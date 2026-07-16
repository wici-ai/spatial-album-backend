"""Bounded multi-view gsplat training jobs with remote-VRAM preload metadata.

The HTTP/job layer is CPU-safe. A configured runner owns COLMAP preparation and
actual gsplat execution; the contract requires it to honor the preload options
written into manifest.json and exported through WICI_GSPLAT_* variables.
"""
from __future__ import annotations

import json
import os
import re
import shlex
import shutil
import subprocess
import threading
import time
import uuid
from dataclasses import dataclass, field
from email import policy
from email.parser import BytesParser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol
from urllib.parse import urlsplit

PROTOCOL = "wici.gsplat-training.v1"
JOB_ID_RE = re.compile(r"^[A-Za-z0-9_-]{8,80}$")
CANDIDATE_ID_RE = re.compile(r"^[A-Za-z0-9@._:-]{1,160}$")
TERMINAL = {"succeeded", "failed", "cancelled"}


class TrainingError(Exception):
    def __init__(self, status: int, code: str, message: str, *, retryable: bool = False) -> None:
        super().__init__(message)
        self.status = int(status)
        self.code = code
        self.message = message
        self.retryable = retryable

    def payload(self) -> dict[str, Any]:
        return {
            "protocol": PROTOCOL,
            "error": {"code": self.code, "message": self.message, "retryable": self.retryable},
        }


@dataclass(frozen=True)
class Limits:
    max_jobs: int = 2
    max_images: int = 64
    max_upload_bytes: int = 512 * 1024 * 1024
    max_image_bytes: int = 24 * 1024 * 1024
    max_manifest_bytes: int = 64 * 1024
    min_images: int = 2

    def public(self) -> dict[str, int]:
        return {
            "maxJobs": self.max_jobs,
            "minImages": self.min_images,
            "maxImages": self.max_images,
            "maxUploadBytes": self.max_upload_bytes,
            "maxImageBytes": self.max_image_bytes,
        }


@dataclass(frozen=True)
class TrainingImage:
    candidate_id: str
    filename: str
    content_type: str
    data: bytes


@dataclass(frozen=True)
class TrainingUpload:
    manifest: dict[str, Any]
    images: tuple[TrainingImage, ...]


Progress = Callable[[int, int, str], None]


class Runner(Protocol):
    def run(
        self,
        workspace: Path,
        manifest: Mapping[str, Any],
        progress: Progress,
        cancelled: threading.Event,
    ) -> Path: ...


class SubprocessRunner:
    """Runs an operator-provided worker without a shell.

    The worker receives a workspace containing images/ and manifest.json. It
    must place result.splat or result.ply in the output directory. JSON stdout
    lines may report {"step": N, "totalSteps": M, "stage": "training"}.
    """

    def __init__(self, command: str, result_name: str = "result.splat") -> None:
        self.argv = tuple(shlex.split(command))
        if not self.argv:
            raise ValueError("training command must not be empty")
        self.result_name = result_name

    def run(self, workspace: Path, manifest: Mapping[str, Any], progress: Progress, cancelled: threading.Event) -> Path:
        output = workspace / "output"
        output.mkdir(parents=True, exist_ok=True)
        options = manifest["options"]
        env = {
            **os.environ,
            "WICI_GSPLAT_JOB_DIR": str(workspace),
            "WICI_GSPLAT_IMAGES_DIR": str(workspace / "images"),
            "WICI_GSPLAT_OUTPUT_DIR": str(output),
            "WICI_GSPLAT_MANIFEST": str(workspace / "manifest.json"),
            "WICI_GSPLAT_PRELOAD_TRAIN_DATA": "1",
            "WICI_GSPLAT_PRELOAD_CHUNK_BYTES": str(options["preloadChunkBytes"]),
            "WICI_GSPLAT_BATCH_SIZE": str(options["batchSize"]),
            "WICI_GSPLAT_EQUIVALENT_STEPS": str(options["equivalentSteps"]),
        }
        process = subprocess.Popen(
            self.argv,
            cwd=workspace,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        log_path = workspace / "worker.log"
        with log_path.open("w", encoding="utf-8") as log:
            assert process.stdout is not None
            while True:
                if cancelled.is_set() and process.poll() is None:
                    process.terminate()
                line = process.stdout.readline()
                if line:
                    log.write(line)
                    log.flush()
                    try:
                        event = json.loads(line)
                    except json.JSONDecodeError:
                        event = None
                    if isinstance(event, dict) and isinstance(event.get("step"), int):
                        progress(
                            event["step"],
                            int(event.get("totalSteps", options["equivalentSteps"])),
                            str(event.get("stage", "training")),
                        )
                elif process.poll() is not None:
                    break
                else:
                    time.sleep(0.02)
        if cancelled.is_set():
            raise TrainingError(HTTPStatus.CONFLICT, "job_cancelled", "training job was cancelled")
        if process.returncode != 0:
            raise RuntimeError(f"training worker exited with {process.returncode}")
        expected = output / self.result_name
        if expected.is_file():
            return expected
        alternatives = sorted(output.glob("*.splat")) + sorted(output.glob("*.ply"))
        if not alternatives:
            raise RuntimeError("training worker did not produce a .splat or .ply result")
        return alternatives[0]


@dataclass
class _Job:
    job_id: str
    workspace: Path
    manifest: dict[str, Any]
    image_count: int
    created_at: float
    state: str = "queued"
    stage: str = "queued"
    step: int = 0
    total_steps: int = 0
    started_at: float | None = None
    finished_at: float | None = None
    result: Path | None = None
    error: str | None = None
    cancel: threading.Event = field(default_factory=threading.Event)


class JobManager:
    def __init__(
        self,
        root: Path,
        runner: Runner | None,
        *,
        limits: Limits = Limits(),
        clock: Callable[[], float] = time.time,
        token: Callable[[], str] = lambda: uuid.uuid4().hex,
    ) -> None:
        self.root = root
        self.runner = runner
        self.limits = limits
        self.clock = clock
        self.token = token
        self._jobs: dict[str, _Job] = {}
        self._lock = threading.Lock()
        root.mkdir(parents=True, exist_ok=True)

    @property
    def enabled(self) -> bool:
        return self.runner is not None

    def public(self) -> dict[str, Any]:
        with self._lock:
            active = sum(job.state not in TERMINAL for job in self._jobs.values())
        return {
            "enabled": self.enabled,
            "protocol": PROTOCOL,
            "preloadTrainingData": True,
            "preloadChunkBytes": 768 * 1024,
            "limits": self.limits.public(),
            "activeJobs": active,
        }

    def create(self, upload: TrainingUpload) -> dict[str, Any]:
        if self.runner is None:
            raise TrainingError(HTTPStatus.SERVICE_UNAVAILABLE, "training_disabled", "gsplat training worker is not configured")
        with self._lock:
            active = sum(job.state not in TERMINAL for job in self._jobs.values())
            if active >= self.limits.max_jobs:
                raise TrainingError(HTTPStatus.TOO_MANY_REQUESTS, "job_capacity", "training job capacity reached", retryable=True)
            job_id = self._new_id_locked()
            workspace = self.root / job_id
            images_dir = workspace / "images"
            images_dir.mkdir(parents=True)
            manifest = dict(upload.manifest)
            mapping: list[dict[str, str]] = []
            for index, image in enumerate(upload.images):
                suffix = _safe_suffix(image.filename, image.content_type)
                stored_name = f"{index:03d}{suffix}"
                (images_dir / stored_name).write_bytes(image.data)
                mapping.append({"candidateId": image.candidate_id, "file": f"images/{stored_name}"})
            manifest["images"] = mapping
            (workspace / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
            job = _Job(
                job_id,
                workspace,
                manifest,
                len(upload.images),
                self.clock(),
                total_steps=int(manifest["options"]["equivalentSteps"]),
            )
            self._jobs[job_id] = job
        threading.Thread(target=self._execute, args=(job,), daemon=True, name=f"gsplat-{job_id}").start()
        return self.payload(job)

    def _new_id_locked(self) -> str:
        for _ in range(16):
            candidate = self.token()
            if JOB_ID_RE.fullmatch(candidate) and candidate not in self._jobs:
                return candidate
        raise TrainingError(HTTPStatus.SERVICE_UNAVAILABLE, "job_id_unavailable", "could not allocate job id", retryable=True)

    def _execute(self, job: _Job) -> None:
        with self._lock:
            if job.cancel.is_set():
                job.state, job.stage, job.finished_at = "cancelled", "cancelled", self.clock()
                return
            job.state, job.stage, job.started_at = "running", "preparing", self.clock()
        try:
            assert self.runner is not None
            result = self.runner.run(job.workspace, job.manifest, lambda s, t, g: self._progress(job, s, t, g), job.cancel)
            with self._lock:
                if job.cancel.is_set():
                    job.state, job.stage = "cancelled", "cancelled"
                else:
                    job.result, job.state, job.stage = result, "succeeded", "ready"
                    job.step = job.total_steps
                job.finished_at = self.clock()
        except TrainingError as exc:
            with self._lock:
                job.state = "cancelled" if job.cancel.is_set() or exc.code == "job_cancelled" else "failed"
                job.stage = job.state
                job.error = None if job.state == "cancelled" else exc.message
                job.finished_at = self.clock()
        except Exception as exc:  # Keep internal details out of the wire response.
            with self._lock:
                job.state, job.stage, job.error, job.finished_at = "failed", "failed", type(exc).__name__, self.clock()

    def _progress(self, job: _Job, step: int, total_steps: int, stage: str) -> None:
        with self._lock:
            if job.state == "running":
                job.step = max(0, min(int(step), int(total_steps)))
                job.total_steps = max(1, int(total_steps))
                job.stage = stage[:80]

    def get(self, job_id: str) -> _Job:
        if not JOB_ID_RE.fullmatch(job_id):
            raise TrainingError(HTTPStatus.NOT_FOUND, "job_not_found", "training job not found")
        with self._lock:
            job = self._jobs.get(job_id)
        if job is None:
            raise TrainingError(HTTPStatus.NOT_FOUND, "job_not_found", "training job not found")
        return job

    def payload(self, job: _Job) -> dict[str, Any]:
        with self._lock:
            elapsed = (job.finished_at or self.clock()) - (job.started_at or job.created_at)
            progress = 0.0 if job.total_steps <= 0 else min(1.0, job.step / job.total_steps)
            return {
                "protocol": PROTOCOL,
                "jobId": job.job_id,
                "state": job.state,
                "stage": job.stage,
                "step": job.step,
                "totalSteps": job.total_steps,
                "progress": progress,
                "imageCount": job.image_count,
                "preloadTrainingData": True,
                "preloadChunkBytes": int(job.manifest["options"]["preloadChunkBytes"]),
                "elapsedSeconds": max(0.0, elapsed),
                **({"statusUrl": f"training/jobs/{job.job_id}", "resultUrl": f"training/jobs/{job.job_id}/result"}),
                **({"error": job.error} if job.error else {}),
            }

    def cancel_job(self, job_id: str) -> None:
        job = self.get(job_id)
        job.cancel.set()
        with self._lock:
            if job.state == "queued":
                job.state, job.stage, job.finished_at = "cancelled", "cancelled", self.clock()

    def result(self, job_id: str) -> Path:
        job = self.get(job_id)
        with self._lock:
            if job.state != "succeeded" or job.result is None:
                raise TrainingError(HTTPStatus.CONFLICT, "result_not_ready", "training result is not ready", retryable=job.state not in TERMINAL)
            result = job.result
        if not result.is_file() or job.workspace not in result.parents:
            raise TrainingError(HTTPStatus.INTERNAL_SERVER_ERROR, "result_unavailable", "training result is unavailable", retryable=True)
        return result

    def close(self, job_id: str) -> None:
        job = self.get(job_id)
        self.cancel_job(job_id)
        with self._lock:
            if job.state not in TERMINAL:
                raise TrainingError(HTTPStatus.CONFLICT, "job_running", "cancel requested; retry deletion after the worker stops", retryable=True)
            del self._jobs[job_id]
        shutil.rmtree(job.workspace, ignore_errors=True)


def parse_multipart(content_type: str, body: bytes, limits: Limits) -> TrainingUpload:
    message = BytesParser(policy=policy.default).parsebytes(
        f"Content-Type: {content_type}\r\nMIME-Version: 1.0\r\n\r\n".encode("ascii") + body
    )
    if not message.is_multipart():
        raise TrainingError(HTTPStatus.BAD_REQUEST, "invalid_multipart", "request must be multipart/form-data")
    protocol: str | None = None
    manifest_raw: bytes | None = None
    images: list[TrainingImage] = []
    for part in message.iter_parts():
        name = part.get_param("name", header="content-disposition")
        data = part.get_payload(decode=True) or b""
        if name == "protocol":
            if protocol is not None:
                raise TrainingError(HTTPStatus.BAD_REQUEST, "invalid_multipart", "protocol must appear once")
            protocol = data.decode("utf-8", "strict")
        elif name == "manifest":
            if manifest_raw is not None or len(data) > limits.max_manifest_bytes:
                raise TrainingError(HTTPStatus.BAD_REQUEST, "invalid_manifest", "manifest is duplicated or too large")
            manifest_raw = data
        elif name == "image":
            candidate_id = part.get("X-Wici-Candidate-Id", "")
            if not CANDIDATE_ID_RE.fullmatch(candidate_id):
                raise TrainingError(HTTPStatus.BAD_REQUEST, "invalid_candidate", "each image requires a valid X-Wici-Candidate-Id")
            if len(data) <= 0 or len(data) > limits.max_image_bytes:
                raise TrainingError(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, "image_too_large", "image is empty or exceeds per-image limit")
            images.append(TrainingImage(candidate_id, part.get_filename() or candidate_id, part.get_content_type(), data))
        else:
            raise TrainingError(HTTPStatus.BAD_REQUEST, "invalid_multipart", f"unknown multipart field {name}")
    if protocol != PROTOCOL or manifest_raw is None:
        raise TrainingError(HTTPStatus.BAD_REQUEST, "invalid_protocol", f"protocol must be {PROTOCOL}")
    if not limits.min_images <= len(images) <= limits.max_images:
        raise TrainingError(HTTPStatus.BAD_REQUEST, "invalid_image_count", f"image count must be {limits.min_images}..{limits.max_images}")
    try:
        raw = json.loads(manifest_raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise TrainingError(HTTPStatus.BAD_REQUEST, "invalid_manifest", "manifest must be UTF-8 JSON") from exc
    manifest = _manifest(raw, limits)
    actual = [image.candidate_id for image in images]
    selected = manifest["selectedCandidateIds"]
    if len(set(actual)) != len(actual) or set(actual) != set(selected):
        raise TrainingError(HTTPStatus.BAD_REQUEST, "image_manifest_mismatch", "uploaded images must exactly match selectedCandidateIds")
    order = {candidate_id: index for index, candidate_id in enumerate(selected)}
    images.sort(key=lambda image: order[image.candidate_id])
    return TrainingUpload(manifest, tuple(images))


def _manifest(value: Any, limits: Limits) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TrainingError(HTTPStatus.BAD_REQUEST, "invalid_manifest", "manifest must be an object")
    allowed = {"protocol", "sceneId", "anchorCandidateId", "selectedCandidateIds", "options"}
    if set(value) != allowed or value.get("protocol") != PROTOCOL:
        raise TrainingError(HTTPStatus.BAD_REQUEST, "invalid_manifest", "manifest fields or protocol are invalid")
    scene_id = value.get("sceneId")
    anchor = value.get("anchorCandidateId")
    selected = value.get("selectedCandidateIds")
    options = value.get("options")
    if not isinstance(scene_id, str) or not scene_id or len(scene_id) > 160:
        raise TrainingError(HTTPStatus.BAD_REQUEST, "invalid_manifest", "sceneId is invalid")
    if not isinstance(selected, list) or not limits.min_images <= len(selected) <= limits.max_images:
        raise TrainingError(HTTPStatus.BAD_REQUEST, "invalid_manifest", "selectedCandidateIds count is invalid")
    if any(not isinstance(item, str) or not CANDIDATE_ID_RE.fullmatch(item) for item in selected) or len(set(selected)) != len(selected):
        raise TrainingError(HTTPStatus.BAD_REQUEST, "invalid_manifest", "selectedCandidateIds are invalid")
    if anchor not in selected:
        raise TrainingError(HTTPStatus.BAD_REQUEST, "invalid_manifest", "anchorCandidateId must be selected")
    if not isinstance(options, dict) or set(options) - {"preloadTrainingData", "preloadChunkBytes", "batchSize", "equivalentSteps"}:
        raise TrainingError(HTTPStatus.BAD_REQUEST, "invalid_manifest", "options are invalid")
    if options.get("preloadTrainingData") is not True:
        raise TrainingError(HTTPStatus.BAD_REQUEST, "preload_required", "preloadTrainingData must be true")
    chunk = options.get("preloadChunkBytes", 768 * 1024)
    batch = options.get("batchSize", 8)
    steps = options.get("equivalentSteps", 30_000)
    if not isinstance(chunk, int) or not 64 * 1024 <= chunk < 1024 * 1024:
        raise TrainingError(HTTPStatus.BAD_REQUEST, "invalid_manifest", "preloadChunkBytes must be 64 KiB..<1 MiB")
    if not isinstance(batch, int) or not 1 <= batch <= 8:
        raise TrainingError(HTTPStatus.BAD_REQUEST, "invalid_manifest", "batchSize must be 1..8")
    if not isinstance(steps, int) or not 1 <= steps <= 100_000:
        raise TrainingError(HTTPStatus.BAD_REQUEST, "invalid_manifest", "equivalentSteps must be 1..100000")
    return {
        "protocol": PROTOCOL,
        "sceneId": scene_id,
        "anchorCandidateId": anchor,
        "selectedCandidateIds": list(selected),
        "options": {
            "preloadTrainingData": True,
            "preloadChunkBytes": chunk,
            "batchSize": batch,
            "equivalentSteps": steps,
        },
    }


def _safe_suffix(filename: str, content_type: str) -> str:
    suffix = Path(filename).suffix.lower()
    if suffix in {".jpg", ".jpeg", ".png", ".webp"}:
        return suffix
    return ".png" if content_type == "image/png" else ".jpg"


class TrainingRouteMixin:
    def _training_json(self, status: int, payload: Mapping[str, Any]) -> None:
        data = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET,POST,DELETE,OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type,X-Wici-Candidate-Id")
        self.end_headers()
        self.wfile.write(data)

    def dispatch_training(self, manager: JobManager, method: str) -> bool:
        path = urlsplit(self.path).path
        if path != "/training/jobs" and not path.startswith("/training/jobs/"):
            return False
        try:
            if method == "POST" and path == "/training/jobs":
                length = _content_length(self, manager.limits.max_upload_bytes)
                upload = parse_multipart(self.headers.get("Content-Type", ""), self.rfile.read(length), manager.limits)
                self._training_json(HTTPStatus.ACCEPTED, manager.create(upload))
                return True
            suffix = path[len("/training/jobs/") :]
            result_request = suffix.endswith("/result")
            job_id = suffix[:-len("/result")] if result_request else suffix
            if not job_id or "/" in job_id:
                raise TrainingError(HTTPStatus.NOT_FOUND, "route_not_found", "route not found")
            if method == "GET" and result_request:
                result = manager.result(job_id)
                data = result.read_bytes()
                content_type = "application/octet-stream" if result.suffix == ".splat" else "application/octet-stream"
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(data)))
                self.send_header("Content-Disposition", f'attachment; filename="{result.name}"')
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(data)
                return True
            if method == "GET" and not result_request:
                self._training_json(HTTPStatus.OK, manager.payload(manager.get(job_id)))
                return True
            if method == "DELETE" and not result_request:
                manager.close(job_id)
                self.send_response(HTTPStatus.NO_CONTENT)
                self.send_header("Content-Length", "0")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                return True
            raise TrainingError(HTTPStatus.NOT_FOUND, "route_not_found", "route not found")
        except TrainingError as exc:
            self._training_json(exc.status, exc.payload())
        except Exception:
            self._training_json(HTTPStatus.INTERNAL_SERVER_ERROR, TrainingError(500, "internal_error", "internal server error", retryable=True).payload())
        return True


def _content_length(handler: BaseHTTPRequestHandler, limit: int) -> int:
    try:
        length = int(handler.headers.get("Content-Length", ""))
    except ValueError as exc:
        raise TrainingError(HTTPStatus.BAD_REQUEST, "invalid_request", "valid Content-Length is required") from exc
    if length <= 0:
        raise TrainingError(HTTPStatus.BAD_REQUEST, "invalid_request", "request body is empty")
    if length > limit:
        handler.close_connection = True
        raise TrainingError(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, "body_too_large", "request body exceeds training upload limit")
    return length


def make_handler(manager: JobManager) -> type[BaseHTTPRequestHandler]:
    class Handler(TrainingRouteMixin, BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: Any) -> None:
            return

        def do_GET(self) -> None:
            if not self.dispatch_training(manager, "GET"):
                self._training_json(404, TrainingError(404, "route_not_found", "route not found").payload())

        def do_POST(self) -> None:
            if not self.dispatch_training(manager, "POST"):
                self._training_json(404, TrainingError(404, "route_not_found", "route not found").payload())

        def do_DELETE(self) -> None:
            if not self.dispatch_training(manager, "DELETE"):
                self._training_json(404, TrainingError(404, "route_not_found", "route not found").payload())

    return Handler
