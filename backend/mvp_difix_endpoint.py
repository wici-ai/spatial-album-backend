#!/usr/bin/env python3
"""HTTP endpoint for the form-1 on-release Difix refinement."""
from __future__ import annotations

import argparse
import base64
from email import policy
from email.parser import BytesParser
import gc
import io
import json
import os
import re
import subprocess
import sys
import threading
import time
import tempfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

from PIL import Image, ImageEnhance, ImageFilter, ImageOps
from execution_control import controlled_http_task


VIEWER_ROOT = Path(__file__).resolve().parent
ASSETS_ROOT = VIEWER_ROOT / "assets"
DEFAULT_DIFIX_ROOT = Path("/models/difix")
DIFIX_ROOT = Path(os.environ.get("MVP_DIFIX_ROOT", str(DEFAULT_DIFIX_ROOT)))
DIFIX_SRC = Path(os.environ.get("MVP_DIFIX_SRC", "/opt/Difix3D/src"))
if DIFIX_SRC.exists():
    sys.path.insert(0, str(DIFIX_SRC))

SOURCE_IMAGE = Path(
    os.environ.get(
        "MVP_DIFIX_SOURCE",
        str(DIFIX_ROOT / "source_photo_original.jpg"),
    )
)
if not SOURCE_IMAGE.exists():
    SOURCE_IMAGE = ASSETS_ROOT / "default_difix_source.jpg"

MODEL_ID = os.environ.get("MVP_DIFIX_MODEL", "nvidia/difix_ref")
PROMPT = os.environ.get("MVP_DIFIX_PROMPT", "remove degradation")
DIFIX_SEED = int(os.environ.get("MVP_DIFIX_SEED", "0"))
MOCK = os.environ.get("MVP_DIFIX_MOCK", "0") == "1"
KEEP_LOADED = os.environ.get("MVP_DIFIX_KEEP_LOADED", "0") == "1"
INPROC = os.environ.get("MVP_DIFIX_INPROC", "0") == "1"
MIN_FREE_GB = float(os.environ.get("MVP_DIFIX_MIN_FREE_GB", "14"))
MAX_BODY_BYTES = int(os.environ.get("MVP_DIFIX_MAX_BODY_MB", "96")) * 1024 * 1024
MAX_REFINE_SIDE = int(os.environ.get("MVP_DIFIX_MAX_SIDE", "1280"))
SAVE_DIR = Path(os.environ["MVP_DIFIX_SAVE_DIR"]) if os.environ.get("MVP_DIFIX_SAVE_DIR") else None
DIFIX_DTYPE = os.environ.get("MVP_DIFIX_DTYPE", "bf16")
SOURCE_CACHE_DIR = Path(os.environ.get("MVP_DIFIX_SOURCE_CACHE_DIR", str(Path(tempfile.gettempdir()) / "wici-mvp-difix-sources")))
GALLERY_SOURCE_URL = os.environ.get("MVP_DIFIX_GALLERY_SOURCE_URL", "http://127.0.0.1:8772/source")
SOURCE_FETCH_TIMEOUT = float(os.environ.get("MVP_DIFIX_SOURCE_FETCH_TIMEOUT", "15"))
SOURCE_FETCH_MAX_BYTES = int(float(os.environ.get("MVP_DIFIX_SOURCE_FETCH_MAX_MB", "40")) * 1024 * 1024)
RESPONSE_FORMAT = os.environ.get("MVP_DIFIX_RESPONSE_FORMAT", "JPEG").upper()
RESPONSE_JPEG_QUALITY = int(os.environ.get("MVP_DIFIX_RESPONSE_JPEG_QUALITY", "90"))

_lock = threading.Lock()
_source_cache_lock = threading.Lock()
_source_cache: dict[str, Path] = {}
_pipe: Any | None = None
_torch: Any | None = None


def log_event(event: str, **fields: Any) -> None:
    print(json.dumps({"event": event, **fields}, separators=(",", ":")), flush=True)


WORKER_CODE = r"""
import json
import os
import sys
import time

import torch
from PIL import Image, ImageOps

seed_path, out_path, meta_path, source_path, difix_src, model_id, prompt, seed_value = sys.argv[1:]
if difix_src and difix_src not in sys.path:
    sys.path.insert(0, difix_src)
from pipeline_difix import DifixPipeline

seed = Image.open(seed_path).convert("RGB")
source = Image.open(source_path).convert("RGB")
ref = ImageOps.fit(source, seed.size, method=Image.Resampling.LANCZOS, centering=(0.5, 0.5))

t0 = time.time()
print(json.dumps({"event": "mvp_difix_worker_load_begin", "seed": int(seed_value), "size": list(seed.size)}), file=sys.stderr, flush=True)
pipe = DifixPipeline.from_pretrained(model_id, trust_remote_code=True, torch_dtype=torch.bfloat16)
print(json.dumps({"event": "mvp_difix_worker_cuda_begin", "elapsedSeconds": time.time() - t0}), file=sys.stderr, flush=True)
pipe = pipe.to("cuda")
print(json.dumps({"event": "mvp_difix_worker_cuda_done", "elapsedSeconds": time.time() - t0}), file=sys.stderr, flush=True)
pipe.set_progress_bar_config(disable=True)
seed_int = int(seed_value)
generator = torch.Generator(device="cuda").manual_seed(seed_int)
print(json.dumps({"event": "mvp_difix_worker_infer_begin", "elapsedSeconds": time.time() - t0}), file=sys.stderr, flush=True)
with torch.inference_mode():
    out = pipe(
        prompt,
        image=seed,
        ref_image=ref,
        num_inference_steps=1,
        timesteps=[199],
        guidance_scale=0.0,
        generator=generator,
    ).images[0]
torch.cuda.synchronize()
print(json.dumps({"event": "mvp_difix_worker_infer_done", "elapsedSeconds": time.time() - t0}), file=sys.stderr, flush=True)
out.save(out_path)
meta = {
    "mock": False,
    "difixSeconds": time.time() - t0,
    "source": source_path,
    "workerPid": os.getpid(),
    "seed": seed_int,
}
open(meta_path, "w", encoding="utf-8").write(json.dumps(meta))
os._exit(0)
"""


def data_url_to_image(value: str, mode: str = "RGB") -> Image.Image:
    if "," not in value:
        raise ValueError("expected data URL")
    _, encoded = value.split(",", 1)
    raw = base64.b64decode(encoded)
    return Image.open(io.BytesIO(raw)).convert(mode)


def image_bytes_to_image(raw: bytes, mode: str = "RGB") -> Image.Image:
    return Image.open(io.BytesIO(raw)).convert(mode)


def safe_cache_key(value: Any) -> str:
    text = str(value or "source")
    text = re.sub(r"[^A-Za-z0-9_.-]+", "-", text).strip(".-")
    return text[:120] or "source"


def cache_source_path(source_id: Any, source_path: Path) -> None:
    if not source_id:
        return
    with _source_cache_lock:
        _source_cache[str(source_id)] = source_path


def cached_source_path(source_id: Any) -> Path | None:
    if not source_id:
        return None
    if is_user_photo_id(normalize_source_photo_id(source_id)):
        return None
    with _source_cache_lock:
        source_path = _source_cache.get(str(source_id))
    if source_path and source_path.exists():
        return source_path
    return None


def normalize_source_photo_id(source_id: Any) -> str | None:
    if not source_id:
        return None
    photo_id = Path(str(source_id)).name.strip()
    if not photo_id:
        return None
    if photo_id.endswith("-original"):
        photo_id = photo_id[: -len("-original")]
    return photo_id or None


def is_user_photo_id(photo_id: str | None) -> bool:
    return bool(photo_id and str(photo_id).startswith("user-"))


def cache_resolved_source(source_id: Any, source_path: Path, photo_id: str | None = None) -> None:
    if is_user_photo_id(photo_id or normalize_source_photo_id(source_id)):
        return
    cache_source_path(source_id, source_path)
    if photo_id and photo_id != source_id:
        cache_source_path(photo_id, source_path)


def resolve_gallery_source_file(payload: dict[str, Any]) -> tuple[Path, dict[str, Any]] | None:
    value = payload.get("sourceFile") or payload.get("sourceJpg") or payload.get("source_file")
    if not value:
        return None
    name = Path(str(value)).name
    if not name:
        return None
    source_id = payload.get("sourceId") or payload.get("photoId") or name
    for candidate in (ASSETS_ROOT / name, VIEWER_ROOT / name):
        if candidate.exists() and candidate.is_file():
            cache_resolved_source(source_id, candidate, normalize_source_photo_id(source_id))
            return candidate, {
                "mode": "server_file",
                "requestId": source_id,
                "file": name,
                "path": str(candidate),
                "cache": "stored",
            }
    raise FileNotFoundError(f"request sourceFile does not exist under viewer root: {name}")


def resolve_gallery_source_id(payload: dict[str, Any]) -> tuple[Path, dict[str, Any]] | None:
    source_id = payload.get("sourceId") or payload.get("photoId")
    if not source_id:
        return None
    photo_id = normalize_source_photo_id(source_id)
    if is_user_photo_id(photo_id):
        return None
    names = [Path(str(source_id)).name]
    if photo_id and photo_id not in names:
        names.append(photo_id)
    candidates: list[str] = []
    for name in names:
        candidates.append(name)
        if not name.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
            candidates.append(f"{name}.jpg")
    for candidate_name in dict.fromkeys(candidates):
        candidate = ASSETS_ROOT / candidate_name
        if candidate.exists() and candidate.is_file():
            cache_resolved_source(source_id, candidate, photo_id)
            return candidate, {
                "mode": "server_photo_id",
                "requestId": source_id,
                "photoId": photo_id,
                "file": candidate_name,
                "path": str(candidate),
                "cache": "stored",
            }
    return None


def response_filename(headers: Any, fallback: str) -> str:
    disposition = headers.get("Content-Disposition") if headers else None
    if disposition:
        match = re.search(r'filename="?([^";]+)"?', disposition)
        if match:
            return Path(match.group(1)).name
    return fallback


def resolve_gallery_service_source(payload: dict[str, Any]) -> tuple[Path, dict[str, Any]] | None:
    source_id = payload.get("sourceId") or payload.get("photoId")
    photo_id = normalize_source_photo_id(source_id)
    if not photo_id:
        return None
    if is_user_photo_id(photo_id):
        return None

    url = f"{GALLERY_SOURCE_URL}?photoId={quote(photo_id)}"
    request = Request(url, headers={"User-Agent": "WiCiMvpDifix/1.0"})
    try:
        with urlopen(request, timeout=SOURCE_FETCH_TIMEOUT) as response:
            status = getattr(response, "status", 200)
            if status != 200:
                raise FileNotFoundError(f"gallery source service returned HTTP {status} for photoId={photo_id!r}")
            content_length = response.headers.get("Content-Length")
            if content_length and int(content_length) > SOURCE_FETCH_MAX_BYTES:
                raise ValueError(
                    f"gallery source is too large for photoId={photo_id!r}: "
                    f"{content_length} bytes > {SOURCE_FETCH_MAX_BYTES}"
                )
            raw = response.read(SOURCE_FETCH_MAX_BYTES + 1)
            if len(raw) > SOURCE_FETCH_MAX_BYTES:
                raise ValueError(
                    f"gallery source is too large for photoId={photo_id!r}: "
                    f">{SOURCE_FETCH_MAX_BYTES} bytes"
                )
            filename = response_filename(response.headers, f"{safe_cache_key(photo_id)}.jpg")
            suffix = Path(filename).suffix.lower()
            if suffix not in {".jpg", ".jpeg", ".png", ".webp"}:
                suffix = ".jpg"
            SOURCE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
            source_path = SOURCE_CACHE_DIR / f"{safe_cache_key(photo_id)}{suffix}"
            source_path.write_bytes(raw)
            with Image.open(source_path) as image:
                size = list(image.size)
            cache_resolved_source(source_id, source_path, photo_id)
            return source_path, {
                "mode": "gallery_service",
                "requestId": source_id,
                "photoId": photo_id,
                "url": url,
                "path": str(source_path),
                "cache": "stored",
                "size": size,
                "headerSize": [
                    int(response.headers["X-Source-Width"]) if response.headers.get("X-Source-Width") else None,
                    int(response.headers["X-Source-Height"]) if response.headers.get("X-Source-Height") else None,
                ],
            }
    except HTTPError as exc:
        if exc.code == 404:
            return None
        raise FileNotFoundError(f"gallery source service failed for photoId={photo_id!r}: HTTP {exc.code}") from exc
    except URLError as exc:
        raise FileNotFoundError(f"gallery source service unavailable for photoId={photo_id!r}: {exc.reason!r}") from exc


def cache_data_url_source(value: str, source_id: Any, tmp_path: Path | None) -> tuple[Path, dict[str, Any]]:
    image = data_url_to_image(value, "RGB")
    photo_id = normalize_source_photo_id(source_id)
    if source_id and not is_user_photo_id(photo_id):
        SOURCE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        source_path = SOURCE_CACHE_DIR / f"{safe_cache_key(source_id)}.jpg"
        image.save(source_path, format="JPEG", quality=95)
        cache_resolved_source(source_id, source_path, photo_id)
        return source_path, {
            "mode": "data_url_cached",
            "requestId": source_id,
            "path": str(source_path),
            "cache": "stored",
        }
    if tmp_path is None:
        raise ValueError("source data URL requires a temp directory")
    source_path = tmp_path / "request_source.jpg"
    image.save(source_path, format="JPEG", quality=95)
    return source_path, {
        "mode": "data_url_request_local" if is_user_photo_id(photo_id) else "data_url",
        "requestId": source_id,
        "photoId": photo_id,
        "path": str(source_path),
        "cache": "none",
    }


def resolve_request_source(payload: dict[str, Any], tmp_path: Path | None = None) -> tuple[Path, dict[str, Any]]:
    source_id = payload.get("sourceId") or payload.get("photoId")

    file_source = resolve_gallery_source_file(payload)
    if file_source is not None:
        return file_source

    value = payload.get("source") or payload.get("sourceDataUrl") or payload.get("sourcePng")
    if value:
        if isinstance(value, str) and value.startswith("data:"):
            return cache_data_url_source(value, source_id, tmp_path)
        if isinstance(value, str):
            source_path = Path(value)
            if not source_path.exists():
                raise FileNotFoundError(f"request source does not exist: {source_path}")
            cache_resolved_source(source_id, source_path, normalize_source_photo_id(source_id))
            return source_path, {
                "mode": "path",
                "requestId": source_id,
                "path": str(source_path),
                "cache": "stored" if source_id else "none",
            }

    path_value = payload.get("sourcePath") or payload.get("source_path")
    if path_value:
        source_path = Path(str(path_value))
        if not source_path.exists():
            raise FileNotFoundError(f"request sourcePath does not exist: {source_path}")
        cache_resolved_source(source_id, source_path, normalize_source_photo_id(source_id))
        return source_path, {
            "mode": "path",
            "requestId": source_id,
            "path": str(source_path),
            "cache": "stored" if source_id else "none",
        }

    id_source = resolve_gallery_source_id(payload)
    if id_source is not None:
        return id_source

    service_source = resolve_gallery_service_source(payload)
    if service_source is not None:
        return service_source

    cached = cached_source_path(source_id)
    if cached is not None:
        return cached, {
            "mode": "server_cache",
            "requestId": source_id,
            "path": str(cached),
            "cache": "hit",
        }

    raise FileNotFoundError(f"could not resolve source image for sourceId={source_id!r}")


def image_to_data_url(image: Image.Image, image_format: str | None = None) -> str:
    buf = io.BytesIO()
    fmt = (image_format or RESPONSE_FORMAT).upper()
    if fmt in {"JPG", "JPEG"}:
        image.convert("RGB").save(buf, format="JPEG", quality=RESPONSE_JPEG_QUALITY, optimize=True)
        return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")
    image.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def image_to_response_bytes(image: Image.Image) -> tuple[bytes, str]:
    buf = io.BytesIO()
    fmt = RESPONSE_FORMAT.upper()
    if fmt in {"JPG", "JPEG"}:
        image.convert("RGB").save(buf, format="JPEG", quality=RESPONSE_JPEG_QUALITY, optimize=True)
        return buf.getvalue(), "image/jpeg"
    image.save(buf, format="PNG")
    return buf.getvalue(), "image/png"


def parse_multipart_form(content_type: str, raw_body: bytes) -> tuple[dict[str, str], dict[str, dict[str, Any]]]:
    if "multipart/form-data" not in content_type.lower():
        raise ValueError(f"expected multipart/form-data, got {content_type!r}")
    message = BytesParser(policy=policy.default).parsebytes(
        f"Content-Type: {content_type}\r\nMIME-Version: 1.0\r\n\r\n".encode("utf-8") + raw_body
    )
    fields: dict[str, str] = {}
    files: dict[str, dict[str, Any]] = {}
    for part in message.iter_parts():
        disposition = part.get("Content-Disposition", "")
        if "form-data" not in disposition:
            continue
        name = part.get_param("name", header="content-disposition")
        if not name:
            continue
        filename = part.get_param("filename", header="content-disposition")
        content = part.get_payload(decode=True) or b""
        content_type_part = part.get_content_type() or "application/octet-stream"
        if filename:
            files[str(name)] = {
                "filename": str(filename),
                "contentType": content_type_part,
                "bytes": content,
            }
        else:
            fields[str(name)] = content.decode(part.get_content_charset() or "utf-8", errors="replace")
    return fields, files


def clear_cuda_cache() -> None:
    gc.collect()
    if _torch is None or not hasattr(_torch, "cuda") or not _torch.cuda.is_available():
        return
    _torch.cuda.empty_cache()
    if hasattr(_torch.cuda, "ipc_collect"):
        _torch.cuda.ipc_collect()
    if hasattr(_torch.cuda, "reset_peak_memory_stats"):
        try:
            _torch.cuda.reset_peak_memory_stats()
        except Exception:
            pass


def clamp_size(size: tuple[int, int]) -> tuple[int, int]:
    w, h = size
    if MAX_REFINE_SIDE <= 0 or max(w, h) <= MAX_REFINE_SIDE:
        return w, h
    scale = MAX_REFINE_SIDE / max(w, h)
    return max(1, round(w * scale)), max(1, round(h * scale))


def normalize_request_images(
    preview: Image.Image,
    seed: Image.Image,
    refine_mask: Image.Image,
) -> tuple[Image.Image, Image.Image, Image.Image, dict[str, Any]]:
    original = {"preview": list(preview.size), "seed": list(seed.size), "refineMask": list(refine_mask.size)}
    target_size = clamp_size(seed.size)
    if seed.size != target_size:
        seed = seed.resize(target_size, Image.Resampling.LANCZOS)
    if preview.size != target_size:
        preview = preview.resize(target_size, Image.Resampling.LANCZOS)
    if refine_mask.size != target_size:
        refine_mask = refine_mask.resize(target_size, Image.Resampling.NEAREST)
    return preview, seed, refine_mask, {
        "original": original,
        "normalized": list(target_size),
        "maxSide": MAX_REFINE_SIDE,
    }


def json_response(handler: BaseHTTPRequestHandler, status: int, data: dict[str, Any]) -> None:
    raw = json.dumps(data, separators=(",", ":")).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(raw)))
    handler.send_header("Access-Control-Allow-Origin", "*")
    handler.send_header("Access-Control-Allow-Methods", "GET,POST,OPTIONS")
    handler.send_header("Access-Control-Allow-Headers", "Content-Type, Cache-Control, Pragma")
    handler.send_header("Cross-Origin-Resource-Policy", "cross-origin")
    handler.end_headers()
    handler.wfile.write(raw)


def image_response(
    handler: BaseHTTPRequestHandler,
    status: int,
    raw: bytes,
    content_type: str,
    *,
    request_id: str,
    timing: dict[str, Any],
) -> None:
    handler.send_response(status)
    handler.send_header("Content-Type", content_type)
    handler.send_header("Content-Length", str(len(raw)))
    handler.send_header("Access-Control-Allow-Origin", "*")
    handler.send_header("Access-Control-Allow-Methods", "GET,POST,OPTIONS")
    handler.send_header("Access-Control-Allow-Headers", "Content-Type, Cache-Control, Pragma")
    handler.send_header("Cross-Origin-Resource-Policy", "cross-origin")
    handler.send_header("X-Difix-Request-Id", request_id)
    handler.send_header("X-Difix-Timing-Json", json.dumps(timing, separators=(",", ":")))
    handler.end_headers()
    handler.wfile.write(raw)


def gpu_free_enough() -> tuple[bool, dict[str, Any]]:
    global _torch
    if MOCK:
        return True, {"mock": True}
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=memory.free,memory.total",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=5,
        ).strip().splitlines()[0]
        free_mb, total_mb = [float(part.strip()) for part in out.split(",")[:2]]
        info = {
            "cuda": True,
            "freeGb": round(free_mb / 1024, 3),
            "totalGb": round(total_mb / 1024, 3),
            "minFreeGb": MIN_FREE_GB,
            "maxRefineSide": MAX_REFINE_SIDE,
            "source": "nvidia-smi",
        }
        return free_mb >= MIN_FREE_GB * 1024, info
    except Exception:
        pass
    try:
        import torch

        _torch = torch
        if not torch.cuda.is_available():
            return False, {"cuda": False, "message": "CUDA is not available"}
        free, total = torch.cuda.mem_get_info()
        info = {
            "cuda": True,
            "freeGb": round(free / (1024**3), 3),
            "totalGb": round(total / (1024**3), 3),
            "minFreeGb": MIN_FREE_GB,
            "maxRefineSide": MAX_REFINE_SIDE,
        }
        return free >= MIN_FREE_GB * (1024**3), info
    except Exception as exc:
        return False, {"cuda": False, "message": repr(exc)}


def load_pipe() -> Any:
    global _pipe, _torch
    if _pipe is not None:
        log_event("mvp_difix_pipe_reuse")
        return _pipe
    import torch
    from pipeline_difix import DifixPipeline

    _torch = torch
    t0 = time.time()
    log_event("mvp_difix_load_begin", model=MODEL_ID)
    pipe = DifixPipeline.from_pretrained(MODEL_ID, trust_remote_code=True, torch_dtype=torch.bfloat16)
    log_event("mvp_difix_cuda_begin", elapsedSeconds=time.time() - t0)
    pipe = pipe.to("cuda")
    log_event("mvp_difix_cuda_done", elapsedSeconds=time.time() - t0)
    pipe.set_progress_bar_config(disable=True)
    _pipe = pipe
    log_event("mvp_difix_load_done", elapsedSeconds=time.time() - t0)
    return pipe


def unload_pipe(clear_cache: bool = True) -> None:
    global _pipe
    if KEEP_LOADED:
        if clear_cache:
            clear_cuda_cache()
        return
    _pipe = None
    if clear_cache:
        clear_cuda_cache()


def mock_difix(seed: Image.Image) -> Image.Image:
    sharpened = seed.filter(ImageFilter.UnsharpMask(radius=1.15, percent=185, threshold=2))
    sharpened = ImageEnhance.Contrast(sharpened).enhance(1.04)
    return Image.blend(seed, sharpened, 0.86)


def run_difix(seed: Image.Image, source_path: Path) -> tuple[Image.Image, dict[str, Any]]:
    if MOCK:
        t0 = time.time()
        return mock_difix(seed), {"mock": True, "difixSeconds": time.time() - t0}

    ok, gpu = gpu_free_enough()
    if not ok:
        raise GpuBusy(gpu)

    if not INPROC:
        with tempfile.TemporaryDirectory(prefix="wici-mvp-difix-") as tmp:
            tmp_path = Path(tmp)
            seed_path = tmp_path / "seed.png"
            out_path = tmp_path / "raw_difix.png"
            meta_path = tmp_path / "meta.json"
            seed.save(seed_path)
            env = os.environ.copy()
            env["PYTHONPATH"] = str(DIFIX_SRC) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
            proc = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    WORKER_CODE,
                    str(seed_path),
                    str(out_path),
                    str(meta_path),
                    str(source_path),
                    str(DIFIX_SRC),
                    MODEL_ID,
                    PROMPT,
                    str(DIFIX_SEED),
                ],
                cwd=str(VIEWER_ROOT),
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=int(os.environ.get("MVP_DIFIX_WORKER_TIMEOUT", "360")),
            )
            if proc.returncode != 0:
                raise RuntimeError(
                    "Difix worker failed "
                    + json.dumps(
                        {
                            "returncode": proc.returncode,
                            "stdout": proc.stdout[-2000:],
                            "stderr": proc.stderr[-4000:],
                        },
                        separators=(",", ":"),
                    )
                )
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            return Image.open(out_path).convert("RGB"), meta

    start = time.time()
    log_event("mvp_difix_inproc_begin", seed=DIFIX_SEED, size=list(seed.size), source=str(source_path))
    pipe = load_pipe()
    log_event("mvp_difix_inproc_loaded", elapsedSeconds=time.time() - start)
    torch = _torch
    if torch is None:
        import torch as torch_mod

        torch = torch_mod

    source = Image.open(source_path).convert("RGB")
    ref = ImageOps.fit(source, seed.size, method=Image.Resampling.LANCZOS, centering=(0.5, 0.5))
    generator = torch.Generator(device="cuda").manual_seed(DIFIX_SEED)
    out = None
    try:
        log_event("mvp_difix_infer_begin", elapsedSeconds=time.time() - start)
        with torch.inference_mode():
            out = pipe(
                PROMPT,
                image=seed,
                ref_image=ref,
                num_inference_steps=1,
                timesteps=[199],
                guidance_scale=0.0,
                generator=generator,
            ).images[0]
        torch.cuda.synchronize()
        latency = time.time() - start
        result = out.convert("RGB")
        log_event("mvp_difix_infer_done", elapsedSeconds=latency)
        return result, {"mock": False, "difixSeconds": latency, "source": str(source_path), "seed": DIFIX_SEED}
    finally:
        del generator, ref, source
        if out is not None:
            del out
        unload_pipe(clear_cache=False)


def composite(preview: Image.Image, difix: Image.Image, refine_mask: Image.Image) -> Image.Image:
    if difix.size != preview.size:
        difix = difix.resize(preview.size, Image.Resampling.LANCZOS)
    if refine_mask.size != preview.size:
        refine_mask = refine_mask.resize(preview.size, Image.Resampling.BILINEAR)
    mask = refine_mask.convert("L")
    return Image.composite(difix, preview, mask)


def save_artifacts(request_id: str, images: dict[str, Image.Image], meta: dict[str, Any]) -> None:
    if SAVE_DIR is None:
        return
    out = SAVE_DIR / request_id
    out.mkdir(parents=True, exist_ok=True)
    for name, image in images.items():
        image.save(out / f"{name}.png")
    (out / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


class GpuBusy(RuntimeError):
    def __init__(self, info: dict[str, Any]):
        super().__init__("GPU is busy")
        self.info = info


class Handler(BaseHTTPRequestHandler):
    server_version = "WiCiMvpDifix/1.0"

    def log_message(self, fmt: str, *args: Any) -> None:
        print(fmt % args, flush=True)

    def do_OPTIONS(self) -> None:
        json_response(self, 204, {})

    def do_GET(self) -> None:
        path = self.path.split("?", 1)[0]
        if path == "/health-lite":
            json_response(
                self,
                200,
                {
                    "ok": True,
                    "mock": MOCK,
                    "gpuChecked": False,
                    "model": MODEL_ID,
                    "source": str(SOURCE_IMAGE),
                    "difixRoot": str(DIFIX_ROOT),
                    "keepLoaded": KEEP_LOADED,
                    "inproc": INPROC,
                    "seed": DIFIX_SEED,
                    "maxRefineSide": MAX_REFINE_SIDE,
                    "dtype": DIFIX_DTYPE,
                    "responseFormat": RESPONSE_FORMAT,
                    "sourceCacheDir": str(SOURCE_CACHE_DIR),
                },
            )
            return
        if path != "/health":
            json_response(self, 404, {"ok": False, "error": "not_found"})
            return
        ok, gpu = gpu_free_enough()
        json_response(
            self,
            200,
            {
                "ok": True,
                "mock": MOCK,
                "gpuReady": ok,
                "gpu": gpu,
                "model": MODEL_ID,
                "source": str(SOURCE_IMAGE),
                "difixRoot": str(DIFIX_ROOT),
                "keepLoaded": KEEP_LOADED,
                "inproc": INPROC,
                "seed": DIFIX_SEED,
                "maxRefineSide": MAX_REFINE_SIDE,
                "dtype": DIFIX_DTYPE,
                "responseFormat": RESPONSE_FORMAT,
                "sourceCacheDir": str(SOURCE_CACHE_DIR),
            },
        )

    @controlled_http_task("wici-spatial-difix.service", {"/refine", "/api/difix-refine"},
                          finish=lambda: _torch.cuda.synchronize() if _torch is not None else None)
    def do_POST(self) -> None:
        path = self.path.split("?", 1)[0]
        if path not in {"/refine", "/api/difix-refine"}:
            json_response(self, 404, {"ok": False, "error": "not_found"})
            return

        request_start = time.perf_counter()
        trace_id = self.headers.get("X-WiCi-Trace-Id") or ""
        timing: dict[str, Any] = {}
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > MAX_BODY_BYTES:
                raise ValueError(f"invalid body length {length}")
            timing["contentLengthBytes"] = length
            t = time.perf_counter()
            raw_body = self.rfile.read(length)
            timing["readBodyMs"] = round((time.perf_counter() - t) * 1000, 1)
            t = time.perf_counter()
            content_type = self.headers.get("Content-Type", "")
            fields, files = parse_multipart_form(content_type, raw_body)
            payload: dict[str, Any] = dict(fields)
            if isinstance(payload.get("metrics"), str):
                try:
                    payload["metrics"] = json.loads(str(payload["metrics"]))
                except Exception:
                    pass
            timing["requestParseMs"] = round((time.perf_counter() - t) * 1000, 1)

            with tempfile.TemporaryDirectory(prefix="wici-mvp-difix-request-") as tmp:
                tmp_path = Path(tmp)
                t = time.perf_counter()
                source_path: Path | None = None
                source_meta: dict[str, Any]
                source_part = files.get("source")
                if source_part and source_part.get("bytes"):
                    suffix = ".png" if "png" in str(source_part.get("contentType", "")).lower() else ".jpg"
                    source_path = tmp_path / f"request_source{suffix}"
                    source_path.write_bytes(source_part["bytes"])
                    source_meta = {
                        "mode": "multipart_source",
                        "requestId": payload.get("sourceId") or payload.get("photoId"),
                        "photoId": normalize_source_photo_id(payload.get("sourceId") or payload.get("photoId")),
                        "path": str(source_path),
                        "bytes": len(source_part["bytes"]),
                        "contentType": source_part.get("contentType"),
                    }
                else:
                    try:
                        source_path, source_meta = resolve_request_source(payload, tmp_path)
                    except Exception as exc:  # noqa: BLE001 - fallback is explicit and request-local.
                        source_meta = {
                            "mode": "request_image_fallback",
                            "requestId": payload.get("sourceId") or payload.get("photoId"),
                            "photoId": normalize_source_photo_id(payload.get("sourceId") or payload.get("photoId")),
                            "error": repr(exc),
                        }
                        log_event(
                            "mvp_difix_source_resolve_failed",
                            sourceId=payload.get("sourceId"),
                            photoId=payload.get("photoId"),
                            error=repr(exc),
                        )
                timing["sourceResolveMs"] = round((time.perf_counter() - t) * 1000, 1)
                t = time.perf_counter()
                preview = image_bytes_to_image(files["preview"]["bytes"], "RGB")
                seed = image_bytes_to_image(files["seed"]["bytes"], "RGB")
                refine_mask = image_bytes_to_image(files["refineMask"]["bytes"], "L")
                timing["imageDecodeMs"] = round((time.perf_counter() - t) * 1000, 1)
                t = time.perf_counter()
                preview, seed, refine_mask, size_meta = normalize_request_images(preview, seed, refine_mask)
                timing["normalizeMs"] = round((time.perf_counter() - t) * 1000, 1)
                if source_path is None:
                    source_path = tmp_path / "request_seed_reference.jpg"
                    seed.save(source_path, format="JPEG", quality=95)
                    source_meta = {
                        **source_meta,
                        "path": str(source_path),
                        "size": list(seed.size),
                        "fallbackSource": "seed",
                    }
                request_id = time.strftime("%Y%m%d-%H%M%S") + f"-{threading.get_ident()}"
                try:
                    with Image.open(source_path) as source_probe:
                        source_meta = {**source_meta, "size": list(source_probe.size)}
                except Exception:
                    pass

                lock_wait_start = time.perf_counter()
                with _lock:
                    timing["lockWaitMs"] = round((time.perf_counter() - lock_wait_start) * 1000, 1)
                    t0 = time.time()
                    t = time.perf_counter()
                    raw_difix, run_meta = run_difix(seed, source_path)
                    timing["runDifixMs"] = round((time.perf_counter() - t) * 1000, 1)
                    t = time.perf_counter()
                    refined = composite(preview, raw_difix, refine_mask)
                    timing["compositeMs"] = round((time.perf_counter() - t) * 1000, 1)
                    latency = time.time() - t0

                timing["handlerBeforeResponseMs"] = round((time.perf_counter() - request_start) * 1000, 1)
                t = time.perf_counter()
                response_bytes, response_mime = image_to_response_bytes(refined)
                timing["responseImageEncodeMs"] = round((time.perf_counter() - t) * 1000, 1)
                timing["responseImageBytes"] = len(response_bytes)
                timing["responseContentType"] = response_mime

                meta = {
                    "traceId": trace_id,
                    "requestId": request_id,
                    "latencySeconds": latency,
                    "run": run_meta,
                    "source": source_meta,
                    "viewerMetrics": payload.get("metrics") if isinstance(payload.get("metrics"), dict) else {},
                    "requestImages": size_meta,
                    "ply": payload.get("ply"),
                    "photoId": payload.get("photoId"),
                    "serverTimingMs": timing,
                }
                save_artifacts(
                    request_id,
                    {
                        "preview": preview,
                        "seed": seed,
                        "refine_mask": refine_mask,
                        "raw_difix": raw_difix,
                        "refined": refined,
                    },
                    meta,
                )
                log_event(
                    "mvp_difix_request_timing",
                    traceId=trace_id,
                    requestId=request_id,
                    photoId=payload.get("photoId"),
                    timingMs=timing,
                    source=source_meta,
                    requestImages=size_meta,
                )
                image_response(
                    self,
                    200,
                    response_bytes,
                    response_mime,
                    request_id=request_id,
                    timing=timing,
                )
                del preview, seed, refine_mask, raw_difix, refined
        except GpuBusy as exc:
            json_response(
                self,
                503,
                {
                    "ok": False,
                    "error": "gpu_busy",
                    "message": "GPU free memory is below the Difix threshold; retry later.",
                    "retryable": True,
                    "retryMs": 15000,
                    "gpu": exc.info,
                },
            )
        except Exception as exc:
            unload_pipe(clear_cache=False)
            retryable = "out of memory" in str(exc).lower() or "cuda" in str(exc).lower()
            json_response(
                self,
                503 if retryable else 500,
                {
                    "ok": False,
                    "error": "refine_failed",
                    "message": repr(exc),
                    "retryable": retryable,
                    "retryMs": 15000 if retryable else 0,
                },
            )
        finally:
            clear_cuda_cache()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default=os.environ.get("MVP_DIFIX_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("MVP_DIFIX_PORT", "8765")))
    args = parser.parse_args()
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(
        json.dumps(
            {
                "event": "mvp_difix_endpoint_start",
                "host": args.host,
                "port": args.port,
                "mock": MOCK,
                "model": MODEL_ID,
                "source": str(SOURCE_IMAGE),
                "difixRoot": str(DIFIX_ROOT),
                "difixSrc": str(DIFIX_SRC),
                "minFreeGb": MIN_FREE_GB,
                "maxRefineSide": MAX_REFINE_SIDE,
                "inproc": INPROC,
                "seed": DIFIX_SEED,
                "dtype": DIFIX_DTYPE,
            },
            separators=(",", ":"),
        ),
        flush=True,
    )
    if INPROC and KEEP_LOADED and not MOCK:
        load_pipe()
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
