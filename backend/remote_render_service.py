#!/usr/bin/env python3
"""CPU-safe remote render protocol and session lifecycle primitives.

This module deliberately imports no renderer, tensor, or GPU package.  The Orbit
process supplies an engine implementing ``RemoteRenderEngine`` in the adapter.
"""
from __future__ import annotations

import io
import json
import math
import re
import threading
import time
import zipfile
from dataclasses import dataclass, field
from email import policy
from email.parser import BytesParser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable, Mapping, Protocol
from urllib.parse import urlsplit


PROTOCOL = "remote-render-v1"
CAMERA_CONVENTION = "opencv-w2c-row-major-4x4"
CAPTURE_MEMBERS = ("manifest.json", "seed.jpg", "preview.jpg", "alpha.png")
SESSION_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{8,128}$")


class ProtocolError(Exception):
    """A stable, client-visible remote-render error."""

    def __init__(
        self,
        status: int,
        code: str,
        message: str,
        *,
        retryable: bool = False,
        details: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.status = int(status)
        self.code = code
        self.message = message
        self.retryable = bool(retryable)
        self.details = dict(details or {})

    def payload(self) -> dict[str, Any]:
        error: dict[str, Any] = {
            "code": self.code,
            "message": self.message,
            "retryable": self.retryable,
        }
        if self.details:
            error["details"] = self.details
        return {"protocol": PROTOCOL, "error": error}


def _bad(code: str, message: str, *, details: Mapping[str, Any] | None = None) -> ProtocolError:
    return ProtocolError(HTTPStatus.BAD_REQUEST, code, message, details=details)


def _object(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise _bad("invalid_request", f"{name} must be an object")
    return value


def _strict_keys(value: Mapping[str, Any], required: set[str], optional: set[str], name: str) -> None:
    missing = sorted(required - value.keys())
    unknown = sorted(value.keys() - required - optional)
    if missing or unknown:
        raise _bad(
            "invalid_request",
            f"{name} has invalid fields",
            details={"missing": missing, "unknown": unknown},
        )


def _integer(value: Any, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise _bad("invalid_request", f"{name} must be an integer >= {minimum}")
    return value


def _number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise _bad("invalid_camera", f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise _bad("invalid_camera", f"{name} must be a finite number")
    return result


def _protocol(value: Any) -> None:
    if value != PROTOCOL:
        raise _bad(
            "unsupported_protocol",
            f"protocol must be {PROTOCOL}",
            details={"supported": [PROTOCOL]},
        )


@dataclass(frozen=True)
class Limits:
    max_sessions: int = 4
    idle_ttl_seconds: float = 120.0
    tombstone_ttl_seconds: float = 300.0
    max_frame_pixels: int = 1280 * 1280
    max_upload_bytes: int = 20 * 1024 * 1024
    max_json_bytes: int = 64 * 1024
    max_capture_bytes: int = 32 * 1024 * 1024

    def __post_init__(self) -> None:
        for name in ("max_sessions", "max_frame_pixels", "max_upload_bytes", "max_json_bytes", "max_capture_bytes"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.idle_ttl_seconds <= 0 or self.tombstone_ttl_seconds <= 0:
            raise ValueError("TTL values must be positive")

    def public(self) -> dict[str, Any]:
        return {
            "maxFramePixels": self.max_frame_pixels,
            "maxUploadBytes": self.max_upload_bytes,
            "sessionIdleTtlSeconds": self.idle_ttl_seconds,
        }


@dataclass(frozen=True)
class Camera:
    convention: str
    w2c: tuple[float, ...]
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int

    def public(self) -> dict[str, Any]:
        return {
            "convention": self.convention,
            "w2c": list(self.w2c),
            "intrinsics": {"fx": self.fx, "fy": self.fy, "cx": self.cx, "cy": self.cy},
            "width": self.width,
            "height": self.height,
        }


@dataclass(frozen=True)
class SessionResource:
    handle: Any
    initial_camera: Camera
    pivot: tuple[float, float, float]
    focus_distance: float
    source_width: int
    source_height: int


@dataclass(frozen=True)
class RenderedFrame:
    preview_jpeg: bytes
    width: int
    height: int
    seed_jpeg: bytes | None = None
    alpha_png: bytes | None = None


class RemoteRenderEngine(Protocol):
    def create_gallery(self, photo_id: str) -> SessionResource: ...
    def create_upload(self, image: bytes, filename: str, content_type: str) -> SessionResource: ...
    def render(self, handle: Any, camera: Camera, mode: str) -> RenderedFrame: ...
    def release(self, handle: Any) -> None: ...


@dataclass
class _Session:
    session_id: str
    resource: SessionResource
    created_at: float
    last_used: float
    latest_request_id: int = -1
    active_requests: int = 0
    delete_pending: bool = False
    released: bool = False
    lock: threading.Lock = field(default_factory=threading.Lock)


class SessionManager:
    """Owns engine resources and enforces request/lifetime ordering."""

    def __init__(
        self,
        engine: RemoteRenderEngine,
        *,
        limits: Limits | None = None,
        clock: Callable[[], float] = time.monotonic,
        token: Callable[[], str] | None = None,
        render_lock: threading.Lock | None = None,
    ) -> None:
        import secrets

        self.engine = engine
        self.limits = limits or Limits()
        self.clock = clock
        self.token = token or (lambda: secrets.token_urlsafe(18))
        self.render_lock = render_lock or threading.Lock()
        self._lock = threading.RLock()
        self._sessions: dict[str, _Session] = {}
        self._tombstones: dict[str, float] = {}

    def _release(self, session: _Session) -> None:
        with self._lock:
            if session.released:
                return
            session.released = True
        self.engine.release(session.resource.handle)

    def _prune_locked(self, now: float) -> list[_Session]:
        for session_id, expired_at in list(self._tombstones.items()):
            if now - expired_at >= self.limits.tombstone_ttl_seconds:
                del self._tombstones[session_id]
        releases: list[_Session] = []
        for session_id, session in list(self._sessions.items()):
            if now - session.last_used < self.limits.idle_ttl_seconds:
                continue
            session.delete_pending = True
            self._tombstones[session_id] = now
            del self._sessions[session_id]
            if session.active_requests == 0:
                releases.append(session)
        return releases

    def sweep(self) -> None:
        with self._lock:
            releases = self._prune_locked(self.clock())
        for session in releases:
            self._release(session)

    def _unique_id_locked(self) -> str:
        for _ in range(8):
            session_id = self.token()
            if (
                isinstance(session_id, str)
                and SESSION_ID_PATTERN.fullmatch(session_id) is not None
                and session_id not in self._sessions
                and session_id not in self._tombstones
            ):
                return session_id
        raise ProtocolError(HTTPStatus.SERVICE_UNAVAILABLE, "token_unavailable", "could not allocate session id", retryable=True)

    def _create(self, factory: Callable[[], SessionResource]) -> dict[str, Any]:
        self.sweep()
        with self._lock:
            if len(self._sessions) >= self.limits.max_sessions:
                raise ProtocolError(HTTPStatus.TOO_MANY_REQUESTS, "session_capacity", "session capacity reached", retryable=True)
        try:
            resource = factory()
        except ProtocolError:
            raise
        except Exception as exc:
            raise ProtocolError(HTTPStatus.SERVICE_UNAVAILABLE, "engine_unavailable", "render engine unavailable", retryable=True) from exc
        try:
            self._validate_resource(resource)
        except Exception as exc:
            # A factory-created handle is manager-owned even when its metadata
            # cannot safely be exposed to a client.
            if isinstance(resource, SessionResource):
                self.engine.release(resource.handle)
            if isinstance(exc, ProtocolError) and exc.status == HTTPStatus.SERVICE_UNAVAILABLE:
                raise
            raise ProtocolError(
                HTTPStatus.SERVICE_UNAVAILABLE,
                "invalid_engine_output",
                "engine returned invalid session metadata",
            ) from exc
        now = self.clock()
        allocation_error: ProtocolError | None = None
        with self._lock:
            releases = self._prune_locked(now)
            if len(self._sessions) >= self.limits.max_sessions:
                overflow = True
                session_id = ""
            else:
                overflow = False
                try:
                    session_id = self._unique_id_locked()
                except ProtocolError as error:
                    session_id = ""
                    allocation_error = error
                else:
                    self._sessions[session_id] = _Session(session_id, resource, now, now)
        for old in releases:
            self._release(old)
        if overflow or allocation_error is not None:
            self.engine.release(resource.handle)
        if allocation_error is not None:
            raise allocation_error
        if overflow:
            raise ProtocolError(HTTPStatus.TOO_MANY_REQUESTS, "session_capacity", "session capacity reached", retryable=True)
        return self._session_payload(session_id, resource)

    def _validate_resource(self, resource: SessionResource) -> None:
        if not isinstance(resource, SessionResource):
            raise ProtocolError(HTTPStatus.SERVICE_UNAVAILABLE, "invalid_engine_output", "engine returned invalid session metadata")
        parse_camera(resource.initial_camera.public(), self.limits)
        if len(resource.pivot) != 3 or any(not math.isfinite(float(value)) for value in resource.pivot):
            raise ProtocolError(HTTPStatus.SERVICE_UNAVAILABLE, "invalid_engine_output", "engine returned invalid pivot")
        if not math.isfinite(float(resource.focus_distance)) or resource.focus_distance <= 0:
            raise ProtocolError(HTTPStatus.SERVICE_UNAVAILABLE, "invalid_engine_output", "engine returned invalid focus distance")
        if (
            isinstance(resource.source_width, bool)
            or isinstance(resource.source_height, bool)
            or not isinstance(resource.source_width, int)
            or not isinstance(resource.source_height, int)
            or resource.source_width <= 0
            or resource.source_height <= 0
        ):
            raise ProtocolError(HTTPStatus.SERVICE_UNAVAILABLE, "invalid_engine_output", "engine returned invalid source dimensions")

    def create_gallery(self, photo_id: str) -> dict[str, Any]:
        if not isinstance(photo_id, str) or not photo_id or len(photo_id) > 256:
            raise _bad("invalid_source", "photoId must be a non-empty string of at most 256 characters")
        return self._create(lambda: self.engine.create_gallery(photo_id))

    def create_upload(self, image: bytes, filename: str, content_type: str) -> dict[str, Any]:
        if not image:
            raise _bad("invalid_source", "image must not be empty")
        if len(image) > self.limits.max_upload_bytes:
            raise ProtocolError(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, "upload_too_large", "image exceeds upload limit")
        if content_type not in {"image/jpeg", "image/png"}:
            raise _bad("invalid_source", "image must be image/jpeg or image/png")
        safe_name = filename.rsplit("/", 1)[-1].rsplit("\\", 1)[-1][:255] or "upload"
        return self._create(lambda: self.engine.create_upload(image, safe_name, content_type))

    def _session_payload(self, session_id: str, resource: SessionResource) -> dict[str, Any]:
        return {
            "protocol": PROTOCOL,
            "sessionId": session_id,
            "camera": resource.initial_camera.public(),
            "pivot": list(resource.pivot),
            "focusDistance": resource.focus_distance,
            "source": {"width": resource.source_width, "height": resource.source_height},
            "limits": self.limits.public(),
        }

    def _lookup_locked(self, session_id: str, now: float) -> _Session:
        # Callers sweep before taking this lookup path, so engine callbacks are
        # never made while the manager lock is held.
        session = self._sessions.get(session_id)
        if session is not None:
            return session
        if session_id in self._tombstones:
            raise ProtocolError(HTTPStatus.GONE, "session_expired", "session expired")
        raise ProtocolError(HTTPStatus.NOT_FOUND, "session_not_found", "session not found")

    def render(self, session_id: str, request: Mapping[str, Any]) -> tuple[str, bytes, dict[str, str]]:
        request_id, mode, camera = parse_frame_request(request, self.limits)
        self.sweep()
        with self._lock:
            session = self._lookup_locked(session_id, self.clock())
            with session.lock:
                if session.delete_pending:
                    raise ProtocolError(HTTPStatus.GONE, "session_expired", "session expired")
                if request_id <= session.latest_request_id:
                    raise ProtocolError(HTTPStatus.CONFLICT, "request_out_of_order", "requestId must increase monotonically")
                session.latest_request_id = request_id
                session.last_used = self.clock()
                session.active_requests += 1
        try:
            with self.render_lock:
                with session.lock:
                    if session.delete_pending:
                        raise ProtocolError(HTTPStatus.GONE, "session_expired", "session expired")
                    if request_id != session.latest_request_id:
                        raise ProtocolError(HTTPStatus.CONFLICT, "request_superseded", "request was superseded", retryable=True)
                try:
                    frame = self.engine.render(session.resource.handle, camera, mode)
                except ProtocolError:
                    raise
                except Exception as exc:
                    raise ProtocolError(HTTPStatus.SERVICE_UNAVAILABLE, "render_unavailable", "render engine unavailable", retryable=True) from exc
            self._validate_frame(frame, camera, mode)
            headers = {
                "Cache-Control": "no-store",
                "X-Wici-Protocol": PROTOCOL,
                "X-Wici-Request-Id": str(request_id),
                "X-Wici-Frame-Width": str(frame.width),
                "X-Wici-Frame-Height": str(frame.height),
            }
            if mode == "interactive":
                return "image/jpeg", frame.preview_jpeg, headers
            return "application/zip", self._capture_zip(session_id, request_id, frame), headers
        finally:
            release = False
            with self._lock:
                with session.lock:
                    session.active_requests -= 1
                    release = session.delete_pending and session.active_requests == 0
            if release:
                self._release(session)

    def _validate_frame(self, frame: RenderedFrame, camera: Camera, mode: str) -> None:
        if frame.width != camera.width or frame.height != camera.height:
            raise ProtocolError(HTTPStatus.SERVICE_UNAVAILABLE, "invalid_engine_output", "engine returned unexpected dimensions")
        if not frame.preview_jpeg.startswith(b"\xff\xd8"):
            raise ProtocolError(HTTPStatus.SERVICE_UNAVAILABLE, "invalid_engine_output", "engine preview is not JPEG")
        if mode == "capture":
            if frame.seed_jpeg is None or not frame.seed_jpeg.startswith(b"\xff\xd8"):
                raise ProtocolError(HTTPStatus.SERVICE_UNAVAILABLE, "invalid_engine_output", "capture seed is not JPEG")
            if frame.alpha_png is None or not frame.alpha_png.startswith(b"\x89PNG\r\n\x1a\n"):
                raise ProtocolError(HTTPStatus.SERVICE_UNAVAILABLE, "invalid_engine_output", "capture alpha must be lossless PNG")

    def _capture_zip(self, session_id: str, request_id: int, frame: RenderedFrame) -> bytes:
        manifest = {
            "protocol": PROTOCOL,
            "sessionId": session_id,
            "requestId": request_id,
            "width": frame.width,
            "height": frame.height,
            "members": {"seed": "seed.jpg", "preview": "preview.jpg", "alpha": "alpha.png"},
        }
        output = io.BytesIO()
        with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("manifest.json", json.dumps(manifest, separators=(",", ":")))
            archive.writestr("seed.jpg", frame.seed_jpeg or b"")
            archive.writestr("preview.jpg", frame.preview_jpeg)
            archive.writestr("alpha.png", frame.alpha_png or b"")
        data = output.getvalue()
        if len(data) > self.limits.max_capture_bytes:
            raise ProtocolError(HTTPStatus.SERVICE_UNAVAILABLE, "capture_too_large", "capture exceeds server limit", retryable=True)
        return data

    def delete(self, session_id: str) -> None:
        self.sweep()
        now = self.clock()
        with self._lock:
            releases: list[_Session] = []
            session = self._sessions.pop(session_id, None)
            if session is None:
                for old in releases:
                    self._release(old)
                if session_id in self._tombstones:
                    raise ProtocolError(HTTPStatus.GONE, "session_expired", "session expired")
                raise ProtocolError(HTTPStatus.NOT_FOUND, "session_not_found", "session not found")
            session.delete_pending = True
            self._tombstones[session_id] = now
            if session.active_requests == 0:
                releases.append(session)
        for old in releases:
            self._release(old)


def parse_gallery_create(value: Any) -> str:
    body = _object(value, "body")
    _strict_keys(body, {"protocol", "source"}, set(), "body")
    _protocol(body["protocol"])
    source = _object(body["source"], "source")
    _strict_keys(source, {"type", "photoId"}, set(), "source")
    if source["type"] != "gallery":
        raise _bad("invalid_source", "JSON source type must be gallery")
    return source["photoId"]


def parse_camera(value: Any, limits: Limits) -> Camera:
    camera = _object(value, "camera")
    _strict_keys(camera, {"convention", "w2c", "intrinsics", "width", "height"}, set(), "camera")
    if camera["convention"] != CAMERA_CONVENTION:
        raise _bad("invalid_camera", f"camera convention must be {CAMERA_CONVENTION}")
    raw_matrix = camera["w2c"]
    if not isinstance(raw_matrix, list) or len(raw_matrix) != 16:
        raise _bad("invalid_camera", "camera.w2c must contain 16 row-major numbers")
    matrix = tuple(_number(item, f"camera.w2c[{index}]") for index, item in enumerate(raw_matrix))
    tolerance = 1e-4
    if any(abs(matrix[12 + i] - expected) > tolerance for i, expected in enumerate((0.0, 0.0, 0.0, 1.0))):
        raise _bad("invalid_camera", "camera.w2c last row must be [0,0,0,1]")
    rows = (matrix[0:3], matrix[4:7], matrix[8:11])
    for i in range(3):
        for j in range(3):
            dot = sum(rows[i][k] * rows[j][k] for k in range(3))
            expected = 1.0 if i == j else 0.0
            if abs(dot - expected) > tolerance:
                raise _bad("invalid_camera", "camera.w2c rotation must be orthonormal")
    determinant = (
        rows[0][0] * (rows[1][1] * rows[2][2] - rows[1][2] * rows[2][1])
        - rows[0][1] * (rows[1][0] * rows[2][2] - rows[1][2] * rows[2][0])
        + rows[0][2] * (rows[1][0] * rows[2][1] - rows[1][1] * rows[2][0])
    )
    if abs(determinant - 1.0) > tolerance:
        raise _bad("invalid_camera", "camera.w2c rotation determinant must be +1")
    intrinsics = _object(camera["intrinsics"], "camera.intrinsics")
    _strict_keys(intrinsics, {"fx", "fy", "cx", "cy"}, set(), "camera.intrinsics")
    width = _integer(camera["width"], "camera.width", minimum=1)
    height = _integer(camera["height"], "camera.height", minimum=1)
    if width * height > limits.max_frame_pixels:
        raise ProtocolError(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, "frame_too_large", "frame exceeds pixel limit")
    fx = _number(intrinsics["fx"], "camera.intrinsics.fx")
    fy = _number(intrinsics["fy"], "camera.intrinsics.fy")
    cx = _number(intrinsics["cx"], "camera.intrinsics.cx")
    cy = _number(intrinsics["cy"], "camera.intrinsics.cy")
    if fx <= 0 or fy <= 0 or not (0 <= cx <= width) or not (0 <= cy <= height):
        raise _bad("invalid_camera", "intrinsics require positive focal lengths and principal point inside the viewport")
    return Camera(CAMERA_CONVENTION, matrix, fx, fy, cx, cy, width, height)


def parse_frame_request(value: Any, limits: Limits) -> tuple[int, str, Camera]:
    body = _object(value, "body")
    _strict_keys(body, {"protocol", "requestId", "mode", "camera"}, set(), "body")
    _protocol(body["protocol"])
    request_id = _integer(body["requestId"], "requestId", minimum=0)
    mode = body["mode"]
    if mode not in {"interactive", "capture"}:
        raise _bad("invalid_request", "mode must be interactive or capture")
    return request_id, mode, parse_camera(body["camera"], limits)


def parse_multipart(content_type: str, body: bytes, limits: Limits) -> tuple[bytes, str, str]:
    message = BytesParser(policy=policy.default).parsebytes(
        f"Content-Type: {content_type}\r\nMIME-Version: 1.0\r\n\r\n".encode("ascii") + body
    )
    if not message.is_multipart():
        raise _bad("invalid_multipart", "request must be multipart/form-data")
    fields: dict[str, str] = {}
    image: tuple[bytes, str, str] | None = None
    for part in message.iter_parts():
        name = part.get_param("name", header="content-disposition")
        if name == "image":
            if image is not None:
                raise _bad("invalid_multipart", "multipart must contain exactly one image")
            data = part.get_payload(decode=True) or b""
            image = (data, part.get_filename() or "upload", part.get_content_type())
        elif name in {"protocol", "sourceType"}:
            if name in fields:
                raise _bad("invalid_multipart", f"duplicate multipart field {name}")
            fields[name] = (part.get_payload(decode=True) or b"").decode("utf-8", "strict")
        else:
            raise _bad("invalid_multipart", f"unknown multipart field {name}")
    if set(fields) != {"protocol", "sourceType"} or image is None:
        raise _bad("invalid_multipart", "multipart requires protocol, sourceType, and image")
    _protocol(fields["protocol"])
    if fields["sourceType"] != "image":
        raise _bad("invalid_source", "multipart sourceType must be image")
    if len(image[0]) > limits.max_upload_bytes:
        raise ProtocolError(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, "upload_too_large", "image exceeds upload limit")
    return image


def make_handler(manager: SessionManager) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        server_version = "WiCiRemoteRender/1"

        def log_message(self, format: str, *args: Any) -> None:
            return

        def _json(self, status: int, payload: Mapping[str, Any]) -> None:
            data = json.dumps(payload, separators=(",", ":")).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

        def _error(self, error: ProtocolError) -> None:
            self._json(error.status, error.payload())

        def _body(self, limit: int) -> bytes:
            raw_length = self.headers.get("Content-Length")
            try:
                length = int(raw_length or "")
            except ValueError as exc:
                raise _bad("invalid_request", "valid Content-Length is required") from exc
            if length < 0:
                raise _bad("invalid_request", "valid Content-Length is required")
            if length > limit:
                raise ProtocolError(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, "body_too_large", "request body exceeds limit")
            data = self.rfile.read(length)
            if len(data) != length:
                raise _bad("invalid_request", "request body is truncated")
            return data

        def _json_body(self) -> Any:
            content_type = self.headers.get_content_type()
            if content_type != "application/json":
                raise _bad("invalid_content_type", "Content-Type must be application/json")
            try:
                return json.loads(self._body(manager.limits.max_json_bytes).decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise _bad("invalid_json", "request body must be valid UTF-8 JSON") from exc

        def do_POST(self) -> None:
            try:
                path = urlsplit(self.path).path
                if path == "/render/sessions":
                    if self.headers.get_content_type() == "multipart/form-data":
                        body = self._body(manager.limits.max_upload_bytes + manager.limits.max_json_bytes)
                        image, filename, content_type = parse_multipart(self.headers.get("Content-Type", ""), body, manager.limits)
                        payload = manager.create_upload(image, filename, content_type)
                    else:
                        payload = manager.create_gallery(parse_gallery_create(self._json_body()))
                    self._json(HTTPStatus.CREATED, payload)
                    return
                prefix, suffix = "/render/sessions/", "/frames"
                if path.startswith(prefix) and path.endswith(suffix):
                    session_id = path[len(prefix) : -len(suffix)]
                    if not session_id or "/" in session_id:
                        raise ProtocolError(HTTPStatus.NOT_FOUND, "route_not_found", "route not found")
                    mime, data, headers = manager.render(session_id, self._json_body())
                    self.send_response(HTTPStatus.OK)
                    self.send_header("Content-Type", mime)
                    self.send_header("Content-Length", str(len(data)))
                    for name, value in headers.items():
                        self.send_header(name, value)
                    self.end_headers()
                    self.wfile.write(data)
                    return
                raise ProtocolError(HTTPStatus.NOT_FOUND, "route_not_found", "route not found")
            except ProtocolError as error:
                self._error(error)
            except Exception:
                self._error(ProtocolError(HTTPStatus.INTERNAL_SERVER_ERROR, "internal_error", "internal server error", retryable=True))

        def do_DELETE(self) -> None:
            try:
                path = urlsplit(self.path).path
                prefix = "/render/sessions/"
                if not path.startswith(prefix) or "/" in path[len(prefix) :] or not path[len(prefix) :]:
                    raise ProtocolError(HTTPStatus.NOT_FOUND, "route_not_found", "route not found")
                manager.delete(path[len(prefix) :])
                self.send_response(HTTPStatus.NO_CONTENT)
                self.send_header("Content-Length", "0")
                self.end_headers()
            except ProtocolError as error:
                self._error(error)

    return Handler


def loopback_server(manager: SessionManager, port: int = 0) -> ThreadingHTTPServer:
    """Construct an unstarted loopback-only server, primarily for adapters/tests."""

    return ThreadingHTTPServer(("127.0.0.1", port), make_handler(manager))
