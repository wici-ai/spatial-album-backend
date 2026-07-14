#!/usr/bin/env python3
"""Resident Gaussian-splat orbit preview endpoint."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import mimetypes
import os
import re
import shutil
import struct
import subprocess
import tempfile
import threading
import time
import traceback
from dataclasses import dataclass
from email import policy
from email.parser import BytesParser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, quote, unquote, urlparse
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import numpy as np
import torch
from PIL import Image, ImageColor, ImageOps
from plyfile import PlyData

from gsplat.rendering import rasterization

try:
    from .remote_render_service import (
        CAMERA_CONVENTION,
        PROTOCOL as REMOTE_RENDER_PROTOCOL,
        Camera as RemoteCamera,
        Limits as RemoteLimits,
        ProtocolError as RemoteProtocolError,
        RemoteRouteMixin,
        RenderedFrame as RemoteRenderedFrame,
        SessionManager as RemoteSessionManager,
        SessionResource as RemoteSessionResource,
    )
except ImportError:  # Direct script execution from backend/.
    from remote_render_service import (  # type: ignore[no-redef]
        CAMERA_CONVENTION,
        PROTOCOL as REMOTE_RENDER_PROTOCOL,
        Camera as RemoteCamera,
        Limits as RemoteLimits,
        ProtocolError as RemoteProtocolError,
        RemoteRouteMixin,
        RenderedFrame as RemoteRenderedFrame,
        SessionManager as RemoteSessionManager,
        SessionResource as RemoteSessionResource,
    )


VIEWER_ROOT = Path(__file__).resolve().parent
REPO_ROOT = VIEWER_ROOT.parent
ASSETS_ROOT = VIEWER_ROOT / "assets"
PHOTOS_ROOT = VIEWER_ROOT / "photos"
PHOTOS_BAKE_ROOT = VIEWER_ROOT / "photos_bake"
PLYS_ROOT = VIEWER_ROOT / "plys"
PHOTOS_USER_ROOT = VIEWER_ROOT / "photos_user"
USER_UPLOADS_ROOT = PHOTOS_USER_ROOT / "sources"
USER_BAKE_ROOT = PHOTOS_USER_ROOT / "bakes"
USER_GALLERY_PATH = PHOTOS_USER_ROOT / "user_gallery.json"
GALLERY_PATH = VIEWER_ROOT / "photos" / "gallery.json"
CACHE_DIR = Path(os.environ.get("WICI_ORBIT_CACHE_DIR", "/tmp/wici-orbit-preview-cache"))
SPLAT_STREAM_CACHE_DIR = Path(os.environ.get("WICI_SPLAT_STREAM_CACHE_DIR", str(CACHE_DIR / "splat-stream")))
THUMB_CACHE_DIR = Path(os.environ.get("WICI_GALLERY_THUMB_CACHE_DIR", str(CACHE_DIR / "thumbs")))
DEFAULT_THUMB_SIZE = int(os.environ.get("WICI_GALLERY_THUMB_SIZE", "320"))
MAX_GPU_ASSETS = int(os.environ.get("WICI_ORBIT_MAX_GPU_ASSETS", "4"))
DEFAULT_WIDTH = int(os.environ.get("WICI_ORBIT_WIDTH", "480"))
DEFAULT_HEIGHT = int(os.environ.get("WICI_ORBIT_HEIGHT", str(DEFAULT_WIDTH)))
DEFAULT_MAX_HEIGHT = int(os.environ.get("WICI_ORBIT_MAX_HEIGHT", "720"))
DEFAULT_FPS = int(os.environ.get("WICI_ORBIT_FPS", "24"))
DEFAULT_WEBP_WIDTH = int(os.environ.get("WICI_ORBIT_WEBP_WIDTH", "360"))
DEFAULT_WEBP_HEIGHT = int(os.environ.get("WICI_ORBIT_WEBP_HEIGHT", str(DEFAULT_WEBP_WIDTH)))
DEFAULT_WEBP_FPS = int(os.environ.get("WICI_ORBIT_WEBP_FPS", "12"))
DEFAULT_SECONDS = float(os.environ.get("WICI_ORBIT_SECONDS", "2.5"))
DEFAULT_AZ_DEG = float(os.environ.get("WICI_ORBIT_AZ_DEG", "5"))
DEFAULT_POL_DEG = float(os.environ.get("WICI_ORBIT_POL_DEG", "1"))
DEFAULT_PORTRAIT_AZ_DEG = float(os.environ.get("WICI_ORBIT_PORTRAIT_AZ_DEG", "1"))
DEFAULT_PORTRAIT_POL_DEG = float(os.environ.get("WICI_ORBIT_PORTRAIT_POL_DEG", "5"))
DEFAULT_BITRATE_KBPS = int(os.environ.get("WICI_ORBIT_BITRATE_KBPS", "450"))
DEFAULT_WEBP_BITRATE_KBPS = int(os.environ.get("WICI_ORBIT_WEBP_BITRATE_KBPS", "180"))
DEFAULT_WEBP_QUALITY = int(os.environ.get("WICI_ORBIT_WEBP_QUALITY", "32"))
DEFAULT_BG_HEX = os.environ.get("WICI_ORBIT_BG", "101012")
SHARP_INFER_URL = os.environ.get("WICI_SHARP_INFER_URL", "http://127.0.0.1:8770/infer")
SHARP_INFER_UPLOAD_URL = os.environ.get(
    "WICI_SHARP_INFER_UPLOAD_URL",
    SHARP_INFER_URL.rsplit("/", 1)[0] + "/infer_upload",
)
ORBIT_INPROC_SHARP = os.environ.get("WICI_ORBIT_INPROC_SHARP", "0").strip().lower() in {"1", "true", "yes", "on"}
INGEST_MAX_BODY_BYTES = int(float(os.environ.get("WICI_INGEST_MAX_UPLOAD_MB", "20")) * 1024 * 1024)
INGEST_MAX_SOURCE_SIDE = int(os.environ.get("WICI_INGEST_MAX_SOURCE_SIDE", "1800"))
INGEST_MAX_PIXELS = int(float(os.environ.get("WICI_INGEST_MAX_PIXELS", "16000000")))
INGEST_MIN_FREE_BYTES = int(float(os.environ.get("WICI_INGEST_MIN_FREE_GB", "10")) * 1024 * 1024 * 1024)
INGEST_JPEG_QUALITY = int(os.environ.get("WICI_INGEST_JPEG_QUALITY", "92"))
INGEST_SHARP_TIMEOUT_SECONDS = float(os.environ.get("WICI_INGEST_SHARP_TIMEOUT_SECONDS", "180"))
REMOTE_RENDER_ENABLED = os.environ.get("WICI_REMOTE_RENDER_ENABLED", "1").strip().lower() not in {"0", "false", "no", "off"}
REMOTE_INTERACTIVE_JPEG_QUALITY = int(os.environ.get("WICI_REMOTE_INTERACTIVE_JPEG_QUALITY", "82"))
REMOTE_CAPTURE_JPEG_QUALITY = int(os.environ.get("WICI_REMOTE_CAPTURE_JPEG_QUALITY", "92"))
REMOTE_LIMITS = RemoteLimits(
    max_sessions=int(os.environ.get("WICI_REMOTE_MAX_SESSIONS", "4")),
    idle_ttl_seconds=float(os.environ.get("WICI_REMOTE_SESSION_TTL_SECONDS", "120")),
    max_frame_pixels=int(os.environ.get("WICI_REMOTE_MAX_FRAME_PIXELS", str(1280 * 1280))),
    max_upload_bytes=int(float(os.environ.get("WICI_REMOTE_MAX_UPLOAD_MB", "20")) * 1024 * 1024),
)
SPLAT_STREAM_SHUFFLE_SEED = int(os.environ.get("WICI_SPLAT_STREAM_SHUFFLE_SEED", "1337"))
SH_C0 = 0.28209479177387814
SPLAT_ROW_BYTES = 32
VERSION = "1"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
INGEST_COMPACT_ACCEPT_HEADER = "X-Wici-Splat-Accept"
INGEST_COMPACT_STREAM_HEADER = "X-Wici-Splat-Stream"
COMPACT_SPLAT_FORMAT = "fp16v1"
COMPACT_SPLAT_IDENTITY_ENCODING = "identity"
COMPACT_SPLAT_MAGIC = b"WICISF16"
COMPACT_SPLAT_VERSION = 1
COMPACT_SPLAT_ROW_BYTES = 20
COMPACT_SPLAT_HEADER = struct.Struct("<8sIIII")
COMPACT_SPLAT_HEADER_BYTES = COMPACT_SPLAT_HEADER.size

_lock = threading.Lock()
_inproc_sharp_lock = threading.Lock()
_inproc_sharp_module: Any | None = None
_asset_cache: dict[str, "LoadedAsset"] = {}
_gallery_cache: tuple[float, list[dict[str, Any]]] | None = None
_user_gallery_cache: tuple[float, list[dict[str, Any]]] | None = None
_last_preview: dict[str, Any] | None = None
_last_ingest: dict[str, Any] | None = None
_image_dim_cache: dict[str, tuple[int, int, int, int]] = {}
_ply_camera_cache: dict[str, tuple[int, int, int, int, np.ndarray]] = {}


@dataclass
class AssetSpec:
    path: Path
    kind: str
    source_width: int
    source_height: int
    k_source: np.ndarray
    photo_id: str | None
    title: str | None


@dataclass
class LoadedAsset:
    spec: AssetSpec
    means: torch.Tensor
    quats: torch.Tensor
    scales: torch.Tensor
    opacities: torch.Tensor
    colors_linear: torch.Tensor
    count: int
    loaded_seconds: float
    last_used: float


@dataclass
class RemoteAssetHandle:
    """Session-owned tensor bundle; never placed in the legacy asset cache."""

    asset: LoadedAsset | None
    released: bool = False


@dataclass(frozen=True)
class RenderParams:
    width: int
    height: int
    fps: int
    seconds: float
    frames: int
    az_deg: float
    pol_deg: float
    bitrate_kbps: int
    bg_hex: str
    output_format: str


@dataclass
class OrbitFrameBuffer:
    frames: list[np.ndarray]
    frame_times: list[float]
    render_seconds: float


@dataclass(frozen=True)
class IngestUpload:
    image: bytes
    filename: str
    fields: dict[str, str]


def log_event(event: str, **fields: Any) -> None:
    print(json.dumps({"event": event, **fields}, separators=(",", ":")), flush=True)


def json_response(handler: BaseHTTPRequestHandler, status: int, data: Any) -> None:
    raw = json.dumps(data, separators=(",", ":")).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(raw)))
    handler.send_header("Access-Control-Allow-Origin", "*")
    handler.send_header("Access-Control-Allow-Methods", "GET,POST,DELETE,OPTIONS")
    handler.send_header("Access-Control-Allow-Headers", f"Content-Type, {INGEST_COMPACT_ACCEPT_HEADER}, {INGEST_COMPACT_STREAM_HEADER}")
    handler.send_header("Cross-Origin-Resource-Policy", "cross-origin")
    handler.end_headers()
    if status != 204:
        handler.wfile.write(raw)


def image_response(handler: BaseHTTPRequestHandler, path: Path, *, cache_hit: bool) -> None:
    data_len = path.stat().st_size
    handler.send_response(200)
    handler.send_header("Content-Type", "image/jpeg")
    handler.send_header("Content-Length", str(data_len))
    handler.send_header("Cache-Control", "public, max-age=86400")
    handler.send_header("Access-Control-Allow-Origin", "*")
    handler.send_header("Access-Control-Allow-Methods", "GET,OPTIONS")
    handler.send_header("Access-Control-Expose-Headers", "X-Thumbnail-Cache, X-Thumbnail-Size-Bytes")
    handler.send_header("Cross-Origin-Resource-Policy", "cross-origin")
    handler.send_header("X-Thumbnail-Cache", "hit" if cache_hit else "miss")
    handler.send_header("X-Thumbnail-Size-Bytes", str(data_len))
    handler.end_headers()
    with path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            handler.wfile.write(chunk)


def source_response(handler: BaseHTTPRequestHandler, path: Path, *, record: dict[str, Any]) -> None:
    data_len = path.stat().st_size
    content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    handler.send_response(200)
    handler.send_header("Content-Type", content_type)
    handler.send_header("Content-Length", str(data_len))
    handler.send_header("Content-Disposition", f'inline; filename="{path.name}"')
    handler.send_header("Cache-Control", "public, max-age=86400")
    handler.send_header("Access-Control-Allow-Origin", "*")
    handler.send_header("Access-Control-Allow-Methods", "GET,POST,OPTIONS")
    handler.send_header(
        "Access-Control-Expose-Headers",
        "X-Source-Photo-Id, X-Source-Width, X-Source-Height, X-Source-Size-Bytes",
    )
    handler.send_header("Cross-Origin-Resource-Policy", "cross-origin")
    handler.send_header("X-Source-Photo-Id", str(record["photoId"]))
    if record.get("sourceWidth"):
        handler.send_header("X-Source-Width", str(int(record["sourceWidth"])))
    if record.get("sourceHeight"):
        handler.send_header("X-Source-Height", str(int(record["sourceHeight"])))
    handler.send_header("X-Source-Size-Bytes", str(data_len))
    handler.end_headers()
    with path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            handler.wfile.write(chunk)


def orbit_response(
    handler: BaseHTTPRequestHandler,
    path: Path,
    *,
    filename: str,
    meta: dict[str, Any],
    cache_hit: bool,
) -> None:
    data_len = path.stat().st_size
    output_format = str(meta.get("format") or "mp4")
    content_type = "image/webp" if output_format == "webp" else "video/mp4"
    handler.send_response(200)
    handler.send_header("Content-Type", content_type)
    handler.send_header("Content-Length", str(data_len))
    handler.send_header("Content-Disposition", f'inline; filename="{filename}"')
    handler.send_header("Cache-Control", "public, max-age=86400")
    handler.send_header("Access-Control-Allow-Origin", "*")
    handler.send_header("Access-Control-Allow-Methods", "GET,OPTIONS")
    handler.send_header(
        "Access-Control-Expose-Headers",
        (
            "X-Orbit-Cache, X-Orbit-Seconds, X-Orbit-Generate-Seconds, "
            "X-Orbit-Frames, X-Orbit-FPS, X-Orbit-Size-Bytes, X-Orbit-Width, "
            "X-Orbit-Height, X-Orbit-Format, X-Orbit-Az-Deg, X-Orbit-Pol-Deg, "
            "X-Orbit-Photo-Id"
        ),
    )
    handler.send_header("Cross-Origin-Resource-Policy", "cross-origin")
    handler.send_header("X-Orbit-Cache", "hit" if cache_hit else "miss")
    handler.send_header("X-Orbit-Format", output_format)
    handler.send_header("X-Orbit-Seconds", f"{float(meta.get('durationSeconds', 0.0)):.6f}")
    handler.send_header("X-Orbit-Generate-Seconds", f"{float(meta.get('generateSeconds', 0.0)):.6f}")
    handler.send_header("X-Orbit-Frames", str(int(meta.get("frames", 0))))
    handler.send_header("X-Orbit-FPS", str(int(meta.get("fps", 0))))
    handler.send_header("X-Orbit-Size-Bytes", str(data_len))
    handler.send_header("X-Orbit-Width", str(int(meta.get("width", 0))))
    handler.send_header("X-Orbit-Height", str(int(meta.get("height", 0))))
    handler.send_header("X-Orbit-Az-Deg", str(meta.get("azDeg", "")))
    handler.send_header("X-Orbit-Pol-Deg", str(meta.get("polDeg", "")))
    if meta.get("photoId"):
        handler.send_header("X-Orbit-Photo-Id", str(meta["photoId"]))
    handler.end_headers()
    with path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            handler.wfile.write(chunk)


def orbit_bytes_response(
    handler: BaseHTTPRequestHandler,
    data: bytes,
    *,
    filename: str,
    meta: dict[str, Any],
) -> None:
    output_format = str(meta.get("format") or "webp")
    content_type = "image/webp" if output_format == "webp" else "application/octet-stream"
    handler.send_response(200)
    handler.send_header("Content-Type", content_type)
    handler.send_header("Content-Length", str(len(data)))
    handler.send_header("Content-Disposition", f'inline; filename="{filename}"')
    handler.send_header("Cache-Control", "no-store")
    handler.send_header("Access-Control-Allow-Origin", "*")
    handler.send_header("Access-Control-Allow-Methods", "POST,OPTIONS")
    handler.send_header(
        "Access-Control-Expose-Headers",
        (
            "X-Orbit-Storage-Policy, X-Orbit-Seconds, X-Orbit-Generate-Seconds, "
            "X-Orbit-Frames, X-Orbit-FPS, X-Orbit-Size-Bytes, X-Orbit-Width, "
            "X-Orbit-Height, X-Orbit-Format, X-Orbit-Az-Deg, X-Orbit-Pol-Deg, "
            "X-Orbit-Photo-Id, X-Orbit-Timing-Json"
        ),
    )
    handler.send_header("Cross-Origin-Resource-Policy", "cross-origin")
    handler.send_header("X-Orbit-Storage-Policy", "stateless")
    handler.send_header("X-Orbit-Format", output_format)
    handler.send_header("X-Orbit-Seconds", f"{float(meta.get('durationSeconds', 0.0)):.6f}")
    handler.send_header("X-Orbit-Generate-Seconds", f"{float(meta.get('generateSeconds', 0.0)):.6f}")
    handler.send_header("X-Orbit-Frames", str(int(meta.get("frames", 0))))
    handler.send_header("X-Orbit-FPS", str(int(meta.get("fps", 0))))
    handler.send_header("X-Orbit-Size-Bytes", str(len(data)))
    handler.send_header("X-Orbit-Width", str(int(meta.get("width", 0))))
    handler.send_header("X-Orbit-Height", str(int(meta.get("height", 0))))
    handler.send_header("X-Orbit-Az-Deg", str(meta.get("azDeg", "")))
    handler.send_header("X-Orbit-Pol-Deg", str(meta.get("polDeg", "")))
    if meta.get("photoId"):
        handler.send_header("X-Orbit-Photo-Id", str(meta["photoId"]))
    timing = meta.get("timing") if isinstance(meta.get("timing"), dict) else {}
    handler.send_header("X-Orbit-Timing-Json", json.dumps(timing, separators=(",", ":")))
    handler.end_headers()
    handler.wfile.write(data)


def ply_response(
    handler: BaseHTTPRequestHandler,
    path: Path,
    *,
    photo_id: str | None,
    requested_density: str,
    served_density: str,
    fallback: bool,
) -> None:
    data_len = path.stat().st_size
    filename = f"{photo_id or path.stem}.ply"
    handler.send_response(200)
    handler.send_header("Content-Type", "application/octet-stream")
    handler.send_header("Content-Length", str(data_len))
    handler.send_header("Content-Disposition", f'attachment; filename="{filename}"')
    handler.send_header("Cache-Control", "public, max-age=86400")
    handler.send_header("Access-Control-Allow-Origin", "*")
    handler.send_header("Access-Control-Allow-Methods", "GET,OPTIONS")
    handler.send_header(
        "Access-Control-Expose-Headers",
        (
            "X-Splat-Format, X-Splat-Density, X-Splat-Requested-Density, "
            "X-Splat-Fallback, X-Splat-Size-Bytes, X-Splat-Photo-Id, X-Splat-Path"
        ),
    )
    handler.send_header("Cross-Origin-Resource-Policy", "cross-origin")
    handler.send_header("X-Splat-Format", "ply")
    handler.send_header("X-Splat-Density", served_density)
    handler.send_header("X-Splat-Requested-Density", requested_density)
    handler.send_header("X-Splat-Fallback", "true" if fallback else "false")
    handler.send_header("X-Splat-Size-Bytes", str(data_len))
    handler.send_header("X-Splat-Path", str(path))
    if photo_id:
        handler.send_header("X-Splat-Photo-Id", photo_id)
    handler.end_headers()
    with path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            handler.wfile.write(chunk)


def splat_stream_response(handler: BaseHTTPRequestHandler, path: Path, *, meta: dict[str, Any], cache_hit: bool) -> None:
    data_len = path.stat().st_size
    photo_id = meta.get("photoId") or path.stem
    density = str(meta.get("servedDensity") or "unknown")
    filename = f"{photo_id}-{density}-shuffled.splat"
    handler.send_response(200)
    handler.send_header("Content-Type", "application/octet-stream")
    handler.send_header("Content-Length", str(data_len))
    handler.send_header("Content-Disposition", f'attachment; filename="{filename}"')
    handler.send_header("Cache-Control", "public, max-age=86400")
    handler.send_header("Accept-Ranges", "bytes")
    handler.send_header("Access-Control-Allow-Origin", "*")
    handler.send_header("Access-Control-Allow-Methods", "GET,OPTIONS")
    handler.send_header(
        "Access-Control-Expose-Headers",
        (
            "X-Splat-Stream-Cache, X-Splat-Stream-Format, X-Splat-Stream-Row-Bytes, "
            "X-Splat-Stream-Num-Gaussians, X-Splat-Stream-Size-Bytes, "
            "X-Splat-Stream-Shuffle-Seed, X-Splat-Stream-Density, "
            "X-Splat-Stream-Requested-Density, X-Splat-Stream-Fallback, "
            "X-Splat-Stream-Source-Path, X-Splat-Stream-Photo-Id, "
            "X-Src-Width, X-Src-Height, X-Cam-Fx, X-Cam-Fy, X-Cam-Cx, X-Cam-Cy"
        ),
    )
    handler.send_header("Cross-Origin-Resource-Policy", "cross-origin")
    handler.send_header("X-Splat-Stream-Cache", "hit" if cache_hit else "miss")
    handler.send_header("X-Splat-Stream-Format", "splat")
    handler.send_header("X-Splat-Stream-Row-Bytes", str(SPLAT_ROW_BYTES))
    handler.send_header("X-Splat-Stream-Num-Gaussians", str(int(meta.get("numGaussians", 0))))
    handler.send_header("X-Splat-Stream-Size-Bytes", str(data_len))
    handler.send_header("X-Splat-Stream-Shuffle-Seed", str(int(meta.get("shuffleSeed", SPLAT_STREAM_SHUFFLE_SEED))))
    handler.send_header("X-Splat-Stream-Density", str(meta.get("servedDensity", "")))
    handler.send_header("X-Splat-Stream-Requested-Density", str(meta.get("requestedDensity", "")))
    handler.send_header("X-Splat-Stream-Fallback", "true" if meta.get("fallback") else "false")
    handler.send_header("X-Splat-Stream-Source-Path", str(meta.get("sourcePath", "")))
    if meta.get("photoId"):
        handler.send_header("X-Splat-Stream-Photo-Id", str(meta["photoId"]))
    for header, key in (
        ("X-Src-Width", "sourceWidth"),
        ("X-Src-Height", "sourceHeight"),
        ("X-Cam-Fx", "camFx"),
        ("X-Cam-Fy", "camFy"),
        ("X-Cam-Cx", "camCx"),
        ("X-Cam-Cy", "camCy"),
    ):
        value = meta.get(key)
        if value is not None:
            handler.send_header(header, str(value))
    handler.end_headers()
    with path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            handler.wfile.write(chunk)


def write_http_chunk(handler: BaseHTTPRequestHandler, data: bytes) -> None:
    if not data:
        return
    handler.wfile.write(f"{len(data):X}\r\n".encode("ascii"))
    handler.wfile.write(data)
    handler.wfile.write(b"\r\n")
    handler.wfile.flush()


def ingest_splat_chunked_response(
    handler: BaseHTTPRequestHandler,
    status: int,
    chunks: Any,
    result: dict[str, Any],
) -> dict[str, Any]:
    photo_id = str(result.get("photoId") or "user-upload")
    timing = result.get("timing") if isinstance(result.get("timing"), dict) else {}
    request_start_counter = timing.pop("_requestStartPerfCounter", None)
    timing_json = json.dumps(timing, separators=(",", ":"))
    splat_format = str(result.get("responseSplatFormat") or result.get("splatFormat") or COMPACT_SPLAT_FORMAT)
    row_bytes = int(result.get("responseSplatRowBytes") or COMPACT_SPLAT_ROW_BYTES)
    payload_size = int(result.get("responsePayloadBytes") or 0)
    fp32_size = int(result.get("fp32SplatBytes") or result.get("splatBytes") or 0)
    header_bytes = int(result.get("responseSplatHeaderBytes") or 0)

    handler.send_response(status)
    handler.send_header("Content-Type", "application/octet-stream")
    handler.send_header("Transfer-Encoding", "chunked")
    handler.send_header("Content-Disposition", f'attachment; filename="{photo_id}.splat"')
    handler.send_header("Cache-Control", "no-store")
    handler.send_header("Access-Control-Allow-Origin", "*")
    handler.send_header("Access-Control-Allow-Methods", "POST,OPTIONS")
    handler.send_header(
        "Access-Control-Expose-Headers",
        (
            "X-Ingest-Photo-Id, X-Ingest-Storage-Policy, X-Ingest-Timing-Json, "
            "X-Splat-Format, X-Splat-Row-Bytes, X-Splat-Num-Gaussians, "
            "X-Splat-Size-Bytes, X-Src-Width, X-Src-Height, X-Cam-Fx, "
            "X-Cam-Fy, X-Cam-Cx, X-Cam-Cy, X-Splat-Encoding, "
            "X-Splat-Payload-Size-Bytes, X-Splat-Fp32-Size-Bytes, "
            "X-Splat-Compressed-Size-Bytes, X-Splat-Header-Bytes, X-Splat-Transfer"
        ),
    )
    handler.send_header("Cross-Origin-Resource-Policy", "cross-origin")
    handler.send_header("X-Ingest-Photo-Id", photo_id)
    handler.send_header("X-Ingest-Storage-Policy", "stateless")
    handler.send_header("X-Ingest-Timing-Json", timing_json)
    handler.send_header("X-Splat-Format", splat_format)
    handler.send_header("X-Splat-Row-Bytes", str(row_bytes))
    handler.send_header("X-Splat-Num-Gaussians", str(int(result.get("numGaussians") or 0)))
    handler.send_header("X-Splat-Size-Bytes", str(payload_size))
    handler.send_header("X-Splat-Encoding", COMPACT_SPLAT_IDENTITY_ENCODING)
    handler.send_header("X-Splat-Payload-Size-Bytes", str(payload_size))
    handler.send_header("X-Splat-Fp32-Size-Bytes", str(fp32_size))
    handler.send_header("X-Splat-Compressed-Size-Bytes", "0")
    handler.send_header("X-Splat-Header-Bytes", str(header_bytes))
    handler.send_header("X-Splat-Transfer", "chunked")
    for header, key in (
        ("X-Src-Width", "sourceWidth"),
        ("X-Src-Height", "sourceHeight"),
        ("X-Cam-Fx", "camFx"),
        ("X-Cam-Fy", "camFy"),
        ("X-Cam-Cx", "camCx"),
        ("X-Cam-Cy", "camCy"),
    ):
        value = result.get(key)
        if value is not None:
            handler.send_header(header, str(value))
    handler.end_headers()

    stream_start = time.perf_counter()
    first_chunk_seconds: float | None = None
    bytes_sent = 0
    chunk_count = 0
    try:
        for chunk in chunks:
            if not chunk:
                continue
            if first_chunk_seconds is None:
                first_chunk_seconds = time.perf_counter() - stream_start
            write_http_chunk(handler, chunk)
            bytes_sent += len(chunk)
            chunk_count += 1
        handler.wfile.write(b"0\r\n\r\n")
        handler.wfile.flush()
    finally:
        timing["streamWriteSeconds"] = time.perf_counter() - stream_start
        timing["firstChunkWriteSeconds"] = first_chunk_seconds
        timing["streamBytesSent"] = bytes_sent
        timing["streamChunkCount"] = chunk_count
        timing["totalSeconds"] = time.perf_counter() - (request_start_counter or stream_start)
    return {
        "bytesSent": bytes_sent,
        "chunkCount": chunk_count,
        "firstChunkWriteSeconds": first_chunk_seconds,
        "streamWriteSeconds": timing.get("streamWriteSeconds"),
    }


def load_gallery() -> list[dict[str, Any]]:
    global _gallery_cache
    if not GALLERY_PATH.exists():
        return []
    mtime = GALLERY_PATH.stat().st_mtime
    if _gallery_cache and _gallery_cache[0] == mtime:
        return _gallery_cache[1]
    data = json.loads(GALLERY_PATH.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"gallery is not a list: {GALLERY_PATH}")
    _gallery_cache = (mtime, data)
    return data


def ensure_user_dirs() -> None:
    USER_UPLOADS_ROOT.mkdir(parents=True, exist_ok=True)
    USER_BAKE_ROOT.mkdir(parents=True, exist_ok=True)


def load_user_gallery() -> list[dict[str, Any]]:
    global _user_gallery_cache
    if not USER_GALLERY_PATH.exists():
        return []
    mtime = USER_GALLERY_PATH.stat().st_mtime
    if _user_gallery_cache and _user_gallery_cache[0] == mtime:
        return _user_gallery_cache[1]
    data = json.loads(USER_GALLERY_PATH.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"user gallery is not a list: {USER_GALLERY_PATH}")
    _user_gallery_cache = (mtime, data)
    return data


def save_user_gallery(entries: list[dict[str, Any]]) -> None:
    global _user_gallery_cache
    ensure_user_dirs()
    tmp_path = USER_GALLERY_PATH.with_suffix(".tmp.json")
    tmp_path.write_text(json.dumps(entries, indent=2, sort_keys=True), encoding="utf-8")
    tmp_path.replace(USER_GALLERY_PATH)
    _user_gallery_cache = (USER_GALLERY_PATH.stat().st_mtime, entries)


def gallery_entry(photo_id: str, *, include_user: bool = True) -> dict[str, Any] | None:
    for entry in load_gallery():
        if entry.get("id") == photo_id:
            return entry
    if include_user:
        for entry in load_user_gallery():
            if entry.get("id") == photo_id:
                return entry
    return None


def slugify_photo_id(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "-", value).strip("-").lower()
    return slug or value.lower()


def photo_id_from_asset_stem(stem: str) -> str:
    name = stem
    if name.startswith("sharp-"):
        name = name[len("sharp-") :]
    name = re.sub(r"-sharp-warm\d+$", "", name)
    for suffix in ("-350k", "-original", "-reproj"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
    return slugify_photo_id(name)


def image_rank(path: Path) -> int:
    try:
        rel = path.resolve().relative_to(REPO_ROOT.resolve())
    except ValueError:
        return 100
    parts = rel.parts
    if parts[:2] == ("backend", "assets"):
        return 0
    if parts[:2] == ("backend", "photos"):
        return 1
    if parts[:2] == ("backend", "plys"):
        return 2
    if parts[:2] == ("backend", "photos_user"):
        return 4
    return 10


def ply_rank(path: Path) -> int:
    stem = path.stem.lower()
    if path.parent == USER_BAKE_ROOT and not stem.endswith("-350k"):
        return 0
    if path.parent == PHOTOS_BAKE_ROOT and not stem.endswith("-350k"):
        return 0
    if path.parent == PLYS_ROOT and stem.startswith("sharp-") and not stem.endswith("-350k") and "warm" not in stem:
        return 1
    if stem.endswith("-350k"):
        return 2
    if "warm" in stem or stem.endswith("-reproj"):
        return 5
    return 3


def iter_files(root: Path, suffixes: set[str]) -> list[Path]:
    if not root.exists():
        return []
    return sorted(path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in suffixes)


def read_image_dimensions(path: Path) -> tuple[int, int]:
    stat = path.stat()
    key = str(path.resolve())
    cached = _image_dim_cache.get(key)
    if cached and cached[0] == stat.st_mtime_ns and cached[1] == stat.st_size:
        return cached[2], cached[3]
    with Image.open(path) as image:
        width, height = ImageOps.exif_transpose(image).size
    _image_dim_cache[key] = (stat.st_mtime_ns, stat.st_size, int(width), int(height))
    return int(width), int(height)


def camera_dict(source_width: int, source_height: int, k_source: np.ndarray) -> dict[str, Any]:
    return {
        "sourceWidth": int(source_width),
        "sourceHeight": int(source_height),
        "camFx": float(k_source[0, 0]),
        "camFy": float(k_source[1, 1]),
        "camCx": float(k_source[0, 2]),
        "camCy": float(k_source[1, 2]),
    }


def apply_record_camera(record: dict[str, Any], source_width: int, source_height: int, k_source: np.ndarray) -> None:
    record.update(camera_dict(source_width, source_height, k_source))


def get_gallery_record(records: dict[str, dict[str, Any]], photo_id: str) -> dict[str, Any]:
    return records.setdefault(
        photo_id,
        {
            "photoId": photo_id,
            "title": photo_id.replace("-", " ").title(),
            "sourcePath": None,
            "sourceWidth": None,
            "sourceHeight": None,
            "plyPath": None,
            "splatPath": None,
            "galleryOrder": None,
        },
    )


def set_record_source(record: dict[str, Any], path: Path, *, width: int | None = None, height: int | None = None) -> None:
    current = record.get("sourcePath")
    if current is None or image_rank(path) < image_rank(Path(str(current))):
        if width is None or height is None:
            try:
                width, height = read_image_dimensions(path)
            except Exception:
                width, height = None, None
        record["sourcePath"] = path
        if width:
            record["sourceWidth"] = int(width)
        if height:
            record["sourceHeight"] = int(height)


def set_record_ply(record: dict[str, Any], path: Path) -> None:
    current = record.get("plyPath")
    if current is None or ply_rank(path) < ply_rank(Path(str(current))):
        record["plyPath"] = path
        try:
            source_width, source_height, k_source = read_ply_camera(path)
            apply_record_camera(record, source_width, source_height, k_source)
        except Exception as exc:
            log_event("ply_camera_read_failed", path=str(path), error=repr(exc))


def set_record_splat(record: dict[str, Any], path: Path) -> None:
    current = record.get("splatPath")
    if current is None or path.name < Path(str(current)).name:
        record["splatPath"] = path


def resolve_entry_path(value: Any, root: Path) -> Path | None:
    if value in {None, ""}:
        return None
    path = Path(str(value))
    if path.is_absolute():
        return path
    if path.parts and path.parts[0] == "backend":
        return REPO_ROOT / path
    return root / path


def entry_int(entry: dict[str, Any], *keys: str) -> int | None:
    for key in keys:
        value = entry.get(key)
        if value in {None, ""}:
            continue
        try:
            return int(value)
        except (TypeError, ValueError):
            continue
    return None


def add_user_gallery_records(records: dict[str, dict[str, Any]]) -> None:
    for order, entry in enumerate(load_user_gallery()):
        photo_id = str(entry.get("id") or "").strip()
        if not photo_id:
            continue
        record = get_gallery_record(records, photo_id)
        record["title"] = entry.get("title") or record["title"]
        record["galleryOrder"] = 100000 + order
        record["userUpload"] = True
        width = entry_int(entry, "width", "sourceWidth")
        height = entry_int(entry, "height", "sourceHeight")
        source = resolve_entry_path(entry.get("source_path") or entry.get("jpg"), PHOTOS_USER_ROOT)
        if source and source.exists():
            set_record_source(record, source, width=width, height=height)
        ply = resolve_entry_path(entry.get("ply") or f"bakes/{photo_id}.ply", PHOTOS_USER_ROOT)
        if ply and ply.exists():
            set_record_ply(record, ply)
        splat = resolve_entry_path(entry.get("splat") or f"bakes/{photo_id}.splat", PHOTOS_USER_ROOT)
        if splat and splat.exists():
            set_record_splat(record, splat)


def build_gallery_records(*, include_user: bool = False) -> list[dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}

    for order, entry in enumerate(load_gallery()):
        photo_id = str(entry.get("id") or "").strip()
        if not photo_id:
            continue
        record = get_gallery_record(records, photo_id)
        record["title"] = entry.get("title") or record["title"]
        record["galleryOrder"] = order
        width = int(entry["width"]) if entry.get("width") else None
        height = int(entry["height"]) if entry.get("height") else None
        jpg = entry.get("jpg")
        if jpg:
            source = PHOTOS_ROOT / str(jpg)
            if source.exists():
                set_record_source(record, source, width=width, height=height)
        splat_name = entry.get("splat")
        if splat_name:
            splat_path = PHOTOS_BAKE_ROOT / str(splat_name)
            if splat_path.exists():
                set_record_splat(record, splat_path)
        ply_path = PHOTOS_BAKE_ROOT / f"{photo_id}.ply"
        if ply_path.exists():
            set_record_ply(record, ply_path)

    if include_user:
        add_user_gallery_records(records)

    image_roots = [PHOTOS_ROOT, PLYS_ROOT]
    if include_user:
        image_roots.append(USER_UPLOADS_ROOT)
    for root in image_roots:
        for image_path in iter_files(root, IMAGE_SUFFIXES):
            photo_id = photo_id_from_asset_stem(image_path.stem)
            set_record_source(get_gallery_record(records, photo_id), image_path)

    bake_roots = [PHOTOS_BAKE_ROOT, PHOTOS_ROOT, PLYS_ROOT]
    if include_user:
        bake_roots.insert(0, USER_BAKE_ROOT)
    for root in bake_roots:
        for ply_path in iter_files(root, {".ply"}):
            photo_id = photo_id_from_asset_stem(ply_path.stem)
            set_record_ply(get_gallery_record(records, photo_id), ply_path)
        for splat_path in iter_files(root, {".splat"}):
            photo_id = photo_id_from_asset_stem(splat_path.stem)
            set_record_splat(get_gallery_record(records, photo_id), splat_path)

    def sort_key(record: dict[str, Any]) -> tuple[int, int, str]:
        has_bake = bool(record.get("plyPath") or record.get("splatPath"))
        gallery_order = record.get("galleryOrder")
        if gallery_order is not None:
            return (0, int(gallery_order), str(record["photoId"]))
        if has_bake:
            return (1, 0, str(record["photoId"]))
        return (2, 0, str(record["photoId"]))

    return sorted(records.values(), key=sort_key)


def public_gallery_items(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    items = []
    for record in records:
        photo_id = str(record["photoId"])
        item = {
            "photoId": photo_id,
            "thumbnailUrl": f"/thumbnail?photoId={quote(photo_id)}",
            "sourceUrl": f"/source?photoId={quote(photo_id)}",
            "hasOrbit": bool(record.get("plyPath") or record.get("splatPath")),
            "hasSplat": bool(record.get("plyPath")),
        }
        if record.get("userUpload"):
            item["userUpload"] = True
        for key in ("sourceWidth", "sourceHeight", "camFx", "camFy", "camCx", "camCy"):
            value = record.get(key)
            if value is not None:
                item[key] = value
        items.append(item)
    return items


def gallery_record_for_photo_id(photo_id: str) -> dict[str, Any] | None:
    for record in build_gallery_records(include_user=True):
        if record["photoId"] == photo_id:
            return record
    return None


def thumbnail_cache_path(photo_id: str) -> Path:
    safe_name = slugify_photo_id(photo_id)
    return THUMB_CACHE_DIR / f"{safe_name}-{DEFAULT_THUMB_SIZE}.jpg"


def get_or_create_thumbnail(record: dict[str, Any], *, force: bool = False) -> tuple[Path, bool]:
    source_path = record.get("sourcePath")
    if not source_path:
        raise FileNotFoundError(f"no source image found for photoId={record['photoId']!r}")
    source = Path(str(source_path))
    if not source.exists():
        raise FileNotFoundError(f"source image does not exist: {source}")
    photo_id = str(record["photoId"])
    out_path = thumbnail_cache_path(photo_id)
    THUMB_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    if (
        out_path.exists()
        and not force
        and out_path.stat().st_mtime >= source.stat().st_mtime
        and out_path.stat().st_size > 0
    ):
        return out_path, True

    resample = getattr(getattr(Image, "Resampling", Image), "LANCZOS")
    with Image.open(source) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        thumb = ImageOps.fit(image, (DEFAULT_THUMB_SIZE, DEFAULT_THUMB_SIZE), method=resample, centering=(0.5, 0.5))
        thumb.save(out_path, format="JPEG", quality=82, optimize=True, progressive=True)
    return out_path, False


def make_even(value: int, minimum: int = 2) -> int:
    value = max(minimum, int(round(value)))
    return value if value % 2 == 0 else value - 1


def clamp_int(value: str | None, default: int, name: str, lo: int, hi: int) -> int:
    if value in {None, ""}:
        return default
    parsed = int(value)
    if parsed < lo or parsed > hi:
        raise ValueError(f"{name} must be in [{lo}, {hi}], got {parsed}")
    return parsed


def clamp_float(value: str | None, default: float, name: str, lo: float, hi: float) -> float:
    if value in {None, ""}:
        return default
    parsed = float(value)
    if parsed < lo or parsed > hi:
        raise ValueError(f"{name} must be in [{lo}, {hi}], got {parsed}")
    return parsed


def bool_param(value: str | None, default: bool = False) -> bool:
    if value in {None, ""}:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def first(params: dict[str, list[str]], *names: str) -> str | None:
    for name in names:
        values = params.get(name)
        if values:
            return values[-1]
    return None


def default_orbit_angles(spec: AssetSpec) -> tuple[float, float]:
    if spec.source_height > spec.source_width:
        return DEFAULT_PORTRAIT_AZ_DEG, DEFAULT_PORTRAIT_POL_DEG
    return DEFAULT_AZ_DEG, DEFAULT_POL_DEG


def source_k_from_dims(width: int, height: int) -> np.ndarray:
    f35 = 30.0
    focal_px = f35 * math.hypot(width, height) / math.hypot(36.0, 24.0)
    return np.array(
        [[focal_px, 0.0, width * 0.5], [0.0, focal_px, height * 0.5], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )


PLY_SCALAR_FORMATS: dict[str, tuple[str, int]] = {
    "char": ("b", 1),
    "int8": ("b", 1),
    "uchar": ("B", 1),
    "uint8": ("B", 1),
    "short": ("h", 2),
    "int16": ("h", 2),
    "ushort": ("H", 2),
    "uint16": ("H", 2),
    "int": ("i", 4),
    "int32": ("i", 4),
    "uint": ("I", 4),
    "uint32": ("I", 4),
    "float": ("f", 4),
    "float32": ("f", 4),
    "double": ("d", 8),
    "float64": ("d", 8),
}


def read_ply_header(path: Path) -> tuple[str, int, list[dict[str, Any]]]:
    elements: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    fmt = ""
    with path.open("rb") as f:
        first = f.readline()
        if first.strip() != b"ply":
            raise ValueError(f"not a PLY file: {path}")
        while True:
            line_raw = f.readline()
            if not line_raw:
                raise ValueError(f"PLY header is missing end_header: {path}")
            line = line_raw.decode("ascii", errors="replace").strip()
            if line == "end_header":
                return fmt, f.tell(), elements
            if not line or line.startswith("comment"):
                continue
            parts = line.split()
            if parts[:1] == ["format"]:
                fmt = parts[1]
            elif parts[:1] == ["element"] and len(parts) == 3:
                current = {"name": parts[1], "count": int(parts[2]), "properties": []}
                elements.append(current)
            elif parts[:1] == ["property"] and current is not None:
                if len(parts) >= 5 and parts[1] == "list":
                    raise ValueError(f"PLY list properties are not supported by the fast camera reader: {path}")
                if len(parts) != 3:
                    raise ValueError(f"unsupported PLY property line {line!r} in {path}")
                if parts[1] not in PLY_SCALAR_FORMATS:
                    raise ValueError(f"unsupported PLY scalar type {parts[1]!r} in {path}")
                current["properties"].append((parts[2], parts[1]))


def fast_read_ply_camera(path: Path) -> tuple[int, int, np.ndarray]:
    fmt, offset, elements = read_ply_header(path)
    if fmt == "binary_little_endian":
        endian = "<"
    elif fmt == "binary_big_endian":
        endian = ">"
    else:
        raise ValueError(f"fast camera reader only supports binary PLY, got format={fmt!r}")

    intrinsic: np.ndarray | None = None
    image_size: np.ndarray | None = None
    with path.open("rb") as f:
        for element in elements:
            properties: list[tuple[str, str]] = element["properties"]
            stride = sum(PLY_SCALAR_FORMATS[type_name][1] for _, type_name in properties)
            count = int(element["count"])
            if element["name"] == "intrinsic":
                prop_names = [name for name, _ in properties]
                if prop_names == ["intrinsic"]:
                    _, type_name = properties[0]
                    code, size = PLY_SCALAR_FORMATS[type_name]
                    f.seek(offset)
                    raw = f.read(count * size)
                    intrinsic = np.asarray(struct.unpack(f"{endian}{count}{code}", raw), dtype=np.float32)
            elif element["name"] == "image_size":
                prop_names = [name for name, _ in properties]
                if prop_names == ["image_size"]:
                    _, type_name = properties[0]
                    code, size = PLY_SCALAR_FORMATS[type_name]
                    f.seek(offset)
                    raw = f.read(count * size)
                    image_size = np.asarray(struct.unpack(f"{endian}{count}{code}", raw), dtype=np.float32)
            offset += count * stride

    if image_size is None or len(image_size) < 2:
        source_width, source_height = 640, 480
    else:
        source_width, source_height = int(image_size[0]), int(image_size[1])

    if intrinsic is None:
        k_source = source_k_from_dims(source_width, source_height)
    elif len(intrinsic) == 9:
        k_source = intrinsic.reshape(3, 3)
    elif len(intrinsic) == 4:
        fx, fy, source_width_f, source_height_f = intrinsic
        source_width, source_height = int(source_width_f), int(source_height_f)
        k_source = np.array(
            [[fx, 0.0, source_width * 0.5], [0.0, fy, source_height * 0.5], [0.0, 0.0, 1.0]],
            dtype=np.float32,
        )
    else:
        raise ValueError(f"unsupported intrinsic element length {len(intrinsic)} in {path}")
    return source_width, source_height, k_source.astype(np.float32)


def safe_resolve_asset_path(value: str) -> Path:
    raw = Path(unquote(value))
    if not raw.is_absolute():
        raw = VIEWER_ROOT / raw
    resolved = raw.resolve()
    allowed_roots = [VIEWER_ROOT.resolve(), Path("/tmp").resolve()]
    for root in allowed_roots:
        try:
            resolved.relative_to(root)
            return resolved
        except ValueError:
            continue
    raise ValueError(f"asset path is outside allowed roots: {resolved}")


def read_ply_camera(path: Path) -> tuple[int, int, np.ndarray]:
    stat = path.stat()
    key = str(path.resolve())
    cached = _ply_camera_cache.get(key)
    if cached and cached[0] == stat.st_mtime_ns and cached[1] == stat.st_size:
        return cached[2], cached[3], cached[4].copy()
    try:
        source_width, source_height, k_source = fast_read_ply_camera(path)
    except Exception:
        ply = PlyData.read(path)
        if "image_size" in ply:
            image_size = np.asarray(ply["image_size"]["image_size"])
            source_width, source_height = int(image_size[0]), int(image_size[1])
        else:
            source_width, source_height = 640, 480
        if "intrinsic" in ply:
            intrinsic = np.asarray(ply["intrinsic"]["intrinsic"], dtype=np.float32)
            if len(intrinsic) == 9:
                k_source = intrinsic.reshape(3, 3)
            elif len(intrinsic) == 4:
                fx, fy, source_width_f, source_height_f = intrinsic
                source_width, source_height = int(source_width_f), int(source_height_f)
                k_source = np.array(
                    [[fx, 0.0, source_width * 0.5], [0.0, fy, source_height * 0.5], [0.0, 0.0, 1.0]],
                    dtype=np.float32,
                )
            else:
                raise ValueError(f"unsupported intrinsic element length {len(intrinsic)} in {path}")
        else:
            k_source = source_k_from_dims(source_width, source_height)
    k_source = k_source.astype(np.float32)
    _ply_camera_cache[key] = (stat.st_mtime_ns, stat.st_size, int(source_width), int(source_height), k_source.copy())
    return int(source_width), int(source_height), k_source.copy()


def entry_source_dims(entry: dict[str, Any]) -> tuple[int | None, int | None]:
    return entry_int(entry, "width", "sourceWidth"), entry_int(entry, "height", "sourceHeight")


def entry_asset_candidates(value: Any, roots: list[Path]) -> list[Path]:
    if value in {None, ""}:
        return []
    path = Path(str(value))
    if path.is_absolute():
        return [path]
    if path.parts and path.parts[0] == "backend":
        return [REPO_ROOT / path]
    candidates = [root / path for root in roots]
    if len(path.parts) > 1:
        candidates.extend(root / path.name for root in roots)
    return candidates


def unique_paths(paths: list[Path]) -> list[Path]:
    seen: set[str] = set()
    out: list[Path] = []
    for path in paths:
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        out.append(path)
    return out


def resolve_asset(params: dict[str, list[str]]) -> AssetSpec:
    photo_id = first(params, "photoId", "photo_id", "id")
    splat_value = first(params, "splat", "path", "asset")
    title: str | None = None
    source_width: int | None = None
    source_height: int | None = None
    k_source: np.ndarray | None = None
    path: Path | None = None

    if splat_value:
        path = safe_resolve_asset_path(splat_value)
        if not path.exists():
            raise FileNotFoundError(f"asset does not exist: {path}")
        if photo_id:
            entry = gallery_entry(photo_id)
            if entry:
                source_width, source_height = entry_source_dims(entry)
                title = entry.get("title")
    elif photo_id:
        entry = gallery_entry(photo_id)
        candidates = []
        if entry:
            source_width, source_height = entry_source_dims(entry)
            title = entry.get("title")
            roots = [PHOTOS_USER_ROOT, USER_BAKE_ROOT, PHOTOS_BAKE_ROOT, PHOTOS_ROOT, PLYS_ROOT]
            candidates.extend(entry_asset_candidates(entry.get("splat"), roots))
            candidates.extend(entry_asset_candidates(entry.get("ply"), roots))
            candidates.extend(candidate.with_suffix(".ply") for candidate in list(candidates) if candidate.suffix.lower() == ".splat")
        candidates.extend(
            [
                USER_BAKE_ROOT / f"{photo_id}.splat",
                USER_BAKE_ROOT / f"{photo_id}.ply",
                PHOTOS_BAKE_ROOT / f"{photo_id}.splat",
                PHOTOS_BAKE_ROOT / f"{photo_id}.ply",
                PHOTOS_ROOT / f"{photo_id}.splat",
                PHOTOS_ROOT / f"{photo_id}.ply",
                PLYS_ROOT / f"sharp-{photo_id}.ply",
                PLYS_ROOT / f"{photo_id}.ply",
                PLYS_ROOT / f"sharp-{photo_id}.splat",
                PLYS_ROOT / f"{photo_id}.splat",
            ]
        )
        candidates = unique_paths(candidates)
        path = next((candidate for candidate in candidates if candidate.exists()), None)
        if path is None:
            raise FileNotFoundError(f"no splat/ply asset found for photoId={photo_id!r}")
    else:
        raise ValueError("provide photoId or splat/path")

    suffix = path.suffix.lower()
    if suffix not in {".splat", ".ply"}:
        raise ValueError(f"asset must be .splat or .ply, got {path}")
    if suffix == ".ply":
        ply_width, ply_height, ply_k = read_ply_camera(path)
        source_width = ply_width
        source_height = ply_height
        k_source = ply_k
    else:
        source_width = source_width or clamp_int(first(params, "sourceWidth", "widthSource"), 0, "sourceWidth", 1, 20000)
        source_height = source_height or clamp_int(first(params, "sourceHeight", "heightSource"), 0, "sourceHeight", 1, 20000)
        if source_width <= 0 or source_height <= 0:
            raise ValueError("sourceWidth and sourceHeight are required for direct .splat paths not in gallery")
        k_source = source_k_from_dims(source_width, source_height)

    return AssetSpec(
        path=path,
        kind=suffix[1:],
        source_width=int(source_width),
        source_height=int(source_height),
        k_source=k_source.astype(np.float32),
        photo_id=photo_id,
        title=title,
    )


def density_param(params: dict[str, list[str]]) -> str:
    value = (first(params, "density") or "350k").strip().lower()
    aliases = {
        "": "350k",
        "default": "350k",
        "app": "350k",
        "mobile": "350k",
        "downsample": "350k",
        "downsampled": "350k",
        "350": "350k",
        "350k": "350k",
        "full": "full",
        "1.18m": "full",
        "1180k": "full",
    }
    if value not in aliases:
        raise ValueError(f"density must be 350k/default or full, got {value!r}")
    return aliases[value]


def density_for_path(path: Path) -> str:
    return "350k" if "350k" in path.stem.lower() else "full"


def downsampled_ply_path(path: Path) -> Path:
    return path.with_name(f"{path.stem}-350k{path.suffix}")


def resolve_ply_asset(params: dict[str, list[str]]) -> tuple[Path, str | None, str, str, bool]:
    photo_id = first(params, "photoId", "photo_id", "id")
    path_value = first(params, "ply", "path", "asset", "splat")
    requested_density = density_param(params)
    preferred: list[Path] = []
    fallback: list[Path] = []

    if path_value:
        path = safe_resolve_asset_path(path_value)
        full_path = path if path.suffix.lower() == ".ply" else path.with_suffix(".ply")
        if requested_density == "full":
            preferred.append(full_path)
        elif density_for_path(full_path) == "350k":
            preferred.append(full_path)
        else:
            preferred.append(downsampled_ply_path(full_path))
            fallback.append(full_path)

    if photo_id:
        entry = gallery_entry(photo_id)
        full_candidates: list[Path] = []
        downsample_candidates: list[Path] = []
        if entry:
            roots = [PHOTOS_USER_ROOT, USER_BAKE_ROOT, PHOTOS_BAKE_ROOT, PHOTOS_ROOT, PLYS_ROOT]
            full_candidates.extend(entry_asset_candidates(entry.get("ply"), roots))
            for splat_path in entry_asset_candidates(entry.get("splat") or f"{photo_id}.splat", roots):
                full_candidates.append(splat_path.with_suffix(".ply"))
        full_candidates.extend(
            [
                USER_BAKE_ROOT / f"{photo_id}.ply",
                PHOTOS_BAKE_ROOT / f"{photo_id}.ply",
                PHOTOS_ROOT / f"{photo_id}.ply",
                PLYS_ROOT / f"sharp-{photo_id}.ply",
                PLYS_ROOT / f"{photo_id}.ply",
            ]
        )
        full_candidates = unique_paths(full_candidates)
        for candidate in full_candidates:
            downsample_candidates.append(downsampled_ply_path(candidate))
        downsample_candidates.extend(
            [
                USER_BAKE_ROOT / f"{photo_id}-350k.ply",
                PHOTOS_BAKE_ROOT / f"{photo_id}-350k.ply",
                PHOTOS_ROOT / f"{photo_id}-350k.ply",
                PLYS_ROOT / f"sharp-{photo_id}-350k.ply",
                PLYS_ROOT / f"{photo_id}-350k.ply",
            ]
        )
        if requested_density == "full":
            preferred.extend(full_candidates)
        else:
            preferred.extend(downsample_candidates)
            fallback.extend(full_candidates)

    for candidate in preferred:
        if candidate.exists() and candidate.suffix.lower() == ".ply":
            return candidate, photo_id, requested_density, density_for_path(candidate), False
    for candidate in fallback:
        if candidate.exists() and candidate.suffix.lower() == ".ply":
            return candidate, photo_id, requested_density, density_for_path(candidate), True
    if photo_id:
        raise FileNotFoundError(f"no .ply asset found for photoId={photo_id!r}")
    raise ValueError("provide photoId or ply/path")


def parse_render_params(params: dict[str, list[str]], spec: AssetSpec) -> RenderParams:
    output_format = (first(params, "format", "fmt") or "mp4").strip().lower()
    if output_format not in {"mp4", "webp"}:
        raise ValueError(f"format must be mp4 or webp, got {output_format!r}")
    default_width = DEFAULT_WEBP_WIDTH if output_format == "webp" else DEFAULT_WIDTH
    default_height = DEFAULT_WEBP_HEIGHT if output_format == "webp" else DEFAULT_HEIGHT
    default_fps = DEFAULT_WEBP_FPS if output_format == "webp" else DEFAULT_FPS
    width = clamp_int(first(params, "width", "w"), default_width, "width", 96, 1280)
    explicit_height = first(params, "height", "h")
    max_height = clamp_int(first(params, "maxHeight", "max_height"), DEFAULT_MAX_HEIGHT, "maxHeight", 96, 1280)
    if explicit_height:
        height = clamp_int(explicit_height, DEFAULT_MAX_HEIGHT, "height", 96, 1280)
    else:
        height = default_height
        if height > max_height:
            scale = max_height / height
            height = max_height
            width = int(round(width * scale))
    width = make_even(width)
    height = make_even(height)
    fps = clamp_int(first(params, "fps"), default_fps, "fps", 8, 60)
    seconds = clamp_float(first(params, "seconds", "duration"), DEFAULT_SECONDS, "seconds", 0.5, 10.0)
    frames = clamp_int(first(params, "frames"), int(round(fps * seconds)), "frames", 8, 240)
    seconds = frames / fps
    default_az, default_pol = default_orbit_angles(spec)
    az_deg = clamp_float(first(params, "azDeg", "az", "azAmpDeg"), default_az, "azDeg", 0.0, 45.0)
    pol_deg = clamp_float(first(params, "polDeg", "pol", "polAmpDeg"), default_pol, "polDeg", 0.0, 20.0)
    bitrate_default = DEFAULT_WEBP_BITRATE_KBPS if output_format == "webp" else DEFAULT_BITRATE_KBPS
    bitrate = clamp_int(first(params, "bitrateKbps", "bitrate"), bitrate_default, "bitrateKbps", 50, 4000)
    bg_hex = (first(params, "bg", "background") or DEFAULT_BG_HEX).strip().lstrip("#")
    ImageColor.getrgb(f"#{bg_hex}")
    return RenderParams(
        width=width,
        height=height,
        fps=fps,
        seconds=seconds,
        frames=frames,
        az_deg=az_deg,
        pol_deg=pol_deg,
        bitrate_kbps=bitrate,
        bg_hex=bg_hex,
        output_format=output_format,
    )


def srgb_to_linear_np(rgb: np.ndarray) -> np.ndarray:
    return np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4).astype(np.float32)


def linear_to_srgb_torch(rgb: torch.Tensor) -> torch.Tensor:
    return torch.where(rgb <= 0.0031308, rgb * 12.92, 1.055 * torch.pow(rgb.clamp_min(0.0), 1 / 2.4) - 0.055)


def load_splat_arrays(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if path.stat().st_size % SPLAT_ROW_BYTES != 0:
        raise ValueError(f"invalid .splat size {path.stat().st_size}; not divisible by {SPLAT_ROW_BYTES}")
    row_dtype = np.dtype(
        [
            ("xyz", "<f4", (3,)),
            ("scale", "<f4", (3,)),
            ("rgba", "u1", (4,)),
            ("rotation", "u1", (4,)),
        ]
    )
    rows = np.fromfile(path, dtype=row_dtype)
    means = rows["xyz"].astype(np.float32, copy=False)
    scales = rows["scale"].astype(np.float32, copy=False)
    rgba = rows["rgba"].astype(np.float32) / 255.0
    rotations = ((rows["rotation"].astype(np.float32) - 128.0) / 128.0).astype(np.float32)
    rotations /= np.linalg.norm(rotations, axis=1, keepdims=True).clip(1e-8)
    colors = srgb_to_linear_np(rgba[:, :3])
    opacities = rgba[:, 3].astype(np.float32)
    return means, rotations, scales, opacities, colors


def ply_stack(vertex: Any, names: list[str]) -> np.ndarray:
    return np.stack([np.asarray(vertex[name], dtype=np.float32) for name in names], axis=1)


def load_ply_arrays(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    ply = PlyData.read(path)
    vertex = ply["vertex"]
    means = ply_stack(vertex, ["x", "y", "z"])
    sh0 = ply_stack(vertex, ["f_dc_0", "f_dc_1", "f_dc_2"])
    scales = np.exp(ply_stack(vertex, ["scale_0", "scale_1", "scale_2"])).astype(np.float32)
    quats = ply_stack(vertex, ["rot_0", "rot_1", "rot_2", "rot_3"])
    quats /= np.linalg.norm(quats, axis=1, keepdims=True).clip(1e-8)
    opacity_logits = np.asarray(vertex["opacity"], dtype=np.float32)
    opacities = (1.0 / (1.0 + np.exp(-opacity_logits))).astype(np.float32)
    coeff_degree0 = math.sqrt(1.0 / (4.0 * math.pi))
    colors = np.clip(sh0 * coeff_degree0 + 0.5, 0.0, 1.0).astype(np.float32)
    color_space = 0
    if "color_space" in ply:
        color_space_data = np.asarray(ply["color_space"]["color_space"])
        if len(color_space_data):
            color_space = int(color_space_data[0])
    if color_space == 0:
        colors = srgb_to_linear_np(colors)
    return means, quats.astype(np.float32), scales, opacities, colors


def load_asset(spec: AssetSpec) -> LoadedAsset:
    key = str(spec.path.resolve())
    cached = _asset_cache.get(key)
    now = time.time()
    if cached:
        cached.last_used = now
        return cached

    t0 = time.perf_counter()
    if spec.kind == "splat":
        means_np, quats_np, scales_np, opacities_np, colors_np = load_splat_arrays(spec.path)
    elif spec.kind == "ply":
        means_np, quats_np, scales_np, opacities_np, colors_np = load_ply_arrays(spec.path)
    else:
        raise ValueError(f"unsupported asset kind {spec.kind}")

    asset = LoadedAsset(
        spec=spec,
        means=torch.from_numpy(means_np).to("cuda", non_blocking=True),
        quats=torch.from_numpy(quats_np).to("cuda", non_blocking=True),
        scales=torch.from_numpy(scales_np).to("cuda", non_blocking=True),
        opacities=torch.from_numpy(opacities_np).to("cuda", non_blocking=True),
        colors_linear=torch.from_numpy(colors_np).to("cuda", non_blocking=True),
        count=int(means_np.shape[0]),
        loaded_seconds=0.0,
        last_used=now,
    )
    torch.cuda.synchronize()
    asset.loaded_seconds = time.perf_counter() - t0
    _asset_cache[key] = asset
    evict_old_assets()
    log_event(
        "orbit_asset_loaded",
        path=str(spec.path),
        kind=spec.kind,
        photoId=spec.photo_id,
        count=asset.count,
        sourceWidth=spec.source_width,
        sourceHeight=spec.source_height,
        loadedSeconds=asset.loaded_seconds,
    )
    return asset


def evict_old_assets() -> None:
    while len(_asset_cache) > MAX_GPU_ASSETS:
        oldest_key, oldest = min(_asset_cache.items(), key=lambda item: item[1].last_used)
        log_event("orbit_asset_evict", path=oldest_key, count=oldest.count)
        del _asset_cache[oldest_key]
        torch.cuda.empty_cache()


def scaled_k(spec: AssetSpec, params: RenderParams) -> torch.Tensor:
    k = spec.k_source.copy()
    scale = max(params.width / spec.source_width, params.height / spec.source_height)
    scaled_width = spec.source_width * scale
    scaled_height = spec.source_height * scale
    crop_x = (scaled_width - params.width) * 0.5
    crop_y = (scaled_height - params.height) * 0.5
    k[0, :] *= scale
    k[1, :] *= scale
    k[0, 2] -= crop_x
    k[1, 2] -= crop_y
    return torch.from_numpy(k.astype(np.float32)).to("cuda")[None]


def normalize_vec(vec: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    return vec / max(norm, 1e-8)


def viewmat_for_offset(az_deg: float, pol_deg: float) -> np.ndarray:
    target = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    radius = 1.0
    az = math.radians(az_deg)
    pol = math.radians(pol_deg)
    camera = target + np.array(
        [
            radius * math.sin(az) * math.cos(pol),
            radius * math.sin(pol),
            -radius * math.cos(az) * math.cos(pol),
        ],
        dtype=np.float32,
    )
    z_axis = normalize_vec(target - camera)
    down_ref = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    x_axis = normalize_vec(np.cross(down_ref, z_axis))
    y_axis = np.cross(z_axis, x_axis)
    c2w = np.eye(4, dtype=np.float32)
    c2w[:3, :3] = np.stack([x_axis, y_axis, z_axis], axis=1)
    c2w[:3, 3] = camera
    return np.linalg.inv(c2w).astype(np.float32)


def bg_linear_tensor(bg_hex: str) -> torch.Tensor:
    rgb = np.array(ImageColor.getrgb(f"#{bg_hex}"), dtype=np.float32) / 255.0
    return torch.from_numpy(srgb_to_linear_np(rgb)).to("cuda")


def render_frame(asset: LoadedAsset, ks: torch.Tensor, viewmat: np.ndarray, params: RenderParams, bg: torch.Tensor) -> np.ndarray:
    view = torch.from_numpy(viewmat).to("cuda")[None]
    colors, alphas, _ = rasterization(
        means=asset.means,
        quats=asset.quats,
        scales=asset.scales,
        opacities=asset.opacities,
        colors=asset.colors_linear,
        viewmats=view,
        Ks=ks,
        width=params.width,
        height=params.height,
        render_mode="RGB",
        rasterize_mode="classic",
        packed=False,
        absgrad=False,
    )
    rgb = colors[0, ..., :3].clamp(0.0, 1.0)
    alpha = alphas[0, ..., :1].clamp(0.0, 1.0)
    rgb = rgb + (1.0 - alpha) * bg[None, None, :]
    rgb = linear_to_srgb_torch(rgb).clamp(0.0, 1.0)
    return (rgb.detach().cpu().numpy() * 255.0).round().astype(np.uint8)


def cache_key(spec: AssetSpec, params: RenderParams) -> str:
    stat = spec.path.stat()
    payload = {
        "version": VERSION,
        "path": str(spec.path.resolve()),
        "mtimeNs": stat.st_mtime_ns,
        "size": stat.st_size,
        "photoId": spec.photo_id,
        "sourceWidth": spec.source_width,
        "sourceHeight": spec.source_height,
        "k": np.round(spec.k_source, 6).tolist(),
        "width": params.width,
        "height": params.height,
        "fps": params.fps,
        "frames": params.frames,
        "azDeg": params.az_deg,
        "polDeg": params.pol_deg,
        "bitrateKbps": params.bitrate_kbps,
        "bg": params.bg_hex,
        "format": params.output_format,
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:24]


def cache_paths(key: str, output_format: str) -> tuple[Path, Path]:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR / f"{key}.{output_format}", CACHE_DIR / f"{key}.json"


def write_meta(path: Path, meta: dict[str, Any]) -> None:
    path.write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")


def read_meta(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def stable_sigmoid(values: np.ndarray) -> np.ndarray:
    positive = values >= 0
    out = np.empty_like(values, dtype=np.float32)
    out[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exp_values = np.exp(values[~positive])
    out[~positive] = exp_values / (1.0 + exp_values)
    return out


def encode_u8(values: np.ndarray) -> np.ndarray:
    return np.clip(np.rint(values), 0.0, 255.0).astype(np.uint8)


def splat_row_dtype() -> np.dtype:
    return np.dtype(
        [
            ("xyz", "<f4", (3,)),
            ("scale", "<f4", (3,)),
            ("rgba", "u1", (4,)),
            ("rotation", "u1", (4,)),
        ]
    )


def validate_splat_bytes(data: bytes) -> int:
    if len(data) <= 0 or len(data) % SPLAT_ROW_BYTES != 0:
        raise ValueError(f"invalid .splat byte length {len(data)}; not divisible by {SPLAT_ROW_BYTES}")
    return len(data) // SPLAT_ROW_BYTES


def convert_ply_to_shuffled_splat(ply_path: Path, out_path: Path, meta_path: Path, meta_base: dict[str, Any]) -> dict[str, Any]:
    tmp_path = out_path.with_suffix(".tmp.splat")
    if tmp_path.exists():
        tmp_path.unlink()
    t0 = time.perf_counter()
    ply = PlyData.read(ply_path)
    vertex = ply["vertex"]
    count = len(vertex.data)
    permutation = np.random.default_rng(SPLAT_STREAM_SHUFFLE_SEED).permutation(count)

    xyz = ply_stack(vertex, ["x", "y", "z"])[permutation]
    scales = np.exp(ply_stack(vertex, ["scale_0", "scale_1", "scale_2"]))[permutation]
    sh0 = ply_stack(vertex, ["f_dc_0", "f_dc_1", "f_dc_2"])
    rgb = encode_u8((0.5 + SH_C0 * sh0) * 255.0)[permutation]
    opacity = stable_sigmoid(np.asarray(vertex["opacity"], dtype=np.float32))
    alpha = encode_u8(opacity * 255.0)[permutation, None]
    quats = ply_stack(vertex, ["rot_0", "rot_1", "rot_2", "rot_3"])
    quats = quats / np.linalg.norm(quats, axis=1, keepdims=True).clip(1e-8)
    rotation = encode_u8(quats * 128.0 + 128.0)[permutation]

    row_dtype = splat_row_dtype()
    rows = np.empty(count, dtype=row_dtype)
    rows["xyz"] = xyz.astype("<f4", copy=False)
    rows["scale"] = scales.astype("<f4", copy=False)
    rows["rgba"] = np.concatenate([rgb, alpha], axis=1)
    rows["rotation"] = rotation
    rows.tofile(tmp_path)
    tmp_path.replace(out_path)

    meta = {
        **meta_base,
        "ok": True,
        "version": VERSION,
        "format": "splat",
        "rowBytes": SPLAT_ROW_BYTES,
        "numGaussians": count,
        "fileBytes": out_path.stat().st_size,
        "expectedBytes": count * SPLAT_ROW_BYTES,
        "shuffleSeed": SPLAT_STREAM_SHUFFLE_SEED,
        "shuffle": "uniform",
        "sourcePath": str(ply_path),
        "generatedSeconds": time.perf_counter() - t0,
        "createdAt": time.time(),
    }
    if meta["fileBytes"] != meta["expectedBytes"]:
        raise RuntimeError(f"invalid splat size {meta['fileBytes']} != {meta['expectedBytes']}")
    write_meta(meta_path, meta)
    log_event("splat_stream_generated", **meta)
    return meta


def convert_ply_to_static_splat(ply_path: Path, out_path: Path, *, force: bool = False) -> dict[str, Any]:
    if out_path.exists() and not force and out_path.stat().st_mtime >= ply_path.stat().st_mtime:
        count = out_path.stat().st_size // SPLAT_ROW_BYTES
        return {
            "ok": True,
            "splat_path": str(out_path),
            "num_gaussians": count,
            "fileBytes": out_path.stat().st_size,
            "cache": "hit",
        }

    tmp_path = out_path.with_suffix(".tmp.splat")
    if tmp_path.exists():
        tmp_path.unlink()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    ply = PlyData.read(ply_path)
    vertex = ply["vertex"]
    count = len(vertex.data)

    xyz = ply_stack(vertex, ["x", "y", "z"])
    scales = np.exp(ply_stack(vertex, ["scale_0", "scale_1", "scale_2"]))
    sh0 = ply_stack(vertex, ["f_dc_0", "f_dc_1", "f_dc_2"])
    rgb = encode_u8((0.5 + SH_C0 * sh0) * 255.0)
    opacity = stable_sigmoid(np.asarray(vertex["opacity"], dtype=np.float32))
    alpha = encode_u8(opacity * 255.0)[:, None]
    quats = ply_stack(vertex, ["rot_0", "rot_1", "rot_2", "rot_3"])
    quats = quats / np.linalg.norm(quats, axis=1, keepdims=True).clip(1e-8)
    rotation = encode_u8(quats * 128.0 + 128.0)

    row_dtype = splat_row_dtype()
    rows = np.empty(count, dtype=row_dtype)
    rows["xyz"] = xyz.astype("<f4", copy=False)
    rows["scale"] = scales.astype("<f4", copy=False)
    rows["rgba"] = np.concatenate([rgb, alpha], axis=1)
    rows["rotation"] = rotation
    rows.tofile(tmp_path)
    tmp_path.replace(out_path)
    file_bytes = out_path.stat().st_size
    if file_bytes != count * SPLAT_ROW_BYTES:
        raise RuntimeError(f"invalid static splat size {file_bytes} != {count * SPLAT_ROW_BYTES}")
    meta = {
        "ok": True,
        "splat_path": str(out_path),
        "num_gaussians": count,
        "fileBytes": file_bytes,
        "generatedSeconds": time.perf_counter() - t0,
        "cache": "miss",
    }
    log_event("static_splat_generated", plyPath=str(ply_path), **meta)
    return meta


def splat_stream_cache_key(
    ply_path: Path,
    *,
    photo_id: str | None,
    requested_density: str,
    served_density: str,
    fallback: bool,
) -> str:
    stat = ply_path.stat()
    payload = {
        "version": VERSION,
        "kind": "shuffled-splat-stream",
        "photoId": photo_id,
        "requestedDensity": requested_density,
        "servedDensity": served_density,
        "fallback": fallback,
        "path": str(ply_path.resolve()),
        "mtimeNs": stat.st_mtime_ns,
        "size": stat.st_size,
        "shuffleSeed": SPLAT_STREAM_SHUFFLE_SEED,
        "rowBytes": SPLAT_ROW_BYTES,
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:24]


def splat_stream_cache_paths(key: str) -> tuple[Path, Path]:
    SPLAT_STREAM_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return SPLAT_STREAM_CACHE_DIR / f"{key}.splat", SPLAT_STREAM_CACHE_DIR / f"{key}.json"


def get_or_create_splat_stream(
    ply_path: Path,
    *,
    photo_id: str | None,
    requested_density: str,
    served_density: str,
    fallback: bool,
    force: bool,
) -> tuple[Path, dict[str, Any], bool]:
    source_width, source_height, k_source = read_ply_camera(ply_path)
    camera_meta = camera_dict(source_width, source_height, k_source)
    key = splat_stream_cache_key(
        ply_path,
        photo_id=photo_id,
        requested_density=requested_density,
        served_density=served_density,
        fallback=fallback,
    )
    out_path, meta_path = splat_stream_cache_paths(key)
    if out_path.exists() and meta_path.exists() and not force:
        meta = read_meta(meta_path)
        if meta.get("ok") and out_path.stat().st_size == int(meta.get("expectedBytes", -1)):
            if any(meta.get(key) is None for key in camera_meta):
                meta.update(camera_meta)
                write_meta(meta_path, meta)
            return out_path, meta, True

    with _lock:
        if out_path.exists() and meta_path.exists() and not force:
            meta = read_meta(meta_path)
            if meta.get("ok") and out_path.stat().st_size == int(meta.get("expectedBytes", -1)):
                if any(meta.get(key) is None for key in camera_meta):
                    meta.update(camera_meta)
                    write_meta(meta_path, meta)
                return out_path, meta, True
        meta = convert_ply_to_shuffled_splat(
            ply_path,
            out_path,
            meta_path,
            {
                "photoId": photo_id,
                "requestedDensity": requested_density,
                "servedDensity": served_density,
                "fallback": fallback,
                **camera_meta,
            },
        )
        return out_path, meta, False


def ffmpeg_version() -> str | None:
    try:
        raw = subprocess.check_output(["ffmpeg", "-version"], text=True, stderr=subprocess.STDOUT, timeout=3)
    except Exception:
        return None
    return raw.splitlines()[0] if raw else None


def ffmpeg_orbit_cmd(params: RenderParams, tmp_path: Path) -> list[str]:
    base = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "warning",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{params.width}x{params.height}",
        "-r",
        str(params.fps),
        "-i",
        "pipe:0",
        "-an",
    ]
    if params.output_format == "webp":
        return [
            *base,
            "-c:v",
            "libwebp",
            "-lossless",
            "0",
            "-compression_level",
            "6",
            "-q:v",
            str(DEFAULT_WEBP_QUALITY),
            "-b:v",
            f"{params.bitrate_kbps}k",
            "-loop",
            "0",
            "-f",
            "webp",
            str(tmp_path),
        ]
    return [
        *base,
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-profile:v",
        "baseline",
        "-level",
        "3.1",
        "-pix_fmt",
        "yuv420p",
        "-b:v",
        f"{params.bitrate_kbps}k",
        "-maxrate",
        f"{max(params.bitrate_kbps, 150)}k",
        "-bufsize",
        f"{max(params.bitrate_kbps * 2, 300)}k",
        "-g",
        str(params.fps),
        "-movflags",
        "+faststart",
        str(tmp_path),
    ]


def render_orbit_frames(asset: LoadedAsset, params: RenderParams) -> OrbitFrameBuffer:
    render_start = time.perf_counter()
    frames: list[np.ndarray] = []
    frame_times: list[float] = []
    ks = scaled_k(asset.spec, params)
    bg = bg_linear_tensor(params.bg_hex)
    with torch.inference_mode():
        for i in range(params.frames):
            phase = 2.0 * math.pi * i / params.frames
            az = -params.az_deg * math.cos(phase)
            pol = params.pol_deg * math.sin(phase)
            view = viewmat_for_offset(az, pol)
            t0 = time.perf_counter()
            frame = render_frame(asset, ks, view, params, bg)
            frame_times.append(time.perf_counter() - t0)
            frames.append(np.ascontiguousarray(frame))
    return OrbitFrameBuffer(
        frames=frames,
        frame_times=frame_times,
        render_seconds=time.perf_counter() - render_start,
    )


def encode_orbit_frames(
    asset: LoadedAsset,
    params: RenderParams,
    frame_buffer: OrbitFrameBuffer,
    out_path: Path,
    meta_path: Path,
    *,
    generate_start: float,
    gpu_lock_seconds: float | None = None,
) -> dict[str, Any]:
    tmp_path = out_path.with_name(
        f"{out_path.stem}.{os.getpid()}.{threading.get_ident()}.{time.perf_counter_ns()}.tmp{out_path.suffix}"
    )
    if tmp_path.exists():
        tmp_path.unlink()
    ffmpeg_cmd = ffmpeg_orbit_cmd(params, tmp_path)

    encode_start = time.perf_counter()
    proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.stdin is None or proc.stderr is None:
        raise RuntimeError("failed to open ffmpeg pipes")
    try:
        for frame in frame_buffer.frames:
            proc.stdin.write(frame.tobytes())
        proc.stdin.close()
        stderr = proc.stderr.read().decode("utf-8", errors="replace")
        return_code = proc.wait()
    except Exception:
        proc.kill()
        proc.wait()
        raise
    if return_code != 0:
        raise RuntimeError(f"ffmpeg failed with code {return_code}: {stderr[-4000:]}")
    tmp_path.replace(out_path)

    encode_seconds = time.perf_counter() - encode_start
    generate_seconds = time.perf_counter() - generate_start
    frame_times = frame_buffer.frame_times
    meta = {
        "ok": True,
        "version": VERSION,
        "format": params.output_format,
        "photoId": asset.spec.photo_id,
        "title": asset.spec.title,
        "assetPath": str(asset.spec.path),
        "assetKind": asset.spec.kind,
        "numGaussians": asset.count,
        "sourceWidth": asset.spec.source_width,
        "sourceHeight": asset.spec.source_height,
        "width": params.width,
        "height": params.height,
        "fps": params.fps,
        "frames": params.frames,
        "durationSeconds": params.frames / params.fps,
        "azDeg": params.az_deg,
        "polDeg": params.pol_deg,
        "bitrateKbps": params.bitrate_kbps,
        "webpQuality": DEFAULT_WEBP_QUALITY if params.output_format == "webp" else None,
        "background": params.bg_hex,
        "fileBytes": out_path.stat().st_size,
        "generateSeconds": generate_seconds,
        "gpuRenderSeconds": frame_buffer.render_seconds,
        "ffmpegEncodeSeconds": encode_seconds,
        "gpuLockSeconds": gpu_lock_seconds,
        "assetLoadedSeconds": asset.loaded_seconds,
        "frameMsAvg": round(float(np.mean(frame_times) * 1000.0), 2) if frame_times else None,
        "frameMsP50": round(float(np.percentile(frame_times, 50) * 1000.0), 2) if frame_times else None,
        "frameMsP95": round(float(np.percentile(frame_times, 95) * 1000.0), 2) if frame_times else None,
        "ffmpeg": ffmpeg_version(),
        "createdAt": time.time(),
    }
    write_meta(meta_path, meta)
    log_event("orbit_preview_generated", **meta)
    return meta


def encode_orbit(asset: LoadedAsset, params: RenderParams, out_path: Path, meta_path: Path) -> dict[str, Any]:
    generate_start = time.perf_counter()
    frame_buffer = render_orbit_frames(asset, params)
    return encode_orbit_frames(asset, params, frame_buffer, out_path, meta_path, generate_start=generate_start)


def get_or_create_preview(spec: AssetSpec, params: RenderParams, force: bool) -> tuple[Path, dict[str, Any], bool]:
    key = cache_key(spec, params)
    out_path, meta_path = cache_paths(key, params.output_format)
    if out_path.exists() and meta_path.exists() and not force:
        meta = read_meta(meta_path)
        if meta.get("ok") and meta.get("format", "mp4") == params.output_format:
            return out_path, meta, True

    with _lock:
        if out_path.exists() and meta_path.exists() and not force:
            meta = read_meta(meta_path)
            if meta.get("ok") and meta.get("format", "mp4") == params.output_format:
                return out_path, meta, True
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available")
        lock_start = time.perf_counter()
        asset = load_asset(spec)
        generate_start = time.perf_counter()
        frame_buffer = render_orbit_frames(asset, params)
        gpu_lock_seconds = time.perf_counter() - lock_start
    meta = encode_orbit_frames(
        asset,
        params,
        frame_buffer,
        out_path,
        meta_path,
        generate_start=generate_start,
        gpu_lock_seconds=gpu_lock_seconds,
    )
    return out_path, meta, False


def smi_processes() -> list[dict[str, Any]]:
    try:
        raw = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader,nounits"],
            text=True,
            timeout=3,
        ).strip()
    except Exception:
        return []
    rows = []
    for line in raw.splitlines():
        if not line.strip():
            continue
        parts = [part.strip() for part in line.split(",", 2)]
        if len(parts) == 3:
            rows.append({"pid": parts[0], "process": parts[1], "usedMiB": parts[2]})
    return rows


def disk_free_bytes(path: Path) -> int:
    probe = path
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    return shutil.disk_usage(probe).free


def ensure_ingest_space() -> None:
    free_bytes = disk_free_bytes(VIEWER_ROOT)
    if free_bytes < INGEST_MIN_FREE_BYTES:
        raise OSError(
            f"low disk space for ingest: free={free_bytes} bytes, required={INGEST_MIN_FREE_BYTES} bytes"
        )


def parse_ingest_request(handler: BaseHTTPRequestHandler) -> IngestUpload:
    length = int(handler.headers.get("Content-Length", "0"))
    if length <= 0:
        raise ValueError("request body is empty")
    if length > INGEST_MAX_BODY_BYTES:
        raise ValueError(f"upload too large: {length} bytes > {INGEST_MAX_BODY_BYTES} bytes")

    content_type_full = handler.headers.get("Content-Type", "")
    content_type = content_type_full.split(";", 1)[0].strip().lower()
    if content_type != "multipart/form-data":
        raise ValueError("POST /ingest requires multipart/form-data with an image/file field")

    raw = handler.rfile.read(length)
    message = BytesParser(policy=policy.default).parsebytes(
        b"Content-Type: " + content_type_full.encode("utf-8") + b"\r\nMIME-Version: 1.0\r\n\r\n" + raw
    )
    fields: dict[str, str] = {}
    image_payload: bytes | None = None
    filename = "upload.jpg"
    for part in message.iter_parts():
        name = part.get_param("name", header="content-disposition")
        if not name:
            continue
        payload = part.get_payload(decode=True) or b""
        part_filename = part.get_filename()
        if part_filename:
            if name in {"image", "file", "upload"} and image_payload is None:
                image_payload = payload
                filename = Path(part_filename).name or filename
            continue
        fields[name] = payload.decode(part.get_content_charset() or "utf-8", errors="replace").strip()

    if image_payload is None:
        raise ValueError("missing multipart image field; use name=image or name=file")
    return IngestUpload(image=image_payload, filename=filename, fields=fields)


def user_relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PHOTOS_USER_ROOT.resolve()))
    except ValueError:
        return str(path)


def reserve_user_photo_id(upload: IngestUpload) -> str:
    requested = upload.fields.get("photoId") or upload.fields.get("photo_id") or upload.fields.get("id")
    title = upload.fields.get("title") or Path(upload.filename).stem
    curated_ids = {str(entry.get("id")) for entry in load_gallery()}
    if requested:
        candidate = slugify_photo_id(requested)
        if not candidate.startswith("user-"):
            candidate = f"user-{candidate}"
        if candidate in curated_ids:
            raise ValueError(f"requested photoId conflicts with curated gallery id: {candidate}")
        return candidate
    base = slugify_photo_id(requested or title or "upload")
    if not base.startswith("user-"):
        base = f"user-{base}"
    digest = hashlib.sha1(upload.image).hexdigest()[:10]
    existing = set(curated_ids)
    existing.update(str(entry.get("id")) for entry in load_user_gallery())
    for index in range(1000):
        suffix = digest if index == 0 else f"{digest}-{index}"
        candidate = f"{base}-{suffix}"
        if candidate not in existing:
            return candidate
    raise RuntimeError("could not allocate a unique user photoId")


def save_ingest_source_temp(upload: IngestUpload, photo_id: str, tmp_dir: Path) -> dict[str, Any]:
    source_path = tmp_dir / f"{photo_id}.jpg"
    resample = getattr(getattr(Image, "Resampling", Image), "LANCZOS")
    with Image.open(io.BytesIO(upload.image)) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        original_width, original_height = image.size
        pixels = original_width * original_height
        scale = 1.0
        if max(original_width, original_height) > INGEST_MAX_SOURCE_SIDE:
            scale = min(scale, INGEST_MAX_SOURCE_SIDE / max(original_width, original_height))
        if pixels > INGEST_MAX_PIXELS:
            scale = min(scale, math.sqrt(INGEST_MAX_PIXELS / pixels))
        if scale < 1.0:
            new_size = (
                max(1, int(round(original_width * scale))),
                max(1, int(round(original_height * scale))),
            )
            image = image.resize(new_size, resample)
        source_width, source_height = image.size
        image.save(source_path, format="JPEG", quality=INGEST_JPEG_QUALITY, optimize=True, progressive=True)
    return {
        "sourcePath": source_path,
        "sourceWidth": int(source_width),
        "sourceHeight": int(source_height),
        "originalWidth": int(original_width),
        "originalHeight": int(original_height),
        "sourceBytes": source_path.stat().st_size,
    }


def call_sharp_infer(image_path: Path, out_path: Path, output_format: str) -> dict[str, Any]:
    if output_format not in {"ply", "splat"}:
        raise ValueError(f"unsupported SHARP output format: {output_format}")
    payload = {
        "image_path": str(image_path),
        "output_format": output_format,
        "out_ply" if output_format == "ply" else "out_splat": str(out_path),
    }
    raw = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    request = Request(
        SHARP_INFER_URL,
        data=raw,
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        method="POST",
    )
    t0 = time.perf_counter()
    try:
        with urlopen(request, timeout=INGEST_SHARP_TIMEOUT_SECONDS) as response:
            body = response.read()
            status = int(response.status)
    except HTTPError as exc:
        body = exc.read()
        raise RuntimeError(f"SHARP /infer failed with HTTP {exc.code}: {body[:1000]!r}") from exc
    if status != 200:
        raise RuntimeError(f"SHARP /infer failed with HTTP {status}: {body[:1000]!r}")
    data = json.loads(body.decode("utf-8"))
    if data.get("ok") is False:
        raise RuntimeError(f"SHARP /infer returned error: {data}")
    data["clientTiming"] = {
        "httpSeconds": time.perf_counter() - t0,
        "requestBytes": len(raw),
        "responseBytes": len(body),
        "outputBytes": out_path.stat().st_size if out_path.exists() else None,
    }
    return data


def load_inproc_sharp_module() -> Any:
    global _inproc_sharp_module
    if _inproc_sharp_module is None:
        import importlib

        _inproc_sharp_module = importlib.import_module("sharp_endpoint")
    return _inproc_sharp_module


def ensure_inproc_sharp_loaded() -> None:
    if not ORBIT_INPROC_SHARP:
        return
    with _inproc_sharp_lock:
        load_inproc_sharp_module().load_model()


def inproc_camera_meta(f_px: float, image_shape: tuple[int, int]) -> dict[str, Any]:
    image_height, image_width = image_shape
    return {
        "sourceWidth": int(image_width),
        "sourceHeight": int(image_height),
        "camFx": float(f_px),
        "camFy": float(f_px),
        "camCx": float(image_width * 0.5),
        "camCy": float(image_height * 0.5),
    }


def predict_inproc_sharp_gaussians(image_path: Path) -> tuple[Any, dict[str, Any]]:
    with _inproc_sharp_lock:
        sharp = load_inproc_sharp_module()
        start = time.perf_counter()
        gaussians, f_px, image_shape, steps = sharp.predict_gaussians(image_path)
        count = sharp.num_gaussians(gaussians)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        seconds = time.perf_counter() - start
    meta = {
        "splat_path": None,
        "output_path": None,
        "output_format": "gpu_tensors",
        "num_gaussians": count,
        "seconds": seconds,
        "precision": getattr(sharp, "PRECISION", None),
        "fast_unproject": getattr(sharp, "USE_FAST_UNPROJECT", None),
        "splat_row_bytes": SPLAT_ROW_BYTES,
        "steps": steps,
        "inproc": True,
        **inproc_camera_meta(f_px, image_shape),
    }
    log_event("orbit_inproc_sharp_done", imagePath=str(image_path), numGaussians=count, elapsedSeconds=seconds, steps=steps)
    return gaussians, meta


def compact_splat_chunks_from_gaussians(gaussians: Any, *, chunk_records: int = 8_192) -> Any:
    import torch.nn.functional as F

    with torch.inference_mode():
        xyz = gaussians.mean_vectors.flatten(0, 1).float()
        scale = gaussians.singular_values.flatten(0, 1).float()
        colors = linear_to_srgb_torch(gaussians.colors.flatten(0, 1).float()).clamp(0.0, 1.0)
        opacities = gaussians.opacities.flatten(0, 1).float().clamp(0.0, 1.0)
        quaternions = F.normalize(gaussians.quaternions.flatten(0, 1).float(), p=2, dim=-1)
        count = int(xyz.shape[0])
        yield COMPACT_SPLAT_HEADER.pack(
            COMPACT_SPLAT_MAGIC,
            COMPACT_SPLAT_VERSION,
            COMPACT_SPLAT_ROW_BYTES,
            count,
            0,
        )

        permutation = np.random.default_rng(SPLAT_STREAM_SHUFFLE_SEED).permutation(count)
        row_dtype = np.dtype(
            [
                ("xyz", "<f2", (3,)),
                ("scale", "<f2", (3,)),
                ("rgba", "u1", (4,)),
                ("rotation", "u1", (4,)),
            ]
        )
        device = xyz.device
        for start in range(0, count, chunk_records):
            perm_chunk = permutation[start : start + chunk_records]
            idx = torch.as_tensor(perm_chunk, device=device, dtype=torch.long)
            rows = np.empty(len(perm_chunk), dtype=row_dtype)
            rows["xyz"] = xyz.index_select(0, idx).detach().cpu().numpy().astype("<f2", copy=False)
            rows["scale"] = scale.index_select(0, idx).detach().cpu().numpy().astype("<f2", copy=False)
            rgb = (colors.index_select(0, idx) * 255.0).round().to(torch.uint8).detach().cpu().numpy()
            alpha = (opacities.index_select(0, idx) * 255.0).round().to(torch.uint8).detach().cpu().numpy().reshape(-1, 1)
            rows["rgba"] = np.concatenate([rgb, alpha], axis=1)
            rows["rotation"] = (
                (quaternions.index_select(0, idx) * 128.0) + 128.0
            ).round().clamp(0, 255).to(torch.uint8).detach().cpu().numpy()
            yield rows.tobytes()


def loaded_asset_from_inproc_gaussians(spec: AssetSpec, gaussians: Any) -> LoadedAsset:
    import torch.nn.functional as F

    t0 = time.perf_counter()
    means = gaussians.mean_vectors.flatten(0, 1).float().contiguous()
    quats = F.normalize(gaussians.quaternions.flatten(0, 1).float(), p=2, dim=-1).contiguous()
    scales = gaussians.singular_values.flatten(0, 1).float().contiguous()
    opacities = gaussians.opacities.flatten(0, 1).float().clamp(0.0, 1.0).contiguous()
    colors_linear = gaussians.colors.flatten(0, 1).float().clamp(0.0, 1.0).contiguous()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return LoadedAsset(
        spec=spec,
        means=means,
        quats=quats,
        scales=scales,
        opacities=opacities,
        colors_linear=colors_linear,
        count=int(means.shape[0]),
        loaded_seconds=time.perf_counter() - t0,
        last_used=time.time(),
    )


def load_remote_gallery_asset(spec: AssetSpec) -> LoadedAsset:
    """Load an uncached tensor bundle whose lifetime belongs to one session."""

    started = time.perf_counter()
    if spec.kind == "splat":
        means_np, quats_np, scales_np, opacities_np, colors_np = load_splat_arrays(spec.path)
    elif spec.kind == "ply":
        means_np, quats_np, scales_np, opacities_np, colors_np = load_ply_arrays(spec.path)
    else:
        raise ValueError(f"unsupported asset kind {spec.kind}")
    asset = LoadedAsset(
        spec=spec,
        means=torch.from_numpy(means_np).to("cuda", non_blocking=True),
        quats=torch.from_numpy(quats_np).to("cuda", non_blocking=True),
        scales=torch.from_numpy(scales_np).to("cuda", non_blocking=True),
        opacities=torch.from_numpy(opacities_np).to("cuda", non_blocking=True),
        colors_linear=torch.from_numpy(colors_np).to("cuda", non_blocking=True),
        count=int(means_np.shape[0]),
        loaded_seconds=0.0,
        last_used=time.time(),
    )
    torch.cuda.synchronize()
    asset.loaded_seconds = time.perf_counter() - started
    return asset


def remote_camera_for_spec(spec: AssetSpec) -> RemoteCamera:
    width, height = DEFAULT_WIDTH, DEFAULT_HEIGHT
    scale = max(width / spec.source_width, height / spec.source_height)
    k = spec.k_source.copy()
    k[0, :] *= scale
    k[1, :] *= scale
    k[0, 2] -= (spec.source_width * scale - width) * 0.5
    k[1, 2] -= (spec.source_height * scale - height) * 0.5
    initial_w2c = viewmat_for_offset(0.0, 0.0).reshape(-1)
    return RemoteCamera(
        convention=CAMERA_CONVENTION,
        w2c=tuple(float(value) for value in initial_w2c),
        fx=float(k[0, 0]),
        fy=float(k[1, 1]),
        cx=float(k[0, 2]),
        cy=float(k[1, 2]),
        width=width,
        height=height,
    )


def remote_resource(asset: LoadedAsset) -> RemoteSessionResource:
    return RemoteSessionResource(
        handle=RemoteAssetHandle(asset),
        initial_camera=remote_camera_for_spec(asset.spec),
        pivot=(0.0, 0.0, 1.0),
        focus_distance=1.0,
        source_width=asset.spec.source_width,
        source_height=asset.spec.source_height,
    )


def encode_remote_image(array: np.ndarray, image_format: str, *, quality: int | None = None) -> bytes:
    output = io.BytesIO()
    image = Image.fromarray(array)
    save_options: dict[str, Any] = {}
    if quality is not None:
        save_options.update(quality=quality, optimize=True)
    image.save(output, format=image_format, **save_options)
    return output.getvalue()


class OrbitRemoteRenderEngine:
    """Maps remote-render-v1 resources to Orbit's existing gsplat pipeline."""

    def create_gallery(self, photo_id: str) -> RemoteSessionResource:
        try:
            # Only photoId crosses this boundary; direct path/splat parameters
            # are intentionally impossible on the remote session route.
            spec = resolve_asset({"photoId": [photo_id]})
            return remote_resource(load_remote_gallery_asset(spec))
        except FileNotFoundError as exc:
            raise RemoteProtocolError(404, "asset_not_found", "gallery asset not found") from exc
        except RemoteProtocolError:
            raise
        except torch.cuda.OutOfMemoryError as exc:
            raise RemoteProtocolError(503, "gpu_out_of_memory", "GPU memory exhausted", retryable=True) from exc
        except (ValueError, OSError) as exc:
            raise RemoteProtocolError(400, "invalid_source", "gallery source is invalid") from exc
        except RuntimeError as exc:
            raise self._runtime_error(exc) from exc

    def create_upload(self, image: bytes, filename: str, content_type: str) -> RemoteSessionResource:
        if not ORBIT_INPROC_SHARP:
            raise RemoteProtocolError(
                503,
                "sharp_unavailable",
                "in-process SHARP is required for upload sessions",
                retryable=False,
            )
        try:
            with Image.open(io.BytesIO(image)) as decoded:
                decoded = ImageOps.exif_transpose(decoded)
                width, height = decoded.size
                if width <= 0 or height <= 0 or width * height > INGEST_MAX_PIXELS:
                    raise ValueError("upload image dimensions exceed limits")
                decoded = decoded.convert("RGB")
                longest = max(width, height)
                if longest > INGEST_MAX_SOURCE_SIDE:
                    scale = INGEST_MAX_SOURCE_SIDE / float(longest)
                    decoded = decoded.resize(
                        (max(1, round(width * scale)), max(1, round(height * scale))),
                        Image.Resampling.LANCZOS,
                    )
            with tempfile.TemporaryDirectory(prefix="wici-remote-render-") as temp_dir:
                source_path = Path(temp_dir) / "source.jpg"
                decoded.save(source_path, format="JPEG", quality=INGEST_JPEG_QUALITY, optimize=True)
                gaussians, meta = predict_inproc_sharp_gaussians(source_path)
            source_width = int(meta["sourceWidth"])
            source_height = int(meta["sourceHeight"])
            spec = AssetSpec(
                path=Path("remote-upload"),
                kind="inproc",
                source_width=source_width,
                source_height=source_height,
                k_source=np.array(
                    [
                        [float(meta["camFx"]), 0.0, float(meta["camCx"])],
                        [0.0, float(meta["camFy"]), float(meta["camCy"])],
                        [0.0, 0.0, 1.0],
                    ],
                    dtype=np.float32,
                ),
                photo_id=None,
                title=Path(filename).stem,
            )
            return remote_resource(loaded_asset_from_inproc_gaussians(spec, gaussians))
        except RemoteProtocolError:
            raise
        except (Image.UnidentifiedImageError, ValueError, OSError) as exc:
            raise RemoteProtocolError(400, "invalid_source", "uploaded image is invalid") from exc
        except torch.cuda.OutOfMemoryError as exc:
            raise RemoteProtocolError(503, "gpu_out_of_memory", "GPU memory exhausted", retryable=True) from exc
        except RuntimeError as exc:
            raise self._runtime_error(exc) from exc

    def render(self, handle: RemoteAssetHandle, camera: RemoteCamera, mode: str) -> RemoteRenderedFrame:
        if handle.released or handle.asset is None:
            raise RemoteProtocolError(410, "session_expired", "session resource was released")
        asset = handle.asset
        try:
            with torch.inference_mode():
                view = torch.tensor(camera.w2c, dtype=torch.float32, device="cuda").reshape(1, 4, 4)
                k = torch.tensor(
                    [[camera.fx, 0.0, camera.cx], [0.0, camera.fy, camera.cy], [0.0, 0.0, 1.0]],
                    dtype=torch.float32,
                    device="cuda",
                )[None]
                colors, alphas, _ = rasterization(
                    means=asset.means,
                    quats=asset.quats,
                    scales=asset.scales,
                    opacities=asset.opacities,
                    colors=asset.colors_linear,
                    viewmats=view,
                    Ks=k,
                    width=camera.width,
                    height=camera.height,
                    render_mode="RGB",
                    rasterize_mode="classic",
                    packed=False,
                    absgrad=False,
                )
                seed_linear = colors[0, ..., :3].clamp(0.0, 1.0)
                alpha = alphas[0, ..., :1].clamp(0.0, 1.0)
                bg_rgb = np.array(ImageColor.getrgb(f"#{DEFAULT_BG_HEX}"), dtype=np.float32) / 255.0
                bg_linear = torch.from_numpy(srgb_to_linear_np(bg_rgb)).to(seed_linear.device)
                preview_linear = seed_linear + (1.0 - alpha) * bg_linear[None, None, :]
                seed = (linear_to_srgb_torch(seed_linear).clamp(0.0, 1.0) * 255.0).round().byte().cpu().numpy()
                preview = (linear_to_srgb_torch(preview_linear).clamp(0.0, 1.0) * 255.0).round().byte().cpu().numpy()
                alpha_u8 = (alpha[..., 0] * 255.0).round().byte().cpu().numpy()
            preview_jpeg = encode_remote_image(
                preview,
                "JPEG",
                quality=REMOTE_CAPTURE_JPEG_QUALITY if mode == "capture" else REMOTE_INTERACTIVE_JPEG_QUALITY,
            )
            return RemoteRenderedFrame(
                preview_jpeg=preview_jpeg,
                width=camera.width,
                height=camera.height,
                seed_jpeg=encode_remote_image(seed, "JPEG", quality=REMOTE_CAPTURE_JPEG_QUALITY) if mode == "capture" else None,
                alpha_png=encode_remote_image(alpha_u8, "PNG") if mode == "capture" else None,
            )
        except torch.cuda.OutOfMemoryError as exc:
            raise RemoteProtocolError(503, "gpu_out_of_memory", "GPU memory exhausted", retryable=True) from exc
        except RuntimeError as exc:
            raise self._runtime_error(exc) from exc

    def release(self, handle: RemoteAssetHandle) -> None:
        if handle.released:
            return
        handle.released = True
        handle.asset = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @staticmethod
    def _runtime_error(exc: RuntimeError) -> RemoteProtocolError:
        message = str(exc).lower()
        if "out of memory" in message:
            return RemoteProtocolError(503, "gpu_out_of_memory", "GPU memory exhausted", retryable=True)
        if "cuda" in message:
            return RemoteProtocolError(503, "gpu_unavailable", "CUDA render engine unavailable", retryable=True)
        return RemoteProtocolError(503, "render_unavailable", "render engine unavailable", retryable=True)


def run_inproc_sharp_startup_warmup() -> None:
    if not ORBIT_INPROC_SHARP:
        return
    warmup_start = time.perf_counter()
    try:
        with tempfile.TemporaryDirectory(prefix="wici-orbit-sharp-warmup-") as tmp:
            tmp_dir = Path(tmp)
            source_path = tmp_dir / "warmup.jpg"
            width = 512
            height = 512
            x = np.linspace(0, 255, width, dtype=np.uint8)[None, :]
            y = np.linspace(0, 255, height, dtype=np.uint8)[:, None]
            image = np.zeros((height, width, 3), dtype=np.uint8)
            image[..., 0] = x
            image[..., 1] = y
            image[..., 2] = ((x.astype(np.uint16) + y.astype(np.uint16)) // 2).astype(np.uint8)
            Image.fromarray(image, "RGB").save(source_path, format="JPEG", quality=92)

            gaussians, sharp_meta = predict_inproc_sharp_gaussians(source_path)
            source_width = int(sharp_meta["sourceWidth"])
            source_height = int(sharp_meta["sourceHeight"])
            spec = AssetSpec(
                path=tmp_dir / "warmup.splat",
                kind="splat",
                source_width=source_width,
                source_height=source_height,
                k_source=source_k_from_dims(source_width, source_height),
                photo_id="startup-warmup",
                title="startup-warmup",
            )
            render_params = RenderParams(
                width=DEFAULT_WEBP_WIDTH,
                height=DEFAULT_WEBP_HEIGHT,
                fps=DEFAULT_WEBP_FPS,
                seconds=8 / DEFAULT_WEBP_FPS,
                frames=8,
                az_deg=DEFAULT_PORTRAIT_AZ_DEG,
                pol_deg=DEFAULT_PORTRAIT_POL_DEG,
                bitrate_kbps=DEFAULT_WEBP_BITRATE_KBPS,
                bg_hex=DEFAULT_BG_HEX,
                output_format="webp",
            )
            asset = loaded_asset_from_inproc_gaussians(spec, gaussians)
            generate_start = time.perf_counter()
            frame_buffer = render_orbit_frames(asset, render_params)
            meta = encode_orbit_frames(
                asset,
                render_params,
                frame_buffer,
                tmp_dir / "warmup.webp",
                tmp_dir / "warmup.json",
                generate_start=generate_start,
                gpu_lock_seconds=frame_buffer.render_seconds,
            )
            log_event(
                "orbit_inproc_sharp_warmup_done",
                elapsedSeconds=time.perf_counter() - warmup_start,
                sharpSeconds=sharp_meta.get("seconds"),
                assetLoadSeconds=asset.loaded_seconds,
                gpuRenderSeconds=frame_buffer.render_seconds,
                ffmpegEncodeSeconds=meta.get("ffmpegEncodeSeconds"),
                frames=render_params.frames,
                width=render_params.width,
                height=render_params.height,
                bytes=meta.get("fileBytes"),
            )
            del gaussians, asset, frame_buffer
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    except Exception as exc:  # noqa: BLE001 - warmup must not block startup.
        log_event(
            "orbit_inproc_sharp_warmup_failed",
            elapsedSeconds=time.perf_counter() - warmup_start,
            error=repr(exc),
            traceback=traceback.format_exc(limit=8),
        )


def gallery_payload_for_photo(photo_id: str) -> dict[str, Any]:
    record = gallery_record_for_photo_id(photo_id)
    if record is None:
        raise FileNotFoundError(f"unknown photoId={photo_id!r}")
    item = public_gallery_items([record])[0]
    quoted = quote(photo_id)
    item.update(
        {
            "orbitPreviewUrl": f"/orbit_preview?photoId={quoted}",
            "orbitWebpUrl": f"/orbit_preview?photoId={quoted}&format=webp",
            "splatStreamUrl": f"/splat_stream?photoId={quoted}&density=full",
            "splatUrl": f"/splat?photoId={quoted}&density=full",
        }
    )
    return item


def run_ingest_compact_stream(upload: IngestUpload, *, request_start: float) -> tuple[Any, dict[str, Any]]:
    if not ORBIT_INPROC_SHARP:
        raise RuntimeError("compact chunked ingest streaming requires WICI_ORBIT_INPROC_SHARP=1")
    timing: dict[str, Any] = {
        "mode": "stateless_sharp_splat_stream_chunked",
        "inputBytes": len(upload.image),
        "compactTransportRequested": True,
        "compactStreamRequested": True,
        "_requestStartPerfCounter": request_start,
    }
    t0 = time.perf_counter()
    photo_id = reserve_user_photo_id(upload)
    timing["reservePhotoIdSeconds"] = time.perf_counter() - t0
    with tempfile.TemporaryDirectory(prefix="wici-ingest-") as tmp:
        tmp_dir = Path(tmp)
        t0 = time.perf_counter()
        source_meta = save_ingest_source_temp(upload, photo_id, tmp_dir)
        timing["sourceDecodeResizeTempWriteSeconds"] = time.perf_counter() - t0
        timing["sourceBytes"] = source_meta["sourceBytes"]
        t0 = time.perf_counter()
        gaussians, sharp_meta = predict_inproc_sharp_gaussians(Path(source_meta["sourcePath"]))
        timing["sharpInprocSeconds"] = time.perf_counter() - t0
        timing["sharpPath"] = "inproc"

    timing["sharpServerSeconds"] = sharp_meta.get("seconds")
    timing["sharpServerSteps"] = sharp_meta.get("steps")
    timing["sharpClient"] = sharp_meta.get("clientTiming")
    timing["responseSplatFormat"] = COMPACT_SPLAT_FORMAT
    timing["responseSplatRowBytes"] = COMPACT_SPLAT_ROW_BYTES
    timing["responseSplatEncoding"] = COMPACT_SPLAT_IDENTITY_ENCODING
    timing["responseTransfer"] = "chunked"
    gaussian_count = int(sharp_meta["num_gaussians"])
    compact_payload_bytes = COMPACT_SPLAT_HEADER_BYTES + gaussian_count * COMPACT_SPLAT_ROW_BYTES
    fp32_bytes = gaussian_count * SPLAT_ROW_BYTES
    timing["responseBytes"] = compact_payload_bytes
    timing["compactPayloadBytes"] = compact_payload_bytes
    timing["compressedPayloadBytes"] = 0
    timing["fp32EquivalentBytes"] = fp32_bytes
    timing["compactRatioVsFp32"] = compact_payload_bytes / max(1, fp32_bytes)
    timing["gzipRatioVsCompact"] = None
    timing["gzipRatioVsFp32"] = None
    timing["splatBytes"] = fp32_bytes
    timing["numGaussians"] = gaussian_count
    timing["handlerBeforeResponseSeconds"] = time.perf_counter() - request_start

    if all(sharp_meta.get(key) is not None for key in ("sourceWidth", "sourceHeight", "camFx", "camFy", "camCx", "camCy")):
        camera_meta = {
            "sourceWidth": int(sharp_meta["sourceWidth"]),
            "sourceHeight": int(sharp_meta["sourceHeight"]),
            "camFx": float(sharp_meta["camFx"]),
            "camFy": float(sharp_meta["camFy"]),
            "camCx": float(sharp_meta["camCx"]),
            "camCy": float(sharp_meta["camCy"]),
        }
    else:
        camera_meta = camera_dict(
            int(source_meta["sourceWidth"]),
            int(source_meta["sourceHeight"]),
            source_k_from_dims(int(source_meta["sourceWidth"]), int(source_meta["sourceHeight"])),
        )

    result = {
        "ok": True,
        "photoId": photo_id,
        "filename": upload.filename,
        "userUpload": True,
        "hasSplat": True,
        "hasOrbit": False,
        "splatFormat": "splat",
        "splatRowBytes": SPLAT_ROW_BYTES,
        "responseSplatFormat": COMPACT_SPLAT_FORMAT,
        "responseSplatRowBytes": COMPACT_SPLAT_ROW_BYTES,
        "responseSplatEncoding": COMPACT_SPLAT_IDENTITY_ENCODING,
        "responseSplatHeaderBytes": COMPACT_SPLAT_HEADER_BYTES,
        "responseTransfer": "chunked",
        "responsePayloadBytes": compact_payload_bytes,
        "responseBytes": compact_payload_bytes,
        "fp32SplatBytes": fp32_bytes,
        "compactPayloadBytes": compact_payload_bytes,
        "compressedPayloadBytes": 0,
        "numGaussians": gaussian_count,
        "splatBytes": fp32_bytes,
        "originalWidth": source_meta["originalWidth"],
        "originalHeight": source_meta["originalHeight"],
        **camera_meta,
        "storage": {
            "policy": "stateless",
            "persistentFilesWritten": 0,
            "serverCache": "none",
        },
        "limits": {
            "maxUploadBytes": INGEST_MAX_BODY_BYTES,
            "maxSourceSide": INGEST_MAX_SOURCE_SIDE,
            "maxPixels": INGEST_MAX_PIXELS,
        },
        "timing": timing,
        "sharp": sharp_meta,
    }
    return compact_splat_chunks_from_gaussians(gaussians), result


def run_ingest_preview(upload: IngestUpload, params: dict[str, list[str]]) -> tuple[bytes, dict[str, Any]]:
    request_start = time.perf_counter()
    timing: dict[str, Any] = {
        "mode": "stateless_sharp_splat_preview",
        "inputBytes": len(upload.image),
    }
    t0 = time.perf_counter()
    photo_id = reserve_user_photo_id(upload)
    timing["reservePhotoIdSeconds"] = time.perf_counter() - t0
    with tempfile.TemporaryDirectory(prefix="wici-ingest-preview-") as tmp:
        tmp_dir = Path(tmp)
        t0 = time.perf_counter()
        source_meta = save_ingest_source_temp(upload, photo_id, tmp_dir)
        timing["sourceDecodeResizeTempWriteSeconds"] = time.perf_counter() - t0
        timing["sourceBytes"] = source_meta["sourceBytes"]

        splat_path = tmp_dir / f"{photo_id}.splat"
        inproc_gaussians = None
        t0 = time.perf_counter()
        if ORBIT_INPROC_SHARP:
            inproc_gaussians, sharp_meta = predict_inproc_sharp_gaussians(Path(source_meta["sourcePath"]))
            timing["sharpInprocSeconds"] = time.perf_counter() - t0
            timing["sharpPath"] = "inproc"
            splat_bytes = int(sharp_meta["num_gaussians"]) * SPLAT_ROW_BYTES
        else:
            sharp_meta = call_sharp_infer(Path(source_meta["sourcePath"]), splat_path, "splat")
            timing["sharpHttpSeconds"] = time.perf_counter() - t0
            timing["sharpPath"] = "http"
            splat_bytes = splat_path.stat().st_size
            if splat_bytes % SPLAT_ROW_BYTES != 0:
                raise ValueError(f"invalid .splat size {splat_bytes}; not divisible by {SPLAT_ROW_BYTES}")
        timing["sharpClient"] = sharp_meta.get("clientTiming")
        timing["sharpServer"] = sharp_meta
        timing["splatBytes"] = splat_bytes
        timing["numGaussians"] = splat_bytes // SPLAT_ROW_BYTES

        source_width = int(sharp_meta.get("sourceWidth") or source_meta["sourceWidth"])
        source_height = int(sharp_meta.get("sourceHeight") or source_meta["sourceHeight"])
        spec = AssetSpec(
            path=splat_path,
            kind="splat",
            source_width=source_width,
            source_height=source_height,
            k_source=source_k_from_dims(source_width, source_height),
            photo_id=photo_id,
            title=upload.fields.get("title") or Path(upload.filename).stem,
        )
        render_query = {key: list(value) for key, value in params.items()}
        render_query["format"] = ["webp"]
        render_params = parse_render_params(render_query, spec)
        out_path = tmp_dir / f"{photo_id}-orbit.webp"
        meta_path = tmp_dir / f"{photo_id}-orbit.json"
        cache_key_for_temp_asset = str(splat_path.resolve())
        with _lock:
            lock_start = time.perf_counter()
            try:
                t0 = time.perf_counter()
                if inproc_gaussians is not None:
                    asset = loaded_asset_from_inproc_gaussians(spec, inproc_gaussians)
                else:
                    asset = load_asset(spec)
                timing["assetLoadSeconds"] = time.perf_counter() - t0
                generate_start = time.perf_counter()
                frame_buffer = render_orbit_frames(asset, render_params)
            finally:
                if not ORBIT_INPROC_SHARP and cache_key_for_temp_asset in _asset_cache:
                    del _asset_cache[cache_key_for_temp_asset]
                    torch.cuda.empty_cache()
            gpu_lock_seconds = time.perf_counter() - lock_start
        meta = encode_orbit_frames(
            asset,
            render_params,
            frame_buffer,
            out_path,
            meta_path,
            generate_start=generate_start,
            gpu_lock_seconds=gpu_lock_seconds,
        )
        data = out_path.read_bytes()

    timing["totalSeconds"] = time.perf_counter() - request_start
    meta["timing"] = timing
    meta["storage"] = {
        "policy": "stateless",
        "persistentFilesWritten": 0,
        "serverCache": "none",
    }
    meta["fileBytes"] = len(data)
    return data, meta


_remote_render_manager: RemoteSessionManager | None = (
    RemoteSessionManager(OrbitRemoteRenderEngine(), limits=REMOTE_LIMITS, render_lock=_lock)
    if REMOTE_RENDER_ENABLED
    else None
)


class Handler(RemoteRouteMixin, BaseHTTPRequestHandler):
    server_version = "WiCiOrbitPreview/1.0"
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt: str, *args: Any) -> None:
        print(fmt % args, flush=True)

    def do_OPTIONS(self) -> None:
        json_response(self, 204, {})

    def do_GET(self) -> None:
        global _last_preview
        parsed = urlparse(self.path)
        route = parsed.path
        params = parse_qs(parsed.query)

        if route == "/health":
            remote_stats = _remote_render_manager.public_stats() if _remote_render_manager is not None else {
                "activeSessions": 0,
                "maxSessions": REMOTE_LIMITS.max_sessions,
                "recentExpiredSessions": 0,
            }
            json_response(
                self,
                200,
                {
                    "ok": True,
                    "cuda": torch.cuda.is_available(),
                    "torch": torch.__version__,
                    "cudaVersion": torch.version.cuda,
                    "gsplat": "available",
                    "capabilities": [REMOTE_RENDER_PROTOCOL] if _remote_render_manager is not None else [],
                    "remoteRender": {
                        "enabled": _remote_render_manager is not None,
                        "protocol": REMOTE_RENDER_PROTOCOL,
                        "cameraConvention": CAMERA_CONVENTION,
                        "limits": REMOTE_LIMITS.public(),
                        "sessions": remote_stats,
                    },
                    "cacheDir": str(CACHE_DIR),
                    "userRoot": str(PHOTOS_USER_ROOT),
                    "userGalleryCount": len(load_user_gallery()),
                    "ingest": {
                        "enabled": True,
                        "storagePolicy": "stateless",
                        "sharpInferUrl": SHARP_INFER_URL,
                        "sharpInferUploadUrl": SHARP_INFER_UPLOAD_URL,
                        "responseFormat": "application/octet-stream .splat",
                        "maxUploadBytes": INGEST_MAX_BODY_BYTES,
                        "maxSourceSide": INGEST_MAX_SOURCE_SIDE,
                        "maxPixels": INGEST_MAX_PIXELS,
                        "persistentFilesWritten": 0,
                        "inprocSharp": ORBIT_INPROC_SHARP,
                        "sharpHttpFallbackUrl": SHARP_INFER_URL,
                    },
                    "loadedAssets": [
                        {
                            "path": str(asset.spec.path),
                            "photoId": asset.spec.photo_id,
                            "count": asset.count,
                            "sourceWidth": asset.spec.source_width,
                            "sourceHeight": asset.spec.source_height,
                            "loadedSeconds": asset.loaded_seconds,
                        }
                        for asset in _asset_cache.values()
                    ],
                    "defaults": {
                        "width": DEFAULT_WIDTH,
                        "height": DEFAULT_HEIGHT,
                        "webpWidth": DEFAULT_WEBP_WIDTH,
                        "webpHeight": DEFAULT_WEBP_HEIGHT,
                        "maxHeight": DEFAULT_MAX_HEIGHT,
                        "fps": DEFAULT_FPS,
                        "webpFps": DEFAULT_WEBP_FPS,
                        "seconds": DEFAULT_SECONDS,
                        "azDeg": DEFAULT_AZ_DEG,
                        "polDeg": DEFAULT_POL_DEG,
                        "landscapeAzDeg": DEFAULT_AZ_DEG,
                        "landscapePolDeg": DEFAULT_POL_DEG,
                        "portraitAzDeg": DEFAULT_PORTRAIT_AZ_DEG,
                        "portraitPolDeg": DEFAULT_PORTRAIT_POL_DEG,
                        "bitrateKbps": DEFAULT_BITRATE_KBPS,
                        "webpBitrateKbps": DEFAULT_WEBP_BITRATE_KBPS,
                        "webpQuality": DEFAULT_WEBP_QUALITY,
                        "formats": ["mp4", "webp"],
                        "background": DEFAULT_BG_HEX,
                    },
                    "lastPreview": _last_preview,
                    "lastIngest": _last_ingest,
                    "processes": smi_processes(),
                },
            )
            return

        if route == "/gallery":
            try:
                include_user = bool_param(first(params, "includeUser", "include_user", "user"), False)
                records = build_gallery_records(include_user=include_user)
                json_response(self, 200, public_gallery_items(records))
            except Exception as exc:  # noqa: BLE001 - client needs the exact blocker.
                log_event("gallery_failed", error=repr(exc), traceback=traceback.format_exc(limit=8))
                json_response(
                    self,
                    500,
                    {
                        "ok": False,
                        "error": "gallery_failed",
                        "message": repr(exc),
                    },
                )
            return

        if route == "/source":
            try:
                photo_id = first(params, "photoId", "photo_id", "id")
                if not photo_id:
                    raise ValueError("provide photoId")
                record = gallery_record_for_photo_id(photo_id)
                if record is None:
                    raise FileNotFoundError(f"unknown photoId={photo_id!r}")
                source_path = record.get("sourcePath")
                if not source_path:
                    raise FileNotFoundError(f"no source image found for photoId={photo_id!r}")
                source = Path(str(source_path))
                if not source.exists():
                    raise FileNotFoundError(f"source image does not exist: {source}")
                source_response(self, source, record=record)
            except Exception as exc:  # noqa: BLE001 - client needs the exact blocker.
                log_event("source_serve_failed", error=repr(exc), traceback=traceback.format_exc(limit=8))
                json_response(
                    self,
                    404 if isinstance(exc, FileNotFoundError) else 500,
                    {
                        "ok": False,
                        "error": "source_serve_failed",
                        "message": repr(exc),
                    },
                )
            return

        if route == "/thumbnail":
            try:
                photo_id = first(params, "photoId", "photo_id", "id")
                if not photo_id:
                    raise ValueError("provide photoId")
                record = gallery_record_for_photo_id(photo_id)
                if record is None:
                    raise FileNotFoundError(f"unknown photoId={photo_id!r}")
                force = bool_param(first(params, "force", "refresh"), False)
                thumb_path, cache_hit = get_or_create_thumbnail(record, force=force)
                image_response(self, thumb_path, cache_hit=cache_hit)
            except Exception as exc:  # noqa: BLE001 - client needs the exact blocker.
                log_event("thumbnail_failed", error=repr(exc), traceback=traceback.format_exc(limit=8))
                json_response(
                    self,
                    404 if isinstance(exc, FileNotFoundError) else 500,
                    {
                        "ok": False,
                        "error": "thumbnail_failed",
                        "message": repr(exc),
                    },
                )
            return

        if route == "/splat":
            try:
                path, photo_id, requested_density, served_density, fallback = resolve_ply_asset(params)
                ply_response(
                    self,
                    path,
                    photo_id=photo_id,
                    requested_density=requested_density,
                    served_density=served_density,
                    fallback=fallback,
                )
            except Exception as exc:  # noqa: BLE001 - client needs the exact blocker.
                log_event("splat_serve_failed", error=repr(exc), traceback=traceback.format_exc(limit=8))
                json_response(
                    self,
                    404 if isinstance(exc, FileNotFoundError) else 500,
                    {
                        "ok": False,
                        "error": "splat_serve_failed",
                        "message": repr(exc),
                        "format": "ply",
                    },
                )
            return

        if route == "/splat_stream":
            try:
                path, photo_id, requested_density, served_density, fallback = resolve_ply_asset(params)
                force = bool_param(first(params, "force", "refresh"), False)
                out_path, meta, cache_hit = get_or_create_splat_stream(
                    path,
                    photo_id=photo_id,
                    requested_density=requested_density,
                    served_density=served_density,
                    fallback=fallback,
                    force=force,
                )
                splat_stream_response(self, out_path, meta=meta, cache_hit=cache_hit)
            except Exception as exc:  # noqa: BLE001 - client needs the exact blocker.
                log_event("splat_stream_failed", error=repr(exc), traceback=traceback.format_exc(limit=8))
                retryable = "out of memory" in str(exc).lower()
                json_response(
                    self,
                    404 if isinstance(exc, FileNotFoundError) else 503 if retryable else 500,
                    {
                        "ok": False,
                        "error": "splat_stream_failed",
                        "message": repr(exc),
                        "retryable": retryable,
                    },
                )
            return

        if route != "/orbit_preview":
            json_response(self, 404, {"ok": False, "error": "not_found"})
            return

        request_start = time.perf_counter()
        try:
            spec = resolve_asset(params)
            render_params = parse_render_params(params, spec)
            force = bool_param(first(params, "force", "refresh"), False)
            out_path, meta, cache_hit = get_or_create_preview(spec, render_params, force)
            _last_preview = {
                **meta,
                "cache": "hit" if cache_hit else "miss",
                "handlerSeconds": time.perf_counter() - request_start,
            }
            name_id = spec.photo_id or spec.path.stem
            filename = f"{name_id}-orbit-{render_params.width}x{render_params.height}.{render_params.output_format}"
            orbit_response(self, out_path, filename=filename, meta=meta, cache_hit=cache_hit)
        except Exception as exc:  # noqa: BLE001 - client needs the exact blocker.
            log_event("orbit_preview_failed", error=repr(exc), traceback=traceback.format_exc(limit=8))
            retryable = "cuda" in str(exc).lower() or "out of memory" in str(exc).lower()
            json_response(
                self,
                404 if isinstance(exc, FileNotFoundError) else 503 if retryable else 500,
                {
                    "ok": False,
                    "error": "orbit_preview_failed",
                    "message": repr(exc),
                    "retryable": retryable,
                },
            )

    def do_POST(self) -> None:
        global _last_ingest, _last_preview
        if self.dispatch_remote(_remote_render_manager, "POST"):
            return
        request_start = time.perf_counter()
        parsed = urlparse(self.path)
        route = parsed.path
        params = parse_qs(parsed.query)
        if route == "/ingest_preview":
            try:
                t0 = time.perf_counter()
                upload = parse_ingest_request(self)
                parse_seconds = time.perf_counter() - t0
                preview_bytes, meta = run_ingest_preview(upload, params)
                meta["timing"]["parseMultipartSeconds"] = parse_seconds
                meta["timing"]["handlerBeforeResponseSeconds"] = time.perf_counter() - request_start
                _last_preview = {
                    **meta,
                    "cache": "none",
                    "handlerSeconds": time.perf_counter() - request_start,
                }
                log_event(
                    "ingest_preview_done",
                    photoId=meta.get("photoId"),
                    timing=meta.get("timing"),
                    bytes=len(preview_bytes),
                )
                name_id = meta.get("photoId") or Path(upload.filename).stem
                orbit_bytes_response(
                    self,
                    preview_bytes,
                    filename=f"{name_id}-orbit.webp",
                    meta=meta,
                )
            except Exception as exc:  # noqa: BLE001 - client needs the exact blocker.
                log_event("ingest_preview_failed", error=repr(exc), traceback=traceback.format_exc(limit=8))
                message = repr(exc)
                if isinstance(exc, OSError) and "low disk space" in str(exc):
                    status = 507
                elif isinstance(exc, ValueError):
                    status = 400
                else:
                    status = 500
                json_response(
                    self,
                    status,
                    {
                        "ok": False,
                        "error": "ingest_preview_failed",
                        "message": message,
                        "maxUploadBytes": INGEST_MAX_BODY_BYTES,
                        "diskFreeBytes": disk_free_bytes(VIEWER_ROOT),
                    },
                )
            return

        if route != "/ingest":
            json_response(self, 404, {"ok": False, "error": "not_found"})
            return
        try:
            t0 = time.perf_counter()
            upload = parse_ingest_request(self)
            parse_seconds = time.perf_counter() - t0
            chunks, result = run_ingest_compact_stream(upload, request_start=request_start)
            result["timing"]["parseMultipartSeconds"] = parse_seconds
            result["timing"]["handlerBeforeResponseSeconds"] = time.perf_counter() - request_start
            _last_ingest = {
                "photoId": result.get("photoId"),
                "timing": result.get("timing"),
                "responseBytes": result.get("responseBytes"),
                "splatBytes": result.get("splatBytes"),
                "responseSplatFormat": result.get("responseSplatFormat"),
                "responseSplatEncoding": result.get("responseSplatEncoding"),
                "responseTransfer": result.get("responseTransfer"),
                "numGaussians": result.get("numGaussians"),
                "storage": result.get("storage"),
                "createdAt": time.time(),
            }
            log_event(
                "ingest_stream_response_start",
                photoId=result.get("photoId"),
                timing=result.get("timing"),
                storage=result.get("storage"),
            )
            stream_stats = ingest_splat_chunked_response(self, 201, chunks, result)
            _last_ingest = {
                "photoId": result.get("photoId"),
                "timing": result.get("timing"),
                "responseBytes": stream_stats.get("bytesSent"),
                "splatBytes": result.get("splatBytes"),
                "responseSplatFormat": result.get("responseSplatFormat"),
                "responseSplatEncoding": result.get("responseSplatEncoding"),
                "responseTransfer": result.get("responseTransfer"),
                "numGaussians": result.get("numGaussians"),
                "storage": result.get("storage"),
                "createdAt": time.time(),
            }
            log_event(
                "ingest_done",
                photoId=result.get("photoId"),
                timing=result.get("timing"),
                storage=result.get("storage"),
                streamStats=stream_stats,
            )
        except Exception as exc:  # noqa: BLE001 - client needs the exact blocker.
            log_event("ingest_failed", error=repr(exc), traceback=traceback.format_exc(limit=8))
            message = repr(exc)
            if isinstance(exc, OSError) and "low disk space" in str(exc):
                status = 507
            elif isinstance(exc, ValueError):
                status = 400
            else:
                status = 500
            json_response(
                self,
                status,
                {
                    "ok": False,
                    "error": "ingest_failed",
                    "message": message,
                    "maxUploadBytes": INGEST_MAX_BODY_BYTES,
                    "diskFreeBytes": disk_free_bytes(VIEWER_ROOT),
                },
            )

    def do_DELETE(self) -> None:
        if self.dispatch_remote(_remote_render_manager, "DELETE"):
            return
        json_response(self, 404, {"ok": False, "error": "not_found"})


class Server(ThreadingHTTPServer):
    daemon_threads = True


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default=os.environ.get("WICI_ORBIT_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("WICI_ORBIT_PORT", "8772")))
    args = parser.parse_args()
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    ensure_inproc_sharp_loaded()
    run_inproc_sharp_startup_warmup()
    log_event(
        "orbit_preview_endpoint_start",
        host=args.host,
        port=args.port,
        viewerRoot=str(VIEWER_ROOT),
        cacheDir=str(CACHE_DIR),
        torch=torch.__version__,
        cuda=torch.version.cuda,
        ldPreload=os.environ.get("LD_PRELOAD", ""),
        inprocSharp=ORBIT_INPROC_SHARP,
    )
    server = Server((args.host, args.port), Handler)
    try:
        server.serve_forever()
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
