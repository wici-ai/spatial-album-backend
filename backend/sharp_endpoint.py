#!/usr/bin/env python3
"""Resident SHARP inference endpoint used in-process by orbit."""
from __future__ import annotations

import argparse
import gc
import json
import os
import shutil
import tempfile
import threading
import time
from email import policy
from email.parser import BytesParser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any


DEFAULT_CKPT = "/models/torch/hub/checkpoints/sharp_2572gikvuh.pt"
CKPT = Path(os.environ.get("SHARP_CKPT", DEFAULT_CKPT))
MAX_BODY_BYTES = int(os.environ.get("SHARP_MAX_BODY_MB", "128")) * 1024 * 1024
PRECISION = os.environ.get("SHARP_PRECISION", "fp16").strip().lower()
PROFILE_STEPS = os.environ.get("SHARP_PROFILE_STEPS", "0").strip().lower() in {"1", "true", "yes"}
USE_CHANNELS_LAST = os.environ.get("SHARP_CHANNELS_LAST", "0").strip().lower() in {"1", "true", "yes"}
USE_CUDNN_BENCHMARK = os.environ.get("SHARP_CUDNN_BENCHMARK", "0").strip().lower() in {"1", "true", "yes"}
USE_FAST_UNPROJECT = os.environ.get("SHARP_FAST_UNPROJECT", "1").strip().lower() not in {"0", "false", "no"}
REQUESTED_SDP_BACKEND = os.environ.get("SHARP_SDP_BACKEND", "").strip().lower()
REQUESTED_LINALG_BACKEND = os.environ.get("SHARP_LINALG_BACKEND", "").strip().lower()
FAST_DECOMP_BACKEND = os.environ.get("SHARP_FAST_DECOMP", "analytic").strip().lower()
INTERNAL_SHAPE = (1536, 1536)
SPLAT_ROW_BYTES = 32

_lock = threading.Lock()
_model: Any | None = None
_torch: Any | None = None
_device: Any | None = None
_loaded_seconds: float | None = None
_sdp_backend: str | None = None
_linalg_backend: str | None = None


def log_event(event: str, **fields: Any) -> None:
    print(json.dumps({"event": event, **fields}, separators=(",", ":")), flush=True)


def configure_sdp_backend(torch: Any, capability: tuple[int, int]) -> str:
    if not hasattr(torch.backends, "cuda"):
        return "unavailable"

    requested = REQUESTED_SDP_BACKEND
    legacy_disable = os.environ.get("SHARP_DISABLE_FUSED_SDP")
    if not requested:
        if legacy_disable is not None and legacy_disable.strip().lower() not in {"0", "false", "no"}:
            requested = "math"
        elif capability == (12, 0) and str(torch.version.cuda or "").startswith("12.8"):
            requested = "mem_efficient"
        else:
            requested = "auto"

    aliases = {
        "efficient": "mem_efficient",
        "mem": "mem_efficient",
        "memory_efficient": "mem_efficient",
        "flash_attention": "flash",
        "efficient_attention": "mem_efficient",
        "fused": "auto",
        "default": "auto",
        "": "auto",
    }
    requested = aliases.get(requested, requested)
    enabled = {
        "flash": requested in {"auto", "flash"},
        "mem_efficient": requested in {"auto", "mem_efficient"},
        "cudnn": requested in {"auto", "cudnn"},
        "math": requested in {"auto", "math"},
    }
    if not any(enabled.values()):
        raise ValueError(f"unsupported SHARP_SDP_BACKEND={REQUESTED_SDP_BACKEND!r}")

    for name, value in (
        ("enable_flash_sdp", enabled["flash"]),
        ("enable_mem_efficient_sdp", enabled["mem_efficient"]),
        ("enable_cudnn_sdp", enabled["cudnn"]),
        ("enable_math_sdp", enabled["math"]),
    ):
        fn = getattr(torch.backends.cuda, name, None)
        if fn is not None:
            fn(value)

    return requested


def configure_linalg_backend(torch: Any) -> str | None:
    fn = getattr(getattr(torch.backends, "cuda", None), "preferred_linalg_library", None)
    if fn is None:
        return None
    requested = REQUESTED_LINALG_BACKEND
    if requested:
        fn(requested)
    current = fn()
    return str(current).rsplit(".", 1)[-1].lower()


def json_response(handler: BaseHTTPRequestHandler, status: int, data: dict[str, Any]) -> None:
    raw = json.dumps(data, separators=(",", ":")).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(raw)))
    handler.send_header("Access-Control-Allow-Origin", "*")
    handler.send_header("Access-Control-Allow-Methods", "GET,POST,OPTIONS")
    handler.send_header("Access-Control-Allow-Headers", "Content-Type")
    handler.send_header("Cross-Origin-Resource-Policy", "cross-origin")
    handler.end_headers()
    handler.wfile.write(raw)


def splat_response(handler: BaseHTTPRequestHandler, status: int, data: bytes, result: dict[str, Any], timings: dict[str, float]) -> None:
    handler.send_response(status)
    handler.send_header("Content-Type", "application/octet-stream")
    handler.send_header("Content-Length", str(len(data)))
    handler.send_header("Access-Control-Allow-Origin", "*")
    handler.send_header("Access-Control-Allow-Methods", "GET,POST,OPTIONS")
    handler.send_header("Access-Control-Allow-Headers", "Content-Type")
    handler.send_header(
        "Access-Control-Expose-Headers",
        (
            "X-Sharp-Seconds, X-Sharp-Total-Seconds, X-Sharp-Parse-Seconds, "
            "X-Sharp-Num-Gaussians, X-Sharp-Splat-Bytes, X-Sharp-Precision, "
            "X-Sharp-Steps-Json, X-Src-Width, X-Src-Height, X-Cam-Fx, "
            "X-Cam-Fy, X-Cam-Cx, X-Cam-Cy"
        ),
    )
    handler.send_header("Cross-Origin-Resource-Policy", "cross-origin")
    handler.send_header("Content-Disposition", 'attachment; filename="sharp-live.splat"')
    handler.send_header("X-Sharp-Seconds", f"{float(result.get('seconds', 0.0)):.6f}")
    handler.send_header("X-Sharp-Total-Seconds", f"{float(timings.get('total', 0.0)):.6f}")
    handler.send_header("X-Sharp-Parse-Seconds", f"{float(timings.get('parse', 0.0)):.6f}")
    handler.send_header("X-Sharp-Num-Gaussians", str(int(result.get("num_gaussians", 0))))
    handler.send_header("X-Sharp-Splat-Bytes", str(len(data)))
    handler.send_header("X-Sharp-Precision", PRECISION)
    handler.send_header("X-Sharp-Steps-Json", json.dumps(result.get("steps") or {}, separators=(",", ":")))
    for key, header in (
        ("sourceWidth", "X-Src-Width"),
        ("sourceHeight", "X-Src-Height"),
        ("camFx", "X-Cam-Fx"),
        ("camFy", "X-Cam-Fy"),
        ("camCx", "X-Cam-Cx"),
        ("camCy", "X-Cam-Cy"),
    ):
        if result.get(key) is not None:
            handler.send_header(header, str(result[key]))
    handler.end_headers()
    handler.wfile.write(data)


def load_model() -> Any:
    global _device, _linalg_backend, _loaded_seconds, _model, _sdp_backend, _torch
    if _model is not None:
        return _model

    t0 = time.perf_counter()
    log_event("sharp_load_begin", checkpoint=str(CKPT))
    import torch
    from sharp.models import PredictorParams, create_predictor

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    cap = torch.cuda.get_device_capability()
    if cap != (12, 0):
        raise RuntimeError(f"expected CUDA capability (12, 0), got {cap}")
    _sdp_backend = configure_sdp_backend(torch, cap)
    _linalg_backend = configure_linalg_backend(torch)

    sd = torch.load(CKPT, weights_only=True, map_location="cpu")
    m = create_predictor(PredictorParams())
    m.load_state_dict(sd)
    m.eval().to("cuda")
    if USE_CHANNELS_LAST:
        m.to(memory_format=torch.channels_last)
    torch.backends.cudnn.benchmark = USE_CUDNN_BENCHMARK
    torch.cuda.synchronize()

    _torch = torch
    _device = torch.device("cuda")
    _model = m
    _loaded_seconds = time.perf_counter() - t0
    log_event(
        "sharp_load_done",
        elapsedSeconds=_loaded_seconds,
        torch=torch.__version__,
        cudaDevice=torch.cuda.get_device_name(0),
        cudaCapability=list(cap),
        precision=PRECISION,
        channelsLast=USE_CHANNELS_LAST,
        cudnnBenchmark=USE_CUDNN_BENCHMARK,
        fastUnproject=USE_FAST_UNPROJECT,
        fastDecomp=FAST_DECOMP_BACKEND if USE_FAST_UNPROJECT else None,
        sdpBackend=_sdp_backend,
        linalgBackend=_linalg_backend,
    )
    return _model


def autocast_dtype(torch: Any) -> Any | None:
    if PRECISION in {"fp16", "float16", "half"}:
        return torch.float16
    if PRECISION in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if PRECISION in {"fp32", "float32", "none", "off"}:
        return None
    raise ValueError(f"unsupported SHARP_PRECISION={PRECISION!r}; use fp32, fp16, or bf16")


def num_gaussians(gaussians: Any) -> int:
    mean_vectors = gaussians.mean_vectors
    if len(mean_vectors.shape) >= 2:
        return int(mean_vectors.flatten(0, -2).shape[0])
    return int(mean_vectors.shape[0])


def det3(matrix: Any) -> Any:
    return (
        matrix[..., 0, 0] * (matrix[..., 1, 1] * matrix[..., 2, 2] - matrix[..., 1, 2] * matrix[..., 2, 1])
        - matrix[..., 0, 1] * (matrix[..., 1, 0] * matrix[..., 2, 2] - matrix[..., 1, 2] * matrix[..., 2, 0])
        + matrix[..., 0, 2] * (matrix[..., 1, 0] * matrix[..., 2, 1] - matrix[..., 1, 1] * matrix[..., 2, 0])
    )


def ensure_rotation_matrix(matrix: Any) -> Any:
    sign = _torch.where(det3(matrix)[..., None] < 0, -1.0, 1.0).to(matrix.dtype)
    out = matrix.clone()
    out[..., :, 2] = out[..., :, 2] * sign
    return out


def normalize_vector(vector: Any, eps: float = 1e-12) -> Any:
    return vector / vector.norm(dim=-1, keepdim=True).clamp(min=eps)


def fallback_axis(vector: Any) -> Any:
    abs_vector = vector.abs()
    idx = abs_vector.argmin(dim=-1)
    basis = _torch.zeros_like(vector)
    basis.scatter_(-1, idx[..., None], 1.0)
    return basis


def orthogonal_vector(vector: Any) -> Any:
    axis = fallback_axis(vector)
    out = _torch.cross(vector, axis, dim=-1)
    return normalize_vector(out)


def eigenvector_for_value(matrix: Any, eigenvalue: Any) -> Any:
    eye = _torch.eye(3, device=matrix.device, dtype=matrix.dtype)
    shifted = matrix - eigenvalue[..., None, None] * eye
    row0 = shifted[..., 0, :]
    row1 = shifted[..., 1, :]
    row2 = shifted[..., 2, :]
    candidates = _torch.stack(
        [
            _torch.cross(row0, row1, dim=-1),
            _torch.cross(row0, row2, dim=-1),
            _torch.cross(row1, row2, dim=-1),
        ],
        dim=-2,
    )
    norms = (candidates * candidates).sum(dim=-1)
    idx = norms.argmax(dim=-1)
    vector = candidates.gather(-2, idx[..., None, None].expand(idx.shape + (1, 3))).squeeze(-2)
    fallback = fallback_axis(vector)
    vector = _torch.where(norms.max(dim=-1).values[..., None] > 1e-20, vector, fallback)
    return normalize_vector(vector)


def symmetric_eigh_3x3(matrix: Any) -> tuple[Any, Any]:
    # Closed-form eigensolver for symmetric 3x3 matrices. This avoids cuSolver's
    # batched eigh path, which is broken for this workload on sm_120/CUDA 12.8.
    matrix = (matrix + matrix.transpose(-1, -2)) * 0.5
    a00 = matrix[..., 0, 0]
    a01 = matrix[..., 0, 1]
    a02 = matrix[..., 0, 2]
    a11 = matrix[..., 1, 1]
    a12 = matrix[..., 1, 2]
    a22 = matrix[..., 2, 2]

    q = (a00 + a11 + a22) / 3.0
    p1 = a01 * a01 + a02 * a02 + a12 * a12
    p2 = (a00 - q) ** 2 + (a11 - q) ** 2 + (a22 - q) ** 2 + 2.0 * p1
    p = _torch.sqrt(_torch.clamp(p2 / 6.0, min=1e-20))
    eye = _torch.eye(3, device=matrix.device, dtype=matrix.dtype)
    b = (matrix - q[..., None, None] * eye) / p[..., None, None]
    r = _torch.clamp(det3(b) * 0.5, min=-1.0, max=1.0)
    phi = _torch.acos(r) / 3.0
    two_p = 2.0 * p
    eig0 = q + two_p * _torch.cos(phi)
    eig2 = q + two_p * _torch.cos(phi + (2.0 * _torch.pi / 3.0))
    eig1 = 3.0 * q - eig0 - eig2
    analytic_values = _torch.stack([eig0, eig1, eig2], dim=-1)
    diagonal_values = _torch.stack([a00, a11, a22], dim=-1).sort(dim=-1, descending=True).values
    values = _torch.where((p2 > 1e-20)[..., None], analytic_values, diagonal_values)
    values = values.sort(dim=-1, descending=True).values

    vec0 = eigenvector_for_value(matrix, values[..., 0])
    vec1 = eigenvector_for_value(matrix, values[..., 1])
    vec1 = vec1 - (vec1 * vec0).sum(dim=-1, keepdim=True) * vec0
    vec1 = _torch.where((vec1.norm(dim=-1, keepdim=True) > 1e-10), vec1, orthogonal_vector(vec0))
    vec1 = normalize_vector(vec1)
    vec2 = normalize_vector(_torch.cross(vec0, vec1, dim=-1))
    rotations = _torch.stack([vec0, vec1, vec2], dim=-1)
    return values, rotations


def rotation_matrices_to_quaternions(matrix: Any) -> Any:
    import torch.nn.functional as F

    m00 = matrix[..., 0, 0]
    m01 = matrix[..., 0, 1]
    m02 = matrix[..., 0, 2]
    m10 = matrix[..., 1, 0]
    m11 = matrix[..., 1, 1]
    m12 = matrix[..., 1, 2]
    m20 = matrix[..., 2, 0]
    m21 = matrix[..., 2, 1]
    m22 = matrix[..., 2, 2]

    q_abs = _torch.sqrt(
        _torch.clamp(
            _torch.stack(
                [
                    1.0 + m00 + m11 + m22,
                    1.0 + m00 - m11 - m22,
                    1.0 - m00 + m11 - m22,
                    1.0 - m00 - m11 + m22,
                ],
                dim=-1,
            ),
            min=0.0,
        )
    )
    quat_by_rijk = _torch.stack(
        [
            _torch.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], dim=-1),
            _torch.stack([m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], dim=-1),
            _torch.stack([m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], dim=-1),
            _torch.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], dim=-1),
        ],
        dim=-2,
    )
    quat_candidates = quat_by_rijk / (2.0 * q_abs[..., None].clamp(min=0.1))
    idx = q_abs.argmax(dim=-1)
    quaternions = quat_candidates.gather(
        -2, idx[..., None, None].expand(idx.shape + (1, 4))
    ).squeeze(-2)
    return F.normalize(quaternions, p=2, dim=-1)


def fast_unproject_gaussians(gaussians_ndc: Any, extrinsics: Any, intrinsics: Any, image_shape: tuple[int, int]) -> Any:
    from sharp.utils.gaussians import Gaussians3D, compose_covariance_matrices, get_unprojection_matrix

    if _torch is None:
        raise RuntimeError("SHARP model was not initialized correctly")

    transform = get_unprojection_matrix(extrinsics, intrinsics, image_shape)[:3]
    transform_linear = transform[..., :3, :3]
    transform_offset = transform[..., :3, 3]

    mean_vectors = gaussians_ndc.mean_vectors @ transform_linear.T + transform_offset
    covariance_matrices = compose_covariance_matrices(
        gaussians_ndc.quaternions.float(), gaussians_ndc.singular_values.float()
    )
    covariance_matrices = (
        transform_linear @ covariance_matrices @ transform_linear.transpose(-1, -2)
    )
    covariance_matrices = (covariance_matrices + covariance_matrices.transpose(-1, -2)) * 0.5

    if FAST_DECOMP_BACKEND in {"analytic", "closed_form", "closed-form"}:
        eigenvalues, rotations = symmetric_eigh_3x3(covariance_matrices.float())
    elif FAST_DECOMP_BACKEND == "eigh":
        eigenvalues, rotations = _torch.linalg.eigh(covariance_matrices.float())
        eigenvalues = eigenvalues.flip(-1)
        rotations = rotations.flip(-1)
    else:
        raise ValueError(f"unsupported SHARP_FAST_DECOMP={FAST_DECOMP_BACKEND!r}")
    eigenvalues = _torch.clamp(eigenvalues, min=0.0)
    rotations = ensure_rotation_matrix(rotations)

    quaternions = rotation_matrices_to_quaternions(rotations).to(dtype=gaussians_ndc.quaternions.dtype)
    singular_values = _torch.sqrt(eigenvalues).to(dtype=gaussians_ndc.singular_values.dtype)

    return Gaussians3D(
        mean_vectors=mean_vectors,
        singular_values=singular_values,
        quaternions=quaternions,
        colors=gaussians_ndc.colors,
        opacities=gaussians_ndc.opacities,
    )


def predict_gaussians(image_path: Path) -> tuple[Any, float, tuple[int, int], dict[str, float]]:
    import torch.nn.functional as F
    from sharp.utils import io
    from sharp.utils.gaussians import unproject_gaussians

    model = load_model()
    if _torch is None or _device is None:
        raise RuntimeError("SHARP model was not initialized correctly")

    steps: dict[str, float] = {}
    step_start = time.perf_counter()

    def mark(step: str) -> None:
        nonlocal step_start
        if PROFILE_STEPS:
            _torch.cuda.synchronize()
        now = time.perf_counter()
        steps[step] = now - step_start
        step_start = now

    image, _, f_px = io.load_rgb(image_path)
    mark("io_load_rgb")

    with _torch.inference_mode():
        image_pt = _torch.from_numpy(image.copy()).float().to(_device).permute(2, 0, 1) / 255.0
        _, height, width = image_pt.shape
        disparity_factor = _torch.tensor([f_px / width], device=_device, dtype=_torch.float32)
        mark("to_device")

        image_resized_pt = F.interpolate(
            image_pt[None],
            size=(INTERNAL_SHAPE[1], INTERNAL_SHAPE[0]),
            mode="bilinear",
            align_corners=True,
        )
        if USE_CHANNELS_LAST:
            image_resized_pt = image_resized_pt.contiguous(memory_format=_torch.channels_last)
        mark("interpolate")

        dtype = autocast_dtype(_torch)
        with _torch.autocast("cuda", dtype=dtype, enabled=dtype is not None):
            gaussians_ndc = model(image_resized_pt, disparity_factor)
        mark("forward")

        intrinsics = (
            _torch.tensor(
                [
                    [f_px, 0, width / 2, 0],
                    [0, f_px, height / 2, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1],
                ],
                device=_device,
                dtype=_torch.float32,
            )
        )
        intrinsics_resized = intrinsics.clone()
        intrinsics_resized[0] *= INTERNAL_SHAPE[0] / width
        intrinsics_resized[1] *= INTERNAL_SHAPE[1] / height
        if USE_FAST_UNPROJECT:
            gaussians = fast_unproject_gaussians(
                gaussians_ndc, _torch.eye(4, device=_device), intrinsics_resized, INTERNAL_SHAPE
            )
        else:
            gaussians = unproject_gaussians(
                gaussians_ndc, _torch.eye(4, device=_device), intrinsics_resized, INTERNAL_SHAPE
            )
        mark("unproject")

    return gaussians, f_px, (height, width), steps


def gaussians_to_splat_bytes(gaussians: Any) -> bytes:
    import numpy as np
    import torch.nn.functional as F
    from sharp.utils import color_space as cs_utils

    if _torch is None:
        raise RuntimeError("SHARP model was not initialized correctly")

    xyz = gaussians.mean_vectors.flatten(0, 1).float()
    scale = gaussians.singular_values.flatten(0, 1).float()
    colors = cs_utils.linearRGB2sRGB(gaussians.colors.flatten(0, 1).float()).clamp(0.0, 1.0)
    opacities = gaussians.opacities.flatten(0, 1).float().clamp(0.0, 1.0)
    quaternions = F.normalize(gaussians.quaternions.flatten(0, 1).float(), p=2, dim=-1)

    rgb = (colors * 255.0).round().to(_torch.uint8).detach().cpu().numpy()
    alpha = (opacities * 255.0).round().to(_torch.uint8).detach().cpu().numpy().reshape(-1, 1)
    rgba = np.concatenate([rgb, alpha], axis=1)
    rotation = ((quaternions * 128.0) + 128.0).round().clamp(0, 255).to(_torch.uint8).detach().cpu().numpy()

    row_dtype = np.dtype(
        [
            ("xyz", "<f4", (3,)),
            ("scale", "<f4", (3,)),
            ("rgba", "u1", (4,)),
            ("rotation", "u1", (4,)),
        ]
    )
    rows = np.empty(xyz.shape[0], dtype=row_dtype)
    rows["xyz"] = xyz.detach().cpu().numpy().astype("<f4", copy=False)
    rows["scale"] = scale.detach().cpu().numpy().astype("<f4", copy=False)
    rows["rgba"] = rgba
    rows["rotation"] = rotation
    return rows.tobytes()


def run_inference(image_path: Path, out_path: Path, output_format: str) -> dict[str, Any]:
    from sharp.utils.gaussians import save_ply

    out_path.parent.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    gaussians, f_px, image_shape, steps = predict_gaussians(image_path)
    count = num_gaussians(gaussians)

    write_start = time.perf_counter()
    if output_format == "splat":
        out_path.write_bytes(gaussians_to_splat_bytes(gaussians))
        output_key = "splat_path"
    elif output_format == "ply":
        save_ply(gaussians, f_px, image_shape, out_path)
        output_key = "ply_path"
    else:
        raise ValueError(f"unsupported output_format: {output_format}")
    if _torch is not None:
        _torch.cuda.synchronize()
    steps[f"write_{output_format}"] = time.perf_counter() - write_start
    seconds = time.perf_counter() - start
    image_height, image_width = image_shape
    camera = {
        "sourceWidth": int(image_width),
        "sourceHeight": int(image_height),
        "camFx": float(f_px),
        "camFy": float(f_px),
        "camCx": float(image_width * 0.5),
        "camCy": float(image_height * 0.5),
    }

    log_event(
        "sharp_infer_done",
        imagePath=str(image_path),
        outputPath=str(out_path),
        outputFormat=output_format,
        numGaussians=count,
        elapsedSeconds=seconds,
        precision=PRECISION,
        fastUnproject=USE_FAST_UNPROJECT,
        steps=steps,
    )
    return {
        output_key: str(out_path),
        "output_path": str(out_path),
        "output_format": output_format,
        "num_gaussians": count,
        "seconds": seconds,
        "precision": PRECISION,
        "fast_unproject": USE_FAST_UNPROJECT,
        "splat_row_bytes": SPLAT_ROW_BYTES if output_format == "splat" else None,
        "steps": steps,
        **camera,
    }


def run_inference_splat_bytes(image_path: Path) -> tuple[bytes, dict[str, Any]]:
    start = time.perf_counter()
    gaussians, f_px, image_shape, steps = predict_gaussians(image_path)
    count = num_gaussians(gaussians)

    encode_start = time.perf_counter()
    splat = gaussians_to_splat_bytes(gaussians)
    if _torch is not None:
        _torch.cuda.synchronize()
    steps["encode_splat"] = time.perf_counter() - encode_start
    seconds = time.perf_counter() - start
    image_height, image_width = image_shape
    camera = {
        "sourceWidth": int(image_width),
        "sourceHeight": int(image_height),
        "camFx": float(f_px),
        "camFy": float(f_px),
        "camCx": float(image_width * 0.5),
        "camCy": float(image_height * 0.5),
    }

    log_event(
        "sharp_infer_upload_done",
        imagePath=str(image_path),
        outputFormat="splat",
        outputPath=None,
        numGaussians=count,
        elapsedSeconds=seconds,
        precision=PRECISION,
        fastUnproject=USE_FAST_UNPROJECT,
        steps=steps,
    )
    return splat, {
        "splat_path": None,
        "output_path": None,
        "output_format": "splat",
        "num_gaussians": count,
        "seconds": seconds,
        "precision": PRECISION,
        "fast_unproject": USE_FAST_UNPROJECT,
        "splat_row_bytes": SPLAT_ROW_BYTES,
        "steps": steps,
        "streamed": True,
        **camera,
    }


def clear_cuda_cache() -> None:
    gc.collect()
    if _torch is None or not hasattr(_torch, "cuda") or not _torch.cuda.is_available():
        return
    _torch.cuda.empty_cache()
    if hasattr(_torch.cuda, "ipc_collect"):
        try:
            _torch.cuda.ipc_collect()
        except Exception:
            pass


def parse_json_body(raw: bytes) -> dict[str, str]:
    payload = json.loads(raw.decode("utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("JSON body must be an object")
    return {str(key): str(value) for key, value in payload.items()}


def parse_multipart_body(content_type: str, raw: bytes) -> dict[str, str]:
    message = BytesParser(policy=policy.default).parsebytes(
        b"Content-Type: " + content_type.encode("utf-8") + b"\r\nMIME-Version: 1.0\r\n\r\n" + raw
    )
    fields: dict[str, str] = {}
    temp_dir = Path(tempfile.mkdtemp(prefix="sharp-upload-"))
    fields["__temp_dir"] = str(temp_dir)
    for part in message.iter_parts():
        name = part.get_param("name", header="content-disposition")
        if not name:
            continue
        filename = part.get_filename()
        payload = part.get_payload(decode=True) or b""
        if filename:
            safe_name = Path(filename).name or "upload.jpg"
            upload_path = temp_dir / safe_name
            upload_path.write_bytes(payload)
            if name in {"image", "file", "upload"} and "image_path" not in fields:
                fields["image_path"] = str(upload_path)
            fields[name] = str(upload_path)
        else:
            fields[name] = payload.decode(part.get_content_charset() or "utf-8").strip()
    return fields


def parse_infer_request(handler: BaseHTTPRequestHandler) -> tuple[Path, Path, str, Path | None]:
    length = int(handler.headers.get("Content-Length", "0"))
    if length <= 0:
        raise ValueError("request body is empty")
    if length > MAX_BODY_BYTES:
        raise ValueError(f"request body is too large: {length} bytes")

    content_type = handler.headers.get("Content-Type", "application/json").split(";", 1)[0].lower()
    raw = handler.rfile.read(length)
    if content_type == "multipart/form-data":
        fields = parse_multipart_body(handler.headers.get("Content-Type", ""), raw)
    else:
        fields = parse_json_body(raw)
    temp_dir_value = fields.pop("__temp_dir", "")
    temp_dir = Path(temp_dir_value) if temp_dir_value else None

    image_path_value = fields.get("image_path")
    if not image_path_value:
        raise ValueError("missing image_path")

    output_format_value = fields.get("output_format", "").strip().lower()
    if output_format_value and output_format_value not in {"ply", "splat"}:
        raise ValueError("output_format must be 'ply' or 'splat'")

    out_splat_value = fields.get("out_splat")
    out_ply_value = fields.get("out_ply")

    if out_splat_value:
        out_path_value = out_splat_value
        output_format = "splat"
    elif out_ply_value:
        out_path_value = out_ply_value
        output_format = output_format_value or ("splat" if Path(out_ply_value).suffix == ".splat" else "ply")
    else:
        stamp = time.strftime("%Y%m%d-%H%M%S")
        output_format = output_format_value or "splat"
        out_path_value = f"/tmp/sharp_endpoint/{Path(image_path_value).stem}-{stamp}.{output_format}"

    return Path(image_path_value), Path(out_path_value), output_format, temp_dir


def cleanup_upload_temp(temp_dir: Path | None) -> None:
    if not temp_dir:
        return
    try:
        if temp_dir.parent == Path(tempfile.gettempdir()) and temp_dir.name.startswith("sharp-upload-"):
            shutil.rmtree(temp_dir, ignore_errors=True)
    except Exception as exc:
        log_event("sharp_upload_cleanup_error", tempDir=str(temp_dir), error=repr(exc))


class Handler(BaseHTTPRequestHandler):
    server_version = "WiCiSharpEndpoint/1.0"

    def log_message(self, fmt: str, *args: Any) -> None:
        print(fmt % args, flush=True)

    def do_OPTIONS(self) -> None:
        json_response(self, 204, {})

    def do_GET(self) -> None:
        path = self.path.split("?", 1)[0]
        if path != "/health":
            json_response(self, 404, {"ok": False, "error": "not_found"})
            return
        json_response(
            self,
            200,
            {
                "ok": True,
                "loaded": _model is not None,
                "checkpoint": str(CKPT),
                "load_seconds": _loaded_seconds,
                "precision": PRECISION,
                "profile_steps": PROFILE_STEPS,
                "channels_last": USE_CHANNELS_LAST,
                "cudnn_benchmark": USE_CUDNN_BENCHMARK,
                "fast_unproject": USE_FAST_UNPROJECT,
                "fast_decomp": FAST_DECOMP_BACKEND if USE_FAST_UNPROJECT else None,
                "sdp_backend": _sdp_backend,
                "linalg_backend": _linalg_backend,
            },
        )

    def do_POST(self) -> None:
        path = self.path.split("?", 1)[0]
        if path not in {"/infer", "/infer_upload"}:
            json_response(self, 404, {"ok": False, "error": "not_found"})
            return

        temp_dir: Path | None = None
        try:
            request_start = time.perf_counter()
            image_path, out_path, output_format, temp_dir = parse_infer_request(self)
            parse_seconds = time.perf_counter() - request_start
            if not image_path.exists():
                raise FileNotFoundError(f"image_path does not exist: {image_path}")

            if path == "/infer_upload":
                with _lock:
                    splat, result = run_inference_splat_bytes(image_path)
                splat_response(
                    self,
                    200,
                    splat,
                    result,
                    {
                        "parse": parse_seconds,
                        "total": time.perf_counter() - request_start,
                    },
                )
            else:
                with _lock:
                    result = run_inference(image_path, out_path, output_format)
                json_response(self, 200, result)
        except Exception as exc:
            log_event("sharp_infer_error", error=repr(exc))
            json_response(self, 500, {"ok": False, "error": type(exc).__name__, "message": str(exc)})
        finally:
            clear_cuda_cache()
            if path == "/infer_upload":
                cleanup_upload_temp(temp_dir)


class Server(ThreadingHTTPServer):
    daemon_threads = True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8770)
    args = parser.parse_args()

    load_model()
    server = Server((args.host, args.port), Handler)
    log_event("sharp_endpoint_listening", host=args.host, port=args.port)
    try:
        server.serve_forever()
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
