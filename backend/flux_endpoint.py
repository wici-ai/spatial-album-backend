#!/usr/bin/env python3
"""Resident FLUX.1-Fill endpoint for form-1 peripheral generation."""
from __future__ import annotations

import argparse
import base64
import gc
import io
import json
import math
import os
import subprocess
import sys
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image


VIEWER_ROOT = Path(__file__).resolve().parent
MODEL_ROOT = Path(os.environ.get("WICI_MODEL_ROOT", "/models"))
DEFAULT_OUT_DIR = MODEL_ROOT / "flux/flux_peripheral"
OUT_DIR = Path(os.environ.get("WICI_FLUX_OUT_DIR", str(DEFAULT_OUT_DIR)))
CONFIG_DIR = Path(os.environ.get("WICI_FLUX_CONFIG_DIR", str(OUT_DIR / "hf_flux1_fill_dev_config")))
PROMPT_EMBEDS = Path(os.environ.get("WICI_FLUX_PROMPT_EMBEDS", str(OUT_DIR / "prompt_embeds.pt")))
TRANSFORMER_CKPT = Path(
    os.environ.get("WICI_FLUX_TRANSFORMER", "/models/ComfyUI/models/diffusion_models/flux1-fill-dev.safetensors")
)
VAE_CKPT = Path(os.environ.get("WICI_FLUX_VAE", "/models/ComfyUI/models/vae/ae.safetensors"))
TOKEN_FILE = Path(os.environ.get("WICI_HF_TOKEN_FILE", "/models/hf-token"))
PROMPT = "fill in the missing background to match the surrounding scene, with consistent texture, color and lighting."
QUANT_MODE = os.environ.get("WICI_FLUX_QUANT_MODE", "bnb8").strip().lower()
NUM_STEPS = int(os.environ.get("WICI_FLUX_NUM_STEPS", "20"))
GUIDANCE = float(os.environ.get("WICI_FLUX_GUIDANCE", "30.0"))
SEED = int(os.environ.get("WICI_FLUX_SEED", "0"))
MAX_BODY_BYTES = int(os.environ.get("WICI_FLUX_MAX_BODY_MB", "64")) * 1024 * 1024
DEFAULT_DILATE_PX = int(os.environ.get("WICI_FLUX_MASK_DILATE_PX", "24"))
DEFAULT_FEATHER_PX = int(os.environ.get("WICI_FLUX_FEATHER_PX", "24"))
DEFAULT_POISSON = os.environ.get("WICI_FLUX_POISSON", "false").strip().lower() in {"1", "true", "yes", "on"}
TURBO_LORA_ENABLED = os.environ.get("WICI_FLUX_TURBO_LORA", "false").strip().lower() in {"1", "true", "yes", "on"}
TURBO_LORA_PATH = Path(
    os.environ.get("WICI_FLUX_TURBO_LORA_PATH", "/models/ComfyUI/models/loras/FLUX.1-Turbo-Alpha.safetensors")
)
TURBO_LORA_SCALE = float(os.environ.get("WICI_FLUX_TURBO_LORA_SCALE", "1.0"))

_lock = threading.Lock()
_pipe: Any | None = None
_prompt_embeds: torch.Tensor | None = None
_pooled_prompt_embeds: torch.Tensor | None = None
_load_meta: dict[str, Any] = {}
_last_fill: dict[str, Any] | None = None
_peak_vram_mib: int = 0
_cv2_missing_logged = False


def env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    lowered = value.strip().lower()
    if lowered in {"1", "true", "yes", "on"}:
        return True
    if lowered in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean, got {value!r}")


COMPILE_ENABLED = env_bool("WICI_FLUX_COMPILE", False)
COMPILE_MODE = os.environ.get("WICI_FLUX_COMPILE_MODE", "default").strip().lower()
COMPILE_DYNAMIC = env_bool("WICI_FLUX_COMPILE_DYNAMIC", True)
BNB8_BF16_IO = env_bool("WICI_FLUX_BNB8_BF16_IO", False)

if QUANT_MODE not in {"bnb8", "fp8"}:
    raise ValueError(f"WICI_FLUX_QUANT_MODE must be bnb8 or fp8, got {QUANT_MODE!r}")
if COMPILE_MODE not in {"default", "reduce-overhead", "max-autotune"}:
    raise ValueError(
        "WICI_FLUX_COMPILE_MODE must be default, reduce-overhead, or max-autotune, "
        f"got {COMPILE_MODE!r}"
    )


def log_event(event: str, **fields: Any) -> None:
    print(json.dumps({"event": event, **fields}, separators=(",", ":")), flush=True)


def smi_memory() -> dict[str, Any]:
    try:
        raw = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
            text=True,
        ).strip()
        used_s, total_s = [part.strip() for part in raw.split(",", 1)]
        return {"usedMiB": int(used_s), "totalMiB": int(total_s), "raw": raw}
    except Exception as exc:  # noqa: BLE001 - health should survive nvidia-smi failures.
        return {"error": repr(exc)}


def smi_processes() -> list[dict[str, Any]]:
    try:
        raw = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader,nounits"],
            text=True,
        ).strip()
    except Exception:
        return []
    rows = []
    for line in raw.splitlines():
        parts = [part.strip() for part in line.split(",", 2)]
        if len(parts) == 3:
            rows.append({"pid": parts[0], "process": parts[1], "usedMiB": parts[2]})
    return rows


def update_peak_vram() -> None:
    global _peak_vram_mib
    mem = smi_memory()
    used = int(mem.get("usedMiB") or 0)
    _peak_vram_mib = max(_peak_vram_mib, used)


def clear_cuda_cache() -> None:
    gc.collect()
    torch.cuda.empty_cache()
    if hasattr(torch.cuda, "ipc_collect"):
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass


class LoraLinearWrapper(torch.nn.Module):
    def __init__(self, base: torch.nn.Module, lora_a: torch.Tensor, lora_b: torch.Tensor, scale: float) -> None:
        super().__init__()
        self.base = base
        self.register_buffer("lora_A", lora_a.contiguous(), persistent=False)
        self.register_buffer("lora_B", lora_b.contiguous(), persistent=False)
        self.scale = float(scale)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        base_out = self.base(hidden_states)
        lora_hidden = torch.nn.functional.linear(hidden_states.to(dtype=self.lora_A.dtype), self.lora_A)
        lora_out = torch.nn.functional.linear(lora_hidden, self.lora_B)
        if self.scale != 1.0:
            lora_out = lora_out * self.scale
        return base_out + lora_out.to(dtype=base_out.dtype)


def replace_submodule(root: torch.nn.Module, module_name: str, replacement: torch.nn.Module) -> None:
    if "." in module_name:
        parent_name, child_name = module_name.rsplit(".", 1)
        parent = root.get_submodule(parent_name)
    else:
        parent = root
        child_name = module_name
    setattr(parent, child_name, replacement)


def apply_filtered_turbo_lora(transformer: torch.nn.Module) -> dict[str, Any]:
    status: dict[str, Any] = {
        "enabled": TURBO_LORA_ENABLED,
        "loaded": False,
        "path": str(TURBO_LORA_PATH),
        "scale": TURBO_LORA_SCALE,
        "applied": 0,
        "skippedMissing": [],
        "skippedIncomplete": [],
        "skippedShape": [],
    }
    if not TURBO_LORA_ENABLED:
        return status
    if not TURBO_LORA_PATH.exists():
        raise FileNotFoundError(f"Turbo LoRA not found: {TURBO_LORA_PATH}")

    from safetensors.torch import load_file

    raw = load_file(str(TURBO_LORA_PATH), device="cpu")
    pairs: dict[str, dict[str, torch.Tensor]] = {}
    for key, tensor in raw.items():
        if not key.startswith("transformer.") or ".lora_" not in key or not key.endswith(".weight"):
            continue
        if key.endswith(".lora_A.weight"):
            module_name = key.removeprefix("transformer.").removesuffix(".lora_A.weight")
            pairs.setdefault(module_name, {})["A"] = tensor
        elif key.endswith(".lora_B.weight"):
            module_name = key.removeprefix("transformer.").removesuffix(".lora_B.weight")
            pairs.setdefault(module_name, {})["B"] = tensor

    status["pairsTotal"] = len(pairs)
    for module_name, pair in sorted(pairs.items()):
        if "A" not in pair or "B" not in pair:
            status["skippedIncomplete"].append(module_name)
            continue
        try:
            module = transformer.get_submodule(module_name)
        except AttributeError:
            status["skippedMissing"].append(module_name)
            continue
        weight = getattr(module, "weight", None)
        if weight is None:
            status["skippedMissing"].append(module_name)
            continue
        lora_a = pair["A"]
        lora_b = pair["B"]
        expected = (int(lora_b.shape[0]), int(lora_a.shape[1]))
        actual = tuple(int(dim) for dim in weight.shape)
        rank_ok = len(lora_a.shape) == 2 and len(lora_b.shape) == 2 and lora_a.shape[0] == lora_b.shape[1]
        if not rank_ok or actual != expected:
            status["skippedShape"].append(
                {
                    "module": module_name,
                    "weight": list(actual),
                    "loraA": list(lora_a.shape),
                    "loraB": list(lora_b.shape),
                }
            )
            continue
        device = weight.device
        wrapper = LoraLinearWrapper(
            module,
            lora_a.to(device=device, dtype=torch.float16),
            lora_b.to(device=device, dtype=torch.float16),
            TURBO_LORA_SCALE,
        )
        replace_submodule(transformer, module_name, wrapper)
        status["applied"] += 1
    del raw
    gc.collect()
    status["loaded"] = status["applied"] > 0
    log_event("flux_turbo_lora_loaded", **status)
    return status


def quantize_transformer_fp8(transformer: torch.nn.Module) -> dict[str, Any]:
    from torchao.quantization import Float8DynamicActivationFloat8WeightConfig, quantize_
    from torchao.quantization.granularity import PerRow

    status: dict[str, Any] = {
        "mode": "fp8",
        "library": "torchao",
        "config": "Float8DynamicActivationFloat8WeightConfig",
        "granularity": "PerRow",
        "weightDtype": "float8_e4m3fn",
        "activationDtype": "float8_e4m3fn",
        "realFp8ComputeProbe": False,
    }
    quant_start = time.perf_counter()
    quantize_(transformer, Float8DynamicActivationFloat8WeightConfig(granularity=PerRow()))
    status["quantizeSeconds"] = time.perf_counter() - quant_start
    status["float8WeightCount"] = sum(1 for module in transformer.modules() if module.__class__.__name__ == "Linear" and module.weight.__class__.__name__ == "Float8Tensor")

    probe = torch.nn.Linear(128, 128, bias=False).cuda().eval().bfloat16()
    quantize_(probe, Float8DynamicActivationFloat8WeightConfig(granularity=PerRow()))
    x = torch.randn(256, 128, device="cuda", dtype=torch.bfloat16)
    torch.cuda.synchronize()
    _ = probe(x)
    torch.cuda.synchronize()
    status["realFp8ComputeProbe"] = probe.weight.__class__.__name__ == "Float8Tensor"
    del probe, x
    return status


def maybe_compile_transformer(pipe: Any) -> dict[str, Any]:
    status: dict[str, Any] = {
        "enabled": COMPILE_ENABLED,
        "compiled": False,
        "mode": COMPILE_MODE,
        "dynamic": COMPILE_DYNAMIC,
        "backend": "inductor",
    }
    if not COMPILE_ENABLED:
        return status
    if not hasattr(torch, "compile"):
        raise RuntimeError("WICI_FLUX_COMPILE is enabled but torch.compile is unavailable")

    compile_kwargs: dict[str, Any] = {"dynamic": COMPILE_DYNAMIC}
    if COMPILE_MODE != "default":
        compile_kwargs["mode"] = COMPILE_MODE
    compile_start = time.perf_counter()
    try:
        pipe.transformer = torch.compile(pipe.transformer, **compile_kwargs)
    except Exception as exc:  # noqa: BLE001 - startup should report the exact blocker.
        status["error"] = repr(exc)
        log_event("flux_compile_transformer_failed", **status)
        raise RuntimeError(f"torch.compile failed while wrapping transformer: {exc!r}") from exc
    status["compiled"] = True
    status["wrapSeconds"] = time.perf_counter() - compile_start
    log_event("flux_compile_transformer_wrapped", **status)
    return status


def json_response(handler: BaseHTTPRequestHandler, status: int, payload: dict[str, Any]) -> None:
    raw = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(raw)))
    handler.send_header("Access-Control-Allow-Origin", "*")
    handler.send_header("Access-Control-Allow-Methods", "GET,POST,OPTIONS")
    handler.send_header("Access-Control-Allow-Headers", "Content-Type, Cache-Control, Pragma")
    handler.send_header("Cross-Origin-Resource-Policy", "cross-origin")
    handler.end_headers()
    if status != 204:
        handler.wfile.write(raw)


def data_url_to_image(value: str, mode: str = "RGB") -> Image.Image:
    if not isinstance(value, str):
        raise ValueError("image value must be a data URL or local path string")
    if value.startswith("data:"):
        _, encoded = value.split(",", 1)
        return Image.open(io.BytesIO(base64.b64decode(encoded))).convert(mode)
    path = Path(value)
    if not path.is_absolute():
        path = VIEWER_ROOT / value
    resolved = path.resolve()
    root = VIEWER_ROOT.resolve()
    if root not in resolved.parents and resolved != root:
        raise ValueError("path image must be under viewer root")
    return Image.open(resolved).convert(mode)


def image_to_data_url(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    encoded = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def ceil16(value: int) -> int:
    return int(math.ceil(value / 16) * 16)


def parse_nonnegative_int(value: Any, default: int, name: str) -> int:
    if value is None:
        value = default
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a nonnegative integer")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a nonnegative integer") from exc
    if parsed < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return parsed


def parse_bool(value: Any, default: bool, name: str) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "on"}:
            return True
        if lowered in {"0", "false", "no", "off"}:
            return False
    raise ValueError(f"{name} must be a boolean")


def dilate_once(mask_bool: np.ndarray) -> np.ndarray:
    padded = np.pad(mask_bool.astype(bool), 1, mode="constant", constant_values=False)
    return (
        padded[1:-1, 1:-1]
        | padded[:-2, 1:-1]
        | padded[2:, 1:-1]
        | padded[1:-1, :-2]
        | padded[1:-1, 2:]
        | padded[:-2, :-2]
        | padded[:-2, 2:]
        | padded[2:, :-2]
        | padded[2:, 2:]
    )


def dilate_mask(mask_bool: np.ndarray, radius_px: int) -> np.ndarray:
    if radius_px <= 0:
        return mask_bool.copy()
    out = mask_bool.astype(bool, copy=True)
    for _ in range(radius_px):
        out = dilate_once(out)
    return out


def erode_mask(mask_bool: np.ndarray) -> np.ndarray:
    padded = np.pad(mask_bool.astype(bool), 1, mode="constant", constant_values=True)
    return (
        padded[1:-1, 1:-1]
        & padded[:-2, 1:-1]
        & padded[2:, 1:-1]
        & padded[1:-1, :-2]
        & padded[1:-1, 2:]
        & padded[:-2, :-2]
        & padded[:-2, 2:]
        & padded[2:, :-2]
        & padded[2:, 2:]
    )


def mask_feather_alpha(mask_bool: np.ndarray, feather_px: int) -> np.ndarray:
    alpha = np.zeros(mask_bool.shape, dtype=np.float32)
    if feather_px <= 0:
        alpha[mask_bool] = 1.0
        return alpha

    frontier = mask_bool.astype(bool, copy=True)
    for step in range(1, feather_px + 1):
        if not frontier.any():
            break
        eroded = erode_mask(frontier)
        layer = frontier & ~eroded
        alpha[layer] = step / feather_px
        frontier = eroded
    alpha[frontier] = 1.0
    return alpha


def pad_image_and_mask(
    image: Image.Image, mask: Image.Image, dilate_px: int = 0
) -> tuple[Image.Image, Image.Image, np.ndarray, np.ndarray]:
    base = image.convert("RGB")
    mask_l = mask.convert("L")
    if mask_l.size != base.size:
        mask_l = mask_l.resize(base.size, Image.Resampling.NEAREST)
    input_mask_bool = np.array(mask_l) > 127
    mask_bool = dilate_mask(input_mask_bool, dilate_px)
    w, h = base.size
    pw, ph = ceil16(w), ceil16(h)
    padded = Image.new("RGB", (pw, ph))
    padded.paste(base, (0, 0))
    if pw > w:
        padded.paste(base.crop((w - 1, 0, w, h)).resize((pw - w, h)), (w, 0))
    if ph > h:
        padded.paste(padded.crop((0, h - 1, pw, h)).resize((pw, ph - h)), (0, h))

    mask_np = np.zeros((ph, pw), dtype=np.uint8)
    mask_np[:h, :w] = mask_bool.astype(np.uint8) * 255
    if pw > w:
        mask_np[:h, w:pw] = 255
    if ph > h:
        mask_np[h:ph, :] = 255
    return padded, Image.fromarray(mask_np, mode="L"), mask_bool, input_mask_bool


def composite_byte_exact(base: Image.Image, generated: Image.Image, mask_bool: np.ndarray) -> tuple[Image.Image, bool]:
    w, h = base.size
    base_np = np.array(base.convert("RGB"))
    gen_np = np.array(generated.convert("RGB").crop((0, 0, w, h)))
    out_np = base_np.copy()
    out_np[mask_bool] = gen_np[mask_bool]
    exact = bool(np.array_equal(out_np[~mask_bool], base_np[~mask_bool]))
    return Image.fromarray(out_np, mode="RGB"), exact


def poisson_clone(
    base_np: np.ndarray, generated_np: np.ndarray, mask_bool: np.ndarray
) -> tuple[np.ndarray | None, bool, dict[str, Any]]:
    global _cv2_missing_logged
    try:
        import cv2  # type: ignore[import-not-found]
    except Exception as exc:  # noqa: BLE001 - cv2 is intentionally optional.
        if not _cv2_missing_logged:
            log_event("flux_poisson_unavailable", note=f"OpenCV unavailable; using feather only: {exc!r}")
            _cv2_missing_logged = True
        return None, False, {"poissonError": repr(exc)}

    if not mask_bool.any():
        return None, False, {"poissonError": "empty_mask"}
    h, w = mask_bool.shape
    touches_border = bool(mask_bool[0, :].any() or mask_bool[-1, :].any() or mask_bool[:, 0].any() or mask_bool[:, -1].any())
    y_idx, x_idx = np.nonzero(mask_bool)
    center_x = int(round(float(x_idx.mean())))
    center_y = int(round(float(y_idx.mean())))
    bbox = [int(x_idx.min()), int(y_idx.min()), int(x_idx.max() + 1), int(y_idx.max() + 1)]
    meta: dict[str, Any] = {
        "poissonMode": "padded_centroid_crop_normal_clone",
        "poissonCenter": [center_x, center_y],
        "poissonMaskBbox": bbox,
        "poissonMaskTouchesBorder": touches_border,
    }
    try:
        crop_context = 32
        crop: tuple[int, int, int, int] | None = None
        for canvas_pad in (128, 256, 512):
            base_pad = np.pad(base_np, ((canvas_pad, canvas_pad), (canvas_pad, canvas_pad), (0, 0)), mode="edge")
            generated_pad = np.pad(
                generated_np, ((canvas_pad, canvas_pad), (canvas_pad, canvas_pad), (0, 0)), mode="edge"
            )
            mask_pad = np.pad(mask_bool, ((canvas_pad, canvas_pad), (canvas_pad, canvas_pad)), mode="constant")
            y_pad, x_pad = np.nonzero(mask_pad)
            center_x_pad = int(round(float(x_pad.mean())))
            center_y_pad = int(round(float(y_pad.mean())))
            x0, x1 = int(x_pad.min()), int(x_pad.max() + 1)
            y0, y1 = int(y_pad.min()), int(y_pad.max() + 1)
            half_w = max(center_x_pad - x0, x1 - 1 - center_x_pad) + crop_context
            half_h = max(center_y_pad - y0, y1 - 1 - center_y_pad) + crop_context
            crop_x0 = center_x_pad - half_w
            crop_x1 = center_x_pad + half_w + 1
            crop_y0 = center_y_pad - half_h
            crop_y1 = center_y_pad + half_h + 1
            if crop_x0 >= 0 and crop_y0 >= 0 and crop_x1 <= generated_pad.shape[1] and crop_y1 <= generated_pad.shape[0]:
                crop = (crop_x0, crop_y0, crop_x1, crop_y1)
                break
        if crop is None:
            raise ValueError(
                f"poisson crop exceeded padded canvas after max pad: crop={(crop_x0, crop_y0, crop_x1, crop_y1)} "
                f"canvas={generated_pad.shape[1]}x{generated_pad.shape[0]}"
            )
        crop_x0, crop_y0, crop_x1, crop_y1 = crop

        src = cv2.cvtColor(generated_pad[crop_y0:crop_y1, crop_x0:crop_x1], cv2.COLOR_RGB2BGR)
        dst = cv2.cvtColor(base_pad, cv2.COLOR_RGB2BGR)
        clone_mask = mask_pad[crop_y0:crop_y1, crop_x0:crop_x1].astype(np.uint8) * 255
        cloned = cv2.seamlessClone(src, dst, clone_mask, (center_x_pad, center_y_pad), cv2.NORMAL_CLONE)
        cloned_rgb = cv2.cvtColor(cloned, cv2.COLOR_BGR2RGB)[canvas_pad : canvas_pad + h, canvas_pad : canvas_pad + w]
        meta.update(
            {
                "poissonApplied": True,
                "poissonCanvasPadPx": canvas_pad,
                "poissonCropContextPx": crop_context,
                "poissonCrop": [int(crop_x0 - canvas_pad), int(crop_y0 - canvas_pad), int(crop_x1 - canvas_pad), int(crop_y1 - canvas_pad)],
            }
        )
        return cloned_rgb, True, meta
    except Exception as exc:  # noqa: BLE001 - optional tone matching should not fail the fill.
        log_event("flux_poisson_failed", note=f"OpenCV seamlessClone failed; using feather only: {exc!r}")
        meta["poissonError"] = repr(exc)
        return None, False, meta


def composite_feathered(
    base: Image.Image,
    generated: Image.Image,
    mask_bool: np.ndarray,
    feather_px: int,
    poisson: bool,
) -> tuple[Image.Image, bool, dict[str, Any]]:
    if feather_px == 0 and not poisson:
        output, exact = composite_byte_exact(base, generated, mask_bool)
        return output, exact, {"alphaPx": int(mask_bool.sum()), "poissonApplied": False, "poisson": {}}

    w, h = base.size
    base_np = np.array(base.convert("RGB"))
    gen_np = np.array(generated.convert("RGB").crop((0, 0, w, h)))
    alpha = mask_feather_alpha(mask_bool, feather_px)
    source_np = gen_np
    poisson_applied = False
    poisson_meta: dict[str, Any] = {}
    if poisson:
        poisson_np, poisson_applied, poisson_meta = poisson_clone(base_np, gen_np, mask_bool)
        if poisson_np is not None:
            source_np = poisson_np

    alpha_3 = alpha[:, :, None]
    out_np = np.rint(base_np.astype(np.float32) * (1.0 - alpha_3) + source_np.astype(np.float32) * alpha_3)
    out_np = np.clip(out_np, 0, 255).astype(np.uint8)
    alpha_positive = alpha > 0.0
    exact = bool(np.array_equal(out_np[~alpha_positive], base_np[~alpha_positive]))
    return Image.fromarray(out_np, mode="RGB"), exact, {
        "alphaPx": int(alpha_positive.sum()),
        "poissonApplied": poisson_applied,
        "poisson": poisson_meta,
    }


def patch_bnb_prequantized() -> None:
    import diffusers.loaders.single_file_model as sfm
    import diffusers.quantizers.auto as auto

    orig_from_config = auto.DiffusersAutoQuantizer.from_config

    def from_config_not_prequantized(cls, quantization_config, **kwargs):
        kwargs.setdefault("pre_quantized", False)
        return orig_from_config(quantization_config, **kwargs)

    auto.DiffusersAutoQuantizer.from_config = classmethod(from_config_not_prequantized)
    sfm.DiffusersAutoQuantizer.from_config = classmethod(from_config_not_prequantized)


def ensure_official_config() -> None:
    required = [
        CONFIG_DIR / "model_index.json",
        CONFIG_DIR / "transformer/config.json",
        CONFIG_DIR / "scheduler/scheduler_config.json",
        CONFIG_DIR / "vae/config.json",
        CONFIG_DIR / "text_encoder/config.json",
        CONFIG_DIR / "text_encoder_2/config.json",
        CONFIG_DIR / "tokenizer/vocab.json",
        CONFIG_DIR / "tokenizer_2/tokenizer.json",
    ]
    if all(path.exists() for path in required):
        return
    token = os.environ.get("HF_TOKEN")
    if not token and TOKEN_FILE.exists():
        token = TOKEN_FILE.read_text(encoding="utf-8").strip()
    if not token:
        missing = [str(path) for path in required if not path.exists()]
        raise RuntimeError(f"official FLUX config missing and no HF token available: {missing[:3]}")
    from huggingface_hub import snapshot_download

    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    saved_env = {name: os.environ.get(name) for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_ENDPOINT")}
    for name in saved_env:
        os.environ.pop(name, None)
    try:
        snapshot_download(
            repo_id="black-forest-labs/FLUX.1-Fill-dev",
            local_dir=str(CONFIG_DIR),
            allow_patterns=["model_index.json", "**/*.json", "scheduler/*", "tokenizer*/*"],
            ignore_patterns=[
                "*.safetensors",
                "**/*.safetensors",
                "*.bin",
                "**/*.bin",
                "*.pt",
                "**/*.pt",
                "*.ckpt",
                "**/*.ckpt",
                "*.gguf",
                "**/*.gguf",
            ],
            token=token,
            local_files_only=False,
        )
    finally:
        for name, value in saved_env.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def ensure_prompt_embeds() -> None:
    if PROMPT_EMBEDS.exists():
        return
    from safetensors.torch import load_file
    from transformers import CLIPTextConfig, CLIPTextModel, CLIPTokenizer, T5Config, T5EncoderModel, T5TokenizerFast

    text_encoder = Path(os.environ.get("WICI_FLUX_CLIP_L", "/models/ComfyUI/models/text_encoders/clip_l.safetensors"))
    text_encoder_2 = Path(
        os.environ.get("WICI_FLUX_T5XXL", "/models/ComfyUI/models/text_encoders/t5xxl_fp8_e4m3fn.safetensors")
    )
    if not text_encoder.exists() or not text_encoder_2.exists():
        raise FileNotFoundError("local CLIP/T5 weights are required to create prompt cache")

    PROMPT_EMBEDS.parent.mkdir(parents=True, exist_ok=True)

    clip_tok = CLIPTokenizer.from_pretrained(str(CONFIG_DIR / "tokenizer"), local_files_only=True)
    clip_cfg = CLIPTextConfig.from_pretrained(str(CONFIG_DIR / "text_encoder"), local_files_only=True)
    old_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.float16)
    clip = CLIPTextModel(clip_cfg)
    torch.set_default_dtype(old_dtype)
    clip_sd = {k.removeprefix("text_model."): v for k, v in load_file(str(text_encoder), device="cpu").items()}
    clip.load_state_dict(clip_sd, strict=True)
    clip.eval().to("cuda")
    with torch.inference_mode():
        text_inputs = clip_tok([PROMPT], padding="max_length", max_length=77, truncation=True, return_tensors="pt")
        pooled = clip(text_inputs.input_ids.to("cuda"), output_hidden_states=False).pooler_output
        pooled = pooled.detach().to("cpu", dtype=torch.float16)
    del clip, clip_sd
    gc.collect()
    torch.cuda.empty_cache()

    t5_tok = T5TokenizerFast.from_pretrained(str(CONFIG_DIR / "tokenizer_2"), local_files_only=True)
    t5_cfg = T5Config.from_pretrained(str(CONFIG_DIR / "text_encoder_2"), local_files_only=True)
    old_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    t5 = T5EncoderModel(t5_cfg)
    torch.set_default_dtype(old_dtype)
    t5_sd = {
        k: v.to(dtype=torch.bfloat16) if v.is_floating_point() else v
        for k, v in load_file(str(text_encoder_2), device="cpu").items()
    }
    t5.load_state_dict(t5_sd, strict=True)
    del t5_sd
    gc.collect()
    t5.eval().to("cuda")
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        text_inputs = t5_tok([PROMPT], padding="max_length", max_length=512, truncation=True, return_tensors="pt")
        embeds = t5(text_inputs.input_ids.to("cuda"), output_hidden_states=False)[0]
        embeds = embeds.detach().to("cpu", dtype=torch.bfloat16)
    del t5
    gc.collect()
    torch.cuda.empty_cache()
    torch.save(
        {
            "prompt": PROMPT,
            "prompt_embeds": embeds,
            "pooled_prompt_embeds": pooled,
            "max_sequence_length": 512,
            "clip_prefix_fixed": True,
        },
        PROMPT_EMBEDS,
    )


def load_model() -> None:
    global _pipe, _prompt_embeds, _pooled_prompt_embeds, _load_meta
    if _pipe is not None:
        return
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    ensure_official_config()
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    ensure_prompt_embeds()
    config_seconds = time.perf_counter() - t0

    if QUANT_MODE == "bnb8":
        patch_bnb_prequantized()
    from diffusers import AutoencoderKL, FlowMatchEulerDiscreteScheduler, FluxFillPipeline
    from diffusers import FluxTransformer2DModel

    t = time.perf_counter()
    quant_status: dict[str, Any]
    if QUANT_MODE == "bnb8":
        from diffusers import BitsAndBytesConfig

        quant = BitsAndBytesConfig(load_in_8bit=True)
        transformer = FluxTransformer2DModel.from_single_file(
            str(TRANSFORMER_CKPT),
            config=str(CONFIG_DIR / "transformer"),
            local_files_only=True,
            token=False,
            quantization_config=quant,
            torch_dtype=torch.float16,
            device="cuda",
        )
        quant_status = {"mode": "bnb8", "library": "bitsandbytes", "loadIn8bit": True}
        prompt_dtype = torch.bfloat16 if BNB8_BF16_IO else torch.float16
    else:
        transformer = FluxTransformer2DModel.from_single_file(
            str(TRANSFORMER_CKPT),
            config=str(CONFIG_DIR / "transformer"),
            local_files_only=True,
            token=False,
            torch_dtype=torch.bfloat16,
            device="cpu",
        )
        quant_status = quantize_transformer_fp8(transformer)
        transformer.to("cuda")
        torch.cuda.synchronize()
        prompt_dtype = torch.bfloat16
    torch.cuda.synchronize()
    transformer_seconds = time.perf_counter() - t
    update_peak_vram()

    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        str(CONFIG_DIR), subfolder="scheduler", local_files_only=True
    )
    vae_dtype = torch.bfloat16 if QUANT_MODE == "fp8" or (QUANT_MODE == "bnb8" and BNB8_BF16_IO) else torch.float16
    vae = AutoencoderKL.from_single_file(
        str(VAE_CKPT),
        config=str(CONFIG_DIR / "vae"),
        local_files_only=True,
        token=False,
        torch_dtype=vae_dtype,
    ).to("cuda")
    _pipe = FluxFillPipeline(
        scheduler=scheduler,
        vae=vae,
        text_encoder=None,
        tokenizer=None,
        text_encoder_2=None,
        tokenizer_2=None,
        transformer=transformer,
    )
    lora_status = apply_filtered_turbo_lora(transformer)
    linear8_count = sum(1 for m in transformer.modules() if m.__class__.__name__ == "Linear8bitLt")
    float8_weight_count = sum(
        1 for module in transformer.modules() if getattr(module, "weight", None).__class__.__name__ == "Float8Tensor"
    )
    compile_status = maybe_compile_transformer(_pipe)
    cache = torch.load(PROMPT_EMBEDS, map_location="cpu")
    _prompt_embeds = cache["prompt_embeds"].to(device="cuda", dtype=prompt_dtype)
    _pooled_prompt_embeds = cache["pooled_prompt_embeds"].to(device="cuda", dtype=prompt_dtype)
    _pipe.set_progress_bar_config(disable=True)
    update_peak_vram()
    _load_meta = {
        "configSeconds": config_seconds,
        "transformerLoadSeconds": transformer_seconds,
        "quant": quant_status,
        "steps": NUM_STEPS,
        "guidance": GUIDANCE,
        "prompt": PROMPT,
        "officialConfig": str(CONFIG_DIR / "transformer/config.json"),
        "linear8Count": linear8_count,
        "float8WeightCount": float8_weight_count,
        "promptDtype": str(prompt_dtype),
        "vaeDtype": str(vae_dtype),
        "bnb8Bf16Io": BNB8_BF16_IO,
        "turboLora": lora_status,
        "compile": compile_status,
        "residentVram": smi_memory(),
    }
    log_event("flux_load_done", **_load_meta)


def run_fill(
    image: Image.Image,
    mask: Image.Image,
    photo_id: Any = None,
    dilate_px: int = DEFAULT_DILATE_PX,
    feather_px: int = DEFAULT_FEATHER_PX,
    poisson: bool = DEFAULT_POISSON,
    input_decode_seconds: float = 0.0,
) -> tuple[Image.Image, dict[str, Any]]:
    if _pipe is None or _prompt_embeds is None or _pooled_prompt_embeds is None:
        raise RuntimeError("FLUX pipeline is not loaded")
    timing: dict[str, Any] = {}
    pad_start = time.perf_counter()
    base = image.convert("RGB")
    padded, padded_mask, mask_bool, input_mask_bool = pad_image_and_mask(base, mask, dilate_px)
    pad_seconds = time.perf_counter() - pad_start
    torch.cuda.reset_peak_memory_stats()
    start_vram = smi_memory()
    update_peak_vram()
    generator = torch.Generator(device="cpu").manual_seed(SEED)
    step_end_times: list[float] = []

    def record_step_end(_pipe_obj: Any, _step: int, _timestep: Any, callback_kwargs: dict[str, Any]) -> dict[str, Any]:
        torch.cuda.synchronize()
        step_end_times.append(time.perf_counter())
        return callback_kwargs

    t0 = time.perf_counter()
    with torch.inference_mode():
        generated = _pipe(
            prompt_embeds=_prompt_embeds,
            pooled_prompt_embeds=_pooled_prompt_embeds,
            image=padded,
            mask_image=padded_mask,
            height=padded.height,
            width=padded.width,
            strength=1.0,
            num_inference_steps=NUM_STEPS,
            guidance_scale=GUIDANCE,
            generator=generator,
            max_sequence_length=512,
            callback_on_step_end=record_step_end,
            callback_on_step_end_tensor_inputs=[],
            output_type="pil",
        ).images[0]
    torch.cuda.synchronize()
    pipe_end = time.perf_counter()
    seconds = pipe_end - t0
    update_peak_vram()
    step_count = len(step_end_times)
    if step_count >= 2:
        step_end_span_seconds = step_end_times[-1] - step_end_times[0]
        step_avg_seconds = step_end_span_seconds / (step_count - 1)
        t_pre_seconds = max((step_end_times[0] - t0) - step_avg_seconds, 0.0)
        denoise_seconds = step_avg_seconds * step_count
        t_post_seconds = pipe_end - step_end_times[-1]
    elif step_count == 1:
        step_end_span_seconds = 0.0
        step_avg_seconds = seconds
        t_pre_seconds = 0.0
        denoise_seconds = step_end_times[0] - t0
        t_post_seconds = pipe_end - step_end_times[0]
    else:
        step_end_span_seconds = 0.0
        step_avg_seconds = 0.0
        t_pre_seconds = 0.0
        denoise_seconds = 0.0
        t_post_seconds = seconds
    timing["inputDecodeMs"] = round(input_decode_seconds * 1000, 1)
    timing["padMs"] = round(pad_seconds * 1000, 1)
    timing["inputDecodePlusPadMs"] = round((input_decode_seconds + pad_seconds) * 1000, 1)
    timing["pipeMs"] = round(seconds * 1000, 1)
    timing["pipePreMs"] = round(t_pre_seconds * 1000, 1)
    timing["pipeFirstStepEndMs"] = round((step_end_times[0] - t0) * 1000, 1) if step_end_times else None
    timing["pipeLoopStepEndSpanMs"] = round(step_end_span_seconds * 1000, 1)
    timing["pipeDenoiseMs"] = round(denoise_seconds * 1000, 1)
    timing["pipeStepAvgMs"] = round(step_avg_seconds * 1000, 1)
    timing["pipePostMs"] = round(t_post_seconds * 1000, 1)
    timing["pipeStepCount"] = step_count
    composite_start = time.perf_counter()
    output, exact, composite_meta = composite_feathered(base, generated, mask_bool, feather_px, poisson)
    timing["compositeMs"] = round((time.perf_counter() - composite_start) * 1000, 1)
    meta = {
        "photoId": photo_id,
        "seconds": seconds,
        "steps": NUM_STEPS,
        "guidance": GUIDANCE,
        "quant": _load_meta.get("quant"),
        "turboLora": _load_meta.get("turboLora"),
        "compile": _load_meta.get("compile"),
        "size": list(base.size),
        "paddedSize": [padded.width, padded.height],
        "dilatePx": dilate_px,
        "featherPx": feather_px,
        "poisson": poisson,
        "poissonApplied": composite_meta.get("poissonApplied", False),
        "poissonMeta": composite_meta.get("poisson", {}),
        "inputMaskPx": int(input_mask_bool.sum()),
        "maskPx": int(mask_bool.sum()),
        "alphaPx": composite_meta["alphaPx"],
        "outsideMaskByteExact": exact,
        "startVram": start_vram,
        "endVram": smi_memory(),
        "peakVram": {"usedMiB": _peak_vram_mib, "totalMiB": smi_memory().get("totalMiB")},
        "torchPeakAllocMiB": round(torch.cuda.max_memory_allocated() / (1024**2), 1),
        "textEncodersNone": {
            "text_encoder": getattr(_pipe, "text_encoder", None) is None,
            "text_encoder_2": getattr(_pipe, "text_encoder_2", None) is None,
        },
        "timing": timing,
    }
    return output, meta


class Handler(BaseHTTPRequestHandler):
    server_version = "WiCiFluxFill/1.0"

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
                "loaded": _pipe is not None,
                "steps": NUM_STEPS,
                "guidance": GUIDANCE,
                "defaults": {
                    "dilatePx": DEFAULT_DILATE_PX,
                    "featherPx": DEFAULT_FEATHER_PX,
                    "poisson": DEFAULT_POISSON,
                    "quantMode": QUANT_MODE,
                    "steps": NUM_STEPS,
                    "guidance": GUIDANCE,
                    "turboLora": TURBO_LORA_ENABLED,
                    "bnb8Bf16Io": BNB8_BF16_IO,
                    "compile": {
                        "enabled": COMPILE_ENABLED,
                        "mode": COMPILE_MODE,
                        "dynamic": COMPILE_DYNAMIC,
                    },
                },
                "load": _load_meta,
                "lastFill": _last_fill,
                "vram": smi_memory(),
                "peakVramMiB": _peak_vram_mib,
                "processes": smi_processes(),
            },
        )

    def do_POST(self) -> None:
        global _last_fill
        path = self.path.split("?", 1)[0]
        if path != "/fill":
            json_response(self, 404, {"ok": False, "error": "not_found"})
            return
        request_start = time.perf_counter()
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > MAX_BODY_BYTES:
                raise ValueError(f"invalid body length {length}")
            raw_body = self.rfile.read(length)
            payload = json.loads(raw_body.decode("utf-8"))
            decode_start = time.perf_counter()
            image = data_url_to_image(payload.get("image") or payload.get("imagePng") or payload.get("imageUrl"), "RGB")
            mask = data_url_to_image(payload.get("mask") or payload.get("maskPng") or payload.get("maskUrl"), "L")
            input_decode_seconds = time.perf_counter() - decode_start
            dilate_px = parse_nonnegative_int(payload.get("dilatePx"), DEFAULT_DILATE_PX, "dilatePx")
            feather_px = parse_nonnegative_int(payload.get("featherPx"), DEFAULT_FEATHER_PX, "featherPx")
            poisson = parse_bool(payload.get("poisson"), DEFAULT_POISSON, "poisson")
            with _lock:
                output, meta = run_fill(
                    image,
                    mask,
                    payload.get("photoId"),
                    dilate_px,
                    feather_px,
                    poisson,
                    input_decode_seconds,
                )
            encode_start = time.perf_counter()
            output_data_url = image_to_data_url(output)
            meta["timing"]["outputEncodeMs"] = round((time.perf_counter() - encode_start) * 1000, 1)
            meta["handlerMs"] = round((time.perf_counter() - request_start) * 1000, 1)
            measured_ms = (
                meta["timing"]["inputDecodePlusPadMs"]
                + meta["timing"]["pipePreMs"]
                + meta["timing"]["pipeDenoiseMs"]
                + meta["timing"]["pipePostMs"]
                + meta["timing"]["compositeMs"]
                + meta["timing"]["outputEncodeMs"]
            )
            meta["timing"]["handlerMs"] = meta["handlerMs"]
            meta["timing"]["otherHandlerMs"] = round(max(meta["handlerMs"] - measured_ms, 0.0), 1)
            _last_fill = meta
            log_event("flux_fill_done", **meta)
            json_response(
                self,
                200,
                {
                    "ok": True,
                    "image": output_data_url,
                    "seconds": meta["seconds"],
                    "metrics": meta,
                },
            )
            del image, mask, output
        except Exception as exc:  # noqa: BLE001 - client needs exact blocker.
            retryable = "out of memory" in str(exc).lower() or "cuda" in str(exc).lower()
            log_event("flux_fill_failed", error=repr(exc), traceback=traceback.format_exc(limit=8), retryable=retryable)
            json_response(
                self,
                503 if retryable else 500,
                {
                    "ok": False,
                    "error": "flux_fill_failed",
                    "message": repr(exc),
                    "retryable": retryable,
                    "vram": smi_memory(),
                },
            )
        finally:
            clear_cuda_cache()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default=os.environ.get("WICI_FLUX_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("WICI_FLUX_PORT", "8771")))
    args = parser.parse_args()
    log_event(
        "flux_endpoint_start",
        host=args.host,
        port=args.port,
        configDir=str(CONFIG_DIR),
        promptEmbeds=str(PROMPT_EMBEDS),
        quantMode=QUANT_MODE,
        steps=NUM_STEPS,
        guidance=GUIDANCE,
        bnb8Bf16Io=BNB8_BF16_IO,
        compile={
            "enabled": COMPILE_ENABLED,
            "mode": COMPILE_MODE,
            "dynamic": COMPILE_DYNAMIC,
        },
        turboLora={
            "enabled": TURBO_LORA_ENABLED,
            "path": str(TURBO_LORA_PATH),
            "scale": TURBO_LORA_SCALE,
        },
        defaults={
            "dilatePx": DEFAULT_DILATE_PX,
            "featherPx": DEFAULT_FEATHER_PX,
            "poisson": DEFAULT_POISSON,
            "quantMode": QUANT_MODE,
            "steps": NUM_STEPS,
            "guidance": GUIDANCE,
            "turboLora": TURBO_LORA_ENABLED,
            "bnb8Bf16Io": BNB8_BF16_IO,
            "compile": {
                "enabled": COMPILE_ENABLED,
                "mode": COMPILE_MODE,
                "dynamic": COMPILE_DYNAMIC,
            },
        },
    )
    load_model()
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
