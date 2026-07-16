#!/usr/bin/env python3
"""Prepare COLMAP data and run the pinned gsplat trainer with VRAM preload."""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
import numpy as np
from PIL import Image, ImageOps
from plyfile import PlyData

SH_C0 = 0.28209479177387814
SPLAT_ROW_BYTES = 32


def event(stage: str, **fields: object) -> None:
    print(json.dumps({"stage": stage, **fields}, separators=(",", ":")), flush=True)


def run(command: list[str], stage: str, *, parse_training: bool = False, progress_scale: int = 1) -> None:
    event(stage)
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
    assert process.stdout is not None
    for line in process.stdout:
        handled = False
        if parse_training:
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                payload = None
            if isinstance(payload, dict) and payload.get("event") == "training_progress":
                event(
                    "training",
                    step=int(payload["step"]) * progress_scale,
                    totalSteps=int(payload["totalSteps"]) * progress_scale,
                )
                handled = True
            elif isinstance(payload, dict) and payload.get("event") == "preload_ready":
                event("preloading", **{key: payload[key] for key in ("images", "bytes", "chunkBytes", "elapsedSeconds")})
                handled = True
        if not handled:
            sys.stderr.write(line)
    code = process.wait()
    if code != 0:
        raise RuntimeError(f"{stage} command exited with {code}")


def prepare_images(source_dir: Path, dataset: Path, factor: int) -> None:
    images = dataset / "images"
    images_factor = dataset / f"images_{factor}"
    images.mkdir(parents=True, exist_ok=True)
    images_factor.mkdir(parents=True, exist_ok=True)
    source_files = sorted(path for path in source_dir.iterdir() if path.is_file())
    if len(source_files) < 2:
        raise ValueError("at least two source images are required")
    resample = getattr(Image, "Resampling", Image).LANCZOS
    for index, source in enumerate(source_files):
        name = f"{index:03d}.jpg"
        with Image.open(source) as decoded:
            decoded = ImageOps.exif_transpose(decoded).convert("RGB")
            decoded.save(images / name, format="JPEG", quality=95, optimize=True)
            resized = decoded.resize((max(1, round(decoded.width / factor)), max(1, round(decoded.height / factor))), resample)
            resized.save(images_factor / name, format="JPEG", quality=95, optimize=True)


def colmap(dataset: Path) -> None:
    database = dataset / "database.db"
    sparse = dataset / "sparse"
    sparse.mkdir(parents=True, exist_ok=True)
    common = ["--database_path", str(database)]
    run([
        "colmap", "feature_extractor", *common,
        "--image_path", str(dataset / "images"),
        "--ImageReader.single_camera", "1",
        "--ImageReader.camera_model", "PINHOLE",
        "--SiftExtraction.use_gpu", "0",
    ], "colmap_features")
    run(["colmap", "exhaustive_matcher", *common, "--SiftMatching.use_gpu", "0"], "colmap_matching")
    run([
        "colmap", "mapper", *common,
        "--image_path", str(dataset / "images"),
        "--output_path", str(sparse),
    ], "colmap_mapping")
    model = sparse / "0"
    if not (model / "cameras.bin").is_file() or not (model / "images.bin").is_file():
        raise RuntimeError("COLMAP did not produce a registered sparse/0 model")


def _ply_stack(vertex, names: list[str]) -> np.ndarray:
    return np.stack([np.asarray(vertex[name], dtype=np.float32) for name in names], axis=-1)


def _encode_u8(values: np.ndarray) -> np.ndarray:
    return np.clip(np.rint(values), 0.0, 255.0).astype(np.uint8)


def _sigmoid(values: np.ndarray) -> np.ndarray:
    positive = values >= 0
    out = np.empty_like(values, dtype=np.float32)
    out[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exp_values = np.exp(values[~positive])
    out[~positive] = exp_values / (1.0 + exp_values)
    return out


def convert_ply_to_splat(ply_path: Path, splat_path: Path) -> dict[str, int]:
    vertex = PlyData.read(ply_path)["vertex"]
    count = len(vertex.data)
    dtype = np.dtype([
        ("xyz", "<f4", (3,)),
        ("scale", "<f4", (3,)),
        ("rgba", "u1", (4,)),
        ("rotation", "u1", (4,)),
    ])
    rows = np.empty(count, dtype=dtype)
    rows["xyz"] = _ply_stack(vertex, ["x", "y", "z"]).astype("<f4", copy=False)
    rows["scale"] = np.exp(_ply_stack(vertex, ["scale_0", "scale_1", "scale_2"])).astype("<f4", copy=False)
    sh0 = _ply_stack(vertex, ["f_dc_0", "f_dc_1", "f_dc_2"])
    rgb = _encode_u8((0.5 + SH_C0 * sh0) * 255.0)
    alpha = _encode_u8(_sigmoid(np.asarray(vertex["opacity"], dtype=np.float32)) * 255.0)[:, None]
    rows["rgba"] = np.concatenate([rgb, alpha], axis=1)
    quats = _ply_stack(vertex, ["rot_0", "rot_1", "rot_2", "rot_3"])
    quats /= np.linalg.norm(quats, axis=1, keepdims=True).clip(1e-8)
    rows["rotation"] = _encode_u8(quats * 128.0 + 128.0)
    temporary = splat_path.with_suffix(".tmp.splat")
    rows.tofile(temporary)
    temporary.replace(splat_path)
    if splat_path.stat().st_size != count * SPLAT_ROW_BYTES:
        raise RuntimeError("invalid generated splat byte length")
    return {"numGaussians": count, "splatBytes": splat_path.stat().st_size}


def main() -> int:
    workspace = Path(os.environ["WICI_GSPLAT_JOB_DIR"]).resolve()
    source_dir = Path(os.environ["WICI_GSPLAT_IMAGES_DIR"]).resolve()
    output_dir = Path(os.environ["WICI_GSPLAT_OUTPUT_DIR"]).resolve()
    manifest = json.loads(Path(os.environ["WICI_GSPLAT_MANIFEST"]).read_text(encoding="utf-8"))
    options = manifest["options"]
    if options.get("preloadTrainingData") is not True or os.environ.get("WICI_GSPLAT_PRELOAD_TRAIN_DATA") != "1":
        raise ValueError("VRAM preload is required for this worker")
    factor = int(os.environ.get("WICI_GSPLAT_DATA_FACTOR", "4"))
    batch = int(options["batchSize"])
    equivalent_steps = int(options["equivalentSteps"])
    actual_steps = max(1, equivalent_steps // batch)
    dataset = workspace / "dataset"
    train_result = output_dir / "training"
    output_dir.mkdir(parents=True, exist_ok=True)
    prepare_images(source_dir, dataset, factor)
    colmap(dataset)
    trainer = Path(os.environ.get("WICI_GSPLAT_TRAINER", "/opt/sota/gsplat/examples/simple_trainer.py"))
    python = os.environ.get("WICI_GSPLAT_PYTHON", "/opt/venvs/orbit/bin/python")
    command = [
        python, str(trainer), "default",
        "--disable-viewer",
        "--disable-video",
        "--data-dir", str(dataset),
        "--data-factor", str(factor),
        "--patch-size", os.environ.get("WICI_GSPLAT_PATCH_SIZE", "256"),
        "--result-dir", str(train_result),
        "--max-steps", str(equivalent_steps),
        "--steps-scaler", str(1.0 / batch),
        "--eval-steps", "999999",
        "--save-steps", str(equivalent_steps),
        "--save-ply",
        "--ply-steps", str(equivalent_steps),
        "--batch-size", str(batch),
        "--preload-train-data",
        "--preload-chunk-bytes", str(options["preloadChunkBytes"]),
        "--sh-degree", os.environ.get("WICI_GSPLAT_SH_DEGREE", "0"),
        "--ssim-lambda", os.environ.get("WICI_GSPLAT_SSIM_LAMBDA", "0"),
        "--test-every", "8",
    ]
    run(command, "training", parse_training=True, progress_scale=batch)
    candidates = sorted((train_result / "ply").glob("point_cloud_*.ply"))
    if not candidates:
        raise RuntimeError("gsplat trainer did not export a PLY")
    result_ply = output_dir / "result.ply"
    shutil.copy2(candidates[-1], result_ply)
    result = output_dir / "result.splat"
    splat_meta = convert_ply_to_splat(result_ply, result)
    stats_files = sorted((train_result / "stats").glob("train_step*_rank0.json"))
    stats = json.loads(stats_files[-1].read_text(encoding="utf-8")) if stats_files else {}
    (output_dir / "result.json").write_text(json.dumps({
        "protocol": "wici.gsplat-training.v1",
        "actualSteps": actual_steps,
        "equivalentSteps": equivalent_steps,
        "preloadTrainingData": True,
        "preloadChunkBytes": options["preloadChunkBytes"],
        "stats": stats,
        **splat_meta,
    }, indent=2), encoding="utf-8")
    event("ready", step=equivalent_steps, totalSteps=equivalent_steps, resultBytes=result.stat().st_size)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
