#!/opt/venvs/flux/bin/python
"""Download model assets into the /models bind mount.

Draft helper used by install.sh via `docker compose run --rm downloader`.
It mirrors the paths consumed by the current endpoints.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import time
from pathlib import Path
from urllib.request import urlretrieve

from huggingface_hub import hf_hub_download, snapshot_download
from huggingface_hub.errors import GatedRepoError, HfHubHTTPError


MODEL_ROOT = Path(os.environ.get("WICI_MODEL_DIR", "/models")).resolve()
REPO_ROOT = Path(os.environ.get("WICI_REPO_ROOT", "/opt/wici/spatial-album-backend")).resolve()
HF_HOME = MODEL_ROOT / "huggingface"
TORCH_HOME = MODEL_ROOT / "torch"

SHARP_URL = "https://ml-site.cdn-apple.com/models/sharp/sharp_2572gikvuh.pt"
FLUX_REPO = "black-forest-labs/FLUX.1-Fill-dev"
DIFIX_REPO = "nvidia/difix_ref"
TEXT_ENCODER_REPO = "comfyanonymous/flux_text_encoders"


def log(message: str) -> None:
    print(message, flush=True)


def token_value() -> str | None:
    value = os.environ.get("HF_TOKEN")
    if value:
        return value.strip()
    token_file = MODEL_ROOT / "hf-token"
    if token_file.exists():
        value = token_file.read_text(encoding="utf-8").strip()
        if value:
            return value
    return None


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def download_url(url: str, target: Path, min_bytes: int = 1) -> None:
    ensure_dir(target.parent)
    if target.exists() and target.stat().st_size >= min_bytes:
        log(f"present {target} bytes={target.stat().st_size}")
        return
    tmp = target.with_suffix(target.suffix + ".tmp")
    if tmp.exists():
        tmp.unlink()
    log(f"download {url} -> {target}")
    urlretrieve(url, tmp)
    tmp.replace(target)
    log(f"downloaded {target} bytes={target.stat().st_size}")


def hf_file(repo_id: str, filename: str, local_dir: Path, token: str | None = None) -> Path:
    ensure_dir(local_dir)
    target = local_dir / filename
    if target.exists() and target.stat().st_size > 0:
        log(f"present {target} bytes={target.stat().st_size}")
        return target
    log(f"hf download {repo_id}/{filename} -> {local_dir}")
    path = Path(hf_hub_download(repo_id=repo_id, filename=filename, local_dir=str(local_dir), token=token))
    log(f"downloaded {path} bytes={path.stat().st_size}")
    return path


def ensure_default_difix_source() -> None:
    target = MODEL_ROOT / "difix" / "source_photo_original.jpg"
    if target.exists() and target.stat().st_size > 0:
        log(f"present {target} bytes={target.stat().st_size}")
        return
    candidates = [
        REPO_ROOT / "backend/assets/default_difix_source.jpg",
    ]
    for candidate in candidates:
        if candidate.exists():
            ensure_dir(target.parent)
            shutil.copyfile(candidate, target)
            log(f"created default Difix source {target} from {candidate}")
            return
    raise FileNotFoundError("no default source image found at backend/assets/default_difix_source.jpg")


def main() -> int:
    os.environ["HF_HOME"] = str(HF_HOME)
    os.environ["TORCH_HOME"] = str(TORCH_HOME)
    ensure_dir(MODEL_ROOT)
    ensure_dir(HF_HOME)
    ensure_dir(TORCH_HOME)

    manifest: dict[str, object] = {
        "generatedAt": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "modelRoot": str(MODEL_ROOT),
        "assets": [],
    }
    assets: list[dict[str, object]] = manifest["assets"]  # type: ignore[assignment]

    sharp_target = TORCH_HOME / "hub/checkpoints/sharp_2572gikvuh.pt"
    download_url(SHARP_URL, sharp_target, min_bytes=2_000_000_000)
    assets.append({"name": "sharp", "source": SHARP_URL, "path": str(sharp_target)})

    log(f"hf snapshot {DIFIX_REPO} -> HF cache {HF_HOME}")
    snapshot_download(repo_id=DIFIX_REPO, token=token_value(), local_files_only=False)
    assets.append({"name": "difix_ref", "source": f"https://huggingface.co/{DIFIX_REPO}"})
    ensure_default_difix_source()

    flux_token = token_value()
    if not flux_token:
        raise RuntimeError(
            "FLUX.1-Fill-dev is gated. Put a Hugging Face token with accepted FLUX terms "
            f"in {MODEL_ROOT / 'hf-token'} or set HF_TOKEN."
        )

    comfy_root = MODEL_ROOT / "ComfyUI/models"
    hf_file(FLUX_REPO, "flux1-fill-dev.safetensors", comfy_root / "diffusion_models", token=flux_token)
    hf_file(FLUX_REPO, "ae.safetensors", comfy_root / "vae", token=flux_token)
    hf_file(TEXT_ENCODER_REPO, "clip_l.safetensors", comfy_root / "text_encoders")
    hf_file(TEXT_ENCODER_REPO, "t5xxl_fp8_e4m3fn.safetensors", comfy_root / "text_encoders")
    assets.append({"name": "flux_fill", "source": f"https://huggingface.co/{FLUX_REPO}"})
    assets.append({"name": "flux_text_encoders", "source": f"https://huggingface.co/{TEXT_ENCODER_REPO}"})

    config_dir = MODEL_ROOT / "flux/flux_peripheral/hf_flux1_fill_dev_config"
    log(f"hf snapshot FLUX configs/tokenizers -> {config_dir}")
    snapshot_download(
        repo_id=FLUX_REPO,
        local_dir=str(config_dir),
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
        token=flux_token,
        local_files_only=False,
    )

    manifest_path = MODEL_ROOT / "weights-manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    log(f"wrote {manifest_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (GatedRepoError, HfHubHTTPError, RuntimeError, FileNotFoundError) as exc:
        print(f"weight download failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
