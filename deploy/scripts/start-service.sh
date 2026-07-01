#!/usr/bin/env bash
set -euo pipefail

service="${1:-}"
repo_root="${WICI_REPO_ROOT:-/opt/wici/spatial-album-backend}"
backend="${WICI_BACKEND_DIR:-${repo_root}/backend}"

if [[ -z "${service}" ]]; then
  echo "usage: wici-start-service {orbit|difix|flux}" >&2
  exit 64
fi
if [[ ! -d "${backend}" ]]; then
  echo "missing mounted backend directory: ${backend}" >&2
  exit 66
fi

mkdir -p /models /runtime /home/wici /models/ComfyUI
ln -sfn /models/ComfyUI /home/wici/ComfyUI

export HOME=/home/wici
export HF_HOME="${HF_HOME:-/models/huggingface}"
export TORCH_HOME="${TORCH_HOME:-/models/torch}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

cuda_lib_path() {
  local roots=(
    /opt/venvs/orbit/lib/python3.12/site-packages/nvidia
    /opt/venvs/difix/lib/python3.12/site-packages/nvidia
    /opt/venvs/flux/lib/python3.12/site-packages/nvidia
    /usr/local/lib/python3.12/dist-packages/nvidia
    /usr/lib/python3/dist-packages/nvidia
  )
  local existing=()
  local root
  for root in "${roots[@]}"; do
    [[ -d "${root}" ]] && existing+=("${root}")
  done
  [[ "${#existing[@]}" -gt 0 ]] || return 0
  find "${existing[@]}" -maxdepth 2 -type d -name lib 2>/dev/null | sort -u | paste -sd: -
}

CUDA_PY_LIBS="$(cuda_lib_path)"
if [[ -n "${CUDA_PY_LIBS}" ]]; then
  export LD_LIBRARY_PATH="${CUDA_PY_LIBS}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi

cd "${backend}"

case "${service}" in
  orbit)
    export PYTORCH_JIT="${PYTORCH_JIT:-0}"
    export SHARP_CKPT="${SHARP_CKPT:-/models/torch/hub/checkpoints/sharp_2572gikvuh.pt}"
    export SHARP_PRECISION="${SHARP_PRECISION:-fp16}"
    export SHARP_FAST_UNPROJECT="${SHARP_FAST_UNPROJECT:-1}"
    export SHARP_SDP_BACKEND="${SHARP_SDP_BACKEND:-mem_efficient}"
    export SHARP_FAST_DECOMP="${SHARP_FAST_DECOMP:-analytic}"
    export WICI_ORBIT_INPROC_SHARP="${WICI_ORBIT_INPROC_SHARP:-1}"
    export PYTHONPATH="/opt/sota/ml-sharp/src${PYTHONPATH:+:${PYTHONPATH}}"
    exec /opt/venvs/orbit/bin/python "${backend}/orbit_preview_endpoint.py" \
      --host "${WICI_ORBIT_HOST:-0.0.0.0}" \
      --port "${WICI_ORBIT_PORT:-8772}"
    ;;

  difix)
    export MVP_DIFIX_INPROC="${MVP_DIFIX_INPROC:-1}"
    export MVP_DIFIX_KEEP_LOADED="${MVP_DIFIX_KEEP_LOADED:-1}"
    export MVP_DIFIX_MAX_SIDE="${MVP_DIFIX_MAX_SIDE:-960}"
    export MVP_DIFIX_MIN_FREE_GB="${MVP_DIFIX_MIN_FREE_GB:-8}"
    export MVP_DIFIX_ROOT="${MVP_DIFIX_ROOT:-/models/difix}"
    export MVP_DIFIX_SRC="${MVP_DIFIX_SRC:-/opt/Difix3D/src}"
    export MVP_DIFIX_SOURCE="${MVP_DIFIX_SOURCE:-/models/difix/source_photo_original.jpg}"
    export MVP_DIFIX_GALLERY_SOURCE_URL="${MVP_DIFIX_GALLERY_SOURCE_URL:-http://orbit:8772/source}"
    export PYTHONPATH="/opt/Difix3D/src${PYTHONPATH:+:${PYTHONPATH}}"
    exec /opt/venvs/difix/bin/python "${backend}/mvp_difix_endpoint.py" \
      --host "${MVP_DIFIX_HOST:-0.0.0.0}" \
      --port "${MVP_DIFIX_PORT:-8765}"
    ;;

  flux)
    flux_sp=/opt/venvs/flux/lib/python3.12/site-packages
    export WICI_HF_TOKEN_FILE="${WICI_HF_TOKEN_FILE:-/models/hf-token}"
    export WICI_FLUX_QUANT_MODE="${WICI_FLUX_QUANT_MODE:-bnb8}"
    export WICI_FLUX_BNB8_BF16_IO="${WICI_FLUX_BNB8_BF16_IO:-1}"
    export WICI_FLUX_OUT_DIR="${WICI_FLUX_OUT_DIR:-/models/flux/flux_peripheral}"
    export WICI_FLUX_CONFIG_DIR="${WICI_FLUX_CONFIG_DIR:-/models/flux/flux_peripheral/hf_flux1_fill_dev_config}"
    export WICI_FLUX_PROMPT_EMBEDS="${WICI_FLUX_PROMPT_EMBEDS:-/models/flux/flux_peripheral/prompt_embeds.pt}"
    export WICI_FLUX_TRANSFORMER="${WICI_FLUX_TRANSFORMER:-/models/ComfyUI/models/diffusion_models/flux1-fill-dev.safetensors}"
    export WICI_FLUX_VAE="${WICI_FLUX_VAE:-/models/ComfyUI/models/vae/ae.safetensors}"
    export PYTHONPATH="${flux_sp}${PYTHONPATH:+:${PYTHONPATH}}"
    exec /opt/venvs/flux/bin/python "${backend}/flux_endpoint.py" \
      --host "${WICI_FLUX_HOST:-0.0.0.0}" \
      --port "${WICI_FLUX_PORT:-8771}"
    ;;

  *)
    echo "unknown service: ${service}" >&2
    exit 64
    ;;
esac
