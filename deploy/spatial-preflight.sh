#!/usr/bin/env bash
set -uo pipefail

DEPLOY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="${WICI_ENV_FILE:-${DEPLOY_DIR}/.env}"
COMPOSE_FILE="${WICI_COMPOSE_FILE:-${DEPLOY_DIR}/docker-compose.yml}"
IMAGE="${WICI_BACKEND_IMAGE:-spatial-album-backend:cu128}"

FAILED=0

pass() { printf 'PASS: %s\n' "$*"; }
fail() { printf 'FAIL: %s\n' "$*" >&2; FAILED=1; }

if [[ -r "${ENV_FILE}" ]]; then
  set -a
  # deploy/.env contains shell-compatible KEY=VALUE assignments.
  # shellcheck disable=SC1090
  . "${ENV_FILE}"
  set +a
  pass "environment file is readable: ${ENV_FILE}"
else
  fail "environment file is not readable: ${ENV_FILE}"
fi

MODEL_DIR="${WICI_MODEL_DIR:-${HOME}/.cache/wici-form1/models}"
BACKEND_PORT="${WICI_BACKEND_PORT:-54228}"
PROJECT_NAME="${COMPOSE_PROJECT_NAME:-wici-form1}"

if [[ -n "${WICI_DOCKER_CMD:-}" ]]; then
  read -r -a DOCKER <<<"${WICI_DOCKER_CMD}"
else
  DOCKER=(docker)
fi

if ! "${DOCKER[@]}" info >/dev/null 2>&1; then
  if command -v sudo >/dev/null 2>&1 && sudo -n docker info >/dev/null 2>&1; then
    DOCKER=(sudo -n docker)
  else
    fail "Docker daemon is not readable"
  fi
fi

if "${DOCKER[@]}" info >/dev/null 2>&1; then
  pass "Docker daemon is readable"

  if "${DOCKER[@]}" compose --env-file "${ENV_FILE}" -f "${COMPOSE_FILE}" config --quiet >/dev/null 2>&1; then
    pass "Compose configuration is valid"
  else
    fail "Compose configuration is invalid"
  fi

  if "${DOCKER[@]}" image inspect "${IMAGE}" >/dev/null 2>&1; then
    image_id="$("${DOCKER[@]}" image inspect --format '{{.Id}}' "${IMAGE}" 2>/dev/null | cut -c1-19)"
    pass "backend image exists: ${IMAGE} (${image_id})"
  else
    fail "backend image is missing: ${IMAGE}"
  fi

  if "${DOCKER[@]}" image inspect caddy:2.8-alpine >/dev/null 2>&1; then
    pass "Caddy image exists: caddy:2.8-alpine"
  else
    fail "Caddy image is missing: caddy:2.8-alpine"
  fi

  runtimes="$("${DOCKER[@]}" info --format '{{json .Runtimes}}' 2>/dev/null || true)"
  if grep -q '"nvidia"' <<<"${runtimes}"; then
    pass "Docker NVIDIA runtime is registered"
  elif [[ -s /etc/cdi/nvidia.yaml || -s /var/run/cdi/nvidia.yaml ]]; then
    pass "NVIDIA CDI device specification is present"
  else
    fail "Docker has neither a registered NVIDIA runtime nor an NVIDIA CDI specification"
  fi
fi

if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
  gpu_name="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -n1)"
  pass "host NVIDIA device is available: ${gpu_name}"
else
  fail "host NVIDIA device is unavailable"
fi

check_file() {
  local label="$1" path="$2" min_bytes="$3"
  if [[ ! -f "${path}" ]]; then
    fail "${label} is missing: ${path}"
    return
  fi
  local bytes
  bytes="$(stat -c '%s' "${path}" 2>/dev/null || printf '0')"
  if [[ "${bytes}" =~ ^[0-9]+$ ]] && (( bytes >= min_bytes )); then
    pass "${label} is present: ${bytes} bytes"
  else
    fail "${label} is too small: ${bytes} bytes (minimum ${min_bytes})"
  fi
}

if [[ -d "${MODEL_DIR}/torch" ]]; then
  pass "Torch cache path resolves: ${MODEL_DIR}/torch"
else
  fail "Torch cache path is missing or broken: ${MODEL_DIR}/torch"
fi
if [[ -d "${MODEL_DIR}/huggingface" ]]; then
  pass "Hugging Face cache path resolves: ${MODEL_DIR}/huggingface"
else
  fail "Hugging Face cache path is missing or broken: ${MODEL_DIR}/huggingface"
fi

check_file "SHARP checkpoint" "${MODEL_DIR}/torch/hub/checkpoints/sharp_2572gikvuh.pt" 2000000000

difix_cache="${MODEL_DIR}/huggingface/hub/models--nvidia--difix_ref"
if [[ -d "${difix_cache}" ]] \
  && find -L "${difix_cache}/snapshots" -type f -name model_index.json -size +0c -print -quit 2>/dev/null | grep -q . \
  && find -L "${difix_cache}/snapshots" -type f -name '*.safetensors' -size +1000000c -print -quit 2>/dev/null | grep -q .; then
  pass "DiFix Hugging Face snapshot is present: ${difix_cache}"
else
  fail "DiFix Hugging Face snapshot is missing or incomplete: ${difix_cache}"
fi

check_file "DiFix source photo" "${MODEL_DIR}/difix/source_photo_original.jpg" 10000
check_file "FLUX transformer" "${MODEL_DIR}/ComfyUI/models/diffusion_models/flux1-fill-dev.safetensors" 20000000000
check_file "FLUX VAE" "${MODEL_DIR}/ComfyUI/models/vae/ae.safetensors" 300000000
check_file "FLUX CLIP-L encoder" "${MODEL_DIR}/ComfyUI/models/text_encoders/clip_l.safetensors" 200000000
check_file "FLUX T5 encoder" "${MODEL_DIR}/ComfyUI/models/text_encoders/t5xxl_fp8_e4m3fn.safetensors" 4000000000
check_file "FLUX model index" "${MODEL_DIR}/flux/flux_peripheral/hf_flux1_fill_dev_config/model_index.json" 1
check_file "FLUX prompt embeddings" "${MODEL_DIR}/flux/flux_peripheral/prompt_embeds.pt" 1000000

if [[ -s "${DEPLOY_DIR}/caddy/Caddyfile" ]]; then
  pass "Caddy configuration is present"
else
  fail "Caddy configuration is missing or empty"
fi

listener="$(ss -H -ltn "sport = :${BACKEND_PORT}" 2>/dev/null || true)"
if [[ -z "${listener}" ]]; then
  pass "backend port ${BACKEND_PORT} is free"
elif "${DOCKER[@]}" info >/dev/null 2>&1; then
  owners="$("${DOCKER[@]}" ps --filter "publish=${BACKEND_PORT}" --format '{{.Names}}' 2>/dev/null || true)"
  if grep -Eq "^${PROJECT_NAME}-caddy-[0-9]+$" <<<"${owners}"; then
    pass "backend port ${BACKEND_PORT} is owned by expected Caddy container: $(tr '\n' ' ' <<<"${owners}" | xargs)"
  else
    fail "backend port ${BACKEND_PORT} is occupied by an unknown owner"
  fi
else
  fail "backend port ${BACKEND_PORT} is occupied and Docker ownership cannot be checked"
fi

if (( FAILED != 0 )); then
  printf 'PREFLIGHT_FAILED\n' >&2
  exit 1
fi

printf 'PREFLIGHT_OK\n'
