#!/usr/bin/env bash
set -euo pipefail

MIN_VRAM_MIB="${MIN_VRAM_MIB:-24576}"
MIN_DISK_GB="${MIN_DISK_GB:-80}"
MIN_CUDA_VERSION="${MIN_CUDA_VERSION:-12.8}"
MIN_DRIVER_MAJOR="${MIN_DRIVER_MAJOR:-570}"
MIN_COMPUTE_CAP="${MIN_COMPUTE_CAP:-12.0}"
DISK_PATH="${1:-${WICI_MODEL_DIR:-${HOME}/.cache/wici-form1/models}}"
DISK_REQUIREMENT_DETAIL="about 32 GiB Docker image footprint + about 35 GiB model weights + build/download margin"

pass() { printf 'PASS: %s\n' "$*"; }
fail() { printf 'FAIL: %s\n' "$*" >&2; FAILED=1; }
warn() { printf 'WARN: %s\n' "$*" >&2; }

FAILED=0

is_wsl() {
  grep -qi microsoft /proc/version 2>/dev/null
}

version_ge() {
  [[ "$(printf '%s\n%s\n' "$2" "$1" | sort -V | head -n1)" == "$2" ]]
}

cap_ge() {
  local cap="$1" min="$2"
  version_ge "$cap" "$min"
}

if is_wsl; then
  pass "WSL2 detected; Docker Desktop must have WSL integration and GPU passthrough enabled"
else
  pass "Linux native environment detected"
fi

if ! command -v nvidia-smi >/dev/null 2>&1; then
  fail "nvidia-smi was not found. Install the NVIDIA driver first."
else
  pass "nvidia-smi found"
  smi_text="$(nvidia-smi || true)"
  cuda_version="$(printf '%s\n' "${smi_text}" | sed -n 's/.*CUDA Version: \([0-9.]*\).*/\1/p' | head -n1)"
  if [[ -n "${cuda_version}" ]] && version_ge "${cuda_version}" "${MIN_CUDA_VERSION}"; then
    pass "reported CUDA version ${cuda_version} >= ${MIN_CUDA_VERSION}"
  else
    fail "reported CUDA version ${cuda_version:-unknown} is below ${MIN_CUDA_VERSION}"
  fi

  mapfile -t gpu_rows < <(nvidia-smi --query-gpu=name,driver_version,memory.total,compute_cap --format=csv,noheader,nounits 2>/dev/null || true)
  if [[ "${#gpu_rows[@]}" -eq 0 ]]; then
    fail "could not query GPU details from nvidia-smi"
  fi

  usable_gpu=0
  for row in "${gpu_rows[@]}"; do
    IFS=',' read -r name driver mem_mib compute_cap <<<"${row}"
    name="$(echo "${name}" | xargs)"
    driver="$(echo "${driver}" | xargs)"
    mem_mib="$(echo "${mem_mib}" | xargs)"
    compute_cap="$(echo "${compute_cap}" | xargs)"
    driver_major="${driver%%.*}"

    if [[ "${driver_major}" =~ ^[0-9]+$ ]] && (( driver_major >= MIN_DRIVER_MAJOR )); then
      pass "${name}: driver ${driver} >= ${MIN_DRIVER_MAJOR}.x"
    else
      fail "${name}: driver ${driver:-unknown} is below required ${MIN_DRIVER_MAJOR}.x for this cu128 Blackwell stack"
    fi

    if [[ "${mem_mib}" =~ ^[0-9]+$ ]] && (( mem_mib >= MIN_VRAM_MIB )); then
      pass "${name}: VRAM ${mem_mib} MiB >= ${MIN_VRAM_MIB} MiB"
    else
      fail "${name}: VRAM ${mem_mib:-unknown} MiB is below required ${MIN_VRAM_MIB} MiB"
    fi

    if [[ -n "${compute_cap}" ]] && cap_ge "${compute_cap}" "${MIN_COMPUTE_CAP}"; then
      pass "${name}: compute capability ${compute_cap} >= ${MIN_COMPUTE_CAP}"
      usable_gpu=1
    else
      fail "${name}: compute capability ${compute_cap:-unknown} is below ${MIN_COMPUTE_CAP}; current SHARP endpoint expects Blackwell sm_120"
    fi
  done
  if (( usable_gpu == 0 )); then
    fail "no GPU satisfies the compute capability requirement"
  fi
fi

probe="${DISK_PATH}"
while [[ ! -e "${probe}" && "${probe}" != "/" ]]; do
  probe="$(dirname "${probe}")"
done
if [[ ! -e "${probe}" ]]; then
  probe="/"
fi
avail_kb="$(df -Pk "${probe}" | awk 'NR==2 {print $4}')"
avail_gb="$(( avail_kb / 1024 / 1024 ))"
if (( avail_gb >= MIN_DISK_GB )); then
  pass "free disk at ${probe}: ${avail_gb} GiB >= ${MIN_DISK_GB} GiB (${DISK_REQUIREMENT_DETAIL})"
else
  fail "free disk at ${probe}: ${avail_gb} GiB is below ${MIN_DISK_GB} GiB required (${DISK_REQUIREMENT_DETAIL})"
fi
if (( avail_gb < 100 )); then
  warn "80 GiB is the hard minimum; 100+ GiB is safer when Docker build cache and partial downloads are present"
fi

if (( FAILED != 0 )); then
  exit 1
fi
pass "GPU preflight passed"
