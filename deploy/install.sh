#!/usr/bin/env bash
set -euo pipefail

DEPLOY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${DEPLOY_DIR}/.." && pwd)"
MODEL_DIR="${WICI_MODEL_DIR:-${HOME}/.cache/wici-form1/models}"
RUNTIME_DIR="${WICI_RUNTIME_DIR:-${HOME}/.cache/wici-form1/runtime}"
BACKEND_PORT="${WICI_BACKEND_PORT:-54228}"
REQUIRED_DISK_GB="${WICI_REQUIRED_DISK_GB:-80}"
COMPOSE_FILE="${DEPLOY_DIR}/docker-compose.yml"
ENV_FILE="${DEPLOY_DIR}/.env"
DOCKER="${WICI_DOCKER_CMD:-docker}"
MDNS_ENABLED="${WICI_ENABLE_MDNS:-1}"
MDNS_SERVICE_TYPE="${WICI_MDNS_SERVICE_TYPE:-_wici-backend._tcp}"
MDNS_VERSION="${WICI_MDNS_VERSION:-1}"
MDNS_BASE_PATH="${WICI_MDNS_BASE_PATH:-/}"
AVAHI_SERVICE_FILE="/etc/avahi/services/wici-backend.service"
MDNS_ENV_FILE="/etc/wici/wici-mdns-health.env"
MDNS_SCRIPT_FILE="/usr/local/bin/wici-mdns-health-advertise"
MDNS_UNIT_FILE="/etc/systemd/system/wici-mdns-health.service"

log() { printf '[wici-install] %s\n' "$*"; }
die() { printf '[wici-install] ERROR: %s\n' "$*" >&2; exit 1; }
have() { command -v "$1" >/dev/null 2>&1; }
is_wsl() { grep -qi microsoft /proc/version 2>/dev/null; }

compose() {
  ${DOCKER} compose --env-file "${ENV_FILE}" -f "${COMPOSE_FILE}" "$@"
}

require_sudo() {
  if [[ "${EUID}" -eq 0 ]]; then
    "$@"
  elif have sudo; then
    sudo "$@"
  else
    die "sudo is required for this step"
  fi
}

maybe_install_docker_linux() {
  if have docker || [[ "${DOCKER}" != "docker" ]]; then
    return
  fi
  if is_wsl; then
    die "Docker is not available in WSL. Install Docker Desktop for Windows, enable WSL integration, then rerun."
  fi
  [[ "$(uname -s)" == "Linux" ]] || die "Docker auto-install is only drafted for Linux"
  if [[ "${WICI_ASSUME_YES:-0}" != "1" ]]; then
    read -r -p "Docker is missing. Install Docker Engine with sudo now? [y/N] " answer
    [[ "${answer}" == "y" || "${answer}" == "Y" ]] || die "Docker install declined"
  fi
  log "installing Docker Engine via get.docker.com"
  curl -fsSL https://get.docker.com | require_sudo sh
}

ensure_docker_usable() {
  maybe_install_docker_linux
  if [[ "${DOCKER}" == "docker" ]]; then
    have docker || die "docker command is not available"
  fi
  if ! ${DOCKER} info >/dev/null 2>&1; then
    if [[ "${DOCKER}" == "docker" ]] && have sudo && sudo docker info >/dev/null 2>&1; then
      DOCKER="sudo docker"
      log "using sudo for Docker commands"
    else
      die "docker daemon is not reachable"
    fi
  fi
  if ! ${DOCKER} compose version >/dev/null 2>&1; then
    if is_wsl; then
      die "docker compose v2 is required. Install/repair Docker Desktop Compose integration."
    fi
    if [[ "${WICI_ASSUME_YES:-0}" != "1" ]]; then
      read -r -p "Docker Compose v2 is missing. Install docker-compose-v2 with sudo now? [y/N] " answer
      [[ "${answer}" == "y" || "${answer}" == "Y" ]] || die "Docker Compose install declined"
    fi
    require_sudo apt-get update
    require_sudo apt-get install -y docker-compose-v2
  fi
  ${DOCKER} compose version >/dev/null 2>&1 || die "docker compose v2 is required"
  log "docker and compose are available"
}

install_nvidia_toolkit_linux() {
  if is_wsl; then
    die "NVIDIA Container Toolkit is managed by Docker Desktop in WSL. Enable GPU support in Docker Desktop and rerun."
  fi
  [[ -r /etc/os-release ]] || die "cannot detect Linux distribution for NVIDIA Container Toolkit install"
  . /etc/os-release
  case "${ID_LIKE:-} ${ID:-}" in
    *debian*|*ubuntu*)
      log "installing NVIDIA Container Toolkit"
      curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
        | require_sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
      curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
        | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
        | require_sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list >/dev/null
      require_sudo apt-get update
      require_sudo apt-get install -y nvidia-container-toolkit
      require_sudo nvidia-ctk runtime configure --runtime=docker
      require_sudo systemctl restart docker
      ;;
    *)
      die "auto-install for NVIDIA Container Toolkit is only drafted for Debian/Ubuntu"
      ;;
  esac
}

ensure_nvidia_container_runtime() {
  log "checking Docker GPU runtime"
  if ${DOCKER} run --rm --gpus all nvidia/cuda:12.8.1-base-ubuntu24.04 nvidia-smi >/dev/null 2>&1; then
    log "Docker GPU runtime works"
    return
  fi
  if [[ "${WICI_ASSUME_YES:-0}" != "1" ]]; then
    read -r -p "Docker GPU runtime is missing. Install/configure NVIDIA Container Toolkit now? [y/N] " answer
    [[ "${answer}" == "y" || "${answer}" == "Y" ]] || die "NVIDIA Container Toolkit install declined"
  fi
  install_nvidia_toolkit_linux
  ${DOCKER} run --rm --gpus all nvidia/cuda:12.8.1-base-ubuntu24.04 nvidia-smi >/dev/null 2>&1 \
    || die "Docker GPU runtime still does not work"
}

ensure_health_gated_mdns() {
  if [[ "${MDNS_ENABLED}" == "0" ]]; then
    log "mDNS advertisement disabled by WICI_ENABLE_MDNS=0"
    if have systemctl; then
      require_sudo systemctl disable --now wici-mdns-health.service >/dev/null 2>&1 || true
    fi
    require_sudo rm -f "${AVAHI_SERVICE_FILE}" || true
    return
  fi
  [[ "$(uname -s)" == "Linux" ]] || die "Avahi mDNS install is only supported on Linux"

  if is_wsl; then
    log "warning: Avahi inside WSL2 may not be visible on the physical LAN; Windows host Bonjour/mDNS publishing may be required"
  fi

  if ! have avahi-daemon || ! have avahi-browse; then
    if [[ -r /etc/os-release ]]; then
      . /etc/os-release
    else
      die "cannot detect Linux distribution for Avahi install"
    fi
    case "${ID_LIKE:-} ${ID:-}" in
      *debian*|*ubuntu*)
        log "installing Avahi mDNS publisher"
        require_sudo apt-get update
        require_sudo apt-get install -y avahi-daemon avahi-utils
        ;;
      *)
        die "auto-install for Avahi is only drafted for Debian/Ubuntu"
        ;;
    esac
  fi

  have systemctl || die "health-gated mDNS requires systemd"
  [[ -x "${DEPLOY_DIR}/avahi/health-advertise.sh" ]] || die "missing ${DEPLOY_DIR}/avahi/health-advertise.sh"
  [[ -f "${DEPLOY_DIR}/avahi/wici-mdns-health.service" ]] || die "missing ${DEPLOY_DIR}/avahi/wici-mdns-health.service"

  require_sudo install -d -m 0755 /etc/wici
  require_sudo install -d -m 0755 /etc/avahi/services

  local tmp
  tmp="$(mktemp)"
  cat > "${tmp}" <<EOF
WICI_BACKEND_PORT=${BACKEND_PORT}
WICI_MDNS_HEALTH_URL=http://127.0.0.1:${BACKEND_PORT}/orbit/health
WICI_MDNS_SERVICE_FILE=${AVAHI_SERVICE_FILE}
WICI_MDNS_SERVICE_TYPE=${MDNS_SERVICE_TYPE}
WICI_MDNS_VERSION=${MDNS_VERSION}
WICI_MDNS_BASE_PATH=${MDNS_BASE_PATH}
WICI_MDNS_POLL_SECONDS=${WICI_MDNS_POLL_SECONDS:-10}
EOF
  require_sudo install -m 0644 "${tmp}" "${MDNS_ENV_FILE}"
  rm -f "${tmp}"
  require_sudo install -m 0755 "${DEPLOY_DIR}/avahi/health-advertise.sh" "${MDNS_SCRIPT_FILE}"
  require_sudo install -m 0644 "${DEPLOY_DIR}/avahi/wici-mdns-health.service" "${MDNS_UNIT_FILE}"

  require_sudo rm -f "${AVAHI_SERVICE_FILE}"
  require_sudo systemctl enable --now avahi-daemon
  require_sudo systemctl daemon-reload
  require_sudo systemctl enable wici-mdns-health.service >/dev/null
  require_sudo systemctl restart wici-mdns-health.service

  log "installed health-gated mDNS advertiser for ${MDNS_SERVICE_TYPE} on tcp/${BACKEND_PORT}"
  log "advertisement is present only while http://127.0.0.1:${BACKEND_PORT}/orbit/health returns ok"
}

write_env_file() {
  mkdir -p "${MODEL_DIR}" "${RUNTIME_DIR}"
  cat > "${ENV_FILE}" <<EOF
WICI_MODEL_DIR=${MODEL_DIR}
WICI_RUNTIME_DIR=${RUNTIME_DIR}
WICI_BACKEND_PORT=${BACKEND_PORT}
COMPOSE_PROJECT_NAME=wici-form1
EOF
  log "wrote ${ENV_FILE}"
}

ensure_hf_token() {
  local token_file="${MODEL_DIR}/hf-token"
  if [[ -s "${token_file}" || -n "${HF_TOKEN:-}" ]]; then
    return
  fi
  cat <<'EOF'
FLUX.1-Fill-dev is gated on Hugging Face.
Use a token for an account that has accepted the FLUX.1-dev non-commercial terms.
EOF
  read -r -s -p "Hugging Face token: " token
  printf '\n'
  [[ -n "${token}" ]] || die "Hugging Face token is required for FLUX.1-Fill-dev"
  umask 077
  printf '%s\n' "${token}" > "${token_file}"
  log "stored token at ${token_file}"
}

lan_ip() {
  if is_wsl && have powershell.exe; then
    powershell.exe -NoProfile -Command "(Get-NetIPConfiguration | Where-Object { \$_.IPv4DefaultGateway -ne \$null -and \$_.NetAdapter.Status -eq 'Up' } | Select-Object -First 1 -ExpandProperty IPv4Address).IPAddress" 2>/dev/null \
      | tr -d '\r' | awk 'NF {print; exit}' && return
  fi
  ip route get 1.1.1.1 2>/dev/null | awk '{for (i=1;i<=NF;i++) if ($i=="src") {print $(i+1); exit}}' && return
  hostname -I 2>/dev/null | awk '{print $1; exit}'
}

main() {
  if [[ "${1:-}" == "--mdns-only" ]]; then
    ensure_health_gated_mdns
    return
  fi

  log "repo: ${REPO_ROOT}"
  log "model dir: ${MODEL_DIR}"
  log "disk preflight requires ${REQUIRED_DISK_GB} GiB free: about 32 GiB Docker image + about 35 GiB weights + margin"
  MIN_DISK_GB="${REQUIRED_DISK_GB}" "${DEPLOY_DIR}/gpu-preflight.sh" "${MODEL_DIR}"
  ensure_docker_usable
  ensure_nvidia_container_runtime
  write_env_file
  ensure_hf_token

  log "building backend image"
  compose build orbit

  log "downloading model weights into ${MODEL_DIR}"
  compose --profile tools run --rm downloader

  log "starting services"
  compose up -d orbit difix flux caddy
  ensure_health_gated_mdns

  ip_addr="$(lan_ip)"
  [[ -n "${ip_addr}" ]] || ip_addr="127.0.0.1"
  cat <<EOF

WiCi local backend is starting.
The Android app can also discover it with NSD:
  ${MDNS_SERVICE_TYPE} on port ${BACKEND_PORT}

Base URL for the app:
  http://${ip_addr}:${BACKEND_PORT}

Local health checks:
  http://127.0.0.1:${BACKEND_PORT}/orbit/health
  http://127.0.0.1:${BACKEND_PORT}/difix/health
  http://127.0.0.1:${BACKEND_PORT}/flux/health

EOF
}

main "$@"
