#!/usr/bin/env bash
set -euo pipefail

BACKEND_PORT="${WICI_BACKEND_PORT:-54228}"
HEALTH_URL="${WICI_MDNS_HEALTH_URL:-http://127.0.0.1:${BACKEND_PORT}/orbit/health}"
SERVICE_FILE="${WICI_MDNS_SERVICE_FILE:-/etc/avahi/services/wici-backend.service}"
SERVICE_TYPE="${WICI_MDNS_SERVICE_TYPE:-_wici-backend._tcp}"
SERVICE_NAME="${WICI_MDNS_SERVICE_NAME:-WiCi Backend on %h}"
TXT_VERSION="${WICI_MDNS_VERSION:-1}"
TXT_BASE="${WICI_MDNS_BASE_PATH:-/}"
POLL_SECONDS="${WICI_MDNS_POLL_SECONDS:-10}"
CURL_TIMEOUT="${WICI_MDNS_CURL_TIMEOUT:-3}"

log() {
  printf '[wici-mdns] %s\n' "$*"
}

service_xml() {
  cat <<EOF
<?xml version="1.0" standalone='no'?>
<!DOCTYPE service-group SYSTEM "avahi-service.dtd">
<service-group>
  <name replace-wildcards="yes">${SERVICE_NAME}</name>
  <service>
    <type>${SERVICE_TYPE}</type>
    <port>${BACKEND_PORT}</port>
    <txt-record>version=${TXT_VERSION}</txt-record>
    <txt-record>base=${TXT_BASE}</txt-record>
  </service>
</service-group>
EOF
}

is_healthy() {
  local body
  body="$(curl -fsS --max-time "${CURL_TIMEOUT}" "${HEALTH_URL}" 2>/dev/null || true)"
  [[ "${body}" =~ \"ok\"[[:space:]]*:[[:space:]]*true ]]
}

publish() {
  local dir tmp
  dir="$(dirname "${SERVICE_FILE}")"
  install -d -m 0755 "${dir}"
  tmp="$(mktemp "${dir}/wici-backend.service.tmp.XXXXXX")"
  service_xml > "${tmp}"
  chmod 0644 "${tmp}"
  if [[ -f "${SERVICE_FILE}" ]] && cmp -s "${tmp}" "${SERVICE_FILE}"; then
    rm -f "${tmp}"
    return
  fi
  mv "${tmp}" "${SERVICE_FILE}"
  log "advertising ${SERVICE_TYPE} on tcp/${BACKEND_PORT}; health=${HEALTH_URL}"
}

withdraw() {
  if [[ -f "${SERVICE_FILE}" ]]; then
    rm -f "${SERVICE_FILE}"
    log "withdrew ${SERVICE_TYPE}; health=${HEALTH_URL}"
  fi
}

shutdown() {
  withdraw
  exit 0
}

trap shutdown INT TERM

while true; do
  if is_healthy; then
    publish
  else
    withdraw
  fi
  sleep "${POLL_SECONDS}"
done
