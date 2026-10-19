#!/usr/bin/env bash

# =============================================================================
# SGLang HiCache L2/L3 Validation on Ascend NPU
# =============================================================================
#
# What this script does:
#   1. Starts an NPU MemCache MetaService.
#   2. Starts SGLang Prefill and Decode servers on one machine.
#   3. Starts the SGLang PD router.
#   4. Runs a cold request, flushes L1/L2, and replays the request.
#   5. Passes only when KV cache is restored from MemCache L3 and output matches.
#
# Fastest way to run:
#   cp sglang/hicache/example.env sglang/hicache/local.env
#   # Edit MODEL_PATH and the MemFabric addresses in local.env.
#   CONFIG_FILE=sglang/hicache/local.env \
#     bash sglang/hicache/run_npu_memcache_l2_l3.sh a3
#
# Direct run with built-in defaults:
#   bash sglang/hicache/run_npu_memcache_l2_l3.sh a3
#   bash sglang/hicache/run_npu_memcache_l2_l3.sh a5
#
# Show all options:
#   bash sglang/hicache/run_npu_memcache_l2_l3.sh --help
#
# =============================================================================

set -Eeuo pipefail


# =============================================================================
# 1. USER INTERFACE AND LOGGING
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

usage() {
  cat <<'EOF'
Usage:
  run_npu_memcache_l2_l3.sh {a3|a5}

Hardware profiles:
  a3    Use device_sdma for the HiCache L2/L3 MemCache path.
  a5    Use device_urma for the HiCache L2/L3 MemCache path.

Recommended:
  cp sglang/hicache/example.env sglang/hicache/local.env
  CONFIG_FILE=sglang/hicache/local.env \
    bash sglang/hicache/run_npu_memcache_l2_l3.sh a3

Useful controls:
  CONFIG_FILE=<path>    Load configuration from an env file.
  DRY_RUN=1             Validate and print configuration without launching.
  KEEP_SERVERS=1        Keep all services running after a successful test.
  PD_PROTOCOL=<value>   Override the Prefill-to-Decode transfer protocol.

All supported variables are documented in sglang/README.md and example.env.
EOF
}

section() {
  printf '\n===============================================================================\n'
  printf '%s\n' "$1"
  printf '===============================================================================\n'
}

step() {
  printf '\n==> %s\n' "$1"
}

info() {
  printf '    %s\n' "$1"
}

warn() {
  printf 'WARNING: %s\n' "$1" >&2
}

die() {
  printf 'ERROR: %s\n' "$1" >&2
  exit 1
}


# =============================================================================
# 2. OPTIONAL CONFIG FILE AND HARDWARE PROFILE
# =============================================================================

if [[ -n "${CONFIG_FILE:-}" ]]; then
  [[ -r "${CONFIG_FILE}" ]] || die "CONFIG_FILE is not readable: ${CONFIG_FILE}"
  # shellcheck disable=SC1090
  set -a
  source "${CONFIG_FILE}"
  set +a
fi

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

PROFILE="${1:-${NPU_PROFILE:-}}"
case "${PROFILE}" in
  a3)
    DEFAULT_MEMCACHE_PROTOCOL="device_sdma"
    ;;
  a5)
    DEFAULT_MEMCACHE_PROTOCOL="device_urma"
    ;;
  *)
    usage >&2
    exit 2
    ;;
esac


# =============================================================================
# 3. CONFIGURATION
# =============================================================================
# Override any value below with an environment variable or CONFIG_FILE.

# ---- Model and output --------------------------------------------------------
PYTHON_BIN="${PYTHON_BIN:-python}"
MODEL_PATH="${MODEL_PATH:-/home/weights/Qwen3-32B-W8A8}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen}"
LOG_DIR="${LOG_DIR:-/home/sxy/logs/sglang-hicache-${PROFILE}}"

# ---- NPU placement -----------------------------------------------------------
PREFILL_DEVICES="${PREFILL_DEVICES:-12,13}"
DECODE_DEVICES="${DECODE_DEVICES:-14,15}"
TP_SIZE="${TP_SIZE:-2}"

# ---- Service ports -----------------------------------------------------------
PREFILL_PORT="${PREFILL_PORT:-30000}"
DECODE_PORT="${DECODE_PORT:-40000}"
ROUTER_PORT="${ROUTER_PORT:-8000}"
BOOTSTRAP_PORT="${BOOTSTRAP_PORT:-31000}"

# ---- Transport protocols -----------------------------------------------------
# PD_PROTOCOL controls Prefill -> Decode KV transfer.
# MEMCACHE_PROTOCOL controls HiCache L2 <-> L3 transfer.
PD_PROTOCOL="${PD_PROTOCOL:-device_rdma}"
MEMCACHE_PROTOCOL="${MEMCACHE_PROTOCOL:-${DEFAULT_MEMCACHE_PROTOCOL}}"
PREFILL_MF_STORE_URL="${PREFILL_MF_STORE_URL:-tcp://127.0.0.1:31001}"
DECODE_MF_STORE_URL="${DECODE_MF_STORE_URL:-${PREFILL_MF_STORE_URL}}"
MF_LOG_LEVEL="${ASCEND_MF_LOG_LEVEL:-0}"

# ---- MemCache service and capacity ------------------------------------------
META_SERVICE_URL="${META_SERVICE_URL:-tcp://127.0.0.1:5000}"
CONFIG_STORE_URL="${CONFIG_STORE_URL:-tcp://127.0.0.1:6000}"
MEMCACHE_METRICS_URL="${MEMCACHE_METRICS_URL:-http://127.0.0.1:18000}"
MEMCACHE_WORLD_SIZE="${MEMCACHE_WORLD_SIZE:-256}"
MEMCACHE_DRAM_SIZE="${MEMCACHE_DRAM_SIZE:-8GB}"

# device_urma requires max_dram_size * world_size > 32 TiB. This value is an
# address-space ceiling, not an eager allocation. The defaults span 64 TiB.
MEMCACHE_MAX_DRAM_SIZE="${MEMCACHE_MAX_DRAM_SIZE:-256GB}"
HICACHE_SIZE_GB="${HICACHE_SIZE_GB:-4}"

# ---- Runtime controls --------------------------------------------------------
STARTUP_TIMEOUT="${STARTUP_TIMEOUT:-1200}"
KEEP_SERVERS="${KEEP_SERVERS:-0}"
DRY_RUN="${DRY_RUN:-0}"

META_CONFIG_PATH="${LOG_DIR}/metaservice-config.json"
LOCAL_CONFIG_PATH="${LOG_DIR}/localservice-config.json"


# =============================================================================
# 4. PROCESS LIFECYCLE HELPERS
# =============================================================================

PIDS=()
PROCESS_NAMES=()
LAST_PID=""

start_background_process() {
  local name=$1
  local log_file=$2
  shift 2

  step "Starting ${name}"
  info "Log: ${log_file}"
  "$@" >"${log_file}" 2>&1 &
  LAST_PID=$!
  PIDS+=("${LAST_PID}")
  PROCESS_NAMES+=("${name}")
  info "PID: ${LAST_PID}"
}

wait_for_health() {
  local name=$1
  local url=$2
  local pid=$3
  local log_file=$4
  local deadline=$((SECONDS + STARTUP_TIMEOUT))

  step "Waiting for ${name} health check"
  info "URL: ${url}"

  while (( SECONDS < deadline )); do
    if ! kill -0 "${pid}" 2>/dev/null; then
      printf '\n%s exited before becoming healthy. Last 100 log lines:\n' "${name}" >&2
      tail -n 100 "${log_file}" >&2 || true
      return 1
    fi
    if curl -fsS "${url}" >/dev/null 2>&1; then
      info "${name} is ready."
      return 0
    fi
    sleep 2
  done

  printf '\nTimed out waiting for %s. Last 100 log lines:\n' "${name}" >&2
  tail -n 100 "${log_file}" >&2 || true
  return 1
}

cleanup() {
  local status=$?

  if [[ "${KEEP_SERVERS}" == "1" && ${status} -eq 0 ]]; then
    section "SERVICES LEFT RUNNING"
    info "KEEP_SERVERS=1 was set."
    for i in "${!PIDS[@]}"; do
      info "${PROCESS_NAMES[i]}: PID ${PIDS[i]}"
    done
    return
  fi

  if (( ${#PIDS[@]} > 0 )); then
    section "CLEANUP"
    info "Stopping ${#PIDS[@]} service process(es) ..."
  fi

  # Send SIGTERM first so SGLang can release NPU resources gracefully.
  for ((i=${#PIDS[@]}-1; i>=0; i--)); do
    if kill -0 "${PIDS[i]}" 2>/dev/null; then
      info "Stopping ${PROCESS_NAMES[i]} (PID ${PIDS[i]})"
      kill "${PIDS[i]}" 2>/dev/null || true
    fi
  done

  # Allow graceful shutdown, then kill only the exact processes we launched.
  local deadline=$((SECONDS + 30))
  while (( SECONDS < deadline )); do
    local any_alive=0
    for pid in "${PIDS[@]}"; do
      if kill -0 "${pid}" 2>/dev/null; then
        any_alive=1
        break
      fi
    done
    (( any_alive == 0 )) && break
    sleep 1
  done

  for pid in "${PIDS[@]}"; do
    if kill -0 "${pid}" 2>/dev/null; then
      warn "PID ${pid} did not stop gracefully; sending SIGKILL."
      kill -KILL "${pid}" 2>/dev/null || true
    fi
    wait "${pid}" 2>/dev/null || true
  done

  if (( status != 0 )); then
    printf '\nValidation failed. Inspect logs under: %s\n' "${LOG_DIR}" >&2
  fi
}

trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM


# =============================================================================
# 5. PREFLIGHT CHECKS
# =============================================================================

require_command() {
  command -v "$1" >/dev/null 2>&1 || die "Required command not found: $1"
}

device_count() {
  local devices=$1
  local commas=${devices//[^,]/}
  printf '%d' "$(( ${#commas} + 1 ))"
}

print_configuration() {
  cat <<EOF
Hardware profile        : ${PROFILE^^}
Model                   : ${MODEL_PATH}
Served model name       : ${SERVED_MODEL_NAME}
Prefill devices / TP    : ${PREFILL_DEVICES} / ${TP_SIZE}
Decode devices / TP     : ${DECODE_DEVICES} / ${TP_SIZE}
PD protocol             : ${PD_PROTOCOL}
HiCache L2/L3 protocol  : ${MEMCACHE_PROTOCOL}
Prefill MF store URL    : ${PREFILL_MF_STORE_URL}
Decode MF store URL     : ${DECODE_MF_STORE_URL}
HiCache L2 size/rank    : ${HICACHE_SIZE_GB} GiB
MemCache DRAM/rank      : ${MEMCACHE_DRAM_SIZE}
MemCache max DRAM/rank  : ${MEMCACHE_MAX_DRAM_SIZE}
MemCache world size     : ${MEMCACHE_WORLD_SIZE}
Prefill / Decode ports  : ${PREFILL_PORT} / ${DECODE_PORT}
Router / Bootstrap port : ${ROUTER_PORT} / ${BOOTSTRAP_PORT}
Log directory           : ${LOG_DIR}
EOF
}

section "PREFLIGHT CHECKS"

require_command "${PYTHON_BIN}"
require_command curl

[[ "${TP_SIZE}" =~ ^[1-9][0-9]*$ ]] || die "TP_SIZE must be a positive integer."
for port in "${PREFILL_PORT}" "${DECODE_PORT}" "${ROUTER_PORT}" "${BOOTSTRAP_PORT}"; do
  [[ "${port}" =~ ^[1-9][0-9]{0,4}$ ]] || die "All service ports must be integers from 1 to 65535."
  (( port >= 1 && port <= 65535 )) || die "Service port is outside 1-65535: ${port}"
done

[[ "${PREFILL_PORT}" != "${DECODE_PORT}" ]] || die "Prefill and Decode ports must differ."
[[ "${PREFILL_PORT}" != "${ROUTER_PORT}" ]] || die "Prefill and router ports must differ."
[[ "${DECODE_PORT}" != "${ROUTER_PORT}" ]] || die "Decode and router ports must differ."

[[ -n "${PREFILL_DEVICES}" ]] || die "PREFILL_DEVICES must not be empty."
[[ -n "${DECODE_DEVICES}" ]] || die "DECODE_DEVICES must not be empty."
prefill_device_count=$(device_count "${PREFILL_DEVICES}")
decode_device_count=$(device_count "${DECODE_DEVICES}")
(( prefill_device_count >= TP_SIZE )) || die "PREFILL_DEVICES has fewer devices than TP_SIZE."
(( decode_device_count >= TP_SIZE )) || die "DECODE_DEVICES has fewer devices than TP_SIZE."

if [[ "${MODEL_PATH}" == /* && ! -e "${MODEL_PATH}" ]]; then
  die "MODEL_PATH does not exist: ${MODEL_PATH}"
fi

step "Checking required Python packages"
"${PYTHON_BIN}" -c "import memcache_hybrid, sglang, sglang_router"
info "Python dependencies are available."

if [[ "${PREFILL_MF_STORE_URL}" == *"127.0.0.1"* ]]; then
  warn "PREFILL_MF_STORE_URL uses loopback. Set the local MemFabric-reachable IP if required."
fi

section "RESOLVED CONFIGURATION"
print_configuration

if [[ "${DRY_RUN}" == "1" ]]; then
  section "DRY RUN COMPLETE"
  info "Configuration is valid. No service was launched."
  exit 0
fi


# =============================================================================
# 6. GENERATE MEMCACHE CONFIGURATION
# =============================================================================

section "GENERATE MEMCACHE CONFIGURATION"
mkdir -p "${LOG_DIR}"

cat >"${META_CONFIG_PATH}" <<EOF
{
  "meta_service_url": "${META_SERVICE_URL}",
  "config_store_url": "${CONFIG_STORE_URL}",
  "metrics_url": "${MEMCACHE_METRICS_URL}",
  "log_level": "info",
  "log_path": "${LOG_DIR}/memcache"
}
EOF

cat >"${LOCAL_CONFIG_PATH}" <<EOF
{
  "meta_service_url": "${META_SERVICE_URL}",
  "config_store_url": "${CONFIG_STORE_URL}",
  "log_level": "info",
  "world_size": ${MEMCACHE_WORLD_SIZE},
  "protocol": "${MEMCACHE_PROTOCOL}",
  "dram_size": "${MEMCACHE_DRAM_SIZE}",
  "max_dram_size": "${MEMCACHE_MAX_DRAM_SIZE}"
}
EOF

info "MetaService config : ${META_CONFIG_PATH}"
info "LocalService config: ${LOCAL_CONFIG_PATH}"


# =============================================================================
# 7. START NPU MEMCACHE METASERVICE
# =============================================================================

section "START NPU MEMCACHE METASERVICE"

start_background_process \
  "NPU MemCache MetaService" \
  "${LOG_DIR}/memcache-meta.log" \
  "${PYTHON_BIN}" -m sglang.srt.mem_cache.storage.npu_memcache.start_meta_service \
    --config_path "${META_CONFIG_PATH}"
META_PID=${LAST_PID}

sleep 3
if ! kill -0 "${META_PID}" 2>/dev/null; then
  tail -n 100 "${LOG_DIR}/memcache-meta.log" >&2 || true
  die "NPU MemCache MetaService failed to start."
fi
info "MetaService process is running."


# =============================================================================
# 8. START SGLANG PREFILL AND DECODE SERVERS
# =============================================================================

section "START SGLANG PREFILL AND DECODE SERVERS"

# Arguments shared by Prefill and Decode. Role-specific options are kept below
# so the differences are visible without searching through the script.
COMMON_ARGS=(
  --model-path "${MODEL_PATH}"
  --served-model-name "${SERVED_MODEL_NAME}"
  --tp-size "${TP_SIZE}"
  --dp-size 1
  --device npu
  --attention-backend ascend
  --host 0.0.0.0
  --trust-remote-code
  --schedule-policy fcfs
  --chunked-prefill-size -1
  --schedule-conservativeness 0.3
  --disable-overlap-schedule
  --mem-fraction-static 0.8
  --context-length 20000
  --max-running-requests 64
  --enable-metrics
  --enable-cache-report
)

# ---- Prefill: owns HiCache L1/L2/L3 and writes to NPU MemCache --------------
start_background_process \
  "SGLang Prefill server on NPUs ${PREFILL_DEVICES}" \
  "${LOG_DIR}/sglang-prefill.log" \
  env \
    ASCEND_RT_VISIBLE_DEVICES="${PREFILL_DEVICES}" \
    ASCEND_MF_STORE_URL="${PREFILL_MF_STORE_URL}" \
    ASCEND_MF_TRANSFER_PROTOCOL="${PD_PROTOCOL}" \
    ASCEND_MF_LOG_LEVEL="${MF_LOG_LEVEL}" \
    SGLANG_NPU_MEMCACHE_ENABLE_WARMUP=1 \
  "${PYTHON_BIN}" -m sglang.launch_server \
    "${COMMON_ARGS[@]}" \
    --disaggregation-mode prefill \
    --port "${PREFILL_PORT}" \
    --disaggregation-bootstrap-port "${BOOTSTRAP_PORT}" \
    --disaggregation-transfer-backend ascend \
    --enable-hierarchical-cache \
    --hicache-size "${HICACHE_SIZE_GB}" \
    --hicache-write-policy write_through \
    --hicache-io-backend kernel_ascend \
    --hicache-mem-layout page_first_direct \
    --hicache-storage-backend npu_memcache \
    --hicache-storage-prefetch-policy wait_complete \
    --hicache-storage-backend-extra-config "@${LOCAL_CONFIG_PATH}"
PREFILL_PID=${LAST_PID}

# ---- Decode: receives Prefill KV through the independent PD transport --------
start_background_process \
  "SGLang Decode server on NPUs ${DECODE_DEVICES}" \
  "${LOG_DIR}/sglang-decode.log" \
  env \
    ASCEND_RT_VISIBLE_DEVICES="${DECODE_DEVICES}" \
    ASCEND_MF_STORE_URL="${DECODE_MF_STORE_URL}" \
    ASCEND_MF_TRANSFER_PROTOCOL="${PD_PROTOCOL}" \
    ASCEND_MF_LOG_LEVEL="${MF_LOG_LEVEL}" \
  "${PYTHON_BIN}" -m sglang.launch_server \
    "${COMMON_ARGS[@]}" \
    --disaggregation-mode decode \
    --port "${DECODE_PORT}" \
    --disaggregation-transfer-backend ascend \
    --max-prefill-tokens 36000
DECODE_PID=${LAST_PID}

wait_for_health \
  "Prefill server" \
  "http://127.0.0.1:${PREFILL_PORT}/health" \
  "${PREFILL_PID}" \
  "${LOG_DIR}/sglang-prefill.log"

wait_for_health \
  "Decode server" \
  "http://127.0.0.1:${DECODE_PORT}/health" \
  "${DECODE_PID}" \
  "${LOG_DIR}/sglang-decode.log"


# =============================================================================
# 9. START PD ROUTER
# =============================================================================

section "START PD ROUTER"

start_background_process \
  "SGLang PD router" \
  "${LOG_DIR}/sglang-router.log" \
  "${PYTHON_BIN}" -m sglang_router.launch_router \
    --pd-disaggregation \
    --policy cache_aware \
    --prefill "http://127.0.0.1:${PREFILL_PORT}" "${BOOTSTRAP_PORT}" \
    --decode "http://127.0.0.1:${DECODE_PORT}" \
    --host 0.0.0.0 \
    --port "${ROUTER_PORT}"
ROUTER_PID=${LAST_PID}

wait_for_health \
  "PD router" \
  "http://127.0.0.1:${ROUTER_PORT}/health" \
  "${ROUTER_PID}" \
  "${LOG_DIR}/sglang-router.log"


# =============================================================================
# 10. RUN L2/L3 ROUND-TRIP VALIDATION
# =============================================================================

section "RUN HICACHE L2/L3 VALIDATION"
info "Profile: ${PROFILE^^}"
info "PD transport: ${PD_PROTOCOL}"
info "HiCache L2/L3 transport: ${MEMCACHE_PROTOCOL}"

"${PYTHON_BIN}" "${SCRIPT_DIR}/verify_npu_memcache_l2_l3.py" \
  --router-url "http://127.0.0.1:${ROUTER_PORT}" \
  --prefill-url "http://127.0.0.1:${PREFILL_PORT}" \
  --output-dir "${LOG_DIR}"

section "VALIDATION PASSED"
info "Artifacts: ${LOG_DIR}"
info "Summary  : ${LOG_DIR}/summary.json"
