#!/usr/bin/env bash
# Single-node NPU HiCache L2/L3 validation with a PD-disaggregated SGLang server.
#
# Usage:
#   bash sglang/hicache/run_npu_memcache_l2_l3.sh a3
#   bash sglang/hicache/run_npu_memcache_l2_l3.sh a5
#
# Important overrides:
#   MODEL_PATH=/home/weights/Qwen3-32B-W8A8
#   PREFILL_DEVICES=12,13 DECODE_DEVICES=14,15
#   PREFILL_MF_STORE_URL=tcp://<local-ip>:31001
#   DECODE_MF_STORE_URL=tcp://<local-ip>:31001
#   PD_PROTOCOL=device_rdma  # independent from the HiCache L2/L3 protocol
#   KEEP_SERVERS=1  # keep processes running after validation

set -Eeuo pipefail

PROFILE="${1:-${NPU_PROFILE:-}}"
case "${PROFILE}" in
  a3)
    DEFAULT_PROTOCOL="device_sdma"
    ;;
  a5)
    DEFAULT_PROTOCOL="device_urma"
    ;;
  *)
    echo "Usage: $0 {a3|a5}" >&2
    exit 2
    ;;
esac

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
MODEL_PATH="${MODEL_PATH:-/home/weights/Qwen3-32B-W8A8}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen}"
LOG_DIR="${LOG_DIR:-/home/sxy/logs/sglang-hicache-${PROFILE}}"

PREFILL_DEVICES="${PREFILL_DEVICES:-12,13}"
DECODE_DEVICES="${DECODE_DEVICES:-14,15}"
TP_SIZE="${TP_SIZE:-2}"
PREFILL_PORT="${PREFILL_PORT:-30000}"
DECODE_PORT="${DECODE_PORT:-40000}"
ROUTER_PORT="${ROUTER_PORT:-8000}"
BOOTSTRAP_PORT="${BOOTSTRAP_PORT:-31000}"

PD_PROTOCOL="${PD_PROTOCOL:-device_rdma}"
MEMCACHE_PROTOCOL="${MEMCACHE_PROTOCOL:-${DEFAULT_PROTOCOL}}"
PREFILL_MF_STORE_URL="${PREFILL_MF_STORE_URL:-tcp://127.0.0.1:31001}"
DECODE_MF_STORE_URL="${DECODE_MF_STORE_URL:-${PREFILL_MF_STORE_URL}}"
MF_LOG_LEVEL="${ASCEND_MF_LOG_LEVEL:-0}"

META_SERVICE_URL="${META_SERVICE_URL:-tcp://127.0.0.1:5000}"
CONFIG_STORE_URL="${CONFIG_STORE_URL:-tcp://127.0.0.1:6000}"
MEMCACHE_METRICS_URL="${MEMCACHE_METRICS_URL:-http://127.0.0.1:18000}"
MEMCACHE_WORLD_SIZE="${MEMCACHE_WORLD_SIZE:-256}"
MEMCACHE_DRAM_SIZE="${MEMCACHE_DRAM_SIZE:-8GB}"
# device_urma requires max_dram_size * world_size to span more than 32 TiB.
# This is an address-space ceiling, not an eager allocation.
MEMCACHE_MAX_DRAM_SIZE="${MEMCACHE_MAX_DRAM_SIZE:-256GB}"
HICACHE_SIZE_GB="${HICACHE_SIZE_GB:-4}"
STARTUP_TIMEOUT="${STARTUP_TIMEOUT:-1200}"
KEEP_SERVERS="${KEEP_SERVERS:-0}"

mkdir -p "${LOG_DIR}"
META_CONFIG_PATH="${LOG_DIR}/metaservice-config.json"
LOCAL_CONFIG_PATH="${LOG_DIR}/localservice-config.json"
PIDS=()

cleanup() {
  local status=$?
  if [[ "${KEEP_SERVERS}" == "1" && ${status} -eq 0 ]]; then
    echo "KEEP_SERVERS=1; leaving processes running: ${PIDS[*]}"
    return
  fi
  for ((i=${#PIDS[@]}-1; i>=0; i--)); do
    if kill -0 "${PIDS[i]}" 2>/dev/null; then
      kill "${PIDS[i]}" 2>/dev/null || true
    fi
  done
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
      kill -KILL "${pid}" 2>/dev/null || true
    fi
    wait "${pid}" 2>/dev/null || true
  done
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

wait_for_health() {
  local name=$1
  local url=$2
  local pid=$3
  local log_file=$4
  local deadline=$((SECONDS + STARTUP_TIMEOUT))

  while (( SECONDS < deadline )); do
    if ! kill -0 "${pid}" 2>/dev/null; then
      echo "${name} exited before becoming healthy. Last log lines:" >&2
      tail -n 100 "${log_file}" >&2 || true
      return 1
    fi
    if curl -fsS "${url}" >/dev/null 2>&1; then
      echo "${name} is ready: ${url}"
      return 0
    fi
    sleep 2
  done

  echo "Timed out waiting for ${name}: ${url}" >&2
  tail -n 100 "${log_file}" >&2 || true
  return 1
}

echo "Checking Python dependencies ..."
"${PYTHON_BIN}" -c "import memcache_hybrid, sglang, sglang_router"

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

echo "Starting MemCache MetaService ..."
"${PYTHON_BIN}" -m sglang.srt.mem_cache.storage.npu_memcache.start_meta_service \
  --config_path "${META_CONFIG_PATH}" \
  >"${LOG_DIR}/memcache-meta.log" 2>&1 &
META_PID=$!
PIDS+=("${META_PID}")
sleep 3
if ! kill -0 "${META_PID}" 2>/dev/null; then
  echo "MemCache MetaService failed to start:" >&2
  tail -n 100 "${LOG_DIR}/memcache-meta.log" >&2 || true
  exit 1
fi

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

echo "Starting Prefill on NPUs ${PREFILL_DEVICES} ..."
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
  --hicache-storage-backend-extra-config "@${LOCAL_CONFIG_PATH}" \
  >"${LOG_DIR}/sglang-prefill.log" 2>&1 &
PREFILL_PID=$!
PIDS+=("${PREFILL_PID}")

echo "Starting Decode on NPUs ${DECODE_DEVICES} ..."
ASCEND_RT_VISIBLE_DEVICES="${DECODE_DEVICES}" \
ASCEND_MF_STORE_URL="${DECODE_MF_STORE_URL}" \
ASCEND_MF_TRANSFER_PROTOCOL="${PD_PROTOCOL}" \
ASCEND_MF_LOG_LEVEL="${MF_LOG_LEVEL}" \
"${PYTHON_BIN}" -m sglang.launch_server \
  "${COMMON_ARGS[@]}" \
  --disaggregation-mode decode \
  --port "${DECODE_PORT}" \
  --disaggregation-transfer-backend ascend \
  --max-prefill-tokens 36000 \
  >"${LOG_DIR}/sglang-decode.log" 2>&1 &
DECODE_PID=$!
PIDS+=("${DECODE_PID}")

wait_for_health \
  "Prefill" "http://127.0.0.1:${PREFILL_PORT}/health" \
  "${PREFILL_PID}" "${LOG_DIR}/sglang-prefill.log"
wait_for_health \
  "Decode" "http://127.0.0.1:${DECODE_PORT}/health" \
  "${DECODE_PID}" "${LOG_DIR}/sglang-decode.log"

echo "Starting PD router ..."
"${PYTHON_BIN}" -m sglang_router.launch_router \
  --pd-disaggregation \
  --policy cache_aware \
  --prefill "http://127.0.0.1:${PREFILL_PORT}" "${BOOTSTRAP_PORT}" \
  --decode "http://127.0.0.1:${DECODE_PORT}" \
  --host 0.0.0.0 \
  --port "${ROUTER_PORT}" \
  >"${LOG_DIR}/sglang-router.log" 2>&1 &
ROUTER_PID=$!
PIDS+=("${ROUTER_PID}")

wait_for_health \
  "Router" "http://127.0.0.1:${ROUTER_PORT}/health" \
  "${ROUTER_PID}" "${LOG_DIR}/sglang-router.log"

echo "Running ${PROFILE^^} HiCache validation: PD=${PD_PROTOCOL}, L3=${MEMCACHE_PROTOCOL}"
"${PYTHON_BIN}" "${SCRIPT_DIR}/verify_npu_memcache_l2_l3.py" \
  --router-url "http://127.0.0.1:${ROUTER_PORT}" \
  --prefill-url "http://127.0.0.1:${PREFILL_PORT}" \
  --output-dir "${LOG_DIR}"

echo "Validation artifacts: ${LOG_DIR}"
