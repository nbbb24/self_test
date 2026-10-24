#!/usr/bin/env bash
# Source: https://github.com/shun8686/sglang/wiki/%5BA5%5D-DeepSeek-V4-Flash-%E5%8D%95%E6%9C%BA%E6%B7%B7%E9%83%A8
# Retrieved: 2026-10-23. DeepSeek-V4-Flash, four-NPU mixed deployment.
# A5 Wiki configuration; this local copy has not been tested on NPU hardware.
# Start a compatible MemCache MetaService separately on ports 5006/6006.
# Wiki model/operator paths and host tuning are retained; inspect before running.
set -eo pipefail

usage() {
  cat <<'EOF'
Usage: bash hicache_A5.sh [serve|bench|--help]
  serve   Start the Wiki's four-NPU DeepSeek-V4-Flash server (default).
  bench   Run the Wiki's shared-prefix benchmark against 127.0.0.1:31000.
Environment overrides: MODEL_PATH, ASCEND_RT_VISIBLE_DEVICES (default: 4,5,6,7).
MemCache: device_urma, meta tcp://127.0.0.1:5006, store tcp://127.0.0.1:6006.
Start MemCache separately and activate the matching SGLang/Ascend environment.
Serve applies the Wiki's CPU governor and sysctl tuning, requiring permissions.
A5 Wiki configuration; this local copy has not been tested on NPU hardware.
EOF
}

if (( $# > 1 )); then
  usage >&2
  exit 2
fi
case "${1:-serve}" in
  -h|--help) usage; exit 0 ;;
  serve) ;;
  bench)
MODEL_PATH="${MODEL_PATH:-/mnt/share/weight/DeepSeek-V4-Flash-0731}"
python3 -m sglang.benchmark.serving \
  --host 127.0.0.1 \
  --port 31000 \
  --backend sglang \
  --dataset-name generated-shared-prefix \
  --gsp-num-groups 3 \
  --gsp-prompts-per-group 32 \
  --gsp-system-prompt-len 28800 \
  --gsp-question-len 3200 \
  --gsp-output-len 1024 \
  --max-concurrency 32 \
  --random-range-ratio 1 \
  --seed 100 \
  --model "$MODEL_PATH" \
  --tokenizer "$MODEL_PATH" \
  --warmup-requests 1 \
  --cache-report
    exit 0
    ;;
  *) usage >&2; exit 2 ;;
esac
echo performance | tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
sysctl -w vm.swappiness=0
sysctl -w kernel.numa_balancing=0

source /usr/local/Ascend/ascend-toolkit/latest/opp/vendors/customize/bin/set_env.bash
source /usr/local/Ascend/ascend-toolkit/latest/opp/vendors/custom_transformer/bin/set_env.bash
source /usr/local/Ascend/ascend-toolkit/latest/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh


export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export STREAMS_PER_DEVICE=32
export SGLANG_SET_CPU_AFFINITY=1

#TEST
export TASK_QUEUE_ENABLE=1
export INF_NAN_MODE_FORCE_DISABLE=1
export SGLANG_DEFAULT_THINKING=1
export SGLANG_DSV4_REASONING_EFFORT=max

#HCCL deepep
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-4,5,6,7}"
export DEEPEP_HCCL_BUFFSIZE=2048
export HCCL_SOCKET_IFNAME=lo
export GLOO_SOCKET_IFNAME=lo

# 蚂蚁搬家，ROUND*TOKENS≥chunkedprefillsize/tp*dp
export DEEPEP_NORMAL_COMBINE_ENABLE_LONG_SEQ=1
export DEEPEP_NORMAL_LONG_SEQ_ROUND=16
export DEEPEP_NORMAL_LONG_SEQ_PER_ROUND_TOKENS=2048

export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=256


# dsv4
export IS_DEEPSEEK_V4=1
export USE_FUSED_HC_PRE_ASCENDC=1
export SGLANG_DSV4_NPU_FUSED_COMPRESSOR=1
export SGLANG_DSV4_NPU_FUSED_COMPRESSOR_PREFILL=1


# skip gpu branch
export SGLANG_OPT_USE_OVERLAP_STORE_CACHE=False
export FORCE_DRAFT_MODEL_NON_QUANT=1
export SGLANG_DSV4_FP4_EXPERTS=True
export SGLANG_OPT_FUSE_WQA_WKV=0
export SGLANG_OPT_BF16_FP32_GEMM_ALGO=torch
export SGLANG_OPT_USE_FUSED_HASH_TOPK=False
export SGLANG_OPT_USE_TILELANG_MHC_PRE=False
export SGLANG_OPT_DEEPGEMM_HC_PRENORM=False
export SGLANG_OPT_USE_TILELANG_MHC_POST=False
export SGLANG_OPT_FP8_WO_A_GEMM=False

#export PYTHONPATH=/home/l00567497/sglang/python:$PYTHONPATH
export MODEL_PATH="${MODEL_PATH:-/mnt/share/weight/DeepSeek-V4-Flash-0731/}"

export ASCEND_SLOG_PRINT_TO_STDOUT=0  # 1/0 Plog是否打屏（推荐为0）
export ASCEND_GLOBAL_LOG_LEVEL=3  # 日志等级 0: debug 1: info 2: warning 3: error

# perfermance
export SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1
export SGLANG_NPU_USE_MULTI_STREAM=1
export USE_NPU_MOE_GATING_TOP_K=1
export ASCEND_LAUNCH_BLOCKING=1
export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=256


# Dspark
export SGLANG_RAGGED_VERIFY_MODE=static
export SGLANG_DSPARK_FAST_KERNEL=0
export SGLANG_DSPARK_FAST_SAMPLING=0
export SGLANG_DSPARK_A5_EXTRA_OPS_SO=/usr/local/Ascend/extra-ops/vllm_ascend_C.cpython-312-aarch64-linux-gnu.so
export ASCEND_CUSTOM_OPP_PATH=/usr/local/Ascend/extra-ops/vendors/custom_transformer/:$ASCEND_CUSTOM_OPP_PATH
export LD_LIBRARY_PATH=/usr/local/Ascend/extra-ops/vendors/custom_transformer/op_api/lib:$LD_LIBRARY_PATH

python3 -m sglang.launch_server --model-path "${MODEL_PATH}" \
    --max-total-tokens 65536 \
    --page-size 128 \
    --tp-size 4 \
    --trust-remote-code \
    --attention-backend dsv4 \
    --device npu \
    --watchdog-timeout 9000 \
    --host 0.0.0.0 --port 31000 \
    --mem-fraction-static 0.66 \
    --max-running-requests 128 \
    --chunked-prefill-size 131072 \
    --max-prefill-tokens 131072 \
    --cuda-graph-bs-decode 1 2 4 8 16\
    --kv-cache-dtype auto \
    --enable-dp-lm-head \
    --enable-dp-attention --dp-size 4  \
    --reasoning-parser deepseek-v4 \
    --speculative-algorithm DSPARK \
    --speculative-draft-model-path "${MODEL_PATH}" \
    --speculative-draft-model-quantization modelslim \
    --speculative-draft-attention-backend ascend \
    --speculative-num-draft-tokens 6 \
    --speculative-dspark-block-size 5 \
    --enable-hierarchical-cache \
    --hicache-io-backend kernel_ascend \
    --enable-cache-report \
    --hicache-ratio 2 \
    --hicache-write-policy write_through \
    --hicache-storage-backend npu_memcache \
    --hicache-storage-backend-extra-config '{"meta_service_url":"tcp://127.0.0.1:5006", "config_store_url":"tcp://127.0.0.1:6006", "log_level":"info", "world_size":4, "protocol": "device_urma", "dram_size": "30GB"}' \
    2>&1 | tee "sglang_server_A5_$(date +%Y%m%d_%H%M%S).log"
