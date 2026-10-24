#!/bin/bash
# RECONSTRUCTED AFTER THE RUN from recorded edits; not an original snapshot.
set -eo pipefail
source /home/sxy/hicache_A5_20261023_7d19/runtime.env
source /usr/local/Ascend/ascend-toolkit/latest/set_env.sh
export ASCEND_RT_VISIBLE_DEVICES=4
export SGLANG_SET_CPU_AFFINITY=0
export CC="$RUN/clang-private.sh"
export PYTHONPATH="$RUN/python:${PYTHONPATH:-}"
export GLOO_SOCKET_IFNAME=lo
export ASCEND_GLOBAL_LOG_LEVEL=3
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export SGLANG_NPU_MEMCACHE_ENABLE_WARMUP=1
exec python3 -B -m sglang.launch_server \
  --model-path /home/weights/Qwen3-0.6B \
  --device npu --attention-backend ascend --tp-size 1 \
  --host 127.0.0.1 --port 31000 \
  --mem-fraction-static 0.15 --max-total-tokens 8192 \
  --context-length 8192 --max-running-requests 2 --chunked-prefill-size 2048 \
  --disable-cuda-graph --page-size 128 \
  --enable-metrics --enable-cache-report --return-output-ids \
  --enable-hierarchical-cache --hicache-size 2 \
  --hicache-io-backend kernel_ascend --hicache-mem-layout page_first_kv_split \
  --hicache-write-policy write_through \
  --hicache-storage-backend npu_memcache \
  --hicache-storage-prefetch-policy wait_complete \
  --hicache-storage-backend-extra-config @/home/sxy/hicache_A5_20261023_7d19/local.json
