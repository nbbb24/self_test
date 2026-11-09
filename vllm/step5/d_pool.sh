#!/usr/bin/env bash
# 容器内执行：bash D.sh NODE；换机器修改下方IP数组或MASTER_IP/LOCAL_IP。
# 适用139性能配置：async关闭、synthetic MTP3接受率目标0.6，不是精度认证。
# 镜像必须已包含core/pull_worker白名单修复，或容器已只读挂载两补丁。
# 本脚本不修改安装包；容器需NPU/驱动、只读模型以及可写cache。
set -euo pipefail
NODE_ID=${1:?node index required}
[[ "$NODE_ID" =~ ^[0-3]$ ]] || exit 2
NODE_IPS=(10.208.128.46 10.208.128.47 10.208.128.48 10.208.128.49)
LOCAL_IP="${LOCAL_IP:-${NODE_IPS[NODE_ID]}}"
MASTER_IP="${MASTER_IP:-${NODE_IPS[0]}}"
COMM_IFACE="${COMM_IFACE:-bond4}"
MODEL_PATH="${MODEL_PATH:-/model/target}"
PYTHON_BIN="${PYTHON_BIN:-/opt/upgrade/.venv/bin/python}"
RUNTIME_ENV="${RUNTIME_ENV:-/opt/upgrade/runtime-env.sh}"
CACHE_ROOT="${CACHE_ROOT:-/cache}"
DP_START_RANK=$((NODE_ID * 2))
unset PYTHONPATH LD_PRELOAD VLLM_SERVER_DEV_MODE ASCEND_LAUNCH_BLOCKING VLLM_PP_LAYER_PARTITION
set +u
source "$RUNTIME_ENV"
set -u
export CLEAN_JIT_CACHE=0 PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONPYCACHEPREFIX=$CACHE_ROOT/pycache TORCH_EXTENSIONS_DIR=$CACHE_ROOT/torch-extensions
export XDG_CACHE_HOME=$CACHE_ROOT/xdg TRITON_CACHE_DIR=$CACHE_ROOT/triton HF_HOME=$CACHE_ROOT/hf
export VLLM_CACHE_ROOT=$CACHE_ROOT/vllm ASCEND_CACHE_PATH=$CACHE_ROOT/ascend
export OMP_NUM_THREADS=1 TASK_QUEUE_ENABLE=0 PYTHONUNBUFFERED=1
export VLLM_USE_V2_MODEL_RUNNER=1 VLLM_ASCEND_ENABLE_FA3=1 VLLM_ASCEND_ENABLE_STEP5_FP8=1
export VLLM_ASCEND_TARGET_EXACT_SEQ_LENS=0 VLLM_ASCEND_MTP_EXACT_SEQ_LENS=0
export VLLM_WORKER_MULTIPROC_METHOD=spawn VLLM_ENGINE_READY_TIMEOUT_S=1800
export VLLM_KV_CACHE_LAYOUT=BLNHC VLLM_HOST_IP="$LOCAL_IP" HCCL_IF_IP="$LOCAL_IP"
export HCCL_SOCKET_IFNAME="$COMM_IFACE" GLOO_SOCKET_IFNAME="$COMM_IFACE" TP_SOCKET_IFNAME="$COMM_IFACE"
export HCCL_EXEC_TIMEOUT=240 HCCL_CONNECT_TIMEOUT=240 HCCL_BUFFSIZE=1024 HCCL_BUFFSIZE_EP=4096
export HCCL_OP_EXPANSION_MODE=CCU_MS HCCL_IF_BASE_PORT=55000
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True TRITON_ALL_BLOCKS_PARALLEL=1 VLLM_USE_FASTOKENS=1
export VLLM_PLUGINS=ascend,ascend_model,ascend_kv_connector,ascend_model_loader
export VLLM_SERVER_DEV_MODE=1
# === TENT + memfabric ===
source /usr/local/memfabric_hybrid/set_env.sh
# source /usr/local/Ascend/ascend-toolkit/set_env.sh
export MC_USE_TENT=1
export MOONCAKE_LOCAL_HOSTNAME="$LOCAL_IP"
export MC_TENT_CONF=./mooncake_run/memfabric_tent.json
export MF_PROTOCOL=device_urma
export ASCEND_MF_TRANSFER_PROTOCOL=device_urma
export MEMFABRIC_LIB_DIR=/opt/upgrade/.venv/lib/python3.12/site-packages/memfabric_hybrid/lib
export MOONCAKE_LIB_DIR=/opt/upgrade/.venv/lib/python3.12/site-packages/mooncake
export LD_LIBRARY_PATH="$MEMFABRIC_LIB_DIR:$MOONCAKE_LIB_DIR:${LD_LIBRARY_PATH:-}"

# === Store 池化 ===
export MOONCAKE_CONFIG_PATH=./mooncake_run/mooncake.json
export PYTHONHASHSEED=0
# export MC_STORE_USE_HUGEPAGE=1
# export MC_STORE_HUGEPAGE_SIZE=2MB
export ASCEND_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
mkdir -p "$TORCH_EXTENSIONS_DIR" "$XDG_CACHE_HOME" "$TRITON_CACHE_DIR" "$VLLM_CACHE_ROOT"
ulimit -l unlimited || true
SPECULATIVE_CONFIG_JSON='{"method":"mtp","num_speculative_tokens":3,"enforce_eager":false,"rejection_sample_method":"synthetic","synthetic_acceptance_length":2.8}'
PROFILER_CONFIG_JSON='{"profiler":"torch","torch_profiler_dir":"/run-cluster/profiler139","torch_profiler_with_stack":true,"torch_profiler_record_shapes":true,"torch_profiler_with_memory":false,"ignore_frontend":true,"capture_torch_profiler":false,"max_iterations":128}'
args=("$MODEL_PATH" --trust-remote-code --served-model-name step5
  --load-format safetensors --safetensors-load-strategy prefetch --quantization ascend --dtype bfloat16
  --distributed-executor-backend mp --data-parallel-backend mp
  --tensor-parallel-size 4 --pipeline-parallel-size 1 --enable-expert-parallel
  --all2all-backend allgather_reducescatter
  --data-parallel-size 8 --data-parallel-size-local 2
  --nnodes 4 --node-rank "$NODE_ID" --master-addr "$MASTER_IP" --master-port 29990
  --data-parallel-address "$MASTER_IP" --data-parallel-rpc-port 13790
  --max-model-len 1048576 --max-num-seqs 32 --max-num-batched-tokens 4096
  --gpu-memory-utilization 0.85 --block-size 128 --enable-prefix-caching --async-scheduling
  --speculative-config "$SPECULATIVE_CONFIG_JSON"
  --profiler-config "$PROFILER_CONFIG_JSON"
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY","max_cudagraph_capture_size":512}'
  --additional-config '{"enable_cpu_binding":true,"enable_shared_expert_dp":true,"multistream_overlap_shared_expert":true,"enable_force_eplb":false}'
  --kv-transfer-config \
    '{
        "kv_connector": "MultiConnector",
        "kv_role": "kv_consumer",
        "engine_id":"dp8-D139",
        "kv_load_failure_policy":"fail",
        "kv_connector_extra_config": {
            "connectors": [
                {
                    "kv_connector": "MooncakeConnectorV2",
                    "kv_role": "kv_consumer",
                    "kv_port": "22490",
                },
                {
                    "kv_connector": "AscendStoreConnector",
                    "kv_role": "kv_consumer",
                    "kv_connector_extra_config": {
                        "backend": "mooncake",
                        "lookup_rpc_port": "1",
                        "use_layerwise": false,
                        "load_async": true
                    }
                }
            ]
        }
    }'
  --api-server-count 2 --data-parallel-hybrid-lb --data-parallel-start-rank "$DP_START_RANK"
  --host "$LOCAL_IP" --port 19390)
cd /tmp
exec "$PYTHON_BIN" -B -m vllm.entrypoints.cli.main serve "${args[@]}"