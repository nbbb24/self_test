#!/bin/bash

usage() {
    echo "Usage: bash bench_start.sh [-t read|write|write_read_same_process] [-p processes] [-b batch]"
    echo "       [-s bytes] [-n calls] [-w warmups] [-d 1|2] [-e memcache|mooncake] [-l npu|cpu]"
    echo "       [-m MODEL|--model MODEL]"
    echo "Models: deepseek-r1 (default), glm-5.1, glm-5.2, deepseek-v4-flash, deepseek-v4-pro, deepseek-v4.1-flash"
    echo "Model profiles use 128 original tokens and BF16 bytes; V4/V4.1 include only global prefix cache."
    echo "Selecting -m/--model implies -d 2 unless -d is explicitly supplied. -s applies only to -d 1."
}

TEST_CASE="read"
PROCESS_COUNT=1
BATCH_SIZE=16
BLOCK_SIZE=1048576
CALL_COUNT=100
WARMUP_COUNT=2
DATA_DIM=1
BACKEND="memcache"
LOCAL_TYPE="npu"
MODEL="deepseek-r1"
MODEL_SELECTED=0
DATA_DIM_SELECTED=0

# Normalize long options while retaining getopts support for existing short options.
NORMALIZED_ARGS=()
while (($#)); do
    case "$1" in
    --model)
        if (($# < 2)); then
            echo "--model requires a value" >&2
            exit 1
        fi
        NORMALIZED_ARGS+=(-m "$2")
        shift 2
        ;;
    --model=*)
        NORMALIZED_ARGS+=(-m "${1#*=}")
        shift
        ;;
    --help)
        NORMALIZED_ARGS+=(-h)
        shift
        ;;
    *)
        NORMALIZED_ARGS+=("$1")
        shift
        ;;
    esac
done
set -- "${NORMALIZED_ARGS[@]}"
while getopts ':p:b:s:n:w:d:e:t:l:m:h' OPT; do
    case "$OPT" in
    t) TEST_CASE="$OPTARG" ;;
    p) PROCESS_COUNT="$OPTARG" ;;
    b) BATCH_SIZE="$OPTARG" ;;
    s) BLOCK_SIZE="$OPTARG" ;;
    n) CALL_COUNT="$OPTARG" ;;
    w) WARMUP_COUNT="$OPTARG" ;;
    d) DATA_DIM="$OPTARG"; DATA_DIM_SELECTED=1 ;;
    e) BACKEND="$OPTARG" ;;
    l) LOCAL_TYPE="$OPTARG" ;;
    m) MODEL="$OPTARG"; MODEL_SELECTED=1 ;;
    h) usage; exit 0 ;;
    :) echo "-$OPTARG requires a value" >&2; exit 1 ;;
    ?) usage >&2; exit 1 ;;
    esac
done
shift "$((OPTIND - 1))"
if (($#)); then
    echo "Unexpected argument: $1" >&2
    exit 1
fi
if ((MODEL_SELECTED && !DATA_DIM_SELECTED)); then
    DATA_DIM=2
fi

source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/memcache_hybrid/set_env.sh
source /usr/local/memfabric_hybrid/set_env.sh
export HCOM_SET_LOG_LEVEL=3
# export MMC_LOCAL_CONFIG_PATH=$PWD/mmc-local.conf
# export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
set -x
python3 "$SCRIPT_DIR/run_mutil_process.py" "$TEST_CASE" "$PROCESS_COUNT" "$BATCH_SIZE" "$BLOCK_SIZE" \
    "$CALL_COUNT" "$WARMUP_COUNT" "$DATA_DIM" "$BACKEND" "$LOCAL_TYPE" --model "$MODEL"
