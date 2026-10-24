#!/bin/bash
set -eo pipefail
source /home/sxy/hicache_A5_llama8b_20261023_b28e/runtime.env
source /usr/local/Ascend/ascend-toolkit/latest/set_env.sh
export ASCEND_RT_VISIBLE_DEVICES=4
exec python3 -B "$RUN/supervisor.py"
