#!/bin/bash
set -eo pipefail
source /home/sxy/hicache_A5_qwen14b_20261023_c39f/runtime.env
source /usr/local/Ascend/ascend-toolkit/latest/set_env.sh
export ASCEND_RT_VISIBLE_DEVICES=4
exec python3 -B "$RUN/supervisor.py"
