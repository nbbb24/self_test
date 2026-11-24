#!/usr/bin/env python
# coding=utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
# MemCache_Hybrid is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

import argparse
import multiprocessing as mp
import os

from model_profiles import DEFAULT_MODEL, MODEL_CHOICES


def create_parser():
    parser = argparse.ArgumentParser(description="Benchmark MemCache put/get with synthetic cache layouts.")
    parser.add_argument("testcase", choices=("read", "write", "write_read_same_process"))
    parser.add_argument("process_count", type=int)
    parser.add_argument("batch_size", type=int)
    parser.add_argument("block_size", type=int)
    parser.add_argument("call_count", type=int)
    parser.add_argument("warmup_count", type=int)
    parser.add_argument("data_dim", type=int, choices=(1, 2))
    parser.add_argument("backend", choices=("memcache", "mooncake"))
    parser.add_argument("local_type", choices=("npu", "cpu"))
    parser.add_argument(
        "-m",
        "--model",
        type=str.lower,
        choices=MODEL_CHOICES,
        default=DEFAULT_MODEL,
        help="128-token BF16 cache profile for data_dim=2; V4/V4.1 include only global prefix cache",
    )
    return parser


def run_worker(worker, worker_args, sync):
    try:
        worker(*worker_args)
    except BaseException:
        # Release peers waiting at a barrier if initialization or a transfer fails.
        sync.abort()
        raise


def run_workers(worker, args):
    sync = mp.Barrier(args.process_count)
    base_device = int(os.environ.get("MMC_DEVICE_ID", 0))
    processes = []
    common_args = (
        args.batch_size,
        args.block_size,
        args.call_count,
        args.warmup_count,
        args.data_dim,
        args.backend,
        args.local_type,
        args.process_count,
        sync,
        args.model,
    )
    for index in range(args.process_count):
        worker_args = (base_device + index, *common_args)
        process = mp.Process(target=run_worker, args=(worker, worker_args, sync))
        process.start()
        processes.append(process)
    for process in processes:
        process.join()
    failed = [(process.pid, process.exitcode) for process in processes if process.exitcode != 0]
    if failed:
        print(f"{args.testcase} workers failed: {failed}")
        return False
    print(f"{args.testcase} workers finished.")
    return True


def main(argv=None):
    parser = create_parser()
    args = parser.parse_args(argv)
    for name in ("process_count", "batch_size", "block_size", "call_count"):
        if getattr(args, name) <= 0:
            parser.error(f"{name} must be positive")
    if args.warmup_count < 0:
        parser.error("warmup_count must not be negative")
    print(f"parent PID:{os.getpid()}, parameters:{vars(args)}")
    if args.data_dim == 1:
        print("data_dim=1 uses block_size; the model profile applies only to data_dim=2.")
    # Keep --help and parameter validation usable without torch/Ascend installed.
    from mutil_process import read_worker, write_read_worker, write_worker

    workers = {"read": read_worker, "write": write_worker, "write_read_same_process": write_read_worker}
    return 0 if run_workers(workers[args.testcase], args) else 1


if __name__ == "__main__":
    raise SystemExit(main())
