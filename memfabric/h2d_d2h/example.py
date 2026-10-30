# Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
# MemFabric_Hybrid is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

"""Run local H2D/D2H examples using the allocation and copy interfaces."""

import argparse
import json
import logging
import platform
import re
import statistics
import time

if __package__:
    from .interface import (
        ALIGNMENT,
        COPY_APIS,
        DEVICE_POLICIES,
        HOST_ALLOCATORS,
        CopyPlan,
        Runtime,
        fail,
        positive_size,
    )
else:
    from interface import ALIGNMENT, COPY_APIS, DEVICE_POLICIES, HOST_ALLOCATORS, CopyPlan, Runtime, fail, positive_size

LOGGER = logging.getLogger(__name__)
SIZE_UNITS = {"": 1, "K": 1024, "M": 1 << 20, "G": 1 << 30}


def benchmark(plan: CopyPlan, *, warmup=5, iterations=20, rounds=3):
    for name, value in (("iterations", iterations), ("rounds", rounds)):
        positive_size(value, name)
    if isinstance(warmup, bool) or not isinstance(warmup, int) or warmup < 0:
        fail("warmup must be a nonnegative integer", warmup=warmup)
    expected = bytearray(bytes(range(256)) * (plan.length // 256))
    for index in range(plan.batch_size):
        offset = index * plan.block_size
        expected[offset : offset + 8] = index.to_bytes(8, "little")
    plan.runtime.write(plan.src, expected)
    for _ in range(warmup):
        plan.run()
    samples = []
    for _ in range(rounds):
        plan.runtime.fill(plan.dst, 0xA5, plan.length)
        start = time.perf_counter_ns()
        for _ in range(iterations):
            plan.run()
        elapsed = time.perf_counter_ns() - start
        if elapsed <= 0:
            fail("nonpositive measurement duration", elapsed_ns=elapsed, api=plan.api)
        actual = plan.runtime.read(plan.dst, plan.length)
        if actual != expected:
            first = next(index for index, (want, got) in enumerate(zip(expected, actual)) if want != got)
            fail(
                "data verification failed",
                api=plan.api,
                first_offset=first,
                expected=expected[first],
                actual=actual[first],
                block_size=plan.block_size,
                batch=plan.batch_size,
            )
        samples.append(elapsed / iterations / 1000)
    latency = statistics.median(samples)
    return {
        "direction": "h2d" if plan.kind == 1 else "d2h",
        "api": plan.api,
        "host_allocator": plan.src.allocator if plan.kind == 1 else plan.dst.allocator,
        "device_allocator": plan.dst.allocator if plan.kind == 1 else plan.src.allocator,
        "device": plan.runtime.device,
        "soc": plan.runtime.soc,
        "acl_library": plan.runtime.acl.library,
        "python_version": platform.python_version(),
        "host_registration": "mf_managed"
        if plan.runtime.pool
        else ("acl_mapped" if (plan.src.device_alias or plan.dst.device_alias) else "none"),
        "layout": plan.layout,
        "seed": plan.seed,
        "block_size": plan.block_size,
        "batch_size": plan.batch_size,
        "bytes_per_batch": plan.length,
        "iterations": iterations,
        "warmup": warmup,
        "rounds": rounds,
        "batch_latency_us": latency,
        "bandwidth_gbps": plan.length / (latency * 1000),
        "samples_batch_latency_us": samples,
        "timing": "host_wall_completion_includes_python",
        "verified": True,
        "native_dispatch": "MF-managed; not traced" if plan.pool else plan.api,
    }


def parse_sizes(value):
    def one(text):
        match = re.fullmatch(r"\s*(\d+)\s*([KMG]?)B?\s*", text.upper())
        if match is None:
            fail("invalid size; use bytes or K/M/G suffix", value=text)
        return positive_size(int(match[1]) * SIZE_UNITS[match[2]])

    if ":" in value:
        parts = value.split(":")
        if len(parts) != 2:
            fail("invalid doubling range", value=value)
        first, last = map(one, parts)
        if first > last:
            fail("range start exceeds end", value=value)
        sizes = []
        while first <= last:
            sizes.append(first)
            first *= 2
    else:
        sizes = sorted({one(part) for part in value.split(",")})
    if any(size % ALIGNMENT for size in sizes):
        fail("sizes must be multiples of 4096 bytes", value=value)
    return sizes


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=0, help="logical device ID under ASCEND_RT_VISIBLE_DEVICES")
    parser.add_argument("--host-alloc", choices=HOST_ALLOCATORS, default="acl_pinned")
    parser.add_argument(
        "--host-register", action="store_true", help="explicit ACL mapped registration; not for mf_pool"
    )
    parser.add_argument("--device-policy", choices=DEVICE_POLICIES, default="huge_first")
    parser.add_argument("--api", default="acl_sync,acl_async", help="comma-separated: " + ",".join(COPY_APIS))
    parser.add_argument("--direction", choices=("h2d", "d2h", "both"), default="both")
    parser.add_argument("--block-size", default="16K:2M", help="value, list, or doubling range")
    parser.add_argument("--batch-size", type=int, help="default: 128 MiB / block size, minimum 1")
    parser.add_argument("--warmup-count", type=int, default=5)
    parser.add_argument("--copy-count", type=int, default=20)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--layout", choices=("contiguous", "shuffled"), default="contiguous")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--protocol", choices=("device_rdma", "device_urma"), default="device_rdma")
    parser.add_argument("--store-url", default="tcp://127.0.0.1:12070", help="used only with mf_pool")
    parser.add_argument("--acl-library", help="explicit libascendcl.so path")
    parser.add_argument("--json", action="store_true", help="emit one JSON object per measurement")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.ERROR, format="%(levelname)s %(message)s")
    try:
        sizes = parse_sizes(args.block_size)
        apis = args.api.split(",")
        if any(api not in COPY_APIS for api in apis):
            fail("unsupported copy API", apis=apis, supported=COPY_APIS)
        if args.host_register and (args.host_alloc == "mf_pool" or any(api.startswith("mf_") for api in apis)):
            fail("explicit ACL registration cannot be combined with MF-owned registration", allocator=args.host_alloc)
        if any(api.startswith("mf_") for api in apis) and args.host_alloc != "mf_pool":
            fail("MF copy APIs require --host-alloc mf_pool", apis=apis, allocator=args.host_alloc)
        for name, value in (("copy_count", args.copy_count), ("rounds", args.rounds)):
            positive_size(value, name)
        if args.warmup_count < 0:
            fail("warmup count must be nonnegative", value=args.warmup_count)
        if args.batch_size is not None:
            positive_size(args.batch_size, "batch_size")
        batches = {size: args.batch_size or max(1, (128 << 20) // size) for size in sizes}
        capacity = max(size * batch for size, batch in batches.items())
        if capacity > 4 << 30:
            fail("benchmark payload exceeds 4 GiB limit", capacity=capacity)
        with Runtime(
            args.device, library=args.acl_library, protocol=args.protocol, store_url=args.store_url
        ) as runtime:
            for api in apis:
                if api == "acl_batch":
                    runtime.acl.require("aclrtMemcpyBatch")
            host = runtime.allocate_host(capacity, args.host_alloc, register=args.host_register)
            device = runtime.allocate_device(capacity, args.device_policy)
            for size in sizes:
                for direction in ("h2d", "d2h") if args.direction == "both" else (args.direction,):
                    src, dst = (host, device) if direction == "h2d" else (device, host)
                    for api in apis:
                        plan = runtime.prepare_copy(
                            src=src,
                            dst=dst,
                            block_size=size,
                            batch_size=batches[size],
                            api=api,
                            layout=args.layout,
                            seed=args.seed,
                        )
                        result = benchmark(
                            plan, warmup=args.warmup_count, iterations=args.copy_count, rounds=args.rounds
                        )
                        result.update(
                            host_registered=args.host_register,
                            protocol=args.protocol if runtime.pool else None,
                            acl_library=runtime.acl.library,
                        )
                        if args.json:
                            print(json.dumps(result), flush=True)
                        else:
                            print(
                                f"{direction:3} {api:10} block={size} batch={batches[size]} "
                                f"{result['bandwidth_gbps']:.3f} GB/s "
                                f"{result['batch_latency_us']:.3f} us/batch verified=PASS",
                                flush=True,
                            )
        return 0
    except Exception as error:
        LOGGER.error("H2D/D2H benchmark failed: %s; configuration=%s", error, vars(args))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
