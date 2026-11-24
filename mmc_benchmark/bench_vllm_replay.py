import argparse
import hashlib
import json
import logging
import multiprocessing as mp
import time

from vllm_layout import load_layout

logger = logging.getLogger(__name__)


def replay_worker(device, layout, args, barrier):
    import torch
    import torch_npu
    from memcache_hybrid import G2L, L2G, DistributedObjectStore

    torch_npu.npu.set_device(device)
    store = DistributedObjectStore()
    ret = store.init(device)
    if ret:
        raise RuntimeError(f"ERROR store init failed: device={device}, result={ret}")
    barrier.wait()
    buffers = []
    addresses = []
    for region in layout["regions"]:
        size = region["bytes"]
        raw = torch.randint(0, 256, (size + 2097151,), dtype=torch.uint8, device="npu")
        offset = (-raw.data_ptr()) % 2097152
        tensor = raw[offset : offset + size]
        ret = store.register_buffer(tensor.data_ptr(), size)
        if ret not in (None, 0):
            raise RuntimeError(f"ERROR registration failed: device={device}, bytes={size}, result={ret}")
        buffers.append(tensor)
        addresses.append(tensor.data_ptr())
    torch_npu.npu.synchronize()
    groups = []
    prefix = f"vllm_{layout['sha256'][:16]}_{device}_"
    for group in layout["groups"]:
        pointers = [
            [addresses[s["region"]] + s["offset"] + block * s["stride"] for s in group["segments"]]
            for block in range(group["keys"])
        ]
        sizes = [[s["bytes"] for s in group["segments"]] for _ in range(group["keys"])]
        groups.append((group, pointers, sizes))
    calls = []
    for call in range(args.call_count):
        calls.append(
            [
                ([f"{prefix}{call}_g{g['group']}_b{block}" for block in range(g["keys"])], ptrs, sizes)
                for g, ptrs, sizes in groups
            ]
        )
    total_bytes = args.call_count * sum(sum(map(sum, sizes)) for _, _, sizes in groups)

    def transfer(index, write, warmup=None):
        for keys, ptrs, sizes in calls[index]:
            if warmup is not None:
                keys = [f"warmup_{warmup}_{key}" for key in keys]
            result = (
                store.batch_put_from_layers(keys, ptrs, sizes, L2G)
                if write
                else store.batch_get_into_layers(keys, ptrs, sizes, G2L)
            )
            if len(result) != len(keys) or any(result):
                raise RuntimeError(
                    f"ERROR transfer failed: device={device}, write={write}, keys={keys}, result={result}"
                )
            if warmup is not None:
                result = store.remove_batch(keys)
                if len(result) != len(keys) or any(result):
                    raise RuntimeError(f"ERROR warmup removal failed: device={device}, keys={keys}, result={result}")

    def checksums():
        host = [tensor.cpu().numpy().tobytes() for tensor in buffers]
        values = []
        for group, _, _ in groups:
            digest = hashlib.sha256()
            for block in range(group["keys"]):
                for segment in group["segments"]:
                    start = segment["offset"] + block * segment["stride"]
                    digest.update(host[segment["region"]][start : start + segment["bytes"]])
            values.append(digest.hexdigest())
        return tuple(values)

    src = checksums()
    for index in range(args.write_warmup):
        transfer(index % args.call_count, True, index)
    barrier.wait()
    start = time.perf_counter()
    for index in range(args.call_count):
        transfer(index, True)
    write_duration = time.perf_counter() - start
    barrier.wait()
    for tensor in buffers:
        tensor.zero_()
    torch_npu.npu.synchronize()
    for index in range(args.read_warmup):
        transfer((index + device) % args.call_count, False)
    barrier.wait()
    start = time.perf_counter()
    for index in range(args.call_count):
        transfer((index + device) % args.call_count, False)
    read_duration = time.perf_counter() - start
    barrier.wait()
    dst = checksums()
    if src != dst:
        raise RuntimeError(f"ERROR checksum mismatch: device={device}, source={src}, destination={dst}")
    print(f"write data sum for check data ==> device_id:{device}, checksums:{src}", flush=True)
    print(f"read data sum for check data ==> device_id:{device}, checksums:{dst}", flush=True)
    for phase, duration in (("write", write_duration), ("read", read_duration)):
        print(
            f"device_id:{device} {phase}_total_size:{total_bytes} bytes, "
            f"call count:{args.call_count}, api_calls:{args.call_count * len(groups)}, "
            f"total_time:{duration * 1e6:.2f} us, avg_time:{duration * 1e6 / args.call_count:.2f} us, "
            f"bw:{total_bytes / 1024**3 / duration:.3f} GiB/s",
            flush=True,
        )
    for tensor in buffers:
        ret = store.unregister_buffer(tensor.data_ptr(), tensor.numel())
        if ret:
            raise RuntimeError(f"ERROR unregister failed: device={device}, bytes={tensor.numel()}, result={ret}")
    del tensor, raw
    buffers.clear()
    torch_npu.npu.synchronize()
    torch_npu.npu.empty_cache()
    barrier.wait()
    ret = store.close()
    if ret not in (None, 0):
        raise RuntimeError(f"ERROR store close failed: device={device}, result={ret}")
    time.sleep(10)


def worker_entry(device, layout, args, barrier):
    try:
        replay_worker(device, layout, args, barrier)
    except BaseException:
        logger.exception("ERROR replay worker failed: device=%d, layout=%s", device, args.layout)
        barrier.abort()
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description="Replay vLLM bulk KV group buffers through MemCache.")
    parser.add_argument("--layout", required=True)
    parser.add_argument("--process-count", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--call-count", type=int, default=100)
    parser.add_argument("--write-warmup", type=int, default=10)
    parser.add_argument("--read-warmup", type=int, default=100)
    args = parser.parse_args(argv)
    if min(args.process_count, args.batch_size, args.call_count) <= 0 or min(args.write_warmup, args.read_warmup) < 0:
        parser.error("counts must be positive and warmups must be nonnegative")
    layout = load_layout(args.layout, args.batch_size)
    print(
        json.dumps(
            {
                "parameters": vars(args),
                "layout_sha256": layout["sha256"],
                "metadata": layout["metadata"],
                "groups": [
                    {
                        "group": g["group"],
                        "keys": g["keys"],
                        "segments_per_key": len(g["segments"]),
                        "bytes_per_key": sum(s["bytes"] for s in g["segments"]),
                    }
                    for g in layout["groups"]
                ],
            }
        ),
        flush=True,
    )
    context = mp.get_context("spawn")
    barrier = context.Barrier(args.process_count)
    processes = [
        context.Process(target=worker_entry, args=(device, layout, args, barrier))
        for device in range(args.process_count)
    ]
    for process in processes:
        process.start()
    for process in processes:
        process.join()
    failed = [(p.pid, p.exitcode) for p in processes if p.exitcode]
    if failed:
        logger.error("ERROR replay failed: workers=%s, layout=%s", failed, args.layout)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
