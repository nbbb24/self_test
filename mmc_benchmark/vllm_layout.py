import hashlib
import json
from pathlib import Path


def export_layout(worker, output, metadata, group_key_counts):
    regions = []
    storage_ids = {}
    for caches in worker.kv_caches.values():
        for tensor in worker._as_cache_tuple(caches):
            storage = tensor.untyped_storage()
            if storage.data_ptr() not in storage_ids:
                storage_ids[storage.data_ptr()] = len(regions)
                regions.append(
                    {
                        "base": tensor.data_ptr(),
                        "end": tensor.data_ptr()
                        + (1 + sum((n - 1) * s for n, s in zip(tensor.shape, tensor.stride()))) * tensor.element_size(),
                    }
                )
            region = regions[storage_ids[storage.data_ptr()]]
            region["base"] = min(region["base"], tensor.data_ptr())
            region["end"] = max(
                region["end"],
                tensor.data_ptr()
                + (1 + sum((n - 1) * s for n, s in zip(tensor.shape, tensor.stride()))) * tensor.element_size(),
            )
    for region in regions:
        region["bytes"] = region["end"] - region["base"]
    groups = []
    for group_id, (block_start, key_count) in group_key_counts.items():
        segments = []
        for addr, size, stride in zip(
            worker.group_kv_caches_base_addr[group_id],
            worker.group_block_len[group_id],
            worker.group_block_stride[group_id],
        ):
            region_id = next(i for i, r in enumerate(regions) if r["base"] <= addr < r["base"] + r["bytes"])
            segments.append(
                {
                    "region": region_id,
                    "offset": addr - regions[region_id]["base"] + block_start * stride,
                    "bytes": size,
                    "stride": stride,
                }
            )
        groups.append({"group": group_id, "keys": key_count, "segments": segments})
    from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.metadata import (
        ChunkedTokenDatabase,
    )

    database = ChunkedTokenDatabase.__new__(ChunkedTokenDatabase)
    database.block_size = worker.grouped_block_size
    for name in ("group_kv_caches_base_addr", "group_block_len", "group_block_stride", "group_cache_entry_segments"):
        setattr(database, name, getattr(worker, name))
    for group in groups:
        group_id = group["group"]
        block_start, key_count = group_key_counts[group_id]
        block_tokens = worker.grouped_block_size[group_id]
        for index in range(key_count):
            actual_addrs, actual_sizes, _ = database.prepare_value(
                0, block_tokens, [], kv_cache_group_id=group_id, block_id=block_start + index
            )
            replay_addrs = [regions[s["region"]]["base"] + s["offset"] + index * s["stride"] for s in group["segments"]]
            if actual_addrs != replay_addrs or actual_sizes != [s["bytes"] for s in group["segments"]]:
                raise ValueError(f"ERROR connector layout mismatch: group={group_id}, block={index}")
    metadata["prepare_value_verified"] = True
    result = {"schema": 1, "metadata": metadata, "regions": [{"bytes": r["bytes"]} for r in regions], "groups": groups}
    Path(output).write_text(json.dumps(result, indent=2) + "\n")
    return result


def load_layout(filename, batch_size):
    layout = json.loads(Path(filename).read_text())
    if layout.get("schema") != 1 or layout["metadata"]["batch_size"] != batch_size:
        raise ValueError(f"ERROR incompatible layout schema/batch_size: file={filename}, batch_size={batch_size}")
    for group in layout["groups"]:
        if group["keys"] <= 0 or not group["segments"]:
            raise ValueError(f"ERROR empty group: file={filename}, group={group}")
        for segment in group["segments"]:
            if not 0 <= segment["region"] < len(layout["regions"]):
                raise ValueError(f"ERROR invalid region: file={filename}, segment={segment}")
            end = segment["offset"] + (group["keys"] - 1) * segment["stride"] + segment["bytes"]
            if min(segment["offset"], segment["stride"]) < 0 or segment["bytes"] <= 0:
                raise ValueError(f"ERROR invalid segment: file={filename}, segment={segment}")
            if end > layout["regions"][segment["region"]]["bytes"]:
                raise ValueError(f"ERROR segment exceeds region: file={filename}, segment={segment}, end={end}")
    layout["sha256"] = hashlib.sha256(Path(filename).read_bytes()).hexdigest()
    return layout
