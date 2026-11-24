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
import math
import os
import time
from dataclasses import dataclass
from time import sleep

import torch
import torch_npu

from model_profiles import DEFAULT_MODEL, TOKENS_PER_BLOCK, get_key_prefix, get_model_sizes


g_local_type = "npu"
PRINT_DATA_SUM = True
# Limit the temporary memory used by each checksum reduction.
CHECKSUM_AGG_LAYER_NUM = 1
MISALIGNED_ACCESS = True
GIB_BYTES = 1024**3
MICROSECONDS_PER_SECOND = 1_000_000
WRITE_HOLD_SECONDS = 30 * 60
READ_HOLD_SECONDS = 10


def set_device(device_id):
    import acl

    acl.init()
    ret = acl.rt.set_device(device_id)
    if ret != 0:
        raise RuntimeError("acl set device failed")


def tensor_sum(tensor: torch.Tensor):
    if CHECKSUM_AGG_LAYER_NUM <= 0:
        raise ValueError("CHECKSUM_AGG_LAYER_NUM must be positive")
    checksum = 0
    for start_layer in range(0, tensor.shape[0], CHECKSUM_AGG_LAYER_NUM):
        end_layer = min(start_layer + CHECKSUM_AGG_LAYER_NUM, tensor.shape[0])
        checksum += int(tensor[start_layer:end_layer].sum(dtype=torch.int64).item())
    return checksum


def allocate_aligned_tensor(shape, dtype=torch.float32, alignment=2 * 1024 * 1024):
    num_elements = math.prod(shape)
    element_size = torch.finfo(dtype).bits // 8 if dtype.is_floating_point else torch.iinfo(dtype).bits // 8
    total_bytes = num_elements * element_size
    buffer = torch.randint(0, 256, (total_bytes + alignment - 1,), dtype=dtype, device=g_local_type)
    address = buffer.data_ptr()
    aligned_address = (address + alignment - 1) & ~(alignment - 1)
    offset = (aligned_address - address) // element_size
    aligned_tensor = buffer[offset : offset + num_elements].view(*shape)
    print(f"==== Aligned tensor address: {aligned_tensor.data_ptr():x}, {total_bytes=}, {shape=}")
    return aligned_tensor


def malloc_npu_blocks(min_block_size, layer_num, block_num):
    tensor = allocate_aligned_tensor((layer_num, block_num, min_block_size), torch.uint8)
    if g_local_type == "npu":
        torch_npu.npu.current_stream().synchronize()
    return tensor


@dataclass
class BenchmarkBuffers:
    tensors: list
    segment_indices: list
    sizes: list

    def pointers(self, batch_index):
        return [self.tensors[group][layer][batch_index].data_ptr() for group, layer in self.segment_indices]

    def checksums(self):
        return tuple(tensor_sum(tensor) for tensor in self.tensors) if PRINT_DATA_SUM else ()

    def unregister(self, store):
        for tensor in self.tensors:
            size = tensor.numel() * tensor.element_size()
            ret = store.unregister_buffer(tensor.data_ptr(), size)
            if ret != 0:
                raise RuntimeError(f"Failed to unregister buffer ptr:{tensor.data_ptr()} size:{size}, error:{ret}")


def allocate_benchmark_buffers(store, data_dim, block_size, batch_size, model=DEFAULT_MODEL):
    sizes = get_model_sizes(model) if data_dim == 2 else [block_size]
    # Equal-sized segments share a tensor and registration, retaining the R1 allocation layout.
    group_sizes = []
    group_counts = []
    group_by_size = {}
    segment_indices = []
    for size in sizes:
        if size not in group_by_size:
            group_by_size[size] = len(group_sizes)
            group_sizes.append(size)
            group_counts.append(0)
        group = group_by_size[size]
        segment_indices.append((group, group_counts[group]))
        group_counts[group] += 1
    tensors = []
    for size, count in zip(group_sizes, group_counts):
        tensor = malloc_npu_blocks(size, count, batch_size)
        ret = store.register_buffer(tensor.data_ptr(), size * count * batch_size)
        if ret not in (None, 0):
            raise RuntimeError(f"Failed to register benchmark buffer, error:{ret}")
        tensors.append(tensor)
    print(
        f"==== model:{model if data_dim == 2 else 'contiguous'}, data_dim:{data_dim}, "
        f"tokens_per_block:{TOKENS_PER_BLOCK if data_dim == 2 else 'n/a'}, "
        f"segments_per_key:{len(sizes)}, bytes_per_key:{sum(sizes)}, "
        f"cache_dtype:{'bf16-payload' if data_dim == 2 else 'raw-bytes'}"
    )
    return BenchmarkBuffers(tensors, segment_indices, sizes)


def build_batch_arguments(device_id, call_indices, batch_size, buffers, key_prefix):
    keys_list = []
    buffs_list = []
    sizes_list = []
    batch_pointers = [buffers.pointers(index) for index in range(batch_size)]
    for call_index in call_indices:
        keys_list.append([f"{key_prefix}{device_id}_{call_index}_{index}" for index in range(batch_size)])
        buffs_list.append(batch_pointers)
        sizes_list.append([buffers.sizes for _ in range(batch_size)])
    return keys_list, buffs_list, sizes_list


def calculate_total_size(sizes_list):
    return sum(sum(size) for sizes in sizes_list for size in sizes)


def init_mooncake(device_id):
    from mooncake_store import Mooncakestore, MooncakeConfig

    config = MooncakeConfig(
        device=device_id,
        protocol="rdma",
        device_name="",
        local_hostname="192.168.1.2",  # Change to your local IP.
        metadata_server="P2PHANDSHAKE",
        global_segment_size=64 * GIB_BYTES,
        local_buffer_size=128 * 1024 * 1024,
        master_server_address="192.168.1.1:50051",
    )
    return Mooncakestore(config)


def init_benchmark_store(device_id, backend, local_type):
    global g_local_type
    g_local_type = local_type
    set_device(device_id)
    if backend == "mooncake":
        # The Mooncake wrapper does not use the direction argument.
        return init_mooncake(device_id), 0, 1
    from memcache_hybrid import DistributedObjectStore, G2H, G2L, H2G, L2G

    store = DistributedObjectStore()
    ret = store.init(device_id)
    if ret != 0:
        raise RuntimeError(f"Failed to initialize pid:{os.getpid()} device:{device_id}, error:{ret}")
    return store, L2G if local_type == "npu" else H2G, G2L if local_type == "npu" else G2H


def put_batches(store, arguments, direction):
    for keys, buffs, sizes in zip(*arguments):
        results = store.batch_put_from_layers(keys, buffs, sizes, direction)
        if len(results) != len(keys) or any(result != 0 for result in results):
            raise RuntimeError(f"Failed to put keys:{keys}, results:{results}")


def get_batches(store, arguments, direction):
    for keys, buffs, sizes in zip(*arguments):
        results = store.batch_get_into_layers(keys, buffs, sizes, direction)
        if len(results) != len(keys) or any(result != 0 for result in results):
            raise RuntimeError(f"Failed to get keys:{keys}, results:{results}")


def warmup_put(store, arguments, direction, warmup_count):
    keys_list, buffs_list, sizes_list = arguments
    for index in range(warmup_count):
        batch_index = index % len(keys_list)
        keys = [f"warmup_{index}_{key}" for key in keys_list[batch_index]]
        put_batches(store, ([keys], [buffs_list[batch_index]], [sizes_list[batch_index]]), direction)
        results = store.remove_batch(keys)
        if len(results) != len(keys) or any(result != 0 for result in results):
            raise RuntimeError(f"Failed to remove warmup keys:{keys}, results:{results}")


def format_result(device_id, phase, total_bytes, call_count, duration, warmup_count=0):
    duration_us = duration * MICROSECONDS_PER_SECOND
    bandwidth = total_bytes / GIB_BYTES / duration
    return (
        f"device_id:{device_id} {phase}_total_size:{total_bytes} bytes, "
        f"single_size:{total_bytes / call_count:.0f} bytes, call count:{call_count}, "
        f"warmup count:{warmup_count}, total_time:{duration_us:.2f} us, "
        f"avg_time:{duration_us / call_count:.2f} us, bw:{bandwidth:.3f} GiB/s"
    )


def get_read_order(device_id, call_count, bytes_per_key):
    start = (math.ceil(GIB_BYTES / bytes_per_key) * device_id) % call_count if MISALIGNED_ACCESS else 0
    return list(range(start, call_count)) + list(range(start))


def write_worker(
    device_id,
    batch_size,
    block_size,
    call_count,
    warmup_count,
    data_dim,
    backend,
    local_type,
    process_count,
    sync,
    model=DEFAULT_MODEL,
):
    store, write_direction, _ = init_benchmark_store(device_id, backend, local_type)
    buffers = allocate_benchmark_buffers(store, data_dim, block_size, batch_size, model)
    checksum = buffers.checksums()
    prefix = get_key_prefix(model, data_dim)
    arguments = build_batch_arguments(device_id, range(call_count), batch_size, buffers, prefix)
    warmup_put(store, arguments, write_direction, warmup_count)
    sync.wait()
    start = time.perf_counter()
    put_batches(store, arguments, write_direction)
    duration = time.perf_counter() - start
    sync.wait()
    print(format_result(device_id, "write", calculate_total_size(arguments[2]), call_count, duration, warmup_count))
    if PRINT_DATA_SUM:
        print(f"write data sum for check data ==> device_id:{device_id}, checksums:{checksum}")
    print(f"===== device:{device_id} write finish, wait read testing ......")
    sleep(WRITE_HOLD_SECONDS)


def read_worker(
    device_id,
    batch_size,
    block_size,
    call_count,
    warmup_count,
    data_dim,
    backend,
    local_type,
    process_count,
    sync,
    model=DEFAULT_MODEL,
):
    store, _, read_direction = init_benchmark_store(device_id, backend, local_type)
    buffers = allocate_benchmark_buffers(store, data_dim, block_size, batch_size, model)
    prefix = get_key_prefix(model, data_dim)
    order = get_read_order(device_id, call_count, sum(buffers.sizes))
    arguments = build_batch_arguments(device_id, order, batch_size, buffers, prefix)
    get_batches(store, arguments, read_direction)
    sync.wait()
    start = time.perf_counter()
    get_batches(store, arguments, read_direction)
    duration = time.perf_counter() - start
    sync.wait()
    print(format_result(device_id, "read", calculate_total_size(arguments[2]), call_count, duration, call_count))
    if PRINT_DATA_SUM:
        print(f"read data sum for check data ==> device_id:{device_id}, checksums:{buffers.checksums()}")
    sleep(READ_HOLD_SECONDS)


def write_read_worker(
    device_id,
    batch_size,
    block_size,
    call_count,
    warmup_count,
    data_dim,
    backend,
    local_type,
    process_count,
    sync,
    model=DEFAULT_MODEL,
):
    store, write_direction, read_direction = init_benchmark_store(device_id, backend, local_type)
    # Wait for local services before allocating objects in the global pool.
    sync.wait()
    prefix = get_key_prefix(model, data_dim)
    source = allocate_benchmark_buffers(store, data_dim, block_size, batch_size, model)
    source_checksums = source.checksums()
    write_arguments = build_batch_arguments(device_id, range(call_count), batch_size, source, prefix)
    warmup_put(store, write_arguments, write_direction, warmup_count)
    sync.wait()
    start = time.perf_counter()
    put_batches(store, write_arguments, write_direction)
    write_duration = time.perf_counter() - start
    sync.wait()
    total_bytes = calculate_total_size(write_arguments[2])
    write_result = format_result(device_id, "write", total_bytes, call_count, write_duration, warmup_count)
    if backend != "mooncake":
        source.unregister(store)
        del source
        del write_arguments

    destination = allocate_benchmark_buffers(store, data_dim, block_size, batch_size, model)
    order = get_read_order(device_id, call_count, sum(destination.sizes))
    read_arguments = build_batch_arguments(device_id, order, batch_size, destination, prefix)
    get_batches(store, read_arguments, read_direction)
    sync.wait()
    start = time.perf_counter()
    get_batches(store, read_arguments, read_direction)
    read_duration = time.perf_counter() - start
    sync.wait()
    destination_checksums = destination.checksums()
    if PRINT_DATA_SUM:
        print(f"write data sum for check data ==> device_id:{device_id}, checksums:{source_checksums}")
        print(f"read data sum for check data ==> device_id:{device_id}, checksums:{destination_checksums}")
        if source_checksums != destination_checksums:
            raise RuntimeError(f"Checksum mismatch pid:{os.getpid()} device:{device_id}")
    print(write_result)
    print(format_result(device_id, "read", total_bytes, call_count, read_duration, call_count))
    sleep(READ_HOLD_SECONDS)
