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
from time import sleep

import torch
import torch_npu


g_local_type = "npu"

# the block size of 128 token, the model is deepseek r1 w8a8
layers_num = 61
k_size = 128 * 1024
v_size = 16 * 1024
k_sizes = [k_size for _ in range(layers_num)]
v_sizes = [v_size for _ in range(layers_num)]
layers_block_size = [item for pair in zip(k_sizes, v_sizes) for item in pair]
key_prefix: str = "key_"
PRINT_DATA_SUM = True
# Number of layers reduced by one checksum operation. A larger value reduces the
# number of checksum operations but increases the temporary NPU memory usage.
# Set it to 61 to reduce all layers at once, which is equivalent to the original checksum behavior.
CHECKSUM_AGG_LAYER_NUM = 1
MISALIGNED_ACCESS = True


def set_device(device_id):
    import acl

    acl.init()
    ret = acl.rt.set_device(device_id)
    if ret != 0:
        raise RuntimeError("acl set device failed")


def tensor_sum(tensor: torch.Tensor):
    if tensor is None:
        return 0
    if CHECKSUM_AGG_LAYER_NUM <= 0:
        raise ValueError("CHECKSUM_AGG_LAYER_NUM must be positive")

    checksum = 0
    layer_num = tensor.shape[0]
    for start_layer in range(0, layer_num, CHECKSUM_AGG_LAYER_NUM):
        end_layer = min(start_layer + CHECKSUM_AGG_LAYER_NUM, layer_num)
        checksum += int(tensor[start_layer:end_layer].sum(dtype=torch.int64).item())
    return checksum


def allocate_aligned_tensor(shape, dtype=torch.float32, alignment=2 * 1024 * 1024):
    num_elements = torch.prod(torch.tensor(shape)).item()
    element_size = torch.finfo(dtype).bits // 8 if dtype.is_floating_point else torch.iinfo(dtype).bits // 8
    total_bytes = num_elements * element_size

    padding = alignment - 1
    buffer = torch.randint(0, 256, (total_bytes + padding,), dtype=dtype, device=g_local_type)

    address = buffer.data_ptr()
    aligned_address = (address + alignment - 1) & ~(alignment - 1)
    offset = (aligned_address - address) // element_size

    aligned_tensor = buffer[offset : offset + num_elements].view(*shape)
    print(
        f"==== Aligned tensor address: {aligned_tensor.data_ptr():x}, {num_elements=}, "
        f"{element_size=}, {total_bytes=}, {dtype=}, {shape=}"
    )
    return aligned_tensor


def malloc_npu_blocks(min_block_size: int, layer_num: int, block_num: int):
    raw_blocks = allocate_aligned_tensor((layer_num, block_num, min_block_size), torch.uint8)
    torch_npu.npu.current_stream().synchronize()
    return raw_blocks


def get_col_tensors_by_index(tensors, layer_num, block_index):
    block_tensor = []
    for li in range(layer_num):
        block_tensor.append(tensors[li][block_index])
    return block_tensor


def get_col_tensors_ptr_by_index(tensors, layer_num, block_index):
    block_ptrs = []
    for li in range(layer_num):
        block_ptrs.append(tensors[li][block_index].data_ptr())
    return block_ptrs


def allocate_benchmark_buffers(store, data_dim, block_size, batch_size):
    one_dim_tensor = None
    k_tensors = None
    v_tensors = None
    if data_dim == 2:
        k_tensors = malloc_npu_blocks(max(k_sizes, default=0), len(k_sizes), batch_size)
        v_tensors = malloc_npu_blocks(max(v_sizes, default=0), len(v_sizes), batch_size)
        store.register_buffer(k_tensors.data_ptr(), max(k_sizes, default=0) * len(k_sizes) * batch_size)
        store.register_buffer(v_tensors.data_ptr(), max(v_sizes, default=0) * len(v_sizes) * batch_size)
    else:
        one_dim_tensor = malloc_npu_blocks(max(block_size, default=0), 1, batch_size)
        store.register_buffer(one_dim_tensor.data_ptr(), max(block_size, default=0) * batch_size)
    return one_dim_tensor, k_tensors, v_tensors


def unregister_benchmark_buffers(store, data_dim, block_size, batch_size, one_dim_tensor, k_tensors, v_tensors):
    if data_dim == 2:
        buffers = (
            (k_tensors, max(k_sizes, default=0) * len(k_sizes) * batch_size),
            (v_tensors, max(v_sizes, default=0) * len(v_sizes) * batch_size),
        )
    else:
        buffers = ((one_dim_tensor, max(block_size, default=0) * batch_size),)

    for tensor, size in buffers:
        ret = store.unregister_buffer(tensor.data_ptr(), size)
        if ret != 0:
            raise RuntimeError(f"Failed to unregister buffer ptr:{tensor.data_ptr()} size:{size}, error:{ret}")


def build_batch_arguments(device_id, call_indices, batch_size, data_dim, block_size, one_dim_tensor, k_tensors,
                          v_tensors):
    keys_list = []
    buffs_list = []
    sizes_list = []
    for call_index in call_indices:
        keys = []
        buffs = []
        sizes = []
        for batch_index in range(batch_size):
            keys.append(f"{key_prefix}{device_id}_{call_index}_{batch_index}")
            if data_dim == 2:
                block_buffs = [
                    item
                    for pair in zip(
                        get_col_tensors_ptr_by_index(k_tensors, len(k_sizes), batch_index),
                        get_col_tensors_ptr_by_index(v_tensors, len(v_sizes), batch_index),
                    )
                    for item in pair
                ]
                sizes.append(layers_block_size)
            else:
                block_buffs = get_col_tensors_ptr_by_index(one_dim_tensor, 1, batch_index)
                sizes.append(block_size)
            buffs.append(block_buffs)
        keys_list.append(keys)
        buffs_list.append(buffs)
        sizes_list.append(sizes)
    return keys_list, buffs_list, sizes_list


def calculate_total_size(sizes_list):
    return sum(sum(size) for sizes in sizes_list for size in sizes)


def init_mooncake(device_id: int):
    from mooncake_store import Mooncakestore, MooncakeConfig

    config = MooncakeConfig(
        device=device_id,
        protocol='rdma',
        device_name='',
        local_hostname='192.168.1.2',  # Change to your local IP
        metadata_server='P2PHANDSHAKE',
        global_segment_size=1024 * 1024 * 1024 * 64,
        local_buffer_size=128 * 1024 * 1024,
        master_server_address='192.168.1.1:50051',
    )  # Change to your master server
    store = Mooncakestore(config)
    return store


def write_worker(*args):
    device_id = args[0]
    batch_size = args[1]
    block_size = [args[2]]
    call_count = args[3]
    warmup_count = args[4]
    data_dim = args[5]
    backend = args[6]
    global g_local_type
    g_local_type = args[7]
    process_count = args[8]
    sync = args[9]

    set_device(device_id)
    print(f"npu:{device_id} 开始，PID: {os.getpid()}")
    if backend == "mooncake":
        store = init_mooncake(device_id)
        print(f"==== Start to init mooncake device:{device_id}")
    else:
        from memcache_hybrid import DistributedObjectStore, L2G, G2L, G2H

        store = DistributedObjectStore()
        print(f"==== Start to init memcache device:{device_id}")
        res = store.init(device_id)
        if res != 0:
            raise f"Failed to start pid:{os.getpid()} deviceId:{device_id}"
    print(f"==== Success to init device:{device_id}")
    one_dim_tensor = None
    k_tensors = None
    v_tensors = None
    k_sum = 0
    v_sum = 0
    one_dim_sum = 0
    if data_dim == 2:
        k_tensors = malloc_npu_blocks(max(k_sizes, default=0), len(k_sizes), batch_size)
        v_tensors = malloc_npu_blocks(max(v_sizes, default=0), len(v_sizes), batch_size)
        store.register_buffer(k_tensors.data_ptr(), max(k_sizes, default=0) * len(k_sizes) * batch_size)
        store.register_buffer(v_tensors.data_ptr(), max(v_sizes, default=0) * len(v_sizes) * batch_size)
        if PRINT_DATA_SUM:
            k_sum = tensor_sum(k_tensors)
            v_sum = tensor_sum(v_tensors)
    else:
        one_dim_tensor = malloc_npu_blocks(max(block_size, default=0), 1, batch_size)
        store.register_buffer(one_dim_tensor.data_ptr(), max(block_size, default=0) * batch_size)
        if PRINT_DATA_SUM:
            one_dim_sum = tensor_sum(one_dim_tensor)
    keys_list = []
    buffs_list = []
    sizes_list = []
    for i in range(call_count):
        keys = []
        buffs = []
        sizes = []
        for j in range(batch_size):
            key = key_prefix + str(device_id) + '_' + str(i) + '_' + str(j)
            keys.append(key)
            if data_dim == 2:
                block_buffs = [
                    item
                    for pair in zip(
                        get_col_tensors_ptr_by_index(k_tensors, len(k_sizes), j),
                        get_col_tensors_ptr_by_index(v_tensors, len(v_sizes), j),
                    )
                    for item in pair
                ]
                sizes.append(layers_block_size)
            else:
                block_buffs = get_col_tensors_ptr_by_index(one_dim_tensor, 1, j)
                sizes.append(block_size)
            buffs.append(block_buffs)
        keys_list.append(keys)
        buffs_list.append(buffs)
        sizes_list.append(sizes)

    for warmup_index in range(warmup_count):
        keys = [f"warmup_{device_id}_{warmup_index}_{batch_index}" for batch_index in range(batch_size)]
        buffs = buffs_list[warmup_index % call_count]
        sizes = sizes_list[warmup_index % call_count]
        write_ret = store.batch_put_from_layers(keys, buffs, sizes, L2G)
        if any(x != 0 for x in write_ret):
            raise RuntimeError(f"warmup put failed for device {device_id}")
        remove_ret = store.remove_batch(keys)
        if any(x != 0 for x in remove_ret):
            raise RuntimeError(f"warmup remove failed for device {device_id}")

    sync.wait()
    start = time.perf_counter()
    for keys, buffs, sizes in zip(keys_list, buffs_list, sizes_list):
        write_ret = store.batch_put_from_layers(keys, buffs, sizes, L2G)
        if any(x != 0 for x in write_ret):
            raise f"Failed to put pid:{os.getpid()} deviceId:{device_id}"
    end = time.perf_counter()
    sync.wait()
    duration_us = (end - start) * 1_000_000
    total_size_bytes = sum(sum(size) for size in sizes for sizes in sizes_list)
    total_size_gb = total_size_bytes / (1024 * 1024 * 1024)
    total_duration_seconds = duration_us / 1_000_000
    bandwidth_gb_per_sec = total_size_gb / total_duration_seconds
    print(
        f"\033[91mdevice_id:{device_id} write_total_size:{total_size_bytes} bytes, "
        f"single_size:{total_size_bytes / call_count:.0f} bytes, call count:{call_count}, "
        f"warmup count:{warmup_count}, "
        f"total_time:{duration_us:.2f} us, avg_time:{duration_us / call_count:.2f} us, "
        f"bw:{bandwidth_gb_per_sec:.3f} GB/s\033[0m\n"
    )

    sleep(1)
    if PRINT_DATA_SUM:
        print(f"write data sum for check data ==> {device_id=} {k_sum=}, {v_sum}, {one_dim_sum}\n")
    sleep(1)
    print(f"===== npu:{device_id} write finish, wait read testing ......\n")
    sleep(30 * 60)


def read_worker(*args):
    device_id = args[0]
    batch_size = args[1]
    block_size = [args[2]]
    call_count = args[3]
    warmup_count = args[4]
    data_dim = args[5]
    backend = args[6]
    global g_local_type
    g_local_type = args[7]
    process_count = args[8]
    sync = args[9]

    set_device(device_id)
    print(f"npu:{device_id} 开始，PID: {os.getpid()}")
    if backend == "mooncake":
        store = init_mooncake(device_id)
        print(f"==== Start to init mooncake device:{device_id}")
    else:
        from memcache_hybrid import DistributedObjectStore, L2G, G2L, G2H

        store = DistributedObjectStore()
        print(f"==== Start to init memcache device:{device_id}")
        res = store.init(device_id)
        if res != 0:
            raise f"Failed to start pid:{os.getpid()} deviceId:{device_id}"
    print(f"==== Success to init device:{device_id}")
    one_dim_tensor = None
    k_tensors = None
    v_tensors = None
    k_sum = 0
    v_sum = 0
    one_dim_sum = 0
    if data_dim == 2:
        read_size = max(k_sizes, default=0) * len(k_sizes) + max(v_sizes, default=0) * len(v_sizes)
        k_tensors = malloc_npu_blocks(max(k_sizes, default=0), len(k_sizes), batch_size)
        v_tensors = malloc_npu_blocks(max(v_sizes, default=0), len(v_sizes), batch_size)
        store.register_buffer(k_tensors.data_ptr(), max(k_sizes, default=0) * len(k_sizes) * batch_size)
        store.register_buffer(v_tensors.data_ptr(), max(v_sizes, default=0) * len(v_sizes) * batch_size)
    else:
        read_size = max(block_size, default=0)
        one_dim_tensor = malloc_npu_blocks(max(block_size, default=0), 1, batch_size)
        store.register_buffer(one_dim_tensor.data_ptr(), max(block_size, default=0) * batch_size)
    if g_local_type == "npu":
        direct_t = G2L
    else:
        direct_t = G2H
    # 实测此步骤很耗时，提前准备
    keys_list = []
    buffs_list = []
    sizes_list = []
    # 错开读取位置，防止流量都到一个打到numa
    start_index = (math.ceil((1 << 30) / read_size) * device_id) % call_count if MISALIGNED_ACCESS else 0
    for i in range(start_index, call_count):
        keys = []
        buffs = []
        sizes = []
        for j in range(batch_size):
            key = key_prefix + str(device_id) + '_' + str(i) + '_' + str(j)
            keys.append(key)
            if data_dim == 2:
                block_buffs = [
                    item
                    for pair in zip(
                        get_col_tensors_ptr_by_index(k_tensors, len(k_sizes), j),
                        get_col_tensors_ptr_by_index(v_tensors, len(v_sizes), j),
                    )
                    for item in pair
                ]
                sizes.append(layers_block_size)
            else:
                block_buffs = get_col_tensors_ptr_by_index(one_dim_tensor, 1, j)
                sizes.append(block_size)
            buffs.append(block_buffs)

        keys_list.append(keys)
        buffs_list.append(buffs)
        sizes_list.append(sizes)

    for i in range(start_index):
        keys = []
        buffs = []
        sizes = []
        for j in range(batch_size):
            key = key_prefix + str(device_id) + '_' + str(i) + '_' + str(j)
            keys.append(key)
            if data_dim == 2:
                block_buffs = [
                    item
                    for pair in zip(
                        get_col_tensors_ptr_by_index(k_tensors, len(k_sizes), j),
                        get_col_tensors_ptr_by_index(v_tensors, len(v_sizes), j),
                    )
                    for item in pair
                ]
                sizes.append(layers_block_size)
            else:
                block_buffs = get_col_tensors_ptr_by_index(one_dim_tensor, 1, j)
                sizes.append(block_size)
            buffs.append(block_buffs)
        keys_list.append(keys)
        buffs_list.append(buffs)
        sizes_list.append(sizes)

    for keys, buffs, sizes in zip(keys_list, buffs_list, sizes_list):
        read_ret = store.batch_get_into_layers(keys, buffs, sizes, direct_t)
        if any(x != 0 for x in read_ret):
            raise f"Failed to get pid:{os.getpid()} deviceId:{device_id}"

    sync.wait()
    start = time.perf_counter()
    for keys, buffs, sizes in zip(keys_list, buffs_list, sizes_list):
        read_ret = store.batch_get_into_layers(keys, buffs, sizes, direct_t)
        if any(x != 0 for x in read_ret):
            raise f"Failed to get pid:{os.getpid()} deviceId:{device_id}"
    end = time.perf_counter()
    sync.wait()
    duration_us = (end - start) * 1_000_000
    total_size_bytes = sum(sum(size) for size in sizes for sizes in sizes_list)
    total_size_gb = total_size_bytes / (1024 * 1024 * 1024)
    total_duration_seconds = duration_us / 1_000_000
    bandwidth_gb_per_sec = total_size_gb / total_duration_seconds
    if data_dim == 2 and PRINT_DATA_SUM:
        k_sum = tensor_sum(k_tensors)
        v_sum = tensor_sum(v_tensors)
    elif PRINT_DATA_SUM:
        one_dim_sum = tensor_sum(one_dim_tensor)

    print(
        f"\033[91mdevice_id:{device_id} read_total_size:{total_size_bytes} bytes, "
        f"single_size:{total_size_bytes / call_count:.0f} bytes, call count:{call_count}, "
        f"total_time:{duration_us:.2f} us, avg_time:{duration_us / call_count:.2f} us, "
        f"bw:{bandwidth_gb_per_sec:.3f} GB/s\033[0m\n"
    )

    sleep(1)
    if PRINT_DATA_SUM:
        print(f"read data sum for check data ==> {device_id=} {k_sum=}, {v_sum}, {one_dim_sum}")
    sleep(10)


def write_read_worker(*args):
    device_id = args[0]
    batch_size = args[1]
    block_size = [args[2]]
    call_count = args[3]
    warmup_count = args[4]
    data_dim = args[5]
    backend = args[6]
    global g_local_type
    g_local_type = args[7]
    sync = args[9]

    set_device(device_id)
    print(f"npu:{device_id} write_read开始，PID: {os.getpid()}")
    write_direction = 0
    read_to_npu_direction = 1
    read_to_cpu_direction = 3
    if backend == "mooncake":
        store = init_mooncake(device_id)
        print(f"==== Start to init mooncake device:{device_id}")
    else:
        from memcache_hybrid import DistributedObjectStore, G2H, G2L, L2G

        store = DistributedObjectStore()
        print(f"==== Start to init memcache device:{device_id}")
        res = store.init(device_id)
        if res != 0:
            raise RuntimeError(f"Failed to start pid:{os.getpid()} deviceId:{device_id}")
        write_direction = L2G
        read_to_npu_direction = G2L
        read_to_cpu_direction = G2H
    print(f"==== Success to init write_read device:{device_id}")

    # Ensure all local services are registered before MetaService starts allocating objects for the write phase.
    sync.wait()

    src_one_dim, src_k, src_v = allocate_benchmark_buffers(store, data_dim, block_size, batch_size)
    src_k_sum = tensor_sum(src_k) if data_dim == 2 and PRINT_DATA_SUM else 0
    src_v_sum = tensor_sum(src_v) if data_dim == 2 and PRINT_DATA_SUM else 0
    src_one_dim_sum = tensor_sum(src_one_dim) if data_dim != 2 and PRINT_DATA_SUM else 0
    write_keys, write_buffs, write_sizes = build_batch_arguments(
        device_id,
        range(call_count),
        batch_size,
        data_dim,
        block_size,
        src_one_dim,
        src_k,
        src_v,
    )

    for warmup_index in range(warmup_count):
        keys = [f"warmup_{device_id}_{warmup_index}_{batch_index}" for batch_index in range(batch_size)]
        buffs = write_buffs[warmup_index % call_count]
        sizes = write_sizes[warmup_index % call_count]
        write_ret = store.batch_put_from_layers(keys, buffs, sizes, write_direction)
        if any(result != 0 for result in write_ret):
            raise RuntimeError(f"warmup put failed for device {device_id}")
        remove_ret = store.remove_batch(keys)
        if any(result != 0 for result in remove_ret):
            raise RuntimeError(f"warmup remove failed for device {device_id}")

    sync.wait()
    write_start = time.perf_counter()
    for keys, buffs, sizes in zip(write_keys, write_buffs, write_sizes):
        write_ret = store.batch_put_from_layers(keys, buffs, sizes, write_direction)
        if any(result != 0 for result in write_ret):
            raise RuntimeError(f"Failed to put pid:{os.getpid()} deviceId:{device_id}")
    write_end = time.perf_counter()
    # batch_put_from_layers waits for its copy tasks. This barrier makes every process wait until all writers finish.
    sync.wait()

    total_size_bytes = calculate_total_size(write_sizes)
    total_size_gb = total_size_bytes / (1024 * 1024 * 1024)
    write_duration_us = (write_end - write_start) * 1_000_000
    write_bandwidth = total_size_gb / (write_duration_us / 1_000_000)
    write_result = (
        f"\033[91mdevice_id:{device_id} write_total_size:{total_size_bytes} bytes, "
        f"single_size:{total_size_bytes / call_count:.0f} bytes, call count:{call_count}, "
        f"warmup count:{warmup_count}, total_time:{write_duration_us:.2f} us, "
        f"avg_time:{write_duration_us / call_count:.2f} us, bw:{write_bandwidth:.3f} GB/s\033[0m"
    )

    if backend != "mooncake":
        unregister_benchmark_buffers(
            store,
            data_dim,
            block_size,
            batch_size,
            src_one_dim,
            src_k,
            src_v,
        )
        del write_buffs
        src_one_dim = None
        src_k = None
        src_v = None

    dst_one_dim, dst_k, dst_v = allocate_benchmark_buffers(store, data_dim, block_size, batch_size)
    read_size = (
        max(k_sizes, default=0) * len(k_sizes) + max(v_sizes, default=0) * len(v_sizes)
        if data_dim == 2
        else max(block_size, default=0)
    )
    start_index = (math.ceil((1 << 30) / read_size) * device_id) % call_count if MISALIGNED_ACCESS else 0
    read_order = list(range(start_index, call_count)) + list(range(start_index))
    read_keys, read_buffs, read_sizes = build_batch_arguments(
        device_id,
        read_order,
        batch_size,
        data_dim,
        block_size,
        dst_one_dim,
        dst_k,
        dst_v,
    )
    direct_t = read_to_npu_direction if g_local_type == "npu" else read_to_cpu_direction

    for keys, buffs, sizes in zip(read_keys, read_buffs, read_sizes):
        read_ret = store.batch_get_into_layers(keys, buffs, sizes, direct_t)
        if any(result != 0 for result in read_ret):
            raise RuntimeError(f"Failed to warm up get pid:{os.getpid()} deviceId:{device_id}")

    sync.wait()
    read_start = time.perf_counter()
    for keys, buffs, sizes in zip(read_keys, read_buffs, read_sizes):
        read_ret = store.batch_get_into_layers(keys, buffs, sizes, direct_t)
        if any(result != 0 for result in read_ret):
            raise RuntimeError(f"Failed to get pid:{os.getpid()} deviceId:{device_id}")
    read_end = time.perf_counter()
    sync.wait()

    read_duration_us = (read_end - read_start) * 1_000_000
    read_bandwidth = total_size_gb / (read_duration_us / 1_000_000)
    dst_k_sum = tensor_sum(dst_k) if data_dim == 2 and PRINT_DATA_SUM else 0
    dst_v_sum = tensor_sum(dst_v) if data_dim == 2 and PRINT_DATA_SUM else 0
    dst_one_dim_sum = tensor_sum(dst_one_dim) if data_dim != 2 and PRINT_DATA_SUM else 0
    read_result = (
        f"\033[91mdevice_id:{device_id} read_total_size:{total_size_bytes} bytes, "
        f"single_size:{total_size_bytes / call_count:.0f} bytes, call count:{call_count}, "
        f"total_time:{read_duration_us:.2f} us, avg_time:{read_duration_us / call_count:.2f} us, "
        f"bw:{read_bandwidth:.3f} GB/s\033[0m"
    )
    if PRINT_DATA_SUM:
        print(
            f"write data sum for check data ==> {device_id=} {src_k_sum=}, {src_v_sum=}, "
            f"{src_one_dim_sum=}\n"
        )
        print(
            f"read data sum for check data ==> {device_id=} {dst_k_sum=}, {dst_v_sum=}, "
            f"{dst_one_dim_sum=}"
        )
        if (src_k_sum, src_v_sum, src_one_dim_sum) != (dst_k_sum, dst_v_sum, dst_one_dim_sum):
            raise RuntimeError(f"Checksum mismatch pid:{os.getpid()} deviceId:{device_id}")
    print(write_result)
    print(read_result)
    sleep(10)
