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
"""CPU memory contract tests; do not require torch, Ascend or a running MetaService."""

import contextlib
import ctypes
import importlib.util
import io
import itertools
import math
import os
from pathlib import Path
import subprocess
import sys
import types
import unittest
from unittest.mock import Mock, patch

from model_profiles import DEFAULT_MODEL, MODEL_CHOICES, get_key_prefix, get_model_sizes
from run_mutil_process import create_parser, run_worker, run_workers

TEST_BATCH_SIZE = 2
TEST_CALL_COUNT = 2
TEST_BLOCK_BYTES = 256
TEST_WARMUP_COUNT = 1
TEST_PATTERN_STEP = 37
BYTE_VALUES = 256
TEST_WRITE_DIRECTION = 11
TEST_READ_DIRECTION = 22
ALLOCATION_IDS = itertools.count(1)


class CpuTensor:
    def __init__(self, shape, storage=None, offset=0):
        self.shape = shape
        self.storage = storage if storage is not None else (ctypes.c_ubyte * math.prod(shape))()
        self.offset = offset

    def __getitem__(self, index):
        stride = math.prod(self.shape[1:])
        return CpuTensor(self.shape[1:], self.storage, self.offset + index * stride)

    def data_ptr(self):
        return ctypes.addressof(self.storage) + self.offset

    def numel(self):
        return math.prod(self.shape)

    @staticmethod
    def element_size():
        return 1


def allocate_cpu_blocks(size, count, batch):
    tensor = CpuTensor((count, batch, size))
    allocation_id = next(ALLOCATION_IDS)
    for layer in range(count):
        for index in range(batch):
            value = (allocation_id + layer + index * TEST_PATTERN_STEP) % BYTE_VALUES
            ctypes.memset(tensor[layer][index].data_ptr(), value, size)
    return tensor


class MemoryStore:
    def __init__(self, corrupt_reads=False):
        self.objects = {}
        self.registered = {}
        self.corrupt_reads = corrupt_reads
        self.directions = []

    def register_buffer(self, pointer, size):
        self.registered[pointer] = size
        return 0

    def unregister_buffer(self, pointer, size):
        assert self.registered.pop(pointer) == size
        return 0

    def validate(self, pointers, sizes):
        assert len(pointers) == len(sizes)
        for pointer, size in zip(pointers, sizes):
            assert size > 0
            assert any(base <= pointer and pointer + size <= base + length for base, length in self.registered.items())

    def batch_put_from_layers(self, keys, pointers, sizes, direction):
        self.directions.append(direction)
        for key, ptrs, lengths in zip(keys, pointers, sizes):
            self.validate(ptrs, lengths)
            self.objects[key] = b"".join(ctypes.string_at(ptr, size) for ptr, size in zip(ptrs, lengths))
        return [0] * len(keys)

    def batch_get_into_layers(self, keys, pointers, sizes, direction):
        self.directions.append(direction)
        for key, ptrs, lengths in zip(keys, pointers, sizes):
            self.validate(ptrs, lengths)
            data = self.objects[key]
            assert len(data) == sum(lengths)
            offset = 0
            for pointer, size in zip(ptrs, lengths):
                ctypes.memmove(pointer, data[offset : offset + size], size)
                offset += size
            if self.corrupt_reads:
                byte = ctypes.c_ubyte.from_address(ptrs[0])
                byte.value = (byte.value + 1) % BYTE_VALUES
        return [0] * len(keys)

    def remove_batch(self, keys):
        for key in keys:
            del self.objects[key]
        return [0] * len(keys)


def load_workers():
    name = "benchmark_workers_under_test"
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name("mutil_process.py"))
    module = importlib.util.module_from_spec(spec)
    torch_stub = types.SimpleNamespace(Tensor=object, float32=object())
    with patch.dict(sys.modules, {name: module, "torch": torch_stub, "torch_npu": types.SimpleNamespace()}):
        spec.loader.exec_module(module)
    return module


class BenchmarkModelTests(unittest.TestCase):
    def test_profile_payloads(self):
        expected = {
            "deepseek-r1": (122, 8_994_816),
            "glm-5.1": (234, 14_057_472),
            "glm-5.2": (177, 12_189_696),
            "deepseek-v4-flash": (62, 880_640),
            "deepseek-v4-pro": (91, 1_260_544),
            "deepseek-v4.1-flash": (8, 409_600),
        }
        self.assertEqual(set(MODEL_CHOICES), set(expected))
        for model, payload in expected.items():
            with self.subTest(model=model):
                sizes = get_model_sizes(model)
                self.assertEqual((len(sizes), sum(sizes)), payload)
                self.assertTrue(all(size > 0 for size in sizes))
        with self.assertRaises(ValueError):
            get_model_sizes("unknown")

    def test_keys_preserve_legacy_and_isolate_models(self):
        self.assertEqual(get_key_prefix(DEFAULT_MODEL, 2), "key_")
        self.assertEqual(get_key_prefix("glm-5.1", 1), "key_")
        prefixes = [get_key_prefix(model, 2) for model in MODEL_CHOICES]
        self.assertEqual(len(set(prefixes)), len(MODEL_CHOICES))

    def run_memory_worker(self, workers, worker, store, model, data_dim=2):
        with (
            patch.object(workers, "malloc_npu_blocks", side_effect=allocate_cpu_blocks),
            patch.object(
                workers,
                "tensor_sum",
                side_effect=lambda tensor: sum(ctypes.string_at(tensor.data_ptr(), tensor.numel())),
            ),
            patch.object(
                workers, "init_benchmark_store", return_value=(store, TEST_WRITE_DIRECTION, TEST_READ_DIRECTION)
            ),
            patch.object(workers, "sleep"),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            worker(
                0,
                TEST_BATCH_SIZE,
                TEST_BLOCK_BYTES,
                TEST_CALL_COUNT,
                TEST_WARMUP_COUNT,
                data_dim,
                "memcache",
                "cpu",
                1,
                Mock(),
                model,
            )

    def test_all_models_roundtrip_with_actual_cpu_pointers(self):
        for model in MODEL_CHOICES:
            with self.subTest(model=model):
                workers = load_workers()
                store = MemoryStore()
                self.run_memory_worker(workers, workers.write_read_worker, store, model)
                self.assertEqual(len(store.objects), TEST_BATCH_SIZE * TEST_CALL_COUNT)
                self.assertEqual({len(data) for data in store.objects.values()}, {sum(get_model_sizes(model))})
                self.assertEqual(set(store.directions), {TEST_WRITE_DIRECTION, TEST_READ_DIRECTION})
                self.assertFalse(any(key.startswith("warmup_") for key in store.objects))

    def test_separate_write_read_and_contiguous_mode(self):
        for model, data_dim in (("glm-5.2", 2), (DEFAULT_MODEL, 1)):
            with self.subTest(model=model, data_dim=data_dim):
                workers = load_workers()
                store = MemoryStore()
                self.run_memory_worker(workers, workers.write_worker, store, model, data_dim)
                self.run_memory_worker(workers, workers.read_worker, store, model, data_dim)
                expected = sum(get_model_sizes(model)) if data_dim == 2 else TEST_BLOCK_BYTES
                self.assertEqual({len(data) for data in store.objects.values()}, {expected})

    def test_checksum_rejects_corruption(self):
        workers = load_workers()
        with self.assertRaisesRegex(RuntimeError, "Checksum mismatch"):
            self.run_memory_worker(
                workers, workers.write_read_worker, MemoryStore(corrupt_reads=True), "deepseek-v4.1-flash"
            )

    def test_model_reaches_every_child_process(self):
        for testcase in ("read", "write", "write_read_same_process"):
            args = create_parser().parse_args(
                [
                    testcase,
                    "2",
                    "1",
                    "256",
                    "1",
                    "0",
                    "2",
                    "memcache",
                    "npu",
                    "--model",
                    "glm-5.2",
                ]
            )
            process = Mock(exitcode=0)
            with (
                patch("run_mutil_process.mp.Process", return_value=process) as factory,
                patch.dict(os.environ, {"MMC_DEVICE_ID": "3"}),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                self.assertTrue(run_workers(Mock(), args))
            self.assertEqual(factory.call_count, args.process_count)
            for index, call in enumerate(factory.call_args_list):
                child_args = call.kwargs["args"][1]
                self.assertEqual(child_args[0], 3 + index)
                self.assertEqual(child_args[-1], "glm-5.2")

    def test_failure_releases_barrier(self):
        sync = Mock()
        with self.assertRaises(ValueError):
            run_worker(Mock(side_effect=ValueError("failed")), (), sync)
        sync.abort.assert_called_once()

    def test_cpu_and_npu_copy_directions(self):
        workers = load_workers()
        store = Mock()
        store.init.return_value = 0
        flags = types.SimpleNamespace(
            DistributedObjectStore=Mock(return_value=store),
            H2G="host-to-global",
            G2H="global-to-host",
            L2G="npu-to-global",
            G2L="global-to-npu",
        )
        for local_type, expected in (("cpu", (flags.H2G, flags.G2H)), ("npu", (flags.L2G, flags.G2L))):
            with (
                self.subTest(local_type=local_type),
                patch.object(workers, "set_device"),
                patch.dict(sys.modules, {"memcache_hybrid": flags}),
            ):
                actual_store, write_direction, read_direction = workers.init_benchmark_store(0, "memcache", local_type)
                self.assertIs(actual_store, store)
                self.assertEqual((write_direction, read_direction), expected)

    def test_child_failure_is_reported(self):
        args = create_parser().parse_args(["read", "1", "1", "256", "1", "0", "2", "memcache", "npu"])
        with (
            patch("run_mutil_process.mp.Process", return_value=Mock(exitcode=1)),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            self.assertFalse(run_workers(Mock(), args))

    def test_help_and_invalid_model_without_ascend(self):
        script = str(Path(__file__).with_name("run_mutil_process.py"))
        result = subprocess.run([sys.executable, script, "--help"], capture_output=True, text=True, check=False)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("glm-5.2", result.stdout)
        result = subprocess.run(
            [
                sys.executable,
                script,
                "read",
                "1",
                "1",
                "256",
                "1",
                "0",
                "2",
                "memcache",
                "npu",
                "--model",
                "unknown",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("invalid choice", result.stderr)
        self.assertNotIn("ModuleNotFoundError", result.stderr)


if __name__ == "__main__":
    unittest.main()
