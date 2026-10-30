# Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
# MemFabric_Hybrid is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

import ctypes
import io
import json
import types
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

import example
import interface as bench

BLOCK = 4096
PAYLOAD = 3 * BLOCK


def pointer(value):
    return value.value if isinstance(value, ctypes.c_void_p) else value


class FakeAcl:
    def __init__(self, library=None):
        self.library = "fake-libascendcl.so"
        self.lib = self
        self.allocations, self.pending, self.events = {}, [], []
        self.noop, self.fail_copy = False, False

    def require(self, name):
        return getattr(self, name)

    def aclInit(self, config):
        self.events.append("init")
        return 0

    def aclFinalize(self):
        self.events.append("finalize")
        return 0

    def aclrtSetDevice(self, device):
        return 0

    def aclrtResetDevice(self, device):
        self.events.append("reset")
        return 0

    def aclrtCreateStream(self, output):
        ctypes.cast(output, ctypes.POINTER(ctypes.c_void_p))[0] = 123
        return 0

    def aclrtDestroyStream(self, stream):
        self.events.append("destroy_stream")
        return 0

    def aclrtMalloc(self, output, size, policy):
        storage = ctypes.create_string_buffer(size)
        address = ctypes.addressof(storage)
        self.allocations[address] = storage
        ctypes.cast(output, ctypes.POINTER(ctypes.c_void_p))[0] = address
        return 0

    def aclrtMallocHost(self, output, size):
        return self.aclrtMalloc(output, size, 0)

    def aclrtFree(self, address):
        self.events.append("free")
        del self.allocations[pointer(address)]
        return 0

    def aclrtFreeHost(self, address):
        return self.aclrtFree(address)

    def aclrtHostRegister(self, address, size, kind, output):
        ctypes.cast(output, ctypes.POINTER(ctypes.c_void_p))[0] = address + 0x10000000
        return 0

    def aclrtHostUnregister(self, address):
        self.events.append("unregister")
        return 0

    def aclrtMemcpy(self, dst, capacity, src, size, kind):
        ctypes.memmove(dst, src, size)
        return 0

    def aclrtMemcpyAsync(self, dst, capacity, src, size, kind, stream):
        if self.fail_copy:
            return 42
        if not self.noop:
            self.pending.append((dst, src, size))
        return 0

    def aclrtSynchronizeStream(self, stream):
        self.events.append("sync")
        for dst, src, size in self.pending:
            ctypes.memmove(dst, src, size)
        self.pending.clear()
        return 0

    def aclrtMemset(self, dst, capacity, value, size):
        ctypes.memset(dst, value, size)
        return 0

    def aclrtMemcpyBatch(self, dsts, capacities, srcs, sizes, count, attrs, indexes, attr_count, fail_index):
        self.events.append("batch")
        for index in range(count):
            ctypes.memmove(dsts[index], srcs[index], sizes[index])
        return 0


class FakeBmHandle:
    def __init__(self, size, events):
        self.storage = ctypes.create_string_buffer(size)
        self.ptr = ctypes.addressof(self.storage)
        self.gva = self.ptr + (1 << 48)
        self.events, self.registrations = events, []

    def join(self):
        return 0

    def leave(self):
        self.events.append("bm_leave")
        return 0

    def destroy(self):
        self.events.append("bm_destroy")

    def peer_rank_ptr(self, rank, mem_type):
        return self.gva

    def gva_to_va(self, gva, mem_type):
        return self.ptr

    def register(self, address, size):
        self.registrations.append(address)
        return 0

    def unregister(self, address):
        self.events.append("bm_unregister")
        self.registrations.remove(address)
        return 0

    def copy_data(self, src, dst, size, kind, flags):
        self.events.append(("bm_copy", kind))
        if src >= self.gva:
            src = src - self.gva + self.ptr
        if dst >= self.gva:
            dst = dst - self.gva + self.ptr
        ctypes.memmove(dst, src, size)
        return 0

    def copy_data_batch(self, srcs, dsts, sizes, count, kind, flags):
        for src, dst, size in zip(srcs, dsts, sizes):
            self.copy_data(src, dst, size, kind, flags)
        return 0


class BenchmarkTests(unittest.TestCase):
    def setUp(self):
        self.acl = FakeAcl()
        self.patch = patch.object(bench, "Acl", return_value=self.acl)
        self.patch.start()
        self.addCleanup(self.patch.stop)

    def test_raw_apis_both_directions_and_layouts(self):
        with bench.Runtime() as runtime:
            host, device = runtime.allocate_host(PAYLOAD, "mmap"), runtime.allocate_device(PAYLOAD)
            for api in ("acl_sync", "acl_async", "acl_batch"):
                for layout in ("contiguous", "shuffled"):
                    for src, dst in ((host, device), (device, host)):
                        with self.subTest(api=api, layout=layout, source=src.kind):
                            plan = runtime.prepare_copy(
                                src=src, dst=dst, block_size=BLOCK, batch_size=3, api=api, layout=layout
                            )
                            result = example.benchmark(plan, warmup=1, iterations=2, rounds=2)
                            self.assertTrue(result["verified"])
                            self.assertGreater(result["bandwidth_gbps"], 0)
                            self.assertEqual(len(result["samples_batch_latency_us"]), 2)

    def test_async_run_completes_pending_work(self):
        with bench.Runtime() as runtime:
            host, device = runtime.allocate_host(BLOCK, "mmap"), runtime.allocate_device(BLOCK)
            runtime.fill(host, 23, BLOCK)
            runtime.copy(src=host, dst=device, api="acl_async")
            self.assertEqual(runtime.read(device), bytearray([23] * BLOCK))
            self.assertEqual(self.acl.pending, [])

    def test_pinned_registration_uses_host_pointer_and_releases_in_order(self):
        with bench.Runtime() as runtime:
            host = runtime.allocate_host(BLOCK, register=True)
            self.assertEqual(host.ptr % BLOCK, 0)
            self.assertNotEqual(host.ptr, host.device_alias)
            runtime.fill(host, 17, BLOCK)
            self.assertEqual(runtime.read(host), bytearray([17] * BLOCK))
        self.assertLess(self.acl.events.index("unregister"), self.acl.events.index("free"))
        self.assertLess(self.acl.events.index("free"), self.acl.events.index("destroy_stream"))
        self.assertEqual(self.acl.allocations, {})

    def test_cleanup_drains_stream_before_free_after_exception(self):
        with self.assertRaisesRegex(RuntimeError, "interrupted"):
            with bench.Runtime() as runtime:
                host, device = runtime.allocate_host(BLOCK), runtime.allocate_device(BLOCK)
                self.acl.aclrtMemcpyAsync(device.ptr, BLOCK, host.ptr, BLOCK, 1, runtime.stream)
                raise RuntimeError("interrupted")
        self.assertLess(self.acl.events.index("sync"), self.acl.events.index("free"))
        self.assertEqual(self.acl.pending, [])

    def test_noop_after_warmup_fails_verification(self):
        with bench.Runtime() as runtime:
            host, device = runtime.allocate_host(BLOCK, "mmap"), runtime.allocate_device(BLOCK)
            plan = runtime.prepare_copy(src=host, dst=device, block_size=BLOCK)
            original = plan.run
            calls = 0

            def lose_measured_copy():
                nonlocal calls
                calls += 1
                self.acl.noop = calls > 1
                original()

            with patch.object(plan, "run", side_effect=lose_measured_copy):
                with (
                    self.assertLogs(bench.LOGGER, "ERROR"),
                    self.assertRaisesRegex(bench.BenchmarkError, "verification"),
                ):
                    example.benchmark(plan, warmup=1, iterations=1, rounds=1)

    def test_missing_batch_api_does_not_fallback(self):
        with bench.Runtime() as runtime:
            host, device = runtime.allocate_host(BLOCK, "mmap"), runtime.allocate_device(BLOCK)
            with patch.object(self.acl, "require", side_effect=bench.BenchmarkError("missing batch")):
                with self.assertRaisesRegex(bench.BenchmarkError, "missing batch"):
                    runtime.prepare_copy(src=host, dst=device, block_size=BLOCK, api="acl_batch")

    def test_native_copy_failure_propagates_and_drains_stream(self):
        with bench.Runtime() as runtime:
            host, device = runtime.allocate_host(BLOCK, "mmap"), runtime.allocate_device(BLOCK)
            self.acl.fail_copy = True
            with self.assertLogs(bench.LOGGER, "ERROR"), self.assertRaisesRegex(bench.BenchmarkError, "42"):
                runtime.copy(src=host, dst=device, api="acl_async")
            self.assertIn("sync", self.acl.events)

    def test_batch_failure_does_not_execute_another_api(self):
        with bench.Runtime() as runtime:
            host, device = runtime.allocate_host(BLOCK, "mmap"), runtime.allocate_device(BLOCK)
            plan = runtime.prepare_copy(src=host, dst=device, block_size=BLOCK, api="acl_batch")
            with patch.object(self.acl, "aclrtMemcpyBatch", return_value=19):
                with self.assertLogs(bench.LOGGER, "ERROR"), self.assertRaisesRegex(bench.BenchmarkError, "19"):
                    plan.run()
            self.assertEqual(self.acl.pending, [])

    def test_registration_failure_releases_allocations(self):
        with patch.object(self.acl, "aclrtHostRegister", return_value=21):
            with self.assertLogs(bench.LOGGER, "ERROR"), self.assertRaisesRegex(bench.BenchmarkError, "21"):
                with bench.Runtime() as runtime:
                    runtime.allocate_host(BLOCK, register=True)
        self.assertEqual(self.acl.allocations, {})
        self.assertNotIn("unregister", self.acl.events)

    def test_invalid_inputs_are_rejected_before_copy(self):
        with bench.Runtime() as runtime:
            host, device = runtime.allocate_host(BLOCK, "mmap"), runtime.allocate_device(BLOCK)
            for block, batch in ((0, 1), (BLOCK, 0), (1, 1), (BLOCK, 2), (True, 1)):
                with self.subTest(block=block, batch=batch):
                    with self.assertLogs(bench.LOGGER, "ERROR"), self.assertRaises(bench.BenchmarkError):
                        runtime.prepare_copy(src=host, dst=device, block_size=block, batch_size=batch)

    def test_closed_runtime_and_foreign_buffers_are_rejected(self):
        with bench.Runtime() as runtime:
            host, device = runtime.allocate_host(BLOCK, "mmap"), runtime.allocate_device(BLOCK)
            plan = runtime.prepare_copy(src=host, dst=device, block_size=BLOCK)
            foreign = bench.Buffer(host.ptr, BLOCK, "host", "mmap", bench.Runtime())
            with self.assertLogs(bench.LOGGER, "ERROR"), self.assertRaises(bench.BenchmarkError):
                runtime.copy(src=foreign, dst=device)
        with self.assertLogs(bench.LOGGER, "ERROR"), self.assertRaises(bench.BenchmarkError):
            plan.run()

    def test_mapped_registration_kernel_constraint(self):
        with bench.Runtime() as runtime:
            with patch.object(bench.platform, "release", return_value="5.10.0-test"):
                with self.assertLogs(bench.LOGGER, "ERROR"), self.assertRaisesRegex(bench.BenchmarkError, "pinned"):
                    runtime.allocate_host(BLOCK, "mmap", register=True)

    def test_runtime_initialization_failure_rolls_back(self):
        with patch.object(self.acl, "aclrtCreateStream", return_value=17):
            with self.assertLogs(bench.LOGGER, "ERROR"), self.assertRaises(bench.BenchmarkError):
                with bench.Runtime():
                    self.fail("initialization should fail")
        self.assertEqual(self.acl.events, ["init", "reset", "finalize"])

    def test_bm_pool_uses_gva_for_mf_and_va_for_acl(self):
        with bench.Runtime() as runtime:
            events = self.acl.events
            module = types.ModuleType("memfabric_hybrid")
            handle = FakeBmHandle(2 << 20, events)
            module.initialize = lambda: 0
            module.uninitialize = lambda: events.append("mf_uninit")
            module.set_log_level = lambda level: 0
            module.get_last_err_msg = lambda: ""
            module.bm = types.SimpleNamespace(
                BmConfig=types.SimpleNamespace,
                BmDataOpType=types.SimpleNamespace(DEVICE_RDMA=1, DEVICE_URMA=2),
                BmMemType=types.SimpleNamespace(HOST=0, LOCAL_HOST=2),
                BmCopyType=types.SimpleNamespace(G2L=3, L2G=4),
                initialize=lambda *args: 0,
                uninitialize=lambda flags: events.append("bm_uninit"),
                create2=lambda **kwargs: handle,
            )
            with patch.dict("sys.modules", {"memfabric_hybrid": module}):
                host = runtime.allocate_host(PAYLOAD, "mf_pool")
                device = runtime.allocate_device(PAYLOAD)
                for api in ("mf_copy", "mf_batch", "acl_async"):
                    for src, dst in ((host, device), (device, host)):
                        with self.subTest(api=api, source=src.kind):
                            plan = runtime.prepare_copy(src=src, dst=dst, block_size=BLOCK, batch_size=3, api=api)
                            self.assertTrue(example.benchmark(plan, warmup=1, iterations=2, rounds=1)["verified"])
                self.assertEqual(handle.registrations, [device.ptr])
        self.assertLess(events.index("bm_unregister"), events.index("bm_leave"))
        self.assertLess(events.index("bm_leave"), events.index("bm_destroy"))

    def test_size_parsing_and_overflow(self):
        self.assertEqual(example.parse_sizes("4K:16K"), [BLOCK, 2 * BLOCK, 4 * BLOCK])
        self.assertEqual(example.parse_sizes("1M,4KB,1M"), [BLOCK, 1 << 20])
        for value in ("0", "3K", "1M:4K", "4K:8K:16K", "", "1T", str(1 << 128)):
            with self.subTest(value=value):
                with self.assertLogs(bench.LOGGER, "ERROR"), self.assertRaises(bench.BenchmarkError):
                    example.parse_sizes(value)

    def test_cli_emits_verified_json(self):
        output = io.StringIO()
        with redirect_stdout(output):
            ret = example.main(
                [
                    "--host-alloc",
                    "mmap",
                    "--block-size",
                    "4K",
                    "--batch-size",
                    "3",
                    "--api",
                    "acl_sync,acl_async,acl_batch",
                    "--warmup-count",
                    "0",
                    "--copy-count",
                    "1",
                    "--rounds",
                    "1",
                    "--json",
                ]
            )
        rows = [json.loads(line) for line in output.getvalue().splitlines()]
        self.assertEqual(ret, 0)
        self.assertEqual(len(rows), 6)
        self.assertTrue(all(row["verified"] for row in rows))

    def test_cli_rejects_invalid_configuration_before_acl_initialization(self):
        cases = (
            ["--api", "invalid"],
            ["--batch-size", "0"],
            ["--warmup-count", "-1"],
            ["--copy-count", "0"],
            ["--api", "mf_batch"],
            ["--host-alloc", "mf_pool", "--host-register"],
            ["--block-size", "1G", "--batch-size", "5"],
        )
        for argv in cases:
            with self.subTest(argv=argv), self.assertLogs(level="ERROR"):
                self.assertEqual(example.main(argv), 1)
        self.assertEqual(self.acl.events, [])


if __name__ == "__main__":
    unittest.main()
