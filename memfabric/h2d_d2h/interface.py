# Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
# MemFabric_Hybrid is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

"""Python interfaces for ACL/MemFabric memory allocation and completed copies."""

import ctypes
import ctypes.util
import logging
import mmap
import os
import platform
import random
import re
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path

LOGGER = logging.getLogger(__name__)
ALIGNMENT = 4096
HOST_ALLOCATORS = ("mmap", "aligned", "acl_pinned", "mf_pool")
COPY_APIS = ("acl_sync", "acl_async", "acl_batch", "mf_copy", "mf_batch")
DEVICE_POLICIES = {"huge_first": 0, "huge_only": 1, "normal_only": 2}


class BenchmarkError(RuntimeError):
    pass


def fail(message, **context):
    detail = " ".join(f"{key}={value!r}" for key, value in context.items())
    LOGGER.error("%s; %s", message, detail)
    raise BenchmarkError(f"{message}; {detail}")


def check(ret, operation, **context):
    if ret != 0:
        fail("native operation failed", operation=operation, ret=ret, **context)


def positive_size(size, name="size"):
    if isinstance(size, bool) or not isinstance(size, int) or size <= 0 or size > ctypes.c_size_t(-1).value:
        fail("expected a positive size_t integer", parameter=name, value=size)
    return size


class MemLocation(ctypes.Structure):
    _fields_ = [("id", ctypes.c_uint32), ("type", ctypes.c_uint32)]


class MemcpyBatchAttr(ctypes.Structure):
    _fields_ = [("dstLoc", MemLocation), ("srcLoc", MemLocation), ("reserved", ctypes.c_uint8 * 16)]


class Acl:
    def __init__(self, library=None):
        if platform.system() != "Linux":
            fail("ACL execution requires Linux with CANN and an NPU", platform=platform.system())
        if library is None:
            root = os.environ.get("ASCEND_HOME_PATH")
            library = str(Path(root) / "lib64/libascendcl.so") if root else "libascendcl.so"
        self.library = str(library)
        self.lib = ctypes.CDLL(self.library)
        ptr, size, ptr_out = ctypes.c_void_p, ctypes.c_size_t, ctypes.POINTER(ctypes.c_void_p)
        self.bind("aclInit", [ctypes.c_char_p])
        self.bind("aclFinalize", [])
        self.bind("aclrtSetDevice", [ctypes.c_int32])
        self.bind("aclrtResetDevice", [ctypes.c_int32])
        self.bind("aclrtCreateStream", [ptr_out])
        self.bind("aclrtDestroyStream", [ptr])
        self.bind("aclrtSynchronizeStream", [ptr])
        self.bind("aclrtMalloc", [ptr_out, size, ctypes.c_uint32])
        self.bind("aclrtFree", [ptr])
        self.bind("aclrtMallocHost", [ptr_out, size])
        self.bind("aclrtFreeHost", [ptr])
        copy_args = [ptr, size, ptr, size, ctypes.c_uint32]
        self.bind("aclrtMemcpy", copy_args)
        self.bind("aclrtMemcpyAsync", [*copy_args, ptr])
        self.bind("aclrtMemset", [ptr, size, ctypes.c_int32, size])
        self.bind("aclrtHostRegister", [ptr, ctypes.c_uint64, ctypes.c_uint32, ptr_out], required=False)
        self.bind("aclrtHostUnregister", [ptr], required=False)
        self.bind("aclrtGetSocName", [], result=ctypes.c_char_p, required=False)
        self.bind(
            "aclrtMemcpyBatch",
            [
                ctypes.POINTER(ptr),
                ctypes.POINTER(size),
                ctypes.POINTER(ptr),
                ctypes.POINTER(size),
                size,
                ctypes.POINTER(MemcpyBatchAttr),
                ctypes.POINTER(size),
                size,
                ctypes.POINTER(size),
            ],
            required=False,
        )

    def bind(self, name, args, result=ctypes.c_int32, required=True):
        function = getattr(self.lib, name, None)
        if function is None and required:
            fail("required ACL symbol is unavailable", symbol=name, library=self.library)
        if function is not None:
            function.argtypes, function.restype = args, result

    def require(self, name):
        function = getattr(self.lib, name, None)
        if function is None:
            fail("requested ACL API is unavailable; no fallback is performed", api=name, library=self.library)
        return function


@dataclass(frozen=True)
class Buffer:
    ptr: int
    nbytes: int
    kind: str
    allocator: str
    runtime: "Runtime"
    storage: object = None
    gva: int = 0
    device_alias: int = 0


class BmPool:
    def __init__(self, runtime, size):
        import memfabric_hybrid as mf
        from memfabric_hybrid import bm

        self.mf, self.bm, self.runtime = mf, bm, runtime
        self.size = (size + (2 << 20) - 1) // (2 << 20) * (2 << 20)
        self.registered = set()
        with ExitStack() as setup:
            check(mf.set_log_level(3), "mf.set_log_level")
            check(mf.initialize(), "mf.initialize")
            setup.callback(mf.uninitialize)
            cfg = bm.BmConfig()
            cfg.auto_ranking, cfg.rank_id, cfg.start_store = False, 0, True
            check(
                bm.initialize(runtime.store_url, 1, runtime.device, cfg),
                "bm.initialize",
                store=runtime.store_url,
                device=runtime.device,
            )
            setup.callback(bm.uninitialize, 0)
            protocol = getattr(bm.BmDataOpType, runtime.protocol.upper())
            self.handle = bm.create2(
                id=0, local_dram_size=self.size, max_dram_size=self.size, data_op_type=protocol, enable_56bits_gva=True
            )
            if self.handle is None:
                fail("bm.create2 failed", size=self.size, protocol=runtime.protocol, detail=mf.get_last_err_msg())
            setup.callback(self.handle.destroy)
            check(self.handle.join(), "BM join", protocol=runtime.protocol, size=self.size)
            setup.callback(self.leave)
            self.gva = self.handle.peer_rank_ptr(0, bm.BmMemType.HOST)
            self.ptr = self.handle.gva_to_va(self.gva, bm.BmMemType.LOCAL_HOST) if self.gva else 0
            if not self.ptr:
                fail("BM local DRAM address is unavailable", size=self.size, detail=mf.get_last_err_msg())
            runtime.stack.enter_context(setup.pop_all())

    def leave(self):
        check(self.handle.leave(), "BM leave", detail=self.mf.get_last_err_msg())

    def register_device(self, buffer):
        if buffer.ptr not in self.registered:
            check(
                self.handle.register(buffer.ptr, buffer.nbytes),
                "BM register device",
                size=buffer.nbytes,
                detail=self.mf.get_last_err_msg(),
            )
            self.registered.add(buffer.ptr)
            self.runtime.stack.callback(self.unregister_device, buffer.ptr)

    def unregister_device(self, ptr):
        check(self.handle.unregister(ptr), "BM unregister device", detail=self.mf.get_last_err_msg())


class Runtime:
    """Own an ACL runtime, one stream and all allocated buffers until context exit."""

    def __init__(self, device=0, *, library=None, protocol="device_rdma", store_url="tcp://127.0.0.1:12070"):
        if isinstance(device, bool) or not isinstance(device, int) or not 0 <= device < 2**31:
            fail("invalid logical device ID", device=device)
        if protocol not in ("device_rdma", "device_urma"):
            fail("unsupported BM protocol", protocol=protocol)
        self.device, self.library, self.protocol, self.store_url = device, library, protocol, store_url
        self.active, self.entered, self.pool = False, False, None
        self.stack = ExitStack()
        self.stream = ctypes.c_void_p()

    def __enter__(self):
        if self.entered:
            fail("Runtime contexts cannot be entered more than once", device=self.device)
        self.entered = True
        self.acl = Acl(self.library)
        with ExitStack() as setup:
            check(self.acl.lib.aclInit(None), "aclInit", library=self.acl.library)
            setup.callback(self.cleanup, "aclFinalize")
            check(self.acl.lib.aclrtSetDevice(self.device), "aclrtSetDevice", device=self.device)
            setup.callback(self.cleanup, "aclrtResetDevice", self.device)
            check(self.acl.lib.aclrtCreateStream(ctypes.byref(self.stream)), "aclrtCreateStream", device=self.device)
            setup.callback(self.cleanup, "aclrtDestroyStream", self.stream)
            self.stack = setup.pop_all()
        self.active = True
        get_soc = getattr(self.acl.lib, "aclrtGetSocName", None)
        self.soc = (get_soc() or b"").decode("utf-8", errors="replace") if get_soc else None
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.active = False
        with self.stack:
            self.cleanup("aclrtSynchronizeStream", self.stream)

    def cleanup(self, name, *args):
        check(getattr(self.acl.lib, name)(*args), name, device=self.device)

    def require_active(self):
        if not self.active:
            fail("Runtime must be used inside its context", device=self.device)

    def allocate_host(self, size, allocator="acl_pinned", *, register=False):
        self.require_active()
        positive_size(size)
        if allocator not in HOST_ALLOCATORS or size % ALIGNMENT:
            fail("unsupported allocator or size not aligned to 4096 bytes", allocator=allocator, size=size)
        if allocator == "mf_pool" and register:
            fail("BM owns pool registration; explicit ACL registration is unsupported", allocator=allocator)
        kernel = re.match(r"(\d+)\.(\d+)", platform.release())
        if register and allocator != "acl_pinned" and kernel and tuple(map(int, kernel.groups())) <= (5, 10):
            fail(
                "ACL mapped registration requires pinned memory on kernels <= 5.10",
                allocator=allocator,
                kernel=platform.release(),
            )
        storage, alias = None, ctypes.c_void_p()
        if allocator == "mf_pool":
            if self.pool is not None:
                fail("only one BM pool buffer is supported per Runtime", size=size)
            self.pool = BmPool(self, size)
            return Buffer(self.pool.ptr, size, "host", allocator, self, gva=self.pool.gva)
        if allocator == "mmap":
            storage = mmap.mmap(-1, size)
            self.stack.callback(storage.close)
            ptr = ctypes.addressof(ctypes.c_char.from_buffer(storage))
        elif allocator == "aligned":
            libc = ctypes.CDLL(ctypes.util.find_library("c") or "libc.so.6")
            libc.posix_memalign.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t, ctypes.c_size_t]
            libc.posix_memalign.restype = ctypes.c_int
            libc.free.argtypes, libc.free.restype = [ctypes.c_void_p], None
            base = ctypes.c_void_p()
            check(libc.posix_memalign(ctypes.byref(base), ALIGNMENT, size), "posix_memalign", size=size)
            self.stack.callback(libc.free, base)
            ptr, storage = base.value, libc
        else:
            base = ctypes.c_void_p()
            positive_size(size + ALIGNMENT - 1)
            check(self.acl.lib.aclrtMallocHost(ctypes.byref(base), size + ALIGNMENT - 1), "aclrtMallocHost", size=size)
            self.stack.callback(self.cleanup, "aclrtFreeHost", base)
            ptr = (base.value + ALIGNMENT - 1) // ALIGNMENT * ALIGNMENT
        if register:
            function = self.acl.require("aclrtHostRegister")
            self.acl.require("aclrtHostUnregister")
            check(function(ptr, size, 0, ctypes.byref(alias)), "aclrtHostRegister", size=size, allocator=allocator)
            self.stack.callback(self.cleanup, "aclrtHostUnregister", ptr)
        return Buffer(ptr, size, "host", allocator, self, storage=storage, device_alias=alias.value or 0)

    def allocate_device(self, size, policy="huge_first"):
        self.require_active()
        positive_size(size)
        if policy not in DEVICE_POLICIES:
            fail("unsupported device memory policy", policy=policy)
        ptr = ctypes.c_void_p()
        check(
            self.acl.lib.aclrtMalloc(ctypes.byref(ptr), size, DEVICE_POLICIES[policy]),
            "aclrtMalloc",
            size=size,
            policy=policy,
        )
        self.stack.callback(self.cleanup, "aclrtFree", ptr)
        return Buffer(ptr.value, size, "device", f"acl_{policy}", self)

    def prepare_copy(self, *, src, dst, block_size, batch_size=1, api="acl_async", layout="contiguous", seed=0):
        """Prepare address arrays once; run() completes the entire batch before returning."""
        return CopyPlan(self, src, dst, block_size, batch_size, api, layout, seed)

    def copy(self, *, src, dst, size=None, api="acl_sync"):
        size = min(src.nbytes, dst.nbytes) if size is None else size
        self.prepare_copy(src=src, dst=dst, block_size=size, api=api).run()

    def write(self, buffer, data):
        self.require_buffer(buffer)
        if not isinstance(data, (bytes, bytearray)) or not 0 < len(data) <= buffer.nbytes:
            fail("write expects nonempty bytes within buffer capacity", capacity=buffer.nbytes)
        storage = (ctypes.c_char * len(data)).from_buffer_copy(data)
        if buffer.kind == "host":
            ctypes.memmove(buffer.ptr, storage, len(data))
        else:
            check(
                self.acl.lib.aclrtMemcpy(buffer.ptr, buffer.nbytes, ctypes.addressof(storage), len(data), 1),
                "aclrtMemcpy seed H2D",
                size=len(data),
            )

    def read(self, buffer, size=None):
        self.require_buffer(buffer)
        size = buffer.nbytes if size is None else positive_size(size)
        if size > buffer.nbytes:
            fail("read exceeds buffer capacity", size=size, capacity=buffer.nbytes)
        data = bytearray(size)
        storage = (ctypes.c_char * size).from_buffer(data)
        if buffer.kind == "host":
            ctypes.memmove(storage, buffer.ptr, size)
        else:
            check(
                self.acl.lib.aclrtMemcpy(ctypes.addressof(storage), size, buffer.ptr, size, 2),
                "aclrtMemcpy verify D2H",
                size=size,
            )
        return data

    def fill(self, buffer, value, size):
        self.require_buffer(buffer)
        positive_size(size)
        if size > buffer.nbytes or not isinstance(value, int) or not 0 <= value <= 255:
            fail("invalid fill size or byte", size=size, value=value, capacity=buffer.nbytes)
        if buffer.kind == "host":
            ctypes.memset(buffer.ptr, value, size)
        else:
            check(self.acl.lib.aclrtMemset(buffer.ptr, buffer.nbytes, value, size), "aclrtMemset", size=size)

    def require_buffer(self, buffer):
        self.require_active()
        if not isinstance(buffer, Buffer) or buffer.runtime is not self:
            fail("buffer must belong to this Runtime", device=self.device)


class CopyPlan:
    def __init__(self, runtime, src, dst, block_size, batch_size, api, layout, seed):
        runtime.require_buffer(src)
        runtime.require_buffer(dst)
        positive_size(block_size, "block_size")
        positive_size(batch_size, "batch_size")
        if block_size % ALIGNMENT or block_size * batch_size > min(src.nbytes, dst.nbytes):
            fail(
                "copy must use 4096-byte blocks within both buffers",
                block_size=block_size,
                batch_size=batch_size,
                source_capacity=src.nbytes,
                destination_capacity=dst.nbytes,
            )
        if src.kind == dst.kind or api not in COPY_APIS or layout not in ("contiguous", "shuffled"):
            fail("unsupported copy configuration", source=src.kind, destination=dst.kind, api=api, layout=layout)
        self.runtime, self.src, self.dst, self.api, self.layout = runtime, src, dst, api, layout
        self.seed = seed
        self.block_size, self.batch_size, self.length = block_size, batch_size, block_size * batch_size
        self.kind = 1 if src.kind == "host" else 2
        order = list(range(batch_size))
        if layout == "shuffled":
            random.Random(seed).shuffle(order)
        self.sources = [src.ptr + index * block_size for index in order]
        self.destinations = [dst.ptr + index * block_size for index in order]
        self.sizes = [block_size] * batch_size
        self.pool = None
        if api == "acl_batch":
            runtime.acl.require("aclrtMemcpyBatch")
            self.native_sources = (ctypes.c_void_p * batch_size)(*self.sources)
            self.native_destinations = (ctypes.c_void_p * batch_size)(*self.destinations)
            self.native_sizes = (ctypes.c_size_t * batch_size)(*self.sizes)
            host, device = MemLocation(0, 0), MemLocation(runtime.device, 1)
            self.attr = MemcpyBatchAttr(device if self.kind == 1 else host, host if self.kind == 1 else device)
            self.attr_index, self.fail_index = ctypes.c_size_t(0), ctypes.c_size_t(0)
        elif api.startswith("mf_"):
            host_buffer, device_buffer = (src, dst) if self.kind == 1 else (dst, src)
            if host_buffer.allocator != "mf_pool" or host_buffer.device_alias:
                fail("MF copies require the local BM DRAM pool and an application device buffer", api=api)
            self.pool = runtime.pool
            self.pool.register_device(device_buffer)
            pool_base = host_buffer.gva
            if self.kind == 1:
                self.sources = [pool_base + index * block_size for index in order]
            else:
                self.destinations = [pool_base + index * block_size for index in order]
            types = self.pool.bm.BmCopyType
            self.mf_type = types.G2L if self.kind == 1 else types.L2G

    def run(self):
        """Submit one batch and wait for completion, including for acl_async."""
        self.runtime.require_active()
        lib = self.runtime.acl.lib
        if self.api in ("acl_sync", "acl_async"):
            function = lib.aclrtMemcpy if self.api == "acl_sync" else lib.aclrtMemcpyAsync
            tail = () if self.api == "acl_sync" else (self.runtime.stream,)
            for index, (src, dst) in enumerate(zip(self.sources, self.destinations)):
                ret = function(dst, self.block_size, src, self.block_size, self.kind, *tail)
                if ret != 0:
                    if self.api == "acl_async":
                        check(lib.aclrtSynchronizeStream(self.runtime.stream), "aclrtSynchronizeStream after failure")
                    check(ret, self.api, block_size=self.block_size, index=index, direction=self.kind)
            if self.api == "acl_async":
                check(lib.aclrtSynchronizeStream(self.runtime.stream), "aclrtSynchronizeStream", batch=self.batch_size)
        elif self.api == "acl_batch":
            self.fail_index.value = self.batch_size
            ret = lib.aclrtMemcpyBatch(
                self.native_destinations,
                self.native_sizes,
                self.native_sources,
                self.native_sizes,
                self.batch_size,
                ctypes.byref(self.attr),
                ctypes.byref(self.attr_index),
                1,
                ctypes.byref(self.fail_index),
            )
            check(
                ret,
                "aclrtMemcpyBatch",
                fail_index=self.fail_index.value,
                block_size=self.block_size,
                batch=self.batch_size,
                direction=self.kind,
            )
        elif self.api == "mf_batch":
            ret = self.pool.handle.copy_data_batch(
                self.sources, self.destinations, self.sizes, self.batch_size, self.mf_type, 0
            )
            if ret != 0:
                check(
                    ret,
                    self.api,
                    block_size=self.block_size,
                    batch=self.batch_size,
                    detail=self.pool.mf.get_last_err_msg(),
                )
        else:
            for index, (src, dst) in enumerate(zip(self.sources, self.destinations)):
                ret = self.pool.handle.copy_data(src, dst, self.block_size, self.mf_type, 0)
                if ret != 0:
                    check(
                        ret, self.api, index=index, block_size=self.block_size, detail=self.pool.mf.get_last_err_msg()
                    )
