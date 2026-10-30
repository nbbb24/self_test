# Python H2D/D2H benchmark

独立的 Python 内存分配、拷贝封装及本地 H2D/D2H benchmark。本目录独立运行，不修改 `mfcli`、MemFabric 库或其构建系统。
原始 ACL 模式使用标准库 `ctypes` 加载 `libascendcl.so`，无需编译扩展，也不依赖 PyACL、NumPy 或 Torch。
BM 池模式复用已安装的 `memfabric_hybrid` Python 绑定。

## 文件职责

- [`interface.py`](interface.py)：包装内存分配、注册、拷贝、同步和释放，提供 `Runtime`、`Buffer`、`CopyPlan`。
- [`example.py`](example.py)：调用接口模拟 H2D load / D2H offload 的本地搬运流程，组织数据准备、预热、复制、
  完成校验和性能统计，并提供命令行入口。
- [`test_interface.py`](test_interface.py)：接口和示例流程的 CPU 模拟后端回归测试。

`interface.py` 可单独导入复用，不执行示例。`example.py` 使用它完成“分配缓冲区 → 准备块数据 →
调用传输接口 → 等待完成 → 校验 → 释放”的流程；BM 模式中的 Host 缓冲区是真实 BM DRAM 池。

## 环境

- Linux、Python >= 3.10、NPU 驱动和 CANN。先加载 CANN 环境，使 `ASCEND_HOME_PATH` 或动态库搜索路径有效。
- BM 模式额外要求正确安装 MemFabric 及所选协议依赖。`device_rdma` / `device_urma` 需要与机型、版本匹配。
- 本工具拥有 ACL 初始化、设备、stream 和释放的完整生命周期。请在独立进程中运行，勿嵌入已经初始化 ACL、Torch NPU
  或 MemFabric 的进程；同一进程使用一个 `Runtime` 上下文。
- `--device` 是 `ASCEND_RT_VISIBLE_DEVICES` 下的逻辑编号。

## 命令行

在仓库根目录运行。先使用小负载确认接口可用：

```bash
python memfabric/h2d_d2h/example.py \
  --host-alloc acl_pinned --api acl_sync,acl_async \
  --block-size 64K --batch-size 16 --copy-count 5 --rounds 2
```

比较锁页内存上的三个 ACL 接口：

```bash
python memfabric/h2d_d2h/example.py \
  --host-alloc acl_pinned --api acl_sync,acl_async,acl_batch \
  --block-size 16K:2M --json
```

将 `--host-alloc` 换成 `mmap` 或 `aligned` 比较普通内存。一次命令选择一种分配方式，多个接口复用同一对缓冲区。
若显式请求的接口在当前 CANN 中不存在或执行失败，程序记录 ERROR 并以非零退出码结束，不回退到其他接口。

使用真实 BM DRAM 池，同时比较池内存上的 ACL 路径和 MF 路径：

```bash
source /path/to/memfabric_hybrid/set_env.sh
python memfabric/h2d_d2h/example.py \
  --host-alloc mf_pool --protocol device_rdma \
  --api acl_async,mf_copy,mf_batch \
  --block-size 16K:2M --layout shuffled --json
```

此场景创建 `world_size=1` 的 BM DRAM 池，应用 HBM 由 ACL 分配：

```text
D2H / offload: 应用 HBM -> 本 rank BM DRAM 池，MF 使用 L2G
H2D / load:    本 rank BM DRAM 池 -> 应用 HBM，MF 使用 G2L
```

ACL 使用池的本地 Host VA；MF 使用同一池的 GVA，并注册应用 HBM。它保留真实池内存、地址转换和 MF 调用路径，
结果代表本地搬运；跨 rank 或跨机的完整路径仍应使用 `mfcli benchmark` 测量。

## Python 接口

从仓库根目录启动 Python，或将此目录加入模块搜索路径。以下接口已经实现：

```python
from memfabric.h2d_d2h.interface import Runtime
from memfabric.h2d_d2h.example import benchmark

with Runtime(device=0) as runtime:
    host = runtime.allocate_host(128 << 20, allocator="acl_pinned")
    device = runtime.allocate_device(128 << 20, policy="huge_first")

    runtime.fill(host, 23, host.nbytes)
    runtime.copy(src=host, dst=device, api="acl_async")

    plan = runtime.prepare_copy(
        src=device,
        dst=host,
        block_size=64 << 10,
        batch_size=2048,
        api="acl_batch",
        layout="shuffled",
        seed=0,
    )
    result = benchmark(plan, warmup=5, iterations=20, rounds=3)
    print(result)
```

- `allocate_host(size, allocator, register=False)` 返回保存地址、容量、所有者及必要底层对象的 `Buffer`。
- `allocate_device(size, policy)` 使用 `aclrtMalloc`，返回 Device `Buffer`。
- `prepare_copy(...)` 提前构造地址数组，返回可反复执行的 `CopyPlan`。`plan.run()` 完成一批复制并等待完成。
- `example.benchmark(plan, ...)` 在接口之上组织预热、计时和完整校验；这些示例逻辑位于 `example.py`。
- `copy(src=..., dst=..., size=None, api=...)` 执行单块复制，默认取两个缓冲区容量的较小值。
- `write(buffer, data)` / `read(buffer, size=None)` 提供同步准备及读回能力；`fill(buffer, value, size)` 设置指定范围。
- 所有缓冲区及 plan 只能在其所属 `Runtime` 上下文中使用，退出时先同步，再按依赖顺序释放。
- 拷贝块大小须为正的 4 KiB 倍数，且 `block_size * batch_size` 不能超过任一缓冲区容量。

## 可选后端

| 选择 | 实际调用 / 行为 |
| --- | --- |
| `--host-alloc mmap` | 独立匿名 mmap，保留映射对象 |
| `--host-alloc aligned` | libc `posix_memalign`，4 KiB 对齐 |
| `--host-alloc acl_pinned` | `aclrtMallocHost`，保留原始指针供 `aclrtFreeHost` 释放 |
| `--host-alloc mf_pool` | `bm.create2` 创建真实本地 DRAM 池，分配行为由 MemFabric 决定 |
| `--device-policy huge_first/huge_only/normal_only` | `aclrtMalloc` 对应策略；实际支持由设备和 CANN 决定 |
| `--api acl_sync` | 每块调用 `aclrtMemcpy` |
| `--api acl_async` | 每块调用 `aclrtMemcpyAsync`，每批末尾 `aclrtSynchronizeStream` |
| `--api acl_batch` | 一次 `aclrtMemcpyBatch`，同步完成；需要该导出 C 符号 |
| `--api mf_copy` | 每块调用 `handle.copy_data`，`flags=0`、默认 stream 的同步路径 |
| `--api mf_batch` | 一次 `handle.copy_data_batch`，相同同步语义 |

`--host-register` 可对外部分配的 Host 缓冲区执行 `aclrtHostRegister(MAPPED)`，结束时反注册。
注册得到的 Device alias 单独保存，ACL H2D/D2H 使用原始 Host 地址。内核 <= 5.10 时仅允许 `acl_pinned`；
机型与 CANN 的其他限制以接口执行结果及官方文档为准。池内存及 MF 接口由 MF 管理注册，禁止叠加显式 ACL 注册。
注册不等价于统一的锁页分配策略，也不保证改变异步接口的完成语义。

## 测量口径

- 默认方向为 H2D、D2H，块大小 `16K:2M`；自动 batch 取 `max(1, 128 MiB / block_size)`。
- 默认预热 5 次，每轮计时复制 20 批，共 3 轮。可用 `--warmup-count`、`--copy-count`、`--rounds` 修改。
- 地址数组、初始化、触页、注册、预热、目标重置和校验均在计时之外。每批的完成等待在计时之内。
- `shuffled` 用固定 seed 打乱所有块的提交顺序，源、目标偏移一一对应，块不重叠；它模拟乱序块访问，
  尚不模拟不同源/目标布局、跨 rank 访问、计算重叠或多 stream 并发。
- 每轮计时前用不同的 poison 字节重置目标；源数据包含块编号，每轮完成后完整读回并逐字节比较。
  任意一轮校验失败均停止，不输出通过结果。
- 输出各轮平均批次延迟的中位数和对应有效带宽；`GB/s` 使用十进制 `10^9` 字节。
- `batch_latency_us` 是完整一批的平均完成延迟；计时包含 Python 循环、ctypes / pybind 调用及返回码检查开销。
  数据可用于 Python 应用路径比较，不能直接解释为纯硬件 DMA 延迟。逐块 Python 提交与单次 native batch 的差异
  包含提交开销；当前版本未增加 native 计时循环。
- MF 接口可能在库内按机型分流或回退；JSON 将底层派发标为 `MF-managed; not traced`，不宣称它一定使用 ACL batch。
- CLI 限制每批 payload 不超过 4 GiB；测试时还需要源、目标和校验数据的额外内存。锁页配额及大页资源需满足要求。
- NUMA 比较可在同一环境下用 `numactl` 控制进程策略，并核查实际分配位置；本版未封装 BM NUMA flags。
- `--json` 输出逐条结果，包含接口、分配器、块布局、各轮延迟、完成计时口径和校验状态；底层库日志仍可能另行输出。

## 验证与限制

CPU 回归无需 CANN，不会加载 NPU 动态库：

```bash
python -m unittest discover -s memfabric/h2d_d2h -p 'test_interface.py' -v
```

测试使用会延迟完成复制的模拟 ACL 后端，覆盖完成同步、GVA/VA 使用、资源释放、配置失败及预热后的复制失效。
模拟后端不能证明真实 CANN ABI、硬件支持或性能。当前版本尚未完成 NPU 真机验证；上机后应先执行前述小负载命令，
再分别验证分配器、接口、方向与 BM 场景。`aligned` 的 libc 分配路径也需在 Linux 上验证。
验收时保留机型、拓扑、驱动/CANN/MemFabric 版本、NUMA 配置、命令及 JSON 原始数据。

首版未覆盖 BM HBM 池、Host 大页独立分配、ACL batch async、Torch 分配器和多卡并发。

参考：[mfcli benchmark](https://gitcode.com/nbbb24/memfabric_hybrid/blob/bb6a851c/src/smem/python/memfabric_hybrid/memfabric_hybrid/benchmark/README.md)、
[PR #1546](https://gitcode.com/Ascend/memfabric_hybrid/pull/1546)、
[ACL Host 注册说明](https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/910/API/runtimeapi/aclcppdevg_03_1804.html)、
[ACL batch 属性定义](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/910/API/runtimeapi/aclcppdevg_03_1942.html)。

## 来源与许可证

从 [MemFabric Hybrid fork 的 h2d_d2h 分支](https://gitcode.com/nbbb24/memfabric_hybrid/tree/h2d_d2h)
移入，源提交为 `bb6a851c`（2026-10-30）。`interface.py`、`example.py` 和 `test_interface.py` 保持源版本内容，
本仓适配限于文档路径、入口索引和许可证副本。

代码遵循 [Mulan PSL v2](LICENSE)，保留原始版权及许可证声明。
