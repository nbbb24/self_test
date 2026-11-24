# benchmark

提供python benchmark，以方便测试memcache内存池put/get性能。
主要涉及如下步骤。

## 一、启动元数据服务

```shell
export MMC_META_CONFIG_PATH=${your config path}/mmc-meta.conf;
bash bench_start_meta.sh
```

## 二、写数据测试

```shell
export MMC_LOCAL_CONFIG_PATH=${your config path}/mmc-local.conf;
bash bench_start.sh -t write -p 1 -b 16 -s 1048576 -n 100 -w 10 -d 1 -e memcache -l npu
```

## 三、读数据测试

```shell
export MMC_LOCAL_CONFIG_PATH=${your config path}/mmc-local.conf;
bash bench_start.sh -t read -p 1 -b 16 -s 1048576 -n 100 -d 1 -e memcache -l npu
```

## 四、同一进程先 write 再 read

```shell
export MMC_LOCAL_CONFIG_PATH=${your config path}/mmc-local.conf;
bash bench_start.sh -t write_read_same_process -p 1 -b 16 -s 1048576 -n 100 -w 10 -d 1 -e memcache -l npu
```

`write_read_same_process` 为每张卡启动一个子进程并初始化一次 Store，使用独立的 src/dst buffer。write
不指定目标 rank，由 MemCache 自行分配；所有子进程写完并通过 Barrier 同步后，再按原有 read 流程
执行预热和正式读取，最后比较 src/dst checksum。

## 入参说明

| **参数**        | **可选项**               | **说明** |
|----------------|--------------------------|----------|
| -t             | [read, write, write_read_same_process] | 控制读取、写入或同进程顺序模式      |
| -p             | [1, 32]                  | process count, 运行的进程数量, 受限于NPU的个数      |
| -b             | [1, ]                    | batch size, 单次IO key的个数      |
| -s             | [1, ]                    | block size, 单次IO key对应的value的大小，单位Byte      |
| -n             | [10, ]                   | call count, 循环测试次数，受限于内存池总容量，会生成 batch_size \* process_count \* call_count个key   |
| -w             | [0, ]                    | 写入预热次数。预热在正式写入计时前执行，不计入性能结果；每轮预热使用临时 key，并在计时前删除。默认值为 2；读测试已在正式计时前完成预热，无需配置该参数。 |
| -d             | [1, 2]                   | 测试连续tensor（1）还是离散tensor（2），离散tensor默认为61\*128KB + 61\*16KB，连续tensor大小为block size   |
| -e             | [memcache, mooncake]     | 内存池后端，默认支持memcache，mooncake须安装相关软件     |
| -l             | [npu, cpu]               | 测试数据tensor位于NPU还是CPU      |

## 五、128 token 的离散缓存基线

当前脚本以 DeepSeek R1 的缓存大小模拟数据传输，不执行模型推理。使用 `-d 2` 时，每个 key 对应
61 层数据，每层由 128 KiB 和 16 KiB 两个 buffer 组成，共 122 个 buffer、8,994,816 Byte
（8.578125 MiB）。该模式的数据大小由 `mutil_process.py` 中的配置决定，`-s` 不生效。

```shell
export MMC_META_CONFIG_PATH=$PWD/mmc-meta.conf
bash bench_start_meta.sh

# 在另一终端执行；按测试机器实际情况选择可用 NPU。
export MMC_LOCAL_CONFIG_PATH=$PWD/mmc-local.conf
export ASCEND_RT_VISIBLE_DEVICES=0
bash bench_start.sh -t write_read_same_process -p 1 -b 16 -n 100 -w 10 -d 2 -e memcache -l npu
```

以上命令的正式数据为 1,600 个 key，原始 value 总量约 13.403 GiB；多进程时按进程数增加。
内存池还需留出元数据、分配对齐及副本等额外空间。写入与读取分别计时，预热不计入正式结果。
脚本按 1024 的幂换算带宽，因此日志中标记为 `GB/s` 的数值实际为 GiB/s。

## 六、不同模型的适配位置

MemCache 的 `batch_put_from_layers(keys, buffs, sizes, direct)` 和
`batch_get_into_layers(keys, buffs, sizes, direct)` 接口可以复用。模型差异主要体现在每个 key 的
buffer 数量、地址及实际字节数；`direct` 由 CPU/NPU 和拷贝方向决定，不由模型名称决定。

| 位置 | 需要适配的内容 |
|------|----------------|
| `mutil_process.py` 的数据配置 | 按模型生成有序缓存组件：主 KV、RoPE、indexer K、量化 scale，以及需要保存的状态。 |
| `allocate_benchmark_buffers` 及各 worker 的分配逻辑 | 为不同大小的组件分配并注册内存；共享缓存按实际拥有者分配，避免重复传输。 |
| `build_batch_arguments` 及独立读写 worker 的参数构造 | 生成与布局一致的地址列表和大小列表，不假定每层始终有一对等长 K/V 列表。 |
| 大小统计及 checksum | 按实际传输的有效字节计算，不把分配 padding 计入带宽或未写入区域的校验。 |
| `bench_start.sh`、`run_mutil_process.py` | 后续增加模型 profile 选择，并将配置传递到子进程。 |

GLM-5.1/5.2 需考虑 DSA indexer 缓存及索引共享；DeepSeek V4 Flash/Pro 需考虑不同层的压缩率；
V4.1 还需按共享 KV 的拥有者去重。全局前缀缓存与 SWA/压缩尾部状态应分别描述和统计。
实际布局须与所用推理框架、KV 精度及 TP/PP 配置一致，不能仅凭模型权重量化精度推算。

当前同步版本仍固定使用上述 R1 基线，尚未实现这些模型 profile。

本目录的核心脚本同步自 `nbbb24/memcache` 的 `mmc_bm` 分支，源提交为 `9b80c51c`。
启动脚本采用通用默认参数，配置路径和可见设备由调用前的环境变量指定。
