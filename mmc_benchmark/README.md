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
| -n             | [1, ]                    | call count, 循环测试次数，受限于内存池总容量，会生成 batch_size \* process_count \* call_count个key   |
| -w             | [0, ]                    | 写入预热次数。预热在正式写入计时前执行，不计入性能结果；每轮预热使用临时 key，并在计时前删除。默认值为 2；读测试已在正式计时前完成预热，无需配置该参数。 |
| -d             | [1, 2]                   | 连续 tensor（1）大小由 -s 决定；离散缓存（2）大小由 -m 指定的模型决定。未传 -m/-d 时保持默认值 1。 |
| -m, --model    | 见下表                    | 模型缓存布局，默认 deepseek-r1；显式选择模型且未传 -d 时自动使用 -d 2。 |
| -e             | [memcache, mooncake]     | 内存池后端，默认支持memcache，mooncake须安装相关软件     |
| -l             | [npu, cpu]               | 测试数据tensor位于NPU还是CPU      |

## 五、选择模型

```shell
export MMC_LOCAL_CONFIG_PATH=$PWD/mmc-local.conf
export ASCEND_RT_VISIBLE_DEVICES=0
bash bench_start.sh -t write_read_same_process -m glm-5.2 -p 1 -b 4 -n 10 -w 2
# 长参数等价：--model glm-5.2 或 --model=glm-5.2
bash bench_start.sh --help
```

预设模拟每个 key 对应 **128 个原始 token、BF16 缓存字节量**，不加载权重，不执行模型推理。
GLM 包含主 MLA latent、RoPE 和独立 indexer K；V4/V4.1 仅包含全局前缀 KV 与 indexer K，
不包含 SWA、压缩尾部状态及 MTP。共享缓存按实际拥有者计数，不为复用层重复分配。

| `-m/--model` | 每 key 的 buffer 数 | 每 key 的有效字节数 | 组成（KiB） |
|-------------|---------------------|--------------------|-------------|
| `deepseek-r1` | 122 | 8,994,816（8.578125 MiB） | 61 × (128 + 16) |
| `glm-5.1` | 234 | 14,057,472（13.40625 MiB） | 78 × (128 + 16 + 32) |
| `glm-5.2` | 177 | 12,189,696（11.625 MiB） | 78 × (128 + 16) + 21 × 32 |
| `deepseek-v4-flash` | 62 | 880,640（860 KiB） | 21 × (32 + 8) + 20 × 1 |
| `deepseek-v4-pro` | 91 | 1,260,544（1231 KiB） | 30 × (32 + 8) + 31 × 1 |
| `deepseek-v4.1-flash` | 8 | 409,600（400 KiB） | 3 × (64 + 16) + (128 + 32) |

上述预设以调研时的官方配置为依据：[GLM-5.1](https://huggingface.co/zai-org/GLM-5.1/blob/main/config.json)、
[GLM-5.2](https://huggingface.co/zai-org/GLM-5.2/blob/main/config.json)、
[V4 Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/blob/main/inference/config.json)、
[V4 Pro](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/blob/main/inference/config.json)、
[V4.1 Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/main/config.json)。
GLM-5.2 的完整 indexer 位于第 0、1、2、6、10、……、74 层，共 21 层；V4.1 的 KV 来源为第 2、8、14、20 层。
这些 BF16 预设不代表推理框架的原生 FP8/FP4、scale、padding 或 TP/PP 分片布局；例如 V4.1 此处使用 400 KiB，
不是原生量化全局缓存的 111.25 KiB。实际部署布局需要另行核对并调整 `model_profiles.py`。

显式传 `-d 1` 时继续测 `-s` 指定的连续字节块，模型布局不生效；`-d 2` 时 `-s` 不生效。
连续模式和默认 R1 模式保留原有 key；其他模型使用包含模型名、BF16、128 token 和布局版本的独立 key 前缀。
独立启动 write/read 时，两次命令必须使用相同的模型、数据模式、设备、batch 和 call count。

原始 value 总量为 `bytes_per_key × batch_size × call_count × process_count`，还需为元数据、
分配对齐和副本等预留容量。例如 R1 在 `-b 16 -n 100 -p 1` 下约为 13.403 GiB。
日志输出模型名、每 key 的 buffer 数和有效字节数；带宽单位为 GiB/s。

## 六、适配位置与验证

`model_profiles.py` 定义每个 key 的有序 buffer 大小；`mutil_process.py` 将等长组件分组分配并注册，
再按预设顺序生成 MemCache put/get 的地址和大小列表。write、read、write_read_same_process 三种模式
共用这套逻辑，checksum 只覆盖实际传输的数据，不含 padding。`direct` 由 CPU/NPU 与读写方向决定。

直接调用 Python 入口仍兼容原来的 9 个位置参数，可追加 `-m/--model`；数据模式需显式传 2：

```shell
python3 run_mutil_process.py write_read_same_process 1 4 1048576 10 2 2 memcache npu --model glm-5.2
```

不依赖 torch、Ascend 或 MetaService 的 CPU 内存契约测试：

```shell
# 在 benchmark 目录执行
python3 -m unittest -v test_benchmark_models.py
```

测试覆盖预设大小、所有模型的真实 CPU 指针 put/get 往返、独立读写、连续模式、checksum 错误、
模型参数传递及错误模型校验。这些检查不能替代 Ascend 环境的功能和性能实测。
