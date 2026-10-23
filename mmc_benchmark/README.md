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
