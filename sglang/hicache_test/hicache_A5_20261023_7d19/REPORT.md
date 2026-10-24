# A5 SGLang HiCache + NPU MemCache 基础功能验证

日期：2026-10-23（Asia/Shanghai）
目标：<A5_HOST>，容器 sglang-sxy
目录：/home/sxy/hicache_A5_20261023_7d19

## 结果：基础功能通过

模型 /home/weights/Qwen3-0.6B，物理 NPU 4，TP=1，device_urma。
冷请求输入 1111 tokens，L3 命中 0。清空 L1/L2 后回读，device=0、host=0、storage=1024，storage_backend=NpuMemcacheStore。两次输出的 16 个 token ID 和文本完全一致。
最终断言见 validation/summary.json；两份原始响应保留，并记录 SHA256。
这不是 DeepSeek-V4 四卡混部、DSPARK 或性能压测通过。

## 测试范围变化

原 Wiki 的 DeepSeek-V4 配置预检失败：现有权重缺少 DSPARK 配置和对应权重，报 DSpark draft requires markov_rank > 0; got markov_rank=0。
用户授权关闭 DSPARK，并要求最基础测试，因此采用小模型单卡验证 HiCache。

## 原始运行状态与最终判定

实际冷请求、flush_cache、回读请求已完成。旧验证器将新增的 storage_backend 字符串转成整数导致解析异常，监督器随后清理服务。
run-status.json 和 verify.log 保留原始失败状态。随后直接对两份保存的真实响应做严格离线断言，全部通过；没有重写原始响应。最终功能判定为 validation/summary.json。
verify.py 已改为只解析 device/host/storage 数值字段，并要求 output_ids 非空；修复后检查了语法，未再次占卡重跑。回读后的 Prometheus 快照因解析异常未采集，回读证据来自原始响应及服务日志。

## 最终配置与环境处理

basic.sh、meta.json、local.json 是最终运行配置。仅使用 NPU 4，独立端口 31000/5006/6006/18006，监听 127.0.0.1。
L2=2 GiB，MemCache DRAM=2GB，world_size=4 是容量上限，并非启动四个 rank。
禁用 graph，启用 ASCEND_USE_FIA=1、kernel_ascend、page_first_kv_split、write_through、wait_complete、缓存报告和 token ID 返回。
未使用 DSPARK 或 Wiki 的额外 A5 算子库。

SGLang SHA：d0a75654368a470d8484e3ea45b06e500fee57af；MemCache/MemFabric 1.3.0；PyTorch/torch_npu 2.13.0。完整版本见 versions.txt。
复用已有容器安装，没有修改产品源码、重新构建或安装软件。

仅对本次进程使用的兼容处理：
- clang-private.sh 优先容器 /lib/aarch64-linux-gnu，避开宿主挂载 libm 链接脚本的无效 /lib64 路径。
- python/sitecustomize.py 统一 torch_npu 别名化 RNG 函数的 Dynamo 跟踪规则，避免重复注册；只经本次 PYTHONPATH 加载，不改安装包。
- 加载已有 CANN/ATB 环境，ASCEND_USE_FIA=1 避开失败的 SelfAttentionOperation。
失败证据保留于 attempt1/attempt2/attempt3。

## 隔离与清理

未执行 CPU governor/sysctl 调优，未停止他人进程，未修改模型或他人文件。
显式生成的配置、脚本、日志、缓存和临时文件均位于本目录。
监督器只控制本次独立进程组。测试服务及 MetaService 已退出，容器检查只剩原有两个 bash 会话。
npu-before.txt 与 npu-after.txt 显示原先 0/2/3 号卡进程仍在，测试后的 4 号卡无驻留进程。这不能证明共享主机完全没有瞬时资源竞争。
evidence.log 保留 MemCache 初始化、请求、flush 及 device_urma 清理证据。

## 复现

先确认 NPU 4 和上述端口空闲。重复运行会覆盖顶层日志，应先将已有产物归档到 /home/sxy 下。
容器内执行：
    source /home/sxy/hicache_A5_20261023_7d19/runtime.env
    source /usr/local/Ascend/ascend-toolkit/latest/set_env.sh
    python3 -B "$RUN/supervisor.py"
监督器会启动独立 MetaService 与模型服务、运行验证器并清理自己的进程组。