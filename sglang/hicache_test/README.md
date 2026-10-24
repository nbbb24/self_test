# A5 SGLang HiCache + NPU MemCache 实验归档

整理日期：2026-10-24；实验日期：2026-10-23。

## 从这里开始

1. 阅读本文件了解结论与范围。
2. 阅读「复现流程.md」查看环境、执行步骤和隔离要求。
3. 每个实验目录的 EXPERIMENT.md / REPORT.md 包含本轮参数、实际命令和结论。
4. validation/summary.json 与原始响应是结果证据；basic.sh / supervisor.py / verify.py 是脚本。

| 实验目录 | 模型 | 输入tokens | 冷请求L3 | 回读L3 | 结果 |
| --- | --- | ---: | ---: | ---: | --- |
| hicache_A5_20261023_7d19 | Qwen3-0.6B | 1111 | 0 | 1024 | 真实请求通过；旧解析器异常后对原始响应离线核验 |
| hicache_A5_llama8b_20261023_b28e | Llama-3.1-8B-Instruct | 1274 | 0 | 1152 | 验证器退出0 |
| hicache_A5_qwen14b_20261023_c39f | Qwen3-14B | 1111 | 0 | 1024 | 验证器退出0 |

三轮均为单卡物理NPU4、TP=1、同一实例完成Prefill和Decode，没有PD分离或PD Router。
协议device_urma用于HiCache存储访问。flush L1/L2后，两级命中均为0；L3回读成功，16个输出token及文本与冷请求一致。
这些结果不代表多卡、跨节点、性能或长时间稳定性通过。
原Wiki DeepSeek-V4/DSPARK配置因现有权重缺少markov配置而未通过预检；不得将基础测试结果标为该配置通过。

## 文件说明

- basic.sh：每轮最终模型服务启动命令。
- run.sh：实验02/03入口（实验01通过runtime.env + supervisor.py执行）。
- runtime.env：本轮目录、缓存、临时文件和日志路径。
- meta.json / local.json：独立MemCache配置。
- supervisor.py：启动服务、健康检查、执行验证、只清理本次进程组。
- verify.py：冷请求→清空L1/L2→L3回读→输出一致性校验。
- clang-private.sh / python/sitecustomize.py：本次进程专用环境兼容措施，不修改共享安装。
- validation/：原始响应、summary和指标（实验01无回读后的metrics快照）。
- versions.txt / loaded-libraries.json：软件版本、实际库映射；实验02映射因采集过晚为空，实验03已完整采集。
- npu-before/after.txt、cleanup-checks.json：设备与清理证据。
- attempt1/2/3：实验01调试失败日志，reconstructed脚本为事后还原，非原始快照。
- SHA256SUMS.json：本归档所有文件的校验清单（清单自身除外）。

## 重要区别

实验01的run-status.json保留解析器失败状态，最终功能判定见validation/summary.json；不能只看状态文件而忽略说明。
模型服务结束阶段有-9退出码，报告明确保留，不声称完全优雅退出。测试后确认无本次残留进程、端口释放，原有其他卡进程仍在。

本包不含模型权重、编译缓存、临时文件、密钥或依赖安装包。脚本是Linux/Ascend环境脚本，Windows用于阅读和保存。
归档保留原始绝对路径以保证证据一致；复现请使用新的/home/sxy子目录，勿直接覆盖历史实验。
