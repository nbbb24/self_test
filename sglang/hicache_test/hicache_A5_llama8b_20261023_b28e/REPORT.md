# 实验 02：Llama-3.1-8B-Instruct / A5 / HiCache + MemCache

## 目的与配置
更换为 LlamaForCausalLM 架构，验证同一 HiCache L3 基础路径。模型只读使用 /home/weights/Llama-3.1-8B-Instruct。
单卡物理 NPU 4、TP=1、device_urma、write_through、wait_complete。
max-total-tokens=4096、context-length=4096、max-running-requests=2、mem-fraction-static=0.30、L2=2 GiB、MemCache DRAM=2GB。
复用实验01的进程内兼容配置，不修改共享安装。不是 DSPARK/DeepSeek-V4/四卡性能测试。

## 脚本
run.sh 是统一入口；basic.sh 是完整启动命令；runtime.env 隔离缓存及日志。
meta.json/local.json 为独立 MemCache 配置；supervisor.py 管理服务、验证和清理；verify.py 为已修正解析的验证器。
clang-private.sh、python/sitecustomize.py 保留兼容处理。
本目录仅对应这次启动；监督器拒绝覆盖已有 run-status.json。

## 实际执行命令（宿主机）
docker exec -d sglang-sxy bash -c 'exec bash /home/sxy/hicache_A5_llama8b_20261023_b28e/run.sh > /home/sxy/hicache_A5_llama8b_20261023_b28e/supervisor.log 2>&1'

监督器在服务健康后执行：
python3 -B verify.py --router-url http://127.0.0.1:31000 --prefill-url http://127.0.0.1:31000 --prompt-repeat 16 --max-tokens 16 --settle-seconds 10 --output-dir validation
相对路径基于本实验目录。只发送两次确定性请求；flush 只指向本次服务和独立 MemCache。

## 结果
通过：run-status.json=PASSED，verify_exitcode=0。
冷请求 1274 tokens，缓存命中 0；清空 L1/L2 后 device=0、host=0、storage=1152，后端 NpuMemcacheStore。
两次生成的16个 token ID及文本完全相同。validation/summary.json 是机器可读判定。
两份原始响应和 backup/restore metrics 均已保存。此次直接运行修复后的验证器完成，没有依赖离线补判。
清理后 NPU4 无驻留进程，测试端口释放，原先其他卡进程仍在，见 cleanup-checks.json。
服务退出码为 -9，MetaService 退出码为0；无遗留进程，URMA CloseDevice 日志成功。退出码不表述为完全优雅退出。
尝试读取进程库映射时服务已经清理，loaded-libraries.json 为空；依赖版本来自 versions.txt，加载证据补充见 evidence.log。
不据此推断跨节点传输、并发性能、长时间稳定性或其他模型均通过。

## 约束
所有显式生成文件在 /home/sxy；不安装软件、不改系统设置、不改模型与他人文件，只清理本次创建的进程组。
保留 npu-before/after、原始响应、metrics、版本与脚本校验值。
复现时复制脚本到新的 /home/sxy 子目录并替换旧目录引用；确认卡与端口空闲后运行。禁止覆盖已完成实验。
