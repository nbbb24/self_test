# 实验03：Qwen3-14B / A5 / HiCache + NPU MemCache

## 目的与边界
只读使用 /home/weights/Qwen3-14B，更大模型验证基础 L3 回读。单卡物理NPU4、TP=1、device_urma。
不启用 DSPARK，不做四卡、跨节点或性能结论。

## 脚本及配置
run.sh 为入口；basic.sh 保存完整模型启动命令；runtime.env 将缓存/临时文件/日志放本目录。
meta.json/local.json 是独立MemCache配置；supervisor.py 自动启动、验证、清理并拒绝覆盖旧状态。
verify.py 执行冷请求、清空L1/L2、回读和输出一致性校验。
clang-private.sh/python/sitecustomize.py 复用已验证的进程内兼容措施，不更改共享软件。
启动前保存 artifact-sha256.json、模型config快照；版本与实际动态库映射另存。
max-total-tokens=4096，context-length=4096，max-running-requests=2，mem-fraction-static=0.40；L2=2GiB，MemCache DRAM=2GB。

## 实际怎么跑
先在宿主机 npu-smi info 确认4号卡无驻留进程；容器 ps 确认没有旧测试服务。
宿主机执行：
docker exec -d sglang-sxy bash -c 'exec bash /home/sxy/hicache_A5_qwen14b_20261023_c39f/run.sh > /home/sxy/hicache_A5_qwen14b_20261023_c39f/supervisor.log 2>&1'
监督器检查5006/6006/18006/31000端口空闲，启动MetaService和模型服务，健康后执行：
python3 -B verify.py --router-url http://127.0.0.1:31000 --prefill-url http://127.0.0.1:31000 --prompt-repeat 16 --max-tokens 16 --settle-seconds 10 --output-dir validation
相对路径基于本实验目录。只发送两次确定性验证请求；服务内部另有启动预热/健康检查。

## 结论
通过。run-status.json=PASSED，verify_exitcode=0。
冷请求 1111 tokens，L3命中0；清空L1/L2后 device=0、host=0、storage=1024，storage_backend=NpuMemcacheStore。
两次输出的16个token ID及文本完全一致；详见 validation/summary.json 与原始响应。
backup/restore metrics 均保存。实际库映射记录了4个进程，见 loaded-libraries.json。
测试后NPU4无驻留进程，端口释放，其他卡的原有进程仍在，见 cleanup-checks.json、npu-before/after.txt、processes-after.txt。
关闭阶段模型服务退出码-9，MetaService退出0；服务记录了调度进程收到终止信号后的退出清理，URMA CloseDevice成功，不能把-9表述为正常退出码。
功能断言通过不等于关闭流程完全优雅，也不能推断多卡/跨机/性能/长稳测试通过。

## 隔离与复现
不改系统设置、共享安装、模型或他人文件，只控制本次独立进程组。
显式生成文件全在 /home/sxy。结束后复核NPU、进程与端口。
若复现，复制脚本/配置到新的 /home/sxy 子目录并替换绝对路径，确认卡和端口空闲后运行；不要覆盖本目录。
