# 实验 01：Qwen3-0.6B / A5 / HiCache + MemCache

## 脚本和配置
- basic.sh：最终模型启动脚本；runtime.env：本轮目录、缓存与日志隔离。
- meta.json / local.json：独立 MemCache，device_urma。
- clang-private.sh / python/sitecustomize.py：仅影响本轮进程的兼容配置。
- supervisor.py：启动、健康检查、验证、清理。
- verify.before-schema-fix.py：实测时使用的验证器；verify.py：修复字符串字段解析后的版本。
- artifact-sha256.json / versions.txt：最终脚本校验值及软件版本。

## 实际怎么跑
通过 root 密钥连接 <A5_HOST>；容器为 sglang-sxy。
宿主机先执行 npu-smi info 和容器 ps 确认空闲卡，使用物理卡 4。
容器内 source runtime.env、source /usr/local/Ascend/ascend-toolkit/latest/set_env.sh，
然后 python3 -B /home/sxy/hicache_A5_20261023_7d19/supervisor.py。
监督器用 basic.sh 启动服务，用以下参数执行验证器：
python3 -B verify.py --router-url http://127.0.0.1:31000 --prefill-url http://127.0.0.1:31000 --prompt-repeat 16 --max-tokens 16 --settle-seconds 10 --output-dir validation
所有上述相对路径均相对于本实验目录。

## 结论
真实请求已完成。冷请求 1111 tokens / L3=0；flush L1/L2 后 storage=1024，device=host=0；16 个输出 token 和文本相同。
旧验证器把 storage_backend 字符串当整数导致解析失败；最终读取保存的两份原始响应做断言，通过，详见 validation/summary.json。
原 run-status.json 的 FAILED 是验证器解析错误，保留不覆盖。修复后未再次占卡运行。
服务已清理，4 号卡无驻留；其他原有 NPU 进程仍在。

## 调试轮次
原 Wiki DSPARK 预检失败：markov_rank=0，无相应权重（dspark-preflight.log）。
attempt1：Triton 链接 libm 失败。
attempt2：私有编译路径解决链接；PyTorch seed 别名冲突仍导致 fallback，基础 SelfAttentionOperation 失败。
attempt3：加载已有 ATB/custom 环境后仍有 seed 冲突及 SelfAttentionOperation 失败。
最终：统一进程内 RNG 跟踪规则并开启 ASCEND_USE_FIA=1，原生 Qwen3 正常运行并完成真实 L3 回读。
历史启动脚本在当时逐步修改，没有逐轮完整快照；子目录日志是原始证据，下面生成的 reconstructed 文件明确为事后按修改记录还原，不冒充实测时保存的文件。
原始 Wiki 副本与未启用 DSPARK 的说明见 REPORT.md。

## 复现注意
本目录是已完成实验，重新执行旧监督器会覆盖日志。先复制配置到 /home/sxy 下的新目录并替换所有目录引用、检查空闲卡与端口，再运行；不要覆盖本目录。
