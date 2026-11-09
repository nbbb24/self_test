### 宿主机关闭验签
```
for i in {0..7}; do npu-smi set -t custom-op-secverify-enable -i $i -d 1; done;
for i in {0..7}; do npu-smi set -t custom-op-secverify-mode -i $i -d 0; done;
```

### 拉起容器
```
bash docker.sh
```

### 安装memfabric
```
git clone https://gitcode.com/Ascend/memfabric_hybrid.git
cd memfabric_hybrid
```
```
git checkout release/1.3
bash script/build_and_pack_run.sh
```
安装output目录下的run包
```
bash output/memfabric_hybrird-xxxxxxx.run
```
安装算子包
```
mfcli kernel install
```
出现以下字段则说明安装成功
```
Detected supported CANN version: 9.2.0
Installed: /usr/local/Ascend/cann-9.2.0/opp/vendors/cust/op_impl/aicpu/hybm/kernel/cann-hybm-compat.tar.gz
Installed: /usr/local/Ascend/cann-9.2.0/opp/vendors/cust/op_impl/aicpu/hybm/config/libcann_hybm_kernel.json
Installed: /usr/local/Ascend/cann-9.2.0/opp/vendors/cust/op_impl/aicpu/hybm/config/cann_hybm_kernel_version
Updated: /usr/local/Ascend/cann-9.2.0/conf/ascend_package_load.ini
```
以下报错可以忽略，不会用到该组件
```
MemFabric Hybrid AccOffload:
  Start installing, this may take minutes...
2026-11-05T06:40:08+0000 ERROR memfabric_hybrid.cli: failed to install acc_offload operator extend library: extend build failed at stage 'cmake_build', exit code 2, stderr: CMake Warning (unused-cli):
  Manually-specified variables were not used by the project:
    DYNAMIC_MODE
    INCLUDE_DIR
```
也可以通过以下指令确认版本是否为1.3.1
```
mfcli kernel version
```
### mooncake whl包

#### pip install以下链接的whl包
可以从3种渠道获取，然后pip install即可
1. https://github.com/kvcache-ai/Mooncake/releases/tag/v0.3.14
2. https://pypi.org/project/mooncake-transfer-engine-memfabric/#files
3. 直接下载
```
wget -c -P /home \
  'https://github.com/kvcache-ai/Mooncake/releases/download/v0.3.14/mooncake_transfer_engine_memfabric-0.3.14-cp312-cp312-manylinux_2_35_aarch64.whl'
```
#### 检查Mooncake 本体、engine 扩展和 Tent 模块
```
python3 -c "import mooncake; import mooncake.engine; print('mooncake OK', mooncake.__file__)"
python3 -c "from mooncake import tent; print('tent OK', tent.__file__)"
```
#### 加载路径与动态库检查，预期结果应指向 wheel 所在的 site-packages/mooncake/，不应指向 /usr/local/Ascend/cann-*/python/site-packages/mooncake 中的旧副本。
```
cd /tmp
python3 -c "import mooncake; print(mooncake.__file__)"
```

#### Tent+MF检查
```
bash prepare.sh
```
#### 遇到问题可以参考
https://gitcode.com/logn/window_feature_doc/wiki/Mooncake%EF%BC%88MemFabric%E7%89%88%EF%BC%89%E5%AE%8C%E6%95%B4%E6%8C%87%E5%8D%97.md

### 脚本
在现有脚本基础上较重要的环境变量，注意修改脚本的ip网卡、配置路径等，具体可以参考p/d脚本

若要使用大页则参考mooncake文档，宿主机配置大页并设置环境变量：https://kvcache-ai.github.io/Mooncake/design/transfer-engine/transport/memfabric_transport.html#important-notes
```
# === TENT + memfabric ===
export MC_USE_TENT=1
export MC_TENT_CONF=./mooncake_run/memfabric_tent.json
export MOONCAKE_LOCAL_HOSTNAME="$LOCAL_IP"
export MF_PROTOCOL=device_urma
export ASCEND_MF_TRANSFER_PROTOCOL=device_urma
export MEMFABRIC_LIB_DIR=/opt/upgrade/.venv/lib/python3.12/site-packages/memfabric_hybrid/lib
export MOONCAKE_LIB_DIR=/opt/upgrade/.venv/lib/python3.12/site-packages/mooncake
export LD_LIBRARY_PATH="$MEMFABRIC_LIB_DIR:$MOONCAKE_LIB_DIR:${LD_LIBRARY_PATH:-}"

# === Store 池化 ===
export MOONCAKE_CONFIG_PATH=./mooncake_run/mooncake.json
export PYTHONHASHSEED=0
#export MC_STORE_USE_HUGEPAGE=1
#export MC_STORE_HUGEPAGE_SIZE=2MB
```


### 拉起mooncake master
```
bash start_mooncake.sh
```

### 拉起p
```
bash p_pool.sh x
```

### 拉起d
```
bash d.sh x
```

### proxy
```
bash proxy.sh
```
### curl一条
```
bash single_curl.sh
```

### benchmark
```
bash benchmark_vllm.sh
```

### 若要清除prefix cache(慎用！)
```
bash clear_cache.sh
```