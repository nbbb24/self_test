export MC_USE_TENT=1
export MF_PROTOCOL=device_urma
export ASCEND_MF_TRANSFER_PROTOCOL=device_urma
export MOONCAKE_LOCAL_HOSTNAME="$LOCAL_IP"
export MC_TENT_CONF=./mooncake_run/memfabric_tent.json
export MEMFABRIC_LIB_DIR=/opt/upgrade/.venv/lib/python3.12/site-packages/memfabric_hybrid/lib
export MOONCAKE_LIB_DIR=/opt/upgrade/.venv/lib/python3.12/site-packages/mooncake
export LD_LIBRARY_PATH="$MEMFABRIC_LIB_DIR:$MOONCAKE_LIB_DIR:${LD_LIBRARY_PATH:-}"


cat > /tmp/check_tent_memfabric.py <<'EOF'
import ctypes, importlib, json, os, sys
errors = []

if os.environ.get("MC_USE_TENT") != "1":
    errors.append("MC_USE_TENT 必须为 1")
if os.environ.get("MF_PROTOCOL") not in ("device_sdma", "device_rdma", "device_urma"):
    errors.append("MF_PROTOCOL 应为 device_sdma 或 device_rdma/urma")

conf_path = os.environ.get("MC_TENT_CONF", "")
if not conf_path or not os.path.isfile(conf_path):
    errors.append("MC_TENT_CONF 未设置或文件不存在: " + repr(conf_path))
else:
    try:
        conf = json.load(open(conf_path))
        enabled = [k for k, v in conf.get("transports", {}).items() if v.get("enable")]
        if enabled != ["memfabric"]:
            errors.append(f"Tent transports 必须仅启用 memfabric，当前: {enabled}")
    except Exception as e:
        errors.append("Tent 配置无法解析: " + str(e))

try:
    m = importlib.import_module("mooncake.engine")
    print("mooncake.engine ->", m.__file__)
    for name in ("initialize", "get_rpc_port", "batch_register_memory", "batch_transfer_sync_write"):
        if not hasattr(m.TransferEngine, name):
            errors.append("TransferEngine 缺少方法 " + name)
except Exception as e:
    errors.append("无法导入 mooncake.engine: " + str(e))

mf_dir = os.environ.get("MEMFABRIC_LIB_DIR", "")
try:
    lib = ctypes.CDLL(os.path.join(mf_dir, "libmf_smem.so"), mode=ctypes.RTLD_GLOBAL)
    for symbol in ("smem_trans_init", "smem_trans_create", "smem_trans_batch_register_mem",
                   "smem_trans_batch_copy", "smem_trans_batch_status_query"):
        if not hasattr(lib, symbol):
            errors.append("libmf_smem.so 缺少符号 " + symbol)
    print("libmf_smem.so ->", os.path.join(mf_dir, "libmf_smem.so"))
except Exception as e:
    errors.append(f"无法加载 libmf_smem.so，请检查 MEMFABRIC_LIB_DIR 和 LD_LIBRARY_PATH: {e}")

if errors:
    print("FAIL:")
    [print("  -", error) for error in errors]
    sys.exit(1)
print("OK: Tent + MemFabric 预检通过")
EOF
python3 /tmp/check_tent_memfabric.py