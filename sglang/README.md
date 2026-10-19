# SGLang Self Tests

This directory contains manual SGLang validation workloads that are too
hardware-specific or operationally expensive for a normal unit-test suite.

## Tests

### Ascend HiCache L2/L3 with NPU MemCache

Files:

- [`hicache/run_npu_memcache_l2_l3.sh`](hicache/run_npu_memcache_l2_l3.sh)
  launches a single-node, four-NPU SGLang PD deployment and the NPU MemCache
  MetaService.
- [`hicache/verify_npu_memcache_l2_l3.py`](hicache/verify_npu_memcache_l2_l3.py)
  performs the cache round-trip and enforces the pass criteria.

The test was prepared against SGLang `upstream/main` commit
`ed0b8f433f887995b1ce361b15aaf9f6ca93483f`.

#### What the test validates

1. A cold, long-prefix request populates SGLang L1, HiCache L2, and NPU
   MemCache L3 through the write-through path.
2. The Prefill server's `/flush_cache` endpoint clears L1 and L2 while leaving
   L3 intact.
3. Replaying the exact prompt restores KV cache from L3 into L2/L1.
4. The second response reports at least 256 tokens from the `storage` cache
   source.
5. Cold and restored requests produce identical output text and token IDs.

This is stronger than a launch-only smoke test: a successful request without a
reported L3 hit does not pass.

#### Prerequisites

- A Linux host with four available Ascend NPUs.
- A working SGLang NPU environment with `sglang_router` installed.
- `memcache_hybrid` installed with support for the selected transport.
- A local model compatible with the SGLang NPU backend. The default path is
  `/home/weights/Qwen3-32B-W8A8`.
- `curl` and Python 3.10 or newer.

The default device assignment matches the original test host:

- Prefill: devices `12,13`
- Decode: devices `14,15`
- Tensor parallel size: `2`

#### Quick start

A3 uses `device_sdma` for the HiCache L2/L3 path:

```bash
bash sglang/hicache/run_npu_memcache_l2_l3.sh a3
```

A5 uses `device_urma` for the HiCache L2/L3 path:

```bash
bash sglang/hicache/run_npu_memcache_l2_l3.sh a5
```

Example with explicit machine settings:

```bash
MODEL_PATH=/home/weights/Qwen3-32B-W8A8 \
PREFILL_DEVICES=12,13 \
DECODE_DEVICES=14,15 \
PREFILL_MF_STORE_URL=tcp://10.0.0.10:31001 \
DECODE_MF_STORE_URL=tcp://10.0.0.10:31001 \
bash sglang/hicache/run_npu_memcache_l2_l3.sh a3
```

Use an address reachable by MemFabric on the local machine if loopback is not
valid for the selected device transport.

#### PD transport versus HiCache transport

The two transports are configured independently:

- `MEMCACHE_PROTOCOL` controls the HiCache L2/L3 path. It defaults to
  `device_sdma` on A3 and `device_urma` on A5.
- `PD_PROTOCOL` controls Prefill-to-Decode KV transfer. It defaults to
  `device_rdma`, matching the original deployment script.

To use URMA for both paths on A5:

```bash
PD_PROTOCOL=device_urma \
bash sglang/hicache/run_npu_memcache_l2_l3.sh a5
```

Keeping the two settings separate makes transport failures easier to isolate.

#### Configuration overrides

| Variable | Default | Description |
| --- | --- | --- |
| `MODEL_PATH` | `/home/weights/Qwen3-32B-W8A8` | Model directory or identifier. |
| `SERVED_MODEL_NAME` | `qwen` | API model name. |
| `PREFILL_DEVICES` | `12,13` | Prefill NPU visibility list. |
| `DECODE_DEVICES` | `14,15` | Decode NPU visibility list. |
| `TP_SIZE` | `2` | Tensor parallel size for each server. |
| `PD_PROTOCOL` | `device_rdma` | Prefill-to-Decode MemFabric protocol. |
| `MEMCACHE_PROTOCOL` | Hardware profile | HiCache L2/L3 MemCache protocol. |
| `HICACHE_SIZE_GB` | `4` | L2 host cache size per rank, in GiB. |
| `MEMCACHE_DRAM_SIZE` | `8GB` | Actual MemCache DRAM contribution. |
| `MEMCACHE_MAX_DRAM_SIZE` | `256GB` | Per-rank address-space ceiling. |
| `MEMCACHE_WORLD_SIZE` | `256` | Maximum MemCache rank count. |
| `LOG_DIR` | `/home/sxy/logs/sglang-hicache-<profile>` | Logs and result artifacts. |
| `KEEP_SERVERS` | `0` | Set to `1` to keep services after a passing run. |

For `device_urma`, MemCache requires
`max_dram_size * world_size > 32 TiB`. The defaults provide a 64 TiB virtual
span without eagerly allocating that amount of memory.

#### Result artifacts

The output directory contains:

- `cold-response.json`
- `l3-restore-response.json`
- `summary.json`
- `metrics-after-backup.txt`
- `metrics-after-restore.txt`
- Prefill, Decode, router, and MemCache service logs

A successful run ends with a message similar to:

```text
PASS: restored 4096 prompt tokens from NPU MemCache L3; generated output is identical.
```

The launcher terminates its child services after the test unless
`KEEP_SERVERS=1` is set.

#### Upstream references

- [SGLang NPU MemCache backend documentation](https://github.com/sgl-project/sglang/tree/main/python/sglang/srt/mem_cache/storage/npu_memcache)
- [Ascend MemCache configuration](https://gitcode.com/Ascend/memcache/blob/master/doc/memcache_config.md)
- [Ascend MemCache Python API](https://gitcode.com/Ascend/memcache/blob/master/doc/memcache_python_api.md)
