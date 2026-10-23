# Ascend NPU HiCache L2/L3 Validation

This directory contains a single-node test for moving SGLang KV cache between
HiCache L2 and NPU MemCache L3.

| Hardware profile | HiCache L2/L3 transport | Command argument |
| --- | --- | --- |
| Ascend A3 | `device_sdma` | `a3` |
| Ascend A5 | `device_urma` | `a5` |

The Prefill-to-Decode transport is configured separately and defaults to
`device_rdma` for both profiles.

## DeepSeek-V4-Flash Wiki copies: A3 and A5

[`hicache_A5.sh`](hicache_A5.sh) and [`hicache_A3.sh`](hicache_A3.sh) are
standalone copies adapted from the [A5 single-node mixed-deployment Wiki](https://github.com/shun8686/sglang/wiki/%5BA5%5D-DeepSeek-V4-Flash-%E5%8D%95%E6%9C%BA%E6%B7%B7%E9%83%A8), retrieved on 2026-10-23.
They use the Wiki's DeepSeek-V4-Flash, four NPUs, TP=4/DP=4, DSPARK,
HiCache and generated-shared-prefix benchmark configuration.

- A5 retains `device_urma`.
- A3 changes only the MemCache protocol to `device_sdma`. DeepSeek-V4/DSPARK
  compatibility and the retained A5 extra-ops paths are **not validated on A3**.

Prepare the matching SGLang/Ascend environment, model and custom operators.
Start a compatible MemCache MetaService separately at `tcp://127.0.0.1:5006`
with config store `tcp://127.0.0.1:6006`. The scripts retain the Wiki's
`world_size=4` and `dram_size=30GB`; check these against the installed MemCache
version. Serve also retains the Wiki's CPU governor and sysctl tuning, which
requires sufficient permissions and affects the host.

```bash
# On A5, start the server; use hicache_A3.sh on A3.
MODEL_PATH=/path/to/DeepSeek-V4-Flash \
ASCEND_RT_VISIBLE_DEVICES=4,5,6,7 \
  bash sglang/hicache/hicache_A5.sh serve

# After the server is ready, run in another terminal with the same environment.
MODEL_PATH=/path/to/DeepSeek-V4-Flash \
  bash sglang/hicache/hicache_A5.sh bench
```

No argument means `serve`; `--help` prints usage without touching the host.
Only the model path and device list have environment overrides. Edit other
Wiki paths/settings in the scripts for the target machine. Server logs are
written to `sglang_server_A3_<timestamp>.log` or
`sglang_server_A5_<timestamp>.log` in the working directory.
The Wiki's detached logging pipeline is fixed, model paths are quoted, and
the duplicate benchmark `--backend` flag is removed. Pipeline failures return
a nonzero exit status.

Review the benchmark's `Storage (NpuMemcacheStore)` cache-hit count. These
scripts do not automatically assert an L3 hit or output consistency and have
not been run on NPU hardware here. They do not call the older PD round-trip
launcher; `CONFIG_FILE`, `DRY_RUN` and `KEEP_SERVERS` belong to that older
launcher. The remaining sections below document that separate test.

## Quick start

Run these commands from the root of the `self_test` repository:

```bash
cp sglang/hicache/example.env sglang/hicache/local.env
vim sglang/hicache/local.env

CONFIG_FILE=sglang/hicache/local.env \
  bash sglang/hicache/run_npu_memcache_l2_l3.sh a3
```

For an A5 machine, replace the final `a3` with `a5`.

The launcher performs preflight checks, starts every required service, runs the
validation, writes result artifacts, and stops the services. A successful run
ends with `VALIDATION PASSED` and exits with status `0`.

## What the test proves

The validation exercises a real cache round trip:

```text
Cold request
    -> SGLang L1 device cache
    -> HiCache L2 host cache
    -> NPU MemCache L3

Flush L1 and L2, preserve L3
    -> replay the same prompt
    -> restore KV cache from L3
    -> compare generated output
```

The run passes only when all of the following are true:

1. The cold request reports no L3 storage hit.
2. The replay restores at least 256 prompt tokens from the `storage` source.
3. The cold and restored requests return identical output token IDs.
4. The cold and restored requests return identical output text.

A server that merely starts or returns a response is not considered a pass.

## Files

| File | Purpose |
| --- | --- |
| [`run_npu_memcache_l2_l3.sh`](run_npu_memcache_l2_l3.sh) | Starts MemCache, Prefill, Decode, and the PD router, then runs the validation. |
| [`verify_npu_memcache_l2_l3.py`](verify_npu_memcache_l2_l3.py) | Clears caches, sends deterministic requests, checks L3 evidence, and writes the summary. |
| [`example.env`](example.env) | Copy-and-edit configuration template. |

## Prerequisites

Prepare one Linux host with:

- Four available Ascend NPUs: two for Prefill and two for Decode by default.
- An SGLang NPU environment that can import `sglang` and `sglang_router`.
- `memcache_hybrid` with support for the selected device transport.
- A model supported by the SGLang NPU backend.
- Python 3.10 or newer, Bash, and `curl`.
- Free local ports for Prefill, Decode, the router, and MemCache services.

The scripts were prepared against SGLang `upstream/main` commit
`ed0b8f433f887995b1ce361b15aaf9f6ca93483f`. If a newer revision changes a
HiCache flag or API response, record the tested revision with the result.

Before launching, verify the Python environment:

```bash
python -c "import memcache_hybrid, sglang, sglang_router; print('dependencies OK')"
```

## Step 1: Create the machine configuration

Copy the committed template. `local.env` is ignored by Git so machine-specific
paths and addresses are not accidentally committed.

```bash
cp sglang/hicache/example.env sglang/hicache/local.env
vim sglang/hicache/local.env
```

At minimum, review these values:

```bash
MODEL_PATH=/home/weights/Qwen3-32B-W8A8
PREFILL_MF_STORE_URL=tcp://10.0.0.10:31001
DECODE_MF_STORE_URL=tcp://10.0.0.10:31001
PREFILL_DEVICES=12,13
DECODE_DEVICES=14,15
TP_SIZE=2
```

Use the local IP address that MemFabric can reach. `127.0.0.1` may not be valid
for a device transport even though every process runs on the same machine.

Do not set `MEMCACHE_PROTOCOL` unless intentionally overriding the profile:

- `a3` selects `device_sdma`.
- `a5` selects `device_urma`.

`PD_PROTOCOL` controls Prefill-to-Decode KV transfer and is independent from
the HiCache L2/L3 transport. Its default is `device_rdma`.

## Step 2: Check the resolved configuration

Use a dry run before occupying NPUs:

```bash
DRY_RUN=1 \
CONFIG_FILE=sglang/hicache/local.env \
  bash sglang/hicache/run_npu_memcache_l2_l3.sh a3
```

The dry run verifies the model path, required commands, Python packages, device
counts, and service port values, then prints the resolved configuration. It
does not launch a service.

To display the short command reference:

```bash
bash sglang/hicache/run_npu_memcache_l2_l3.sh --help
```

## Step 3: Run the validation

### Ascend A3: device SDMA

```bash
CONFIG_FILE=sglang/hicache/local.env \
  bash sglang/hicache/run_npu_memcache_l2_l3.sh a3
```

### Ascend A5: device URMA

```bash
CONFIG_FILE=sglang/hicache/local.env \
  bash sglang/hicache/run_npu_memcache_l2_l3.sh a5
```

The launcher shows clearly separated stages:

1. Preflight checks and resolved configuration.
2. MemCache configuration generation and MetaService startup.
3. Prefill and Decode server startup and health checks.
4. PD router startup and health check.
5. Cold request, L1/L2 flush, and L3 restore request.
6. Result reporting and process cleanup.

Pressing `Ctrl+C` or encountering a failure triggers cleanup. The launcher sends
`SIGTERM` first and uses `SIGKILL` only for an exact child PID that does not stop
within the grace period.

## Step 4: Review the result

The default output directory is:

```text
/home/sxy/logs/sglang-hicache-a3
/home/sxy/logs/sglang-hicache-a5
```

Set `LOG_DIR` in `local.env` to use another location. Important artifacts are:

| Artifact | What to inspect |
| --- | --- |
| `summary.json` | Final `passed` value, cache details, token match, and errors. |
| `cold-response.json` | Baseline response after all cache levels are cleared. |
| `l3-restore-response.json` | Response after L1/L2 are flushed and KV is restored from L3. |
| `metrics-after-backup.txt` | HiCache backup metrics before the L1/L2 flush. |
| `metrics-after-restore.txt` | Prefetch and restore metrics after the replay. |
| `memcache-meta.log` | NPU MemCache MetaService output. |
| `sglang-prefill.log` | Prefill server and HiCache storage activity. |
| `sglang-decode.log` | Decode server output. |
| `sglang-router.log` | PD router output. |

The primary machine-readable result is:

```bash
python -m json.tool /home/sxy/logs/sglang-hicache-a3/summary.json
```

A passing summary contains `"passed": true` and a positive
`l3_restore.cached_tokens_details.storage` value of at least 256.

## Useful controls

| Variable | Default | Use |
| --- | --- | --- |
| `CONFIG_FILE` | Unset | Load reusable machine settings from an env file. |
| `MODEL_PATH` | `/home/weights/Qwen3-32B-W8A8` | Select the local model. |
| `PREFILL_DEVICES` | `12,13` | Select the two Prefill NPUs. |
| `DECODE_DEVICES` | `14,15` | Select the two Decode NPUs. |
| `TP_SIZE` | `2` | Set tensor parallelism for each server. |
| `PD_PROTOCOL` | `device_rdma` | Select Prefill-to-Decode transfer. |
| `MEMCACHE_PROTOCOL` | Profile default | Override the HiCache L2/L3 transport. |
| `HICACHE_SIZE_GB` | `4` | Set L2 host cache capacity per rank. |
| `LOG_DIR` | Profile-specific path | Select the output and log directory. |
| `STARTUP_TIMEOUT` | `1200` | Set the health-check timeout in seconds. |
| `DRY_RUN` | `0` | Check and print configuration without launching. |
| `KEEP_SERVERS` | `0` | Keep services alive after a successful validation. |

For `device_urma`, MemCache requires
`MEMCACHE_MAX_DRAM_SIZE * MEMCACHE_WORLD_SIZE > 32 TiB`. The defaults define a
64 TiB virtual address span; they do not eagerly allocate 64 TiB of memory.

## Keep the deployment running

To inspect endpoints or run additional requests after the automatic validation:

```bash
KEEP_SERVERS=1 \
CONFIG_FILE=sglang/hicache/local.env \
  bash sglang/hicache/run_npu_memcache_l2_l3.sh a3
```

The launcher prints the service PIDs. Stop those exact PIDs gracefully when
finished; do not use a broad process-name kill on a shared host.

## Troubleshooting

### A service exits before its health check

Read the log path printed immediately before the failure. The launcher also
prints the final 100 lines of the failed service log.

Common causes are an incorrect model path, occupied NPU, unavailable port,
missing Python package, or an unsupported SGLang command-line flag.

### MemFabric or transport initialization fails

Verify that `PREFILL_MF_STORE_URL` and `DECODE_MF_STORE_URL` use an interface
visible to the selected devices. Confirm that the installed `memcache_hybrid`
build supports `device_sdma` on A3 or `device_urma` on A5.

### The request succeeds but the test reports no L3 hit

Inspect `sglang-prefill.log`, `metrics-after-backup.txt`, and
`metrics-after-restore.txt`. Check that:

- The Prefill server started with `npu_memcache` as its storage backend.
- Write-through backup completed before the replay.
- The second request used exactly the same prompt.
- `/flush_cache` cleared L1/L2 without clearing the storage backend.

Increase the write-through settling time by running the verifier manually with
a larger `--settle-seconds` value only after confirming the services are still
running.

### URMA reports an address-space configuration error

Keep `MEMCACHE_MAX_DRAM_SIZE=256GB` and `MEMCACHE_WORLD_SIZE=256`, or choose
values whose product is greater than 32 TiB.

## Upstream references

- [SGLang NPU MemCache backend](https://github.com/sgl-project/sglang/tree/main/python/sglang/srt/mem_cache/storage/npu_memcache)
- [Ascend MemCache configuration](https://gitcode.com/Ascend/memcache/blob/master/doc/memcache_config.md)
- [Ascend MemCache Python API](https://gitcode.com/Ascend/memcache/blob/master/doc/memcache_python_api.md)
