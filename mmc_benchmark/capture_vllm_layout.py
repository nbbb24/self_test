import argparse
import hashlib
import json
import math
import subprocess
from pathlib import Path
from types import SimpleNamespace

import torch
from vllm.config import set_current_vllm_config
from vllm.engine.arg_utils import EngineArgs
from vllm.v1.attention.backends.utils import (
    get_supported_kv_cache_layouts,
    resolve_kv_cache_layout,
)
from vllm.v1.kv_cache_interface import UniformTypeKVCacheSpecs
from vllm_ascend.attention.dsa_v1 import (
    AscendDSAC4Backend,
    AscendDSAC128Backend,
    AscendDSASWABackend,
)
from vllm_ascend.attention.indexer import AscendSFAIndexerBackend
from vllm_ascend.attention.sfa_v1 import AscendSFABackend
from vllm_ascend.core.kv_cache_config_builder import AscendKVCacheConfigBuilder
from vllm_ascend.core.kv_cache_interface import (
    AscendMLAAttentionSpec,
    AscendSFAIndexerCacheSpec,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.coordinator import (
    AscendStoreCoordinator,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.metadata import (
    infer_cacheable_group_ids,
    infer_group_cache_families,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker import (
    KVPoolWorker,
)
from vllm_ascend.models.deepseek_v4.indexer import (
    AscendDeepseekV4IndexerCache,
    get_indexer_k_dtype,
)
from vllm_ascend.models.deepseek_v4.model import AscendDeepseekV4SWACache
from vllm_ascend.models.layer.attention.layer import DSAAttention
from vllm_ascend.worker.v2 import attn_utils
from vllm_layout import export_layout, load_layout


def main():
    parser = argparse.ArgumentParser(description="Construct bulk vLLM cache views without loading weights.")
    parser.add_argument("--model", choices=["glm-5.2", "deepseek-v4-flash"], required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--block-size", type=int, default=128)
    args = parser.parse_args()
    if min(args.batch_size, args.block_size) <= 0:
        parser.error("batch-size and block-size must be positive")
    cfg = EngineArgs(
        model=args.model_path,
        block_size=args.block_size,
        enforce_eager=True,
        tensor_parallel_size=1,
        max_model_len=32768,
    ).create_engine_config()
    hf = cfg.model_config.hf_text_config
    specs = {}
    backends = {}
    if args.model == "glm-5.2":
        for layer in range(hf.num_hidden_layers):
            name = f"model.layers.{layer}.self_attn.attn"
            attn = attn_utils.MLAAttention.__new__(attn_utils.MLAAttention)
            torch.nn.Module.__init__(attn)
            attn.kv_lora_rank, attn.qk_rope_head_dim = hf.kv_lora_rank, hf.qk_rope_head_dim
            attn.impl = SimpleNamespace(use_flashmla=False)
            cfg.compilation_config.static_forward_context[name] = attn
            specs[name] = AscendMLAAttentionSpec(
                block_size=args.block_size,
                num_kv_heads=1,
                head_size=hf.kv_lora_rank + hf.qk_rope_head_dim,
                dtype=cfg.model_config.dtype,
                cache_dtype_str="auto",
                num_heads=hf.num_attention_heads,
            )
            backends[name] = AscendSFABackend
            if hf.indexer_types[layer] == "full":
                name = f"model.layers.{layer}.self_attn.indexer.k_cache"
                specs[name] = AscendSFAIndexerCacheSpec(
                    block_size=args.block_size,
                    num_kv_heads=1,
                    head_size=hf.index_head_dim,
                    dtype=torch.float8_e4m3fn,
                    scale_dim=1,
                    scale_dtype=torch.float32,
                    cache_sparse_li_c8=True,
                    cache_dtype_str="auto",
                )
                backends[name] = AscendSFAIndexerBackend
        tokens = args.batch_size * args.block_size
    else:
        for layer in range(hf.num_hidden_layers):
            ratio = hf.compress_ratios[layer]
            if ratio:
                name = f"model.layers.{layer}.self_attn.attn"
                specs[name] = DSAAttention.get_kv_cache_spec(
                    SimpleNamespace(
                        compress_ratio=ratio,
                        head_size=hf.head_dim,
                        n_local_heads=hf.num_attention_heads,
                        kv_cache_dtype="auto",
                    ),
                    cfg,
                )
                backends[name] = AscendDSAC4Backend if ratio == 4 else AscendDSAC128Backend
                if ratio == 4:
                    name = f"model.layers.{layer}.self_attn.indexer.k_cache"
                    specs[name] = AscendDeepseekV4IndexerCache.get_kv_cache_spec(
                        SimpleNamespace(
                            compress_ratio=ratio,
                            head_dim=hf.index_head_dim,
                            dtype=get_indexer_k_dtype(cfg),
                            cache_config=cfg.cache_config,
                        ),
                        cfg,
                    )
                    backends[name] = AscendDSAC4Backend
            name = f"model.layers.{layer}.self_attn.swa_cache"
            specs[name] = AscendDeepseekV4SWACache.get_kv_cache_spec(
                SimpleNamespace(
                    head_dim=hf.head_dim,
                    dtype=cfg.model_config.dtype,
                    block_size=args.block_size,
                    window_size=hf.sliding_window,
                    cache_config=cfg.cache_config,
                ),
                cfg,
            )
            backends[name] = AscendDSASWABackend
        tokens = args.batch_size * args.block_size * min(r for r in hf.compress_ratios if r)
    common = [layout.name for layout in get_supported_kv_cache_layouts(list(dict.fromkeys(backends.values())))]
    resolve_kv_cache_layout(cfg, [common], specs.values())
    builder = AscendKVCacheConfigBuilder()
    with set_current_vllm_config(cfg):
        groups = builder.get_kv_cache_groups(cfg, specs)
        cacheable = infer_cacheable_group_ids(groups)
        counts = {}
        num_blocks = 0
        granularity = math.lcm(*(groups[i].kv_cache_spec.block_size for i in cacheable))
        if tokens % granularity:
            raise ValueError(f"ERROR request tokens={tokens} must be divisible by transfer granularity={granularity}")
        families = infer_group_cache_families(groups, getattr(hf, "compress_ratios", None), hf)
        masks = None
        if len(groups) > 1:
            coordinator = AscendStoreCoordinator(
                groups, granularity, args.block_size, [g.kv_cache_spec.block_size for g in groups], families
            )
            masks = coordinator.store_mask(tokens, tokens)
        selected_chunks = {
            i: [j for j in range(tokens // groups[i].kv_cache_spec.block_size) if masks is None or masks[i][j]]
            for i in cacheable
        }
        for i in cacheable:
            count = len(selected_chunks[i])
            if count == 0:
                continue
            counts[i] = (num_blocks, count)
            num_blocks += count
        cache_config = builder.get_kv_cache_config_from_groups(cfg, groups, num_blocks)
        raw = attn_utils._allocate_kv_cache(cache_config, {}, torch.device("cpu"))
        attn_groups = []
        for i, group in enumerate(groups):
            for name in group.layer_names:
                spec = (
                    group.kv_cache_spec.kv_cache_specs[name]
                    if isinstance(group.kv_cache_spec, UniformTypeKVCacheSpecs)
                    else group.kv_cache_spec
                )
                attn_groups.append(attn_utils.AttentionGroup(backends[name], [name], spec, i))
        caches = attn_utils._reshape_kv_cache_v2(
            attn_groups, raw, "auto", [g.kv_cache_spec.block_size for g in groups], {}, cache_config
        )
    worker = KVPoolWorker.__new__(KVPoolWorker)
    worker.kv_caches, worker.num_blocks = caches, num_blocks
    worker.use_layerwise, worker.use_kvpp = False, False
    worker.hf_config, worker.num_layers = hf, hf.num_hidden_layers
    worker._global_to_local_layer = {}
    worker.grouped_block_size = [g.kv_cache_spec.block_size for g in groups]
    worker.cache_transfer_granularity = granularity
    worker._extra_config = {"discard_partial_chunks": True}
    for name in (
        "group_kv_caches_base_addr",
        "group_block_len",
        "group_block_stride",
        "group_layer_cache_entry_offsets",
        "group_cache_entry_segments",
        "group_num_layers",
    ):
        setattr(worker, name, {})
    for i in counts:
        worker._infer_cache_group_metadata(i, groups[i].layer_names)
    repo = Path(attn_utils.__file__).parents[3]
    metadata = {
        "model": args.model,
        "batch_size": args.batch_size,
        "block_size": args.block_size,
        "request_tokens": tokens,
        "transfer_granularity": granularity,
        "tp": 1,
        "pp": 1,
        "mode": "bulk",
        "main_cache_dtype": str(cfg.model_config.dtype),
        "indexer_dtype": "fp8",
        "construction": "vLLM spec/planner/allocator/reshape and KVPoolWorker metadata; no weights or forward",
        "vllm_ascend_commit": subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip(),
        "model_config_sha256": hashlib.sha256((Path(args.model_path) / "config.json").read_bytes()).hexdigest(),
        "group_families": families,
        "selected_chunks": selected_chunks,
        "group_blocks": {
            i: {"start": start, "count": count, "tokens_per_block": groups[i].kv_cache_spec.block_size}
            for i, (start, count) in counts.items()
        },
    }
    source_files = [
        Path(attn_utils.__file__),
        Path(__import__(KVPoolWorker.__module__, fromlist=[""]).__file__),
        Path(__import__(DSAAttention.__module__, fromlist=[""]).__file__),
        Path(__import__(AscendKVCacheConfigBuilder.__module__, fromlist=[""]).__file__),
    ]
    metadata["source_sha256"] = {
        str(p.relative_to(repo)): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files
    }
    export_layout(worker, args.output, metadata, counts)
    layout = load_layout(args.output, args.batch_size)
    print(
        json.dumps(
            {
                "metadata": metadata,
                "groups": [
                    {
                        "group": g["group"],
                        "keys": g["keys"],
                        "segments": len(g["segments"]),
                        "bytes_per_key": sum(s["bytes"] for s in g["segments"]),
                    }
                    for g in layout["groups"]
                ],
            }
        )
    )


if __name__ == "__main__":
    main()
