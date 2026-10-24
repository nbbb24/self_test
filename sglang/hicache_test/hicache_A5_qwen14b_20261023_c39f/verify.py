"""Verify a real HiCache L2 -> L3 -> L2 round trip on an NPU server.

The server must have HiCache enabled with ``npu_memcache`` as its storage
backend. This helper deliberately clears L1/L2 between two identical requests
while leaving L3 intact. The second response must therefore report storage
cache hits, and its output token IDs must match the first response.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any


# =============================================================================
# HTTP HELPERS
# =============================================================================

def _post_json(url: str, body: dict[str, Any] | None, timeout: float) -> Any:
    data = None if body is None else json.dumps(body).encode("utf-8")
    headers = {} if data is None else {"Content-Type": "application/json"}
    request = urllib.request.Request(url, data=data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"POST {url} failed: HTTP {exc.code}: {detail}") from exc
    return json.loads(payload) if payload.strip().startswith(("{", "[")) else payload


def _get_text(url: str, timeout: float) -> str:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return response.read().decode("utf-8")


def _generate(router_url: str, prompt: str, max_tokens: int, timeout: float) -> dict:
    response = _post_json(
        f"{router_url}/generate",
        {
            "text": prompt,
            "sampling_params": {
                "temperature": 0.0,
                "max_new_tokens": max_tokens,
                "ignore_eos": True,
            },
        },
        timeout,
    )
    if not isinstance(response, dict) or "meta_info" not in response:
        raise RuntimeError(f"Unexpected /generate response: {response!r}")
    return response


# =============================================================================
# RESULT AND METRICS HELPERS
# =============================================================================

def _cache_stats(response: dict) -> tuple[int, int, dict[str, int]]:
    meta = response.get("meta_info", {})
    prompt_tokens = int(meta.get("prompt_tokens", 0))
    cached_tokens = int(meta.get("cached_tokens", 0))
    raw_details = meta.get("cached_tokens_details") or {}
    details = {key: int(raw_details.get(key, 0)) for key in ("device", "host", "storage")}
    return prompt_tokens, cached_tokens, details


def _metric_evidence(metrics: str) -> list[str]:
    names = (
        "sglang:backup_pgs",
        "sglang:prefetch_pgs",
        "sglang:backup_bandwidth",
        "sglang:prefetch_bandwidth",
        "sglang:storage_prefetched_tokens_total",
        "sglang:storage_backuped_tokens_total",
    )
    return [
        line
        for line in metrics.splitlines()
        if not line.startswith("#") and any(name in line for name in names)
    ]


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


# =============================================================================
# END-TO-END VALIDATION FLOW
# =============================================================================

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--router-url", default="http://127.0.0.1:8000")
    parser.add_argument("--prefill-url", default="http://127.0.0.1:30000")
    parser.add_argument("--prompt-repeat", type=int, default=96)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--min-storage-tokens", type=int, default=256)
    parser.add_argument("--settle-seconds", type=float, default=5.0)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    args.router_url = args.router_url.rstrip("/")
    args.prefill_url = args.prefill_url.rstrip("/")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # A deterministic prompt that is comfortably above the default HiCache L3
    # prefetch threshold and spans many cache pages.
    paragraph = (
        "大模型推理分为 Prefill 和 Decode 两个阶段。Prefill 并行处理输入并生成 "
        "KV Cache，Decode 逐 token 读取已有缓存。分层缓存会在设备内存、主机内存和 "
        "远端存储之间搬运数据，以降低重复长前缀请求的首 token 延迟。"
    )
    prompt = (paragraph + "\n") * args.prompt_repeat
    prompt += "请用三点总结以上内容。"

    print("[1/5] Clearing L3 and flushing L1/L2 for a clean baseline ...", flush=True)
    _post_json(
        f"{args.prefill_url}/hicache/storage-backend/clear", None, args.timeout
    )
    _post_json(
        f"{args.prefill_url}/flush_cache?{urllib.parse.urlencode({'timeout': 60})}",
        None,
        args.timeout,
    )

    print("[2/5] Sending the cold request (expected L3 miss) ...", flush=True)
    cold = _generate(args.router_url, prompt, args.max_tokens, args.timeout)
    _write_json(args.output_dir / "cold-response.json", cold)
    cold_prompt, cold_cached, cold_details = _cache_stats(cold)
    print(
        f"      cold: cached={cold_cached}/{cold_prompt}, details={cold_details}",
        flush=True,
    )
    if cold_details.get("storage", 0) != 0:
        raise AssertionError(
            "Cold request unexpectedly hit L3 after the storage backend was cleared: "
            f"{cold_details}"
        )

    print(
        f"[3/5] Waiting {args.settle_seconds:g}s for write-through backup ...",
        flush=True,
    )
    time.sleep(args.settle_seconds)
    before_flush_metrics = _get_text(f"{args.prefill_url}/metrics", args.timeout)
    (args.output_dir / "metrics-after-backup.txt").write_text(
        before_flush_metrics, encoding="utf-8"
    )

    # /flush_cache resets the radix tree, device allocator, and HiCache host pool.
    # It intentionally does not call the storage backend's remove_all().
    print("[4/5] Flushing L1/L2 while preserving MemCache L3 ...", flush=True)
    _post_json(
        f"{args.prefill_url}/flush_cache?{urllib.parse.urlencode({'timeout': 60})}",
        None,
        args.timeout,
    )

    print("[5/5] Replaying the prompt (must restore from L3) ...", flush=True)
    warm = _generate(args.router_url, prompt, args.max_tokens, args.timeout)
    _write_json(args.output_dir / "l3-restore-response.json", warm)
    warm_prompt, warm_cached, warm_details = _cache_stats(warm)
    storage_tokens = warm_details.get("storage", 0)
    print(
        f"      restore: cached={warm_cached}/{warm_prompt}, details={warm_details}",
        flush=True,
    )

    after_restore_metrics = _get_text(f"{args.prefill_url}/metrics", args.timeout)
    (args.output_dir / "metrics-after-restore.txt").write_text(
        after_restore_metrics, encoding="utf-8"
    )
    evidence = _metric_evidence(after_restore_metrics)
    if evidence:
        print("      metric evidence:", flush=True)
        for line in evidence:
            print(f"        {line}", flush=True)

    errors: list[str] = []
    if not cold.get('output_ids') or not warm.get('output_ids'):
        errors.append('response is missing nonempty output_ids')
    if storage_tokens < args.min_storage_tokens:
        errors.append(
            f"L3 restored only {storage_tokens} tokens; expected at least "
            f"{args.min_storage_tokens}"
        )
    if cold.get("output_ids") != warm.get("output_ids"):
        errors.append("cold and L3-restored output_ids differ")
    if cold.get("text") != warm.get("text"):
        errors.append("cold and L3-restored output text differs")

    summary = {
        "passed": not errors,
        "cold": {
            "prompt_tokens": cold_prompt,
            "cached_tokens": cold_cached,
            "cached_tokens_details": cold_details,
        },
        "l3_restore": {
            "prompt_tokens": warm_prompt,
            "cached_tokens": warm_cached,
            "cached_tokens_details": warm_details,
        },
        "output_ids_match": cold.get("output_ids") == warm.get("output_ids"),
        "errors": errors,
    }
    _write_json(args.output_dir / "summary.json", summary)

    if errors:
        print("\nFAIL: " + "; ".join(errors), file=sys.stderr)
        return 1

    print(
        f"\nPASS: restored {storage_tokens} prompt tokens from NPU MemCache L3; "
        "generated output is identical.",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
