#!/usr/bin/env python3
"""Benchmark cold, GPU-local, and LMCache-restored GLM-5.2 prompts."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from verify_reuse import (
    RunResult,
    cache_status,
    make_exact_tokens,
    metric_delta,
    request,
    stream_completion,
    vllm_metrics,
    wait_for_objects,
)


def parse_contexts(value: str) -> list[int]:
    suffixes = {"k": 1024, "m": 1024 * 1024}
    contexts: list[int] = []
    for item in value.split(","):
        text = item.strip().lower()
        if not text:
            continue
        multiplier = suffixes.get(text[-1], 1)
        number = text[:-1] if multiplier != 1 else text
        context = int(number) * multiplier
        if context <= 0 or context % 512:
            raise argparse.ArgumentTypeError(
                f"context must be a positive multiple of 512 tokens: {item}"
            )
        contexts.append(context)
    if not contexts:
        raise argparse.ArgumentTypeError("at least one context is required")
    if len(contexts) != len(set(contexts)):
        raise argparse.ArgumentTypeError("contexts must be unique")
    return contexts


def output_digest(result: RunResult) -> str:
    return hashlib.sha256(result.output.encode("utf-8")).hexdigest()


def run_phase(
    api: str,
    model: str,
    prompt: list[int],
    name: str,
    max_tokens: int,
    bytes_per_token: int,
) -> tuple[RunResult, dict[str, Any]]:
    before = vllm_metrics(api)
    result = stream_completion(api, model, prompt, name, max_tokens)
    after = vllm_metrics(api)
    delta = metric_delta(before, after)
    ttft = result.ttft_seconds
    external_hits = int(delta["vllm:external_prefix_cache_hits_total"])
    restored_bytes = external_hits * bytes_per_token
    phase = {
        **asdict(result),
        "output_sha256": output_digest(result),
        "metric_delta": delta,
        "effective_context_tokens_per_second": (
            None if not ttft else result.prompt_tokens / ttft
        ),
        "restored_payload_bytes": restored_bytes,
        "effective_restore_gib_per_second": (
            None if not ttft else restored_bytes / ttft / (1024**3)
        ),
    }
    return result, phase


def describe_phase(label: str, phase: dict[str, Any]) -> None:
    ttft = phase["ttft_seconds"]
    ttft_text = "n/a" if ttft is None else f"{ttft:.3f}s"
    metrics = phase["metric_delta"]
    print(
        f"  {label:<8} TTFT={ttft_text:>9} "
        f"local={metrics['vllm:prefix_cache_hits_total']:>9,.0f} "
        f"external={metrics['vllm:external_prefix_cache_hits_total']:>9,.0f}",
        flush=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--api", default="http://127.0.0.1:5001")
    parser.add_argument("--cache-api", default="http://127.0.0.1:8088")
    parser.add_argument("--model", default="GLM-5.2")
    parser.add_argument(
        "--contexts",
        type=parse_contexts,
        default=parse_contexts("8k,32k,64k,128k"),
    )
    parser.add_argument("--run-id", default=datetime.now().strftime("%Y%m%d-%H%M%S"))
    parser.add_argument("--max-tokens", type=int, default=8)
    parser.add_argument("--evict-requests", type=int, default=2)
    parser.add_argument("--evict-tokens", type=int, default=238080)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.evict_requests < 1:
        parser.error("--evict-requests must be at least one")
    if args.evict_tokens + args.max_tokens > 262144:
        parser.error("eviction request exceeds the live 262,144-token model limit")

    request(f"{args.cache_api}/clear-cache", method="POST", timeout=60)
    baseline_status = cache_status(args.cache_api)
    gpu_meta = baseline_status.get("gpu_context_meta", {})
    registered = baseline_status.get("registered_gpu_ids", [])
    if len(registered) != 4 or len(gpu_meta) != 4:
        raise RuntimeError(
            f"expected four registered GPU contexts, got ids={registered}, meta={gpu_meta}"
        )

    bytes_per_token = sum(
        int(meta["kv_cache_layout"]["cache_size_per_token"])
        for meta in gpu_meta.values()
    )
    chunk_size = int(baseline_status["chunk_size"])
    world_size = int(next(iter(gpu_meta.values()))["world_size"])
    expected_objects = 0

    prompts: dict[int, list[int]] = {}
    rows: list[dict[str, Any]] = []
    print(
        f"GLM-5.2 LMCache matrix: contexts={args.contexts}, "
        f"payload={bytes_per_token:,} bytes/token across {world_size} ranks",
        flush=True,
    )

    for context in args.contexts:
        print(f"\n{context:,} tokens", flush=True)
        prompt = make_exact_tokens(
            args.api,
            args.model,
            context,
            f"glm52-lmcache-matrix-{args.run_id}-{context}",
        )
        prompts[context] = prompt
        cold_result, cold = run_phase(
            args.api,
            args.model,
            prompt,
            f"cold-{context}",
            args.max_tokens,
            bytes_per_token,
        )
        expected_objects += math.ceil(context / chunk_size) * world_size
        wait_for_objects(args.cache_api, expected_objects, timeout=180)
        describe_phase("cold", cold)

        local_result, local = run_phase(
            args.api,
            args.model,
            prompt,
            f"local-{context}",
            args.max_tokens,
            bytes_per_token,
        )
        describe_phase("GPU", local)
        rows.append(
            {
                "context_tokens": context,
                "cold": cold,
                "local_gpu_replay": local,
                "external_lmcache_replay": None,
                "cold_local_outputs_match": cold_result.output == local_result.output,
            }
        )

    fills: list[dict[str, Any]] = []
    print("\nEvicting every sentinel from GPU KV...", flush=True)
    for index in range(args.evict_requests):
        prompt = make_exact_tokens(
            args.api,
            args.model,
            args.evict_tokens,
            f"glm52-lmcache-matrix-evict-{args.run_id}-{index}",
        )
        result, phase = run_phase(
            args.api,
            args.model,
            prompt,
            f"evict-{index + 1}",
            args.max_tokens,
            bytes_per_token,
        )
        expected_objects += math.ceil(args.evict_tokens / chunk_size) * world_size
        wait_for_objects(args.cache_api, expected_objects, timeout=300)
        fills.append(phase)
        print(
            f"  fill {index + 1}: {result.prompt_tokens:,} tokens, "
            f"TTFT={result.ttft_seconds:.3f}s",
            flush=True,
        )

    print("\nReplaying from host RAM...", flush=True)
    for row in rows:
        context = int(row["context_tokens"])
        external_result, external = run_phase(
            args.api,
            args.model,
            prompts[context],
            f"external-{context}",
            args.max_tokens,
            bytes_per_token,
        )
        describe_phase(f"RAM {context // 1024}K", external)
        row["external_lmcache_replay"] = external
        row["all_outputs_match"] = (
            row["cold"]["output"]
            == row["local_gpu_replay"]["output"]
            == external_result.output
        )
        cold_ttft = row["cold"]["ttft_seconds"]
        local_ttft = row["local_gpu_replay"]["ttft_seconds"]
        external_ttft = external["ttft_seconds"]
        row["speedup"] = {
            "local_vs_cold": cold_ttft / local_ttft,
            "external_vs_cold": cold_ttft / external_ttft,
        }

    final_status = cache_status(args.cache_api)
    final_l1 = final_status["storage_manager"]["l1_manager"]
    failures: list[str] = []
    for row in rows:
        context = int(row["context_tokens"])
        external = row["external_lmcache_replay"]
        metrics = external["metric_delta"]
        if not row["all_outputs_match"]:
            failures.append(f"{context}: deterministic outputs differ")
        if int(metrics["vllm:prefix_cache_hits_total"]) != 0:
            failures.append(f"{context}: replay still had local GPU-prefix hits")
        if int(metrics["vllm:external_prefix_cache_hits_total"]) != context:
            failures.append(
                f"{context}: external hits were "
                f"{metrics['vllm:external_prefix_cache_hits_total']:.0f}"
            )
        if int(metrics["vllm:num_preemptions_total"]) != 0:
            failures.append(f"{context}: replay preempted")

    report = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "configuration": {
            "api": args.api,
            "cache_api": args.cache_api,
            "model": args.model,
            "contexts": args.contexts,
            "run_id": args.run_id,
            "max_tokens": args.max_tokens,
            "evict_requests": args.evict_requests,
            "evict_tokens": args.evict_tokens,
            "chunk_size": chunk_size,
            "world_size": world_size,
            "cache_payload_bytes_per_token": bytes_per_token,
        },
        "rows": rows,
        "eviction_fills": fills,
        "registered_gpu_ids": registered,
        "final_l1": final_l1,
        "validation": {
            "passed": not failures,
            "failures": failures,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")

    print(
        f"\nLMCache L1: {final_l1['total_object_count']:,} objects, "
        f"{final_l1['memory_used_bytes'] / (1024**3):.2f} GiB",
        flush=True,
    )
    print(f"Report: {args.output}", flush=True)
    if failures:
        for failure in failures:
            print(f"FAIL: {failure}", flush=True)
        return 1
    print("PASS: every context restored fully from host RAM", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
