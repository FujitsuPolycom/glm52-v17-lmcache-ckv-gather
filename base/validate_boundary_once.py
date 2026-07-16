#!/usr/bin/env python3
"""Run one exact-token completion and wait for LMCache stores to drain."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import time

from verify_reuse import (
    cache_status,
    make_exact_tokens,
    metric_delta,
    request,
    request_json,
    stream_completion,
    vllm_metrics,
)


def wait_for_cache_stores_drained(cache_api: str, timeout: float) -> dict:
    deadline = time.monotonic() + timeout
    latest: dict = {}
    while time.monotonic() < deadline:
        latest = cache_status(cache_api)
        storage = latest["storage_manager"]
        l1 = storage["l1_manager"]
        stores = storage["store_controller"]
        if (
            l1["write_locked_count"] == 0
            and stores["pending_keys_count"] == 0
            and stores["in_flight_task_count"] == 0
        ):
            return latest
        time.sleep(0.5)
    raise TimeoutError(f"LMCache stores did not drain within {timeout}s: {latest}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--api", default="http://127.0.0.1:5001")
    parser.add_argument("--cache-api", default="http://127.0.0.1:8088")
    parser.add_argument("--model", default="GLM-5.2")
    parser.add_argument("--prompt-tokens", type=int, default=258_048)
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--max-model-len", type=int, default=262_144)
    parser.add_argument("--cache-idle-timeout", type=float, default=300)
    parser.add_argument("--run-id", default=datetime.now().strftime("%Y%m%d-%H%M%S"))
    parser.add_argument("--clear-cache", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.prompt_tokens <= 0 or args.max_tokens <= 0:
        parser.error("token counts must be positive")
    if args.prompt_tokens + args.max_tokens > args.max_model_len:
        parser.error("prompt plus completion exceeds --max-model-len")

    model_info = request_json(f"{args.api}/v1/models", timeout=30)
    if args.clear_cache:
        request(f"{args.cache_api}/clear-cache", method="POST", timeout=60)

    prompt = make_exact_tokens(
        args.api,
        args.model,
        args.prompt_tokens,
        f"glm52-boundary-{args.run_id}",
    )
    if len(prompt) != args.prompt_tokens:
        raise RuntimeError(f"token targeting failed: {len(prompt)} != {args.prompt_tokens}")

    metrics_before = vllm_metrics(args.api)
    cache_before = cache_status(args.cache_api)
    started_at = datetime.now(timezone.utc)
    result = stream_completion(
        args.api,
        args.model,
        prompt,
        "boundary",
        args.max_tokens,
    )
    cache_after = wait_for_cache_stores_drained(
        args.cache_api, args.cache_idle_timeout
    )
    metrics_after = vllm_metrics(args.api)
    request_json(f"{args.api}/v1/models", timeout=30)

    l1_before = cache_before["storage_manager"]["l1_manager"]
    l1_after = cache_after["storage_manager"]["l1_manager"]
    payload = {
        "started_at": started_at.isoformat(),
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "arguments": {
            "model": args.model,
            "prompt_tokens": args.prompt_tokens,
            "max_tokens": args.max_tokens,
            "max_model_len": args.max_model_len,
            "run_id": args.run_id,
            "clear_cache": args.clear_cache,
        },
        "model_info": model_info,
        "result": asdict(result),
        "metrics_delta": metric_delta(metrics_before, metrics_after),
        "lmcache": {
            "objects_before": l1_before["total_object_count"],
            "objects_after": l1_after["total_object_count"],
            "bytes_before": l1_before["memory_used_bytes"],
            "bytes_after": l1_after["memory_used_bytes"],
            # This build intentionally TTL-retains completed session metadata.
            "retained_sessions_after": cache_after["active_sessions"],
            "store_controller_after": cache_after["storage_manager"][
                "store_controller"
            ],
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")

    ttft = "n/a" if result.ttft_seconds is None else f"{result.ttft_seconds:.3f}s"
    print(
        f"PASS prompt={result.prompt_tokens:,} completion={result.completion_tokens:,} "
        f"TTFT={ttft} elapsed={result.elapsed_seconds:.3f}s "
        f"LMCache objects={l1_after['total_object_count']:,} "
        f"bytes={l1_after['memory_used_bytes']:,}"
    )
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
