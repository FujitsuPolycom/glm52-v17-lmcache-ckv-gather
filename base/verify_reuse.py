#!/usr/bin/env python3
"""Verify v17 LMCache stores and restores exact-token prompts."""

from __future__ import annotations

import argparse
import json
import math
import re
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass
class RunResult:
    name: str
    prompt_tokens: int
    completion_tokens: int
    ttft_seconds: float | None
    elapsed_seconds: float
    output: str


def request(
    url: str,
    payload: dict[str, Any] | None = None,
    *,
    method: str | None = None,
    timeout: float = 900,
) -> bytes:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        method=method or ("POST" if data is not None else "GET"),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            return response.read()
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"{exc.code} from {url}: {body}") from exc


def request_json(
    url: str,
    payload: dict[str, Any] | None = None,
    *,
    method: str | None = None,
    timeout: float = 900,
) -> dict[str, Any]:
    return json.loads(request(url, payload, method=method, timeout=timeout))


def make_exact_tokens(api: str, model: str, target: int, salt: str) -> list[int]:
    records = max(256, math.ceil(target / 12))
    while True:
        text = "".join(
            f"{salt} record {index:07d}: alpha beta gamma delta epsilon zeta eta theta.\n"
            for index in range(records)
        )
        tokenized = request_json(
            f"{api}/tokenize", {"model": model, "prompt": text}, timeout=300
        )
        tokens = tokenized["tokens"]
        if len(tokens) >= target:
            return tokens[:target]
        records = math.ceil(records * target / max(1, len(tokens)) * 1.05)


def stream_completion(
    api: str,
    model: str,
    prompt: list[int],
    name: str,
    max_tokens: int,
) -> RunResult:
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "seed": 1,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    req = urllib.request.Request(
        f"{api}/v1/completions",
        data=json.dumps(payload).encode("utf-8"),
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    started = time.perf_counter()
    first_token_at: float | None = None
    output: list[str] = []
    usage: dict[str, int] = {}
    try:
        with urllib.request.urlopen(req, timeout=1800) as response:
            for raw_line in response:
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line.startswith("data: "):
                    continue
                body = line[6:]
                if body == "[DONE]":
                    break
                event = json.loads(body)
                if event.get("usage"):
                    usage = event["usage"]
                for choice in event.get("choices", []):
                    text = choice.get("text", "")
                    if text:
                        if first_token_at is None:
                            first_token_at = time.perf_counter()
                        output.append(text)
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"{exc.code} from completion API: {body}") from exc

    finished = time.perf_counter()
    return RunResult(
        name=name,
        prompt_tokens=int(usage.get("prompt_tokens", len(prompt))),
        completion_tokens=int(usage.get("completion_tokens", 0)),
        ttft_seconds=None if first_token_at is None else first_token_at - started,
        elapsed_seconds=finished - started,
        output="".join(output),
    )


METRICS = (
    "vllm:prefix_cache_queries_total",
    "vllm:prefix_cache_hits_total",
    "vllm:external_prefix_cache_queries_total",
    "vllm:external_prefix_cache_hits_total",
    "vllm:num_preemptions_total",
)


def vllm_metrics(api: str) -> dict[str, float]:
    text = request(f"{api}/metrics", timeout=30).decode("utf-8")
    values: dict[str, float] = {}
    for name in METRICS:
        match = re.search(rf"^{re.escape(name)}(?:\{{[^}}]*\}})?\s+(\S+)$", text, re.M)
        values[name] = 0.0 if match is None else float(match.group(1))
    return values


def metric_delta(before: dict[str, float], after: dict[str, float]) -> dict[str, float]:
    return {name: after[name] - before[name] for name in METRICS}


def cache_status(cache_api: str) -> dict[str, Any]:
    return request_json(f"{cache_api}/status", timeout=30)


def wait_for_objects(cache_api: str, minimum: int, timeout: float = 60) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    latest: dict[str, Any] = {}
    while time.monotonic() < deadline:
        latest = cache_status(cache_api)
        count = latest["storage_manager"]["l1_manager"]["total_object_count"]
        if count >= minimum:
            return latest
        time.sleep(1)
    raise RuntimeError(f"LMCache object count did not reach {minimum}: {latest}")


def print_run(result: RunResult) -> None:
    ttft = "n/a" if result.ttft_seconds is None else f"{result.ttft_seconds:.3f}s"
    print(
        f"{result.name:>12}: prompt={result.prompt_tokens:,} "
        f"completion={result.completion_tokens:,} TTFT={ttft} "
        f"elapsed={result.elapsed_seconds:.3f}s"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--api", default="http://127.0.0.1:5001")
    parser.add_argument("--cache-api", default="http://127.0.0.1:8088")
    parser.add_argument("--model", default="GLM-5.2")
    parser.add_argument("--sentinel-tokens", type=int, default=8192)
    parser.add_argument("--run-id", default="default")
    parser.add_argument("--max-tokens", type=int, default=8)
    parser.add_argument("--evict-requests", type=int, default=0)
    parser.add_argument("--evict-tokens", type=int, default=238080)
    parser.add_argument("--require-external-hit", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    request(f"{args.cache_api}/clear-cache", method="POST", timeout=60)
    baseline_status = cache_status(args.cache_api)
    registered = baseline_status.get("registered_gpu_ids", [])
    if len(registered) != 4:
        raise RuntimeError(f"expected four registered GPU contexts, got {registered}")

    sentinel = make_exact_tokens(
        args.api,
        args.model,
        args.sentinel_tokens,
        f"glm52-lmcache-sentinel-{args.run_id}",
    )
    metrics_before = vllm_metrics(args.api)
    cold = stream_completion(
        args.api, args.model, sentinel, "cold", args.max_tokens
    )
    after_cold = wait_for_objects(args.cache_api, 1)
    objects_after_cold = after_cold["storage_manager"]["l1_manager"]
    print_run(cold)
    print(
        f"       cache: objects={objects_after_cold['total_object_count']:,} "
        f"bytes={objects_after_cold['memory_used_bytes']:,}"
    )

    fills: list[RunResult] = []
    for index in range(args.evict_requests):
        prompt = make_exact_tokens(
            args.api,
            args.model,
            args.evict_tokens,
            f"glm52-lmcache-evict-{args.run_id}-{index}",
        )
        result = stream_completion(
            args.api, args.model, prompt, f"evict-{index + 1}", args.max_tokens
        )
        fills.append(result)
        print_run(result)

    objects_before_replay = cache_status(args.cache_api)["storage_manager"]["l1_manager"]
    metrics_before_replay = vllm_metrics(args.api)
    replay = stream_completion(
        args.api, args.model, sentinel, "replay", args.max_tokens
    )
    print_run(replay)
    metrics_after = vllm_metrics(args.api)
    replay_delta = metric_delta(metrics_before_replay, metrics_after)
    total_delta = metric_delta(metrics_before, metrics_after)
    final_status = cache_status(args.cache_api)
    final_l1 = final_status["storage_manager"]["l1_manager"]

    print("replay metric deltas:")
    for name, value in replay_delta.items():
        print(f"  {name}: {value:,.0f}")
    print(
        f"final cache: objects={final_l1['total_object_count']:,} "
        f"bytes={final_l1['memory_used_bytes']:,}"
    )

    external_hits = replay_delta["vllm:external_prefix_cache_hits_total"]
    outputs_match = cold.output == replay.output
    print(f"deterministic output match: {outputs_match}")
    print(f"external replay hits: {external_hits:,.0f} tokens")

    report = {
        "configuration": vars(args) | {"output": str(args.output) if args.output else None},
        "cold": asdict(cold),
        "fills": [asdict(result) for result in fills],
        "replay": asdict(replay),
        "replay_metric_delta": replay_delta,
        "total_metric_delta": total_delta,
        "objects_before_replay": objects_before_replay,
        "final_l1": final_l1,
        "registered_gpu_ids": registered,
        "outputs_match": outputs_match,
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")

    if not outputs_match:
        print("FAIL: deterministic cold and replay outputs differ")
        return 1
    if args.require_external_hit and external_hits <= 0:
        print("FAIL: replay did not report an external LMCache hit")
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
