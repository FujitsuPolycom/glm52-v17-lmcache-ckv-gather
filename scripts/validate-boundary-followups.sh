#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
api="${API:-http://127.0.0.1:5001}"
cache_api="${CACHE_API:-http://127.0.0.1:8088}"
model="${MODEL:-GLM-5.2}"
prompt_tokens="${PROMPT_TOKENS:-258048}"
max_tokens="${MAX_TOKENS:-16}"
followups="${FOLLOWUPS:-4}"
run_id="${RUN_ID:-boundary-$(date -u +%Y%m%dT%H%M%SZ)}"
result_dir="${RESULT_DIR:-${root}/results}"

mkdir -p "${result_dir}"

python3 "${root}/base/validate_boundary_once.py" \
  --api "${api}" \
  --cache-api "${cache_api}" \
  --model "${model}" \
  --prompt-tokens "${prompt_tokens}" \
  --max-tokens "${max_tokens}" \
  --clear-cache \
  --run-id "${run_id}" \
  --output "${result_dir}/${run_id}-boundary.json"

python3 - \
  "${api}" \
  "${model}" \
  "${run_id}" \
  "${followups}" \
  "${result_dir}/${run_id}-followups.json" <<'PY'
import json
import secrets
import sys
import time
import urllib.error
import urllib.request

api, model, run_id, followups_raw, output_path = sys.argv[1:]
results = []
for index in range(int(followups_raw)):
    payload = {
        "model": model,
        "prompt": (
            f"{secrets.token_hex(32)} immediate validation {index} "
            f"for {run_id}: reply with OK."
        ),
        "max_tokens": 16,
        "temperature": 0,
    }
    request = urllib.request.Request(
        f"{api}/v1/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.perf_counter()
    status = None
    error = None
    try:
        with urllib.request.urlopen(request, timeout=180) as response:
            status = response.status
            response.read()
    except urllib.error.HTTPError as exc:
        status = exc.code
        error = exc.read().decode("utf-8", errors="replace")[:1000]
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    result = {
        "index": index,
        "status": status,
        "elapsed_seconds": time.perf_counter() - started,
        "error": error,
    }
    results.append(result)
    print(
        f"FOLLOWUP index={index} status={status} "
        f"elapsed={result['elapsed_seconds']:.3f}s"
    )
    if status != 200:
        break

with open(output_path, "w", encoding="utf-8") as handle:
    json.dump(
        {
            "run_id": run_id,
            "requested": int(followups_raw),
            "results": results,
        },
        handle,
        indent=2,
    )
if len(results) != int(followups_raw) or any(
    item["status"] != 200 for item in results
):
    raise SystemExit(1)
PY

curl -fsS --max-time 5 "${api}/v1/models" >/dev/null
echo "PASS boundary=${prompt_tokens} followups=${followups} run_id=${run_id}"
