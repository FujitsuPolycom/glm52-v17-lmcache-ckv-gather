#!/usr/bin/env bash
set -euo pipefail

required_paths=(
  MODEL_PATH
  HF_CACHE_PATH
  RUNTIME_CACHE_PATH
  RUNTIME_TMP_PATH
)

if [[ ! -f .env ]]; then
  echo "Missing .env; copy .env.example and set the host paths." >&2
  exit 2
fi

set -a
# shellcheck disable=SC1091
source .env
set +a

for variable in "${required_paths[@]}"; do
  value="${!variable:-}"
  if [[ -z "${value}" || "${value}" != /* ]]; then
    echo "${variable} must be an absolute Linux path." >&2
    exit 2
  fi
done

if [[ ! -d "${MODEL_PATH}" ]]; then
  echo "Model path does not exist: ${MODEL_PATH}" >&2
  exit 2
fi

for directory in "${HF_CACHE_PATH}" "${RUNTIME_CACHE_PATH}" "${RUNTIME_TMP_PATH}"; do
  mkdir -p "${directory}"
  test -w "${directory}" || {
    echo "Path is not writable: ${directory}" >&2
    exit 2
  }
done

command -v docker >/dev/null
docker compose version >/dev/null
command -v nvidia-smi >/dev/null

gpu_count="$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)"
if [[ "${gpu_count}" -ne 4 ]]; then
  echo "Expected exactly four visible GPUs, found ${gpu_count}." >&2
  exit 2
fi

memory_kib="$(awk '/MemTotal:/ {print $2}' /proc/meminfo)"
if (( memory_kib < 120 * 1024 * 1024 )); then
  echo "At least 120 GiB host RAM is required for the 48 GiB LMCache profile." >&2
  exit 2
fi

docker compose --env-file .env config --quiet
echo "Preflight passed: 4 GPUs, model present, host paths writable, Compose valid."

