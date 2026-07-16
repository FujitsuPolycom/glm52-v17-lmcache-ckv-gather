#!/usr/bin/env bash
set -euo pipefail

LMCACHE_HOST="${LMCACHE_HOST:-127.0.0.1}"
LMCACHE_PORT="${LMCACHE_PORT:-5555}"
LMCACHE_HTTP_PORT="${LMCACHE_HTTP_PORT:-8088}"
LMCACHE_CHUNK_SIZE="${LMCACHE_CHUNK_SIZE:-512}"
LMCACHE_L1_GB="${LMCACHE_L1_GB:-48}"
LMCACHE_L1_INIT_GB="${LMCACHE_L1_INIT_GB:-48}"
LMCACHE_LOG="${LMCACHE_LOG:-/tmp/lmcache-mp.log}"

if (( MAX_MODEL_LEN % LMCACHE_CHUNK_SIZE != 0 )); then
  echo "MAX_MODEL_LEN=${MAX_MODEL_LEN} must be divisible by LMCACHE_CHUNK_SIZE=${LMCACHE_CHUNK_SIZE}" >&2
  exit 2
fi

export LMCACHE_KV_TRANSFER_CONFIG
LMCACHE_KV_TRANSFER_CONFIG="$(printf '{\"kv_connector\":\"LMCacheMPConnector\",\"kv_role\":\"kv_both\",\"kv_connector_extra_config\":{\"lmcache.mp.host\":\"tcp://%s\",\"lmcache.mp.port\":%s,\"lmcache.mp.mq_timeout\":30,\"lmcache.mp.heartbeat_interval\":5}}' "${LMCACHE_HOST}" "${LMCACHE_PORT}")"

rm -f "${LMCACHE_LOG}"
lmcache server \
  --host "${LMCACHE_HOST}" \
  --port "${LMCACHE_PORT}" \
  --chunk-size "${LMCACHE_CHUNK_SIZE}" \
  --l1-size-gb "${LMCACHE_L1_GB}" \
  --l1-init-size-gb "${LMCACHE_L1_INIT_GB}" \
  --l1-write-ttl-seconds 600 \
  --l1-read-ttl-seconds 300 \
  --eviction-policy LRU \
  --eviction-trigger-watermark 0.90 \
  --eviction-ratio 0.10 \
  --http-port "${LMCACHE_HTTP_PORT}" \
  >"${LMCACHE_LOG}" 2>&1 &
lmcache_pid=$!

cleanup() {
  kill "${lmcache_pid}" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

for _ in $(seq 1 120); do
  if ! kill -0 "${lmcache_pid}" 2>/dev/null; then
    echo "LMCache exited during startup" >&2
    sed -n '1,240p' "${LMCACHE_LOG}" >&2 || true
    exit 1
  fi
  if grep -q "ZMQ cache server is running" "${LMCACHE_LOG}" 2>/dev/null; then
    break
  fi
  sleep 1
done

if ! grep -q "ZMQ cache server is running" "${LMCACHE_LOG}" 2>/dev/null; then
  echo "LMCache did not become ready" >&2
  sed -n '1,240p' "${LMCACHE_LOG}" >&2 || true
  exit 1
fi

echo "LMCache ready: L1=${LMCACHE_L1_GB}GB chunk=${LMCACHE_CHUNK_SIZE} (RAM only)"

# Do not exec: keeping this shell as PID 1 lets its trap stop LMCache when
# vLLM exits or Docker stops the container.
/usr/local/bin/serve-glm52-hybrid-v17.sh &
vllm_pid=$!
wait "${vllm_pid}"
