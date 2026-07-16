#!/usr/bin/env bash
set -euo pipefail

name="glm52-v17-lmcache-smoke"
status_file="/tmp/${name}-status.json"
image="${IMAGE:-glm52-v17-lmcache-ckv-gather:0.1.0-exp.1}"

cleanup() {
  docker rm -f "${name}" >/dev/null 2>&1 || true
  rm -f "${status_file}"
}
trap cleanup EXIT INT TERM
cleanup

docker run -d \
  --name "${name}" \
  --network host \
  --entrypoint lmcache \
  "${image}" \
  server \
  --host 127.0.0.1 \
  --port 15555 \
  --chunk-size 512 \
  --l1-size-gb 1 \
  --l1-init-size-gb 1 \
  --eviction-policy LRU \
  --http-port 18088 >/dev/null

for _ in $(seq 1 60); do
  if [[ "$(docker inspect -f '{{.State.Running}}' "${name}")" != "true" ]]; then
    docker logs "${name}" >&2
    exit 1
  fi
  if curl -fsS http://127.0.0.1:18088/status >"${status_file}"; then
    break
  fi
  sleep 1
done

grep -q "ZMQ cache server is running" < <(docker logs "${name}" 2>&1)
test -s "${status_file}"
echo "LMCache daemon smoke test passed"
cat "${status_file}"
