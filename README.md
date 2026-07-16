# GLM-5.2 v17 LMCache + CKV-gather experimental image

This package builds one pinned container containing:

- Fathomless Firmament v17 for GLM-5.2 on four RTX PRO 6000 GPUs
- LMCache 0.4.6 with TP4/DCP4 MLA compatibility and RAM-backed KV reuse
- an eager-prefill CKV-gather path that leaves decode on the stock v17 path

## Status

This is a local staging package, not a published release. GPU validation of the
multi-future LMCache fix is still pending. Do not use it as a production image.
An empty private staging repository exists under `FujitsuPolycom`; this source
has not been pushed to it.

Validated before the pending fix:

- CKV-gather prefill at 8K, 64K, 96K, 128K, and 192K
- 12-cell decode parity against stock v17
- a 196,610-token request with the CUDA IPC event-lifetime backport
- a 258,048-token request completed, but the engine later faulted while polling
  an overwritten chunk-store future at the first MTP transition

The staged fix retains every outstanding store future and its matching CUDA IPC
event for all 84 chunks in a 258,048-token request.

## Tested configuration

- 4x NVIDIA RTX PRO 6000 Blackwell 96GB
- TP4 / DCP4 / MTP3
- `madeby561/GLM-5.2-MXFP8-NVFP4-NF3-Hybrid`
- NVFP4 MLA KV, 492,800-token GPU KV pool
- 48 GiB RAM LMCache, approximately 1.39M reusable payload tokens
- 262,144 maximum model length

## Build and run

Linux, NVIDIA Container Toolkit, Docker Compose v2, and the model checkpoint are
required. The API and LMCache status endpoint use host networking and have no
authentication; restrict them with the host firewall.

```bash
cp .env.example .env
# Edit .env and set the four absolute host paths.
./scripts/preflight.sh
docker compose up -d --build
docker compose logs -f glm52
```

The OpenAI-compatible API defaults to `http://127.0.0.1:5001` and LMCache status
to `http://127.0.0.1:8088`.

## Validation

Run the model-free release checks locally:

```bash
python3 -m unittest discover -s tests -v
```

After the container is healthy, validate a short cold/store/replay cycle:

```bash
python3 base/verify_reuse.py \
  --api http://127.0.0.1:5001 \
  --cache-api http://127.0.0.1:8088 \
  --model GLM-5.2 \
  --sentinel-tokens 8192 \
  --require-external-hit \
  --output results/reuse-8k.json
```

The 258,048-token boundary test is intentionally omitted from the public quick
start until the staged multi-future fix passes on GPU hardware.

## Layout

- `base/`: LMCache and v17 integration patches
- `gather/`: CKV-gather vLLM overlay and runtime validator
- `benchmarks/`: selected raw results and summaries
- `tests/`: source-only regression and release-layout checks
- `release-manifest.json`: pinned components, limits, and validation state
