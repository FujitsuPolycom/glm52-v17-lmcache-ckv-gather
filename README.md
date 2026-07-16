# GLM-5.2 v17 LMCache + CKV-gather experimental image

This package builds one pinned container containing:

- Fathomless Firmament v17 for GLM-5.2 on four RTX PRO 6000 GPUs
- LMCache 0.4.6 with TP4/DCP4 MLA compatibility and RAM-backed KV reuse
- an eager-prefill CKV-gather path that leaves decode on the stock v17 path

## Credits

The full-CKV gather direction builds on **@koush's** prototype and design notes,
with **@luke's** DCP/MLA technical review and corrections. Handles refer to their
Discord identities. This repository is an independent vLLM port, LMCache
integration, validation effort, and experimental release package for `ai01`.

## Status

This is a GPU-validated **experimental** release, not a production-supported
vLLM distribution. It pins one specific Fathomless Firmament v17 image and
fails closed if the expected vLLM source hashes do not match.

The final edge-case gate passed three cold 258,048-token prompts in one model
process and after a clean host reboot. Every run drained 2,016 LMCache objects
(9,562,226,688 bytes), and all 20 immediate cache-unique follow-up requests
returned HTTP 200. The tested container remained healthy with zero restarts,
no host OOM, and no CUDA or `EngineDeadError` log entries.

The release contains three related corrections:

- LMCache retains every chunk-store future and its matching CUDA IPC event.
- Full-CKV all-gather uses vLLM's current-stream PyNccl communicator before the
  shared arena is reused by another layer.
- The packed B12X indexer exposes only the active page-table width instead of
  stale capacity-only columns after a large request.

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

Run the release regression that found the original asynchronous CUDA fault:

```bash
bash scripts/validate-boundary-followups.sh
```

By default it clears LMCache, sends an exact 258,048-token cold prompt, waits
for every store to drain, then sends four unique small requests immediately.
Expect roughly 90-100 seconds after model warmup on four RTX PRO 6000 GPUs.

## Layout

- `base/`: LMCache and v17 integration patches
- `gather/`: CKV-gather vLLM overlay and runtime validator
- `patches/`: reviewable machine-applicable vLLM diff and patch notes
- `benchmarks/`: selected raw results and summaries
- `tests/`: source-only regression and release-layout checks
- `release-manifest.json`: pinned components, limits, and validation state
