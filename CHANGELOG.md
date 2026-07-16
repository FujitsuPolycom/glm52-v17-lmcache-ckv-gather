# Changelog

## 0.1.0-exp.2 - GPU-validated experimental release

- Routed full-CKV all-gather through the current-stream PyNccl communicator to
  prevent a shared-arena write/reuse race between attention layers.
- Made CKV arena growth fail closed once layer aliases exist.
- Limited packed B12X prefill page tables to their active local-token width.
- Passed three cold 258,048-token runs and 20/20 immediate unique follow-ups,
  including a clean post-reboot regression.
- Added a portable boundary-plus-follow-up validator and reviewable vLLM patch.

## 0.1.0-exp.1 - staged, unpublished

- Added TP4/DCP4-aware LMCache storage for opaque NVFP4 MLA records.
- Added preemption-safe restore and late KV-transfer completion guards.
- Backported LMCache CUDA IPC event lifetime retention.
- Added per-request lists for every outstanding chunk-store future/event pair.
- Added the eager-prefill CKV-gather path with stock-v17 decode fallback.
- Added an exact 84-chunk regression for a 258,048-token request.
- Added a pinned combined Docker build, configurable Compose deployment,
  selected benchmark records, source validation, and release checksums.

This version was superseded before publication.
