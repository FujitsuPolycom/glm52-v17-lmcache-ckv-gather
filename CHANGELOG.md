# Changelog

## 0.1.0-exp.1 - staged, unpublished

- Added TP4/DCP4-aware LMCache storage for opaque NVFP4 MLA records.
- Added preemption-safe restore and late KV-transfer completion guards.
- Backported LMCache CUDA IPC event lifetime retention.
- Added per-request lists for every outstanding chunk-store future/event pair.
- Added the eager-prefill CKV-gather path with stock-v17 decode fallback.
- Added an exact 84-chunk regression for a 258,048-token request.
- Added a pinned combined Docker build, configurable Compose deployment,
  selected benchmark records, source validation, and release checksums.

This version remains blocked on GPU validation of the multi-future fix.

