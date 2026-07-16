# Release gates

## Passed

- [x] Base v17 image pinned by digest
- [x] LMCache version pinned
- [x] Overlay source hashes checked before replacement
- [x] Patch application fails closed when source does not match
- [x] CKV-gather 8K through 128K prefill validation
- [x] Decode parity matrix
- [x] 192K CUDA IPC event-lifetime validation
- [x] Machine-specific paths moved into `.env`
- [x] No credentials included
- [x] Three exact 258,048-token cold-store runs
- [x] 20/20 immediate unique follow-up requests after boundary runs
- [x] Clean post-power-cycle boundary regression
- [x] Zero container restarts, OOM kills, CUDA errors, or engine deaths
- [x] Current-stream CKV all-gather and active-page-width guards installed

## Required before production use

- [x] ai01 and all four GPUs pass health checks
- [x] Pass controlled 258,048-token cold stores and post-prefill decode
- [x] Verify container health and zero restarts after each large test
- [x] Generate source checksums
- [x] Review third-party attribution and select Apache-2.0 for original work
- [ ] Build the combined image from a clean checkout
- [ ] Replay the same 258,048-token prefix from LMCache after a process restart
- [ ] Pass one 260,096-token prompt plus 2,048-token output boundary test
- [ ] Generate and publish a container image digest
- [ ] Replace the local image tag with the approved registry tag

The source release is intentionally marked experimental until the remaining
production gates pass.
