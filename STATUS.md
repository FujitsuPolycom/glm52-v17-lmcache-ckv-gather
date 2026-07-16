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

## Required before publication

- [ ] ai01 returns and all four GPUs pass health checks
- [ ] Build the combined image from this directory
- [ ] Pass one controlled 258,048-token cold store and post-prefill decode
- [ ] Replay the same prefix from LMCache without a CUDA fault
- [ ] Pass one 260,096-token prompt plus 2,048-token output boundary test
- [ ] Verify container health and zero restarts after each large test
- [ ] Generate image digest and source checksums
- [ ] Review third-party attribution and select the release license
- [ ] Replace the local image tag with the approved registry tag
- [ ] Obtain explicit approval before any push or image publication

