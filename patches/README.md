# Patch artifacts

## v17 CKV gather

`v17-ckv-gather.patch` is a unified diff against the exact vLLM source shipped
in the pinned Fathomless Firmament v17 base image. From that vLLM source root:

```bash
git apply --check /path/to/v17-ckv-gather.patch
git apply /path/to/v17-ckv-gather.patch
```

It changes three files:

- `vllm/v1/attention/backends/mla/b12x_mla_sparse.py`
- `vllm/model_executor/layers/attention/mla_attention.py`
- `vllm/model_executor/layers/sparse_attn_indexer.py`

The patch includes the eager full-CKV DCP path, current-stream collective,
stable workspace lifetime, and active page-table width fix.

## LMCache

The LMCache work spans LMCache 0.4.6, vLLM's connector, and the v17 launcher.
It is distributed as fail-closed, idempotent source patchers under
`base/patches/` rather than one misleading cross-repository diff. The pinned
Dockerfile applies them in order and validates the resulting installation.

Do not apply these artifacts to another vLLM or LMCache revision without
reviewing and updating the source hashes and regression tests.
