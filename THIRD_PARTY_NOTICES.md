# Third-party provenance

This experimental package contains or modifies code from the following
Apache-2.0 projects and derivative work:

- `vllm-project/vllm` and the Fathomless Firmament v17 derivative image
- `LMCache/LMCache` version 0.4.6
- LMCache commit `5824ab308906`, retaining CUDA IPC events until transfer
  completion
- DCP/LMCache work adapted from
  `myshytf/glm-5.2-v11-lmcache@b52a18a76efbcf253f7bf8333f5862f4b0af7fd7`
- launch configuration and hardware guidance from
  `local-inference-lab/rtx6kpro`
- full-CKV gather design discussion by Koush and Luke, independently ported and
  validated for the pinned v17/vLLM stack in this repository

The CKV-gather overlay is maintained separately from the LMCache fixes so each
change can later be submitted to its owning upstream project.

Original glue, validation, and packaging in this repository are released under
Apache-2.0. The pinned base image and model retain their own licenses and terms.
