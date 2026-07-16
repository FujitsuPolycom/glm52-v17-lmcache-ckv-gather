from __future__ import annotations

import ast
import hashlib
from pathlib import Path


SITE = Path("/opt/venv/lib/python3.12/site-packages/vllm")
SRC = Path("/opt/vllm/vllm")
FILES = (
    Path("v1/attention/backends/mla/b12x_mla_sparse.py"),
    Path("model_executor/layers/attention/mla_attention.py"),
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


for relative in FILES:
    site_path = SITE / relative
    src_path = SRC / relative
    ast.parse(site_path.read_text(encoding="utf-8"), filename=str(site_path))
    if digest(site_path) != digest(src_path):
        raise RuntimeError(f"vLLM source copies differ after CKV overlay: {relative}")

b12x = (SITE / FILES[0]).read_text(encoding="utf-8")
mla = (SITE / FILES[1]).read_text(encoding="utf-8")
required_b12x = (
    "VLLM_B12X_MLA_CKV_GATHER",
    "_map_global_topk_to_gathered_ckv_kernel",
    "dcp_prefill_ckv_gather_eligible",
    "_dcp_gather_ckv",
    "self._ckv_extend_plan",
)
required_mla = (
    "ckv_gather_used = False",
    "dcp_prefill_ckv_gather_eligible",
    "and not ckv_gather_used",
)
for marker in required_b12x:
    if marker not in b12x:
        raise RuntimeError(f"missing B12X CKV marker: {marker}")
for marker in required_mla:
    if marker not in mla:
        raise RuntimeError(f"missing MLA CKV marker: {marker}")

print("CKV gather overlay validation passed")

