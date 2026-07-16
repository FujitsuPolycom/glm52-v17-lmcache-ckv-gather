#!/usr/bin/env python3
"""Inject LMCache MP arguments into the pinned v17 GLM launcher."""

from pathlib import Path


TARGET = Path("/usr/local/bin/serve-glm52-v16.sh")
MARKER = "AI01-V17-LMCACHE-LAUNCHER-PATCH"


def replace_exact(text: str, old: str, new: str, description: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(
            f"{description}: expected exactly one source match, found {count}"
        )
    return text.replace(old, new, 1)


def main() -> None:
    text = TARGET.read_text()
    if MARKER in text:
        print(f"[{MARKER}] already applied")
        return

    text = replace_exact(
        text,
        "export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True",
        "# AI01-V17-LMCACHE-LAUNCHER-PATCH: CUDA IPC is incompatible with "
        "expandable segments.\nunset PYTORCH_CUDA_ALLOC_CONF",
        "CUDA allocator patch",
    )

    text = replace_exact(
        text,
        'GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"',
        'GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"\n'
        'KV_CACHE_MEMORY_BYTES="${KV_CACHE_MEMORY_BYTES:-}"',
        "optional KV-cache byte limit variable",
    )

    text = replace_exact(
        text,
        'cmd=(vllm serve "${MODEL}" \\\n',
        'kv_cache_args=()\n'
        'if [[ -n "${KV_CACHE_MEMORY_BYTES}" ]]; then\n'
        '  [[ "${KV_CACHE_MEMORY_BYTES}" =~ ^[0-9]+$ ]] || '
        'die "KV_CACHE_MEMORY_BYTES must be an integer byte count"\n'
        '  kv_cache_args=(--kv-cache-memory-bytes "${KV_CACHE_MEMORY_BYTES}")\n'
        'fi\n\n'
        'cmd=(vllm serve "${MODEL}" \\\n',
        "optional KV-cache argument construction",
    )

    text = replace_exact(
        text,
        '  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \\\n'
        '  --max-model-len "${MAX_MODEL_LEN}" \\\n',
        '  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \\\n'
        '  "${kv_cache_args[@]}" \\\n'
        '  --max-model-len "${MAX_MODEL_LEN}" \\\n',
        "optional KV-cache CLI argument",
    )

    old_tail = """  --hf-overrides \"${hf_overrides}\" \\
  \"${spec_arg[@]}\")"""
    new_tail = """  --hf-overrides \"${hf_overrides}\" \\
  --disable-hybrid-kv-cache-manager \\
  --kv-transfer-config \"${LMCACHE_KV_TRANSFER_CONFIG:?LMCache config is required}\" \\
  --override-generation-config '{\"repetition_penalty\":1.05}' \\
  \"${spec_arg[@]}\")  # AI01-V17-LMCACHE-LAUNCHER-PATCH"""
    text = replace_exact(text, old_tail, new_tail, "vLLM argument injection")

    TARGET.write_text(text)
    print(f"[{MARKER}] applied to {TARGET}")


if __name__ == "__main__":
    main()
