#!/usr/bin/env python3
"""Add an opt-in eager-mode switch to the pinned v17 launcher."""

from pathlib import Path


TARGET = Path("/usr/local/bin/serve-glm52-v16.sh")
MARKER = "AI01-CKV-GATHER-EAGER-SWITCH"


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
        'GRAPH="${GRAPH:-$((MAX_NUM_SEQS * 4))}"',
        'GRAPH="${GRAPH:-$((MAX_NUM_SEQS * 4))}"\n'
        'ENFORCE_EAGER="${ENFORCE_EAGER:-0}"',
        "eager-mode variable",
    )
    text = replace_exact(
        text,
        '[[ "${GRAPH}" =~ ^[0-9]+$ ]] || die "GRAPH must be an integer"',
        '[[ "${GRAPH}" =~ ^[0-9]+$ ]] || die "GRAPH must be an integer"\n'
        '[[ "${ENFORCE_EAGER}" =~ ^(0|1)$ ]] || '
        'die "ENFORCE_EAGER must be 0 or 1"',
        "eager-mode validation",
    )
    text = replace_exact(
        text,
        'cmd=(vllm serve "${MODEL}" \\\n',
        'eager_args=()\n'
        'if [[ "${ENFORCE_EAGER}" == "1" ]]; then\n'
        '  eager_args=(--enforce-eager)\n'
        'fi\n\n'
        'cmd=(vllm serve "${MODEL}" \\\n',
        "eager-mode argument construction",
    )
    text = replace_exact(
        text,
        '  --max-cudagraph-capture-size "${GRAPH}" \\\n'
        '  --async-scheduling \\\n',
        '  --max-cudagraph-capture-size "${GRAPH}" \\\n'
        '  "${eager_args[@]}" \\\n'
        '  --async-scheduling \\\n',
        "eager-mode CLI argument",
    )

    TARGET.write_text(text + f"\n# {MARKER}\n")
    print(f"[{MARKER}] applied to {TARGET}")


if __name__ == "__main__":
    main()
