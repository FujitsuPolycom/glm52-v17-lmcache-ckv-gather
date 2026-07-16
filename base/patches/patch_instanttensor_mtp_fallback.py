#!/usr/bin/env python3
"""Keep InstantTensor for the main model and bypass it for the MTP reload."""

from pathlib import Path


PATH = Path(
    "/opt/venv/lib/python3.12/site-packages/"
    "vllm/model_executor/model_loader/weight_utils.py"
)
MARKER = "AI01-INSTANTTENSOR-MTP-FALLBACK"

OLD = '''    with instanttensor.safe_open(
        hf_weights_files, framework="pt", device=device, process_group=process_group
    ) as f:
        for name, tensor in tqdm(
            f.tensors(),
            desc="Loading safetensors using InstantTensor loader",
            disable=not enable_tqdm(use_tqdm_on_load),
            bar_format=_BAR_FORMAT,
            position=tqdm._get_free_pos(),
            total=len(f.keys()),
            mininterval=1.0,
        ):
            if weight_name_prefixes and not _matches_weight_name_prefixes(
                name, weight_name_prefixes
            ):
                continue
            yield name, tensor
'''

NEW = f'''    # {MARKER}: preserve the fast main-model load, but avoid opening
    # InstantTensor for the small, prefix-filtered MTP reload after the target
    # model has consumed VRAM. InstantTensor can abort from a native loader
    # thread on cudaMalloc failure, so this decision must happen before open.
    if weight_name_prefixes and len(hf_weights_files) <= 8:
        logger.warning(
            "Bypassing InstantTensor for filtered %d-shard reload; "
            "using ordinary safetensors to avoid MTP staging-buffer OOM.",
            len(hf_weights_files),
        )
        yield from safetensors_weights_iterator(
            hf_weights_files,
            use_tqdm_on_load,
            weight_name_prefixes=weight_name_prefixes,
        )
        return

    def _iterate_instanttensor():
        with instanttensor.safe_open(
            hf_weights_files,
            framework="pt",
            device=device,
            process_group=process_group,
        ) as f:
            for name, tensor in tqdm(
                f.tensors(),
                desc="Loading safetensors using InstantTensor loader",
                disable=not enable_tqdm(use_tqdm_on_load),
                bar_format=_BAR_FORMAT,
                position=tqdm._get_free_pos(),
                total=len(f.keys()),
                mininterval=1.0,
            ):
                if weight_name_prefixes and not _matches_weight_name_prefixes(
                    name, weight_name_prefixes
                ):
                    continue
                yield name, tensor

    def _is_device_memory_error(exc: RuntimeError) -> bool:
        message = str(exc).lower()
        return "device memory is not enough" in message or "out of memory" in message

    yielded = False
    try:
        for item in _iterate_instanttensor():
            yielded = True
            yield item
        return
    except RuntimeError as exc:
        if yielded or not _is_device_memory_error(exc):
            raise
        logger.warning(
            "InstantTensor could not open before first yield; using ordinary "
            "safetensors for this weight iterator."
        )

    yield from safetensors_weights_iterator(
        hf_weights_files,
        use_tqdm_on_load,
        weight_name_prefixes=weight_name_prefixes,
    )
'''


def main() -> None:
    source = PATH.read_text()
    if MARKER in source:
        print(f"[{MARKER}] already applied")
        return
    count = source.count(OLD)
    if count != 1:
        raise RuntimeError(f"expected exactly one InstantTensor iterator anchor, got {count}")
    PATH.write_text(source.replace(OLD, NEW))
    compile(PATH.read_text(), str(PATH), "exec")
    print(f"[{MARKER}] applied to {PATH}")


if __name__ == "__main__":
    main()
