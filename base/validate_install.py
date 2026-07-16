#!/usr/bin/env python3
"""Fail-closed static and layout checks for the v17 LMCache prototype."""

import gc
from importlib import import_module
from pathlib import Path
from types import SimpleNamespace
import weakref

import torch

from lmcache.v1.gpu_connector.utils import normalize_kv_and_discover_format
from lmcache.integration.vllm.vllm_multi_process_adapter import (
    LMCacheMPWorkerAdapter,
)
from lmcache.v1.kv_layer_groups import KVLayerGroupsManager
from lmcache.utils import EngineType


SITE = Path("/opt/venv/lib/python3.12/site-packages")
VLLM_CONNECTOR = SITE / (
    "vllm/distributed/kv_transfer/kv_connector/v1/lmcache_mp_connector.py"
)
LMCACHE_CONNECTOR = SITE / "lmcache/integration/vllm/lmcache_mp_connector.py"
ADAPTER = SITE / "lmcache/integration/vllm/vllm_multi_process_adapter.py"
SCHEDULER = SITE / "vllm/v1/core/sched/scheduler.py"
LAUNCHER = Path("/usr/local/bin/serve-glm52-v16.sh")
WEIGHT_UTILS = SITE / "vllm/model_executor/model_loader/weight_utils.py"


def require(path: Path, needles: tuple[str, ...]) -> None:
    text = path.read_text()
    missing = [needle for needle in needles if needle not in text]
    if missing:
        raise RuntimeError(f"{path}: missing required markers: {missing}")


def validate_sources() -> None:
    connector_needles = (
        "DCP-LMCACHE-PATCH",
        "block_size * vllm_config.parallel_config.decode_context_parallel_size",
        "dcp_size = vllm_config.parallel_config.decode_context_parallel_size",
        "kv_rank = rank % dcp_size",
    )
    require(VLLM_CONNECTOR, connector_needles)
    require(LMCACHE_CONNECTOR, connector_needles)
    require(
        ADAPTER,
        (
            "DCP-MLA-STORE-PATCH",
            "AI01-LMCACHE-IPC-EVENT-LIFETIME",
            "AI01-LMCACHE-CHUNKED-STORE-FUTURES",
            "self.store_futures.setdefault(request_id, []).append(future)",
            "self.store_events.setdefault(request_id, []).append(event)",
            "self.retrieve_events[request_id] = event",
            "def _poll_store_futures(self)",
        ),
    )
    require(SCHEDULER, ("patched guard", "patched recv guard"))
    require(
        LAUNCHER,
        (
            "AI01-V17-LMCACHE-LAUNCHER-PATCH",
            "--disable-hybrid-kv-cache-manager",
            "--kv-transfer-config",
            "unset PYTORCH_CUDA_ALLOC_CONF",
        ),
    )
    require(
        WEIGHT_UTILS,
        (
            "AI01-INSTANTTENSOR-MTP-FALLBACK",
            "Bypassing InstantTensor for filtered %d-shard reload",
            "len(hf_weights_files) <= 8",
            "yield from safetensors_weights_iterator",
        ),
    )


def validate_nvfp4_layout() -> None:
    # v17 exposes each NVFP4 MLA layer as [num_blocks, block_size, 432]
    # uint8. LMCache should discover the ordinary vLLM MLA format and retain
    # the opaque 432-byte record width without conversion.
    tensors = [
        torch.zeros((8, 64, 432), dtype=torch.uint8),
        torch.zeros((8, 64, 432), dtype=torch.uint8),
    ]
    fmt, normalized = normalize_kv_and_discover_format(tensors, EngineType.VLLM)
    if fmt.name != "NL_X_NB_BS_HS":
        raise RuntimeError(f"unexpected LMCache format: {fmt.name}")
    if any(t.dtype is not torch.uint8 or t.shape[-1] != 432 for t in normalized):
        raise RuntimeError("LMCache changed the opaque NVFP4 tensor layout")

    groups = KVLayerGroupsManager(
        normalized,
        gpu_kv_format=fmt,
        num_blocks=8,
        lmcache_logical_chunk_size=512,
        layout_hints={"inference_engine_logical_block_size": 256},
    )
    if groups.num_groups != 1:
        raise RuntimeError(f"unexpected layer-group count: {groups.num_groups}")
    group = groups.kv_layer_groups[0]
    if group.dtype is not torch.uint8 or group.hidden_dim_size != 432:
        raise RuntimeError(
            f"unexpected NVFP4 group: dtype={group.dtype}, width={group.hidden_dim_size}"
        )
    if group.shape_desc.bs != 64 or group.compress_ratio != 4:
        raise RuntimeError(
            f"unexpected DCP4 layout: bs={group.shape_desc.bs}, "
            f"compression={group.compress_ratio}"
        )
    if group.physical_chunk_size != 128:
        raise RuntimeError(
            f"unexpected physical chunk size: {group.physical_chunk_size}"
        )
    chunk_shape = torch.Size(
        (group.shape_desc.kv_size, group.num_layers, 128, group.hidden_dim_size)
    )
    if chunk_shape != torch.Size((1, 2, 128, 432)):
        raise RuntimeError(f"unexpected NVFP4 chunk shape: {chunk_shape}")


def validate_dcp_mapping() -> None:
    # TP4/DCP4 gives each process one distinct physical MLA KV shard. The
    # scheduler's 64-token block is therefore 256 logical tokens, and the
    # 512-token LMCache chunk contains exactly two scheduler blocks.
    config = SimpleNamespace(
        model_config=object(),
        parallel_config=SimpleNamespace(
            tensor_parallel_size=4,
            decode_context_parallel_size=4,
        ),
    )
    module_names = (
        "lmcache.integration.vllm.lmcache_mp_connector",
        "vllm.distributed.kv_transfer.kv_connector.v1.lmcache_mp_connector",
    )
    for module_name in module_names:
        module = import_module(module_name)
        module.mla_enabled = lambda _: True
        actual = [
            module.extract_world_size_and_kv_rank(4, rank, config)
            for rank in range(4)
        ]
        expected = [(4, rank) for rank in range(4)]
        if actual != expected:
            raise RuntimeError(
                f"{module_name}: incorrect TP4/DCP4 KV mapping: {actual}"
            )


class FakeStoreFuture:
    def __init__(self, chunk: int, done: bool = False, result: bool = True) -> None:
        self.chunk = chunk
        self.done = done
        self.store_result = result
        self.result_calls = 0

    def query(self) -> bool:
        return self.done

    def result(self) -> bool:
        if not self.done:
            raise RuntimeError("result called before future completed")
        self.result_calls += 1
        return self.store_result


class FakeIpcEvent:
    def __init__(self, chunk: int) -> None:
        self.chunk = chunk


def validate_chunked_store_tracking() -> None:
    # 252 Ki tokens is 258,048 exact tokens. At a 3,072-token scheduler
    # step, one request can have 84 store operations in flight under sustained
    # chunked prefill. Exercise that exact failure shape and complete chunks
    # out of order so scalar per-request bookkeeping cannot pass accidentally.
    prompt_tokens = 258_048
    store_step_tokens = 3_072
    chunk_count, remainder = divmod(prompt_tokens, store_step_tokens)
    if remainder or chunk_count != 84:
        raise RuntimeError(
            f"invalid regression geometry: {prompt_tokens=} "
            f"{store_step_tokens=} {chunk_count=} {remainder=}"
        )

    adapter = LMCacheMPWorkerAdapter.__new__(LMCacheMPWorkerAdapter)
    futures = [FakeStoreFuture(chunk) for chunk in range(chunk_count)]
    events = [FakeIpcEvent(chunk) for chunk in range(chunk_count)]
    event_refs = [weakref.ref(event) for event in events]
    adapter.store_futures = {"req": list(futures)}
    adapter.store_events = {"req": list(events)}
    del events
    gc.collect()

    if not all(ref() is not None for ref in event_refs):
        raise RuntimeError("an IPC event was released before its future completed")

    # Complete every even chunk first. Pending future/event pairs must remain
    # aligned and all completed events should become collectible.
    for future in futures[::2]:
        future.done = True

    if adapter._poll_store_futures():
        raise RuntimeError("request finished while 42 chunk futures were pending")
    if adapter.store_futures != {"req": futures[1::2]}:
        raise RuntimeError("out-of-order completed chunk futures were not drained")
    if [event.chunk for event in adapter.store_events["req"]] != list(
        range(1, chunk_count, 2)
    ):
        raise RuntimeError("IPC events did not stay aligned with pending futures")
    if any(future.result_calls != 1 for future in futures[::2]):
        raise RuntimeError("a completed chunk future was not consumed exactly once")

    gc.collect()
    if any(event_refs[index]() is not None for index in range(0, chunk_count, 2)):
        raise RuntimeError("completed IPC events remained strongly referenced")
    if any(event_refs[index]() is None for index in range(1, chunk_count, 2)):
        raise RuntimeError("pending IPC events were released too early")

    # Leave the final chunk pending through a second poll.
    for future in futures[1:-1:2]:
        future.done = True
    if adapter._poll_store_futures():
        raise RuntimeError("request finished before its 84th chunk")
    if adapter.store_futures != {"req": [futures[-1]]}:
        raise RuntimeError("the final pending future was not retained")
    if [event.chunk for event in adapter.store_events["req"]] != [chunk_count - 1]:
        raise RuntimeError("the final pending IPC event was not retained")

    futures[-1].done = True
    if adapter._poll_store_futures() != {"req"}:
        raise RuntimeError("request did not finish after its final chunk future")
    if adapter.store_futures or adapter.store_events:
        raise RuntimeError("completed store tracking state was not released")

    gc.collect()
    if any(ref() is not None for ref in event_refs):
        raise RuntimeError("IPC event references leaked after request completion")

    # Fail closed if future/event state ever becomes misaligned.
    adapter.store_futures = {"bad": [FakeStoreFuture(0)]}
    adapter.store_events = {"bad": []}
    try:
        adapter._poll_store_futures()
    except RuntimeError as exc:
        if "future/event mismatch" not in str(exc):
            raise
    else:
        raise RuntimeError("mismatched store state did not fail closed")


if __name__ == "__main__":
    validate_sources()
    validate_nvfp4_layout()
    validate_dcp_mapping()
    validate_chunked_store_tracking()
    print("v17 LMCache compatibility checks passed")
