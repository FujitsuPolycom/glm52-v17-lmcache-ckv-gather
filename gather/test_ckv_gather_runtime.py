from __future__ import annotations

import os

import torch
import torch.distributed as dist

from vllm import _custom_ops as ops
from vllm.v1.attention.backends.mla.b12x_mla_sparse import (
    _map_global_topk_to_gathered_ckv,
)


BLOCK_SIZE = 64
RECORD_BYTES = 432
TOPK = 2048
INTERLEAVE = 1


def local_length(global_length: int, world: int, rank: int) -> int:
    base, remainder = divmod(global_length, world)
    return base + int(rank < remainder)


def global_token(local_index: int, world: int, rank: int) -> int:
    return (local_index // INTERLEAVE) * world * INTERLEAVE + rank * INTERLEAVE


def main() -> None:
    dist.init_process_group("nccl")
    rank = int(os.environ["LOCAL_RANK"])
    world = dist.get_world_size()
    if world != 4:
        raise RuntimeError(f"runtime check requires four ranks, got {world}")
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)

    global_lens = (259, 131)
    all_lens_cpu = [
        [local_length(length, world, source_rank) for length in global_lens]
        for source_rank in range(world)
    ]
    rank_starts_cpu = []
    for lens in all_lens_cpu:
        starts = [0]
        for length in lens[:-1]:
            starts.append(starts[-1] + length)
        rank_starts_cpu.append(starts)
    totals = [sum(lens) for lens in all_lens_cpu]
    padded = ((max(totals) + BLOCK_SIZE - 1) // BLOCK_SIZE) * BLOCK_SIZE

    local_lens = all_lens_cpu[rank]
    max_blocks = max((length + BLOCK_SIZE - 1) // BLOCK_SIZE for length in local_lens)
    block_table = torch.full(
        (len(global_lens), max_blocks), -1, dtype=torch.int32, device=device
    )
    total_blocks = sum(
        (length + BLOCK_SIZE - 1) // BLOCK_SIZE for length in local_lens
    )
    cache = torch.zeros(
        (total_blocks, BLOCK_SIZE, RECORD_BYTES), dtype=torch.uint8, device=device
    )
    cache_i32 = cache.view(torch.int32)
    next_block = 0
    for req_id, length in enumerate(local_lens):
        blocks = (length + BLOCK_SIZE - 1) // BLOCK_SIZE
        block_table[req_id, :blocks] = torch.arange(
            next_block, next_block + blocks, dtype=torch.int32, device=device
        )
        for local_idx in range(length):
            token = global_token(local_idx, world, rank)
            value = req_id * 10000 + token
            block = next_block + local_idx // BLOCK_SIZE
            offset = local_idx % BLOCK_SIZE
            cache_i32[block, offset, 0] = value
        next_block += blocks

    cu_lens = torch.tensor(
        [0, local_lens[0], sum(local_lens)], dtype=torch.int32, device=device
    )
    local_buffer = torch.zeros((padded, RECORD_BYTES), dtype=torch.uint8, device=device)
    ops.cp_gather_cache(
        src_cache=cache,
        dst=local_buffer[: totals[rank]],
        block_table=block_table,
        cu_seq_lens=cu_lens,
        batch_size=len(global_lens),
    )
    gathered = torch.empty(
        (world * padded, RECORD_BYTES), dtype=torch.uint8, device=device
    )
    dist.all_gather_into_tensor(gathered.view(-1), local_buffer.view(-1))

    req_ids = torch.tensor([0, 1], dtype=torch.int32, device=device)
    topk = torch.full((2, TOPK), -1, dtype=torch.int32, device=device)
    wanted = ([0, 1, 2, 63, 64, 128, 258], [0, 3, 64, 130])
    for row, tokens in enumerate(wanted):
        topk[row, : len(tokens)] = torch.tensor(tokens, dtype=torch.int32, device=device)
    rank_starts = torch.tensor(rank_starts_cpu, dtype=torch.int32, device=device)
    rank_lens = torch.tensor(all_lens_cpu, dtype=torch.int32, device=device)
    selected = torch.empty_like(topk)
    valid_counts = torch.empty((2,), dtype=torch.int32, device=device)
    _map_global_topk_to_gathered_ckv(
        req_ids,
        topk,
        rank_starts,
        rank_lens,
        selected,
        valid_counts,
        dcp_size=world,
        cp_kv_cache_interleave_size=INTERLEAVE,
        padded_rank_tokens=padded,
    )
    torch.cuda.synchronize()

    gathered_i32 = gathered.view(torch.int32)
    counts = valid_counts.cpu().tolist()
    for row, tokens in enumerate(wanted):
        slots = selected[row, : counts[row]].to(torch.int64)
        observed = sorted(gathered_i32[slots, 0].cpu().tolist())
        expected = sorted(row * 10000 + token for token in tokens)
        if observed != expected:
            raise AssertionError(
                f"rank {rank} row {row}: gathered values {observed} != {expected}"
            )
        if not torch.all(selected[row, counts[row] :] == -1):
            raise AssertionError(f"rank {rank} row {row}: invalid tail was not masked")

    dist.barrier()
    if rank == 0:
        print(
            "CKV runtime check passed: native uint8 page gather, NCCL all-gather, "
            "global-topK remap, and invalid-tail masking"
        )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()

