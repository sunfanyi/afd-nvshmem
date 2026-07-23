from __future__ import annotations

import json
import os

import torch
import torch.distributed as dist

from afd_nvshmem import Provider, get_unique_id


ALIGNMENT = 256
CAPACITIES = (1024, 4096)
WAVES = 64


def align_up(value: int) -> int:
    return (value + ALIGNMENT - 1) // ALIGNMENT * ALIGNMENT


def uint64_view(arena: torch.Tensor, offset: int) -> torch.Tensor:
    return arena[offset : offset + 8].view(torch.uint64)


def main() -> None:
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    if world_size != 2:
        raise RuntimeError(f"provider smoke requires two PEs, got {world_size}")

    torch.cuda.set_device(local_rank)
    dist.init_process_group("gloo")

    uid_objects = [get_unique_id() if rank == 0 else b""]
    dist.broadcast_object_list(uid_objects, src=0)
    provider = Provider.init(
        uid_objects[0],
        rank=rank,
        world_size=world_size,
        device=local_rank,
    )

    max_payload_bytes = max(CAPACITIES)
    request_offset = 0
    response_offset = align_up(request_offset + max_payload_bytes)
    request_signal_offset = align_up(response_offset + max_payload_bytes)
    response_signal_offset = request_signal_offset + 8
    request_expected_offset = response_signal_offset + 8
    response_expected_offset = request_expected_offset + 8
    arena_size = align_up(response_expected_offset + 8)

    arena = provider.allocate_arena(arena_size)
    if arena.dtype != torch.uint8 or arena.shape != (arena_size,):
        raise RuntimeError(
            f"unexpected arena tensor: dtype={arena.dtype} shape={arena.shape}"
        )
    if arena.data_ptr() % ALIGNMENT != 0:
        raise RuntimeError("symmetric arena is not 256-byte aligned")

    request = arena[request_offset : request_offset + max_payload_bytes]
    response = arena[response_offset : response_offset + max_payload_bytes]
    request_signal = uint64_view(arena, request_signal_offset)
    response_signal = uint64_view(arena, response_signal_offset)
    request_expected = uint64_view(arena, request_expected_offset)
    response_expected = uint64_view(arena, response_expected_offset)

    pattern = torch.arange(
        max_payload_bytes,
        dtype=torch.int64,
        device=f"cuda:{local_rank}",
    ).to(torch.uint8)
    mismatch_count = torch.zeros((), dtype=torch.int64, device=local_rank)
    stream = torch.cuda.Stream(device=local_rank)
    peer = 1 - rank

    with torch.cuda.stream(stream):
        for wave in range(WAVES):
            capacity = CAPACITIES[wave % len(CAPACITIES)]
            expected_request = torch.bitwise_xor(pattern[:capacity], wave & 0xFF)
            expected_response = torch.bitwise_xor(expected_request, 0xA5)
            if rank == 0:
                request[:capacity].copy_(expected_request)
                provider.put_signal_on_stream(
                    request[:capacity], request_signal, peer
                )
                provider.wait_next_on_stream(response_signal, response_expected)
                mismatch_count.add_(
                    torch.count_nonzero(
                        response[:capacity] != expected_response
                    )
                )
            else:
                provider.wait_next_on_stream(request_signal, request_expected)
                mismatch_count.add_(
                    torch.count_nonzero(request[:capacity] != expected_request)
                )
                response[:capacity].copy_(expected_response)
                provider.put_signal_on_stream(
                    response[:capacity], response_signal, peer
                )

    stream.synchronize()

    local_signal = response_signal if rank == 0 else request_signal
    local_expected = response_expected if rank == 0 else request_expected
    signal_value = int(local_signal.item())
    expected_value = int(local_expected.item())
    if signal_value != WAVES or expected_value != WAVES:
        raise RuntimeError(
            "sequence mismatch: "
            f"rank={rank} signal={signal_value} expected={expected_value}"
        )

    total_mismatches = torch.tensor([int(mismatch_count.item())])
    dist.all_reduce(total_mismatches)
    if total_mismatches.item() != 0:
        raise RuntimeError(f"payload mismatches: {total_mismatches.item()}")

    version_info = provider.version_info
    dist.barrier()
    provider.close()
    dist.barrier()

    if rank == 0:
        print(
            json.dumps(
                {
                    "status": "PASS",
                    "compiled_version": version_info.compiled,
                    "runtime_version": version_info.runtime,
                    "world_size": world_size,
                    "arena_bytes": arena_size,
                    "capacities": CAPACITIES,
                    "waves": WAVES,
                    "mismatches": total_mismatches.item(),
                    "final_sequence": expected_value,
                    "stream": "non-default",
                },
                sort_keys=True,
            )
        )

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
