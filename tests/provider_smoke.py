from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist

import afd_nvshmem
from afd_nvshmem import Provider, get_unique_id


ALIGNMENT = 256
CAPACITIES = (1024, 4096)
WAVES = 64
GRAPH_REPLAYS = 64


def align_up(value: int) -> int:
    return (value + ALIGNMENT - 1) // ALIGNMENT * ALIGNMENT


def uint64_view(arena: torch.Tensor, offset: int) -> torch.Tensor:
    return arena[offset : offset + 8].view(torch.uint64)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--remote-transport",
        choices=("none", "ibgda"),
        default="none",
    )
    args = parser.parse_args()

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    if world_size != 2:
        raise RuntimeError(f"provider smoke requires two PEs, got {world_size}")

    torch.cuda.set_device(local_rank)
    dist.init_process_group("gloo")

    uid_objects = [
        get_unique_id(remote_transport=args.remote_transport) if rank == 0 else b""
    ]
    dist.broadcast_object_list(uid_objects, src=0)
    provider = Provider.init(
        uid_objects[0],
        rank=rank,
        world_size=world_size,
        device=local_rank,
        remote_transport=args.remote_transport,
    )
    mapped_ibgda_plugins = sorted(
        {
            line.rsplit(" ", 1)[-1]
            for line in Path("/proc/self/maps").read_text().splitlines()
            if "transport_ibgda.so.3" in line
        }
    )
    if args.remote_transport == "ibgda":
        private_ibgda_plugin = str(
            Path(afd_nvshmem.__file__).resolve().parent / "afdpriv_transport_ibgda.so.3"
        )
        if private_ibgda_plugin not in mapped_ibgda_plugins:
            raise RuntimeError(
                f"private IBGDA plugin is not loaded: {mapped_ibgda_plugins}"
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
                provider.put_signal_on_stream(request[:capacity], request_signal, peer)
                provider.wait_next_on_stream(response_signal, response_expected)
                mismatch_count.add_(
                    torch.count_nonzero(response[:capacity] != expected_response)
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

    dist.barrier()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        if rank == 0:
            request.add_(1)
            provider.put_signal_on_stream(request, request_signal, peer)
            provider.wait_next_on_stream(response_signal, response_expected)
        else:
            provider.wait_next_on_stream(request_signal, request_expected)
            response.copy_(request)
            response.bitwise_xor_(0xA5)
            provider.put_signal_on_stream(response, response_signal, peer)
    dist.barrier()

    with torch.cuda.stream(stream):
        for _ in range(GRAPH_REPLAYS):
            graph.replay()
    stream.synchronize()

    graph_request = torch.bitwise_xor(pattern, (WAVES - 1) & 0xFF)
    expected_graph_response = torch.bitwise_xor(graph_request.add(GRAPH_REPLAYS), 0xA5)
    if rank == 0:
        mismatch_count.add_(torch.count_nonzero(response != expected_graph_response))

    local_signal = response_signal if rank == 0 else request_signal
    local_expected = response_expected if rank == 0 else request_expected
    signal_value = int(local_signal.item())
    expected_value = int(local_expected.item())
    final_sequence = WAVES + GRAPH_REPLAYS
    if signal_value != final_sequence or expected_value != final_sequence:
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
                    "graph_replays": GRAPH_REPLAYS,
                    "remote_transport": args.remote_transport,
                    "mapped_ibgda_plugins": mapped_ibgda_plugins,
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
