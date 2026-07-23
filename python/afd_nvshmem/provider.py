from __future__ import annotations

from typing import NamedTuple

import torch

from afd_nvshmem import _C  # noqa: F401


class VersionInfo(NamedTuple):
    compiled: tuple[int, int, int]
    runtime: tuple[int, int, int]


def get_unique_id() -> bytes:
    return bytes(torch.ops.afd_nvshmem.get_unique_id())


class Provider:
    def __init__(
        self,
        unique_id: bytes,
        rank: int,
        world_size: int,
        device: int,
    ):
        if not isinstance(unique_id, bytes):
            raise TypeError("unique_id must be bytes")
        versions = torch.ops.afd_nvshmem.init_provider(
            list(unique_id), rank, world_size, device
        )
        self._rank = rank
        self._world_size = world_size
        self._device = device
        self._version_info = VersionInfo(
            compiled=tuple(versions[:3]),
            runtime=tuple(versions[3:]),
        )
        self._arena: torch.Tensor | None = None
        self._closed = False

    @classmethod
    def init(
        cls,
        unique_id: bytes,
        rank: int,
        world_size: int,
        device: int,
    ) -> "Provider":
        return cls(unique_id, rank, world_size, device)

    @property
    def rank(self) -> int:
        return self._rank

    @property
    def world_size(self) -> int:
        return self._world_size

    @property
    def device(self) -> int:
        return self._device

    @property
    def version_info(self) -> VersionInfo:
        return self._version_info

    @property
    def arena(self) -> torch.Tensor | None:
        return self._arena

    @property
    def arena_size(self) -> int:
        return 0 if self._arena is None else self._arena.numel()

    @property
    def closed(self) -> bool:
        return self._closed

    def allocate_arena(self, num_bytes: int) -> torch.Tensor:
        self._check_open()
        if self._arena is not None:
            raise RuntimeError("symmetric arena is already allocated")
        self._arena = torch.ops.afd_nvshmem.allocate_arena(num_bytes)
        return self._arena

    def put_signal_on_stream(
        self,
        payload: torch.Tensor,
        signal: torch.Tensor,
        peer: int,
    ) -> None:
        self._check_ready()
        torch.ops.afd_nvshmem.put_signal_on_stream(payload, signal, peer)

    def wait_next_on_stream(
        self,
        signal: torch.Tensor,
        expected: torch.Tensor,
    ) -> None:
        self._check_ready()
        torch.ops.afd_nvshmem.wait_next_on_stream(signal, expected)

    def close(self) -> None:
        if self._closed:
            return
        torch.ops.afd_nvshmem.close_provider()
        self._arena = None
        self._closed = True

    def _check_open(self) -> None:
        if self._closed:
            raise RuntimeError("afd-nvshmem provider is closed")

    def _check_ready(self) -> None:
        self._check_open()
        if self._arena is None:
            raise RuntimeError("symmetric arena is not allocated")
