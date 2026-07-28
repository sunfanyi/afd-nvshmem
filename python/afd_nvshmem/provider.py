from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, NamedTuple

import torch

from afd_nvshmem import _C  # noqa: F401

_PRIVATE_BOOTSTRAP_UID_PLUGIN = (
    Path(__file__).resolve().parent / "afd_nvshmem_bootstrap_uid.so.3"
)


class VersionInfo(NamedTuple):
    compiled: tuple[int, int, int]
    runtime: tuple[int, int, int]


@contextmanager
def _isolated_runtime_env() -> Iterator[None]:
    """Scope env vars so the private NVSHMEM instance stays isolated.

    The extension links a renamed private copy of the NVSHMEM host runtime so
    it can coexist with other NVSHMEM users in the same process (e.g. DeepEP
    low-latency on the FFN side). The private instance must dlopen its own
    copy of the UID bootstrap plugin instead of the stock one. NVSHMEM reads
    these env vars during unique-id generation and initialization only, so
    the scope covers both call sites.

    Only the built-in P2P transport (NVLink domain, including multi-node
    NVLink via IMEX on GB-series systems) is isolated today: remote
    transports (IBRC/IBGDA/...) are dlopen'd stock plugins with hardcoded
    names that would be shared with the other NVSHMEM instance, so they are
    forced off until renamed private copies are shipped as well. Note that
    IBGDA has its own enable switch independent of NVSHMEM_REMOTE_TRANSPORT.
    """
    overrides = {
        "NVSHMEM_BOOTSTRAP_UID_PLUGIN": str(_PRIVATE_BOOTSTRAP_UID_PLUGIN),
        "NVSHMEM_REMOTE_TRANSPORT": "none",
        "NVSHMEM_IB_ENABLE_IBGDA": "0",
    }
    saved = {key: os.environ.get(key) for key in overrides}
    os.environ.update(overrides)
    try:
        yield
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def get_unique_id() -> bytes:
    with _isolated_runtime_env():
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
        with _isolated_runtime_env():
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
