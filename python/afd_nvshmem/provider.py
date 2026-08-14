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
_PRIVATE_IBGDA_PLUGIN = Path(__file__).resolve().parent / "afdpriv_transport_ibgda.so.3"
_REMOTE_TRANSPORTS = frozenset({"none", "ibgda"})


class VersionInfo(NamedTuple):
    compiled: tuple[int, int, int]
    runtime: tuple[int, int, int]


def _normalize_remote_transport(remote_transport: str) -> str:
    value = remote_transport.lower()
    if value not in _REMOTE_TRANSPORTS:
        raise ValueError(
            "remote_transport must be 'none' or 'ibgda', " f"got {remote_transport!r}"
        )
    return value


@contextmanager
def _isolated_runtime_env(remote_transport: str) -> Iterator[None]:
    """Scope env vars so the private NVSHMEM instance stays isolated.

    The extension links a renamed private copy of the NVSHMEM host runtime so
    it can coexist with other NVSHMEM users in the same process (e.g. DeepEP
    low-latency on the FFN side). The private instance must dlopen its own
    copy of the UID bootstrap plugin instead of the stock one. NVSHMEM reads
    these env vars during unique-id generation and initialization only, so
    the scope covers both call sites.

    The built-in P2P transport covers the NVLink domain, including multi-node
    NVLink via IMEX on GB-series systems. IBGDA is provided as a renamed
    private plugin for peers outside that domain. Other remote transports are
    intentionally unsupported.
    """
    remote_transport = _normalize_remote_transport(remote_transport)
    if remote_transport == "ibgda" and not _PRIVATE_IBGDA_PLUGIN.is_file():
        raise RuntimeError(f"private IBGDA plugin not found: {_PRIVATE_IBGDA_PLUGIN}")

    overrides = {
        "NVSHMEM_BOOTSTRAP_UID_PLUGIN": str(_PRIVATE_BOOTSTRAP_UID_PLUGIN),
        "NVSHMEM_REMOTE_TRANSPORT": remote_transport,
        "NVSHMEM_IB_ENABLE_IBGDA": "1" if remote_transport == "ibgda" else "0",
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


def get_unique_id(*, remote_transport: str = "none") -> bytes:
    with _isolated_runtime_env(remote_transport):
        return bytes(torch.ops.afd_nvshmem.get_unique_id())


class Provider:
    def __init__(
        self,
        unique_id: bytes,
        rank: int,
        world_size: int,
        device: int,
        *,
        remote_transport: str = "none",
    ):
        if not isinstance(unique_id, bytes):
            raise TypeError("unique_id must be bytes")
        with _isolated_runtime_env(remote_transport):
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
        *,
        remote_transport: str = "none",
    ) -> "Provider":
        return cls(
            unique_id,
            rank,
            world_size,
            device,
            remote_transport=remote_transport,
        )

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
        use_signal_set: bool = False,
    ) -> None:
        self._check_ready()
        torch.ops.afd_nvshmem.put_signal_on_stream(
            payload, signal, peer, use_signal_set
        )

    def wait_next_on_stream(
        self,
        signal: torch.Tensor,
        expected: torch.Tensor,
        arrivals: int = 1,
        use_signal_set: bool = False,
    ) -> None:
        self._check_ready()
        torch.ops.afd_nvshmem.wait_next_on_stream(
            signal, expected, arrivals, use_signal_set
        )

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
