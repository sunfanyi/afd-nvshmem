# afd-nvshmem

`afd-nvshmem` is a small native NVSHMEM provider for the SGLang
Attention-FFN Disaggregation data plane.

It owns the complete process lifecycle:

- unique-ID bootstrap and initialization;
- one aligned symmetric byte arena exposed as non-owning PyTorch views;
- stream-ordered put-with-signal;
- GPU-resident dynamic signal wait;
- collective free and finalize.

The provider requires NVSHMEM 3.4.5 or newer. It is built against the target
NVSHMEM headers and libraries; binary wheels are not portable across CPU
architectures, Python ABIs, CUDA GPU architectures, or arbitrary NVSHMEM
runtime versions.

## Private NVSHMEM runtime

The wheel bundles renamed private copies of the NVSHMEM host runtime
(`libafd_nvshmem_host.so.3`) and of the UID bootstrap plugin
(`afd_nvshmem_bootstrap_uid.so.3`), taken from `NVSHMEM_HOME` at build time.
The extension links and loads only these copies, so the provider runs its own
NVSHMEM instance and can coexist in one process with other users of the stock
`libnvshmem_host.so.3` (e.g. DeepEP low-latency), which would otherwise merge
two incompatible worlds and deadlock
(see [NVIDIA/nvshmem#68](https://github.com/NVIDIA/nvshmem/issues/68)).

The isolation currently covers the built-in P2P transport only, which spans
the NVLink domain: same-node peers and, on GB-series systems with IMEX
channels configured, multi-node NVLink (MNNVL) peers in the same fabric.
Remote transports (IBRC/IBGDA/...) are dlopen'd stock plugins with hardcoded
names that would be shared with the other instance, so initialization forces
`NVSHMEM_REMOTE_TRANSPORT=none` and `NVSHMEM_IB_ENABLE_IBGDA=0` for the
private instance; peers outside the NVLink domain are unsupported until
renamed copies of the transport plugins are shipped as well. Ambient
`NVSHMEM_*` variables for other libraries are untouched, and initialization
fails fast if a stock runtime preempted the private symbol binding (e.g. via
`LD_PRELOAD`).

## Build

Install PyTorch, `patchelf`, and the build requirements, then build on the
target machine (`--no-build-isolation` skips the declared build requirements,
so `patchelf` must already be on `PATH`):

```bash
export NVSHMEM_HOME=/path/to/nvshmem
export TORCH_CUDA_ARCH_LIST=10.3  # Use 9.0 for Hopper.
uv build --wheel --no-build-isolation
```

## Usage

```python
from afd_nvshmem import Provider, get_unique_id

uid = get_unique_id()  # Generate on one rank and broadcast as opaque bytes.
provider = Provider.init(uid, rank, world_size, device)
arena = provider.allocate_arena(num_bytes)

provider.put_signal_on_stream(payload, signal, peer)
provider.wait_next_on_stream(signal, expected, arrivals=num_peers)
provider.close()
```

`arrivals` defaults to 1 and must be non-negative. The wait advances
`expected` by `arrivals` and blocks until the signal reaches that value;
`arrivals=0` leaves `expected` unchanged and completes immediately, so a
peerless rank can keep a uniform call sequence. The wait kernel polls the
local signal with the nonblocking `nvshmem_uint64_test` API and is launched
per-PE on the current stream, so wait launches carry no cross-PE ordering
requirement. A successful test also performs NVSHMEM's required consistency
operation before the payload is consumed. Only init, arena allocation, and
`close()` are collective and must be called by all PEs in the same order.
Arena tensors and views become invalid immediately after `close()`.

Run the smoke test with two visible GPUs:

```bash
torchrun --standalone --nproc-per-node=2 tests/provider_smoke.py
```
