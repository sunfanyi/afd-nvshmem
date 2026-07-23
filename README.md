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

## Build

Install PyTorch and the build requirements, then build on the target machine:

```bash
export NVSHMEM_HOME=/path/to/nvshmem
export TORCH_CUDA_ARCH_LIST=10.3  # Use 9.0 for Hopper.
uv build --wheel --no-build-isolation
```

The matching `libnvshmem_host.so.3` must be selected before Python imports
PyTorch. Changing `NVSHMEM_HOME` cannot replace a runtime already loaded into
the process.

## Usage

```python
from afd_nvshmem import Provider, get_unique_id

uid = get_unique_id()  # Generate on one rank and broadcast as opaque bytes.
provider = Provider.init(uid, rank, world_size, device)
arena = provider.allocate_arena(num_bytes)

provider.put_signal_on_stream(payload, signal, peer)
provider.wait_next_on_stream(signal, expected)
provider.close()
```

All ranks must call collective operations in the same order. Arena tensors and
views become invalid immediately after `close()`.

Run the smoke test with two visible GPUs:

```bash
torchrun --standalone --nproc-per-node=2 tests/provider_smoke.py
```
