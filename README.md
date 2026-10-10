# afd-nvshmem

`afd-nvshmem` provides the NVSHMEM transport used by SGLang Attention-FFN
Disaggregation (AFD). It exposes a symmetric memory arena as PyTorch tensors,
stream-ordered transfers with remote signaling, and CUDA Graph-compatible
signal waits. SGLang manages the AFD protocol and buffer layout; this package
provides the communication primitives.

## Requirements

- Linux with a CUDA toolkit and a C++20 compiler compatible with the installed
  PyTorch and NVSHMEM versions.
- Python 3.10 or newer and CUDA-enabled PyTorch 2.11 or newer.
- NVSHMEM 3.4.5 or newer, including the headers, static device library, host
  library, UID bootstrap plugin, and IBGDA transport plugin.
- CMake 3.26 or newer, `uv`, and `patchelf`.

Use one process per GPU. By default, all participating GPUs must be reachable
through NVLink. Multi-node NVLink requires IMEX configuration on supported
systems. For peers outside that domain, select `remote_transport="ibgda"`
and configure the system for NVSHMEM IBGDA. Other remote transports are not
supported.

## Build and install

Install CUDA-enabled PyTorch in the build environment first. From the
repository root, install the remaining build requirements and build a wheel:

```bash
python -m pip install 'scikit-build-core>=0.10' wheel patchelf
export NVSHMEM_HOME=/path/to/nvshmem
uv build --wheel --no-build-isolation
python -m pip install dist/afd_nvshmem-*.whl
```

`NVSHMEM_HOME` must contain `include/` and `lib/`. The build links against
that installation and bundles private copies of its host runtime and plugins.
Build and install with the same PyTorch environment. Wheels depend on the CPU
architecture, Python ABI, CUDA architectures, and the bundled NVSHMEM version.

The build uses the NVSHMEM installation selected by `NVSHMEM_HOME`; it does
not pin a release. Newer versions can be used while the required APIs and
library layout remain compatible.

Set `TORCH_CUDA_ARCH_LIST` to select target GPU architectures, for example:

```bash
TORCH_CUDA_ARCH_LIST="9.0 10.0 10.3" uv build --wheel --no-build-isolation
```

PyTorch configures the extension's CUDA targets. The selected CUDA toolkit
must support every requested architecture.

### NVSHMEM license file

The build requires the license from the NVSHMEM distribution being bundled.
It searches `NVSHMEM_HOME`; if the license is elsewhere, pass its path:

```bash
uv build --wheel --no-build-isolation \
  -C cmake.define.NVSHMEM_LICENSE_FILE=/path/to/nvshmem/License.txt
```

NVIDIA's Python packages may store the file under `.dist-info/licenses/`
rather than `NVSHMEM_HOME`. For an installation from `nvidia-nvshmem-cu13`,
locate it with:

```bash
python - <<'PY'
from importlib.metadata import distribution

dist = distribution("nvidia-nvshmem-cu13")
for path in dist.files:
    if path.name == "License.txt":
        print(dist.locate_file(path))
PY
```

Use the license from the same distribution as `NVSHMEM_HOME`.

## Verify the installation

Run the [provider smoke test](tests/provider_smoke.py) with two visible GPUs
in one NVLink domain:

```bash
torchrun --standalone --nproc-per-node=2 tests/provider_smoke.py
```

The test checks payload transfers on a non-default CUDA stream, signal
counters, CUDA Graph replay, and collective shutdown. A successful run prints
`"status": "PASS"` and zero payload mismatches. The test is also a complete
example of UID exchange, arena layout, and request/response transfers.

For an IBGDA deployment, launch the same test with exactly two processes
across the participating hosts and add `--remote-transport ibgda`.

## Python API

Import `Provider` and `get_unique_id` from `afd_nvshmem`.

| API | Behavior |
| --- | --- |
| `get_unique_id(*, remote_transport="none")` | Returns opaque UID bytes. Generate the UID on one rank before initialization and distribute it to all participating ranks through an external control channel. |
| `Provider.init(unique_id, rank, world_size, device, *, remote_transport="none")` | Initializes the NVSHMEM group. All ranks must use the same UID, world size, and transport. `device` is the process-local CUDA device index. |
| `provider.allocate_arena(num_bytes)` | Collectively allocates one zero-initialized, 256-byte-aligned `torch.uint8` arena. All ranks must allocate the same size and use the same layout. |
| `provider.put_signal_on_stream(payload, signal, peer, use_signal_set=False)` | Copies `payload` to the same arena offset on `peer`, then updates the remote signal at the corresponding offset. |
| `provider.wait_next_on_stream(signal, expected, arrivals=1, use_signal_set=False)` | Enqueues a GPU wait for the next expected arrivals. The default mode increments `expected` by `arrivals` and waits until `signal >= expected`. |
| `provider.close()` | Synchronizes the device, performs a group barrier, frees the arena, and finalizes the private NVSHMEM instance. |

`world_size` must be greater than one. Only one provider can be initialized
per process, and it cannot be reinitialized after `close()`. Initialization,
arena allocation, and shutdown require all ranks to participate.

### Buffers and streams

All communication tensors must be contiguous, one-dimensional views into
the provider's arena. Payloads must be nonempty and use `torch.uint8`;
`signal` and `expected` each contain one 8-byte-aligned `torch.uint64`.
The payload and signal must not overlap, and `signal` and `expected` must
occupy separate locations.
The peer must differ from the local rank.

Transfers and waits use PyTorch's current CUDA stream on the provider's
device. Work consuming a received payload must follow its wait on that
stream, or use an explicit stream dependency. Both operations can be captured
in a CUDA Graph: the expected sequence is updated on the GPU during replay.
In the default ADD mode, `arrivals=0` leaves the counters unchanged and does
not wait.

SET mode (`use_signal_set=True`) sets the remote signal to one; the receiver
waits for one and resets it to zero. Use it only with one writer per signal,
`arrivals=1`, and a protocol that prevents the sender from reusing the slot
before the receiver resets it. Sender and receiver must select the same mode.

Call `close()` on every rank after communication has finished. Retained arena
views become invalid when it returns and must not be used afterward.

## Runtime isolation

An AFD process may also use DeepEP with a different NVSHMEM group. To keep
the groups separate, the wheel contains renamed copies of the host runtime
and its plugins:

- `libafd_nvshmem_host.so.3`
- `afd_nvshmem_bootstrap_uid.so.3`
- `afdpriv_transport_ibgda.so.3`

The extension links to the private host library. UID generation and
initialization temporarily select the private plugins and requested transport,
then restore the previous environment variables. Initialization fails if a
stock NVSHMEM runtime has preempted the private symbols, for example through
`LD_PRELOAD` or `RTLD_GLOBAL`. See
[NVIDIA/nvshmem#68](https://github.com/NVIDIA/nvshmem/issues/68) for discussion
of independent modules sharing NVSHMEM process state.

## License

This project's source is licensed under [Apache-2.0](LICENSE). Bundled
NVSHMEM binaries retain the terms of their distribution; the wheel includes
that distribution's license as `afd_nvshmem/NVSHMEM_LICENSE.txt`. Including
this file does not establish permission to redistribute modified NVSHMEM
binaries.
