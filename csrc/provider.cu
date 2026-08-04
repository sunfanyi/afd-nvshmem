#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include "provider.h"

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>
#include <dlfcn.h>
#include <nvshmem.h>
#include <nvshmemx.h>

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <utility>
#include <vector>

#if NVSHMEM_VENDOR_MAJOR_VERSION < 3 ||                                      \
    (NVSHMEM_VENDOR_MAJOR_VERSION == 3 &&                                    \
     NVSHMEM_VENDOR_MINOR_VERSION < 4) ||                                    \
    (NVSHMEM_VENDOR_MAJOR_VERSION == 3 &&                                    \
     NVSHMEM_VENDOR_MINOR_VERSION == 4 &&                                    \
     NVSHMEM_VENDOR_PATCH_VERSION < 5)
#error "afd-nvshmem requires NVSHMEM >= 3.4.5"
#endif

namespace afd_nvshmem {
namespace {

constexpr size_t kArenaAlignment = 256;

enum class ProviderLifecycle {
  kFresh,
  kReady,
  kFailed,
  kTerminal,
};

struct ProviderState {
  ProviderLifecycle lifecycle = ProviderLifecycle::kFresh;
  int rank = -1;
  int world_size = 0;
  int device = -1;
  void* arena_ptr = nullptr;
  size_t arena_size = 0;
  at::Tensor arena;
};

ProviderState& State() {
  static ProviderState state;
  return state;
}

bool VersionAtLeast(int major, int minor, int patch) {
  if (major != 3) {
    return major > 3;
  }
  if (minor != 4) {
    return minor > 4;
  }
  return patch >= 5;
}

size_t TensorBytes(const at::Tensor& tensor) {
  TORCH_CHECK(tensor.numel() >= 0, "tensor has an invalid element count");
  const auto numel = static_cast<size_t>(tensor.numel());
  const auto element_size = static_cast<size_t>(tensor.element_size());
  TORCH_CHECK(
      numel <= std::numeric_limits<size_t>::max() / element_size,
      "tensor byte size overflows size_t");
  return numel * element_size;
}

void CheckInitialized() {
  TORCH_CHECK(
      State().lifecycle == ProviderLifecycle::kReady,
      "afd-nvshmem provider is not ready");
}

void CheckArenaTensor(
    const at::Tensor& tensor,
    at::ScalarType dtype,
    const char* name) {
  CheckInitialized();
  const ProviderState& state = State();
  TORCH_CHECK(state.arena_ptr != nullptr, "afd-nvshmem arena is not allocated");
  TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
  TORCH_CHECK(
      tensor.get_device() == state.device,
      name,
      " must be on cuda:",
      state.device);
  TORCH_CHECK(tensor.scalar_type() == dtype, name, " has an invalid dtype");
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
  TORCH_CHECK(tensor.dim() == 1, name, " must be one-dimensional");
  TORCH_CHECK(tensor.numel() > 0, name, " must not be empty");

  const uintptr_t base = reinterpret_cast<uintptr_t>(state.arena_ptr);
  const uintptr_t pointer = reinterpret_cast<uintptr_t>(tensor.data_ptr());
  const size_t bytes = TensorBytes(tensor);
  TORCH_CHECK(
      pointer >= base && pointer - base <= state.arena_size &&
          bytes <= state.arena_size - (pointer - base),
      name,
      " must be a view inside the symmetric arena");
}

bool RangesOverlap(
    const void* lhs_pointer,
    size_t lhs_bytes,
    const void* rhs_pointer,
    size_t rhs_bytes) {
  const uintptr_t lhs = reinterpret_cast<uintptr_t>(lhs_pointer);
  const uintptr_t rhs = reinterpret_cast<uintptr_t>(rhs_pointer);
  return lhs < rhs + rhs_bytes && rhs < lhs + lhs_bytes;
}

void CheckSignalTensor(const at::Tensor& tensor, const char* name) {
  CheckArenaTensor(tensor, at::ScalarType::UInt64, name);
  TORCH_CHECK(tensor.numel() == 1, name, " must contain exactly one uint64");
  TORCH_CHECK(
      reinterpret_cast<uintptr_t>(tensor.data_ptr()) % alignof(uint64_t) == 0,
      name,
      " must be 8-byte aligned");
}

__global__ void WaitNextKernel(
    uint64_t* signal,
    uint64_t* expected,
    uint64_t arrivals) {
  if (threadIdx.x == 0 && arrivals != 0) {
    const uint64_t target = (*expected += arrivals);
    nvshmem_signal_wait_until(signal, NVSHMEM_CMP_GE, target);
  }
}

// The extension links a renamed private copy of the NVSHMEM host runtime.
// If a stock libnvshmem_host.so.3 was injected into the global symbol scope
// (LD_PRELOAD, RTLD_GLOBAL), the dynamic linker would silently bind our
// calls to that shared instance, merging two incompatible worlds. Fail fast
// instead of hanging later.
void CheckPrivateRuntimeBinding() {
  Dl_info info{};
  const int status =
      dladdr(reinterpret_cast<const void*>(&nvshmem_barrier_all), &info);
  TORCH_CHECK(
      status != 0 && info.dli_fname != nullptr,
      "failed to resolve the NVSHMEM host runtime binding");
  TORCH_CHECK(
      std::strstr(info.dli_fname, "libafd_nvshmem_host.so") != nullptr,
      "NVSHMEM symbols are bound to '",
      info.dli_fname,
      "' instead of the private libafd_nvshmem_host.so.3; another NVSHMEM "
      "runtime preempted the binding (e.g. via LD_PRELOAD or RTLD_GLOBAL)");
}

}  // namespace

std::vector<int64_t> GetUniqueId() {
  TORCH_CHECK(
      State().lifecycle == ProviderLifecycle::kFresh,
      "unique ID must be generated before provider initialization");
  CheckPrivateRuntimeBinding();

  nvshmemx_uniqueid_t unique_id = NVSHMEMX_UNIQUEID_INITIALIZER;
  const int status = nvshmemx_get_uniqueid(&unique_id);
  TORCH_CHECK(status == 0, "nvshmemx_get_uniqueid failed with status ", status);

  const auto* bytes = reinterpret_cast<const uint8_t*>(&unique_id);
  std::vector<int64_t> result(sizeof(unique_id));
  for (size_t index = 0; index < sizeof(unique_id); ++index) {
    result[index] = bytes[index];
  }
  return result;
}

std::vector<int64_t> InitProvider(
    const std::vector<int64_t>& unique_id,
    int64_t rank,
    int64_t world_size,
    int64_t device) {
  ProviderState& state = State();
  TORCH_CHECK(
      state.lifecycle == ProviderLifecycle::kFresh,
      "afd-nvshmem provider cannot be initialized in its current lifecycle");
  TORCH_CHECK(world_size > 1, "world_size must be greater than one");
  TORCH_CHECK(rank >= 0 && rank < world_size, "rank is outside world_size");
  TORCH_CHECK(device >= 0, "device must be non-negative");
  TORCH_CHECK(
      unique_id.size() == sizeof(nvshmemx_uniqueid_t),
      "unique ID has ",
      unique_id.size(),
      " bytes, expected ",
      sizeof(nvshmemx_uniqueid_t));
  CheckPrivateRuntimeBinding();

  nvshmemx_uniqueid_t native_unique_id = NVSHMEMX_UNIQUEID_INITIALIZER;
  auto* native_bytes = reinterpret_cast<uint8_t*>(&native_unique_id);
  for (size_t index = 0; index < unique_id.size(); ++index) {
    TORCH_CHECK(
        unique_id[index] >= 0 && unique_id[index] <= 255,
        "unique ID byte ",
        index,
        " is outside [0, 255]");
    native_bytes[index] = static_cast<uint8_t>(unique_id[index]);
  }

  C10_CUDA_CHECK(cudaSetDevice(static_cast<int>(device)));

  nvshmemx_init_attr_t attributes = NVSHMEMX_INIT_ATTR_INITIALIZER;
  int status = nvshmemx_set_attr_uniqueid_args(
      static_cast<int>(rank),
      static_cast<int>(world_size),
      &native_unique_id,
      &attributes);
  TORCH_CHECK(
      status == 0,
      "nvshmemx_set_attr_uniqueid_args failed with status ",
      status);

  status = nvshmemx_init_attr(NVSHMEMX_INIT_WITH_UNIQUEID, &attributes);
  if (status != 0) {
    state.lifecycle = ProviderLifecycle::kTerminal;
    TORCH_CHECK(false, "nvshmemx_init_attr failed with status ", status);
  }

  const int actual_rank = nvshmem_my_pe();
  const int actual_world_size = nvshmem_n_pes();
  state.lifecycle = ProviderLifecycle::kReady;
  state.rank = actual_rank;
  state.world_size = actual_world_size;
  state.device = static_cast<int>(device);

  if (actual_rank != rank || actual_world_size != world_size) {
    state.lifecycle = ProviderLifecycle::kFailed;
    TORCH_CHECK(
        false,
        "NVSHMEM topology mismatch: expected rank ",
        rank,
        "/",
        world_size,
        ", got ",
        actual_rank,
        "/",
        actual_world_size);
  }

  int runtime_major = 0;
  int runtime_minor = 0;
  int runtime_patch = 0;
  nvshmemx_vendor_get_version_info(
      &runtime_major, &runtime_minor, &runtime_patch);
  if (!VersionAtLeast(runtime_major, runtime_minor, runtime_patch)) {
    state.lifecycle = ProviderLifecycle::kFailed;
    TORCH_CHECK(
        false,
        "afd-nvshmem requires NVSHMEM >= 3.4.5, got ",
        runtime_major,
        ".",
        runtime_minor,
        ".",
        runtime_patch);
  }

  return {
      NVSHMEM_VENDOR_MAJOR_VERSION,
      NVSHMEM_VENDOR_MINOR_VERSION,
      NVSHMEM_VENDOR_PATCH_VERSION,
      runtime_major,
      runtime_minor,
      runtime_patch,
  };
}

at::Tensor AllocateArena(int64_t num_bytes) {
  CheckInitialized();
  ProviderState& state = State();
  const at::cuda::CUDAGuard device_guard(state.device);
  TORCH_CHECK(state.arena_ptr == nullptr, "symmetric arena is already allocated");
  TORCH_CHECK(num_bytes > 0, "arena size must be positive");
  TORCH_CHECK(
      static_cast<uint64_t>(num_bytes) <=
          static_cast<uint64_t>(std::numeric_limits<size_t>::max()),
      "arena size does not fit size_t");

  const size_t arena_size = static_cast<size_t>(num_bytes);
  void* arena_ptr = nvshmem_align(kArenaAlignment, arena_size);
  TORCH_CHECK(
      arena_ptr != nullptr, "nvshmem_align failed for ", num_bytes, " bytes");
  TORCH_CHECK(
      reinterpret_cast<uintptr_t>(arena_ptr) % kArenaAlignment == 0,
      "NVSHMEM returned a misaligned arena");

  const cudaStream_t stream =
      at::cuda::getCurrentCUDAStream(state.device).stream();
  C10_CUDA_CHECK(cudaMemsetAsync(arena_ptr, 0, arena_size, stream));
  C10_CUDA_CHECK(cudaStreamSynchronize(stream));
  nvshmem_barrier_all();

  auto options = at::TensorOptions()
                     .dtype(at::kByte)
                     .device(at::Device(at::kCUDA, state.device));
  at::Tensor arena = at::from_blob(
      arena_ptr,
      {num_bytes},
      [](void*) {},
      std::move(options));

  state.arena_ptr = arena_ptr;
  state.arena_size = arena_size;
  state.arena = arena;
  return arena;
}

void PutSignalOnStream(
    const at::Tensor& payload,
    const at::Tensor& signal,
    int64_t peer) {
  CheckArenaTensor(payload, at::kByte, "payload");
  CheckSignalTensor(signal, "signal");
  const ProviderState& state = State();
  const at::cuda::CUDAGuard device_guard(state.device);
  TORCH_CHECK(peer >= 0 && peer < state.world_size, "peer is outside world_size");
  TORCH_CHECK(peer != state.rank, "peer must differ from the local rank");

  const size_t payload_bytes = TensorBytes(payload);
  TORCH_CHECK(
      !RangesOverlap(
          payload.data_ptr(),
          payload_bytes,
          signal.data_ptr(),
          TensorBytes(signal)),
      "payload and signal must not overlap");

  const cudaStream_t stream =
      at::cuda::getCurrentCUDAStream(state.device).stream();
  nvshmemx_putmem_signal_on_stream(
      payload.data_ptr(),
      payload.data_ptr(),
      payload_bytes,
      signal.data_ptr<uint64_t>(),
      1,
      NVSHMEM_SIGNAL_ADD,
      static_cast<int>(peer),
      stream);
}

void WaitNextOnStream(
    const at::Tensor& signal,
    const at::Tensor& expected,
    int64_t arrivals) {
  CheckSignalTensor(signal, "signal");
  CheckSignalTensor(expected, "expected");
  TORCH_CHECK(arrivals >= 0, "arrivals must be non-negative");
  const ProviderState& state = State();
  const at::cuda::CUDAGuard device_guard(state.device);
  TORCH_CHECK(
      !RangesOverlap(
          signal.data_ptr(),
          TensorBytes(signal),
          expected.data_ptr(),
          TensorBytes(expected)),
      "signal and expected must not overlap");

  uint64_t* signal_pointer = signal.data_ptr<uint64_t>();
  uint64_t* expected_pointer = expected.data_ptr<uint64_t>();
  uint64_t arrival_count = static_cast<uint64_t>(arrivals);
  void* arguments[] = {&signal_pointer, &expected_pointer, &arrival_count};
  const cudaStream_t stream =
      at::cuda::getCurrentCUDAStream(state.device).stream();
  const int status = nvshmemx_collective_launch(
      reinterpret_cast<const void*>(WaitNextKernel),
      dim3(1),
      dim3(1),
      arguments,
      0,
      stream);
  TORCH_CHECK(
      status == 0,
      "nvshmemx_collective_launch failed with status ",
      status);
}

void CloseProvider() {
  CheckInitialized();
  ProviderState& state = State();
  const at::cuda::CUDAGuard device_guard(state.device);
  C10_CUDA_CHECK(cudaDeviceSynchronize());
  nvshmem_barrier_all();

  void* arena_ptr = state.arena_ptr;
  state.arena = at::Tensor();
  state.arena_ptr = nullptr;
  state.arena_size = 0;
  if (arena_ptr != nullptr) {
    nvshmem_free(arena_ptr);
  }
  nvshmem_finalize();
  state.lifecycle = ProviderLifecycle::kTerminal;
  state.rank = -1;
  state.world_size = 0;
  state.device = -1;
}

}  // namespace afd_nvshmem
