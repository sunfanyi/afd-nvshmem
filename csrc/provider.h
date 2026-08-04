#pragma once

#include <ATen/core/Tensor.h>

#include <cstdint>
#include <vector>

namespace afd_nvshmem {

std::vector<int64_t> GetUniqueId();

std::vector<int64_t> InitProvider(
    const std::vector<int64_t>& unique_id,
    int64_t rank,
    int64_t world_size,
    int64_t device);

at::Tensor AllocateArena(int64_t num_bytes);

void PutSignalOnStream(
    const at::Tensor& payload,
    const at::Tensor& signal,
    int64_t peer);

void WaitNextOnStream(
    const at::Tensor& signal,
    const at::Tensor& expected,
    int64_t arrivals);

void CloseProvider();

}  // namespace afd_nvshmem
