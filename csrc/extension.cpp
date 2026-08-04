#include <Python.h>
#include <torch/library.h>

#include "provider.h"

TORCH_LIBRARY(afd_nvshmem, module) {
  module.def("get_unique_id() -> int[]");
  module.impl("get_unique_id", &afd_nvshmem::GetUniqueId);

  module.def(
      "init_provider(int[] unique_id, int rank, int world_size, int device) "
      "-> int[]");
  module.impl("init_provider", &afd_nvshmem::InitProvider);

  module.def("allocate_arena(int num_bytes) -> Tensor");
  module.impl("allocate_arena", &afd_nvshmem::AllocateArena);

  module.def(
      "put_signal_on_stream(Tensor payload, Tensor signal, int peer) -> ()");
  module.impl("put_signal_on_stream", &afd_nvshmem::PutSignalOnStream);

  module.def(
      "wait_next_on_stream(Tensor signal, Tensor(a!) expected, int arrivals) "
      "-> ()");
  module.impl("wait_next_on_stream", &afd_nvshmem::WaitNextOnStream);

  module.def("close_provider() -> ()");
  module.impl("close_provider", &afd_nvshmem::CloseProvider);
}

PyMODINIT_FUNC PyInit__C() {
  static struct PyModuleDef module = {
      PyModuleDef_HEAD_INIT, "_C", nullptr, 0, nullptr};
  return PyModule_Create(&module);
}
