cmake_minimum_required(VERSION 3.26 FATAL_ERROR)

if(NOT CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES native)
endif()

project(afd-nvshmem LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_SHARED_LIBRARY_PREFIX "")

find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
find_package(CUDAToolkit REQUIRED)

execute_process(
    COMMAND ${Python_EXECUTABLE} -c
            "import torch; print(torch.utils.cmake_prefix_path)"
    OUTPUT_VARIABLE TORCH_CMAKE_PREFIX_PATH
    OUTPUT_STRIP_TRAILING_WHITESPACE
    COMMAND_ERROR_IS_FATAL ANY
)
list(APPEND CMAKE_PREFIX_PATH "${TORCH_CMAKE_PREFIX_PATH}")
find_package(Torch REQUIRED)

set(NVSHMEM_HOME "" CACHE PATH "NVSHMEM installation prefix")
if(NOT NVSHMEM_HOME AND DEFINED ENV{NVSHMEM_HOME})
    set(NVSHMEM_HOME "$ENV{NVSHMEM_HOME}")
endif()
if(NOT NVSHMEM_HOME)
    message(FATAL_ERROR "NVSHMEM_HOME must point to the target NVSHMEM installation")
endif()

set(NVSHMEM_INCLUDE_DIR "${NVSHMEM_HOME}/include")
set(NVSHMEM_DEVICE_LIBRARY "${NVSHMEM_HOME}/lib/libnvshmem_device.a")
set(NVSHMEM_HOST_LIBRARY "${NVSHMEM_HOME}/lib/libnvshmem_host.so.3")

foreach(REQUIRED_PATH
        "${NVSHMEM_INCLUDE_DIR}/nvshmem.h"
        "${NVSHMEM_INCLUDE_DIR}/nvshmemx.h"
        "${NVSHMEM_DEVICE_LIBRARY}"
        "${NVSHMEM_HOST_LIBRARY}")
    if(NOT EXISTS "${REQUIRED_PATH}")
        message(FATAL_ERROR "Required NVSHMEM file not found: ${REQUIRED_PATH}")
    endif()
endforeach()

Python_add_library(
    _C
    MODULE
    WITH_SOABI
    csrc/extension.cpp
    csrc/provider.cu
)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} ${TORCH_CXX_FLAGS}")

target_include_directories(_C PRIVATE "${NVSHMEM_INCLUDE_DIR}")
target_link_libraries(
    _C
    PRIVATE
    ${TORCH_LIBRARIES}
    CUDA::cudart
    "${NVSHMEM_DEVICE_LIBRARY}"
    "${NVSHMEM_HOST_LIBRARY}"
)

set_target_properties(
    _C
    PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    CUDA_RUNTIME_LIBRARY Shared
    CXX_VISIBILITY_PRESET hidden
    CUDA_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN ON
    BUILD_RPATH "${NVSHMEM_HOME}/lib"
    INSTALL_RPATH "${NVSHMEM_HOME}/lib"
)

# NVSHMEM's static device archive must not export a second set of global
# device symbols from this shared extension.
target_link_options(_C PRIVATE "LINKER:--exclude-libs,ALL")

install(TARGETS _C LIBRARY DESTINATION afd_nvshmem)
