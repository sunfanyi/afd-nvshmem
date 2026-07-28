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
set(NVSHMEM_HOST_LIBRARY_SOURCE "${NVSHMEM_HOME}/lib/libnvshmem_host.so.3")
set(NVSHMEM_BOOTSTRAP_UID_SOURCE "${NVSHMEM_HOME}/lib/nvshmem_bootstrap_uid.so.3")
if(NOT EXISTS "${NVSHMEM_BOOTSTRAP_UID_SOURCE}")
    set(NVSHMEM_BOOTSTRAP_UID_SOURCE "${NVSHMEM_HOME}/lib/nvshmem_bootstrap_uid.so")
endif()

foreach(REQUIRED_PATH
        "${NVSHMEM_INCLUDE_DIR}/nvshmem.h"
        "${NVSHMEM_INCLUDE_DIR}/nvshmemx.h"
        "${NVSHMEM_DEVICE_LIBRARY}"
        "${NVSHMEM_HOST_LIBRARY_SOURCE}"
        "${NVSHMEM_BOOTSTRAP_UID_SOURCE}")
    if(NOT EXISTS "${REQUIRED_PATH}")
        message(FATAL_ERROR "Required NVSHMEM file not found: ${REQUIRED_PATH}")
    endif()
endforeach()

# The stock libnvshmem_host.so.3 holds process-global singleton state: two
# users in one process (this extension for the A<->F world, DeepEP
# low-latency for the F-only EP world) would merge into a single incompatible
# world and deadlock. Ship renamed private copies of the host runtime and of
# the UID bootstrap plugin it dlopens, so this extension gets a fully
# isolated NVSHMEM instance. See https://github.com/NVIDIA/nvshmem/issues/68.
find_program(PATCHELF_EXECUTABLE patchelf REQUIRED)

set(AFD_NVSHMEM_HOST_SONAME "libafd_nvshmem_host.so.3")
set(AFD_NVSHMEM_HOST_LIBRARY "${CMAKE_CURRENT_BINARY_DIR}/${AFD_NVSHMEM_HOST_SONAME}")
set(AFD_NVSHMEM_BOOTSTRAP_UID_SONAME "afd_nvshmem_bootstrap_uid.so.3")
set(AFD_NVSHMEM_BOOTSTRAP_UID_PLUGIN
    "${CMAKE_CURRENT_BINARY_DIR}/${AFD_NVSHMEM_BOOTSTRAP_UID_SONAME}")

function(afd_nvshmem_private_copy SOURCE DESTINATION SONAME)
    execute_process(
        COMMAND cp -L "${SOURCE}" "${DESTINATION}"
        COMMAND_ERROR_IS_FATAL ANY)
    execute_process(
        COMMAND chmod u+w "${DESTINATION}"
        COMMAND_ERROR_IS_FATAL ANY)
    execute_process(
        COMMAND "${PATCHELF_EXECUTABLE}" --set-soname "${SONAME}" "${DESTINATION}"
        COMMAND_ERROR_IS_FATAL ANY)
    # Rebind a possible dependency on the stock host runtime to the private
    # copy (no-op when the DT_NEEDED entry is absent).
    execute_process(
        COMMAND "${PATCHELF_EXECUTABLE}" --replace-needed
                libnvshmem_host.so.3 "${AFD_NVSHMEM_HOST_SONAME}" "${DESTINATION}"
        COMMAND_ERROR_IS_FATAL ANY)
endfunction()

afd_nvshmem_private_copy(
    "${NVSHMEM_HOST_LIBRARY_SOURCE}"
    "${AFD_NVSHMEM_HOST_LIBRARY}"
    "${AFD_NVSHMEM_HOST_SONAME}")
afd_nvshmem_private_copy(
    "${NVSHMEM_BOOTSTRAP_UID_SOURCE}"
    "${AFD_NVSHMEM_BOOTSTRAP_UID_PLUGIN}"
    "${AFD_NVSHMEM_BOOTSTRAP_UID_SONAME}")

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
    "${AFD_NVSHMEM_HOST_LIBRARY}"
    ${CMAKE_DL_LIBS}
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
    BUILD_RPATH "${CMAKE_CURRENT_BINARY_DIR}"
    INSTALL_RPATH "$ORIGIN"
)

# NVSHMEM's static device archive must not export a second set of global
# device symbols from this shared extension.
target_link_options(_C PRIVATE "LINKER:--exclude-libs,ALL")

install(TARGETS _C LIBRARY DESTINATION afd_nvshmem)
install(
    FILES "${AFD_NVSHMEM_HOST_LIBRARY}" "${AFD_NVSHMEM_BOOTSTRAP_UID_PLUGIN}"
    DESTINATION afd_nvshmem)

# The wheel redistributes NVSHMEM binaries; ship their license text with it.
set(NVSHMEM_LICENSE_FILE "")
foreach(LICENSE_CANDIDATE
        "${NVSHMEM_HOME}/License.txt"
        "${NVSHMEM_HOME}/LICENSE.txt"
        "${NVSHMEM_HOME}/LICENSE"
        "${NVSHMEM_HOME}/share/doc/nvshmem/License.txt"
        "${NVSHMEM_HOME}/share/doc/nvshmem/copyright")
    if(EXISTS "${LICENSE_CANDIDATE}")
        set(NVSHMEM_LICENSE_FILE "${LICENSE_CANDIDATE}")
        break()
    endif()
endforeach()
if(NVSHMEM_LICENSE_FILE)
    install(
        FILES "${NVSHMEM_LICENSE_FILE}"
        DESTINATION afd_nvshmem
        RENAME "NVSHMEM_LICENSE.txt")
else()
    message(WARNING
        "No NVSHMEM license file found under ${NVSHMEM_HOME}; the wheel "
        "redistributes NVSHMEM binaries without their license text")
endif()
