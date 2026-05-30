cmake_minimum_required(VERSION 3.20)
project(grand-pattern-ptx LANGUAGES C CUDA)

set(CMAKE_C_STANDARD 11)
set(CMAKE_CUDA_STANDARD 17)

find_package(CUDA REQUIRED)

# PTX kernel files
set(PTX_SOURCES
    src/kernels/cosine_similarity.ptx
    src/kernels/batch_predict.ptx
    src/kernels/balance_check.ptx
    src/kernels/decay.ptx
    src/kernels/vibe_compute.ptx
    src/kernels/correlation_matrix.ptx
    src/kernels/merge_candidates.ptx
)

# Copy PTX files to build directory
file(COPY ${PTX_SOURCES} DESTINATION ${CMAKE_BINARY_DIR}/kernels)

# Host code
add_executable(test_ptx_kernels
    src/host/main.c
    src/host/ptx_loader.c
    src/host/test_cosine.c
    src/host/test_batch_predict.c
    src/host/test_balance.c
    src/host/test_decay.c
    src/host/test_vibe.c
    src/host/test_correlation.c
    src/host/test_merge.c
    src/host/test_e2e.c
)

target_include_directories(test_ptx_kernels PRIVATE
    ${CMAKE_SOURCE_DIR}/src/host
    ${CUDA_INCLUDE_DIRS}
)

target_link_libraries(test_ptx_kernels ${CUDA_LIBRARIES} cuda m)

target_compile_definitions(test_ptx_kernels PRIVATE
    PTX_KERNEL_DIR="${CMAKE_BINARY_DIR}/kernels"
)
