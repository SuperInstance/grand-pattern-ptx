//
// ptx_loader.h — PTX kernel loader using CUDA Driver API
// Grand Pattern Fibonacci Dual-Direction Architecture
//

#ifndef PTX_LOADER_H
#define PTX_LOADER_H

#include <cuda.h>
#include <stddef.h>

typedef struct {
    CUmodule module;
    CUfunction cosine_similarity;
    CUfunction batch_predict;
    CUfunction balance_check;
    CUfunction decay;
    CUfunction vibe_compute;
    CUfunction correlation_matrix;
    CUfunction merge_candidates;
} ptx_kernels_t;

// Initialize CUDA driver API and load PTX kernels
// ptx_dir: directory containing .ptx files
int ptx_init(const char *ptx_dir, ptx_kernels_t *kernels);

// Cleanup
void ptx_cleanup(ptx_kernels_t *kernels);

// Allocate device memory and copy data
int ptx_alloc_copy(void **d_ptr, const void *h_ptr, size_t size);
int ptx_alloc(void **d_ptr, size_t size);
int ptx_copy_to_host(void *h_ptr, const void *d_ptr, size_t size);
void ptx_free(void *d_ptr);

// Launch helpers
int ptx_launch_1d(CUfunction kernel, int n, void **args, int block_size);
int ptx_launch_2d(CUfunction kernel, int nx, int ny, void **args, int block_size);
int ptx_synchronize(void);

#endif // PTX_LOADER_H
