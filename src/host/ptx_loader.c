//
// ptx_loader.c — PTX kernel loader using CUDA Driver API
//

#include "ptx_loader.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK_CUDA(call) do { \
    CUresult err = (call); \
    if (err != CUDA_SUCCESS) { \
        const char *err_str; \
        cuGetErrorString(err, &err_str); \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, err_str); \
        return -1; \
    } \
} while(0)

static int load_kernel(CUmodule module, const char *name, CUfunction *func) {
    CHECK_CUDA(cuModuleGetFunction(func, module, name));
    return 0;
}

static int load_ptx_file(CUmodule *module, const char *path) {
    CHECK_CUDA(cuModuleLoad(module, path));
    return 0;
}

int ptx_init(const char *ptx_dir, ptx_kernels_t *kernels) {
    memset(kernels, 0, sizeof(*kernels));

    CHECK_CUDA(cuInit(0));

    CUdevice device;
    CHECK_CUDA(cuDeviceGet(&device, 0));

    CUcontext context;
    CHECK_CUDA(cuCtxCreate(&context, 0, device));

    // Load each PTX file as a separate module
    char path[512];

    snprintf(path, sizeof(path), "%s/cosine_similarity.ptx", ptx_dir);
    if (load_ptx_file(&kernels->module, path) != 0) {
        fprintf(stderr, "Failed to load %s\n", path);
        // Try loading all from a single file
        snprintf(path, sizeof(path), "%s/cosine_similarity.ptx", ptx_dir);
        return -1;
    }

    // Load kernel functions
    if (load_kernel(kernels->module, "cosine_similarity", &kernels->cosine_similarity) != 0) return -1;
    if (load_kernel(kernels->module, "batch_predict", &kernels->batch_predict) != 0) return -1;
    if (load_kernel(kernels->module, "balance_check", &kernels->balance_check) != 0) return -1;
    if (load_kernel(kernels->module, "decay", &kernels->decay) != 0) return -1;
    if (load_kernel(kernels->module, "vibe_compute", &kernels->vibe_compute) != 0) return -1;
    if (load_kernel(kernels->module, "correlation_matrix", &kernels->correlation_matrix) != 0) return -1;
    if (load_kernel(kernels->module, "merge_candidates", &kernels->merge_candidates) != 0) return -1;

    printf("PTX kernels loaded successfully\n");
    return 0;
}

void ptx_cleanup(ptx_kernels_t *kernels) {
    if (kernels->module) {
        cuModuleUnload(kernels->module);
    }
}

int ptx_alloc_copy(void **d_ptr, const void *h_ptr, size_t size) {
    CHECK_CUDA(cuMemAlloc((CUdeviceptr *)d_ptr, size));
    CHECK_CUDA(cuMemcpyHtoD((CUdeviceptr)*d_ptr, h_ptr, size));
    return 0;
}

int ptx_alloc(void **d_ptr, size_t size) {
    CHECK_CUDA(cuMemAlloc((CUdeviceptr *)d_ptr, size));
    return 0;
}

int ptx_copy_to_host(void *h_ptr, const void *d_ptr, size_t size) {
    CHECK_CUDA(cuMemcpyDtoH(h_ptr, (CUdeviceptr)d_ptr, size));
    return 0;
}

void ptx_free(void *d_ptr) {
    if (d_ptr) cuMemFree((CUdeviceptr)d_ptr);
}

int ptx_launch_1d(CUfunction kernel, int n, void **args, int block_size) {
    int grid = (n + block_size - 1) / block_size;
    CHECK_CUDA(cuLaunchKernel(kernel, grid, 1, 1, block_size, 1, 1,
                               0, NULL, args, NULL));
    return 0;
}

int ptx_launch_2d(CUfunction kernel, int nx, int ny, void **args, int block_size) {
    CHECK_CUDA(cuLaunchKernel(kernel, nx, ny, 1, block_size, 1, 1,
                               0, NULL, args, NULL));
    return 0;
}

int ptx_synchronize(void) {
    CHECK_CUDA(cuCtxSynchronize());
    return 0;
}
