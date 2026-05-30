//
// main.c — PTX kernel test runner
// Grand Pattern Fibonacci Dual-Direction Architecture
//

#include "ptx_loader.h"
#include <stdio.h>
#include <stdlib.h>

// External test functions
extern int test_cosine_similarity(ptx_kernels_t *kernels);
extern int test_batch_predict(ptx_kernels_t *kernels);
extern int test_balance(ptx_kernels_t *kernels);
extern int test_decay(ptx_kernels_t *kernels);
extern int test_vibe(ptx_kernels_t *kernels);
extern int test_correlation(ptx_kernels_t *kernels);
extern int test_merge(ptx_kernels_t *kernels);
extern int test_e2e(ptx_kernels_t *kernels);

int main(int argc, char **argv) {
    const char *ptx_dir = PTX_KERNEL_DIR;
    if (argc > 1) ptx_dir = argv[1];

    printf("Grand Pattern — PTX Kernel Tests\n");
    printf("=================================\n");
    printf("Loading PTX from: %s\n\n", ptx_dir);

    ptx_kernels_t kernels;
    if (ptx_init(ptx_dir, &kernels) != 0) {
        fprintf(stderr, "Failed to initialize PTX kernels\n");
        fprintf(stderr, "Note: This requires an NVIDIA GPU with CUDA support.\n");
        fprintf(stderr, "The PTX kernels target sm_70 (Volta+).\n");
        return 1;
    }

    int passed = 0, failed = 0, total = 8;

    if (test_cosine_similarity(&kernels) == 0) passed++; else failed++;
    if (test_batch_predict(&kernels) == 0) passed++; else failed++;
    if (test_balance(&kernels) == 0) passed++; else failed++;
    if (test_decay(&kernels) == 0) passed++; else failed++;
    if (test_vibe(&kernels) == 0) passed++; else failed++;
    if (test_correlation(&kernels) == 0) passed++; else failed++;
    if (test_merge(&kernels) == 0) passed++; else failed++;
    if (test_e2e(&kernels) == 0) passed++; else failed++;

    printf("\n=================================\n");
    printf("Results: %d/%d passed, %d failed\n", passed, total, failed);

    ptx_cleanup(&kernels);
    return failed > 0 ? 1 : 0;
}
