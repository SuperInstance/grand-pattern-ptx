//
// test_cosine.c — Test cosine similarity kernel
//

#include "ptx_loader.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int test_cosine_similarity(ptx_kernels_t *kernels) {
    printf("=== Test: Cosine Similarity ===\n");

    const int n = 4;
    const int dim = 8;

    float h_a[n * dim], h_b[n * dim], h_result[n];

    // Pair 0: identical vectors → sim = 1.0
    for (int j = 0; j < dim; j++) {
        h_a[j] = 1.0f;
        h_b[j] = 1.0f;
    }

    // Pair 1: orthogonal → sim ≈ 0
    for (int j = 0; j < dim; j++) {
        h_a[dim + j] = (j < dim/2) ? 1.0f : 0.0f;
        h_b[dim + j] = (j < dim/2) ? 0.0f : 1.0f;
    }

    // Pair 2: opposite → sim = -1
    for (int j = 0; j < dim; j++) {
        h_a[2*dim + j] = 1.0f;
        h_b[2*dim + j] = -1.0f;
    }

    // Pair 3: general
    for (int j = 0; j < dim; j++) {
        h_a[3*dim + j] = (float)(j + 1);
        h_b[3*dim + j] = (float)(j + 2);
    }

    void *d_a, *d_b, *d_result;
    size_t bytes = n * dim * sizeof(float);

    if (ptx_alloc_copy(&d_a, h_a, bytes) != 0) return -1;
    if (ptx_alloc_copy(&d_b, h_b, bytes) != 0) return -1;
    if (ptx_alloc(&d_result, n * sizeof(float)) != 0) return -1;

    unsigned int u_n = n, u_dim = dim;
    void *args[] = { &d_a, &d_b, &d_result, &u_n, &u_dim };

    if (ptx_launch_1d(kernels->cosine_similarity, n, args, 128) != 0) return -1;
    if (ptx_synchronize() != 0) return -1;
    if (ptx_copy_to_host(h_result, d_result, n * sizeof(float)) != 0) return -1;

    // Verify
    int pass = 1;
    if (fabsf(h_result[0] - 1.0f) > 0.01f) { printf("FAIL: pair 0 sim=%f expected 1.0\n", h_result[0]); pass = 0; }
    if (fabsf(h_result[1]) > 0.01f)         { printf("FAIL: pair 1 sim=%f expected 0.0\n", h_result[1]); pass = 0; }
    if (fabsf(h_result[2] + 1.0f) > 0.01f)  { printf("FAIL: pair 2 sim=%f expected -1.0\n", h_result[2]); pass = 0; }
    printf("Pair 3: sim = %f\n", h_result[3]);

    ptx_free(d_a); ptx_free(d_b); ptx_free(d_result);

    printf("Test Cosine Similarity: %s\n", pass ? "PASSED" : "FAILED");
    return pass ? 0 : -1;
}
