//
// test_batch_predict.c — Test batch prediction kernel
//

#include "ptx_loader.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

int test_batch_predict(ptx_kernels_t *kernels) {
    printf("=== Test: Batch Predict ===\n");

    const int n = 3;
    const int dim = 4;
    float delta = 0.5f;

    float h_p[n * dim], h_pr[n * dim], h_r[n * dim];

    for (int i = 0; i < n * dim; i++) {
        h_p[i] = 1.0f;
        h_pr[i] = 3.0f;
    }

    void *d_p, *d_pr, *d_r;

    if (ptx_alloc_copy(&d_p, h_p, n * dim * sizeof(float)) != 0) return -1;
    if (ptx_alloc_copy(&d_pr, h_pr, n * dim * sizeof(float)) != 0) return -1;
    if (ptx_alloc(&d_r, n * dim * sizeof(float)) != 0) return -1;

    unsigned int u_n = n, u_dim = dim;
    void *args[] = { &d_p, &d_pr, &d_r, &u_n, &u_dim, &delta };

    if (ptx_launch_1d(kernels->batch_predict, n, args, 128) != 0) return -1;
    if (ptx_synchronize() != 0) return -1;
    if (ptx_copy_to_host(h_r, d_r, n * dim * sizeof(float)) != 0) return -1;

    // result = perception + delta * (prediction - perception) = 1 + 0.5 * (3 - 1) = 2.0
    int pass = 1;
    for (int i = 0; i < n * dim; i++) {
        if (fabsf(h_r[i] - 2.0f) > 0.01f) {
            printf("FAIL: result[%d] = %f, expected 2.0\n", i, h_r[i]);
            pass = 0;
        }
    }

    ptx_free(d_p); ptx_free(d_pr); ptx_free(d_r);

    printf("Test Batch Predict: %s\n", pass ? "PASSED" : "FAILED");
    return pass ? 0 : -1;
}
