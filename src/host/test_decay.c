//
// test_decay.c — Test exponential decay kernel
//

#include "ptx_loader.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

int test_decay(ptx_kernels_t *kernels) {
    printf("=== Test: Decay ===\n");

    const int n = 8;
    float h_strengths[n];
    float h_ages[n];

    for (int i = 0; i < n; i++) {
        h_strengths[i] = 1.0f;
        h_ages[i] = (float)i;
    }

    void *d_strengths, *d_ages;

    if (ptx_alloc_copy(&d_strengths, h_strengths, n * sizeof(float)) != 0) return -1;
    if (ptx_alloc_copy(&d_ages, h_ages, n * sizeof(float)) != 0) return -1;

    unsigned int u_n = n;
    float rate = 0.1f;
    void *args[] = { &d_strengths, &d_ages, &u_n, &rate };

    if (ptx_launch_1d(kernels->decay, n, args, 128) != 0) return -1;
    if (ptx_synchronize() != 0) return -1;
    if (ptx_copy_to_host(h_strengths, d_strengths, n * sizeof(float)) != 0) return -1;

    int pass = 1;
    for (int i = 0; i < n; i++) {
        float expected = expf(-0.1f * (float)i);
        if (fabsf(h_strengths[i] - expected) > 0.05f) {
            printf("FAIL: strengths[%d] = %f, expected %f\n", i, h_strengths[i], expected);
            pass = 0;
        }
    }

    // Verify ordering: strengths should decrease with age
    for (int i = 1; i < n; i++) {
        if (h_strengths[i] >= h_strengths[i-1]) {
            printf("FAIL: decay not monotonic at %d\n", i);
            pass = 0;
        }
    }

    ptx_free(d_strengths); ptx_free(d_ages);

    printf("Test Decay: %s\n", pass ? "PASSED" : "FAILED");
    return pass ? 0 : -1;
}
