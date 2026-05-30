//
// test_balance.c — Test balance check kernel
//

#include "ptx_loader.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int test_balance(ptx_kernels_t *kernels) {
    printf("=== Test: Balance Check ===\n");

    const int n = 6;
    unsigned int h_in[n], h_out[n], h_result[n];
    unsigned int h_imbalance = 0;

    // Rooms 0-2: balanced (in == out)
    h_in[0] = 5;  h_out[0] = 5;
    h_in[1] = 10; h_out[1] = 10;
    h_in[2] = 3;  h_out[2] = 3;

    // Rooms 3-5: imbalanced
    h_in[3] = 5;  h_out[3] = 3;
    h_in[4] = 0;  h_out[4] = 1;
    h_in[5] = 7;  h_out[5] = 7;  // balanced

    void *d_in, *d_out, *d_result, *d_imbalance;

    if (ptx_alloc_copy(&d_in, h_in, n * sizeof(unsigned int)) != 0) return -1;
    if (ptx_alloc_copy(&d_out, h_out, n * sizeof(unsigned int)) != 0) return -1;
    if (ptx_alloc(&d_result, n * sizeof(unsigned int)) != 0) return -1;
    if (ptx_alloc_copy(&d_imbalance, &h_imbalance, sizeof(unsigned int)) != 0) return -1;

    unsigned int u_n = n;
    void *args[] = { &d_in, &d_out, &d_result, &d_imbalance, &u_n };

    if (ptx_launch_1d(kernels->balance_check, n, args, 128) != 0) return -1;
    if (ptx_synchronize() != 0) return -1;
    if (ptx_copy_to_host(h_result, d_result, n * sizeof(unsigned int)) != 0) return -1;
    if (ptx_copy_to_host(&h_imbalance, d_imbalance, sizeof(unsigned int)) != 0) return -1;

    int pass = 1;
    if (h_result[0] != 0 || h_result[1] != 0 || h_result[2] != 0) {
        printf("FAIL: balanced rooms should be 0\n"); pass = 0;
    }
    if (h_result[3] != 1 || h_result[4] != 1) {
        printf("FAIL: imbalanced rooms should be 1\n"); pass = 0;
    }
    if (h_result[5] != 0) {
        printf("FAIL: room 5 should be balanced\n"); pass = 0;
    }
    if (h_imbalance != 2) {
        printf("FAIL: imbalance count = %u, expected 2\n", h_imbalance); pass = 0;
    }

    ptx_free(d_in); ptx_free(d_out); ptx_free(d_result); ptx_free(d_imbalance);

    printf("Test Balance Check: %s\n", pass ? "PASSED" : "FAILED");
    return pass ? 0 : -1;
}
