//
// test_correlation.c — Test correlation matrix kernel
//

#include "ptx_loader.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

int test_correlation(ptx_kernels_t *kernels) {
    printf("=== Test: Correlation Matrix ===\n");

    const int n_rooms = 3;
    const int dim = 4;

    float h_vibes[n_rooms * dim];
    float h_matrix[n_rooms * n_rooms];

    // Room 0: [1, 0, 0, 0]
    h_vibes[0] = 1; h_vibes[1] = 0; h_vibes[2] = 0; h_vibes[3] = 0;
    // Room 1: [0, 1, 0, 0]
    h_vibes[4] = 0; h_vibes[5] = 1; h_vibes[6] = 0; h_vibes[7] = 0;
    // Room 2: [1, 0, 0, 0] (same as room 0)
    h_vibes[8] = 1; h_vibes[9] = 0; h_vibes[10] = 0; h_vibes[11] = 0;

    void *d_vibes, *d_matrix;

    if (ptx_alloc_copy(&d_vibes, h_vibes, n_rooms * dim * sizeof(float)) != 0) return -1;
    if (ptx_alloc(&d_matrix, n_rooms * n_rooms * sizeof(float)) != 0) return -1;

    unsigned int u_n = n_rooms, u_dim = dim;
    void *args[] = { &d_vibes, &d_matrix, &u_n, &u_dim };

    if (ptx_launch_2d(kernels->correlation_matrix, n_rooms, n_rooms, args, 128) != 0) return -1;
    if (ptx_synchronize() != 0) return -1;
    if (ptx_copy_to_host(h_matrix, d_matrix, n_rooms * n_rooms * sizeof(float)) != 0) return -1;

    int pass = 1;

    // Diagonal should be 1.0
    for (int i = 0; i < n_rooms; i++) {
        float diag = h_matrix[i * n_rooms + i];
        if (fabsf(diag - 1.0f) > 0.01f) {
            printf("FAIL: diag[%d] = %f, expected 1.0\n", i, diag);
            pass = 0;
        }
    }

    // Room 0 ↔ Room 1: orthogonal → sim = 0
    float sim01 = h_matrix[0 * n_rooms + 1];
    if (fabsf(sim01) > 0.01f) {
        printf("FAIL: sim(0,1) = %f, expected 0.0\n", sim01);
        pass = 0;
    }

    // Room 0 ↔ Room 2: identical → sim = 1.0
    float sim02 = h_matrix[0 * n_rooms + 2];
    if (fabsf(sim02 - 1.0f) > 0.01f) {
        printf("FAIL: sim(0,2) = %f, expected 1.0\n", sim02);
        pass = 0;
    }

    // Symmetry
    for (int i = 0; i < n_rooms; i++) {
        for (int j = 0; j < n_rooms; j++) {
            if (fabsf(h_matrix[i*n_rooms+j] - h_matrix[j*n_rooms+i]) > 0.01f) {
                printf("FAIL: matrix not symmetric at (%d,%d)\n", i, j);
                pass = 0;
            }
        }
    }

    ptx_free(d_vibes); ptx_free(d_matrix);

    printf("Test Correlation Matrix: %s\n", pass ? "PASSED" : "FAILED");
    return pass ? 0 : -1;
}
