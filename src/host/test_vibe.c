//
// test_vibe.c — Test vibe computation kernel
//

#include "ptx_loader.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

int test_vibe(ptx_kernels_t *kernels) {
    printf("=== Test: Vibe Compute ===\n");

    const int n = 2;
    const int dim = 4;
    float dt = 1.0f;

    float h_emb[n * dim], h_vel[n * dim], h_vibes[n * dim];

    // Room 0: embedding = [1,0,0,0], velocity = [0,1,0,0]
    h_emb[0] = 1; h_emb[1] = 0; h_emb[2] = 0; h_emb[3] = 0;
    h_vel[0] = 0; h_vel[1] = 1; h_vel[2] = 0; h_vel[3] = 0;

    // Room 1: embedding = [3,4,0,0], velocity = [0,0,1,0]
    h_emb[4] = 3; h_emb[5] = 4; h_emb[6] = 0; h_emb[7] = 0;
    h_vel[4] = 0; h_vel[5] = 0; h_vel[6] = 1; h_vel[7] = 0;

    void *d_emb, *d_vel, *d_vibes;

    if (ptx_alloc_copy(&d_emb, h_emb, n * dim * sizeof(float)) != 0) return -1;
    if (ptx_alloc_copy(&d_vel, h_vel, n * dim * sizeof(float)) != 0) return -1;
    if (ptx_alloc(&d_vibes, n * dim * sizeof(float)) != 0) return -1;

    unsigned int u_n = n, u_dim = dim;
    void *args[] = { &d_emb, &d_vel, &d_vibes, &u_n, &u_dim, &dt };

    if (ptx_launch_1d(kernels->vibe_compute, n, args, 128) != 0) return -1;
    if (ptx_synchronize() != 0) return -1;
    if (ptx_copy_to_host(h_vibes, d_vibes, n * dim * sizeof(float)) != 0) return -1;

    // Room 0: vibe = [1,1,0,0] → norm = sqrt(2) → normalized = [1/√2, 1/√2, 0, 0]
    // Room 1: vibe = [3,4,1,0] → norm = sqrt(26) → normalized
    int pass = 1;

    // Check Room 0 norm = 1
    float norm0 = 0;
    for (int j = 0; j < dim; j++) norm0 += h_vibes[j] * h_vibes[j];
    if (fabsf(norm0 - 1.0f) > 0.01f) {
        printf("FAIL: Room 0 norm = %f, expected 1.0\n", norm0);
        pass = 0;
    }

    // Check Room 1 norm = 1
    float norm1 = 0;
    for (int j = 0; j < dim; j++) norm1 += h_vibes[dim + j] * h_vibes[dim + j];
    if (fabsf(norm1 - 1.0f) > 0.01f) {
        printf("FAIL: Room 1 norm = %f, expected 1.0\n", norm1);
        pass = 0;
    }

    ptx_free(d_emb); ptx_free(d_vel); ptx_free(d_vibes);

    printf("Test Vibe Compute: %s\n", pass ? "PASSED" : "FAILED");
    return pass ? 0 : -1;
}
