//
// test_merge.c — Test merge candidates kernel
//

#include "ptx_loader.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

int test_merge(ptx_kernels_t *kernels) {
    printf("=== Test: Merge Candidates ===\n");

    const int n = 4;
    const int dim = 4;
    float threshold = 0.9f;

    float h_emb[n * dim], h_strengths[n];
    unsigned int h_candidates[n];
    unsigned int h_count = 0;

    // Embedding 0: [1, 0, 0, 0]
    h_emb[0] = 1; h_emb[1] = 0; h_emb[2] = 0; h_emb[3] = 0;
    // Embedding 1: [1, 0.01, 0, 0] (very similar to 0)
    h_emb[4] = 1; h_emb[5] = 0.01f; h_emb[6] = 0; h_emb[7] = 0;
    // Embedding 2: [0, 1, 0, 0] (orthogonal to 1)
    h_emb[8] = 0; h_emb[9] = 1; h_emb[10] = 0; h_emb[11] = 0;
    // Embedding 3: [0, 1, 0.01, 0] (very similar to 2)
    h_emb[12] = 0; h_emb[13] = 1; h_emb[14] = 0.01f; h_emb[15] = 0;

    for (int i = 0; i < n; i++) h_strengths[i] = 1.0f;

    void *d_emb, *d_strengths, *d_candidates, *d_count;

    if (ptx_alloc_copy(&d_emb, h_emb, n * dim * sizeof(float)) != 0) return -1;
    if (ptx_alloc_copy(&d_strengths, h_strengths, n * sizeof(float)) != 0) return -1;
    if (ptx_alloc(&d_candidates, n * sizeof(unsigned int)) != 0) return -1;
    if (ptx_alloc_copy(&d_count, &h_count, sizeof(unsigned int)) != 0) return -1;

    unsigned int u_n = n, u_dim = dim;
    void *args[] = { &d_emb, &d_strengths, &d_candidates, &d_count, &u_n, &u_dim, &threshold };

    if (ptx_launch_1d(kernels->merge_candidates, n, args, 128) != 0) return -1;
    if (ptx_synchronize() != 0) return -1;
    if (ptx_copy_to_host(h_candidates, d_candidates, n * sizeof(unsigned int)) != 0) return -1;
    if (ptx_copy_to_host(&h_count, d_count, sizeof(unsigned int)) != 0) return -1;

    int pass = 1;
    printf("Candidates: ");
    for (int i = 0; i < n - 1; i++) printf("%u ", h_candidates[i]);
    printf("(count=%u)\n", h_count);

    // Pairs 0↔1 and 2↔3 should be similar (candidates)
    // Pair 1↔2 should NOT be similar
    if (h_candidates[0] != 1) { printf("FAIL: pair 0-1 should merge\n"); pass = 0; }
    if (h_candidates[1] != 0) { printf("FAIL: pair 1-2 should not merge\n"); pass = 0; }
    if (h_candidates[2] != 1) { printf("FAIL: pair 2-3 should merge\n"); pass = 0; }

    ptx_free(d_emb); ptx_free(d_strengths); ptx_free(d_candidates); ptx_free(d_count);

    printf("Test Merge Candidates: %s\n", pass ? "PASSED" : "FAILED");
    return pass ? 0 : -1;
}
