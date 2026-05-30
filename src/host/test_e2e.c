//
// test_e2e.c — End-to-end test: tick → predict → balance → GC on GPU
//

#include "ptx_loader.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int test_e2e(ptx_kernels_t *kernels) {
    printf("=== Test: End-to-End (Tick → Predict → Balance → GC) ===\n");

    const int n_rooms = 4;
    const int dim = 8;
    const int n_entries = n_rooms;  // simplified: 1 entry per room

    // Step 1: Initialize embeddings and velocities
    float h_emb[n_rooms * dim], h_vel[n_rooms * dim];
    float h_predictions[n_rooms * dim], h_vibes[n_rooms * dim];
    float h_strengths[n_rooms];
    float h_ages[n_rooms];
    unsigned int h_z_in[n_rooms], h_z_out[n_rooms], h_balance_result[n_rooms];
    unsigned int h_imbalance = 0;
    unsigned int h_merge_candidates[n_rooms];
    unsigned int h_merge_count = 0;

    for (int i = 0; i < n_rooms; i++) {
        h_strengths[i] = 1.0f;
        h_ages[i] = 0.0f;
        h_z_in[i] = 5;
        h_z_out[i] = 5;
        for (int j = 0; j < dim; j++) {
            h_emb[i * dim + j] = sinf((float)(i * dim + j));
            h_vel[i * dim + j] = 0.1f * cosf((float)(i * dim + j));
            h_predictions[i * dim + j] = h_emb[i * dim + j] + 0.5f;
        }
    }

    // Make one room imbalanced
    h_z_out[2] = 3;  // room 2: in=5, out=3

    void *d_emb, *d_vel, *d_pred, *d_vibes, *d_strengths, *d_ages;
    void *d_z_in, *d_z_out, *d_balance_result, *d_imbalance;
    void *d_merge_cand, *d_merge_count;

    size_t embed_bytes = n_rooms * dim * sizeof(float);
    ptx_alloc_copy(&d_emb, h_emb, embed_bytes);
    ptx_alloc_copy(&d_vel, h_vel, embed_bytes);
    ptx_alloc_copy(&d_pred, h_predictions, embed_bytes);
    ptx_alloc(&d_vibes, embed_bytes);
    ptx_alloc_copy(&d_strengths, h_strengths, n_rooms * sizeof(float));
    ptx_alloc_copy(&d_ages, h_ages, n_rooms * sizeof(float));
    ptx_alloc_copy(&d_z_in, h_z_in, n_rooms * sizeof(unsigned int));
    ptx_alloc_copy(&d_z_out, h_z_out, n_rooms * sizeof(unsigned int));
    ptx_alloc(&d_balance_result, n_rooms * sizeof(unsigned int));
    ptx_alloc_copy(&d_imbalance, &h_imbalance, sizeof(unsigned int));
    ptx_alloc(&d_merge_cand, n_rooms * sizeof(unsigned int));
    ptx_alloc_copy(&d_merge_count, &h_merge_count, sizeof(unsigned int));

    // STEP 2: Decay strengths
    printf("  Step 1: Decay...\n");
    unsigned int u_n = n_rooms, u_dim = dim;
    float decay_rate = 0.1f;
    void *decay_args[] = { &d_strengths, &d_ages, &u_n, &decay_rate };
    ptx_launch_1d(kernels->decay, n_rooms, decay_args, 128);
    ptx_synchronize();

    // STEP 3: Batch predict
    printf("  Step 2: Batch Predict...\n");
    float delta = 0.5f;
    void *predict_args[] = { &d_emb, &d_pred, &d_pred, &u_n, &u_dim, &delta };
    ptx_launch_1d(kernels->batch_predict, n_rooms, predict_args, 128);
    ptx_synchronize();

    // STEP 4: Compute vibes
    printf("  Step 3: Vibe Compute...\n");
    float dt = 1.0f;
    void *vibe_args[] = { &d_emb, &d_vel, &d_vibes, &u_n, &u_dim, &dt };
    ptx_launch_1d(kernels->vibe_compute, n_rooms, vibe_args, 128);
    ptx_synchronize();

    // STEP 5: Balance check
    printf("  Step 4: Balance Check...\n");
    void *balance_args[] = { &d_z_in, &d_z_out, &d_balance_result, &d_imbalance, &u_n };
    ptx_launch_1d(kernels->balance_check, n_rooms, balance_args, 128);
    ptx_synchronize();

    // STEP 6: Merge candidates (GC)
    printf("  Step 5: Merge Candidates (GC)...\n");
    float merge_threshold = 0.9f;
    void *merge_args[] = { &d_emb, &d_strengths, &d_merge_cand, &d_merge_count, &u_n, &u_dim, &merge_threshold };
    ptx_launch_1d(kernels->merge_candidates, n_rooms, merge_args, 128);
    ptx_synchronize();

    // Read back results
    ptx_copy_to_host(h_strengths, d_strengths, n_rooms * sizeof(float));
    ptx_copy_to_host(h_vibes, d_vibes, embed_bytes);
    ptx_copy_to_host(h_balance_result, d_balance_result, n_rooms * sizeof(unsigned int));
    ptx_copy_to_host(&h_imbalance, d_imbalance, sizeof(unsigned int));
    ptx_copy_to_host(h_merge_candidates, d_merge_cand, n_rooms * sizeof(unsigned int));
    ptx_copy_to_host(&h_merge_count, d_merge_count, sizeof(unsigned int));

    // Verify
    int pass = 1;

    // Decay: all strengths should be 1.0 (ages were 0)
    for (int i = 0; i < n_rooms; i++) {
        if (fabsf(h_strengths[i] - 1.0f) > 0.01f) {
            printf("FAIL: strength[%d] = %f after decay\n", i, h_strengths[i]);
            pass = 0;
        }
    }

    // Balance: room 2 should be imbalanced
    if (h_imbalance != 1) {
        printf("FAIL: imbalance count = %u, expected 1\n", h_imbalance);
        pass = 0;
    }
    if (h_balance_result[2] != 1) {
        printf("FAIL: room 2 should be imbalanced\n");
        pass = 0;
    }

    // Vibes should be normalized
    for (int i = 0; i < n_rooms; i++) {
        float norm = 0;
        for (int j = 0; j < dim; j++) {
            norm += h_vibes[i*dim+j] * h_vibes[i*dim+j];
        }
        if (fabsf(norm - 1.0f) > 0.01f) {
            printf("FAIL: vibe[%d] norm = %f, expected 1.0\n", i, norm);
            pass = 0;
        }
    }

    // Cleanup
    ptx_free(d_emb); ptx_free(d_vel); ptx_free(d_pred); ptx_free(d_vibes);
    ptx_free(d_strengths); ptx_free(d_ages);
    ptx_free(d_z_in); ptx_free(d_z_out); ptx_free(d_balance_result); ptx_free(d_imbalance);
    ptx_free(d_merge_cand); ptx_free(d_merge_count);

    printf("Test End-to-End: %s\n", pass ? "PASSED" : "FAILED");
    return pass ? 0 : -1;
}
