/**
 * @file test_sv_fastpf_determinism.c
 * @brief Determinism test: Verify reproducibility with same seed.
 *
 * Critical for production: ensures that given the same RNG seed and observations,
 * the particle filter produces bitwise-identical results.
 *
 * This is essential for:
 * - Debugging (reproduce exact behavior)
 * - Testing (stable results)
 * - Production (checkpoint/restore must be exact)
 */

#include "../test_common.h"
#include "../../src/model_sv.h"
#include "../../third_party/fastpf/include/fastpf.h"
#include "../../third_party/fastpf/src/fastpf_internal.h"  /* For stack allocation */
#include <stdlib.h>
#include <math.h>
#include <string.h>

/* ========================================================================
 * Helper: Run PF for N steps and collect diagnostics
 * ======================================================================== */

typedef struct {
    double* ess_history;
    double* log_norm_const_history;
    double* max_weight_history;
    int* resampled_history;
    size_t n_steps;
} run_diagnostics_t;

/**
 * @brief Run particle filter for N steps and record diagnostics.
 */
static void run_pf_and_collect(fastpf_t* pf, const double* observations, 
                                size_t n_obs, run_diagnostics_t* out) {
    out->ess_history = (double*)malloc(n_obs * sizeof(double));
    out->log_norm_const_history = (double*)malloc(n_obs * sizeof(double));
    out->max_weight_history = (double*)malloc(n_obs * sizeof(double));
    out->resampled_history = (int*)malloc(n_obs * sizeof(int));
    out->n_steps = n_obs;
    
    size_t i;
    for (i = 0; i < n_obs; i++) {
        fastpf_step(pf, &observations[i]);
        
        const fastpf_diagnostics_t* diag = fastpf_get_diagnostics(pf);
        out->ess_history[i] = diag->ess;
        out->log_norm_const_history[i] = diag->log_norm_const;
        out->max_weight_history[i] = diag->max_weight;
        out->resampled_history[i] = diag->resampled;
    }
}

static void free_run_diagnostics(run_diagnostics_t* d) {
    free(d->ess_history);
    free(d->log_norm_const_history);
    free(d->max_weight_history);
    free(d->resampled_history);
}

/* ========================================================================
 * Test: Same Seed = Identical Results
 * ======================================================================== */

TEST(determinism_same_seed) {
    /* Prepare observations */
    const size_t n_obs = 100;
    double* observations = (double*)malloc(n_obs * sizeof(double));
    size_t i;
    for (i = 0; i < n_obs; i++) {
        observations[i] = 0.0001 * ((i % 10) - 5);
    }
    
    /* Run 1: Create and run PF */
    fastpf_model_t model1;
    sv_model_ctx_t* ctx1 = sv_model_create(-13.8, 0.98, 0.15, 7.0, &model1);
    ASSERT_TRUE(ctx1 != NULL);
    
    fastpf_cfg_t cfg1;
    fastpf_cfg_init(&cfg1, 2000, sizeof(double));
    cfg1.rng_seed = 42;
    cfg1.num_threads = 1;  /* Single-threaded for determinism */
    
    fastpf_t pf1;
    ASSERT_EQ(fastpf_init(&pf1, &cfg1, &model1), FASTPF_SUCCESS);
    
    run_diagnostics_t run1;
    run_pf_and_collect(&pf1, observations, n_obs, &run1);
    
    /* Run 2: Same setup, same seed */
    fastpf_model_t model2;
    sv_model_ctx_t* ctx2 = sv_model_create(-13.8, 0.98, 0.15, 7.0, &model2);
    ASSERT_TRUE(ctx2 != NULL);
    
    fastpf_cfg_t cfg2;
    fastpf_cfg_init(&cfg2, 2000, sizeof(double));
    cfg2.rng_seed = 42;  /* Same seed */
    cfg2.num_threads = 1;
    
    fastpf_t pf2;
    ASSERT_EQ(fastpf_init(&pf2, &cfg2, &model2), FASTPF_SUCCESS);
    
    run_diagnostics_t run2;
    run_pf_and_collect(&pf2, observations, n_obs, &run2);
    
    /* Compare results: should be bitwise identical */
    for (i = 0; i < n_obs; i++) {
        /* ESS should match exactly */
        ASSERT_NEAR(run1.ess_history[i], run2.ess_history[i], 1e-15);
        
        /* Log normalizing constant should match exactly */
        ASSERT_NEAR(run1.log_norm_const_history[i], run2.log_norm_const_history[i], 1e-15);
        
        /* Max weight should match exactly */
        ASSERT_NEAR(run1.max_weight_history[i], run2.max_weight_history[i], 1e-15);
        
        /* Resampling decisions should match */
        ASSERT_EQ(run1.resampled_history[i], run2.resampled_history[i]);
    }
    
    /* Cleanup */
    free_run_diagnostics(&run1);
    free_run_diagnostics(&run2);
    fastpf_free(&pf1);
    fastpf_free(&pf2);
    free(ctx1);
    free(ctx2);
    free(observations);
}

/* ========================================================================
 * Test: Different Seeds = Different Results
 * ======================================================================== */

TEST(determinism_different_seeds) {
    /* Sanity check: different seeds should give different results */
    const size_t n_obs = 50;
    double* observations = (double*)malloc(n_obs * sizeof(double));
    size_t i;
    for (i = 0; i < n_obs; i++) {
        observations[i] = 0.0;
    }
    
    /* Run with seed=42 */
    fastpf_model_t model1;
    sv_model_ctx_t* ctx1 = sv_model_create(-13.8, 0.98, 0.15, 7.0, &model1);
    
    fastpf_cfg_t cfg1;
    fastpf_cfg_init(&cfg1, 2000, sizeof(double));
    cfg1.rng_seed = 42;
    cfg1.num_threads = 1;
    
    fastpf_t pf1;
    fastpf_init(&pf1, &cfg1, &model1);
    
    run_diagnostics_t run1;
    run_pf_and_collect(&pf1, observations, n_obs, &run1);
    
    /* Run with seed=123 */
    fastpf_model_t model2;
    sv_model_ctx_t* ctx2 = sv_model_create(-13.8, 0.98, 0.15, 7.0, &model2);
    
    fastpf_cfg_t cfg2;
    fastpf_cfg_init(&cfg2, 2000, sizeof(double));
    cfg2.rng_seed = 123;  /* Different seed */
    cfg2.num_threads = 1;
    
    fastpf_t pf2;
    fastpf_init(&pf2, &cfg2, &model2);
    
    run_diagnostics_t run2;
    run_pf_and_collect(&pf2, observations, n_obs, &run2);
    
    /* Results should differ (with high probability) */
    int differences = 0;
    for (i = 0; i < n_obs; i++) {
        if (fabs(run1.ess_history[i] - run2.ess_history[i]) > 1e-10) {
            differences++;
        }
    }
    
    /* Should see differences in most steps */
    ASSERT_TRUE(differences > 10);
    
    /* Cleanup */
    free_run_diagnostics(&run1);
    free_run_diagnostics(&run2);
    fastpf_free(&pf1);
    fastpf_free(&pf2);
    free(ctx1);
    free(ctx2);
    free(observations);
}

/* ========================================================================
 * Test: Resampling Determinism
 * ======================================================================== */

TEST(determinism_resampling_steps) {
    /* Verify that resampling decisions are deterministic */
    const size_t n_obs = 100;
    double* observations = (double*)malloc(n_obs * sizeof(double));
    
    /* Create observations that will likely trigger resampling */
    size_t i;
    for (i = 0; i < n_obs; i++) {
        /* Alternate between small and large returns to vary weights */
        observations[i] = (i % 2 == 0) ? 0.0001 : 0.001;
    }
    
    /* Run 1 */
    fastpf_model_t model1;
    sv_model_ctx_t* ctx1 = sv_model_create(-13.8, 0.98, 0.15, 7.0, &model1);
    
    fastpf_cfg_t cfg1;
    fastpf_cfg_init(&cfg1, 3000, sizeof(double));
    cfg1.rng_seed = 999;
    cfg1.resample_threshold = 0.7;
    cfg1.num_threads = 1;
    
    fastpf_t pf1;
    fastpf_init(&pf1, &cfg1, &model1);
    
    run_diagnostics_t run1;
    run_pf_and_collect(&pf1, observations, n_obs, &run1);
    
    /* Run 2: Identical setup */
    fastpf_model_t model2;
    sv_model_ctx_t* ctx2 = sv_model_create(-13.8, 0.98, 0.15, 7.0, &model2);
    
    fastpf_cfg_t cfg2;
    fastpf_cfg_init(&cfg2, 3000, sizeof(double));
    cfg2.rng_seed = 999;
    cfg2.resample_threshold = 0.7;
    cfg2.num_threads = 1;
    
    fastpf_t pf2;
    fastpf_init(&pf2, &cfg2, &model2);
    
    run_diagnostics_t run2;
    run_pf_and_collect(&pf2, observations, n_obs, &run2);
    
    /* Verify resampling decisions match exactly */
    int resample_count = 0;
    for (i = 0; i < n_obs; i++) {
        ASSERT_EQ(run1.resampled_history[i], run2.resampled_history[i]);
        if (run1.resampled_history[i]) {
            resample_count++;
        }
    }
    
    /* Should have triggered some resampling */
    ASSERT_TRUE(resample_count > 0);
    
    /* Cleanup */
    free_run_diagnostics(&run1);
    free_run_diagnostics(&run2);
    fastpf_free(&pf1);
    fastpf_free(&pf2);
    free(ctx1);
    free(ctx2);
    free(observations);
}

/* ========================================================================
 * Test: Weight Determinism
 * ======================================================================== */

TEST(determinism_particle_weights) {
    /* Verify that particle weights are deterministic */
    const size_t n_obs = 20;
    double* observations = (double*)malloc(n_obs * sizeof(double));
    size_t i;
    for (i = 0; i < n_obs; i++) {
        observations[i] = 0.0005;
    }
    
    /* Run 1 */
    fastpf_model_t model1;
    sv_model_ctx_t* ctx1 = sv_model_create(-13.8, 0.98, 0.15, 7.0, &model1);
    
    fastpf_cfg_t cfg1;
    fastpf_cfg_init(&cfg1, 1000, sizeof(double));
    cfg1.rng_seed = 555;
    cfg1.num_threads = 1;
    
    fastpf_t pf1;
    fastpf_init(&pf1, &cfg1, &model1);
    
    /* Run filter */
    for (i = 0; i < n_obs; i++) {
        fastpf_step(&pf1, &observations[i]);
    }
    
    const double* weights1 = fastpf_get_weights(&pf1);
    double* weights1_copy = (double*)malloc(1000 * sizeof(double));
    memcpy(weights1_copy, weights1, 1000 * sizeof(double));
    
    /* Run 2: Identical setup */
    fastpf_model_t model2;
    sv_model_ctx_t* ctx2 = sv_model_create(-13.8, 0.98, 0.15, 7.0, &model2);
    
    fastpf_cfg_t cfg2;
    fastpf_cfg_init(&cfg2, 1000, sizeof(double));
    cfg2.rng_seed = 555;
    cfg2.num_threads = 1;
    
    fastpf_t pf2;
    fastpf_init(&pf2, &cfg2, &model2);
    
    for (i = 0; i < n_obs; i++) {
        fastpf_step(&pf2, &observations[i]);
    }
    
    const double* weights2 = fastpf_get_weights(&pf2);
    
    /* Compare all weights */
    for (i = 0; i < 1000; i++) {
        ASSERT_NEAR(weights1_copy[i], weights2[i], 1e-15);
    }
    
    /* Cleanup */
    free(weights1_copy);
    fastpf_free(&pf1);
    fastpf_free(&pf2);
    free(ctx1);
    free(ctx2);
    free(observations);
}

/* ========================================================================
 * Test Runner
 * ======================================================================== */

int main(void) {
    printf("========================================\n");
    printf("SV + fastpf Determinism Tests\n");
    printf("========================================\n\n");
    
    RUN_TEST(determinism_same_seed);
    RUN_TEST(determinism_different_seeds);
    RUN_TEST(determinism_resampling_steps);
    RUN_TEST(determinism_particle_weights);
    
    return test_summary();
}
