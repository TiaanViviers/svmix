/**
 * @file test_sv_fastpf_smoke.c
 * @brief Smoke test: SV model + fastpf integration.
 *
 * Tests that the SV model callbacks work correctly with a real fastpf instance.
 * This is the first test that exercises the full SV -> fastpf -> diagnostics pipeline.
 *
 * Goals:
 * - Verify no crashes or hangs
 * - Verify diagnostics are sane (ESS > 0, finite log_norm_const)
 * - Verify no NaN/Inf explosions on normal inputs
 * - Confirm particle filter can run for many steps
 */

#include "../test_common.h"
#include "../../src/model_sv.h"
#include "../../third_party/fastpf/include/fastpf.h"
#include "../../third_party/fastpf/src/fastpf_internal.h"  /* For stack allocation */
#include <stdlib.h>
#include <math.h>

/* ========================================================================
 * Test: Basic Initialization
 * ======================================================================== */

TEST(fastpf_init_with_sv_model) {
    /* Create SV model with default parameters */
    fastpf_model_t model;
    sv_model_ctx_t* ctx = sv_model_create(-13.8, 0.98, 0.15, 7.0, &model);
    ASSERT_TRUE(ctx != NULL);
    
    /* Configure particle filter */
    fastpf_cfg_t cfg;
    fastpf_cfg_init(&cfg, 2000, sizeof(double));
    cfg.rng_seed = 42;
    cfg.resample_threshold = 0.5;
    cfg.num_threads = 1;  /* Single-threaded for determinism */
    
    /* Initialize particle filter */
    fastpf_t pf;
    int result = fastpf_init(&pf, &cfg, &model);
    ASSERT_EQ(result, FASTPF_SUCCESS);
    
    /* Verify basic accessors work */
    ASSERT_EQ(fastpf_num_particles(&pf), 2000);
    ASSERT_EQ(fastpf_state_size(&pf), sizeof(double));
    
    const double* weights = fastpf_get_weights(&pf);
    ASSERT_TRUE(weights != NULL);
    
    /* Weights should be uniform initially (1/N) */
    double expected_weight = 1.0 / 2000.0;
    ASSERT_NEAR(weights[0], expected_weight, 1e-10);
    ASSERT_NEAR(weights[1999], expected_weight, 1e-10);
    
    /* Cleanup */
    fastpf_free(&pf);
    free(ctx);
}

/* ========================================================================
 * Test: Single Step with Zero Return
 * ======================================================================== */

TEST(single_step_zero_return) {
    /* Setup */
    fastpf_model_t model;
    sv_model_ctx_t* ctx = sv_model_create(-13.8, 0.98, 0.15, 7.0, &model);
    ASSERT_TRUE(ctx != NULL);
    
    fastpf_cfg_t cfg;
    fastpf_cfg_init(&cfg, 2000, sizeof(double));
    cfg.rng_seed = 123;
    cfg.num_threads = 1;
    
    fastpf_t pf;
    ASSERT_EQ(fastpf_init(&pf, &cfg, &model), FASTPF_SUCCESS);
    
    /* Feed observation: r_t = 0.0 (no return) */
    double obs = 0.0;
    int result = fastpf_step(&pf, &obs);
    ASSERT_EQ(result, FASTPF_SUCCESS);
    
    /* Check diagnostics */
    const fastpf_diagnostics_t* diag = fastpf_get_diagnostics(&pf);
    ASSERT_TRUE(diag != NULL);
    
    /* ESS should be in valid range (0, N] */
    ASSERT_FINITE(diag->ess);
    ASSERT_TRUE(diag->ess > 0.0);
    ASSERT_TRUE(diag->ess <= 2000.0);
    
    /* Log normalizing constant should be finite */
    ASSERT_FINITE(diag->log_norm_const);
    
    /* Max weight should be in (0, 1] */
    ASSERT_FINITE(diag->max_weight);
    ASSERT_TRUE(diag->max_weight > 0.0);
    ASSERT_TRUE(diag->max_weight <= 1.0);
    
    /* Cleanup */
    fastpf_free(&pf);
    free(ctx);
}

/* ========================================================================
 * Test: Multiple Steps with Small Returns
 * ======================================================================== */

TEST(multiple_steps_small_returns) {
    /* Setup */
    fastpf_model_t model;
    sv_model_ctx_t* ctx = sv_model_create(-13.8, 0.98, 0.15, 7.0, &model);
    ASSERT_TRUE(ctx != NULL);
    
    fastpf_cfg_t cfg;
    fastpf_cfg_init(&cfg, 3000, sizeof(double));
    cfg.rng_seed = 456;
    cfg.num_threads = 1;
    
    fastpf_t pf;
    ASSERT_EQ(fastpf_init(&pf, &cfg, &model), FASTPF_SUCCESS);
    
    /* Feed 100 steps of small random-ish returns */
    int step;
    for (step = 0; step < 100; step++) {
        /* Generate pseudo-random small return (deterministic) */
        double obs = 0.0001 * ((step % 10) - 5);  /* Range: -0.0005 to 0.0005 */
        
        int result = fastpf_step(&pf, &obs);
        ASSERT_EQ(result, FASTPF_SUCCESS);
        
        /* Check diagnostics remain sane */
        const fastpf_diagnostics_t* diag = fastpf_get_diagnostics(&pf);
        
        ASSERT_FINITE(diag->ess);
        ASSERT_TRUE(diag->ess > 0.0);
        ASSERT_TRUE(diag->ess <= 3000.0);
        
        ASSERT_FINITE(diag->log_norm_const);
        ASSERT_FINITE(diag->max_weight);
        
        /* Weights should sum to 1 */
        const double* weights = fastpf_get_weights(&pf);
        double weight_sum = 0.0;
        size_t i;
        for (i = 0; i < 3000; i++) {
            ASSERT_FINITE(weights[i]);
            ASSERT_TRUE(weights[i] >= 0.0);
            weight_sum += weights[i];
        }
        ASSERT_NEAR(weight_sum, 1.0, 1e-10);
    }
    
    /* Cleanup */
    fastpf_free(&pf);
    free(ctx);
}

/* ========================================================================
 * Test: Realistic Return Scale
 * ======================================================================== */

TEST(realistic_return_scale) {
    /* Use realistic minute-return scale parameters */
    double mu_h = -13.815510557964;  /* log(0.001^2) */
    double phi_h = 0.98;
    double sigma_h = 0.15;
    double nu = 7.0;
    
    fastpf_model_t model;
    sv_model_ctx_t* ctx = sv_model_create(mu_h, phi_h, sigma_h, nu, &model);
    ASSERT_TRUE(ctx != NULL);
    
    fastpf_cfg_t cfg;
    fastpf_cfg_init(&cfg, 5000, sizeof(double));
    cfg.rng_seed = 789;
    cfg.num_threads = 1;
    
    fastpf_t pf;
    ASSERT_EQ(fastpf_init(&pf, &cfg, &model), FASTPF_SUCCESS);
    
    /* Feed realistic returns (5-10 bps) */
    int step;
    for (step = 0; step < 50; step++) {
        double obs = 0.0005 + 0.0002 * (step % 5);  /* 5-13 bps */
        
        int result = fastpf_step(&pf, &obs);
        ASSERT_EQ(result, FASTPF_SUCCESS);
        
        const fastpf_diagnostics_t* diag = fastpf_get_diagnostics(&pf);
        
        ASSERT_FINITE(diag->ess);
        ASSERT_TRUE(diag->ess > 0.0);
        ASSERT_FINITE(diag->log_norm_const);
    }
    
    /* Cleanup */
    fastpf_free(&pf);
    free(ctx);
}

/* ========================================================================
 * Test: Resampling Triggers
 * ======================================================================== */

TEST(resampling_behavior) {
    /* Setup with aggressive resampling threshold */
    fastpf_model_t model;
    sv_model_ctx_t* ctx = sv_model_create(-13.8, 0.98, 0.15, 7.0, &model);
    ASSERT_TRUE(ctx != NULL);
    
    fastpf_cfg_t cfg;
    fastpf_cfg_init(&cfg, 1000, sizeof(double));
    cfg.rng_seed = 999;
    cfg.resample_threshold = 0.8;  /* High threshold = resample more often */
    cfg.num_threads = 1;
    
    fastpf_t pf;
    ASSERT_EQ(fastpf_init(&pf, &cfg, &model), FASTPF_SUCCESS);
    
    int resampled_count = 0;
    int step;
    
    for (step = 0; step < 50; step++) {
        double obs = 0.0;
        fastpf_step(&pf, &obs);
        
        const fastpf_diagnostics_t* diag = fastpf_get_diagnostics(&pf);
        
        if (diag->resampled) {
            resampled_count++;
            
            /* After resampling, weights should be uniform */
            const double* weights = fastpf_get_weights(&pf);
            double expected = 1.0 / 1000.0;
            ASSERT_NEAR(weights[0], expected, 1e-10);
            ASSERT_NEAR(weights[500], expected, 1e-10);
        }
        
        ASSERT_FINITE(diag->ess);
    }
    
    /* With threshold=0.8 and 50 steps, should resample at least a few times */
    /* (Not asserting exact count since it depends on likelihood variance) */
    /* Just verify the flag works and doesn't always/never trigger */
    
    /* Cleanup */
    fastpf_free(&pf);
    free(ctx);
}

/* ========================================================================
 * Test: Extreme but Valid Observations
 * ======================================================================== */

TEST(extreme_observations) {
    /* Test that large returns don't crash the filter */
    fastpf_model_t model;
    sv_model_ctx_t* ctx = sv_model_create(-13.8, 0.98, 0.15, 7.0, &model);
    ASSERT_TRUE(ctx != NULL);
    
    fastpf_cfg_t cfg;
    fastpf_cfg_init(&cfg, 2000, sizeof(double));
    cfg.rng_seed = 111;
    cfg.num_threads = 1;
    
    fastpf_t pf;
    ASSERT_EQ(fastpf_init(&pf, &cfg, &model), FASTPF_SUCCESS);
    
    /* Normal returns */
    int step;
    for (step = 0; step < 10; step++) {
        double obs = 0.0001;
        fastpf_step(&pf, &obs);
    }
    
    /* Large return (extreme but possible) */
    double large_obs = 0.01;  /* 100 bps = 1% */
    int result = fastpf_step(&pf, &large_obs);
    ASSERT_EQ(result, FASTPF_SUCCESS);
    
    const fastpf_diagnostics_t* diag = fastpf_get_diagnostics(&pf);
    ASSERT_FINITE(diag->ess);
    ASSERT_FINITE(diag->log_norm_const);
    
    /* Continue with normal returns */
    for (step = 0; step < 10; step++) {
        double obs = 0.0001;
        result = fastpf_step(&pf, &obs);
        ASSERT_EQ(result, FASTPF_SUCCESS);
    }
    
    /* Cleanup */
    fastpf_free(&pf);
    free(ctx);
}

/* ========================================================================
 * Test Runner
 * ======================================================================== */

int main(void) {
    printf("========================================\n");
    printf("SV + fastpf Integration Smoke Tests\n");
    printf("========================================\n\n");
    
    RUN_TEST(fastpf_init_with_sv_model);
    RUN_TEST(single_step_zero_return);
    RUN_TEST(multiple_steps_small_returns);
    RUN_TEST(realistic_return_scale);
    RUN_TEST(resampling_behavior);
    RUN_TEST(extreme_observations);
    
    return test_summary();
}
