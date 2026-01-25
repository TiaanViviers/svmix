/**
 * @file test_ensemble_openmp.c
 * @brief OpenMP-specific tests for ensemble parallelization.
 *
 * These tests verify that OpenMP thread control works correctly.
 * Only compiled/run when OPENMP=1 is set.
 */

#include "../test_common.h"
#include "../../src/ensemble.h"
#include <stdlib.h>
#include <stdio.h>

#ifdef _OPENMP
#include <omp.h>
#endif

/* ========================================================================
 * Test helpers
 * ======================================================================== */

#ifdef _OPENMP
static sv_params_t default_sv_params(void) {
    sv_params_t params;
    params.mu_h = -13.815510557964;
    params.phi_h = 0.98;
    params.sigma_h = 0.15;
    params.nu = 7.0;
    return params;
}

static ensemble_config_t default_ensemble_config(void) {
    ensemble_config_t config;
    config.lambda = 0.99;
    config.beta = 1.0;
    config.epsilon = 0.05;
    config.num_threads = 0;  /* Auto */
    return config;
}
#endif

/* ========================================================================
 * Test: OpenMP is actually enabled
 * ======================================================================== */

TEST(openmp_enabled) {
#ifdef _OPENMP
    ASSERT_TRUE(1);  /* Test passes if _OPENMP is defined */
    printf("    OpenMP version: %d\n", _OPENMP);
    printf("    Max threads available: %d\n", omp_get_max_threads());
#else
    printf("    INFO: OpenMP not enabled (compile with OPENMP=1 to enable)\n");
    printf("    Skipping OpenMP tests - this is expected behavior\n");
    ASSERT_TRUE(1);  /* Pass, not fail - OpenMP is optional */
#endif
}

/* ========================================================================
 * Test: Thread count configuration
 * ======================================================================== */

TEST(thread_count_explicit) {
#ifdef _OPENMP
    const size_t K = 4;
    sv_params_t params[4];
    unsigned long seeds[4];
    for (size_t i = 0; i < K; i++) {
        params[i] = default_sv_params();
        seeds[i] = 20000 + i;
    }
    
    /* Request 2 threads explicitly */
    ensemble_config_t config = default_ensemble_config();
    config.num_threads = 2;
    
    ensemble_t* ens = ensemble_create(K, params, 500, seeds, &config);
    ASSERT_TRUE(ens != NULL);
    
    /* Verify thread count was set */
    int actual_threads = omp_get_max_threads();
    printf("    Requested: 2 threads, Actual: %d threads\n", actual_threads);
    ASSERT_EQ(actual_threads, 2);
    
    /* Run some steps to verify it works */
    for (int t = 0; t < 10; t++) {
        ASSERT_TRUE(ensemble_step(ens, 0.001) == 0);
    }
    
    ensemble_free(ens);
#else
    printf("    SKIP: OpenMP not enabled\n");
    ASSERT_TRUE(1);
#endif
}

/* ========================================================================
 * Test: Thread count auto (0 = use default)
 * ======================================================================== */

TEST(thread_count_auto) {
#ifdef _OPENMP
    const size_t K = 3;
    sv_params_t params[3];
    unsigned long seeds[3];
    for (size_t i = 0; i < K; i++) {
        params[i] = default_sv_params();
        seeds[i] = 21000 + i;
    }
    
    /* num_threads = 0 means auto */
    ensemble_config_t config = default_ensemble_config();
    config.num_threads = 0;
    
    ensemble_t* ens = ensemble_create(K, params, 500, seeds, &config);
    ASSERT_TRUE(ens != NULL);
    
    /* Should use system default (unchanged) */
    int threads = omp_get_max_threads();
    printf("    Auto thread count: %d\n", threads);
    ASSERT_TRUE(threads > 0);
    
    ensemble_free(ens);
#else
    printf("    SKIP: OpenMP not enabled\n");
    ASSERT_TRUE(1);
#endif
}

/* ========================================================================
 * Test: Determinism preserved with OpenMP
 * ======================================================================== */

TEST(determinism_with_openmp) {
#ifdef _OPENMP
    const size_t K = 3;
    sv_params_t params[3];
    unsigned long seeds[3] = {22000, 22001, 22002};
    
    for (size_t i = 0; i < K; i++) {
        params[i] = default_sv_params();
    }
    
    ensemble_config_t config = default_ensemble_config();
    config.num_threads = 2;
    
    /* Run 1 */
    ensemble_t* ens1 = ensemble_create(K, params, 500, seeds, &config);
    ASSERT_TRUE(ens1 != NULL);
    
    for (int t = 0; t < 30; t++) {
        ASSERT_TRUE(ensemble_step(ens1, 0.001 * (double)(t % 10)) == 0);
    }
    
    double weights1[3];
    ensemble_get_weights(ens1, weights1);
    ensemble_free(ens1);
    
    /* Run 2 with same seeds and thread count */
    ensemble_t* ens2 = ensemble_create(K, params, 500, seeds, &config);
    ASSERT_TRUE(ens2 != NULL);
    
    for (int t = 0; t < 30; t++) {
        ASSERT_TRUE(ensemble_step(ens2, 0.001 * (double)(t % 10)) == 0);
    }
    
    double weights2[3];
    ensemble_get_weights(ens2, weights2);
    ensemble_free(ens2);
    
    /* Results should be identical */
    for (size_t i = 0; i < K; i++) {
        ASSERT_NEAR(weights1[i], weights2[i], 1e-15);
    }
    
    printf("    Determinism verified with 2 OpenMP threads\n");
#else
    printf("    SKIP: OpenMP not enabled\n");
    ASSERT_TRUE(1);
#endif
}

/* ========================================================================
 * Main
 * ======================================================================== */

int main(void) {
    printf("\n========================================\n");
    printf("Ensemble OpenMP Tests\n");
    printf("========================================\n\n");
    
    RUN_TEST(openmp_enabled);
    RUN_TEST(thread_count_explicit);
    RUN_TEST(thread_count_auto);
    RUN_TEST(determinism_with_openmp);
    
    test_summary();
    return g_test_failed > 0 ? 1 : 0;
}
