/**
 * @file test_svmix_api.c
 * @brief Unit tests for public svmix API.
 *
 * Tests cover:
 * - Create/free lifecycle
 * - Parameter validation
 * - Step/belief operations
 * - Weights retrieval
 * - Determinism at API level
 */

#include "../test_common.h"
#include "../../include/svmix.h"
#include <stdlib.h>
#include <math.h>
#include <string.h>

/* ========================================================================
 * Test helpers
 * ======================================================================== */

static svmix_cfg_t default_config(void) {
    svmix_cfg_t cfg;
    cfg.num_models = 3;
    cfg.num_particles = 500;
    cfg.spec = SVMIX_SPEC_VOL;
    cfg.ensemble.lambda = 0.99;
    cfg.ensemble.beta = 1.0;
    cfg.ensemble.epsilon = 0.05;
    cfg.ensemble.num_threads = 0;
    return cfg;
}

static svmix_sv_params_t default_params(void) {
    svmix_sv_params_t params;
    params.mu_h = -13.815510557964;
    params.phi_h = 0.98;
    params.sigma_h = 0.15;
    params.nu = 7.0;
    return params;
}

/* ========================================================================
 * Test: Version string
 * ======================================================================== */

TEST(version_string) {
    const char* version = svmix_version();
    ASSERT_TRUE(version != NULL);
    ASSERT_TRUE(strlen(version) > 0);
    printf("    svmix version: %s\n", version);
}

/* ========================================================================
 * Test: Status strings
 * ======================================================================== */

TEST(status_strings) {
    ASSERT_TRUE(strcmp(svmix_status_string(SVMIX_OK), "Success") == 0);
    ASSERT_TRUE(strlen(svmix_status_string(SVMIX_ERR_NULL_POINTER)) > 0);
    ASSERT_TRUE(strlen(svmix_status_string(SVMIX_ERR_INVALID_PARAM)) > 0);
    ASSERT_TRUE(strlen(svmix_status_string((svmix_status_t)-999)) > 0);  /* Unknown */
}

/* ========================================================================
 * Test: Create and free
 * ======================================================================== */

TEST(create_and_free) {
    svmix_cfg_t cfg = default_config();
    svmix_sv_params_t params[3];
    unsigned long seeds[3] = {1000, 1001, 1002};
    
    for (int i = 0; i < 3; i++) {
        params[i] = default_params();
    }
    
    svmix_t* sv = svmix_create(&cfg, params, seeds);
    ASSERT_TRUE(sv != NULL);
    
    ASSERT_EQ(svmix_get_num_models(sv), 3);
    ASSERT_EQ(svmix_get_timestep(sv), 0);
    
    svmix_free(sv);
    svmix_free(NULL);  /* Should be no-op */
}

/* ========================================================================
 * Test: NULL parameter validation
 * ======================================================================== */

TEST(create_null_params) {
    svmix_cfg_t cfg = default_config();
    svmix_sv_params_t params[3];
    unsigned long seeds[3] = {2000, 2001, 2002};
    
    for (int i = 0; i < 3; i++) {
        params[i] = default_params();
    }
    
    /* NULL config */
    ASSERT_TRUE(svmix_create(NULL, params, seeds) == NULL);
    
    /* NULL params */
    ASSERT_TRUE(svmix_create(&cfg, NULL, seeds) == NULL);
    
    /* NULL seeds */
    ASSERT_TRUE(svmix_create(&cfg, params, NULL) == NULL);
}

/* ========================================================================
 * Test: Invalid configuration
 * ======================================================================== */

TEST(create_invalid_config) {
    svmix_sv_params_t params[3];
    unsigned long seeds[3] = {3000, 3001, 3002};
    
    for (int i = 0; i < 3; i++) {
        params[i] = default_params();
    }
    
    /* num_models = 0 */
    svmix_cfg_t cfg = default_config();
    cfg.num_models = 0;
    ASSERT_TRUE(svmix_create(&cfg, params, seeds) == NULL);
    
    /* num_particles = 0 */
    cfg = default_config();
    cfg.num_particles = 0;
    ASSERT_TRUE(svmix_create(&cfg, params, seeds) == NULL);
    
    /* Invalid spec */
    cfg = default_config();
    cfg.spec = (svmix_spec_t)999;
    ASSERT_TRUE(svmix_create(&cfg, params, seeds) == NULL);
    
    /* Invalid lambda */
    cfg = default_config();
    cfg.ensemble.lambda = 0.0;
    ASSERT_TRUE(svmix_create(&cfg, params, seeds) == NULL);
    
    cfg.ensemble.lambda = 1.5;
    ASSERT_TRUE(svmix_create(&cfg, params, seeds) == NULL);
    
    /* Invalid beta */
    cfg = default_config();
    cfg.ensemble.beta = -1.0;
    ASSERT_TRUE(svmix_create(&cfg, params, seeds) == NULL);
    
    /* Invalid epsilon */
    cfg = default_config();
    cfg.ensemble.epsilon = -0.1;
    ASSERT_TRUE(svmix_create(&cfg, params, seeds) == NULL);
    
    cfg.ensemble.epsilon = 1.0;
    ASSERT_TRUE(svmix_create(&cfg, params, seeds) == NULL);
}

/* ========================================================================
 * Test: Invalid SV parameters
 * ======================================================================== */

TEST(create_invalid_sv_params) {
    svmix_cfg_t cfg = default_config();
    svmix_sv_params_t params[3];
    unsigned long seeds[3] = {4000, 4001, 4002};
    
    for (int i = 0; i < 3; i++) {
        params[i] = default_params();
    }
    
    /* Invalid phi_h */
    params[0].phi_h = 0.0;
    ASSERT_TRUE(svmix_create(&cfg, params, seeds) == NULL);
    
    params[0] = default_params();
    params[1].phi_h = 1.0;
    ASSERT_TRUE(svmix_create(&cfg, params, seeds) == NULL);
    
    /* Invalid sigma_h */
    params[1] = default_params();
    params[2].sigma_h = -0.1;
    ASSERT_TRUE(svmix_create(&cfg, params, seeds) == NULL);
    
    /* Invalid nu */
    params[2] = default_params();
    params[0].nu = 2.0;
    ASSERT_TRUE(svmix_create(&cfg, params, seeds) == NULL);
    
    params[0].nu = 1.0;
    ASSERT_TRUE(svmix_create(&cfg, params, seeds) == NULL);
}

/* ========================================================================
 * Test: Step and belief
 * ======================================================================== */

TEST(step_and_belief) {
    svmix_cfg_t cfg = default_config();
    svmix_sv_params_t params[3];
    unsigned long seeds[3] = {5000, 5001, 5002};
    
    for (int i = 0; i < 3; i++) {
        params[i] = default_params();
    }
    
    svmix_t* sv = svmix_create(&cfg, params, seeds);
    ASSERT_TRUE(sv != NULL);
    
    /* Initial belief should be valid */
    svmix_belief_t belief;
    ASSERT_EQ(svmix_get_belief(sv, &belief), SVMIX_OK);
    ASSERT_EQ(belief.valid, 1);
    ASSERT_FINITE(belief.mean_h);
    ASSERT_FINITE(belief.var_h);
    ASSERT_FINITE(belief.mean_sigma);
    ASSERT_TRUE(belief.var_h >= 0.0);
    
    /* Step through 10 observations */
    for (int t = 0; t < 10; t++) {
        double r = 0.001 * (double)(t - 5);  /* -0.005 to 0.004 */
        ASSERT_EQ(svmix_step(sv, r), SVMIX_OK);
    }
    
    ASSERT_EQ(svmix_get_timestep(sv), 10);
    
    /* Belief should still be valid */
    ASSERT_EQ(svmix_get_belief(sv, &belief), SVMIX_OK);
    ASSERT_EQ(belief.valid, 1);
    ASSERT_FINITE(belief.mean_h);
    ASSERT_FINITE(belief.var_h);
    ASSERT_FINITE(belief.mean_sigma);
    
    svmix_free(sv);
}

/* ========================================================================
 * Test: Step with invalid observation
 * ======================================================================== */

TEST(step_invalid_observation) {
    svmix_cfg_t cfg = default_config();
    svmix_sv_params_t params[3];
    unsigned long seeds[3] = {6000, 6001, 6002};
    
    for (int i = 0; i < 3; i++) {
        params[i] = default_params();
    }
    
    svmix_t* sv = svmix_create(&cfg, params, seeds);
    ASSERT_TRUE(sv != NULL);
    
    /* NaN observation */
    ASSERT_EQ(svmix_step(sv, NAN), SVMIX_ERR_INVALID_PARAM);
    
    /* Inf observation */
    ASSERT_EQ(svmix_step(sv, INFINITY), SVMIX_ERR_INVALID_PARAM);
    
    /* NULL pointer */
    ASSERT_EQ(svmix_step(NULL, 0.0), SVMIX_ERR_NULL_POINTER);
    
    svmix_free(sv);
}

/* ========================================================================
 * Test: Get weights
 * ======================================================================== */

TEST(get_weights) {
    svmix_cfg_t cfg = default_config();
    svmix_sv_params_t params[3];
    unsigned long seeds[3] = {7000, 7001, 7002};
    
    for (int i = 0; i < 3; i++) {
        params[i] = default_params();
    }
    
    svmix_t* sv = svmix_create(&cfg, params, seeds);
    ASSERT_TRUE(sv != NULL);
    
    double weights[3];
    ASSERT_EQ(svmix_get_weights(sv, weights, 3), SVMIX_OK);
    
    /* Initial weights should be uniform */
    double sum = 0.0;
    for (int i = 0; i < 3; i++) {
        ASSERT_TRUE(weights[i] >= 0.0);
        ASSERT_TRUE(weights[i] <= 1.0);
        ASSERT_FINITE(weights[i]);
        sum += weights[i];
    }
    ASSERT_NEAR(sum, 1.0, 1e-10);
    
    /* Step and check weights still valid */
    for (int t = 0; t < 20; t++) {
        ASSERT_EQ(svmix_step(sv, 0.001), SVMIX_OK);
    }
    
    ASSERT_EQ(svmix_get_weights(sv, weights, 3), SVMIX_OK);
    sum = 0.0;
    for (int i = 0; i < 3; i++) {
        ASSERT_TRUE(weights[i] >= 0.0);
        ASSERT_FINITE(weights[i]);
        sum += weights[i];
    }
    ASSERT_NEAR(sum, 1.0, 1e-10);
    
    /* Wrong K */
    ASSERT_EQ(svmix_get_weights(sv, weights, 5), SVMIX_ERR_INVALID_PARAM);
    
    /* NULL pointer */
    ASSERT_EQ(svmix_get_weights(NULL, weights, 3), SVMIX_ERR_NULL_POINTER);
    ASSERT_EQ(svmix_get_weights(sv, NULL, 3), SVMIX_ERR_NULL_POINTER);
    
    svmix_free(sv);
}

/* ========================================================================
 * Test: Belief NULL pointer
 * ======================================================================== */

TEST(belief_null_pointer) {
    svmix_cfg_t cfg = default_config();
    svmix_sv_params_t params[3];
    unsigned long seeds[3] = {8000, 8001, 8002};
    
    for (int i = 0; i < 3; i++) {
        params[i] = default_params();
    }
    
    svmix_t* sv = svmix_create(&cfg, params, seeds);
    ASSERT_TRUE(sv != NULL);
    
    svmix_belief_t belief;
    
    /* NULL svmix */
    ASSERT_EQ(svmix_get_belief(NULL, &belief), SVMIX_ERR_NULL_POINTER);
    
    /* NULL belief */
    ASSERT_EQ(svmix_get_belief(sv, NULL), SVMIX_ERR_NULL_POINTER);
    
    svmix_free(sv);
}

/* ========================================================================
 * Test: Determinism at API level
 * ======================================================================== */

TEST(determinism_same_seeds) {
    svmix_cfg_t cfg = default_config();
    svmix_sv_params_t params[3];
    unsigned long seeds[3] = {9000, 9001, 9002};
    
    for (int i = 0; i < 3; i++) {
        params[i] = default_params();
        /* Vary parameters slightly */
        params[i].phi_h = 0.96 + 0.01 * (double)i;
    }
    
    /* Run 1 */
    svmix_t* sv1 = svmix_create(&cfg, params, seeds);
    ASSERT_TRUE(sv1 != NULL);
    
    double observations[30];
    for (int t = 0; t < 30; t++) {
        observations[t] = 0.001 * (double)(t % 20 - 10);
        ASSERT_EQ(svmix_step(sv1, observations[t]), SVMIX_OK);
    }
    
    svmix_belief_t belief1;
    double weights1[3];
    ASSERT_EQ(svmix_get_belief(sv1, &belief1), SVMIX_OK);
    ASSERT_EQ(svmix_get_weights(sv1, weights1, 3), SVMIX_OK);
    
    svmix_free(sv1);
    
    /* Run 2 with same seeds */
    svmix_t* sv2 = svmix_create(&cfg, params, seeds);
    ASSERT_TRUE(sv2 != NULL);
    
    for (int t = 0; t < 30; t++) {
        ASSERT_EQ(svmix_step(sv2, observations[t]), SVMIX_OK);
    }
    
    svmix_belief_t belief2;
    double weights2[3];
    ASSERT_EQ(svmix_get_belief(sv2, &belief2), SVMIX_OK);
    ASSERT_EQ(svmix_get_weights(sv2, weights2, 3), SVMIX_OK);
    
    svmix_free(sv2);
    
    /* Results should be bitwise-identical */
    ASSERT_EQ(belief1.valid, belief2.valid);
    ASSERT_NEAR(belief1.mean_h, belief2.mean_h, 1e-15);
    ASSERT_NEAR(belief1.var_h, belief2.var_h, 1e-15);
    ASSERT_NEAR(belief1.mean_sigma, belief2.mean_sigma, 1e-15);
    
    for (int i = 0; i < 3; i++) {
        ASSERT_NEAR(weights1[i], weights2[i], 1e-15);
    }
}

/* ========================================================================
 * Test: Different seeds give different results
 * ======================================================================== */

TEST(determinism_different_seeds) {
    svmix_cfg_t cfg = default_config();
    svmix_sv_params_t params[3];
    unsigned long seeds1[3] = {10000, 10001, 10002};
    unsigned long seeds2[3] = {20000, 20001, 20002};
    
    for (int i = 0; i < 3; i++) {
        params[i] = default_params();
    }
    
    /* Run 1 */
    svmix_t* sv1 = svmix_create(&cfg, params, seeds1);
    ASSERT_TRUE(sv1 != NULL);
    
    for (int t = 0; t < 20; t++) {
        ASSERT_EQ(svmix_step(sv1, 0.001), SVMIX_OK);
    }
    
    svmix_belief_t belief1;
    ASSERT_EQ(svmix_get_belief(sv1, &belief1), SVMIX_OK);
    svmix_free(sv1);
    
    /* Run 2 with different seeds */
    svmix_t* sv2 = svmix_create(&cfg, params, seeds2);
    ASSERT_TRUE(sv2 != NULL);
    
    for (int t = 0; t < 20; t++) {
        ASSERT_EQ(svmix_step(sv2, 0.001), SVMIX_OK);
    }
    
    svmix_belief_t belief2;
    ASSERT_EQ(svmix_get_belief(sv2, &belief2), SVMIX_OK);
    svmix_free(sv2);
    
    /* Results should differ (probabilistically certain) */
    int differs = (fabs(belief1.mean_h - belief2.mean_h) > 1e-10) ||
                  (fabs(belief1.var_h - belief2.var_h) > 1e-10);
    ASSERT_TRUE(differs);
}

/* ========================================================================
 * Test: Get functions with NULL
 * ======================================================================== */

TEST(get_functions_null) {
    ASSERT_EQ(svmix_get_num_models(NULL), 0);
    ASSERT_EQ(svmix_get_timestep(NULL), 0);
}

/* ========================================================================
 * Main
 * ======================================================================== */

int main(void) {
    printf("\n========================================\n");
    printf("svmix Public API Tests\n");
    printf("========================================\n\n");
    
    RUN_TEST(version_string);
    RUN_TEST(status_strings);
    RUN_TEST(create_and_free);
    RUN_TEST(create_null_params);
    RUN_TEST(create_invalid_config);
    RUN_TEST(create_invalid_sv_params);
    RUN_TEST(step_and_belief);
    RUN_TEST(step_invalid_observation);
    RUN_TEST(get_weights);
    RUN_TEST(belief_null_pointer);
    RUN_TEST(determinism_same_seeds);
    RUN_TEST(determinism_different_seeds);
    RUN_TEST(get_functions_null);
    
    test_summary();
    return g_test_failed > 0 ? 1 : 0;
}
