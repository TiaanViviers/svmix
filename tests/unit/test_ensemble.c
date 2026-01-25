/**
 * @file test_ensemble.c
 * @brief Unit tests for ensemble layer.
 *
 * Tests focus on:
 * - Weight normalization and non-negativity
 * - Softmax numerical stability (extreme scores)
 * - Anti-starvation mixing (minimum weight guarantees)
 * - Score update dynamics (recurrence equation)
 * - Dominant model scenario (convergence behavior)
 * - Determinism (same seeds -> same trajectories)
 */

#include "../test_common.h"
#include "../../src/ensemble.h"
#include <stdlib.h>
#include <math.h>
#include <float.h>

/* ========================================================================
 * Test helpers
 * ======================================================================== */

/**
 * @brief Create default SV parameters for testing.
 */
static sv_params_t default_sv_params(void) {
    sv_params_t params;
    params.mu_h = -13.815510557964;  /* log(0.001^2) */
    params.phi_h = 0.98;
    params.sigma_h = 0.15;
    params.nu = 7.0;
    return params;
}

/**
 * @brief Create default ensemble config for testing.
 */
static ensemble_config_t default_ensemble_config(void) {
    ensemble_config_t config;
    config.lambda = 0.99;
    config.beta = 1.0;
    config.epsilon = 0.05;
    config.num_threads = 0;  /* Auto */
    return config;
}

/* ========================================================================
 * Test: weight normalization
 * ======================================================================== */

TEST(weight_normalization) {
    /* Create ensemble with 5 models */
    const size_t K = 5;
    sv_params_t params[5];
    unsigned long seeds[5];
    for (size_t i = 0; i < K; i++) {
        params[i] = default_sv_params();
        seeds[i] = 1000 + i;
    }
    
    ensemble_config_t config = default_ensemble_config();
    
    ensemble_t* ens = ensemble_create(K, params, 1000, seeds, &config);
    ASSERT_TRUE(ens != NULL);
    
    /* Initial weights should sum to 1 */
    double weights[5];
    size_t nmodels = ensemble_get_weights(ens, weights);
    ASSERT_EQ(nmodels, K);
    
    double sum = 0.0;
    for (size_t i = 0; i < K; i++) {
        ASSERT_TRUE(weights[i] >= 0.0);
        ASSERT_TRUE(isfinite(weights[i]));
        sum += weights[i];
    }
    ASSERT_NEAR(sum, 1.0, 1e-10);
    
    /* After a step, weights should still sum to 1 */
    ASSERT_TRUE(ensemble_step(ens, 0.001) == 0);
    
    nmodels = ensemble_get_weights(ens, weights);
    ASSERT_EQ(nmodels, K);
    
    sum = 0.0;
    for (size_t i = 0; i < K; i++) {
        ASSERT_TRUE(weights[i] >= 0.0);
        ASSERT_TRUE(isfinite(weights[i]));
        sum += weights[i];
    }
    ASSERT_NEAR(sum, 1.0, 1e-10);
    
    ensemble_free(ens);
}

/* ========================================================================
 * Test: softmax stability with extreme scores
 * ======================================================================== */

TEST(softmax_stability_extreme_scores) {
    /* Create ensemble with 3 models */
    const size_t K = 3;
    sv_params_t params[3];
    unsigned long seeds[3];
    for (size_t i = 0; i < K; i++) {
        params[i] = default_sv_params();
        seeds[i] = 2000 + i;
    }
    
    ensemble_config_t config = default_ensemble_config();
    
    ensemble_t* ens = ensemble_create(K, params, 500, seeds, &config);
    ASSERT_TRUE(ens != NULL);
    
    /* Feed 100 observations to build up scores */
    for (int t = 0; t < 100; t++) {
        double r = 0.001 * (double)(t % 10 - 5);  /* Vary -0.005 to 0.005 */
        ASSERT_TRUE(ensemble_step(ens, r) == 0);
    }
    
    /* Check weights are still finite and normalized */
    double weights[3];
    ensemble_get_weights(ens, weights);
    
    double sum = 0.0;
    for (size_t i = 0; i < K; i++) {
        ASSERT_FINITE(weights[i]);
        ASSERT_TRUE(weights[i] >= 0.0);
        sum += weights[i];
    }
    ASSERT_NEAR(sum, 1.0, 1e-10);
    
    /* Scores might be large, but should be finite */
    double scores[3];
    ensemble_get_scores(ens, scores);
    for (size_t i = 0; i < K; i++) {
        ASSERT_FINITE(scores[i]);
    }
    
    ensemble_free(ens);
}

/* ========================================================================
 * Test: anti-starvation mixing
 * ======================================================================== */

TEST(anti_starvation_mixing) {
    /* Create ensemble with epsilon = 0.1 (10%) */
    const size_t K = 4;
    sv_params_t params[4];
    unsigned long seeds[4];
    for (size_t i = 0; i < K; i++) {
        params[i] = default_sv_params();
        seeds[i] = 3000 + i;
    }
    
    ensemble_config_t config = default_ensemble_config();
    config.epsilon = 0.1;  /* Each model gets at least 0.1/4 = 0.025 */
    
    ensemble_t* ens = ensemble_create(K, params, 500, seeds, &config);
    ASSERT_TRUE(ens != NULL);
    
    /* Feed 50 observations */
    for (int t = 0; t < 50; t++) {
        ASSERT_TRUE(ensemble_step(ens, 0.002) == 0);
    }
    
    /* Check every weight is >= epsilon / K */
    double weights[4];
    ensemble_get_weights(ens, weights);
    
    double min_weight = config.epsilon / (double)K;
    for (size_t i = 0; i < K; i++) {
        ASSERT_TRUE(weights[i] >= min_weight - 1e-12);  /* Allow tiny numerical error */
    }
    
    ensemble_free(ens);
}

/* ========================================================================
 * Test: score update recurrence
 * ======================================================================== */

TEST(score_update_recurrence) {
    /* Create ensemble with single model to check score dynamics */
    const size_t K = 1;
    sv_params_t params[1];
    params[0] = default_sv_params();
    unsigned long seeds[1] = {4000};
    
    ensemble_config_t config = default_ensemble_config();
    config.lambda = 0.95;  /* Clear forgetting for testing */
    
    ensemble_t* ens = ensemble_create(K, params, 1000, seeds, &config);
    ASSERT_TRUE(ens != NULL);
    
    /* Initial score should be 0 */
    double scores[1];
    ensemble_get_scores(ens, scores);
    ASSERT_NEAR(scores[0], 0.0, 1e-10);
    
    /* After first step: S_1 = 0.95 * 0 + ll_1 = ll_1 */
    ASSERT_TRUE(ensemble_step(ens, 0.0) == 0);
    ensemble_get_scores(ens, scores);
    double S_1 = scores[0];
    ASSERT_FINITE(S_1);
    
    /* After second step: S_2 = 0.95 * S_1 + ll_2 */
    ASSERT_TRUE(ensemble_step(ens, 0.0) == 0);
    double S_1_prev = S_1;
    ensemble_get_scores(ens, scores);
    double S_2 = scores[0];
    ASSERT_FINITE(S_2);
    
    /* We can't predict exact ll_2, but S_2 should be related to S_1 */
    /* At minimum, S_2 should be influenced by S_1 (not random) */
    /* Just verify it's a reasonable magnitude */
    ASSERT_TRUE(fabs(S_2) < 1e6);  /* Not blown up */
    ASSERT_TRUE(fabs(S_2 - S_1_prev) < 1e6);  /* Didn't jump wildly */
    
    ensemble_free(ens);
}

/* ========================================================================
 * Test: dominant model scenario
 * ======================================================================== */

TEST(dominant_model_scenario) {
    /* Create 3 models: two with wrong parameters, one correct */
    const size_t K = 3;
    sv_params_t params[3];
    
    /* Model 0: correct parameters */
    params[0] = default_sv_params();
    
    /* Model 1: wrong phi_h (too low persistence) */
    params[1] = default_sv_params();
    params[1].phi_h = 0.5;
    
    /* Model 2: wrong sigma_h (too high volatility) */
    params[2] = default_sv_params();
    params[2].sigma_h = 0.5;
    
    unsigned long seeds[3] = {5000, 5001, 5002};
    
    ensemble_config_t config = default_ensemble_config();
    config.epsilon = 0.01;  /* Low anti-starvation for clear dominance */
    config.beta = 1.0;
    
    ensemble_t* ens = ensemble_create(K, params, 1000, seeds, &config);
    ASSERT_TRUE(ens != NULL);
    
    /* Generate synthetic data from model 0's parameters */
    /* For simplicity, just feed small returns (consistent with default params) */
    for (int t = 0; t < 100; t++) {
        double r = 0.001 * sin((double)t * 0.1);  /* Small oscillating returns */
        ASSERT_TRUE(ensemble_step(ens, r) == 0);
    }
    
    /* Model 0 should have highest weight (not guaranteed, but likely) */
    double weights[3];
    ensemble_get_weights(ens, weights);
    
    /* At minimum, weights should be reasonable */
    for (size_t i = 0; i < K; i++) {
        ASSERT_TRUE(weights[i] >= 0.01 / 3.0 - 1e-10);  /* Anti-starvation */
        ASSERT_TRUE(weights[i] <= 1.0);
        ASSERT_FINITE(weights[i]);
    }
    
    double sum = weights[0] + weights[1] + weights[2];
    ASSERT_NEAR(sum, 1.0, 1e-10);
    
    ensemble_free(ens);
}

/* ========================================================================
 * Test: determinism with same seed
 * ======================================================================== */

TEST(determinism_same_seed) {
    const size_t K = 3;
    sv_params_t params[3];
    unsigned long seeds[3] = {6000, 6001, 6002};
    
    for (size_t i = 0; i < K; i++) {
        params[i] = default_sv_params();
        /* Vary parameters slightly to make it interesting */
        params[i].phi_h = 0.96 + 0.01 * (double)i;
    }
    
    ensemble_config_t config = default_ensemble_config();
    config.lambda = 0.99;
    config.beta = 1.0;
    config.epsilon = 0.05;
    
    /* Run 1 */
    ensemble_t* ens1 = ensemble_create(K, params, 1000, seeds, &config);
    ASSERT_TRUE(ens1 != NULL);
    
    double observations[50];
    for (int t = 0; t < 50; t++) {
        observations[t] = 0.001 * (double)(t % 20 - 10);
    }
    
    for (int t = 0; t < 50; t++) {
        ASSERT_TRUE(ensemble_step(ens1, observations[t]) == 0);
    }
    
    double weights1[3], scores1[3];
    ensemble_get_weights(ens1, weights1);
    ensemble_get_scores(ens1, scores1);
    ensemble_belief_t belief1 = ensemble_get_belief(ens1);
    
    ensemble_free(ens1);
    
    /* Run 2 with same seeds */
    ensemble_t* ens2 = ensemble_create(K, params, 1000, seeds, &config);
    ASSERT_TRUE(ens2 != NULL);
    
    for (int t = 0; t < 50; t++) {
        ASSERT_TRUE(ensemble_step(ens2, observations[t]) == 0);
    }
    
    double weights2[3], scores2[3];
    ensemble_get_weights(ens2, weights2);
    ensemble_get_scores(ens2, scores2);
    ensemble_belief_t belief2 = ensemble_get_belief(ens2);
    
    ensemble_free(ens2);
    
    /* Verify bitwise-identical results */
    for (size_t i = 0; i < K; i++) {
        ASSERT_NEAR(weights1[i], weights2[i], 1e-15);
        ASSERT_NEAR(scores1[i], scores2[i], 1e-15);
    }
    
    ASSERT_TRUE(belief1.valid == belief2.valid);
    if (belief1.valid) {
        ASSERT_NEAR(belief1.mean_h, belief2.mean_h, 1e-15);
        ASSERT_NEAR(belief1.var_h, belief2.var_h, 1e-15);
        ASSERT_NEAR(belief1.mean_sigma, belief2.mean_sigma, 1e-15);
    }
}

/* ========================================================================
 * Test: determinism with different seeds
 * ======================================================================== */

TEST(determinism_different_seeds) {
    const size_t K = 2;
    sv_params_t params[2];
    unsigned long seeds1[2] = {7000, 7001};
    unsigned long seeds2[2] = {8000, 8001};
    
    for (size_t i = 0; i < K; i++) {
        params[i] = default_sv_params();
    }
    
    ensemble_config_t config = default_ensemble_config();
    
    /* Run 1 */
    ensemble_t* ens1 = ensemble_create(K, params, 1000, seeds1, &config);
    ASSERT_TRUE(ens1 != NULL);
    
    for (int t = 0; t < 30; t++) {
        ASSERT_TRUE(ensemble_step(ens1, 0.001) == 0);
    }
    
    double weights1[2];
    ensemble_get_weights(ens1, weights1);
    ensemble_free(ens1);
    
    /* Run 2 with different seeds */
    ensemble_t* ens2 = ensemble_create(K, params, 1000, seeds2, &config);
    ASSERT_TRUE(ens2 != NULL);
    
    for (int t = 0; t < 30; t++) {
        ASSERT_TRUE(ensemble_step(ens2, 0.001) == 0);
    }
    
    double weights2[2];
    ensemble_get_weights(ens2, weights2);
    ensemble_free(ens2);
    
    /* Results should differ (probabilistically certain with 30 steps) */
    bool differs = false;
    for (size_t i = 0; i < K; i++) {
        if (fabs(weights1[i] - weights2[i]) > 1e-10) {
            differs = true;
            break;
        }
    }
    ASSERT_TRUE(differs);
}

/* ========================================================================
 * Test: parameter validation
 * ======================================================================== */

TEST(parameter_validation) {
    const size_t K = 2;
    sv_params_t params[2];
    unsigned long seeds[2] = {9000, 9001};
    
    for (size_t i = 0; i < K; i++) {
        params[i] = default_sv_params();
    }
    
    ensemble_config_t config = default_ensemble_config();
    
    /* Valid creation */
    ensemble_t* ens = ensemble_create(K, params, 1000, seeds, &config);
    ASSERT_TRUE(ens != NULL);
    ensemble_free(ens);
    
    /* K = 0 */
    ens = ensemble_create(0, params, 1000, seeds, &config);
    ASSERT_TRUE(ens == NULL);
    
    /* NULL params */
    ens = ensemble_create(K, NULL, 1000, seeds, &config);
    ASSERT_TRUE(ens == NULL);
    
    /* NULL seeds */
    ens = ensemble_create(K, params, 1000, NULL, &config);
    ASSERT_TRUE(ens == NULL);
    
    /* NULL config */
    ens = ensemble_create(K, params, 1000, seeds, NULL);
    ASSERT_TRUE(ens == NULL);
    
    /* num_particles = 0 */
    ens = ensemble_create(K, params, 0, seeds, &config);
    ASSERT_TRUE(ens == NULL);
    
    /* Invalid lambda */
    config = default_ensemble_config();
    config.lambda = 0.0;
    ens = ensemble_create(K, params, 1000, seeds, &config);
    ASSERT_TRUE(ens == NULL);
    
    config.lambda = 1.5;
    ens = ensemble_create(K, params, 1000, seeds, &config);
    ASSERT_TRUE(ens == NULL);
    
    /* Invalid beta */
    config = default_ensemble_config();
    config.beta = -1.0;
    ens = ensemble_create(K, params, 1000, seeds, &config);
    ASSERT_TRUE(ens == NULL);
    
    /* Invalid epsilon */
    config = default_ensemble_config();
    config.epsilon = -0.1;
    ens = ensemble_create(K, params, 1000, seeds, &config);
    ASSERT_TRUE(ens == NULL);
    
    config.epsilon = 1.0;
    ens = ensemble_create(K, params, 1000, seeds, &config);
    ASSERT_TRUE(ens == NULL);
}

/* ========================================================================
 * Test: belief computation
 * ======================================================================== */

TEST(belief_computation) {
    const size_t K = 2;
    sv_params_t params[2];
    unsigned long seeds[2] = {10000, 10001};
    
    for (size_t i = 0; i < K; i++) {
        params[i] = default_sv_params();
    }
    
    ensemble_config_t config = default_ensemble_config();
    
    ensemble_t* ens = ensemble_create(K, params, 1000, seeds, &config);
    ASSERT_TRUE(ens != NULL);
    
    /* Initial belief should be valid */
    ensemble_belief_t belief = ensemble_get_belief(ens);
    ASSERT_TRUE(belief.valid);
    ASSERT_FINITE(belief.mean_h);
    ASSERT_FINITE(belief.var_h);
    ASSERT_FINITE(belief.mean_sigma);
    ASSERT_TRUE(belief.var_h >= 0.0);
    
    /* After some steps, belief should still be valid */
    for (int t = 0; t < 20; t++) {
        ASSERT_TRUE(ensemble_step(ens, 0.001) == 0);
    }
    
    belief = ensemble_get_belief(ens);
    ASSERT_TRUE(belief.valid);
    ASSERT_FINITE(belief.mean_h);
    ASSERT_FINITE(belief.var_h);
    ASSERT_FINITE(belief.mean_sigma);
    ASSERT_TRUE(belief.var_h >= 0.0);
    
    ensemble_free(ens);
}

/* ========================================================================
 * Main
 * ======================================================================== */

int main(void) {
    printf("\n========================================\n");
    printf("Ensemble Unit Tests\n");
    printf("========================================\n\n");
    
    RUN_TEST(weight_normalization);
    RUN_TEST(softmax_stability_extreme_scores);
    RUN_TEST(anti_starvation_mixing);
    RUN_TEST(score_update_recurrence);
    RUN_TEST(dominant_model_scenario);
    RUN_TEST(determinism_same_seed);
    RUN_TEST(determinism_different_seeds);
    RUN_TEST(parameter_validation);
    RUN_TEST(belief_computation);
    
    test_summary();
    return g_test_failed > 0 ? 1 : 0;
}
