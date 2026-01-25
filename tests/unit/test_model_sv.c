/**
 * @file test_model_sv.c
 * @brief Unit tests for SV model callbacks (no fastpf integration yet).
 *
 * Tests the three SV callbacks in isolation:
 * 1. Prior sampling - matches stationary distribution
 * 2. Transition sampling - AR(1) dynamics
 * 3. Log-likelihood - consistency checks
 */

#include "../test_common.h"
#include "../../src/model_sv.h"
#include "../../third_party/fastpf/include/fastpf.h"
#include <stdlib.h>
#include <math.h>
#include <stdint.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* ========================================================================
 * Test-local RNG (compatible with fastpf_rng_t for testing)
 * ======================================================================== */

/**
 * @brief Test-local RNG structure compatible with fastpf_rng_t.
 * fastpf_rng_t is opaque, but we know from fastpf_internal.h it's just:
 *   struct { uint64_t state; uint64_t inc; };
 * We can create a compatible struct for testing.
 */
typedef struct {
    uint64_t state;
    uint64_t inc;
} test_rng_t;

/* ========================================================================
 * Test Helpers
 * ======================================================================== */

/**
 * @brief Compute sample mean of an array.
 */
static double sample_mean(const double* data, size_t n) {
    double sum = 0.0;
    size_t i;
    for (i = 0; i < n; i++) {
        sum += data[i];
    }
    return sum / (double)n;
}

/**
 * @brief Compute sample variance of an array.
 */
static double sample_variance(const double* data, size_t n) {
    double mean = sample_mean(data, n);
    double sum_sq = 0.0;
    size_t i;
    for (i = 0; i < n; i++) {
        double diff = data[i] - mean;
        sum_sq += diff * diff;
    }
    return sum_sq / (double)(n - 1);  /* Unbiased estimator */
}

/* ========================================================================
 * Test: Parameter Validation
 * ======================================================================== */

TEST(create_valid_params) {
    fastpf_model_t model;
    sv_model_ctx_t* ctx = sv_model_create(-13.8, 0.98, 0.15, 7.0, &model);
    
    ASSERT_TRUE(ctx != NULL);
    ASSERT_TRUE(model.ctx == ctx);
    ASSERT_TRUE(model.prior_sample != NULL);
    ASSERT_TRUE(model.transition_sample != NULL);
    ASSERT_TRUE(model.log_likelihood != NULL);
    ASSERT_TRUE(model.rejuvenate == NULL);  /* v1: no rejuvenation */
    
    free(ctx);
}

TEST(create_null_model_out) {
    sv_model_ctx_t* ctx = sv_model_create(-13.8, 0.98, 0.15, 7.0, NULL);
    ASSERT_TRUE(ctx == NULL);
}

TEST(create_invalid_phi) {
    fastpf_model_t model;
    
    /* phi <= 0 */
    ASSERT_TRUE(sv_model_create(-13.8, 0.0, 0.15, 7.0, &model) == NULL);
    ASSERT_TRUE(sv_model_create(-13.8, -0.5, 0.15, 7.0, &model) == NULL);
    
    /* phi >= 1 */
    ASSERT_TRUE(sv_model_create(-13.8, 1.0, 0.15, 7.0, &model) == NULL);
    ASSERT_TRUE(sv_model_create(-13.8, 1.5, 0.15, 7.0, &model) == NULL);
    
    /* phi too close to 1 (variance blow-up guard) */
    ASSERT_TRUE(sv_model_create(-13.8, 0.99999, 0.15, 7.0, &model) == NULL);
}

TEST(create_invalid_sigma) {
    fastpf_model_t model;
    
    ASSERT_TRUE(sv_model_create(-13.8, 0.98, 0.0, 7.0, &model) == NULL);
    ASSERT_TRUE(sv_model_create(-13.8, 0.98, -0.1, 7.0, &model) == NULL);
}

TEST(create_invalid_nu) {
    fastpf_model_t model;
    
    /* nu <= 2 (no finite variance) */
    ASSERT_TRUE(sv_model_create(-13.8, 0.98, 0.15, 2.0, &model) == NULL);
    ASSERT_TRUE(sv_model_create(-13.8, 0.98, 0.15, 1.5, &model) == NULL);
    ASSERT_TRUE(sv_model_create(-13.8, 0.98, 0.15, 0.0, &model) == NULL);
}

TEST(create_invalid_mu) {
    fastpf_model_t model;
    
    ASSERT_TRUE(sv_model_create(NAN, 0.98, 0.15, 7.0, &model) == NULL);
    ASSERT_TRUE(sv_model_create(INFINITY, 0.98, 0.15, 7.0, &model) == NULL);
}

/* ========================================================================
 * Test: Prior Sampling (Stationary Distribution)
 * ======================================================================== */

TEST(prior_sampling_mean_variance) {
    /* Create model with known parameters */
    double mu_h = -13.8;
    double phi_h = 0.98;
    double sigma_h = 0.15;
    double nu = 7.0;
    
    fastpf_model_t model;
    sv_model_ctx_t* ctx = sv_model_create(mu_h, phi_h, sigma_h, nu, &model);
    ASSERT_TRUE(ctx != NULL);
    
    /* Theoretical stationary distribution: N(mu_h, sigma_h^2 / (1 - phi_h^2)) */
    double theory_mean = mu_h;
    double theory_var = (sigma_h * sigma_h) / (1.0 - phi_h * phi_h);
    
    /* Draw many samples */
    const size_t N = 50000;
    double* samples = (double*)malloc(N * sizeof(double));
    ASSERT_TRUE(samples != NULL);
    
    test_rng_t rng;
    fastpf_rng_seed((fastpf_rng_t*)&rng, 42);  /* Fixed seed for reproducibility */
    
    size_t i;
    for (i = 0; i < N; i++) {
        model.prior_sample(model.ctx, &samples[i], (fastpf_rng_t*)&rng);
        ASSERT_FINITE(samples[i]);  /* Sanity check */
    }
    
    /* Compute sample statistics */
    double sample_mu = sample_mean(samples, N);
    double sample_var = sample_variance(samples, N);
    
    /* Check against theory (Monte Carlo tolerances) */
    double mean_error = fabs(sample_mu - theory_mean);
    double var_rel_error = fabs(sample_var - theory_var) / theory_var;
    
    /* With N=50k, standard error of mean is ~SD/sqrt(N) = 0.754/224 ≈ 0.0034 */
    /* Allow 6-sigma deviation: ~0.02 */
    ASSERT_TRUE(mean_error < 0.02);
    
    /* Variance estimate has higher error; allow 5% relative error */
    ASSERT_TRUE(var_rel_error < 0.05);
    
    free(samples);
    free(ctx);
}

/* ========================================================================
 * Test: Transition Sampling (AR(1) Dynamics)
 * ======================================================================== */

TEST(transition_mean_reversion) {
    /* Create model */
    double mu_h = -13.8;
    double phi_h = 0.98;
    double sigma_h = 0.15;
    double nu = 7.0;
    
    fastpf_model_t model;
    sv_model_ctx_t* ctx = sv_model_create(mu_h, phi_h, sigma_h, nu, &model);
    ASSERT_TRUE(ctx != NULL);
    
    /* Fix h_prev at an offset from mean */
    double h_prev = mu_h + 1.0;
    
    /* Theory: E[h_next | h_prev] = mu + phi*(h_prev - mu) */
    double theory_mean = mu_h + phi_h * (h_prev - mu_h);
    double theory_var = sigma_h * sigma_h;
    
    /* Draw many samples */
    const size_t N = 50000;
    double* samples = (double*)malloc(N * sizeof(double));
    ASSERT_TRUE(samples != NULL);
    
    test_rng_t rng;
    fastpf_rng_seed((fastpf_rng_t*)&rng, 123);
    
    size_t i;
    for (i = 0; i < N; i++) {
        model.transition_sample(model.ctx, &h_prev, &samples[i], (fastpf_rng_t*)&rng);
        ASSERT_FINITE(samples[i]);
    }
    
    double sample_mu = sample_mean(samples, N);
    double sample_var = sample_variance(samples, N);
    
    double mean_error = fabs(sample_mu - theory_mean);
    double var_rel_error = fabs(sample_var - theory_var) / theory_var;
    
    ASSERT_TRUE(mean_error < 0.01);
    ASSERT_TRUE(var_rel_error < 0.05);
    
    free(samples);
    free(ctx);
}

TEST(transition_from_mean) {
    /* Special case: h_prev = mu should give h_next ~ N(mu, sigma^2) */
    double mu_h = -10.0;
    double phi_h = 0.95;
    double sigma_h = 0.2;
    double nu = 7.0;
    
    fastpf_model_t model;
    sv_model_ctx_t* ctx = sv_model_create(mu_h, phi_h, sigma_h, nu, &model);
    ASSERT_TRUE(ctx != NULL);
    
    double h_prev = mu_h;
    
    const size_t N = 50000;
    double* samples = (double*)malloc(N * sizeof(double));
    ASSERT_TRUE(samples != NULL);
    
    test_rng_t rng;
    fastpf_rng_seed((fastpf_rng_t*)&rng, 456);
    
    size_t i;
    for (i = 0; i < N; i++) {
        model.transition_sample(model.ctx, &h_prev, &samples[i], (fastpf_rng_t*)&rng);
    }
    
    double sample_mu = sample_mean(samples, N);
    double sample_var = sample_variance(samples, N);
    
    ASSERT_TRUE(fabs(sample_mu - mu_h) < 0.01);
    ASSERT_TRUE(fabs(sample_var - sigma_h * sigma_h) / (sigma_h * sigma_h) < 0.05);
    
    free(samples);
    free(ctx);
}

/* ========================================================================
 * Test: Log-Likelihood Consistency
 * ======================================================================== */

TEST(likelihood_symmetry) {
    /* Student-t with loc=0 should be symmetric */
    fastpf_model_t model;
    sv_model_ctx_t* ctx = sv_model_create(-13.8, 0.98, 0.15, 7.0, &model);
    ASSERT_TRUE(ctx != NULL);
    
    double h = -13.8;
    double r_pos = 0.001;
    double r_neg = -0.001;
    
    double ll_pos = model.log_likelihood(model.ctx, &h, &r_pos);
    double ll_neg = model.log_likelihood(model.ctx, &h, &r_neg);
    
    ASSERT_FINITE(ll_pos);
    ASSERT_FINITE(ll_neg);
    ASSERT_NEAR(ll_pos, ll_neg, 1e-14);
    
    free(ctx);
}

TEST(likelihood_peak_at_zero) {
    /* Density should be highest at r=0 */
    fastpf_model_t model;
    sv_model_ctx_t* ctx = sv_model_create(-13.8, 0.98, 0.15, 7.0, &model);
    ASSERT_TRUE(ctx != NULL);
    
    double h = -13.8;
    double r0 = 0.0;
    double r1 = 0.0001;
    
    double ll0 = model.log_likelihood(model.ctx, &h, &r0);
    double ll1 = model.log_likelihood(model.ctx, &h, &r1);
    
    ASSERT_FINITE(ll0);
    ASSERT_FINITE(ll1);
    ASSERT_TRUE(ll0 > ll1);  /* Peak at mode */
    
    free(ctx);
}

TEST(likelihood_monotonic_in_abs_r) {
    /* For |r1| < |r2|, should have ll(r1) > ll(r2) */
    fastpf_model_t model;
    sv_model_ctx_t* ctx = sv_model_create(-13.8, 0.98, 0.15, 7.0, &model);
    ASSERT_TRUE(ctx != NULL);
    
    double h = -13.8;
    double r1 = 0.0005;
    double r2 = 0.001;
    double r3 = 0.002;
    
    double ll1 = model.log_likelihood(model.ctx, &h, &r1);
    double ll2 = model.log_likelihood(model.ctx, &h, &r2);
    double ll3 = model.log_likelihood(model.ctx, &h, &r3);
    
    ASSERT_FINITE(ll1);
    ASSERT_FINITE(ll2);
    ASSERT_FINITE(ll3);
    ASSERT_TRUE(ll1 > ll2);
    ASSERT_TRUE(ll2 > ll3);
    
    free(ctx);
}

TEST(likelihood_finite_for_reasonable_h) {
    /* For h in [-20, 0], likelihood should be finite */
    fastpf_model_t model;
    sv_model_ctx_t* ctx = sv_model_create(-13.8, 0.98, 0.15, 7.0, &model);
    ASSERT_TRUE(ctx != NULL);
    
    double r = 0.0005;
    double h_values[] = {-20.0, -15.0, -13.8, -10.0, -5.0, 0.0};
    size_t i;
    
    for (i = 0; i < sizeof(h_values) / sizeof(h_values[0]); i++) {
        double ll = model.log_likelihood(model.ctx, &h_values[i], &r);
        ASSERT_FINITE(ll);
    }
    
    free(ctx);
}

/* ========================================================================
 * Test Runner
 * ======================================================================== */

int main(void) {
    printf("========================================\n");
    printf("SV Model Callback Unit Tests\n");
    printf("========================================\n\n");
    
    /* Parameter validation */
    RUN_TEST(create_valid_params);
    RUN_TEST(create_null_model_out);
    RUN_TEST(create_invalid_phi);
    RUN_TEST(create_invalid_sigma);
    RUN_TEST(create_invalid_nu);
    RUN_TEST(create_invalid_mu);
    
    /* Prior sampling */
    RUN_TEST(prior_sampling_mean_variance);
    
    /* Transition sampling */
    RUN_TEST(transition_mean_reversion);
    RUN_TEST(transition_from_mean);
    
    /* Likelihood */
    RUN_TEST(likelihood_symmetry);
    RUN_TEST(likelihood_peak_at_zero);
    RUN_TEST(likelihood_monotonic_in_abs_r);
    RUN_TEST(likelihood_finite_for_reasonable_h);
    
    return test_summary();
}
