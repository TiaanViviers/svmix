/**
 * @file model_sv.c
 * @brief Stochastic Volatility model implementation for fastpf callbacks.
 *
 * Implements the SV layer that defines:
 * - Latent state: h_t = log-variance
 * - Transition: AR(1) process
 * - Observation: Student-t likelihood
 */

#include "model_sv.h"
#include "model_sv_internal.h"
#include "util.h"
#include "../third_party/fastpf/include/fastpf.h"
#include <stdlib.h>
#include <math.h>
#include <string.h>

/* ========================================================================
 * fastpf Callback: Prior Sampling
 * ======================================================================== */

/**
 * @brief Sample initial log-variance from stationary AR(1) distribution.
 *
 * For AR(1) process: h_t = mu + phi * (h_{t-1} - mu) + sigma * eta_t,
 * the stationary distribution is:
 *   h_0 ~ N(mu, sigma^2 / (1 - phi^2))
 *
 * @param ctx Pointer to sv_model_ctx_t.
 * @param x0_out Output buffer (sizeof(double)) for sampled h_0.
 * @param rng fastpf RNG state.
 */
static void sv_prior_sample(void* ctx, void* x0_out, fastpf_rng_t* rng) {
    const sv_model_ctx_t* model = (const sv_model_ctx_t*)ctx;
    double* h0 = (double*)x0_out;
    
    /* Stationary variance: sigma_h^2 / (1 - phi_h^2) */
    double stationary_var = (model->sigma_h * model->sigma_h) / (1.0 - model->phi_h * model->phi_h);
    double stationary_sd = sqrt(stationary_var);
    
    /* Sample: h_0 = mu_h + stationary_sd * N(0,1) */
    *h0 = model->mu_h + stationary_sd * fastpf_rng_normal(rng);
}

/* ========================================================================
 * fastpf Callback: Transition Sampling
 * ======================================================================== */

/**
 * @brief Sample next log-variance from AR(1) transition model.
 *
 * Implements: h_t = mu_h + phi_h * (h_{t-1} - mu_h) + sigma_h * eta_t
 * where eta_t ~ N(0, 1).
 *
 * @param ctx Pointer to sv_model_ctx_t.
 * @param x_prev Previous state (sizeof(double)) containing h_{t-1}.
 * @param x_out Output buffer (sizeof(double)) for h_t.
 * @param rng fastpf RNG state.
 */
static void sv_transition_sample(void* ctx, const void* x_prev, void* x_out, fastpf_rng_t* rng) {
    const sv_model_ctx_t* model = (const sv_model_ctx_t*)ctx;
    const double* h_prev = (const double*)x_prev;
    double* h_next = (double*)x_out;
    
    /* AR(1): h_t = mu + phi * (h_{t-1} - mu) + sigma * eta_t */
    double mean_reversion = model->mu_h + model->phi_h * (*h_prev - model->mu_h);
    double noise = model->sigma_h * fastpf_rng_normal(rng);
    
    *h_next = mean_reversion + noise;
}

/* ========================================================================
 * fastpf Callback: Log-Likelihood
 * ======================================================================== */

/**
 * @brief Compute log-likelihood of observation given log-variance.
 *
 * Observation model: r_t ~ StudentT(nu, loc=0, scale=exp(h_t/2))
 *
 * @param ctx Pointer to sv_model_ctx_t.
 * @param x Current state (sizeof(double)) containing h_t.
 * @param y Observation (sizeof(double)) containing r_t (return).
 * @return Log-likelihood log p(r_t | h_t).
 */
static double sv_log_likelihood(void* ctx, const void* x, const void* y) {
    const sv_model_ctx_t* model = (const sv_model_ctx_t*)ctx;
    const double* h = (const double*)x;
    const double* r = (const double*)y;
    
    /* Delegate to tested Student-t implementation */
    return svmix_t_logpdf_logvar(*r, *h, model->nu);
}

/* ========================================================================
 * Public API: Model Creation
 * ======================================================================== */

/**
 * @brief Create and configure an SV model for use with fastpf.
 *
 * Allocates and initializes an sv_model_ctx_t with given parameters,
 * then populates a fastpf_model_t with callbacks.
 *
 * @param mu_h Long-run mean of log-variance.
 * @param phi_h Persistence in (0, 1).
 * @param sigma_h Process noise (> 0).
 * @param nu Student-t degrees of freedom (> 2).
 * @param model_out Output: fastpf_model_t to populate with callbacks.
 * @return Pointer to allocated sv_model_ctx_t (caller must free), or NULL on error.
 *
 * Example usage:
 *   fastpf_model_t model;
 *   sv_model_ctx_t* ctx = sv_model_create(-13.8, 0.98, 0.15, 7.0, &model);
 *   if (ctx) {
 *       // Use model with fastpf_init()
 *       // ...
 *       free(ctx);
 *   }
 */
sv_model_ctx_t* sv_model_create(double mu_h, double phi_h, double sigma_h, double nu,
                                 fastpf_model_t* model_out) {
    /* Validate parameters */
    if (!model_out) {
        return NULL;  /* Need valid output struct */
    }
    if (phi_h <= 0.0 || phi_h >= 1.0) {
        return NULL;  /* phi must be in (0, 1) for stationarity */
    }
    if (phi_h > 0.999) {
        return NULL;  /* phi too close to 1 causes stationary variance blow-up */
    }
    if (sigma_h <= 0.0) {
        return NULL;  /* Process noise must be positive */
    }
    if (nu <= 2.0) {
        return NULL;  /* Need finite variance */
    }
    if (!isfinite(mu_h) || !isfinite(sigma_h) || !isfinite(nu)) {
        return NULL;
    }
    
    /* Allocate context */
    sv_model_ctx_t* ctx = (sv_model_ctx_t*)malloc(sizeof(sv_model_ctx_t));
    if (!ctx) {
        return NULL;
    }
    
    /* Initialize parameters */
    ctx->mu_h = mu_h;
    ctx->phi_h = phi_h;
    ctx->sigma_h = sigma_h;
    ctx->nu = nu;
    
    /* Populate fastpf model structure */
    model_out->ctx = ctx;
    model_out->prior_sample = sv_prior_sample;
    model_out->transition_sample = sv_transition_sample;
    model_out->log_likelihood = sv_log_likelihood;
    model_out->rejuvenate = NULL;  /* No rejuvenation in v1 */
    
    return ctx;
}
