/**
 * @file ensemble_internal.h
 * @brief Internal structure definitions for ensemble.
 *
 * This header exposes the internal structures of ensemble_t and
 * ensemble_model_t for use by internal implementation files.
 *
 * DO NOT include this header in public API headers.
 *
 * Pattern: Standard C library practice for serialization/checkpointing.
 */

#ifndef SVMIX_ENSEMBLE_INTERNAL_H
#define SVMIX_ENSEMBLE_INTERNAL_H

#include "ensemble.h"
#include "model_sv_internal.h"
#include "../third_party/fastpf/src/fastpf_internal.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Per-model state in the ensemble.
 *
 * Each model contains:
 *   - fastpf_t: Particle filter instance (embedded for better locality)
 *   - sv_model_ctx_t*: SV model parameters and context
 *   - score: Accumulated log-likelihood score for weighting
 */
struct ensemble_model_t {
    fastpf_t pf;                      /**< Particle filter instance (embedded) */
    struct sv_model_ctx_t* sv_ctx;    /**< SV parameters (owned by ensemble) */
    double score;                     /**< Accumulated score S_i */
};

/**
 * @brief Ensemble of K SV models with mixture weighting.
 *
 * Internal structure containing:
 *   - K models (array of ensemble_model_t)
 *   - Current weights (normalized, size K)
 *   - Configuration (lambda, beta, epsilon, num_threads)
 *   - Timestep counter
 */
struct ensemble_t {
    size_t K;                        /**< Number of models */
    ensemble_model_t* models;        /**< Array of K models */
    double* weights;                 /**< Current weights w_i (size K) */
    ensemble_config_t config;        /**< Hyperparameters */
    size_t timestep;                 /**< Number of steps processed */
};

#ifdef __cplusplus
}
#endif

#endif /* SVMIX_ENSEMBLE_INTERNAL_H */
