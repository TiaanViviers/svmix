/**
 * @file ensemble.h
 * @brief Ensemble of SV models with exponential forgetting and tempered softmax.
 *
 * Architecture:
 *   - K independent SV models, each with its own fastpf_t instance
 *   - Each model accumulates a score S_i via exponential forgetting
 *   - Weights w_i computed via tempered softmax with anti-starvation mixing
 *   - Mixture belief computed as weighted average of model posteriors
 *
 * Key design:
 *   - Ensemble is model-agnostic: only knows fastpf_step() and log_norm_const
 *   - Per-model state is self-contained for easy extension (VOL_DRIFT, etc.)
 *   - Numerical stability: all weight computations in log domain
 *   - Determinism: same seeds produce bitwise-identical trajectories
 */

#ifndef SVMIX_ENSEMBLE_H
#define SVMIX_ENSEMBLE_H

#include <stddef.h>
#include <stdbool.h>

#include "fastpf.h"
#include "../third_party/fastpf/src/fastpf_internal.h"

/* Forward declarations */
struct sv_model_ctx_t;

/* ========================================================================
 * SV parameter structure
 * ======================================================================== */

/**
 * @brief SV model parameters.
 */
typedef struct {
    double mu_h;      /**< Long-run mean of log-variance */
    double phi_h;     /**< Persistence (0 < phi_h < 1) */
    double sigma_h;   /**< Process noise (> 0) */
    double nu;        /**< Student-t degrees of freedom (> 2) */
} sv_params_t;

#ifdef __cplusplus
extern "C" {
#endif

/* ========================================================================
 * Ensemble configuration
 * ======================================================================== */

/**
 * @brief Configuration for ensemble creation.
 *
 * Hyperparameters:
 *   - lambda: Exponential forgetting factor (0 < lambda <= 1)
 *             Recommended: 0.99 - 0.999 for minute-frequency updates
 *             Higher = slower adaptation, lower = more reactive
 *
 *   - beta: Softmax temperature (beta > 0)
 *           Recommended: 0.5 - 1.0 to prevent premature dominance
 *           Higher = more aggressive weighting
 *
 *   - epsilon: Anti-starvation mixing weight (0 <= epsilon < 1)
 *              Each model gets at least epsilon/K weight
 *              Recommended: 0.01 - 0.05
 *
 *   - num_threads: OpenMP thread count (if compiled with -fopenmp)
 *                  0 = use OMP_NUM_THREADS or system default
 *                  >0 = explicitly set thread count
 *                  Ignored if OpenMP not enabled
 */
typedef struct {
    double lambda;        /**< Forgetting factor (0 < lambda <= 1) */
    double beta;          /**< Softmax temperature (beta > 0) */
    double epsilon;       /**< Anti-starvation mix weight (0 <= epsilon < 1) */
    int num_threads;      /**< OpenMP threads (0 = auto, >0 = explicit) */
} ensemble_config_t;

/* ========================================================================
 * Per-model state
 * ======================================================================== */

/**
 * @brief State for a single model in the ensemble.
 *
 * Self-contained: each model owns its PF instance, parameter context,
 * and score accumulator. This makes it easy to add new model types
 * (VOL_DRIFT, etc.) without changing ensemble logic.
 * 
 * NOTE: fastpf_t is embedded directly (not a pointer) for simpler
 * memory management and better cache locality.
 */
typedef struct {
    fastpf_t pf;                      /**< Particle filter instance (embedded) */
    struct sv_model_ctx_t* sv_ctx;    /**< SV parameters (owned by ensemble) */
    double score;                     /**< Accumulated score S_i */
} ensemble_model_t;

/* ========================================================================
 * Ensemble state
 * ======================================================================== */

/**
 * @brief Ensemble of K SV models with mixture weighting.
 *
 * Opaque type: internal layout may change.
 */
typedef struct ensemble_t ensemble_t;

/* ========================================================================
 * Belief summary
 * ======================================================================== */

/**
 * @brief Mixture posterior belief summary.
 *
 * Weighted average of per-model posterior moments.
 */
typedef struct {
    double mean_h;        /**< Mixture posterior mean of log-variance */
    double var_h;         /**< Mixture posterior variance of log-variance */
    double mean_sigma;    /**< exp(mean_h / 2): approx volatility */
    bool valid;           /**< False if computation failed (e.g., all NaN) */
} ensemble_belief_t;

/* ========================================================================
 * API
 * ======================================================================== */

/**
 * @brief Create an ensemble with K models.
 *
 * Each model is initialized with:
 *   - SV parameters from sv_params[i]
 *   - Particle filter with num_particles
 *   - RNG seeded with seeds[i]
 *   - Initial score = 0.0
 *
 * @param K Number of models (K > 0)
 * @param sv_params Array of K parameter sets (not NULL)
 * @param num_particles Particle count per model (> 0)
 * @param seeds Array of K RNG seeds (not NULL)
 * @param config Ensemble hyperparameters (not NULL)
 * @return Ensemble instance, or NULL on error
 *
 * Error conditions:
 *   - K <= 0
 *   - sv_params or seeds is NULL
 *   - Any sv_params[i] fails validation (invalid phi_h, sigma_h, nu, mu_h)
 *   - num_particles <= 0
 *   - config is NULL or contains invalid hyperparameters
 */
ensemble_t* ensemble_create(
    size_t K,
    const sv_params_t* sv_params,
    size_t num_particles,
    const unsigned long* seeds,
    const ensemble_config_t* config
);

/**
 * @brief Free ensemble and all owned resources.
 *
 * @param ens Ensemble instance (NULL-safe)
 */
void ensemble_free(ensemble_t* ens);

/**
 * @brief Process one observation through all models.
 *
 * For each model i:
 *   1. fastpf_step(&model[i].pf, &observation)
 *   2. S_i = lambda * S_i + log_norm_const
 *
 * Then recompute weights via log-stable softmax:
 *   a_i = beta * S_i
 *   m = max(a_i)
 *   w_i = exp(a_i - m) / sum_j exp(a_j - m)
 *   w_i = (1 - epsilon) * w_i + epsilon / K
 *
 * @param ens Ensemble instance (not NULL)
 * @param observation Observed return r_t
 * @return 0 on success, -1 on error
 */
int ensemble_step(ensemble_t* ens, double observation);

/**
 * @brief Get current model weights.
 *
 * @param ens Ensemble instance (not NULL)
 * @param weights_out Output array of size K (not NULL)
 * @return Number of models K, or 0 on error
 */
size_t ensemble_get_weights(const ensemble_t* ens, double* weights_out);

/**
 * @brief Get current model scores.
 *
 * @param ens Ensemble instance (not NULL)
 * @param scores_out Output array of size K (not NULL)
 * @return Number of models K, or 0 on error
 */
size_t ensemble_get_scores(const ensemble_t* ens, double* scores_out);

/**
 * @brief Compute mixture posterior belief.
 *
 * For each model i:
 *   - Compute posterior mean/var of h from particles
 *   - Aggregate via weighted average using w_i
 *
 * @param ens Ensemble instance (not NULL)
 * @return Belief summary (valid=false on error)
 */
ensemble_belief_t ensemble_get_belief(const ensemble_t* ens);

/**
 * @brief Get ensemble configuration (read-only).
 *
 * @param ens Ensemble instance (not NULL)
 * @return Config pointer (NULL on error)
 */
const ensemble_config_t* ensemble_get_config(const ensemble_t* ens);

/**
 * @brief Get number of models in ensemble.
 *
 * @param ens Ensemble instance (not NULL)
 * @return K, or 0 on error
 */
size_t ensemble_get_num_models(const ensemble_t* ens);

#ifdef __cplusplus
}
#endif

#endif /* SVMIX_ENSEMBLE_H */
