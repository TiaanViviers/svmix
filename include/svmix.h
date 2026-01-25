/**
 * @file svmix.h
 * @brief Public C API for svmix: Ensemble stochastic volatility belief engine.
 *
 * svmix provides online Bayesian inference for stochastic volatility models
 * using a mixture of particle filters with adaptive model weighting.
 *
 * Architecture:
 *   - K independent SV models (each with different parameters)
 *   - Each model runs a particle filter (N particles)
 *   - Exponential forgetting + tempered softmax for model weights
 *   - Mixture belief = weighted average of per-model posteriors
 *
 * Key guarantees:
 *   - Deterministic: same seeds + same data = bitwise-identical results
 *   - Checkpointing: save/load state for crash recovery
 *   - Thread-safe creation (but not concurrent step on same instance)
 *
 * Version: 1.0.0
 * Spec: VOL (basic SV with Student-t shocks, AR(1) log-variance)
 */

#ifndef SVMIX_H
#define SVMIX_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ========================================================================
 * Version Information
 * ======================================================================== */

#define SVMIX_VERSION_MAJOR 1
#define SVMIX_VERSION_MINOR 0
#define SVMIX_VERSION_PATCH 0

/**
 * @brief Get svmix version string.
 * @return Version string in format "major.minor.patch"
 */
const char* svmix_version(void);

/* ========================================================================
 * Opaque Handle
 * ======================================================================== */

/**
 * @brief Opaque svmix instance handle.
 *
 * Encapsulates:
 *   - K SV models with particle filters
 *   - Current model weights
 *   - Score accumulators
 *   - Configuration and state
 *
 * Obtain via svmix_create() or svmix_load_checkpoint().
 * Free with svmix_free().
 */
typedef struct svmix_t svmix_t;

/* ========================================================================
 * Error Codes
 * ======================================================================== */

/**
 * @brief Status/error codes returned by svmix API functions.
 *
 * Convention: 0 = success, <0 = error.
 */
typedef enum {
    SVMIX_OK = 0,                    /**< Success */
    SVMIX_ERR_NULL_POINTER = -1,     /**< NULL pointer passed where not allowed */
    SVMIX_ERR_INVALID_PARAM = -2,    /**< Invalid parameter value */
    SVMIX_ERR_ALLOC_FAILED = -3,     /**< Memory allocation failed */
    SVMIX_ERR_FILE_IO = -4,          /**< File I/O error (open/read/write) */
    SVMIX_ERR_CHECKPOINT_CORRUPT = -5, /**< Checkpoint file corrupted or invalid */
    SVMIX_ERR_VERSION_MISMATCH = -6, /**< Checkpoint version incompatible */
    SVMIX_ERR_INTERNAL = -99         /**< Internal error (should not happen) */
} svmix_status_t;

/**
 * @brief Get human-readable error message for status code.
 * @param status Status code
 * @return Error message string (never NULL)
 */
const char* svmix_status_string(svmix_status_t status);

/* ========================================================================
 * Configuration Structures (POD - Plain Old Data)
 * ======================================================================== */

/**
 * @brief Specification ID for model type.
 *
 * Future versions may support additional specs (VOL_DRIFT, etc.).
 * v1 only supports VOL.
 */
typedef enum {
    SVMIX_SPEC_VOL = 1  /**< Basic SV: AR(1) log-variance, Student-t shocks */
} svmix_spec_t;

/**
 * @brief SV model parameters (VOL spec).
 *
 * Model:
 *   h_t = mu_h + phi_h * (h_{t-1} - mu_h) + sigma_h * eta_t
 *   r_t ~ Student-t(nu, 0, exp(h_t/2))
 *
 * Constraints:
 *   - 0 < phi_h <= 0.9999 (stationarity + numerical safety)
 *   - sigma_h > 0
 *   - nu > 2 (finite variance)
 *   - All values must be finite
 */
typedef struct {
    double mu_h;    /**< Long-run mean of log-variance */
    double phi_h;   /**< AR(1) persistence (0 < phi_h <= 0.9999) */
    double sigma_h; /**< Process noise std dev (> 0) */
    double nu;      /**< Student-t degrees of freedom (> 2) */
} svmix_sv_params_t;

/**
 * @brief Ensemble weighting configuration.
 *
 * Hyperparameters:
 *   - lambda: Exponential forgetting factor for scores
 *   - beta: Softmax temperature (higher = more aggressive weighting)
 *   - epsilon: Anti-starvation mixing (each model gets at least eps/K weight)
 *   - num_threads: OpenMP thread count (0 = auto, >0 = explicit)
 *
 * Recommended defaults:
 *   - lambda: 0.99 - 0.999 (minute-frequency data)
 *   - beta: 0.5 - 1.0 (prevents premature dominance)
 *   - epsilon: 0.01 - 0.05 (1-5% floor per model)
 *   - num_threads: 0 (auto) or explicitly set for reproducibility
 */
typedef struct {
    double lambda;   /**< Forgetting factor (0 < lambda <= 1) */
    double beta;     /**< Softmax temperature (> 0) */
    double epsilon;  /**< Anti-starvation weight (0 <= epsilon < 1) */
    int num_threads; /**< OpenMP threads: 0=auto, >0=explicit (requires OPENMP=1) */
} svmix_ensemble_cfg_t;

/**
 * @brief Top-level svmix configuration.
 *
 * Specifies:
 *   - Number of models K
 *   - Particles per model N
 *   - Model specification (VOL for v1)
 *   - Ensemble hyperparameters
 */
typedef struct {
    size_t num_models;                /**< K: number of models (> 0) */
    size_t num_particles;             /**< N: particles per model (> 0) */
    svmix_spec_t spec;                /**< Model specification (SVMIX_SPEC_VOL) */
    svmix_ensemble_cfg_t ensemble;    /**< Ensemble hyperparameters */
} svmix_cfg_t;

/**
 * @brief Mixture posterior belief summary.
 *
 * Weighted average of per-model posterior statistics:
 *   mean_h = sum_i w_i * E[h|model_i]
 *   var_h = sum_i w_i * Var[h|model_i] + sum_i w_i * (E[h|model_i] - mean_h)^2
 *   mean_sigma = exp(mean_h / 2)
 *
 * If valid=0, computation failed (e.g., all models returned NaN).
 */
typedef struct {
    double mean_h;      /**< Mixture posterior mean of log-variance h */
    double var_h;       /**< Mixture posterior variance of h */
    double mean_sigma;  /**< Approximate volatility: exp(mean_h/2) */
    int valid;          /**< 1 if valid, 0 if computation failed */
} svmix_belief_t;

/* ========================================================================
 * Core API
 * ======================================================================== */

/**
 * @brief Create a new svmix instance.
 *
 * Initializes K models with given parameters and seeds.
 * Each model gets an independent particle filter (N particles each).
 * Initial weights are uniform (1/K).
 *
 * @param cfg Configuration (not NULL)
 * @param models Array of K SV parameter sets (not NULL)
 * @param seeds Array of K RNG seeds for reproducibility (not NULL)
 * @return svmix instance on success, NULL on error
 *
 * Error conditions:
 *   - cfg, models, or seeds is NULL
 *   - cfg->num_models or cfg->num_particles is 0
 *   - Any model parameters fail validation
 *   - Memory allocation fails
 *   - cfg->spec is not SVMIX_SPEC_VOL
 *
 * Example:
 *   svmix_cfg_t cfg = {
 *       .num_models = 3,
 *       .num_particles = 1000,
 *       .spec = SVMIX_SPEC_VOL,
 *       .ensemble = {.lambda=0.99, .beta=1.0, .epsilon=0.05, .num_threads=0}
 *   };
 *   svmix_sv_params_t params[3] = { ... };
 *   unsigned long seeds[3] = {42, 43, 44};
 *   svmix_t* sv = svmix_create(&cfg, params, seeds);
 *   if (!sv) { // handle error }
 */
svmix_t* svmix_create(
    const svmix_cfg_t* cfg,
    const svmix_sv_params_t* models,
    const unsigned long* seeds
);

/**
 * @brief Free svmix instance and all resources.
 *
 * Safe to call with NULL pointer (no-op).
 * After calling, the handle is invalid.
 *
 * @param svmix Instance to free (may be NULL)
 */
void svmix_free(svmix_t* svmix);

/**
 * @brief Process one observation (1-minute return).
 *
 * Steps all K models through the observation:
 *   1. Each model runs fastpf_step() with the return
 *   2. Scores updated: S_i = lambda * S_i + log_norm_const
 *   3. Weights recomputed via tempered softmax + anti-starvation
 *
 * @param svmix Instance (not NULL)
 * @param observation Observed return r_t (must be finite)
 * @return SVMIX_OK on success, error code on failure
 *
 * Error conditions:
 *   - svmix is NULL → SVMIX_ERR_NULL_POINTER
 *   - observation is NaN or Inf → SVMIX_ERR_INVALID_PARAM
 *   - Internal particle filter failure → SVMIX_ERR_INTERNAL
 *
 * Thread safety: Not safe to call concurrently on same instance.
 */
svmix_status_t svmix_step(svmix_t* svmix, double observation);

/**
 * @brief Get current mixture belief (posterior summary).
 *
 * Computes weighted average of per-model posterior means/variances.
 * Fast O(K*N) operation (N = particles per model).
 *
 * @param svmix Instance (not NULL)
 * @param belief Output belief structure (not NULL)
 * @return SVMIX_OK on success, error code on failure
 *
 * Error conditions:
 *   - svmix or belief is NULL → SVMIX_ERR_NULL_POINTER
 *
 * If computation fails (e.g., all models return NaN), belief->valid = 0.
 */
svmix_status_t svmix_get_belief(const svmix_t* svmix, svmix_belief_t* belief);

/**
 * @brief Get current model weights.
 *
 * Copies K weights to provided array.
 * Weights sum to 1 and each w_i >= epsilon/K.
 *
 * @param svmix Instance (not NULL)
 * @param weights Output array of size K (not NULL)
 * @param K Expected number of models (must match cfg->num_models)
 * @return SVMIX_OK on success, error code on failure
 *
 * Error conditions:
 *   - svmix or weights is NULL → SVMIX_ERR_NULL_POINTER
 *   - K doesn't match actual model count → SVMIX_ERR_INVALID_PARAM
 *
 * Example:
 *   double weights[3];
 *   if (svmix_get_weights(sv, weights, 3) == SVMIX_OK) {
 *       // weights[0], weights[1], weights[2] are valid
 *   }
 */
svmix_status_t svmix_get_weights(const svmix_t* svmix, double* weights, size_t K);

/**
 * @brief Get number of models in ensemble.
 *
 * @param svmix Instance (not NULL)
 * @return K (number of models), or 0 if svmix is NULL
 */
size_t svmix_get_num_models(const svmix_t* svmix);

/**
 * @brief Get number of timesteps processed.
 *
 * Increments by 1 each time svmix_step() is called successfully.
 *
 * @param svmix Instance (not NULL)
 * @return Number of steps, or 0 if svmix is NULL
 */
size_t svmix_get_timestep(const svmix_t* svmix);

/* ========================================================================
 * Checkpointing
 * ======================================================================== */

/**
 * @brief Save svmix state to checkpoint file.
 *
 * Serializes complete state:
 *   - Configuration (K, N, hyperparameters)
 *   - Model parameters for all K models
 *   - Particle filter states (particles, weights, RNG states)
 *   - Scores and weights
 *   - Timestep counter
 *
 * File format: Binary, includes magic header + version for validation.
 *
 * @param svmix Instance (not NULL)
 * @param filepath Path to checkpoint file (will be created/overwritten)
 * @return SVMIX_OK on success, error code on failure
 *
 * Error conditions:
 *   - svmix or filepath is NULL → SVMIX_ERR_NULL_POINTER
 *   - Cannot open file for writing → SVMIX_ERR_FILE_IO
 *   - Write fails (disk full, etc.) → SVMIX_ERR_FILE_IO
 *
 * Determinism guarantee:
 *   save(A) → load → continue = identical to A continuing directly
 */
svmix_status_t svmix_save_checkpoint(const svmix_t* svmix, const char* filepath);

/**
 * @brief Load svmix state from checkpoint file.
 *
 * Reconstructs complete state from checkpoint.
 * Validates magic header, version, and configuration.
 *
 * @param filepath Path to checkpoint file
 * @param status Output status (may be NULL)
 * @return svmix instance on success, NULL on error
 *
 * Error conditions:
 *   - filepath is NULL → status = SVMIX_ERR_NULL_POINTER, returns NULL
 *   - Cannot open file → status = SVMIX_ERR_FILE_IO
 *   - Corrupted header → status = SVMIX_ERR_CHECKPOINT_CORRUPT
 *   - Version mismatch → status = SVMIX_ERR_VERSION_MISMATCH
 *   - Memory allocation fails → status = SVMIX_ERR_ALLOC_FAILED
 *
 * If status is NULL, error details are lost (but NULL still returned on error).
 *
 * Example:
 *   svmix_status_t status;
 *   svmix_t* sv = svmix_load_checkpoint("checkpoint.dat", &status);
 *   if (!sv) {
 *       fprintf(stderr, "Load failed: %s\n", svmix_status_string(status));
 *   }
 */
svmix_t* svmix_load_checkpoint(const char* filepath, svmix_status_t* status);

#ifdef __cplusplus
}
#endif

#endif /* SVMIX_H */
