/**
 * @file ensemble.c
 * @brief Ensemble implementation: K independent filters + weight manager.
 */

#include "ensemble.h"
#include "model_sv.h"
#include "../third_party/fastpf/src/fastpf_internal.h"  /* For stack allocation */

#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <alloca.h>

#ifdef _OPENMP
#include <omp.h>
#endif

/* ========================================================================
 * Internal ensemble structure
 * ======================================================================== */

struct ensemble_t {
    size_t K;                        /**< Number of models */
    ensemble_model_t* models;        /**< Array of K models */
    double* weights;                 /**< Current weights w_i (size K) */
    ensemble_config_t config;        /**< Hyperparameters */
    size_t timestep;                 /**< Number of steps processed */
};

/* ========================================================================
 * Helper: log-stable softmax with anti-starvation mixing
 * ======================================================================== */

/**
 * @brief Compute weights via log-stable softmax.
 *
 * Input: scores[i] (size K)
 * Output: weights[i] (size K), normalized to sum to 1
 *
 * Algorithm:
 *   a_i = beta * scores[i]
 *   m = max(a_i)
 *   w_i = exp(a_i - m) / sum_j exp(a_j - m)
 *   w_i = (1 - epsilon) * w_i + epsilon / K
 *
 * This avoids overflow/underflow and ensures w_i >= epsilon/K.
 */
static void compute_weights(
    const double* scores,
    size_t K,
    double beta,
    double epsilon,
    double* weights
) {
    /* Find max score for numerical stability */
    double max_score = -DBL_MAX;
    for (size_t i = 0; i < K; i++) {
        if (isfinite(scores[i]) && scores[i] > max_score) {
            max_score = scores[i];
        }
    }

    /* If all scores are non-finite, use uniform weights */
    if (!isfinite(max_score)) {
        for (size_t i = 0; i < K; i++) {
            weights[i] = 1.0 / (double)K;
        }
        return;
    }

    /* Compute exp(beta * (score - max_score)) and sum */
    double sum = 0.0;
    for (size_t i = 0; i < K; i++) {
        if (isfinite(scores[i])) {
            double a = beta * (scores[i] - max_score);
            /* Clamp to prevent underflow */
            if (a < -700.0) {
                weights[i] = 0.0;
            } else {
                weights[i] = exp(a);
            }
        } else {
            weights[i] = 0.0;
        }
        sum += weights[i];
    }

    /* Normalize */
    if (sum > 0.0) {
        for (size_t i = 0; i < K; i++) {
            weights[i] /= sum;
        }
    } else {
        /* All weights underflowed -> uniform */
        for (size_t i = 0; i < K; i++) {
            weights[i] = 1.0 / (double)K;
        }
    }

    /* Anti-starvation mixing: w_i = (1 - eps) * w_i + eps / K */
    double uniform_weight = epsilon / (double)K;
    for (size_t i = 0; i < K; i++) {
        weights[i] = (1.0 - epsilon) * weights[i] + uniform_weight;
    }
}

/* ========================================================================
 * Helper: validate configuration
 * ======================================================================== */

static bool validate_config(const ensemble_config_t* config) {
    if (!config) return false;
    
    /* lambda: (0, 1] */
    if (!isfinite(config->lambda) || config->lambda <= 0.0 || config->lambda > 1.0) {
        return false;
    }
    
    /* beta: > 0 */
    if (!isfinite(config->beta) || config->beta <= 0.0) {
        return false;
    }
    
    /* epsilon: [0, 1) */
    if (!isfinite(config->epsilon) || config->epsilon < 0.0 || config->epsilon >= 1.0) {
        return false;
    }
    
    /* num_threads: >= 0 */
    if (config->num_threads < 0) {
        return false;
    }
    
    return true;
}

/* ========================================================================
 * API: ensemble_create
 * ======================================================================== */

ensemble_t* ensemble_create(
    size_t K,
    const sv_params_t* sv_params,
    size_t num_particles,
    const unsigned long* seeds,
    const ensemble_config_t* config
) {
    /* Validate inputs */
    if (K == 0 || !sv_params || !seeds || !config) {
        return NULL;
    }
    
    if (num_particles == 0) {
        return NULL;
    }
    
    if (!validate_config(config)) {
        return NULL;
    }
    
    /* Allocate ensemble */
    ensemble_t* ens = (ensemble_t*)malloc(sizeof(ensemble_t));
    if (!ens) return NULL;
    
    ens->K = K;
    ens->timestep = 0;
    ens->config = *config;
    
    /* Allocate models array */
    ens->models = (ensemble_model_t*)calloc(K, sizeof(ensemble_model_t));
    if (!ens->models) {
        free(ens);
        return NULL;
    }
    
    /* Allocate weights array */
    ens->weights = (double*)malloc(K * sizeof(double));
    if (!ens->weights) {
        free(ens->models);
        free(ens);
        return NULL;
    }
    
    /* Initialize weights to uniform */
    for (size_t i = 0; i < K; i++) {
        ens->weights[i] = 1.0 / (double)K;
    }
    
    /* Configure OpenMP if enabled */
#ifdef _OPENMP
    if (config->num_threads > 0) {
        omp_set_num_threads(config->num_threads);
    }
#endif
    
    /* Initialize each model */
    for (size_t i = 0; i < K; i++) {
        ensemble_model_t* model = &ens->models[i];
        
        /* Allocate particle filter */
        model->pf = (fastpf_t*)malloc(sizeof(fastpf_t));
        if (!model->pf) {
            /* Clean up previous models */
            for (size_t j = 0; j < i; j++) {
                free(ens->models[j].pf);
                free(ens->models[j].sv_ctx);
            }
            free(ens->weights);
            free(ens->models);
            free(ens);
            return NULL;
        }
        
        /* Create SV model context */
        fastpf_model_t fastpf_model;
        model->sv_ctx = sv_model_create(
            sv_params[i].mu_h,
            sv_params[i].phi_h,
            sv_params[i].sigma_h,
            sv_params[i].nu,
            &fastpf_model
        );
        
        if (!model->sv_ctx) {
            /* Validation failed - clean up and return */
            free(model->pf);
            for (size_t j = 0; j < i; j++) {
                free(ens->models[j].pf);
                free(ens->models[j].sv_ctx);
            }
            free(ens->weights);
            free(ens->models);
            free(ens);
            return NULL;
        }
        
        /* Initialize particle filter */
        fastpf_cfg_t pf_config;
        fastpf_cfg_init(&pf_config, num_particles, sizeof(double));  /* 1D state: h_t */
        pf_config.rng_seed = seeds[i];
        pf_config.resample_threshold = 0.5;
        pf_config.num_threads = config->num_threads;
        
        if (fastpf_init(model->pf, &pf_config, &fastpf_model) != 0) {
            /* PF init failed - clean up */
            free(model->pf);
            free(model->sv_ctx);
            for (size_t j = 0; j < i; j++) {
                free(ens->models[j].pf);
                free(ens->models[j].sv_ctx);
            }
            free(ens->weights);
            free(ens->models);
            free(ens);
            return NULL;
        }
        
        /* Initialize score to 0 */
        model->score = 0.0;
    }
    
    return ens;
}

/* ========================================================================
 * API: ensemble_free
 * ======================================================================== */

void ensemble_free(ensemble_t* ens) {
    if (!ens) return;
    
    if (ens->models) {
        for (size_t i = 0; i < ens->K; i++) {
            /* Free particle filter */
            if (ens->models[i].pf) {
                free(ens->models[i].pf);
            }
            /* Free SV context */
            if (ens->models[i].sv_ctx) {
                free(ens->models[i].sv_ctx);
            }
        }
        free(ens->models);
    }
    
    if (ens->weights) {
        free(ens->weights);
    }
    
    free(ens);
}

/* ========================================================================
 * API: ensemble_step
 * ======================================================================== */

int ensemble_step(ensemble_t* ens, double observation) {
    if (!ens || !ens->models || !ens->weights) {
        return -1;
    }
    
    if (!isfinite(observation)) {
        return -1;
    }
    
    /* Step each model through the observation */
#ifdef _OPENMP
    #pragma omp parallel for
#endif
    for (size_t i = 0; i < ens->K; i++) {
        ensemble_model_t* model = &ens->models[i];
        
        /* Run particle filter step */
        if (fastpf_step(model->pf, &observation) != 0) {
            /* Step failed - set score to -inf to down-weight this model */
            model->score = -DBL_MAX;
            continue;
        }
        
        /* Extract log_norm_const from diagnostics */
        const fastpf_diagnostics_t* diag = fastpf_get_diagnostics(model->pf);
        
        /* Update score with exponential forgetting */
        model->score = ens->config.lambda * model->score + diag->log_norm_const;
    }
    
    /* Collect scores for weight computation (serial) */
    double* scores = (double*)alloca(ens->K * sizeof(double));
    for (size_t i = 0; i < ens->K; i++) {
        scores[i] = ens->models[i].score;
    }
    
    /* Recompute weights via log-stable softmax */
    compute_weights(
        scores,
        ens->K,
        ens->config.beta,
        ens->config.epsilon,
        ens->weights
    );
    
    ens->timestep++;
    return 0;
}

/* ========================================================================
 * API: ensemble_get_weights
 * ======================================================================== */

size_t ensemble_get_weights(const ensemble_t* ens, double* weights_out) {
    if (!ens || !weights_out) return 0;
    
    memcpy(weights_out, ens->weights, ens->K * sizeof(double));
    return ens->K;
}

/* ========================================================================
 * API: ensemble_get_scores
 * ======================================================================== */

size_t ensemble_get_scores(const ensemble_t* ens, double* scores_out) {
    if (!ens || !scores_out) return 0;
    
    for (size_t i = 0; i < ens->K; i++) {
        scores_out[i] = ens->models[i].score;
    }
    return ens->K;
}

/* ========================================================================
 * API: ensemble_get_belief
 * ======================================================================== */

ensemble_belief_t ensemble_get_belief(const ensemble_t* ens) {
    ensemble_belief_t belief = {0};
    belief.valid = false;
    
    if (!ens || !ens->models || !ens->weights) {
        return belief;
    }
    
    /* Compute weighted mixture of posterior means and variances */
    double sum_w_mean_h = 0.0;
    double sum_w_var_h = 0.0;
    double sum_w_mean_h_sq = 0.0;  /* For mixture variance */
    
    for (size_t i = 0; i < ens->K; i++) {
        const ensemble_model_t* model = &ens->models[i];
        const double w = ens->weights[i];
        
        /* Get particle weights from PF */
        const double* pf_weights = fastpf_get_weights(model->pf);
        
        if (!pf_weights) {
            continue;
        }
        
        /* Compute posterior mean of h for this model */
        double mean_h_i = 0.0;
        size_t N = model->pf->cfg.n_particles;
        for (size_t j = 0; j < N; j++) {
            const double* particle = (const double*)fastpf_get_particle(model->pf, j);
            mean_h_i += pf_weights[j] * particle[0];
        }
        
        /* Compute posterior variance of h for this model */
        double var_h_i = 0.0;
        for (size_t j = 0; j < N; j++) {
            const double* particle = (const double*)fastpf_get_particle(model->pf, j);
            double diff = particle[0] - mean_h_i;
            var_h_i += pf_weights[j] * diff * diff;
        }
        
        /* Accumulate for mixture */
        sum_w_mean_h += w * mean_h_i;
        sum_w_var_h += w * var_h_i;
        sum_w_mean_h_sq += w * mean_h_i * mean_h_i;
    }
    
    /* Mixture mean */
    belief.mean_h = sum_w_mean_h;
    
    /* Mixture variance (law of total variance) */
    /* Var[H] = E[Var[H|M]] + Var[E[H|M]] */
    /*        = sum_w_var_h + (sum_w_mean_h_sq - (sum_w_mean_h)^2) */
    belief.var_h = sum_w_var_h + (sum_w_mean_h_sq - sum_w_mean_h * sum_w_mean_h);
    
    /* Approximate volatility */
    belief.mean_sigma = exp(belief.mean_h / 2.0);
    
    /* Validate */
    belief.valid = isfinite(belief.mean_h) && 
                   isfinite(belief.var_h) && 
                   isfinite(belief.mean_sigma) &&
                   belief.var_h >= 0.0;
    
    return belief;
}

/* ========================================================================
 * API: ensemble_get_config
 * ======================================================================== */

const ensemble_config_t* ensemble_get_config(const ensemble_t* ens) {
    if (!ens) return NULL;
    return &ens->config;
}

/* ========================================================================
 * API: ensemble_get_num_models
 * ======================================================================== */

size_t ensemble_get_num_models(const ensemble_t* ens) {
    if (!ens) return 0;
    return ens->K;
}
