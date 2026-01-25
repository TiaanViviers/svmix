/**
 * @file svmix.c
 * @brief Implementation of public svmix API (thin orchestrator).
 *
 * This layer:
 *   - Validates public API inputs
 *   - Translates between public types and internal ensemble types
 *   - Manages ownership and cleanup
 *   - Provides stable ABI for shared library
 *
 * Design: Minimal logic here - forward to ensemble layer.
 */

#include "svmix.h"
#include "svmix_internal.h"
#include "ensemble.h"

#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>

/* ========================================================================
 * Version
 * ======================================================================== */

const char* svmix_version(void) {
    return "1.0.0";
}

/* ========================================================================
 * Error strings
 * ======================================================================== */

const char* svmix_status_string(svmix_status_t status) {
    switch (status) {
        case SVMIX_OK:
            return "Success";
        case SVMIX_ERR_NULL_POINTER:
            return "NULL pointer argument";
        case SVMIX_ERR_INVALID_PARAM:
            return "Invalid parameter value";
        case SVMIX_ERR_ALLOC_FAILED:
            return "Memory allocation failed";
        case SVMIX_ERR_FILE_IO:
            return "File I/O error";
        case SVMIX_ERR_CHECKPOINT_CORRUPT:
            return "Checkpoint file corrupted";
        case SVMIX_ERR_VERSION_MISMATCH:
            return "Checkpoint version incompatible";
        case SVMIX_ERR_INTERNAL:
            return "Internal error";
        default:
            return "Unknown error";
    }
}

/* ========================================================================
 * Validation helpers
 * ======================================================================== */

static int validate_sv_params(const svmix_sv_params_t* params) {
    if (!isfinite(params->mu_h)) return 0;
    if (!isfinite(params->phi_h) || params->phi_h <= 0.0 || params->phi_h > 0.9999) return 0;
    if (!isfinite(params->sigma_h) || params->sigma_h <= 0.0) return 0;
    if (!isfinite(params->nu) || params->nu <= 2.0) return 0;
    return 1;
}

static int validate_ensemble_cfg(const svmix_ensemble_cfg_t* cfg) {
    if (!isfinite(cfg->lambda) || cfg->lambda <= 0.0 || cfg->lambda > 1.0) return 0;
    if (!isfinite(cfg->beta) || cfg->beta <= 0.0) return 0;
    if (!isfinite(cfg->epsilon) || cfg->epsilon < 0.0 || cfg->epsilon >= 1.0) return 0;
    if (cfg->num_threads < 0) return 0;
    return 1;
}

static int validate_config(const svmix_cfg_t* cfg) {
    if (!cfg) return 0;
    if (cfg->num_models == 0) return 0;
    if (cfg->num_particles == 0) return 0;
    if (cfg->spec != SVMIX_SPEC_VOL) return 0;
    if (!validate_ensemble_cfg(&cfg->ensemble)) return 0;
    return 1;
}

/* ========================================================================
 * svmix_create
 * ======================================================================== */

svmix_t* svmix_create(
    const svmix_cfg_t* cfg,
    const svmix_sv_params_t* models,
    const unsigned long* seeds
) {
    /* Validate inputs */
    if (!cfg || !models || !seeds) {
        return NULL;
    }
    
    if (!validate_config(cfg)) {
        return NULL;
    }
    
    /* Validate all model parameters */
    for (size_t i = 0; i < cfg->num_models; i++) {
        if (!validate_sv_params(&models[i])) {
            return NULL;
        }
    }
    
    /* Allocate svmix handle */
    svmix_t* svmix = (svmix_t*)malloc(sizeof(svmix_t));
    if (!svmix) {
        return NULL;
    }
    
    svmix->config = *cfg;
    svmix->timestep = 0;
    svmix->ensemble = NULL;
    svmix->sv_params = NULL;
    svmix->seeds = NULL;
    
    /* Copy SV parameters (for checkpointing) */
    svmix->sv_params = (sv_params_t*)malloc(cfg->num_models * sizeof(sv_params_t));
    if (!svmix->sv_params) {
        free(svmix);
        return NULL;
    }
    
    for (size_t i = 0; i < cfg->num_models; i++) {
        svmix->sv_params[i].mu_h = models[i].mu_h;
        svmix->sv_params[i].phi_h = models[i].phi_h;
        svmix->sv_params[i].sigma_h = models[i].sigma_h;
        svmix->sv_params[i].nu = models[i].nu;
    }
    
    /* Copy seeds (for checkpointing) */
    svmix->seeds = (unsigned long*)malloc(cfg->num_models * sizeof(unsigned long));
    if (!svmix->seeds) {
        free(svmix->sv_params);
        free(svmix);
        return NULL;
    }
    
    memcpy(svmix->seeds, seeds, cfg->num_models * sizeof(unsigned long));
    
    /* Translate ensemble config */
    ensemble_config_t ens_config;
    ens_config.lambda = cfg->ensemble.lambda;
    ens_config.beta = cfg->ensemble.beta;
    ens_config.epsilon = cfg->ensemble.epsilon;
    ens_config.num_threads = cfg->ensemble.num_threads;
    
    /* Create ensemble */
    svmix->ensemble = ensemble_create(
        cfg->num_models,
        svmix->sv_params,
        cfg->num_particles,
        seeds,
        &ens_config
    );
    
    if (!svmix->ensemble) {
        free(svmix->seeds);
        free(svmix->sv_params);
        free(svmix);
        return NULL;
    }
    
    return svmix;
}

/* ========================================================================
 * svmix_free
 * ======================================================================== */

void svmix_free(svmix_t* svmix) {
    if (!svmix) return;
    
    if (svmix->ensemble) {
        ensemble_free(svmix->ensemble);
    }
    
    if (svmix->sv_params) {
        free(svmix->sv_params);
    }
    
    if (svmix->seeds) {
        free(svmix->seeds);
    }
    
    free(svmix);
}

/* ========================================================================
 * svmix_step
 * ======================================================================== */

svmix_status_t svmix_step(svmix_t* svmix, double observation) {
    if (!svmix) {
        return SVMIX_ERR_NULL_POINTER;
    }
    
    if (!isfinite(observation)) {
        return SVMIX_ERR_INVALID_PARAM;
    }
    
    int result = ensemble_step(svmix->ensemble, observation);
    if (result != 0) {
        return SVMIX_ERR_INTERNAL;
    }
    
    svmix->timestep++;
    return SVMIX_OK;
}

/* ========================================================================
 * svmix_get_belief
 * ======================================================================== */

svmix_status_t svmix_get_belief(const svmix_t* svmix, svmix_belief_t* belief) {
    if (!svmix || !belief) {
        return SVMIX_ERR_NULL_POINTER;
    }
    
    ensemble_belief_t ens_belief = ensemble_get_belief(svmix->ensemble);
    
    belief->mean_h = ens_belief.mean_h;
    belief->var_h = ens_belief.var_h;
    belief->mean_sigma = ens_belief.mean_sigma;
    belief->valid = ens_belief.valid ? 1 : 0;
    
    return SVMIX_OK;
}

/* ========================================================================
 * svmix_get_weights
 * ======================================================================== */

svmix_status_t svmix_get_weights(const svmix_t* svmix, double* weights, size_t K) {
    if (!svmix || !weights) {
        return SVMIX_ERR_NULL_POINTER;
    }
    
    if (K != svmix->config.num_models) {
        return SVMIX_ERR_INVALID_PARAM;
    }
    
    size_t actual_K = ensemble_get_weights(svmix->ensemble, weights);
    if (actual_K != K) {
        return SVMIX_ERR_INTERNAL;
    }
    
    return SVMIX_OK;
}

/* ========================================================================
 * svmix_get_num_models
 * ======================================================================== */

size_t svmix_get_num_models(const svmix_t* svmix) {
    if (!svmix) return 0;
    return svmix->config.num_models;
}

/* ========================================================================
 * svmix_get_timestep
 * ======================================================================== */

size_t svmix_get_timestep(const svmix_t* svmix) {
    if (!svmix) return 0;
    return svmix->timestep;
}

/* ========================================================================
 * Checkpointing
 * ======================================================================== */

/* Implemented in checkpoint.c - declarations in checkpoint.h */
/* Prototypes here to avoid circular include */
svmix_status_t svmix_save_checkpoint(const svmix_t* svmix, const char* filepath);
svmix_t* svmix_load_checkpoint(const char* filepath, svmix_status_t* status);
