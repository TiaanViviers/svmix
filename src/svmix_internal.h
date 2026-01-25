/**
 * @file svmix_internal.h
 * @brief Internal structure definitions for svmix.
 *
 * This header exposes the internal structure of svmix_t for use by
 * internal implementation files (checkpoint.c, etc.).
 *
 * DO NOT include this header in public API headers.
 * DO NOT include this header in user code.
 *
 * Pattern: Standard C library practice (see SQLite, Redis, HDF5)
 *   - Public API: svmix.h (opaque pointers only)
 *   - Internal impl: svmix_internal.h (struct definitions)
 */

#ifndef SVMIX_INTERNAL_H
#define SVMIX_INTERNAL_H

#include "svmix.h"
#include "ensemble.h"
#include "ensemble_internal.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Internal svmix instance structure.
 *
 * Contains:
 *   - Ensemble of K models with particle filters
 *   - Configuration snapshot (for checkpointing)
 *   - SV parameters for each model (for checkpointing)
 *   - RNG seeds (for determinism verification)
 *   - Timestep counter
 */
struct svmix_t {
    ensemble_t* ensemble;       /**< Underlying ensemble */
    svmix_cfg_t config;         /**< User configuration (for checkpointing) */
    sv_params_t* sv_params;     /**< SV parameters for K models (owned) */
    unsigned long* seeds;       /**< RNG seeds (owned, for checkpointing) */
    size_t timestep;            /**< Number of steps processed */
};

#ifdef __cplusplus
}
#endif

#endif /* SVMIX_INTERNAL_H */
