/**
 * @file model_sv_internal.h
 * @brief Internal structure definitions for SV model.
 *
 * This header exposes the internal structure of sv_model_ctx_t
 * for use by internal implementation files.
 *
 * DO NOT include this header in public API headers.
 *
 * Pattern: Standard C library practice for serialization/checkpointing.
 */

#ifndef SVMIX_MODEL_SV_INTERNAL_H
#define SVMIX_MODEL_SV_INTERNAL_H

#include "model_sv.h"
#include "ensemble.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief SV model context (internal definition).
 *
 * Contains the parameters for a single SV model:
 *   - mu_h: Long-run mean of log-variance
 *   - phi_h: AR(1) persistence (0 < phi_h < 1)
 *   - sigma_h: Process noise (> 0)
 *   - nu: Student-t degrees of freedom (> 2)
 *
 * This struct IS the same as sv_params_t (for V1).
 * In V2, it will be extended with drift/impulse parameters.
 */
struct sv_model_ctx_t {
    double mu_h;      /**< Long-run mean of log-variance */
    double phi_h;     /**< Persistence parameter in (0, 1) */
    double sigma_h;   /**< Process noise standard deviation (> 0) */
    double nu;        /**< Student-t degrees of freedom (> 2) */
};

#ifdef __cplusplus
}
#endif

#endif /* SVMIX_MODEL_SV_INTERNAL_H */
