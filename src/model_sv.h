/**
 * @file model_sv.h
 * @brief Internal header for SV model implementation.
 *
 * Private API, not exposed to public svmix.h.
 */

#ifndef SVMIX_MODEL_SV_H
#define SVMIX_MODEL_SV_H

#include "../third_party/fastpf/include/fastpf.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Forward declaration */
typedef struct sv_model_ctx_t sv_model_ctx_t;

/**
 * @brief Create and configure an SV model for use with fastpf.
 *
 * Allocates and initializes an sv_model_ctx_t with given parameters,
 * then populates a fastpf_model_t with callbacks.
 *
 * Validates that:
 * - model_out is not NULL
 * - phi_h in (0, 0.9999] for stationarity (upper bound prevents variance blow-up)
 * - sigma_h > 0
 * - nu > 2 for finite variance
 * - All parameters are finite
 *
 * @param mu_h Long-run mean of log-variance.
 * @param phi_h Persistence in (0, 0.9999]. Upper bound prevents stationary variance explosion.
 * @param sigma_h Process noise (> 0).
 * @param nu Student-t degrees of freedom (> 2).
 * @param model_out Output: fastpf_model_t to populate with callbacks (must not be NULL).
 * @return Pointer to allocated sv_model_ctx_t (caller must free), or NULL on error.
 *
 * Example:
 *   fastpf_model_t model;
 *   sv_model_ctx_t* ctx = sv_model_create(-13.8, 0.98, 0.15, 7.0, &model);
 *   if (ctx) {
 *       fastpf_init(&pf, &cfg, &model);
 *       // ... use particle filter ...
 *       fastpf_free(&pf);
 *       free(ctx);
 *   }
 */
sv_model_ctx_t* sv_model_create(double mu_h, double phi_h, double sigma_h, double nu,
                                 fastpf_model_t* model_out);

#ifdef __cplusplus
}
#endif

#endif /* SVMIX_MODEL_SV_H */
