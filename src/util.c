/**
 * @file util.c
 * @brief Internal utility functions for svmix.
 */

#include "util.h"
#include <math.h>

/* ========================================================================
 * Constants
 * ======================================================================== */

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* ========================================================================
 * Student-t distribution
 * ======================================================================== */

/**
 * @brief Compute log-PDF of Student-t distribution with location=0.
 *
 * See util.h for full documentation.
 *
 * Implementation notes:
 * - Uses lgamma() from C99 math library for log-Gamma function
 * - Computes in log-space to avoid underflow
 * - Checks for invalid inputs (nu <= 2, NaN, Inf)
 * - Returns -INFINITY for invalid inputs (zero probability)
 */
double svmix_t_logpdf_logvar(double x, double h, double nu) {
    /* Validity checks */
    if (nu <= 2.0 || !isfinite(nu)) {
        return -INFINITY;
    }
    if (!isfinite(x)) {
        return -INFINITY;
    }
    if (!isfinite(h)) {
        return -INFINITY;
    }

    /*
     * Student-t PDF with scale σ = exp(h/2):
     *   p(x | ν, σ) = [Γ((ν+1)/2) / (Γ(ν/2) * √(νπ) * σ)]
     *                 * [1 + (x/σ)²/ν]^(-(ν+1)/2)
     *
     * Log-PDF:
     *   log p = lgamma((ν+1)/2) - lgamma(ν/2) - log(√(νπ)) - log(σ)
     *           - ((ν+1)/2) * log(1 + (x/σ)²/ν)
     *
     * Since σ = exp(h/2), we have log(σ) = h/2
     * And (x/σ)² = x² * exp(-h)
     */

    const double nu_plus_1 = nu + 1.0;
    const double half_nu_plus_1 = 0.5 * nu_plus_1;

    /* Normalization constant (log-space) */
    /* lgamma((ν+1)/2) - lgamma(ν/2) - log(√(νπ)) - log(σ) */
    /* = lgamma((ν+1)/2) - lgamma(ν/2) - 0.5*log(νπ) - h/2 */
    double log_norm = lgamma(half_nu_plus_1) - lgamma(0.5 * nu)
                      - 0.5 * log(nu * M_PI)
                      - 0.5 * h;  /* This was the bug: should be - 0.5*h, not - 0.5*h from sigma^2 */

    /* Compute (x/σ)² / ν = x² * exp(-h) / ν */
    double x_sq = x * x;
    double z_sq = (x_sq * exp(-h)) / nu;

    /* Log of (1 + z_sq)^(-(nu+1)/2) = -(nu+1)/2 * log(1 + z_sq) */
    double log_kernel = -half_nu_plus_1 * log(1.0 + z_sq);

    return log_norm + log_kernel;
}
