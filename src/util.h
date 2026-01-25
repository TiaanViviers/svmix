/**
 * @file util.h
 * @brief Internal utility functions for svmix.
 *
 * Private header, not exposed in public API.
 */

#ifndef SVMIX_UTIL_H
#define SVMIX_UTIL_H

#ifdef __cplusplus
extern "C" {
#endif

/* ========================================================================
 * Statistical distributions
 * ======================================================================== */

/**
 * @brief Compute log-PDF of Student-t distribution with location=0.
 *
 * Models: X ~ StudentT(nu, loc=0, scale=sigma)
 * where sigma is the Student-t scale parameter.
 *
 * This version takes log-variance h = log(sigma^2) directly, where
 * sigma is the SCALE parameter (NOT the variance of X).
 * 
 * Internally computes: sigma = exp(0.5 * h)
 *
 * IMPORTANT: For Student-t, the variance of X is NOT sigma^2.
 * The variance is: Var(X) = sigma^2 * nu/(nu-2) for nu > 2.
 * 
 * In the SV context:
 * - h_t is the log-variance of returns under a Gaussian approximation
 * - The Student-t scale is set to sigma = sqrt(exp(h_t)) = exp(h_t/2)
 * - The actual variance of returns is exp(h_t) * nu/(nu-2)
 *
 * PDF:
 *   p(x | nu, sigma) = Gamma((nu+1)/2) / (Gamma(nu/2) * sqrt(nu*pi*sigma^2))
 *                      * (1 + x^2 / (nu * sigma^2))^(-(nu+1)/2)
 *
 * Log-PDF:
 *   log p(x | nu, h) = lgamma((nu+1)/2) - lgamma(nu/2)
 *                      - 0.5 * log(nu*pi) - 0.5 * h
 *                      - ((nu+1)/2) * log(1 + x^2 / (nu * exp(h)))
 *
 * Numerically stable for typical ranges of h and nu.
 *
 * @param x      Observation (e.g., return r_t).
 * @param h      Log of SCALE-SQUARED: h = log(sigma^2) where sigma is the Student-t scale.
 * @param nu     Degrees of freedom (must be > 2 for finite variance).
 * @return       Log-probability density, or -INFINITY if inputs invalid.
 *
 * Validity checks:
 * - Requires nu > 2.0 (finite variance constraint)
 * - Requires h is finite (no NaN/Inf)
 * - Requires x is finite
 */
double svmix_t_logpdf_logvar(double x, double h, double nu);

#ifdef __cplusplus
}
#endif

#endif /* SVMIX_UTIL_H */
