/**
 * @file test_student_t.c
 * @brief Unit tests for Student-t log-PDF implementation.
 *
 * Tests the numerically stable implementation of Student-t distribution
 * with location=0 and parameterized by log-variance.
 */

#include "../test_common.h"
#include "../../src/util.h"
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* ========================================================================
 * Test cases
 * ======================================================================== */

/**
 * Test: Basic sanity checks for valid inputs.
 */
TEST(basic_sanity) {
    double logpdf;

    /* At x=0, Student-t is at its mode (maximum density) */
    /* For nu=7, h=0 (variance=1): logpdf should be finite and reasonable */
    logpdf = svmix_t_logpdf_logvar(0.0, 0.0, 7.0);
    ASSERT_FINITE(logpdf);
    ASSERT_TRUE(logpdf < 0.0);  /* Log of probability < 1 */

    /* At x=0 with smaller variance (h=-2, var=exp(-2)≈0.135), density is higher */
    double logpdf_small_var = svmix_t_logpdf_logvar(0.0, -2.0, 7.0);
    ASSERT_FINITE(logpdf_small_var);
    ASSERT_TRUE(logpdf_small_var > logpdf);  /* Smaller variance → higher density at mode */

    /* Non-zero x reduces density */
    double logpdf_nonzero = svmix_t_logpdf_logvar(2.0, 0.0, 7.0);
    ASSERT_FINITE(logpdf_nonzero);
    ASSERT_TRUE(logpdf_nonzero < logpdf);  /* Density lower away from mode */
}

/**
 * Test: Symmetry around zero (loc=0 distribution).
 */
TEST(symmetry) {
    double h = -1.0;
    double nu = 5.0;

    double logpdf_pos = svmix_t_logpdf_logvar(1.5, h, nu);
    double logpdf_neg = svmix_t_logpdf_logvar(-1.5, h, nu);

    ASSERT_FINITE(logpdf_pos);
    ASSERT_FINITE(logpdf_neg);
    ASSERT_NEAR(logpdf_pos, logpdf_neg, 1e-14);  /* Should be bitwise identical */
}

/**
 * Test: Scaling behavior with variance (h = log(sigma^2)).
 * Larger h (larger variance) should reduce density at fixed x.
 */
TEST(variance_scaling) {
    double x = 1.0;
    double nu = 7.0;

    double logpdf_h0 = svmix_t_logpdf_logvar(x, 0.0, nu);   /* var = 1 */
    double logpdf_h2 = svmix_t_logpdf_logvar(x, 2.0, nu);   /* var = e^2 ≈ 7.39 */

    ASSERT_FINITE(logpdf_h0);
    ASSERT_FINITE(logpdf_h2);
    ASSERT_TRUE(logpdf_h2 < logpdf_h0);  /* Larger variance → lower density */
}

/**
 * Test: Heavier tails for smaller nu.
 * At large |x|, smaller nu should give higher log-probability (fatter tails).
 */
TEST(tail_heaviness) {
    double x = 5.0;  /* Far from mode */
    double h = 0.0;

    double logpdf_nu3 = svmix_t_logpdf_logvar(x, h, 3.0);   /* Heavy tails */
    double logpdf_nu30 = svmix_t_logpdf_logvar(x, h, 30.0); /* Closer to Gaussian */

    ASSERT_FINITE(logpdf_nu3);
    ASSERT_FINITE(logpdf_nu30);
    ASSERT_TRUE(logpdf_nu3 > logpdf_nu30);  /* Heavier tails for smaller nu */
}

/**
 * Test: Cross-check against known values.
 * For nu=5, h=0 (var=1, scale=1), x=0: mode of the distribution
 */
TEST(known_value_mode) {
    double logpdf = svmix_t_logpdf_logvar(0.0, 0.0, 5.0);
    double expected = -0.968619589054725;  /* From reference implementation */
    ASSERT_NEAR(logpdf, expected, 1e-12);
}

/**
 * Test: Cross-check at non-zero x.
 * For nu=7, h=log(4)≈1.386, x=2.0: scale=2, non-zero observation
 */
TEST(known_value_nonzero) {
    double h = log(4.0);  /* var=4, scale=2 */
    double logpdf = svmix_t_logpdf_logvar(2.0, h, 7.0);
    double expected = -2.181806901629411;  /* From reference implementation */
    ASSERT_NEAR(logpdf, expected, 1e-12);
}

/**
 * Test: Realistic SV parameters (minute return scale).
 * μ_h = log(1e-6) ≈ -13.815, typical h in [-15, -12], nu=7.
 * 
 * Note: With very small variance, PDF at mode can be > 1 (log-PDF > 0).
 * This is VALID for continuous distributions! Only the integral must equal 1.
 */
TEST(realistic_sv_scale) {
    double nu = 7.0;
    double h = -13.815510557964;  /* log((0.001)^2) */
    double r = 0.0005;  /* 5 bps return */

    double logpdf = svmix_t_logpdf_logvar(r, h, nu);
    ASSERT_FINITE(logpdf);
    /* With tiny variance, density can be very high (PDF > 1, log-PDF > 0) */
    /* Just check it's reasonable and finite */
    ASSERT_TRUE(logpdf < 10.0);  /* Not absurdly large */
}

/**
 * Test: Extreme but valid inputs.
 */
TEST(extreme_valid_inputs) {
    /* Very large variance (h=10, var≈22000) */
    double logpdf1 = svmix_t_logpdf_logvar(0.0, 10.0, 7.0);
    ASSERT_FINITE(logpdf1);
    ASSERT_TRUE(logpdf1 < 0.0);  /* Large variance → low density at mode */

    /* Very small variance (h=-20, var≈2e-9) - density can be very high */
    double logpdf2 = svmix_t_logpdf_logvar(0.0, -20.0, 7.0);
    ASSERT_FINITE(logpdf2);
    ASSERT_TRUE(logpdf2 > 0.0);  /* Tiny variance → very high density (PDF > 1) */

    /* Large |x| with moderate variance - should be far in tail */
    double logpdf3 = svmix_t_logpdf_logvar(100.0, 0.0, 7.0);
    ASSERT_FINITE(logpdf3);
    ASSERT_TRUE(logpdf3 < -25.0);  /* Should be very small probability in tail */
}

/**
 * Test: Invalid inputs should return -INFINITY.
 */
TEST(invalid_inputs) {
    /* nu <= 2 (no finite variance) */
    ASSERT_TRUE(isinf(svmix_t_logpdf_logvar(0.0, 0.0, 2.0)));
    ASSERT_TRUE(isinf(svmix_t_logpdf_logvar(0.0, 0.0, 1.5)));
    ASSERT_TRUE(isinf(svmix_t_logpdf_logvar(0.0, 0.0, 0.0)));
    ASSERT_TRUE(isinf(svmix_t_logpdf_logvar(0.0, 0.0, -1.0)));

    /* NaN inputs */
    ASSERT_TRUE(isinf(svmix_t_logpdf_logvar(NAN, 0.0, 7.0)));
    ASSERT_TRUE(isinf(svmix_t_logpdf_logvar(0.0, NAN, 7.0)));
    ASSERT_TRUE(isinf(svmix_t_logpdf_logvar(0.0, 0.0, NAN)));

    /* Infinite inputs */
    ASSERT_TRUE(isinf(svmix_t_logpdf_logvar(INFINITY, 0.0, 7.0)));
    ASSERT_TRUE(isinf(svmix_t_logpdf_logvar(0.0, INFINITY, 7.0)));
    ASSERT_TRUE(isinf(svmix_t_logpdf_logvar(0.0, 0.0, INFINITY)));
}

/**
 * Test: Numerical stability with extreme h values.
 * The implementation should handle exp(-h) * x^2 gracefully.
 */
TEST(numerical_stability_extreme_h) {
    double x = 1.0;
    double nu = 7.0;

    /* Very large h: variance is huge, but computation should remain stable */
    double logpdf_large_h = svmix_t_logpdf_logvar(x, 50.0, nu);
    ASSERT_FINITE(logpdf_large_h);

    /* Very small h: variance is tiny, but computation should remain stable */
    double logpdf_small_h = svmix_t_logpdf_logvar(x, -50.0, nu);
    ASSERT_FINITE(logpdf_small_h);
}

/**
 * Test: As nu increases, Student-t approaches Gaussian.
 * For large nu, the log-PDF should converge to Gaussian log-PDF.
 */
TEST(large_nu_gaussian_limit) {
    double x = 1.0;
    double h = 0.0;  /* var=1, scale=1 */

    /* Gaussian log-PDF: -0.5*log(2*pi) - 0.5*x^2 = -0.9189 - 0.5 = -1.4189 */
    double gaussian_logpdf = -0.5 * log(2.0 * M_PI) - 0.5 * (x * x);

    /* For nu=100, Student-t should be very close to Gaussian */
    double logpdf_nu100 = svmix_t_logpdf_logvar(x, h, 100.0);
    ASSERT_NEAR(logpdf_nu100, gaussian_logpdf, 0.01);

    /* For nu=1000, even closer */
    double logpdf_nu1000 = svmix_t_logpdf_logvar(x, h, 1000.0);
    ASSERT_NEAR(logpdf_nu1000, gaussian_logpdf, 0.001);
}

/* ========================================================================
 * Test runner
 * ======================================================================== */

int main(void) {
    printf("========================================\n");
    printf("Student-t Log-PDF Unit Tests\n");
    printf("========================================\n\n");

    RUN_TEST(basic_sanity);
    RUN_TEST(symmetry);
    RUN_TEST(variance_scaling);
    RUN_TEST(tail_heaviness);
    RUN_TEST(known_value_mode);
    RUN_TEST(known_value_nonzero);
    RUN_TEST(realistic_sv_scale);
    RUN_TEST(extreme_valid_inputs);
    RUN_TEST(invalid_inputs);
    RUN_TEST(numerical_stability_extreme_h);
    RUN_TEST(large_nu_gaussian_limit);

    return test_summary();
}
