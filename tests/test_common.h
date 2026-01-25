/**
 * @file test_common.h
 * @brief Minimal test framework for svmix unit tests.
 *
 * Provides simple macros for test definition, assertions, and runners.
 * Zero external dependencies.
 */

#ifndef SVMIX_TEST_COMMON_H
#define SVMIX_TEST_COMMON_H

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

/* ========================================================================
 * Color output
 * ======================================================================== */

#define TEST_COLOR_RED     "\033[0;31m"
#define TEST_COLOR_GREEN   "\033[0;32m"
#define TEST_COLOR_YELLOW  "\033[0;33m"
#define TEST_COLOR_RESET   "\033[0m"

/* ========================================================================
 * Global test state
 * ======================================================================== */

static int g_test_passed = 0;
static int g_test_failed = 0;
static const char* g_current_test = NULL;
static int g_current_test_failed = 0;  /* Flag: did current test fail? */

/* ========================================================================
 * Test definition and runner
 * ======================================================================== */

/**
 * @brief Define a test function.
 * Usage: TEST(my_test) { ASSERT_TRUE(1 == 1); }
 */
#define TEST(name) static void test_##name(void)

/**
 * @brief Run a test function and track results.
 * Usage: RUN_TEST(my_test);
 */
#define RUN_TEST(name) \
    do { \
        g_current_test = #name; \
        g_current_test_failed = 0; \
        printf("  Running test: %s ... ", #name); \
        fflush(stdout); \
        test_##name(); \
        if (g_current_test_failed) { \
            /* Assertion already printed FAIL, count was already incremented */ \
        } else { \
            printf(TEST_COLOR_GREEN "PASS" TEST_COLOR_RESET "\n"); \
            g_test_passed++; \
        } \
    } while (0)

/* ========================================================================
 * Assertions
 * ======================================================================== */

/**
 * @brief Assert that a condition is true.
 */
#define ASSERT_TRUE(cond) \
    do { \
        if (!(cond)) { \
            printf(TEST_COLOR_RED "FAIL" TEST_COLOR_RESET "\n"); \
            fprintf(stderr, TEST_COLOR_RED "  ASSERTION FAILED: %s\n" TEST_COLOR_RESET, #cond); \
            fprintf(stderr, "  at %s:%d in test %s\n", __FILE__, __LINE__, g_current_test); \
            g_test_failed++; \
            g_current_test_failed = 1; \
            return; \
        } \
    } while (0)

/**
 * @brief Assert that a condition is false.
 */
#define ASSERT_FALSE(cond) ASSERT_TRUE(!(cond))

/**
 * @brief Assert that two doubles are close within relative tolerance.
 * Uses combined absolute + relative tolerance for robustness near zero.
 */
#define ASSERT_NEAR(a, b, tol) \
    do { \
        double _a = (a); \
        double _b = (b); \
        double _tol = (tol); \
        double _diff = fabs(_a - _b); \
        double _max = fmax(fabs(_a), fabs(_b)); \
        if (_diff > _tol && _diff > _tol * _max) { \
            printf(TEST_COLOR_RED "FAIL" TEST_COLOR_RESET "\n"); \
            fprintf(stderr, TEST_COLOR_RED "  ASSERTION FAILED: %s ≈ %s\n" TEST_COLOR_RESET, #a, #b); \
            fprintf(stderr, "  Expected: %.15g\n", _b); \
            fprintf(stderr, "  Actual:   %.15g\n", _a); \
            fprintf(stderr, "  Diff:     %.15g (tolerance: %.3g)\n", _diff, _tol); \
            fprintf(stderr, "  at %s:%d in test %s\n", __FILE__, __LINE__, g_current_test); \
            g_test_failed++; \
            g_current_test_failed = 1; \
            return; \
        } \
    } while (0)

/**
 * @brief Assert that a value is finite (not NaN or Inf).
 */
#define ASSERT_FINITE(x) \
    do { \
        double _x = (x); \
        if (!isfinite(_x)) { \
            printf(TEST_COLOR_RED "FAIL" TEST_COLOR_RESET "\n"); \
            fprintf(stderr, TEST_COLOR_RED "  ASSERTION FAILED: %s is finite\n" TEST_COLOR_RESET, #x); \
            fprintf(stderr, "  Value: %.15g\n", _x); \
            fprintf(stderr, "  at %s:%d in test %s\n", __FILE__, __LINE__, g_current_test); \
            g_test_failed++; \
            g_current_test_failed = 1; \
            return; \
        } \
    } while (0)

/**
 * @brief Assert that two integers are equal.
 */
#define ASSERT_EQ(a, b) \
    do { \
        long long _a = (long long)(a); \
        long long _b = (long long)(b); \
        if (_a != _b) { \
            printf(TEST_COLOR_RED "FAIL" TEST_COLOR_RESET "\n"); \
            fprintf(stderr, TEST_COLOR_RED "  ASSERTION FAILED: %s == %s\n" TEST_COLOR_RESET, #a, #b); \
            fprintf(stderr, "  Expected: %lld\n", _b); \
            fprintf(stderr, "  Actual:   %lld\n", _a); \
            fprintf(stderr, "  at %s:%d in test %s\n", __FILE__, __LINE__, g_current_test); \
            g_test_failed++; \
            g_current_test_failed = 1; \
            return; \
        } \
    } while (0)

/**
 * @brief Assert that two doubles are bit-exact equal (for determinism tests).
 * Uses memcmp to ensure exact bit-level equality.
 */
#define ASSERT_EQ_DOUBLE(a, b) \
    do { \
        double _a = (a); \
        double _b = (b); \
        if (memcmp(&_a, &_b, sizeof(double)) != 0) { \
            union { double d; unsigned long long u; } _ua, _ub; \
            _ua.d = _a; _ub.d = _b; \
            printf(TEST_COLOR_RED "FAIL" TEST_COLOR_RESET "\n"); \
            fprintf(stderr, TEST_COLOR_RED "  ASSERTION FAILED: %s == %s (bit-exact)\n" TEST_COLOR_RESET, #a, #b); \
            fprintf(stderr, "  Expected: %.17g (0x%016llx)\n", _ub.d, _ub.u); \
            fprintf(stderr, "  Actual:   %.17g (0x%016llx)\n", _ua.d, _ua.u); \
            fprintf(stderr, "  at %s:%d in test %s\n", __FILE__, __LINE__, g_current_test); \
            g_test_failed++; \
            g_current_test_failed = 1; \
            return; \
        } \
    } while (0)

/* ========================================================================
 * Test suite runner
 * ======================================================================== */

/**
 * @brief Print test summary and return exit code.
 * Call this at the end of main() after all RUN_TEST() calls.
 */
static int test_summary(void) {
    printf("\n========================================\n");
    if (g_test_failed == 0) {
        printf(TEST_COLOR_GREEN "All tests passed! (%d/%d)" TEST_COLOR_RESET "\n",
               g_test_passed, g_test_passed);
        printf("========================================\n");
        return 0;
    } else {
        printf(TEST_COLOR_RED "Tests failed: %d/%d" TEST_COLOR_RESET "\n",
               g_test_failed, g_test_passed + g_test_failed);
        printf(TEST_COLOR_GREEN "Tests passed: %d/%d" TEST_COLOR_RESET "\n",
               g_test_passed, g_test_passed + g_test_failed);
        printf("========================================\n");
        return 1;
    }
}

#endif /* SVMIX_TEST_COMMON_H */
