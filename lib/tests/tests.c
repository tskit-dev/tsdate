/*
 * MIT License
 *
 * Copyright (c) 2026 Tskit Developers
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

/*
 * Smoke tests for the low-level tsdate API. These check return codes and
 * memory handling only; numerical correctness is tested from Python.
 */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <CUnit/Basic.h>

#include "tsdate.h"

static void
test_strerror(void)
{
    const int codes[] = { 0, TSD_ERR_GENERIC, TSD_ERR_NO_MEMORY, TSD_ERR_BAD_PARAM,
        TSD_ERR_ZERO_DIVISION, TSD_ERR_INVALID_HYPERGEO };
    const int num_codes = (int) (sizeof(codes) / sizeof(codes[0]));
    const char *msg_i;
    const char *msg_j;
    int i, j;

    for (i = 0; i < num_codes; i++) {
        msg_i = tsd_strerror(codes[i]);
        CU_ASSERT_FATAL(msg_i != NULL);
        CU_ASSERT(strlen(msg_i) > 0);
        for (j = 0; j < i; j++) {
            msg_j = tsd_strerror(codes[j]);
            CU_ASSERT(strcmp(msg_i, msg_j) != 0);
        }
    }
}

static void
test_strerror_unknown(void)
{
    const char *msg = tsd_strerror(-12345);

    CU_ASSERT_FATAL(msg != NULL);
    CU_ASSERT_STRING_EQUAL(msg, "Unknown error");
}

static void
test_trace_error(void)
{
    int ret = tsd_trace_error(TSD_ERR_BAD_PARAM);

    CU_ASSERT_EQUAL(ret, TSD_ERR_BAD_PARAM);
}

static void
test_safe_free(void)
{
    double *x = NULL;

    tsd_safe_free(x);
    CU_ASSERT_PTR_NULL(x);

    x = malloc(10 * sizeof(*x));
    CU_ASSERT_PTR_NOT_NULL_FATAL(x);
    tsd_safe_free(x);
    CU_ASSERT_PTR_NULL(x);
    /* Freeing twice is harmless */
    tsd_safe_free(x);
    CU_ASSERT_PTR_NULL(x);
}

static void
test_bug_assert_passes(void)
{
    tsd_bug_assert(1 + 1 == 2);
}

static void
test_polygamma(void)
{
    const double points[] = { 1e-6, 0.5, 4.9, 8.4, 10.0, 100.0, -0.5, -1.5, -10.5 };
    const int num_points = (int) (sizeof(points) / sizeof(points[0]));
    double value;
    int j, ret;

    for (j = 0; j < num_points; j++) {
        ret = tsd_digamma(points[j], &value);
        CU_ASSERT_EQUAL(ret, 0);
        CU_ASSERT(isfinite(value));
        ret = tsd_trigamma(points[j], &value);
        CU_ASSERT_EQUAL(ret, 0);
        CU_ASSERT(isfinite(value));
    }
    /* NaN is in-band */
    ret = tsd_digamma(NAN, &value);
    CU_ASSERT_EQUAL(ret, 0);
    CU_ASSERT(isnan(value));
    ret = tsd_trigamma(NAN, &value);
    CU_ASSERT_EQUAL(ret, 0);
    CU_ASSERT(isnan(value));
}

static void
test_polygamma_poles(void)
{
    double value;
    int ret;

    ret = tsd_digamma(0.0, &value);
    CU_ASSERT_EQUAL(ret, TSD_ERR_ZERO_DIVISION);
    ret = tsd_trigamma(0.0, &value);
    CU_ASSERT_EQUAL(ret, TSD_ERR_ZERO_DIVISION);
    /* x ** 2 underflows to zero */
    ret = tsd_trigamma(1e-300, &value);
    CU_ASSERT_EQUAL(ret, TSD_ERR_ZERO_DIVISION);
}

static void
test_betaln(void)
{
    double value = tsd_betaln(2.0, 3.0);

    CU_ASSERT_DOUBLE_EQUAL(value, log(1.0 / 12.0), 1e-12);
}

static void
test_gammainc_der(void)
{
    /* x == 0, x > xbig, gaussian, series and continued fraction regimes */
    const double points[][2] = { { 1.0, 0.0 }, { 5.0, 1e9 }, { 2000.0, 2000.0 },
        { 2.0, 1.5 }, { 5.0, 50.0 } };
    const int num_points = (int) (sizeof(points) / sizeof(points[0]));
    double value;
    int j, ret;

    for (j = 0; j < num_points; j++) {
        ret = tsd_gammainc_der(points[j][0], points[j][1], &value);
        CU_ASSERT_EQUAL(ret, 0);
        CU_ASSERT(isfinite(value));
    }
    /* Series with a very small value: the result is truncated to zero */
    ret = tsd_gammainc_der(50.0, 1e-3, &value);
    CU_ASSERT_EQUAL(ret, 0);
    CU_ASSERT_EQUAL(value, 0.0);
}

static void
test_gammainc_der_errors(void)
{
    double value;
    int ret;

    ret = tsd_gammainc_der(-1.0, 1.0, &value);
    CU_ASSERT_EQUAL(ret, TSD_ERR_INVALID_HYPERGEO);
    ret = tsd_gammainc_der(1.0, -1.0, &value);
    CU_ASSERT_EQUAL(ret, TSD_ERR_INVALID_HYPERGEO);
    ret = tsd_gammainc_der(NAN, 1.0, &value);
    CU_ASSERT_EQUAL(ret, TSD_ERR_INVALID_HYPERGEO);
}

static void
test_hyperu_laplace(void)
{
    double value, dvalue;
    int ret;

    ret = tsd_hyperu_laplace(1.0, 2.0, 1.0, &value, &dvalue);
    CU_ASSERT_EQUAL(ret, 0);
    CU_ASSERT(isfinite(value));
    CU_ASSERT(isfinite(dvalue));
}

static void
test_hyperu_laplace_errors(void)
{
    double value, dvalue;
    int ret;

    ret = tsd_hyperu_laplace(1.0, 0.5, 1.0, &value, &dvalue);
    CU_ASSERT_EQUAL(ret, TSD_ERR_INVALID_HYPERGEO);
    ret = tsd_hyperu_laplace(0.0, 1.0, 1.0, &value, &dvalue);
    CU_ASSERT_EQUAL(ret, TSD_ERR_INVALID_HYPERGEO);
    ret = tsd_hyperu_laplace(1.0, 2.0, 0.0, &value, &dvalue);
    CU_ASSERT_EQUAL(ret, TSD_ERR_INVALID_HYPERGEO);
    ret = tsd_hyperu_laplace(1.0, 2.0, NAN, &value, &dvalue);
    CU_ASSERT_EQUAL(ret, TSD_ERR_INVALID_HYPERGEO);
    /* The Laplace root divides by zero */
    ret = tsd_hyperu_laplace(1.0, 2.0, 1e300, &value, &dvalue);
    CU_ASSERT_EQUAL(ret, TSD_ERR_ZERO_DIVISION);
    ret = tsd_hyperu_laplace(1.0, 2.0, INFINITY, &value, &dvalue);
    CU_ASSERT_EQUAL(ret, TSD_ERR_ZERO_DIVISION);
}

static void
test_hyp1f1_laplace(void)
{
    double value;
    int ret;

    ret = tsd_hyp1f1_laplace(1.0, 2.0, 1.0, &value);
    CU_ASSERT_EQUAL(ret, 0);
    CU_ASSERT(isfinite(value));
    ret = tsd_hyp1f1_laplace(1.0, 2.0, 0.0, &value);
    CU_ASSERT_EQUAL(ret, 0);
    CU_ASSERT_EQUAL(value, 0.0);
}

static void
test_hyp1f1_laplace_errors(void)
{
    double value;
    int ret;

    ret = tsd_hyp1f1_laplace(1.0, 1.0, 1.0, &value);
    CU_ASSERT_EQUAL(ret, TSD_ERR_INVALID_HYPERGEO);
    ret = tsd_hyp1f1_laplace(0.0, 1.0, 1.0, &value);
    CU_ASSERT_EQUAL(ret, TSD_ERR_INVALID_HYPERGEO);
    ret = tsd_hyp1f1_laplace(1e-10, 2e-10, 1e8, &value);
    CU_ASSERT_EQUAL(ret, TSD_ERR_ZERO_DIVISION);
}

static void
test_hyp2f1_laplace(void)
{
    /* The interior, z < 0 and the three unity limits, including negative z */
    const double args[][4] = { { 1.0, 2.0, 3.0, 0.5 }, { 1.0, 2.0, 3.0, -0.5 },
        { 1.0, 2.0, 3.0, 0.0 }, { 2.0, 3.0, 4.0, 1 - 1e-8 }, { 1.0, 2.0, 5.0, 1 - 1e-8 },
        { 1.0, 2.0, 3.0, 1 - 1e-8 }, { 2.0, 3.0, 4.0, -1e8 }, { 3.0, 2.0, 5.0, -1e8 },
        { 2.0, 2.0, 5.0, -1e8 } };
    const int num_args = (int) (sizeof(args) / sizeof(args[0]));
    double value;
    int j, ret;

    for (j = 0; j < num_args; j++) {
        ret = tsd_hyp2f1_laplace(args[j][0], args[j][1], args[j][2], args[j][3], &value);
        CU_ASSERT_EQUAL(ret, 0);
        CU_ASSERT(isfinite(value));
    }
}

static void
test_hyp2f1_laplace_errors(void)
{
    double value;
    int ret;

    /* Each condition of the domain, and the root on the boundary */
    ret = tsd_hyp2f1_laplace(1.0, 2.0, 0.0, 0.5, &value);
    CU_ASSERT_EQUAL(ret, TSD_ERR_INVALID_HYPERGEO);
    ret = tsd_hyp2f1_laplace(-1.0, 2.0, 3.0, 0.5, &value);
    CU_ASSERT_EQUAL(ret, TSD_ERR_INVALID_HYPERGEO);
    ret = tsd_hyp2f1_laplace(1.0, -1.0, 3.0, 0.5, &value);
    CU_ASSERT_EQUAL(ret, TSD_ERR_INVALID_HYPERGEO);
    ret = tsd_hyp2f1_laplace(2.0, 1.0, 1.0, 0.5, &value);
    CU_ASSERT_EQUAL(ret, TSD_ERR_INVALID_HYPERGEO);
    ret = tsd_hyp2f1_laplace(1.0, 2.0, 3.0, 1.0, &value);
    CU_ASSERT_EQUAL(ret, TSD_ERR_INVALID_HYPERGEO);
    ret = tsd_hyp2f1_laplace(0.0, 1.0, 1.0, 0.5, &value);
    CU_ASSERT_EQUAL(ret, TSD_ERR_INVALID_HYPERGEO);
    /* z / (z - 1) rounds to one, outside the domain of the unity limit */
    ret = tsd_hyp2f1_laplace(1.0, 2.0, 3.0, -1e16, &value);
    CU_ASSERT_EQUAL(ret, TSD_ERR_INVALID_HYPERGEO);
    /* The Laplace root divides by zero */
    ret = tsd_hyp2f1_laplace(0.0, 1.0, 1e-300, 0.5, &value);
    CU_ASSERT_EQUAL(ret, TSD_ERR_ZERO_DIVISION);
    /* c == a, with a root that rounds into (0, 1) */
    ret = tsd_hyp2f1_laplace(1.1774154985086274, 504.4082462406216, 1.1774154985086274,
        0.1427180165924374, &value);
    CU_ASSERT_EQUAL(ret, TSD_ERR_ZERO_DIVISION);
}

static void
handle_cunit_error(void)
{
    fprintf(
        stderr, "CUnit error occurred: %d: %s\n", CU_get_error(), CU_get_error_msg());
    exit(EXIT_FAILURE);
}

int
main(int argc, char **argv)
{
    int ret;
    CU_pTest test;
    CU_pSuite suite;
    CU_TestInfo tests[] = {
        { "test_strerror", test_strerror },
        { "test_strerror_unknown", test_strerror_unknown },
        { "test_trace_error", test_trace_error },
        { "test_safe_free", test_safe_free },
        { "test_bug_assert_passes", test_bug_assert_passes },
        { "test_polygamma", test_polygamma },
        { "test_polygamma_poles", test_polygamma_poles },
        { "test_betaln", test_betaln },
        { "test_gammainc_der", test_gammainc_der },
        { "test_gammainc_der_errors", test_gammainc_der_errors },
        { "test_hyperu_laplace", test_hyperu_laplace },
        { "test_hyperu_laplace_errors", test_hyperu_laplace_errors },
        { "test_hyp1f1_laplace", test_hyp1f1_laplace },
        { "test_hyp1f1_laplace_errors", test_hyp1f1_laplace_errors },
        { "test_hyp2f1_laplace", test_hyp2f1_laplace },
        { "test_hyp2f1_laplace_errors", test_hyp2f1_laplace_errors },
        CU_TEST_INFO_NULL,
    };
    CU_SuiteInfo suites[] = {
        { .pName = "tsdate", .pInitFunc = NULL, .pCleanupFunc = NULL, .pTests = tests },
        CU_SUITE_INFO_NULL,
    };

    if (CUE_SUCCESS != CU_initialize_registry()) {
        handle_cunit_error();
    }
    if (CUE_SUCCESS != CU_register_suites(suites)) {
        handle_cunit_error();
    }
    CU_basic_set_mode(CU_BRM_VERBOSE);

    if (argc == 1) {
        CU_basic_run_tests();
    } else if (argc == 2) {
        suite = CU_get_suite_by_name("tsdate", CU_get_registry());
        if (suite == NULL) {
            printf("Suite not found\n");
            return EXIT_FAILURE;
        }
        test = CU_get_test_by_name(argv[1], suite);
        if (test == NULL) {
            printf("Test '%s' not found\n", argv[1]);
            return EXIT_FAILURE;
        }
        CU_basic_run_test(suite, test);
    } else {
        printf("usage: ./tests <test_name>\n");
        return EXIT_FAILURE;
    }

    ret = EXIT_SUCCESS;
    if (CU_get_number_of_tests_failed() != 0) {
        printf("Test failed!\n");
        ret = EXIT_FAILURE;
    }
    CU_cleanup_registry();
    return ret;
}
