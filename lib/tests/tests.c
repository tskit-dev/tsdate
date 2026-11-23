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
        TSD_ERR_ZERO_DIVISION, TSD_ERR_INVALID_HYPERGEO,
        TSD_ERR_NONPOSITIVE_OR_NONFINITE_MOMENTS, TSD_ERR_JENSEN_VIOLATION,
        TSD_ERR_KL_MAX_ITERATIONS, TSD_ERR_KL_INVALID_SHAPE,
        TSD_ERR_NONPOSITIVE_CENTRAL_MOMENTS, TSD_ERR_UNSORTED_QUANTILES,
        TSD_ERR_QUANTILE_MAX_ITERATIONS, TSD_ERR_NEGATIVE_SHAPE,
        TSD_ERR_NONPOSITIVE_MOMENTS, TSD_ERR_INVALID_FIXED_AGE,
        TSD_ERR_UNPHASED_INDIVIDUAL_EDGES, TSD_ERR_BLOCK_OUT_OF_RANGE,
        TSD_ERR_PHASE_OUT_OF_RANGE, TSD_ERR_UNPHASED_LIKELIHOOD_NOT_CONSERVED,
        TSD_ERR_MUTATION_NODE_NOT_IN_TREE, TSD_ERR_FIXED_NODES_VIOLATE_CONSTRAINT };
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

/*
 * A stand-in for SciPy's gammaincinv: the inverse of the regularised lower
 * incomplete gamma function by bisection, with the series expansion of the function.
 */
static double
test_gammainc(double a, double x)
{
    double term = 1.0 / a;
    double sum = term;
    int n;

    for (n = 1; n < 1000; n++) {
        term *= x / (a + n);
        sum += term;
    }
    return sum * exp(a * log(x) - x - lgamma(a));
}

static double
test_gammaincinv(double a, double p)
{
    double lo = 0.0;
    double hi = 1.0;
    double mid;
    int j;

    while (test_gammainc(a, hi) < p) {
        hi *= 2;
    }
    for (j = 0; j < 100; j++) {
        mid = (lo + hi) / 2;
        if (test_gammainc(a, mid) < p) {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    return (lo + hi) / 2;
}

static double
test_zero_inv(double a, double p)
{
    return a * 0.0 * p;
}

static double
test_tiny_inv(double a, double p)
{
    return 1e-200 * a * p;
}

static void
test_approximate_log_moments(void)
{
    double logx, xlogx, logx2;
    int ret;

    ret = tsd_approximate_log_moments(2.0, 0.5, &logx, &xlogx, &logx2);
    CU_ASSERT_EQUAL(ret, 0);
    CU_ASSERT(isfinite(logx) && isfinite(xlogx) && isfinite(logx2));
    ret = tsd_approximate_log_moments(0.0, 0.5, &logx, &xlogx, &logx2);
    CU_ASSERT_EQUAL(ret, TSD_ERR_NONPOSITIVE_MOMENTS);
    ret = tsd_approximate_log_moments(1.0, NAN, &logx, &xlogx, &logx2);
    CU_ASSERT_EQUAL(ret, TSD_ERR_NONPOSITIVE_MOMENTS);
    /* mean ** 2 underflows */
    ret = tsd_approximate_log_moments(1e-200, 1.0, &logx, &xlogx, &logx2);
    CU_ASSERT_EQUAL(ret, TSD_ERR_ZERO_DIVISION);
}

static void
test_approximate_gamma_kl(void)
{
    double out[2];
    int ret;

    ret = tsd_approximate_gamma_kl(2.0, 0.5, out);
    CU_ASSERT_EQUAL(ret, 0);
    CU_ASSERT(isfinite(out[0]) && isfinite(out[1]));
    /* the asymptotic lower bound */
    ret = tsd_approximate_gamma_kl(1.0, -1e-10, out);
    CU_ASSERT_EQUAL(ret, 0);
    ret = tsd_approximate_gamma_kl(0.0, -1.0, out);
    CU_ASSERT_EQUAL(ret, TSD_ERR_NONPOSITIVE_OR_NONFINITE_MOMENTS);
    ret = tsd_approximate_gamma_kl(1.0, 0.0, out);
    CU_ASSERT_EQUAL(ret, TSD_ERR_JENSEN_VIOLATION);
    /* the lower bound on the shape is zero */
    ret = tsd_approximate_gamma_kl(INFINITY, -1.0, out);
    CU_ASSERT_EQUAL(ret, TSD_ERR_ZERO_DIVISION);
    /* the trigamma function divides by zero */
    ret = tsd_approximate_gamma_kl(1.0, -1e164, out);
    CU_ASSERT_EQUAL(ret, TSD_ERR_ZERO_DIVISION);
}

static void
test_approximate_gamma_mom(void)
{
    double out[2];
    int ret;

    ret = tsd_approximate_gamma_mom(2.0, 4.0, out);
    CU_ASSERT_EQUAL(ret, 0);
    CU_ASSERT_EQUAL(out[0], 0.0);
    CU_ASSERT_EQUAL(out[1], 0.5);
    ret = tsd_approximate_gamma_mom(2.0, 0.0, out);
    CU_ASSERT_EQUAL(ret, TSD_ERR_NONPOSITIVE_CENTRAL_MOMENTS);
    ret = tsd_approximate_gamma_mom(NAN, 1.0, out);
    CU_ASSERT_EQUAL(ret, TSD_ERR_NONPOSITIVE_CENTRAL_MOMENTS);
}

static void
test_approximate_gamma_iqr(void)
{
    double out[2];
    double shape = 5.0;
    double x1 = test_gammaincinv(shape, 0.25);
    double x2 = test_gammaincinv(shape, 0.75);
    int ret;

    /* Newton iteration recovers the shape */
    ret = tsd_approximate_gamma_iqr(0.25, 0.75, x1, x2, 50.0, test_gammaincinv, out);
    CU_ASSERT_EQUAL(ret, 0);
    CU_ASSERT_DOUBLE_EQUAL(out[0], shape - 1, 1e-6);
    /* capped shape: equal quantiles, a lower bound or an iteration above the cap */
    ret = tsd_approximate_gamma_iqr(0.25, 0.75, 3.0, 3.0, 50.0, test_gammaincinv, out);
    CU_ASSERT_EQUAL(ret, 0);
    CU_ASSERT_EQUAL(out[0], 49.0);
    ret = tsd_approximate_gamma_iqr(0.25, 0.75, x1, x2, 2.0, test_gammaincinv, out);
    CU_ASSERT_EQUAL(ret, 0);
    CU_ASSERT_EQUAL(out[0], 1.0);
    ret = tsd_approximate_gamma_iqr(0.25, 0.75, 1.0, 1.1, 1e5, test_gammaincinv, out);
    CU_ASSERT_EQUAL(ret, 0);
    /* errors */
    ret = tsd_approximate_gamma_iqr(0.25, 0.75, 2.0, 1.0, 50.0, test_gammaincinv, out);
    CU_ASSERT_EQUAL(ret, TSD_ERR_UNSORTED_QUANTILES);
    ret = tsd_approximate_gamma_iqr(0.25, 0.75, 0.0, 0.0, 50.0, test_gammaincinv, out);
    CU_ASSERT_EQUAL(ret, TSD_ERR_ZERO_DIVISION);
    ret = tsd_approximate_gamma_iqr(0.0, 0.75, 1.0, 2.0, 50.0, test_gammaincinv, out);
    CU_ASSERT_EQUAL(ret, TSD_ERR_ZERO_DIVISION);
    ret = tsd_approximate_gamma_iqr(0.25, 0.75, 0.0, 1.0, 50.0, test_gammaincinv, out);
    CU_ASSERT_EQUAL(ret, TSD_ERR_ZERO_DIVISION);
    ret = tsd_approximate_gamma_iqr(0.25, 1.0, 1.0, 2.0, 50.0, test_gammaincinv, out);
    CU_ASSERT_EQUAL(ret, TSD_ERR_NEGATIVE_SHAPE);
    /* a quantile function that returns zero, or underflows when squared */
    ret = tsd_approximate_gamma_iqr(0.25, 0.75, 1.0, 2.0, 50.0, test_zero_inv, out);
    CU_ASSERT_EQUAL(ret, TSD_ERR_ZERO_DIVISION);
    ret = tsd_approximate_gamma_iqr(0.25, 0.75, 1.0, 2.0, 50.0, test_tiny_inv, out);
    CU_ASSERT_EQUAL(ret, TSD_ERR_ZERO_DIVISION);
}

static void
test_average_gammas(void)
{
    const double alpha[] = { 0.5, 3.0 };
    const double beta[] = { 1.0, 2.0 };
    const double zero_rate[] = { 1.0, 0.0 };
    const double pole[] = { -1.0, 3.0 };
    double out[2];
    int ret;

    ret = tsd_average_gammas(2, alpha, beta, out);
    CU_ASSERT_EQUAL(ret, 0);
    CU_ASSERT(isfinite(out[0]) && isfinite(out[1]));
    ret = tsd_average_gammas(0, alpha, beta, out);
    CU_ASSERT_EQUAL(ret, TSD_ERR_ZERO_DIVISION);
    ret = tsd_average_gammas(2, alpha, zero_rate, out);
    CU_ASSERT_EQUAL(ret, TSD_ERR_ZERO_DIVISION);
    /* digamma has a pole at shape 0 */
    ret = tsd_average_gammas(2, pole, beta, out);
    CU_ASSERT_EQUAL(ret, TSD_ERR_ZERO_DIVISION);
}

static void
test_valid(void)
{
    CU_ASSERT_TRUE(tsd_valid_moments(1.0, 1.0));
    CU_ASSERT_FALSE(tsd_valid_moments(NAN, 1.0));
    CU_ASSERT_FALSE(tsd_valid_moments(1.0, 0.0));
    CU_ASSERT_TRUE(tsd_valid_gamma(1.0, 1.0));
    CU_ASSERT_FALSE(tsd_valid_gamma(1.0, INFINITY));
    CU_ASSERT_FALSE(tsd_valid_gamma(0.0, 1.0));
    CU_ASSERT_TRUE(tsd_valid_hyp1f1(1.0, 2.0, 0.5));
    CU_ASSERT_FALSE(tsd_valid_hyp1f1(1.0, NAN, 0.5));
    CU_ASSERT_FALSE(tsd_valid_hyp1f1(2.0, 1.0, 0.5));
    CU_ASSERT_TRUE(tsd_valid_hyperu(1.0, 2.0, 0.5));
    CU_ASSERT_FALSE(tsd_valid_hyperu(1.0, 2.0, INFINITY));
    CU_ASSERT_FALSE(tsd_valid_hyperu(1.0, 2.0, 0.0));
    CU_ASSERT_FALSE(tsd_valid_hyperu(1.0, 1.0, 0.5));
    CU_ASSERT_TRUE(tsd_valid_hyp2f1(1.0, 2.0, 3.0, 0.5));
    CU_ASSERT_FALSE(tsd_valid_hyp2f1(1.0, 2.0, NAN, 0.5));
    CU_ASSERT_FALSE(tsd_valid_hyp2f1(1.0, 2.0, 3.0, NAN));
    CU_ASSERT_FALSE(tsd_valid_hyp2f1(1.0, 2.0, 3.0, 1.0));
    CU_ASSERT_FALSE(tsd_valid_hyp2f1(1.0, 2.0, 3.0, -1e16));
    CU_ASSERT_FALSE(tsd_valid_hyp2f1(0.0, 2.0, 3.0, 0.5));
}

/* approx: node moments */
static void
test_node_moments(void)
{
    int ret;
    double v[5];

    ret = tsd_moments(10, 0.01, 10, 0.01, 1, 1e-4, v, v + 1, v + 2, v + 3, v + 4);
    CU_ASSERT_EQUAL(ret, 0);
    CU_ASSERT_TRUE(isfinite(v[0]));
    ret = tsd_moments(10, 0.01, 10, -0.02, 1, 1e-4, v, v + 1, v + 2, v + 3, v + 4);
    CU_ASSERT_EQUAL(ret, 0);
    CU_ASSERT_TRUE(isnan(v[0]));
    ret = tsd_moments(10, 0.01, 10, 0.01, 1, 1e-4, v, v + 1, v + 2, v + 3, v + 4);
    CU_ASSERT_EQUAL(ret, 0);

    ret = tsd_unphased_moments(
        10, 0.01, 10, 0.01, 1, 1e-4, v, v + 1, v + 2, v + 3, v + 4);
    CU_ASSERT_EQUAL(ret, 0);
    CU_ASSERT_TRUE(isfinite(v[0]));
    ret = tsd_unphased_moments(
        10, -1.0, 10, 0.01, 1, 1e-4, v, v + 1, v + 2, v + 3, v + 4);
    CU_ASSERT_EQUAL(ret, 0);
    CU_ASSERT_TRUE(isnan(v[0]));

    /* a positive, a zero and an invalid child age; negative and NaN ages are errors */
    ret = tsd_rootward_moments(1000, 10, 0.01, 1, 1e-4, v, v + 1, v + 2);
    CU_ASSERT_EQUAL(ret, 0);
    CU_ASSERT_TRUE(isfinite(v[0]));
    ret = tsd_rootward_moments(0, 10, 0.01, 1, 1e-4, v, v + 1, v + 2);
    CU_ASSERT_EQUAL(ret, 0);
    CU_ASSERT_TRUE(isfinite(v[0]));
    ret = tsd_rootward_moments(1000, 10, -1.0, 1, 1e-4, v, v + 1, v + 2);
    CU_ASSERT_EQUAL(ret, 0);
    CU_ASSERT_TRUE(isnan(v[0]));
    ret = tsd_rootward_moments(1e308, 10, 1e10, 1, 1e-4, v, v + 1, v + 2);
    CU_ASSERT_EQUAL(ret, 0);
    CU_ASSERT_TRUE(isnan(v[0]));
    ret = tsd_rootward_moments(-1.0, 10, 0.01, 1, 1e-4, v, v + 1, v + 2);
    CU_ASSERT_EQUAL(ret, TSD_ERR_INVALID_FIXED_AGE);
    ret = tsd_rootward_moments(NAN, 10, 0.01, 1, 1e-4, v, v + 1, v + 2);
    CU_ASSERT_EQUAL(ret, TSD_ERR_INVALID_FIXED_AGE);

    ret = tsd_leafward_moments(1000, 10, 0.01, 1, 1e-4, v, v + 1, v + 2);
    CU_ASSERT_EQUAL(ret, 0);
    CU_ASSERT_TRUE(isfinite(v[0]));
    ret = tsd_leafward_moments(1000, 0, 0.01, 1, 1e-4, v, v + 1, v + 2);
    CU_ASSERT_EQUAL(ret, 0);
    CU_ASSERT_TRUE(isnan(v[0]));
    ret = tsd_leafward_moments(0, 10, 0.01, 1, 1e-4, v, v + 1, v + 2);
    CU_ASSERT_EQUAL(ret, TSD_ERR_INVALID_FIXED_AGE);

    ret = tsd_sideways_moments(1000, 10, 0.01, 1, 1e-4, v, v + 1, v + 2);
    CU_ASSERT_EQUAL(ret, 0);
    CU_ASSERT_TRUE(isfinite(v[0]));
    ret = tsd_sideways_moments(1000, 10, -1.0, 1, 1e-4, v, v + 1, v + 2);
    CU_ASSERT_EQUAL(ret, 0);
    CU_ASSERT_TRUE(isnan(v[0]));
    ret = tsd_sideways_moments(0, 10, 0.01, 1, 1e-4, v, v + 1, v + 2);
    CU_ASSERT_EQUAL(ret, TSD_ERR_INVALID_FIXED_AGE);

    ret = tsd_twin_moments(10, 0.01, 1, 1e-4, v, v + 1, v + 2);
    CU_ASSERT_EQUAL(ret, 0);
    CU_ASSERT_TRUE(isfinite(v[0]));
    ret = tsd_twin_moments(10, -2e-4, 1, 1e-4, v, v + 1, v + 2);
    CU_ASSERT_EQUAL(ret, TSD_ERR_ZERO_DIVISION);
    ret = tsd_twin_moments(10, 1e-300, 1, 0.0, v, v + 1, v + 2);
    CU_ASSERT_EQUAL(ret, TSD_ERR_ZERO_DIVISION);
}

/* Each projection once with usable moments, and once with unusable ones (NaN). */
static void
test_mutation_projections(void)
{
    const double pars_i[2] = { 2.0, 1.0 };
    const double pars_j[2] = { 1.0, 2.0 };
    const double pars_ij[2] = { 1.0, 1.0 };
    double pr_m;
    double proj[2];

    CU_ASSERT_EQUAL(
        tsd_mutation_gamma_projection(pars_i, pars_j, pars_ij, &pr_m, proj), 0);
    CU_ASSERT_EQUAL(pr_m, 1.0);
    CU_ASSERT_EQUAL(
        tsd_mutation_leafward_projection(2.0, pars_j, pars_ij, &pr_m, proj), 0);
    CU_ASSERT_EQUAL(
        tsd_mutation_rootward_projection(1.0, pars_i, pars_ij, &pr_m, proj), 0);
    CU_ASSERT_EQUAL(tsd_mutation_edge_projection(2.0, 1.0, &pr_m, proj), 0);
    CU_ASSERT_EQUAL(pr_m, 1.0);
    CU_ASSERT_EQUAL(
        tsd_mutation_unphased_projection(pars_i, pars_j, pars_ij, &pr_m, proj), 0);
    CU_ASSERT_EQUAL(tsd_mutation_twin_projection(pars_i, pars_ij, &pr_m, proj), 0);
    CU_ASSERT_EQUAL(
        tsd_mutation_sideways_projection(2.0, pars_j, pars_ij, &pr_m, proj), 0);
    CU_ASSERT_EQUAL(tsd_mutation_block_projection(2.0, 1.0, &pr_m, proj), 0);
    CU_ASSERT_TRUE(pr_m > 0.0 && pr_m < 1.0);

    /* Zero variance: the moments are unusable, which is in-band. */
    CU_ASSERT_EQUAL(tsd_mutation_edge_projection(1.0, 1.0, &pr_m, proj), 0);
    CU_ASSERT_TRUE(isnan(pr_m));
    CU_ASSERT_TRUE(isnan(proj[0]));
    CU_ASSERT_TRUE(isnan(proj[1]));
}

/* The node projections: usable moments, and unusable ones (NaN, input copied). */
static void
test_node_projections(void)
{
    const double pars[2] = { 1.0, 2.0 };
    const double bad[2] = { -5.0, 1.0 };
    const double edge[2] = { 2.0, 0.5 };
    double logl;
    double proj_i[2];
    double proj_j[2];

    CU_ASSERT_EQUAL(tsd_gamma_projection(pars, pars, edge, &logl, proj_i, proj_j), 0);
    CU_ASSERT_TRUE(isfinite(logl));
    CU_ASSERT_EQUAL(tsd_gamma_projection(bad, pars, edge, &logl, proj_i, proj_j), 0);
    CU_ASSERT_TRUE(isnan(logl));
    CU_ASSERT_EQUAL(proj_i[0], bad[0]);
    CU_ASSERT_EQUAL(proj_j[1], pars[1]);

    CU_ASSERT_EQUAL(tsd_unphased_projection(pars, pars, edge, &logl, proj_i, proj_j), 0);
    CU_ASSERT_TRUE(isfinite(logl));
    CU_ASSERT_EQUAL(tsd_unphased_projection(bad, pars, edge, &logl, proj_i, proj_j), 0);
    CU_ASSERT_TRUE(isnan(logl));

    CU_ASSERT_EQUAL(tsd_leafward_projection(2.0, pars, edge, &logl, proj_j), 0);
    CU_ASSERT_TRUE(isfinite(logl));
    CU_ASSERT_EQUAL(tsd_leafward_projection(2.0, bad, edge, &logl, proj_j), 0);
    CU_ASSERT_TRUE(isnan(logl));
    CU_ASSERT_EQUAL(proj_j[0], bad[0]);

    CU_ASSERT_EQUAL(tsd_rootward_projection(0.5, pars, edge, &logl, proj_i), 0);
    CU_ASSERT_TRUE(isfinite(logl));
    CU_ASSERT_EQUAL(tsd_rootward_projection(0.5, bad, edge, &logl, proj_i), 0);
    CU_ASSERT_TRUE(isnan(logl));

    CU_ASSERT_EQUAL(tsd_twin_projection(pars, edge, &logl, proj_i), 0);
    CU_ASSERT_TRUE(isfinite(logl));
    CU_ASSERT_EQUAL(tsd_twin_projection(bad, edge, &logl, proj_i), 0);
    CU_ASSERT_TRUE(isnan(logl));

    CU_ASSERT_EQUAL(tsd_sideways_projection(1.0, pars, edge, &logl, proj_j), 0);
    CU_ASSERT_TRUE(isfinite(logl));
    CU_ASSERT_EQUAL(tsd_sideways_projection(1.0, bad, edge, &logl, proj_j), 0);
    CU_ASSERT_TRUE(isnan(logl));
}

/* approx: mutation moments */
static void
test_mutation_moments(void)
{
    double pr, mn, va;

    CU_ASSERT_EQUAL(tsd_mutation_moments(5, 0.01, 6, 0.02, 2, 0.001, &mn, &va), 0);
    CU_ASSERT_TRUE(mn > 0);
    /* t <= 0: the hypergeometric domain check fails, in-band */
    CU_ASSERT_EQUAL(tsd_mutation_moments(5, -1, 6, 0.02, 2, 0.5, &mn, &va), 0);
    CU_ASSERT_TRUE(isnan(mn));
    CU_ASSERT_EQUAL(tsd_mutation_rootward_moments(1, 5, 0.01, 2, 0.001, &mn, &va), 0);
    CU_ASSERT_EQUAL(tsd_mutation_leafward_moments(50, 5, 0.01, 2, 0.001, &mn, &va), 0);
    CU_ASSERT_EQUAL(
        tsd_mutation_unphased_moments(5, 0.01, 6, 0.02, 2, 0.001, &pr, &mn, &va), 0);
    CU_ASSERT_TRUE(pr > 0);
    CU_ASSERT_EQUAL(
        tsd_mutation_unphased_moments(5, -1, 6, 0.02, 2, 0.5, &pr, &mn, &va), 0);
    CU_ASSERT_TRUE(isnan(pr));
    CU_ASSERT_EQUAL(tsd_mutation_twin_moments(5, 0.01, 2, 0.001, &pr, &mn, &va), 0);
    CU_ASSERT_EQUAL(
        tsd_mutation_sideways_moments(10, 5, 0.01, 2, 0.001, &pr, &mn, &va), 0);
    CU_ASSERT_EQUAL(
        tsd_mutation_sideways_moments(10, -5, 0.01, 2, 0.001, &pr, &mn, &va), 0);
    CU_ASSERT_TRUE(isnan(pr));
    CU_ASSERT_EQUAL(tsd_mutation_sideways_moments(0, 5, 0.01, 2, 0.001, &pr, &mn, &va),
        TSD_ERR_INVALID_FIXED_AGE);
    CU_ASSERT_EQUAL(tsd_mutation_edge_moments(10, 2, &mn, &va), 0);
    CU_ASSERT_EQUAL(tsd_mutation_block_moments(10, 2, &pr, &mn, &va), 0);
    CU_ASSERT_EQUAL(
        tsd_mutation_block_moments(0, 2, &pr, &mn, &va), TSD_ERR_INVALID_FIXED_AGE);
    CU_ASSERT_EQUAL(
        tsd_mutation_block_moments(2, 0, &pr, &mn, &va), TSD_ERR_INVALID_FIXED_AGE);
}

/* Two samples in each of two individuals below a root: 0, 1 | 2, 3 | root 4 */
static const tsd_id_t phase_nodes_individual[] = { 0, 0, 1, 1, TSD_NULL };
static const tsd_id_t phase_edges_parent[] = { 4, 4, 4, 4 };
static const tsd_id_t phase_edges_child[] = { 0, 1, 2, 3 };
static const double phase_edges_left[] = { 0, 0, 0, 0 };
static const double phase_edges_right[] = { 10, 10, 10, 10 };
static const tsd_id_t phase_order[] = { 0, 1, 2, 3 };

static void
test_block_singletons(void)
{
    const uint8_t unphased[] = { 1, 1 };
    const tsd_id_t mutations_node[] = { 0, 4, 2 };
    const double mutations_position[] = { 5, 1, 2 };
    const tsd_id_t child_missing[] = { 0, 2, 3, 3 };
    tsd_id_t mutations_block[3];
    tsd_id_t *blocks_edges = NULL;
    double *blocks_stats = NULL;
    tsd_size_t num_blocks = 0;
    int ret;

    ret = tsd_block_singletons(2, 4, 3, unphased, phase_nodes_individual, mutations_node,
        mutations_position, phase_edges_child, phase_edges_left, phase_edges_right,
        phase_order, phase_order, 10.0, &num_blocks, &blocks_stats, &blocks_edges,
        mutations_block);
    CU_ASSERT_EQUAL(ret, 0);
    CU_ASSERT_EQUAL(num_blocks, 2);
    CU_ASSERT_EQUAL(mutations_block[0], 0);
    CU_ASSERT_EQUAL(mutations_block[1], TSD_NULL);
    CU_ASSERT_EQUAL(mutations_block[2], 1);
    CU_ASSERT_EQUAL(blocks_stats[0], 1.0);
    CU_ASSERT_EQUAL(blocks_stats[1], 10.0);
    tsd_safe_free(blocks_stats);
    tsd_safe_free(blocks_edges);

    /* A haplotype edge is missing: the block is never completed */
    ret = tsd_block_singletons(2, 4, 3, unphased, phase_nodes_individual, mutations_node,
        mutations_position, child_missing, phase_edges_left, phase_edges_right,
        phase_order, phase_order, 10.0, &num_blocks, &blocks_stats, &blocks_edges,
        mutations_block);
    CU_ASSERT_EQUAL(ret, TSD_ERR_UNPHASED_INDIVIDUAL_EDGES);
    CU_ASSERT_PTR_NULL(blocks_stats);

    /* No edges and no individuals */
    ret = tsd_block_singletons(0, 0, 0, unphased, phase_nodes_individual, mutations_node,
        mutations_position, phase_edges_child, phase_edges_left, phase_edges_right,
        phase_order, phase_order, 10.0, &num_blocks, &blocks_stats, &blocks_edges,
        mutations_block);
    CU_ASSERT_EQUAL(ret, 0);
    CU_ASSERT_EQUAL(num_blocks, 0);
    tsd_safe_free(blocks_stats);
    tsd_safe_free(blocks_edges);
}

static void
test_reallocate_unphased(void)
{
    const tsd_id_t blocks_edges[] = { 0, 1 };
    const tsd_id_t mutations_block[] = { 0, TSD_NULL };
    const tsd_id_t mutations_block_bad[] = { 1, 0 };
    const tsd_id_t blocks_edges_bad[] = { 0, 7 };
    const double phase[] = { 0.25, NAN };
    const double phase_bad[] = { 1.5, 0.0 };
    const double phase_nan[] = { NAN, 0.0 };
    const double initial[] = { 1.0, 5.0, 0.0, 5.0 };
    double likelihood[4];
    int ret;

    memcpy(likelihood, initial, sizeof(initial));
    ret = tsd_reallocate_unphased(
        2, 2, 1, phase, mutations_block, blocks_edges, likelihood);
    CU_ASSERT_EQUAL(ret, 0);
    CU_ASSERT_DOUBLE_EQUAL(likelihood[0], 0.25, 1e-12);
    CU_ASSERT_DOUBLE_EQUAL(likelihood[2], 0.75, 1e-12);
    memcpy(likelihood, initial, sizeof(initial));
    ret = tsd_reallocate_unphased(
        2, 2, 1, phase_bad, mutations_block, blocks_edges, likelihood);
    CU_ASSERT_EQUAL(ret, TSD_ERR_PHASE_OUT_OF_RANGE);
    /* The mass of the skipped mutation is not allocated */
    memcpy(likelihood, initial, sizeof(initial));
    ret = tsd_reallocate_unphased(
        2, 2, 1, phase_nan, mutations_block, blocks_edges, likelihood);
    CU_ASSERT_EQUAL(ret, TSD_ERR_UNPHASED_LIKELIHOOD_NOT_CONSERVED);
    memcpy(likelihood, initial, sizeof(initial));
    ret = tsd_reallocate_unphased(
        2, 2, 1, phase, mutations_block_bad, blocks_edges, likelihood);
    CU_ASSERT_EQUAL(ret, TSD_ERR_BLOCK_OUT_OF_RANGE);
    memcpy(likelihood, initial, sizeof(initial));
    ret = tsd_reallocate_unphased(
        2, 2, 1, phase, mutations_block, blocks_edges_bad, likelihood);
    CU_ASSERT_EQUAL(ret, TSD_ERR_BLOCK_OUT_OF_RANGE);
}

static void
test_mutation_frequency(void)
{
    const uint8_t nodes_sample[] = { 1, 1, 0, 0, 0 };
    const tsd_id_t mutations_node[] = { 4, 0 };
    const double mutations_position[] = { 5, 1 };
    int32_t freq[2];
    int ret;

    ret = tsd_mutation_frequency(5, 1, 4, 2, nodes_sample, mutations_node,
        mutations_position, phase_edges_parent, phase_edges_child, phase_edges_left,
        phase_edges_right, phase_order, phase_order, 10.0, freq);
    CU_ASSERT_EQUAL(ret, 0);
    CU_ASSERT_EQUAL(freq[0], 2);
    CU_ASSERT_EQUAL(freq[1], 1);
}

static void
test_split_and_relabel(void)
{
    /* Node 1 has edges over [0, 2) and [4, 6): split into node 1 and a copy, 3 */
    const tsd_id_t parent[] = { 2, 2, 2 };
    const tsd_id_t child[] = { 0, 1, 1 };
    const double left[] = { 0, 0, 4 };
    const double right[] = { 6, 2, 6 };
    const uint8_t excluded[] = { 1, 0, 0 };
    const tsd_id_t insert[] = { 0, 1, 2 };
    const tsd_id_t remove[] = { 1, 0, 2 };
    const tsd_id_t mutations_node[] = { 1, 1 };
    const double mutations_position[] = { 1, 5 };
    tsd_id_t new_parent[3], new_child[3], output[2];
    tsd_id_t *nodes_order = NULL;
    tsd_size_t num_split = 0;
    int ret;

    ret = tsd_split_disjoint_nodes(3, 3, parent, child, left, right, excluded,
        new_parent, new_child, &num_split, &nodes_order);
    CU_ASSERT_EQUAL(ret, 0);
    CU_ASSERT_EQUAL(num_split, 1);
    CU_ASSERT_EQUAL(nodes_order[3], 1);
    CU_ASSERT_EQUAL(new_child[2], 3);
    ret = tsd_relabel_mutations_node(4, 3, 2, mutations_node, mutations_position,
        nodes_order, new_parent, new_child, left, right, insert, remove, output);
    CU_ASSERT_EQUAL(ret, 0);
    CU_ASSERT_EQUAL(output[0], 1);
    CU_ASSERT_EQUAL(output[1], 3);
    /* A mutation above a node that is not in any tree */
    {
        const tsd_id_t absent[] = { 3, 3 };

        ret = tsd_relabel_mutations_node(4, 3, 2, absent, mutations_position,
            nodes_order, new_parent, new_child, left, right, insert, remove, output);
        CU_ASSERT_EQUAL(ret, TSD_ERR_MUTATION_NODE_NOT_IN_TREE);
    }
    /* No edges: the mutations are not reached */
    ret = tsd_relabel_mutations_node(4, 0, 2, mutations_node, mutations_position,
        nodes_order, new_parent, new_child, left, right, insert, remove, output);
    CU_ASSERT_EQUAL(ret, 0);
    CU_ASSERT_EQUAL(output[1], TSD_NULL);
    tsd_safe_free(nodes_order);
}

static void
test_constrain_ages(void)
{
    const double times[] = { 0.0, 1.0, 0.5 };
    const uint8_t fixed[] = { 1, 1, 0 };
    const uint8_t all_fixed[] = { 1, 1, 1 };
    const uint8_t free_nodes[] = { 0, 0, 0 };
    const uint8_t fixed_child[] = { 1, 0, 0 };
    const tsd_id_t parent[] = { 1, 2 };
    const tsd_id_t child[] = { 0, 1 };
    double out[3];
    int ret;

    /* Node 2 is younger than its child, node 1, which is fixed */
    ret = tsd_constrain_ages(3, 2, times, fixed, parent, child, 1e-6, 10, out);
    CU_ASSERT_EQUAL(ret, 0);
    CU_ASSERT(out[2] > out[1]);
    /* Free nodes meet in the middle; a fixed parent or child does not move */
    ret = tsd_constrain_ages(3, 2, times, free_nodes, parent, child, 1e-6, 100, out);
    CU_ASSERT_EQUAL(ret, 0);
    ret = tsd_constrain_ages(3, 2, times, fixed_child, parent, child, 1e-6, 100, out);
    CU_ASSERT_EQUAL(ret, 0);
    /* Already consistent */
    ret = tsd_constrain_ages(3, 1, times, fixed, parent, child, 1e-6, 10, out);
    CU_ASSERT_EQUAL(ret, 0);
    /* Forcing without iterations */
    ret = tsd_constrain_ages(3, 2, times, fixed, parent, child, 1e-6, 0, out);
    CU_ASSERT_EQUAL(ret, 0);
    {
        const double bad[] = { 0.0, 1.0, 0.0 };

        ret = tsd_constrain_ages(3, 1, bad, all_fixed, child, parent, 1e-6, 10, out);
        CU_ASSERT_EQUAL(ret, TSD_ERR_FIXED_NODES_VIOLATE_CONSTRAINT);
    }
}

static void
test_contains_unary_nodes(void)
{
    const uint8_t no_mask[] = { 0, 0, 0, 0, 0 };
    const uint8_t all_mask[] = { 1, 1, 1, 1, 1 };
    const tsd_id_t unary_parent[] = { 4, 3 };
    const double left[] = { 0, 0 };
    const double right[] = { 10, 10 };
    const tsd_id_t order[] = { 0, 1 };
    int contains = -1;
    int ret;

    ret = tsd_contains_unary_nodes(5, 4, no_mask, phase_edges_parent, phase_edges_left,
        phase_edges_right, phase_order, phase_order, 10.0, &contains);
    CU_ASSERT_EQUAL(ret, 0);
    CU_ASSERT_EQUAL(contains, 0);
    ret = tsd_contains_unary_nodes(
        5, 2, no_mask, unary_parent, left, right, order, order, 10.0, &contains);
    CU_ASSERT_EQUAL(ret, 0);
    CU_ASSERT_EQUAL(contains, 1);
    ret = tsd_contains_unary_nodes(
        5, 2, all_mask, unary_parent, left, right, order, order, 10.0, &contains);
    CU_ASSERT_EQUAL(ret, 0);
    CU_ASSERT_EQUAL(contains, 0);
}

static void
test_conditional_coalescent_variance(void)
{
    const double val[] = { 1.0, 2.0, 3.0, 4.0 };
    double out[6], variance[11];
    int ret;
    int j;

    ret = tsd_marginalize_over_ancestors(2, 2, val, out);
    CU_ASSERT_EQUAL(ret, 0);
    CU_ASSERT_EQUAL(out[4], 3.0);
    ret = tsd_conditional_coalescent_variance(1, variance);
    CU_ASSERT_EQUAL(ret, 0);
    CU_ASSERT_EQUAL(variance[1], 0.0);
    ret = tsd_conditional_coalescent_variance(10, variance);
    CU_ASSERT_EQUAL(ret, 0);
    for (j = 2; j <= 10; j++) {
        CU_ASSERT(variance[j] > 0.0);
    }
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
        { "test_approximate_log_moments", test_approximate_log_moments },
        { "test_approximate_gamma_kl", test_approximate_gamma_kl },
        { "test_approximate_gamma_mom", test_approximate_gamma_mom },
        { "test_approximate_gamma_iqr", test_approximate_gamma_iqr },
        { "test_average_gammas", test_average_gammas },
        { "test_valid", test_valid },
        { "test_node_moments", test_node_moments },
        { "test_node_projections", test_node_projections },
        { "test_mutation_moments", test_mutation_moments },
        { "test_mutation_projections", test_mutation_projections },
        { "test_block_singletons", test_block_singletons },
        { "test_reallocate_unphased", test_reallocate_unphased },
        { "test_mutation_frequency", test_mutation_frequency },
        { "test_split_and_relabel", test_split_and_relabel },
        { "test_constrain_ages", test_constrain_ages },
        { "test_contains_unary_nodes", test_contains_unary_nodes },
        { "test_conditional_coalescent_variance", test_conditional_coalescent_variance },
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
