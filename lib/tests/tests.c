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
 * Smoke tests for the low-level tsdate API: return codes, error paths and memory
 * handling, with coarse sanity checks of results (finite, NaN, simple values).
 * Numerical correctness is tested from Python, except for the expectation
 * propagation passes on states set directly in the library's struct, and the
 * rescaling, prior and damping functions, which have tables of expected values.
 */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <CUnit/Basic.h>

#include "tsdate.h"

/* Tolerance of the comparisons with the reference implementation */
#define ULPS 4

/* The bits of `x` as an integer that increases with `x` (with -0 equal to 0). */
static int64_t
ordered_bits(double x)
{
    int64_t bits;

    memcpy(&bits, &x, sizeof(bits));
    if (bits < 0) {
        bits = INT64_MIN - bits;
    }
    return bits;
}

/* `actual` is within `ulps` units in the last place of `expected`; NaNs are equal. */
static void
assert_close_ulps(double actual, double expected, int ulps)
{
    int64_t a, b;
    uint64_t distance;

    if (isnan(expected)) {
        CU_ASSERT_TRUE(isnan(actual));
    } else if (!isfinite(expected) || !isfinite(actual)) {
        CU_ASSERT_EQUAL(actual, expected);
    } else {
        a = ordered_bits(actual);
        b = ordered_bits(expected);
        distance = a > b ? (uint64_t) a - (uint64_t) b : (uint64_t) b - (uint64_t) a;
        CU_ASSERT_TRUE(distance <= (uint64_t) ulps);
    }
}

/* |actual - expected| <= atol + rtol * |expected|, as numpy.isclose; NaNs are equal. */
static void
assert_close_tol(double actual, double expected, double rtol, double atol)
{
    if (isnan(expected)) {
        CU_ASSERT_TRUE(isnan(actual));
    } else if (!isfinite(expected)) {
        CU_ASSERT_EQUAL(actual, expected);
    } else {
        CU_ASSERT_TRUE(fabs(actual - expected) <= atol + rtol * fabs(expected));
    }
}

static void
assert_array_close(const double *actual, const double *expected, tsd_size_t n)
{
    tsd_size_t j;

    for (j = 0; j < n; j++) {
        assert_close_ulps(actual[j], expected[j], ULPS);
    }
}

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
        TSD_ERR_UNPHASED_INDIVIDUAL_EDGES, TSD_ERR_BLOCK_OUT_OF_BOUNDS,
        TSD_ERR_PHASE_OUT_OF_RANGE, TSD_ERR_UNPHASED_LIKELIHOOD_NOT_CONSERVED,
        TSD_ERR_MUTATION_NODE_NOT_IN_TREE, TSD_ERR_FIXED_NODES_VIOLATE_CONSTRAINT,
        TSD_ERR_NONPOSITIVE_INTERVALS, TSD_ERR_NEGATIVE_CHANGEPOINT_PARAMETER,
        TSD_ERR_ZERO_EDGE_SPAN, TSD_ERR_NON_INCREASING_BREAKPOINTS,
        TSD_ERR_INVALID_QUANTILE_WIDTH, TSD_ERR_INVALID_POSTERIOR,
        TSD_ERR_TIME_BELOW_FIRST_BREAKPOINT, TSD_ERR_NO_POSTERIORS,
        TSD_ERR_INVALID_MAX_SHAPE, TSD_ERR_INVALID_MIN_STEP,
        TSD_ERR_INVALID_NODE_POSTERIOR, TSD_ERR_INVALID_DAMPING,
        TSD_ERR_TWIN_EDGE_IN_PHASED_PASS, TSD_ERR_NONPOSITIVE_PENALTY,
        TSD_ERR_NODE_OUT_OF_BOUNDS, TSD_ERR_EDGE_OUT_OF_BOUNDS,
        TSD_ERR_INDIVIDUAL_OUT_OF_BOUNDS, TSD_ERR_BAD_EDGE_INTERVAL,
        TSD_ERR_BAD_EDGE_INDEXES, TSD_ERR_EDGES_FORM_CYCLE, TSD_ERR_TOO_FEW_TIPS,
        TSD_ERR_NO_BREAKPOINTS };
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

/* Special functions: arguments, expected values and return codes */
typedef struct {
    double x;
    double expected;
    int ret;
} sf_case1_t;

typedef struct {
    double p, x;
    double expected;
    int ret;
} sf_case2_t;

typedef struct {
    double a, b, x;
    double expected;
    int ret;
} sf_case3_t;

typedef struct {
    double a, b, c, x;
    double expected;
    int ret;
} sf_case4_t;

typedef struct {
    double a, b, x;
    double value, dvalue;
    int ret;
} sf_hyperu_case_t;

/* Special functions: arguments and reference values */
typedef struct {
    double x;
    double expected;
} sf_ref1_t;

typedef struct {
    double x, y;
    double expected;
} sf_ref2_t;

typedef struct {
    double a, b, x;
    double expected;
} sf_ref3_t;

typedef struct {
    double a, b, c, x;
    double expected;
} sf_ref4_t;

typedef struct {
    double a, b, x;
    double value, dvalue;
} sf_hyperu_ref_t;

/* Tolerances of the comparisons with mpmath */
#define POLYGAMMA_RTOL 1e-5
#define POLYGAMMA_ATOL 1e-8
#define LAPLACE_RTOL   2e-2
#define LAPLACE_ATOL   1e-8
/* The unity limit of hyp2f1 is within 1% of the exact value */
#define UNITY_RTOL 1e-2
/* The derivative of the Laplace approximation of hyperu is within 2.5% of the exact */
#define HYPERU_DERIVATIVE_RTOL 2.5e-2
/* Tolerance of the comparison with SciPy, whose results depend on its build */
#define GAMMAINCINV_RTOL 1e-10

/* Expected values of the reference implementation */
static const sf_case1_t digamma_cases[] = {
    { 0.0, 0.0, TSD_ERR_ZERO_DIVISION },
    { -0.0, 0.0, TSD_ERR_ZERO_DIVISION },
    { 4.9406564584124654e-324, -INFINITY, 0 },
    { 9.9999999999999995e-07, -1000000.5772156649, 0 },
    { 1.0000000000000001e-05, -100000.57721566489, 0 },
    { 1.0000000000000003e-05, -100000.57719921565, 0 },
    { 0.5, -1.9635100260214153, 0 },
    { 4.9000000000000004, 1.483737793254901, 0 },
    { 8.4999999999999982, 2.0800908175794208, 0 },
    { 8.5, 2.0800908175794279, 0 },
    { 100.0, 4.6001618527380881, 0 },
    { 1.0000000000000001e+300, 690.77552789821368, 0 },
    { INFINITY, INFINITY, 0 },
    { NAN, NAN, 0 },
    { -0.5, 0.036489973978584861, 0 },
    { -1.0, -25653050788007548.0, 0 },
    { -1.5, 0.70315664064525185, 0 },
    { -10.5, 2.3982391295357801, 0 },
    { -1e-300, 1.0000000000000001e+300, 0 },
    { -INFINITY, NAN, 0 },
};

/* Expected values of the reference implementation */
static const sf_case1_t trigamma_cases[] = {
    { 0.0, 0.0, TSD_ERR_ZERO_DIVISION },
    { -0.0, 0.0, TSD_ERR_ZERO_DIVISION },
    { -1e-300, 0.0, TSD_ERR_ZERO_DIVISION },
    { 1e-300, 0.0, TSD_ERR_ZERO_DIVISION },
    { 4.9406564584124654e-324, 0.0, TSD_ERR_ZERO_DIVISION },
    { 1e-10, 9.9999999999999984e+19, 0 },
    { 0.0001, 100000000.0, 0 },
    { 0.00010000000000000002, 100000001.64469366, 0 },
    { 0.5, 4.9348022005461569, 0 },
    { 4.9000000000000004, 0.22631146419213941, 0 },
    { 5.0, 0.22132295574429867, 0 },
    { 100.0, 0.010050166663333571, 0 },
    { 1.0000000000000001e+300, 1e-300, 0 },
    { INFINITY, 0.0, 0 },
    { NAN, NAN, 0 },
    { -0.5, 8.9348022005432011, 0 },
    { -1.5, 9.3792466449876457, 0 },
    { -10.5, 9.7787577398148109, 0 },
    { -INFINITY, NAN, 0 },
};

/* Expected values of the reference implementation */
static const sf_ref2_t betaln_cases[] = {
    { 1.0, 1.0, 0.0 },
    { 0.5, 0.5, 1.1447298858494002 },
    { 2.0, 3.0, -2.4849066497880004 },
    { 1e-10, 10000000000.0, 23.025848388671875 },
    { 100000.0, 100000.0, -138633.92706134822 },
    { 1.0000000000000001e+300, 1.0000000000000001e+300, -1.3862943611198268e+300 },
    { 1e+308, 1e+308, NAN },
    { 0.0, 1.0, INFINITY },
    { -0.5, 2.0, 1.3862943611198906 },
    { NAN, 1.0, NAN },
    { 1.0, INFINITY, NAN },
};

/* Expected values of the reference implementation */
static const sf_case2_t gammainc_der_cases[] = {
    { 1.0, 0.0, 0.0, 0 },
    { 5.0, 100000000.00000001, 0.0, 0 },
    { 5.0, INFINITY, 0.0, 0 },
    { 2000.0, 2000.0, -0.00892111615358839, 0 },
    { 1000.0000000000001, 1000.0, -0.012617064253050618, 0 },
    { 1000.0, 1000.0, -0.012616713994069394, 0 },
    { 0.5, 1.0, -0.3898372643285174, 0 },
    { 2.0, 1.5, -0.31348863782508085, 0 },
    { 1e-10, 1e-10, -22.448635214909299, 0 },
    { 50.0, 0.001, 0.0, 0 },
    { 0.5, 1.0000000000000002, -0.3898372643285154, 0 },
    { 5.0, 50.0, -1.3227071908086766e-16, 0 },
    { 100.0, 110.0, -0.023808656854890219, 0 },
    { 0.001, 10.0, -4.1816615862918354e-06, 0 },
    { 5.0, 800.0, 0.0, 0 },
    { -1.0, 1.0, 0.0, TSD_ERR_INVALID_HYPERGEO },
    { 0.0, 1.0, 0.0, TSD_ERR_INVALID_HYPERGEO },
    { 1.0, -4.9406564584124654e-324, 0.0, TSD_ERR_INVALID_HYPERGEO },
    { NAN, 1.0, 0.0, TSD_ERR_INVALID_HYPERGEO },
    { 1.0, NAN, 0.0, TSD_ERR_INVALID_HYPERGEO },
};

/* Expected values of the reference implementation */
static const sf_hyperu_case_t hyperu_laplace_cases[] = {
    { 1.0, 2.0, 1.0, 1.1102230246251565e-16, -1.0, 0 },
    { 1.0, 1.0, 1.0, -0.5016052919180235, -0.6944271909999159, 0 },
    { 0.5, 30.0, 0.001, 267.79385089219738, -28998.983352647308, 0 },
    { 50.0, 51.0, 1000.0, -345.38776394910684, -0.049999999999999996, 0 },
    { 0.001, 1.0, 100000000.0, -0.018420680743962414, -9.9999999998999994e-12, 0 },
    { 1.0, INFINITY, 1.0, NAN, NAN, 0 },
    { 4.9406564584124654e-324, 1.0, 1.0, 0.0, -4.9406564584124654e-324, 0 },
    { 1.0, 0.99999999999999989, 1.0, 0.0, 0.0, TSD_ERR_INVALID_HYPERGEO },
    { 0.0, 1.0, 1.0, 0.0, 0.0, TSD_ERR_INVALID_HYPERGEO },
    { NAN, 2.0, 1.0, 0.0, 0.0, TSD_ERR_INVALID_HYPERGEO },
    { 1.0, 2.0, 0.0, 0.0, 0.0, TSD_ERR_INVALID_HYPERGEO },
    { 1.0, 2.0, NAN, 0.0, 0.0, TSD_ERR_INVALID_HYPERGEO },
    { 1.0, 2.0, 1.0000000000000001e+300, 0.0, 0.0, TSD_ERR_ZERO_DIVISION },
    { 1.0, 2.0, INFINITY, 0.0, 0.0, TSD_ERR_ZERO_DIVISION },
    { 1.0, 2.0, 4.9406564584124654e-324, 0.0, 0.0, TSD_ERR_ZERO_DIVISION },
    { 4.9406564584124654e-324, 1.0, 10.0, 0.0, 0.0, TSD_ERR_ZERO_DIVISION },
};

/* Expected values of the reference implementation */
static const sf_case3_t hyp1f1_laplace_cases[] = {
    { 1.0, 2.0, 1.0, 0.5335775438918825, 0 },
    { 1.0, 2.0, 0.0, 0.0, 0 },
    { 1.0, 2.0, -0.0, 0.0, 0 },
    { 0.5, 30.0, -1000.0, -1.7799338025575939, 0 },
    { 50.0, 1050.0, 1000.0, 97.194122480810705, 0 },
    { 0.001, 1.0009999999999999, 1e-10, 9.9826374162340947e-14, 0 },
    { 1.0, 2.0, NAN, NAN, 0 },
    { 1.0, 2.0, INFINITY, NAN, 0 },
    { 1.0, 2.0, 1.0000000000000001e+300, -INFINITY, 0 },
    { 1.0, 1.0, 1.0, 0.0, TSD_ERR_INVALID_HYPERGEO },
    { 0.0, 1.0, 0.0, 0.0, TSD_ERR_INVALID_HYPERGEO },
    { NAN, 2.0, 1.0, 0.0, TSD_ERR_INVALID_HYPERGEO },
    { 1e-10, 2.0000000000000001e-10, 100000000.0, 0.0, TSD_ERR_ZERO_DIVISION },
};

/* Expected values of the reference implementation */
static const sf_case4_t hyp2f1_laplace_cases[] = {
    { 1.0, 2.0, 3.0, 0.5, 0.43844776148964582, 0 },
    { 5.0, 7.0, 9.0, 0.90000000000000002, 8.1454130153612532, 0 },
    { 30.0, 40.0, 35.0, 1e-08, 3.4285713113035854e-07, 0 },
    { 0.0, 1.0, 1.0, 0.0, 0.0, 0 },
    { 1.0, 2.0, 3.0, -0.5, -0.27842079965849342, 0 },
    { 0.5, 30.0, 1.5, -100.0, -4.0851309087682068, 0 },
    { 2.0, 3.0, 4.0, 0.99999998999999995, 19.519293027595719, 0 },
    { 1.0, 2.0, 5.0, 0.99999998999999995, 0.69314718055994551, 0 },
    { 1.0, 2.0, 3.0, 0.99999998999999995, 3.6066211672149588, 0 },
    { 2.0, 3.0, 4.0, -100000000.0, -35.742749223159137, 0 },
    { 3.0, 2.0, 5.0, -100000000.0, -35.049602038676674, 0 },
    { 2.0, 2.0, 5.0, -100000000.0, -31.44298087085901, 0 },
    { 1.0, 2.0, 3.0, -10000000000000000.0, 0.0, TSD_ERR_INVALID_HYPERGEO },
    { 1.0, 2.0, 0.0, 0.5, 0.0, TSD_ERR_INVALID_HYPERGEO },
    { -1.0, 2.0, 3.0, 0.5, 0.0, TSD_ERR_INVALID_HYPERGEO },
    { 1.0, -1.0, 3.0, 0.5, 0.0, TSD_ERR_INVALID_HYPERGEO },
    { 2.0, 1.0, 1.0, 0.5, 0.0, TSD_ERR_INVALID_HYPERGEO },
    { 1.0, 2.0, 3.0, 1.0, 0.0, TSD_ERR_INVALID_HYPERGEO },
    { 1.0, 2.0, 3.0, NAN, 0.0, TSD_ERR_INVALID_HYPERGEO },
    { 0.0, 1.0, 1.0, 0.5, 0.0, TSD_ERR_INVALID_HYPERGEO },
    { 1.0, 2.0, 1.0, 0.5, 0.0, TSD_ERR_INVALID_HYPERGEO },
    { 0.0, 1.0, 1e-300, 0.5, 0.0, TSD_ERR_ZERO_DIVISION },
    { 1.1774154985086274, 504.40824624062162, 1.1774154985086274, 0.14271801659243741,
        0.0, TSD_ERR_ZERO_DIVISION },
};

/* scipy.special.gammaincinv (SciPy 1.18.1); 0 for p = 0, infinity for p = 1 and NaN
 * outside the domain */
static const sf_ref2_t gammaincinv_cases[] = {
    { 0.99999999999999989, 0.5, 0.69314718055994506 },
    { 0.0018725572425875394, 0.93432222410629528, 9.8708778280225384e-17 },
    { 1.4851015014453666e-07, 0.99989259056538193, 4.2708929514115105e-315 },
    { 9.9999999999999694e-311, 0.95988221384026817, NAN },
    { 9.9999999999999694e-311, 0.56112084484672486, NAN },
    { 1.0, 0.64358723829337949, 1.0316657767689139 },
    { 1.3693230946091463e+32, 0.84677972577643534, 1.3693230946091465e+32 },
    { 3.3463277181277915, 0.99999999999894673, 34.949768724325295 },
    { 22.041778641418698, 0.093156553752253735, 16.126803358230266 },
    { 1e-10, 0.99999999999999989, 11.214230215647511 },
    { 2.0, 4.9406564584124654e-324, 3.1434555694052403e-162 },
    { 1000.0, 1e-300, 233.92836429052818 },
    { 1000.0, 0.999, 1100.5780982933145 },
    { 0.67948734660549959, 0.7818881580083944, 1.0450131885835172 },
    { 0.024285952691655305, 0.99728941525830839, 1.4475359182943457 },
    { 0.033000366354340456, 0.98377018057938537, 0.5706929538725517 },
    { 32.69007854626318, 0.99999999999999645, 98.692064965782848 },
    { 1e-300, 0.99999999989999999, 0.0 },
    { 1.0000000000000001e+300, 4.9406564584124654e-324, 1.0000000000000001e+300 },
    { 1e-300, 0.0, 0.0 },
    { 1e-300, 1.0, INFINITY },
    { 1e-300, 1.0000000000000002, NAN },
    { 1.0, 4.9406564584124654e-324, 4.9406564584124654e-324 },
    { 1000000.0, 9.9999999999999995e-21, 990765.90325837862 },
    { INFINITY, 4.9406564584124654e-324, NAN },
    { 2907.5441489134673, 0.85784200187994619, 2965.322214703659 },
    { 64.283331812787964, 0.98660725921411685, 83.323363883964191 },
    { 0.0024060602718259248, 0.17939292031798137, 4.1575279095986522e-311 },
    { 1e-300, -0.10000000000000001, NAN },
    { 1e-10, 0.99999999989999999, 0.2647369819114741 },
    { 0.5, 0.999, 5.4137830853313664 },
    { 100.0, 1e-300, 0.038006988916941875 },
    { 1000.0, 0.99999999989999999, 1214.4995558668779 },
    { 0.0, 4.9406564584124654e-324, NAN },
    { -1.0, 0.0, NAN },
    { NAN, 0.5, NAN },
    { 2.0, NAN, NAN },
    { 0.0001, 0.25, 0.0 },
    { 0.5, 0.25, 0.050765522133810782 },
    { 3.0, 0.94999999999999996, 6.2957936218719883 },
};

/* mpmath psi(0, x) at 15 digits */
static const sf_ref1_t digamma_mpmath[] = {
    { -0.29999999999999999, 2.113309779635399 },
    { 1e-10, -10000000000.577215 },
    { 9.9999999999999995e-07, -1000000.57721402 },
    { 0.01, -100.56088545786866 },
    { 10.0, 2.2517525890667209 },
    { 100.0, 4.6001618527380872 },
    { 1000.0, 6.907255195648812 },
    { 100000.0, 11.512920464961896 },
    { 10000000000.0, 23.025850929890456 },
};

/* mpmath psi(1, x) at 15 digits */
static const sf_ref1_t trigamma_mpmath[] = {
    { -0.29999999999999999, 13.945160267805722 },
    { 1e-10, 1e+20 },
    { 9.9999999999999995e-07, 1000000000001.645 },
    { 0.01, 10001.621213528313 },
    { 10.0, 0.10516633568168575 },
    { 100.0, 0.010050166663333571 },
    { 1000.0, 0.0010005001666666333 },
    { 100000.0, 1.0000050000166667e-05 },
    { 10000000000.0, 1.00000000005e-10 },
};

/* Real part of mpmath log(beta(p, q)) at 15 digits */
static const sf_ref2_t betaln_mpmath[] = {
    { -0.29999999999999999, -0.59999999999999998, 0.41427032493334498 },
    { 1e-10, 2.0000000000000001e-10, 23.431316038048621 },
    { 9.9999999999999995e-07, 1.9999999999999999e-06, 14.220975666069149 },
    { 0.01, 0.02, 5.0103133506979862 },
    { 10.0, 20.0, -19.115327299887046 },
    { 100.0, 200.0, -192.13419227497894 },
    { 1000.0, 2000.0, -1911.8746142144516 },
    { 100000.0, 200000.0, -190958.88527911686 },
    { 10000000000.0, 20000000000.0, -19095425059.235638 },
};

/* Log of mpmath hyp2f1 at 15 digits */
static const sf_ref4_t hyp2f1_laplace_mpmath[] = {
    { 984.72000000000003, 1954.22, 985.72000000000003, -1.1392534691231597,
        -1478.1738051381913 },
    { 936.02999999999997, 1834.8299999999999, 938.02999999999997, -1.0739817123857025,
        -1330.0059628227482 },
    { 413.54000000000002, 451.48000000000002, 414.54000000000002, -12.853557923685116,
        -1181.3138708305196 },
    { 544.64999999999998, 1292.75, 545.64999999999998, -0.71439839463641541,
        -693.46877093725595 },
    { 21.898, 43.078000000000003, 22.898, -1.2310276509604619, -31.917963510674785 },
    { 1000.0, 1007.931, 1001.0, -129.62015503875969, -4904.8528786755833 },
    { 42.006999999999998, 119.15700000000001, 43.006999999999998, -0.70922612322041212,
        -59.975223456559483 },
    { 130.68000000000001, 1122.1800000000001, 132.68000000000001, -0.1533946760574047,
        -152.42035404495687 },
    { 121.70999999999999, 802.61000000000001, 124.70999999999999, -0.80594900849858353,
        -303.45631195318833 },
};

/* Log of mpmath hyp2f1 at 15 digits, for z / (z - 1) close to one */
static const sf_ref4_t hyp2f1_laplace_unity_mpmath[] = {
    { 0.0012723999999999999, 202.86627240000001, 1.0012724, -6614020190.5935307,
        -0.036261751566401573 },
    { 0.0012612000000000001, 1001.9992612000001, 3.0012612000000001, -141574796.04867145,
        -0.031220448825817738 },
    { 0.0011992999999999999, 157.77919929999999, 1.0011992999999999, -1146235430.4403243,
        -0.031774253197474109 },
    { 0.00097179999999999999, 1000.0009718, 1.0009718000000001, -158137496.57453805,
        -0.025619221224780213 },
    { 0.00097110000000000002, 405.48297110000004, 1.0009710999999999,
        -276526839.68553674, -0.025266157355002965 },
    { 0.0015548999999999999, 115.7745549, 1.0015548999999999, -31061255169.177708,
        -0.045842304967148521 },
    { 0.0010028000000000001, 73.491902799999991, 1.0010028, -53058551695.758659,
        -0.029644174580337579 },
    { 0.0015053, 19.999605300000002, 2.0015052999999998, -178245782075.18933,
        -0.042831241419712583 },
};

/* Log of mpmath hyperu at 15 digits, and its derivative with respect to x */
static const sf_hyperu_ref_t hyperu_laplace_mpmath[] = {
    { 1.0, 4.3380000000000001, 0.7752291021399601, 2.6258894260525927,
        -3.3991838624960096 },
    { 1.0, 4.6779999999999999, 0.76981102428276016, 3.1203053508031515,
        -3.8351394948119681 },
    { 2.0, 5.9990000000000006, 5.1995955076999998, -2.202169012479116,
        -0.57735145331711235 },
    { 9.0, 14.251000000000001, 2.5473150302860001, -1.5490292494382205,
        -4.9021077003749038 },
    { 4.0, 18.460000000000001, 11.246026052960001, -3.7214576615414909,
        -0.90331659933332087 },
    { 2.0, 2.9990000000000001, 3.7769370770299999, -2.6582253243210303,
        -0.52944685391153801 },
};

/* Log of mpmath hyp1f1 at 15 digits */
static const sf_ref3_t hyp1f1_laplace_mpmath[] = {
    { 132.47999999999999, 133.47999999999999, 170.98504810547999, 170.15807254764488 },
    { 997.09000000000003, 999.09000000000003, 1492.6979568568001, 1490.8690564465708 },
    { 978.24000000000001, 979.24000000000001, 1031.693903284, 1030.9740566223463 },
    { 713.51999999999998, 722.51999999999998, 739.07796118750002, 732.72108850452321 },
    { 999.99000000000001, 1003.99, 2916.7083719407506, 2911.2536982489523 },
    { 989.08000000000004, 992.08000000000004, 1125.8335771375, 1123.5567170478762 },
    { 28.545999999999999, 29.545999999999999, 38.4316856035, 37.587465253533537 },
    { 30.956, 31.956, 28.834237509375001, 28.184054387129486 },
};

static void
test_digamma(void)
{
    const size_t num_cases = sizeof(digamma_cases) / sizeof(*digamma_cases);
    const size_t num_refs = sizeof(digamma_mpmath) / sizeof(*digamma_mpmath);
    const sf_case1_t *row;
    double value;
    size_t j;
    int ret;

    for (j = 0; j < num_cases; j++) {
        row = &digamma_cases[j];
        ret = tsd_digamma(row->x, &value);
        CU_ASSERT_EQUAL(ret, row->ret);
        if (ret == 0) {
            assert_close_ulps(value, row->expected, ULPS);
        }
    }
    for (j = 0; j < num_refs; j++) {
        ret = tsd_digamma(digamma_mpmath[j].x, &value);
        CU_ASSERT_EQUAL_FATAL(ret, 0);
        assert_close_tol(
            value, digamma_mpmath[j].expected, POLYGAMMA_RTOL, POLYGAMMA_ATOL);
    }
}

static void
test_trigamma(void)
{
    const size_t num_cases = sizeof(trigamma_cases) / sizeof(*trigamma_cases);
    const size_t num_refs = sizeof(trigamma_mpmath) / sizeof(*trigamma_mpmath);
    const sf_case1_t *row;
    double value;
    size_t j;
    int ret;

    for (j = 0; j < num_cases; j++) {
        row = &trigamma_cases[j];
        ret = tsd_trigamma(row->x, &value);
        CU_ASSERT_EQUAL(ret, row->ret);
        if (ret == 0) {
            assert_close_ulps(value, row->expected, ULPS);
        }
    }
    for (j = 0; j < num_refs; j++) {
        ret = tsd_trigamma(trigamma_mpmath[j].x, &value);
        CU_ASSERT_EQUAL_FATAL(ret, 0);
        assert_close_tol(
            value, trigamma_mpmath[j].expected, POLYGAMMA_RTOL, POLYGAMMA_ATOL);
    }
}

static void
test_betaln(void)
{
    const size_t num_cases = sizeof(betaln_cases) / sizeof(*betaln_cases);
    const size_t num_refs = sizeof(betaln_mpmath) / sizeof(*betaln_mpmath);
    const sf_ref2_t *row;
    double value;
    size_t j;

    for (j = 0; j < num_cases; j++) {
        row = &betaln_cases[j];
        value = tsd_betaln(row->x, row->y);
        assert_close_ulps(value, row->expected, ULPS);
    }
    for (j = 0; j < num_refs; j++) {
        row = &betaln_mpmath[j];
        value = tsd_betaln(row->x, row->y);
        assert_close_tol(value, row->expected, POLYGAMMA_RTOL, POLYGAMMA_ATOL);
    }
}

static void
test_gammainc_der(void)
{
    const size_t num_cases = sizeof(gammainc_der_cases) / sizeof(*gammainc_der_cases);
    const sf_case2_t *row;
    double value;
    size_t j;
    int ret;

    for (j = 0; j < num_cases; j++) {
        row = &gammainc_der_cases[j];
        ret = tsd_gammainc_der(row->p, row->x, &value);
        CU_ASSERT_EQUAL(ret, row->ret);
        if (ret == 0) {
            assert_close_ulps(value, row->expected, ULPS);
        }
    }
}

static void
test_hyperu_laplace(void)
{
    const size_t num_cases
        = sizeof(hyperu_laplace_cases) / sizeof(*hyperu_laplace_cases);
    const size_t num_refs
        = sizeof(hyperu_laplace_mpmath) / sizeof(*hyperu_laplace_mpmath);
    const sf_hyperu_case_t *row;
    const sf_hyperu_ref_t *ref;
    double value, dvalue;
    size_t j;
    int ret;

    for (j = 0; j < num_cases; j++) {
        row = &hyperu_laplace_cases[j];
        ret = tsd_hyperu_laplace(row->a, row->b, row->x, &value, &dvalue);
        CU_ASSERT_EQUAL(ret, row->ret);
        if (ret == 0) {
            assert_close_ulps(value, row->value, ULPS);
            assert_close_ulps(dvalue, row->dvalue, ULPS);
        }
    }
    for (j = 0; j < num_refs; j++) {
        ref = &hyperu_laplace_mpmath[j];
        ret = tsd_hyperu_laplace(ref->a, ref->b, ref->x, &value, &dvalue);
        CU_ASSERT_EQUAL_FATAL(ret, 0);
        assert_close_tol(value, ref->value, LAPLACE_RTOL, LAPLACE_ATOL);
        assert_close_tol(dvalue, ref->dvalue, HYPERU_DERIVATIVE_RTOL, LAPLACE_ATOL);
    }
}

static void
test_hyp1f1_laplace(void)
{
    const size_t num_cases
        = sizeof(hyp1f1_laplace_cases) / sizeof(*hyp1f1_laplace_cases);
    const size_t num_refs
        = sizeof(hyp1f1_laplace_mpmath) / sizeof(*hyp1f1_laplace_mpmath);
    const sf_case3_t *row;
    const sf_ref3_t *ref;
    double value;
    size_t j;
    int ret;

    for (j = 0; j < num_cases; j++) {
        row = &hyp1f1_laplace_cases[j];
        ret = tsd_hyp1f1_laplace(row->a, row->b, row->x, &value);
        CU_ASSERT_EQUAL(ret, row->ret);
        if (ret == 0) {
            assert_close_ulps(value, row->expected, ULPS);
        }
    }
    for (j = 0; j < num_refs; j++) {
        ref = &hyp1f1_laplace_mpmath[j];
        ret = tsd_hyp1f1_laplace(ref->a, ref->b, ref->x, &value);
        CU_ASSERT_EQUAL_FATAL(ret, 0);
        assert_close_tol(value, ref->expected, LAPLACE_RTOL, LAPLACE_ATOL);
    }
}

/* Compare tsd_hyp2f1_laplace with `num_refs` mpmath values to within `rtol`. */
static void
check_hyp2f1_laplace_mpmath(const sf_ref4_t *refs, size_t num_refs, double rtol)
{
    double value;
    size_t j;
    int ret;

    for (j = 0; j < num_refs; j++) {
        ret = tsd_hyp2f1_laplace(refs[j].a, refs[j].b, refs[j].c, refs[j].x, &value);
        CU_ASSERT_EQUAL_FATAL(ret, 0);
        assert_close_tol(value, refs[j].expected, rtol, LAPLACE_ATOL);
    }
}

static void
test_hyp2f1_laplace(void)
{
    const size_t num_cases
        = sizeof(hyp2f1_laplace_cases) / sizeof(*hyp2f1_laplace_cases);
    const size_t num_refs
        = sizeof(hyp2f1_laplace_mpmath) / sizeof(*hyp2f1_laplace_mpmath);
    const size_t num_unity_refs
        = sizeof(hyp2f1_laplace_unity_mpmath) / sizeof(*hyp2f1_laplace_unity_mpmath);
    const sf_case4_t *row;
    double value;
    size_t j;
    int ret;

    for (j = 0; j < num_cases; j++) {
        row = &hyp2f1_laplace_cases[j];
        ret = tsd_hyp2f1_laplace(row->a, row->b, row->c, row->x, &value);
        CU_ASSERT_EQUAL(ret, row->ret);
        if (ret == 0) {
            assert_close_ulps(value, row->expected, ULPS);
        }
    }
    check_hyp2f1_laplace_mpmath(hyp2f1_laplace_mpmath, num_refs, LAPLACE_RTOL);
    check_hyp2f1_laplace_mpmath(hyp2f1_laplace_unity_mpmath, num_unity_refs, UNITY_RTOL);
}

static void
test_gammaincinv(void)
{
    const size_t num_cases = sizeof(gammaincinv_cases) / sizeof(*gammaincinv_cases);
    const sf_ref2_t *row;
    double value;
    size_t j;

    for (j = 0; j < num_cases; j++) {
        row = &gammaincinv_cases[j];
        value = tsd_gammaincinv(row->x, row->y);
        assert_close_tol(value, row->expected, GAMMAINCINV_RTOL, 0);
    }
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
    int ret;

    /* Newton iteration */
    ret = tsd_approximate_gamma_iqr(0.25, 0.75, 1.0, 3.0, 50.0, out);
    CU_ASSERT_EQUAL(ret, 0);
    CU_ASSERT(isfinite(out[0]) && isfinite(out[1]));
    /* capped shape: equal quantiles, or a lower bound above the cap */
    ret = tsd_approximate_gamma_iqr(0.25, 0.75, 3.0, 3.0, 50.0, out);
    CU_ASSERT_EQUAL(ret, 0);
    CU_ASSERT_EQUAL(out[0], 49.0);
    ret = tsd_approximate_gamma_iqr(0.25, 0.75, 1.0, 1.1, 2.0, out);
    CU_ASSERT_EQUAL(ret, 0);
    CU_ASSERT_EQUAL(out[0], 1.0);
    /* errors */
    ret = tsd_approximate_gamma_iqr(0.25, 0.75, 2.0, 1.0, 50.0, out);
    CU_ASSERT_EQUAL(ret, TSD_ERR_UNSORTED_QUANTILES);
    ret = tsd_approximate_gamma_iqr(0.25, 0.75, 0.0, 0.0, 50.0, out);
    CU_ASSERT_EQUAL(ret, TSD_ERR_ZERO_DIVISION);
    ret = tsd_approximate_gamma_iqr(0.0, 0.75, 1.0, 2.0, 50.0, out);
    CU_ASSERT_EQUAL(ret, TSD_ERR_ZERO_DIVISION);
    ret = tsd_approximate_gamma_iqr(0.25, 0.75, 0.0, 1.0, 50.0, out);
    CU_ASSERT_EQUAL(ret, TSD_ERR_ZERO_DIVISION);
    /* no convergence, or a step outside the domain of the incomplete gamma
     * derivative */
    ret = tsd_approximate_gamma_iqr(1e-200, 0.99, 1.0, 1.2, 1e300, out);
    CU_ASSERT_EQUAL(ret, TSD_ERR_QUANTILE_MAX_ITERATIONS);
    ret = tsd_approximate_gamma_iqr(1e-250, 0.5, 1.0, 1e100, 1e300, out);
    CU_ASSERT_EQUAL(ret, TSD_ERR_INVALID_HYPERGEO);
    /* the lower quantile is zero, or underflows when squared */
    ret = tsd_approximate_gamma_iqr(1e-10, 0.01, 1.0, 1e300, 50.0, out);
    CU_ASSERT_EQUAL(ret, TSD_ERR_ZERO_DIVISION);
    ret = tsd_approximate_gamma_iqr(1e-200, 0.5, 1.0, 5e199, 50.0, out);
    CU_ASSERT_EQUAL(ret, TSD_ERR_ZERO_DIVISION);
    /* the gradient is zero */
    ret = tsd_approximate_gamma_iqr(0.1, 0.9, 1.0, 1.0000001, 1e20, out);
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

    ret = tsd_block_singletons(5, 2, 4, 3, unphased, phase_nodes_individual,
        mutations_node, mutations_position, phase_edges_child, phase_edges_left,
        phase_edges_right, phase_order, phase_order, 10.0, &num_blocks, &blocks_stats,
        &blocks_edges, mutations_block);
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
    ret = tsd_block_singletons(5, 2, 4, 3, unphased, phase_nodes_individual,
        mutations_node, mutations_position, child_missing, phase_edges_left,
        phase_edges_right, phase_order, phase_order, 10.0, &num_blocks, &blocks_stats,
        &blocks_edges, mutations_block);
    CU_ASSERT_EQUAL(ret, TSD_ERR_UNPHASED_INDIVIDUAL_EDGES);
    CU_ASSERT_PTR_NULL(blocks_stats);

    /* No edges and no individuals */
    ret = tsd_block_singletons(0, 0, 0, 0, unphased, phase_nodes_individual,
        mutations_node, mutations_position, phase_edges_child, phase_edges_left,
        phase_edges_right, phase_order, phase_order, 10.0, &num_blocks, &blocks_stats,
        &blocks_edges, mutations_block);
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
    CU_ASSERT_EQUAL(ret, TSD_ERR_BLOCK_OUT_OF_BOUNDS);
    memcpy(likelihood, initial, sizeof(initial));
    ret = tsd_reallocate_unphased(
        2, 2, 1, phase, mutations_block, blocks_edges_bad, likelihood);
    CU_ASSERT_EQUAL(ret, TSD_ERR_BLOCK_OUT_OF_BOUNDS);
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
test_check_arguments(void)
{
    const tsd_id_t ids[] = { 0, TSD_NULL, 4 };
    const double left_unsorted[] = { 5, 0, 0, 0 };
    const double right_unsorted[] = { 10, 5, 10, 10 };
    const tsd_id_t order_duplicate[] = { 0, 1, 1, 3 };
    const tsd_id_t order_bad[] = { 0, 1, 2, 4 };
    const double left_empty[] = { 0, 0, 0, 10 };
    const double right_long[] = { 10, 10, 10, 11 };
    const double left_nan[] = { 0, 0, 0, NAN };

    CU_ASSERT_EQUAL(tsd_check_ids(3, ids, 5, 1, TSD_ERR_NODE_OUT_OF_BOUNDS), 0);
    CU_ASSERT_EQUAL(tsd_check_ids(3, ids, 5, 0, TSD_ERR_NODE_OUT_OF_BOUNDS),
        TSD_ERR_NODE_OUT_OF_BOUNDS);
    CU_ASSERT_EQUAL(tsd_check_ids(3, ids, 4, 1, TSD_ERR_EDGE_OUT_OF_BOUNDS),
        TSD_ERR_EDGE_OUT_OF_BOUNDS);
    CU_ASSERT_EQUAL(tsd_check_edges(5, 4, phase_edges_parent, phase_edges_child), 0);
    CU_ASSERT_EQUAL(tsd_check_edges(4, 4, phase_edges_parent, phase_edges_child),
        TSD_ERR_NODE_OUT_OF_BOUNDS);
    CU_ASSERT_EQUAL(tsd_check_edges(4, 4, phase_edges_child, phase_edges_parent),
        TSD_ERR_NODE_OUT_OF_BOUNDS);
    CU_ASSERT_EQUAL(tsd_check_edge_sweep(4, phase_edges_left, phase_edges_right,
                        phase_order, phase_order, 10.0),
        0);
    CU_ASSERT_EQUAL(tsd_check_edge_sweep(0, NULL, NULL, NULL, NULL, 10.0), 0);
    CU_ASSERT_EQUAL(tsd_check_edge_sweep(4, left_empty, phase_edges_right, phase_order,
                        phase_order, 10.0),
        TSD_ERR_BAD_EDGE_INTERVAL);
    CU_ASSERT_EQUAL(tsd_check_edge_sweep(
                        4, phase_edges_left, right_long, phase_order, phase_order, 10.0),
        TSD_ERR_BAD_EDGE_INTERVAL);
    CU_ASSERT_EQUAL(tsd_check_edge_sweep(
                        4, left_nan, phase_edges_right, phase_order, phase_order, 10.0),
        TSD_ERR_BAD_EDGE_INTERVAL);
    CU_ASSERT_EQUAL(tsd_check_edge_sweep(4, phase_edges_left, phase_edges_right,
                        order_bad, phase_order, 10.0),
        TSD_ERR_EDGE_OUT_OF_BOUNDS);
    CU_ASSERT_EQUAL(tsd_check_edge_sweep(4, phase_edges_left, phase_edges_right,
                        phase_order, order_bad, 10.0),
        TSD_ERR_EDGE_OUT_OF_BOUNDS);
    CU_ASSERT_EQUAL(tsd_check_edge_sweep(4, phase_edges_left, phase_edges_right,
                        order_duplicate, phase_order, 10.0),
        TSD_ERR_BAD_EDGE_INDEXES);
    CU_ASSERT_EQUAL(tsd_check_edge_sweep(4, phase_edges_left, phase_edges_right,
                        phase_order, order_duplicate, 10.0),
        TSD_ERR_BAD_EDGE_INDEXES);
    CU_ASSERT_EQUAL(tsd_check_edge_sweep(4, left_unsorted, phase_edges_right,
                        phase_order, phase_order, 10.0),
        TSD_ERR_BAD_EDGE_INDEXES);
    CU_ASSERT_EQUAL(tsd_check_edge_sweep(4, phase_edges_left, right_unsorted,
                        phase_order, phase_order, 10.0),
        TSD_ERR_BAD_EDGE_INDEXES);
}

/*
 * Each function given an id out of bounds, edges that cannot be swept, or edges that
 * form a cycle. The fixture has nodes 0-4 and edges from node 4 to nodes 0-3.
 */
static void
test_bad_tree_arguments(void)
{
    const uint8_t flags[] = { 1, 1, 1, 1, 0 };
    const double times[] = { 0, 0, 0, 0, 1 };
    const double constraints[] = { 0, 1, 0, 1, 0, 1, 0, 1, 0, 1 };
    const double likelihoods[] = { 1, 1, 1, 1, 1, 1, 1, 1 };
    const tsd_id_t identity[] = { 0, 1, 2, 3, 4 };
    const tsd_id_t bad_nodes[] = { 0, 1, 2, 5 };
    const tsd_id_t bad_individuals[] = { 0, 0, 1, 2, TSD_NULL };
    const double right_long[] = { 10, 10, 10, 11 };
    const tsd_id_t cycle_parent[] = { 1, 0 };
    const tsd_id_t cycle_child[] = { 0, 1 };
    const double cycle_left[] = { 0, 0 };
    const double cycle_right[] = { 10, 10 };
    const tsd_id_t mutation_node[] = { 0 };
    const tsd_id_t bad_mutation_node[] = { 5 };
    const double mutation_position[] = { 1 };
    tsd_id_t new_parent[4], new_child[4], output[1], mutations_edge[1], index[5];
    tsd_id_t *nodes_order = NULL;
    tsd_id_t *blocks_edges = NULL;
    double *blocks_stats = NULL;
    double *origin = NULL;
    double *adjust = NULL;
    double stats[8], out[5];
    int32_t freq[5];
    tsd_size_t num_split, num_blocks, num_breaks;
    tsd_epoch_stats_t epochs;
    int result;

    /* tsd_split_disjoint_nodes */
    CU_ASSERT_EQUAL(
        tsd_split_disjoint_nodes(5, 4, phase_edges_parent, bad_nodes, phase_edges_left,
            phase_edges_right, flags, new_parent, new_child, &num_split, &nodes_order),
        TSD_ERR_NODE_OUT_OF_BOUNDS);
    CU_ASSERT_PTR_NULL(nodes_order);

    /* tsd_relabel_mutations_node */
    CU_ASSERT_EQUAL(tsd_relabel_mutations_node(5, 4, 1, mutation_node, mutation_position,
                        identity, phase_edges_parent, bad_nodes, phase_edges_left,
                        phase_edges_right, phase_order, phase_order, output),
        TSD_ERR_NODE_OUT_OF_BOUNDS);
    CU_ASSERT_EQUAL(
        tsd_relabel_mutations_node(5, 4, 1, mutation_node, mutation_position, bad_nodes,
            phase_edges_parent, phase_edges_child, phase_edges_left, phase_edges_right,
            phase_order, phase_order, output),
        TSD_ERR_NODE_OUT_OF_BOUNDS);
    CU_ASSERT_EQUAL(
        tsd_relabel_mutations_node(5, 4, 1, bad_mutation_node, mutation_position,
            identity, phase_edges_parent, phase_edges_child, phase_edges_left,
            phase_edges_right, phase_order, phase_order, output),
        TSD_ERR_NODE_OUT_OF_BOUNDS);
    CU_ASSERT_EQUAL(
        tsd_relabel_mutations_node(5, 4, 1, mutation_node, mutation_position, identity,
            phase_edges_parent, phase_edges_child, phase_edges_left, phase_edges_right,
            bad_nodes, phase_order, output),
        TSD_ERR_EDGE_OUT_OF_BOUNDS);

    /* tsd_constrain_ages */
    CU_ASSERT_EQUAL(tsd_constrain_ages(
                        5, 4, times, flags, bad_nodes, phase_edges_child, 1e-6, 1, out),
        TSD_ERR_NODE_OUT_OF_BOUNDS);

    /* tsd_contains_unary_nodes */
    CU_ASSERT_EQUAL(tsd_contains_unary_nodes(5, 4, flags, bad_nodes, phase_edges_left,
                        phase_edges_right, phase_order, phase_order, 10.0, &result),
        TSD_ERR_NODE_OUT_OF_BOUNDS);
    CU_ASSERT_EQUAL(
        tsd_contains_unary_nodes(5, 4, flags, phase_edges_parent, phase_edges_left,
            right_long, phase_order, phase_order, 10.0, &result),
        TSD_ERR_BAD_EDGE_INTERVAL);

    /* tsd_constraints_are_consistent */
    CU_ASSERT_EQUAL(tsd_constraints_are_consistent(
                        5, 4, constraints, phase_edges_parent, bad_nodes, &result),
        TSD_ERR_NODE_OUT_OF_BOUNDS);

    /* tsd_count_mutations */
    CU_ASSERT_EQUAL(
        tsd_count_mutations(5, flags, 1, mutation_node, mutation_position, 4,
            phase_edges_parent, bad_nodes, phase_edges_left, phase_edges_right,
            phase_order, phase_order, 10.0, 0, stats, mutations_edge),
        TSD_ERR_NODE_OUT_OF_BOUNDS);
    CU_ASSERT_EQUAL(
        tsd_count_mutations(5, flags, 1, bad_mutation_node, mutation_position, 4,
            phase_edges_parent, phase_edges_child, phase_edges_left, phase_edges_right,
            phase_order, phase_order, 10.0, 0, stats, mutations_edge),
        TSD_ERR_NODE_OUT_OF_BOUNDS);
    CU_ASSERT_EQUAL(
        tsd_count_mutations(5, flags, 1, mutation_node, mutation_position, 4,
            phase_edges_parent, phase_edges_child, phase_edges_left, right_long,
            phase_order, phase_order, 10.0, 0, stats, mutations_edge),
        TSD_ERR_BAD_EDGE_INTERVAL);
    CU_ASSERT_EQUAL(tsd_count_mutations(2, flags, 1, mutation_node, mutation_position, 2,
                        cycle_parent, cycle_child, cycle_left, cycle_right, phase_order,
                        phase_order, 10.0, 1, stats, mutations_edge),
        TSD_ERR_EDGES_FORM_CYCLE);

    /* tsd_mutational_area and tsd_mutational_timescale */
    CU_ASSERT_EQUAL(tsd_mutational_area(5, times, 4, likelihoods, phase_edges_parent,
                        bad_nodes, &epochs, index),
        TSD_ERR_NODE_OUT_OF_BOUNDS);
    CU_ASSERT_EQUAL(tsd_mutational_timescale(5, times, 4, likelihoods,
                        phase_edges_parent, bad_nodes, 1, &num_breaks, &origin, &adjust),
        TSD_ERR_NODE_OUT_OF_BOUNDS);
    CU_ASSERT_PTR_NULL(origin);

    /* tsd_block_singletons */
    CU_ASSERT_EQUAL(tsd_block_singletons(5, 2, 4, 1, flags, bad_individuals,
                        mutation_node, mutation_position, phase_edges_child,
                        phase_edges_left, phase_edges_right, phase_order, phase_order,
                        10.0, &num_blocks, &blocks_stats, &blocks_edges, output),
        TSD_ERR_INDIVIDUAL_OUT_OF_BOUNDS);
    CU_ASSERT_EQUAL(tsd_block_singletons(5, 2, 4, 1, flags, phase_nodes_individual,
                        mutation_node, mutation_position, bad_nodes, phase_edges_left,
                        phase_edges_right, phase_order, phase_order, 10.0, &num_blocks,
                        &blocks_stats, &blocks_edges, output),
        TSD_ERR_NODE_OUT_OF_BOUNDS);
    CU_ASSERT_EQUAL(tsd_block_singletons(5, 2, 4, 1, flags, phase_nodes_individual,
                        bad_mutation_node, mutation_position, phase_edges_child,
                        phase_edges_left, phase_edges_right, phase_order, phase_order,
                        10.0, &num_blocks, &blocks_stats, &blocks_edges, output),
        TSD_ERR_NODE_OUT_OF_BOUNDS);
    CU_ASSERT_EQUAL(tsd_block_singletons(5, 2, 4, 1, flags, phase_nodes_individual,
                        mutation_node, mutation_position, phase_edges_child,
                        phase_edges_left, right_long, phase_order, phase_order, 10.0,
                        &num_blocks, &blocks_stats, &blocks_edges, output),
        TSD_ERR_BAD_EDGE_INTERVAL);
    CU_ASSERT_PTR_NULL(blocks_stats);
    CU_ASSERT_PTR_NULL(blocks_edges);

    /* tsd_mutation_frequency */
    CU_ASSERT_EQUAL(
        tsd_mutation_frequency(5, 1, 4, 1, flags, mutation_node, mutation_position,
            phase_edges_parent, bad_nodes, phase_edges_left, phase_edges_right,
            phase_order, phase_order, 10.0, freq),
        TSD_ERR_NODE_OUT_OF_BOUNDS);
    CU_ASSERT_EQUAL(
        tsd_mutation_frequency(5, 1, 4, 1, flags, bad_mutation_node, mutation_position,
            phase_edges_parent, phase_edges_child, phase_edges_left, phase_edges_right,
            phase_order, phase_order, 10.0, freq),
        TSD_ERR_NODE_OUT_OF_BOUNDS);
    CU_ASSERT_EQUAL(
        tsd_mutation_frequency(5, 1, 4, 1, flags, mutation_node, mutation_position,
            phase_edges_parent, phase_edges_child, phase_edges_left, right_long,
            phase_order, phase_order, 10.0, freq),
        TSD_ERR_BAD_EDGE_INTERVAL);
    CU_ASSERT_EQUAL(tsd_mutation_frequency(2, 1, 2, 1, flags, mutation_node,
                        mutation_position, cycle_parent, cycle_child, cycle_left,
                        cycle_right, phase_order, phase_order, 10.0, freq),
        TSD_ERR_EDGES_FORM_CYCLE);
}

/* ---- priors ---- */

#define TABLE_MAX_TIPS          20
#define TABLE_MAX_MARGINAL_TIPS 10
#define TABLE_MAX_MOMENTS       3

typedef struct {
    tsd_size_t num_tips;
    tsd_size_t num_moments;
    double val[TABLE_MAX_MARGINAL_TIPS * TABLE_MAX_MOMENTS];
    double out[(TABLE_MAX_MARGINAL_TIPS + 1) * TABLE_MAX_MOMENTS];
    int ret;
} marginalize_row_t;

typedef struct {
    tsd_size_t num_tips;
    double variance[TABLE_MAX_TIPS + 1];
    int ret;
} variance_row_t;

/* Expected values of the reference implementation */
static const marginalize_row_t marginalize_rows[] = {
    { .num_tips = 2,
        .num_moments = 2,
        .val = { 1, 2, 3, 4 },
        .out = { 0, 0, 0, 0, 3, 4 },
        .ret = 0 },
    { .num_tips = 3,
        .num_moments = 2,
        .val = { 6.405920704482398, 2.770888466262316, 0.5056378869683275,
            0.26362359173243805, 8.151375368082697, 9.136280215049444 },
        .out = { 0, 0, 0, 0, 8.151375368082697, 9.136280215049444, 0.5056378869683275,
            0.26362359173243805 },
        .ret = 0 },
    { .num_tips = 6,
        .num_moments = 1,
        .val = { 6.105694180095081, 7.322015953741584, 5.481887415507686,
            9.357216995498906, 8.176950185803168, 0.12711115168446616 },
        .out = { 0, 0, 4.1943175516834055, 8.11004435562237, 7.419552205503297,
            5.481887415507686, 7.322015953741584 },
        .ret = 0 },
    { .num_tips = 5,
        .num_moments = 3,
        .val = { 8.588302338216936, 0.43249719552409716, 7.323588919656446,
            1.8389906439653343, 8.645471331263877, 5.4604660804660075,
            3.0671477163201097, 4.284603489856819, 0.3803647443400834,
            1.3304044373456831, 6.73918170546694, 6.507176164585076, 6.192312603664413,
            3.8984077871926464, 9.972378364313188 },
        .out = { 0, 0, 0, 0, 0, 0, 4.421223665034365, 4.789259532941353,
            7.973616342397445, 2.025101748935454, 5.757350419222893, 4.056451596487079,
            3.0671477163201097, 4.284603489856819, 0.3803647443400834,
            1.8389906439653343, 8.645471331263877, 5.4604660804660075 },
        .ret = 0 },
    { .num_tips = 10,
        .num_moments = 2,
        .val = { 9.810269853884678, 6.886865646358878, 6.5395468350513815,
            6.9156226326523065, 3.9503220973931277, 1.437455399721871, 7.242734567921409,
            5.301007792509686, 3.1713945680336613, 4.909770052434712, 8.905929560055103,
            9.347030807966872, 3.6421724474197954, 5.758145324224633, 3.2865069716518276,
            5.983570298976998, 3.4453211325206197, 3.9770281052287966, 8.913716084847444,
            2.3488601759804593 },
        .out = { 0, 0, 0, 0, 5.623151605758174, 4.568435971625975, 4.799050195244742,
            5.94494257642139, 5.422355848514381, 5.987743974404029, 5.5903751937122115,
            5.348623913661779, 5.40321218005653, 4.413497494215927, 5.163430081689842,
            3.4458516438653746, 4.938045838551611, 2.5965211175582152,
            3.9503220973931277, 1.437455399721871, 6.5395468350513815,
            6.9156226326523065 },
        .ret = 0 },
    { .num_tips = 8,
        .num_moments = 2,
        .val = { 0, 0, 1.75, 4.21937641723356, 0.75, 0.71937641723356,
            0.41666666666666663, 0.21937641723356005, 0.25, 0.08048752834467121, 0.15,
            0.0304875283446712, 0.08333333333333333, 0.0104875283446712,
            0.03571428571428571, 0.002551020408163265 },
        .out = { 0, 0, 0, 0, 0.125, 0.04243197278911563, 0.25000000000000006,
            0.12064625850340137, 0.3750000000000001, 0.23028911564625854, 0.5,
            0.3681859410430839, 0.625, 0.53187641723356, 0.75, 0.71937641723356, 1.75,
            4.21937641723356 },
        .ret = 0 },
    { .num_tips = 6,
        .num_moments = 2,
        .val = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
        .out = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
        .ret = 0 },
    { .num_tips = 6,
        .num_moments = 2,
        .val = { NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN },
        .out = { 0, 0, 0, 0, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN },
        .ret = 0 },
    /* NOTE: non-finite values are propagated by the weighted sum, with no checks */
    { .num_tips = 4,
        .num_moments = 2,
        .val = { 1, 2, INFINITY, 1, 3, 3, 1, NAN },
        .out = { 0, 0, 0, 0, 1.5, NAN, 3, 3, INFINITY, 1 },
        .ret = 0 },
    { .num_tips = 1, .num_moments = 2, .val = { 1, 2 }, .ret = TSD_ERR_TOO_FEW_TIPS },
};

/* Expected values of the reference implementation */
static const variance_row_t variance_rows[] = {
    /* a single tip has zero variance */
    { .num_tips = 1, .variance = { 0, 0 }, .ret = 0 },
    { .num_tips = 2, .variance = { 0, 0, 1 }, .ret = 0 },
    { .num_tips = 3,
        .variance = { 0, 0, 0.1111111111111111, 1.1111111111111112 },
        .ret = 0 },
    { .num_tips = 4,
        .variance = { 0, 0, 0.0763888888888889, 0.1388888888888889, 1.1388888888888888 },
        .ret = 0 },
    { .num_tips = 5,
        .variance = { 0, 0, 0.05555555555555555, 0.10888888888888887,
            0.14888888888888885, 1.148888888888889 },
        .ret = 0 },
    { .num_tips = 7,
        .variance
        = { 0, 0, 0.033197278911564626, 0.07034013605442176, 0.10571428571428576,
            0.13519274376417229, 0.15560090702947849, 1.1556009070294788 },
        .ret = 0 },
    { .num_tips = 10,
        .variance
        = { 0, 0, 0.018567019400352734, 0.041611237087427574, 0.06639581758629377,
            0.09089317208364828, 0.11351221970269593, 0.13295666414714036,
            0.1481418493323255, 0.15814184933232545, 1.158141849332326 },
        .ret = 0 },
    { .num_tips = 15,
        .variance = { 0, 0, 0.009275031033272787, 0.021737639649727542,
            0.03624521961185298, 0.051927046202770555, 0.06808446902686013,
            0.0841377367467945, 0.09959413220318991, 0.11402731806494726,
            0.1270631711007999, 0.13836964240727112, 0.1476492516868806,
            0.15463337867100768, 0.15907782311545215, 1.1590778231154522 },
        .ret = 0 },
    { .num_tips = 20,
        .variance = { 0, 0, 0.005572819114451828, 0.013375800851143278,
            0.022807509758699017, 0.033400737695193194, 0.04477484272918324,
            0.05660993547639791, 0.068631190860954, 0.08059855293878618,
            0.09229959605553426, 0.10354436416820739, 0.11416151633721722,
            0.12399537093168672, 0.1329035891897501, 0.14075532637585275,
            0.1474297329805997, 0.1528147232340108, 0.15680595130418618,
            0.15930595130418657, 1.1593059513041868 },
        .ret = 0 },
    { .num_tips = 0, .ret = TSD_ERR_TOO_FEW_TIPS },
};

static void
test_marginalize_over_ancestors(void)
{
    const size_t num_rows = sizeof(marginalize_rows) / sizeof(marginalize_rows[0]);
    const marginalize_row_t *row;
    double out[(TABLE_MAX_MARGINAL_TIPS + 1) * TABLE_MAX_MOMENTS];
    size_t j;
    int ret;

    for (j = 0; j < num_rows; j++) {
        row = marginalize_rows + j;
        ret = tsd_marginalize_over_ancestors(
            row->num_tips, row->num_moments, row->val, out);
        CU_ASSERT_EQUAL(ret, row->ret);
        if (ret == 0 && row->ret == 0) {
            assert_array_close(out, row->out, (row->num_tips + 1) * row->num_moments);
        }
    }
}

static void
test_conditional_coalescent_variance(void)
{
    const size_t num_rows = sizeof(variance_rows) / sizeof(variance_rows[0]);
    const variance_row_t *row;
    double variance[TABLE_MAX_TIPS + 1];
    size_t j;
    int ret;

    for (j = 0; j < num_rows; j++) {
        row = variance_rows + j;
        ret = tsd_conditional_coalescent_variance(row->num_tips, variance);
        CU_ASSERT_EQUAL(ret, row->ret);
        if (ret == 0 && row->ret == 0) {
            assert_array_close(variance, row->variance, row->num_tips + 1);
        }
    }
}

/* ---- rescaling ---- */

#define TABLE_MAX_COUNTS     8
#define TABLE_MAX_BREAKS     12
#define TABLE_MAX_NODES      6
#define TABLE_MAX_EDGES      4
#define TABLE_MAX_MUTATIONS  5
#define TABLE_MAX_POINTS     5
#define TABLE_MAX_POSTERIORS 4

typedef struct {
    tsd_size_t num_counts;
    double counts[TABLE_MAX_COUNTS];
    int64_t epochs;
    tsd_size_t num_breaks;
    tsd_id_t breaks[TABLE_MAX_BREAKS];
    int ret;
} changepoints_row_t;

typedef struct {
    tsd_size_t num_nodes;
    uint8_t is_sample[TABLE_MAX_NODES];
    tsd_size_t num_mutations;
    tsd_id_t mutations_node[TABLE_MAX_MUTATIONS];
    double mutations_position[TABLE_MAX_MUTATIONS];
    tsd_size_t num_edges;
    tsd_id_t edges_parent[TABLE_MAX_EDGES];
    tsd_id_t edges_child[TABLE_MAX_EDGES];
    double edges_left[TABLE_MAX_EDGES];
    double edges_right[TABLE_MAX_EDGES];
    tsd_id_t indexes_insert[TABLE_MAX_EDGES];
    tsd_id_t indexes_remove[TABLE_MAX_EDGES];
    double sequence_length;
    int size_biased;
    double edges_stats[TABLE_MAX_EDGES][2];
    tsd_id_t mutations_edge[TABLE_MAX_MUTATIONS];
} count_mutations_row_t;

typedef struct {
    tsd_size_t num_nodes;
    double nodes_time[TABLE_MAX_NODES];
    tsd_size_t num_edges;
    double likelihoods[TABLE_MAX_EDGES][2];
    tsd_id_t edges_parent[TABLE_MAX_EDGES];
    tsd_id_t edges_child[TABLE_MAX_EDGES];
    tsd_size_t num_epochs;
    double counts[TABLE_MAX_NODES];
    double offset[TABLE_MAX_NODES];
    double duration[TABLE_MAX_NODES];
    tsd_id_t nodes_index[TABLE_MAX_NODES];
} area_row_t;

typedef struct {
    tsd_size_t num_nodes;
    double nodes_time[TABLE_MAX_NODES];
    tsd_size_t num_edges;
    double likelihoods[TABLE_MAX_EDGES][2];
    tsd_id_t edges_parent[TABLE_MAX_EDGES];
    tsd_id_t edges_child[TABLE_MAX_EDGES];
    int64_t max_intervals;
    tsd_size_t num_breaks;
    double origin[TABLE_MAX_NODES];
    double adjust[TABLE_MAX_NODES];
    int ret;
} timescale_row_t;

typedef struct {
    tsd_size_t num_points;
    double estimate[TABLE_MAX_POINTS];
    uint8_t fixed[TABLE_MAX_POINTS];
    tsd_size_t num_breaks;
    double original[TABLE_MAX_BREAKS];
    double rescaled[TABLE_MAX_BREAKS];
    double out[TABLE_MAX_POINTS];
    int ret;
} point_row_t;

typedef struct {
    tsd_size_t num_posteriors;
    double posteriors[TABLE_MAX_POSTERIORS][2];
    uint8_t fixed[TABLE_MAX_POSTERIORS];
    tsd_size_t num_breaks;
    double original[TABLE_MAX_BREAKS];
    double rescaled[TABLE_MAX_BREAKS];
    double quantile_width;
    double max_shape;
    double out[TABLE_MAX_POSTERIORS][2];
    int ret;
} posterior_row_t;

/* Expected values of the reference implementation */
static const changepoints_row_t changepoints_rows[] = {
    { .num_counts = 3,
        .counts = { 1, 2, 3 },
        .epochs = 1,
        .num_breaks = 2,
        .breaks = { 0, 3 },
        .ret = 0 },
    { .num_counts = 1,
        .counts = { 5 },
        .epochs = 1,
        .num_breaks = 2,
        .breaks = { 0, 1 },
        .ret = 0 },
    { .num_counts = 4,
        .counts = { 0, 0, 4, 0 },
        .epochs = 1,
        .num_breaks = 2,
        .breaks = { 0, 4 },
        .ret = 0 },
    { .num_counts = 3,
        .counts = { 1, 2, 3 },
        .epochs = 10,
        .num_breaks = 11,
        .breaks = { 0, 0, 1, 1, 1, 2, 2, 2, 2, 2, 3 },
        .ret = 0 },
    { .num_counts = 4,
        .counts = { 0, 0, 4, 0 },
        .epochs = 4,
        .num_breaks = 5,
        .breaks = { 0, 2, 2, 2, 4 },
        .ret = 0 },
    { .num_counts = 4,
        .counts = { 0, 0, 5, 5 },
        .epochs = 3,
        .num_breaks = 4,
        .breaks = { 0, 2, 3, 4 },
        .ret = 0 },
    { .num_counts = 4,
        .counts = { 5, 5, 0, 0 },
        .epochs = 3,
        .num_breaks = 4,
        .breaks = { 0, 0, 1, 4 },
        .ret = 0 },
    { .num_counts = 5,
        .counts = { 0, 5, 0, 5, 0 },
        .epochs = 3,
        .num_breaks = 4,
        .breaks = { 0, 1, 3, 5 },
        .ret = 0 },
    { .num_counts = 4,
        .counts = { 1, 1, 1, 1 },
        .epochs = 4,
        .num_breaks = 5,
        .breaks = { 0, 1, 2, 3, 4 },
        .ret = 0 },
    { .num_counts = 4,
        .counts = { 1, 1, 1, 1 },
        .epochs = 2,
        .num_breaks = 3,
        .breaks = { 0, 2, 4 },
        .ret = 0 },
    { .num_counts = 8,
        .counts = { 1.0730290263725388, 0.30845314412528435, 5.375436872608127,
            0.36642711289983193, 0.11536203910680366, 1.7997970856557877,
            0.498639097854449, 0.5514213768237073 },
        .epochs = 3,
        .num_breaks = 4,
        .breaks = { 0, 2, 2, 8 },
        .ret = 0 },
    { .num_counts = 8,
        .counts = { 2.073029026372539, 0, 5.375436872608127, 0.36642711289983193, 0,
            1.7997970856557877, 0.498639097854449, 0.5514213768237073 },
        .epochs = 5,
        .num_breaks = 6,
        .breaks = { 0, 2, 2, 2, 5, 8 },
        .ret = 0 },
    /* NOTE: current behaviour, possibly a bug: with no counts the cumulative
       distribution is 0 / 0 = NaN, so every breakpoint but the last is -1 */
    { .num_counts = 3,
        .counts = { 0, 0, 0 },
        .epochs = 3,
        .num_breaks = 4,
        .breaks = { -1, -1, -1, 3 },
        .ret = 0 },
    { .num_counts = 0,
        .epochs = 3,
        .num_breaks = 4,
        .breaks = { -1, -1, -1, 0 },
        .ret = 0 },
    { .num_counts = 2,
        .counts = { 1, 2 },
        .epochs = 0,
        .ret = TSD_ERR_NONPOSITIVE_INTERVALS },
    { .num_counts = 2,
        .counts = { 1, 2 },
        .epochs = -1,
        .ret = TSD_ERR_NONPOSITIVE_INTERVALS },
};

static void
test_fixed_changepoints(void)
{
    const size_t num_rows = sizeof(changepoints_rows) / sizeof(changepoints_rows[0]);
    const changepoints_row_t *row;
    tsd_id_t *breaks = NULL;
    tsd_size_t num_breaks, k;
    size_t j;
    int ret;

    for (j = 0; j < num_rows; j++) {
        row = changepoints_rows + j;
        ret = tsd_fixed_changepoints(
            row->num_counts, row->counts, row->epochs, &breaks, &num_breaks);
        CU_ASSERT_EQUAL(ret, row->ret);
        if (ret == 0 && row->ret == 0) {
            CU_ASSERT_EQUAL_FATAL(num_breaks, row->num_breaks);
            for (k = 0; k < num_breaks; k++) {
                CU_ASSERT_EQUAL(breaks[k], row->breaks[k]);
            }
        }
        tsd_safe_free(breaks);
        breaks = NULL;
    }
}

/* Expected values of the reference implementation */
static const count_mutations_row_t count_mutations_rows[] = {
    /* a mutation above the root has no edge */
    { .num_nodes = 3,
        .is_sample = { 1, 1, 0 },
        .num_mutations = 4,
        .mutations_node = { 0, 0, 1, 2 },
        .mutations_position = { 1, 2, 3, 5 },
        .num_edges = 2,
        .edges_parent = { 2, 2 },
        .edges_child = { 0, 1 },
        .edges_left = { 0, 0 },
        .edges_right = { 10, 10 },
        .indexes_insert = { 0, 1 },
        .indexes_remove = { 1, 0 },
        .sequence_length = 10,
        .size_biased = 0,
        .edges_stats = { { 2, 10 }, { 1, 10 } },
        .mutations_edge = { 0, 0, 1, -1 } },
    { .num_nodes = 3,
        .is_sample = { 1, 1, 0 },
        .num_mutations = 4,
        .mutations_node = { 0, 0, 1, 2 },
        .mutations_position = { 1, 2, 3, 5 },
        .num_edges = 2,
        .edges_parent = { 2, 2 },
        .edges_child = { 0, 1 },
        .edges_left = { 0, 0 },
        .edges_right = { 10, 10 },
        .indexes_insert = { 0, 1 },
        .indexes_remove = { 1, 0 },
        .sequence_length = 10,
        .size_biased = 1,
        .edges_stats = { { 2, 10 }, { 1, 10 } },
        .mutations_edge = { 0, 0, 1, -1 } },
    /* no mutations */
    { .num_nodes = 3,
        .is_sample = { 1, 1, 0 },
        .num_mutations = 0,
        .num_edges = 2,
        .edges_parent = { 2, 2 },
        .edges_child = { 0, 1 },
        .edges_left = { 0, 0 },
        .edges_right = { 10, 10 },
        .indexes_insert = { 0, 1 },
        .indexes_remove = { 1, 0 },
        .sequence_length = 10,
        .size_biased = 0,
        .edges_stats = { { 0, 10 }, { 0, 10 } } },
    /* no edges */
    { .num_nodes = 3,
        .is_sample = { 1, 1, 0 },
        .num_mutations = 1,
        .mutations_node = { 0 },
        .mutations_position = { 1 },
        .num_edges = 0,
        .sequence_length = 10,
        .size_biased = 1,
        .mutations_edge = { -1 } },
    /* no edges and no mutations */
    { .num_nodes = 3,
        .is_sample = { 1, 1, 0 },
        .num_mutations = 0,
        .num_edges = 0,
        .sequence_length = 10,
        .size_biased = 0 },
    /* a mutation beyond the edge */
    { .num_nodes = 2,
        .is_sample = { 1, 0 },
        .num_mutations = 2,
        .mutations_node = { 0, 0 },
        .mutations_position = { 1, 5 },
        .num_edges = 1,
        .edges_parent = { 1 },
        .edges_child = { 0 },
        .edges_left = { 0 },
        .edges_right = { 4 },
        .indexes_insert = { 0 },
        .indexes_remove = { 0 },
        .sequence_length = 10,
        .size_biased = 0,
        .edges_stats = { { 1, 4 } },
        .mutations_edge = { 0, -1 } },
    { .num_nodes = 2,
        .is_sample = { 1, 0 },
        .num_mutations = 2,
        .mutations_node = { 0, 0 },
        .mutations_position = { 1, 5 },
        .num_edges = 1,
        .edges_parent = { 1 },
        .edges_child = { 0 },
        .edges_left = { 0 },
        .edges_right = { 4 },
        .indexes_insert = { 0 },
        .indexes_remove = { 0 },
        .sequence_length = 10,
        .size_biased = 1,
        .edges_stats = { { 1, 4 } },
        .mutations_edge = { 0, -1 } },
    /* a mutation in a gap between the edges of a node */
    { .num_nodes = 3,
        .is_sample = { 1, 1, 0 },
        .num_mutations = 2,
        .mutations_node = { 0, 0 },
        .mutations_position = { 4, 5 },
        .num_edges = 3,
        .edges_parent = { 2, 2, 2 },
        .edges_child = { 0, 0, 1 },
        .edges_left = { 0, 6, 0 },
        .edges_right = { 4, 10, 10 },
        .indexes_insert = { 0, 2, 1 },
        .indexes_remove = { 0, 2, 1 },
        .sequence_length = 10,
        .size_biased = 0,
        .edges_stats = { { 0, 4 }, { 0, 4 }, { 0, 10 } },
        .mutations_edge = { -1, -1 } },
    { .num_nodes = 3,
        .is_sample = { 1, 1, 0 },
        .num_mutations = 2,
        .mutations_node = { 0, 0 },
        .mutations_position = { 4, 5 },
        .num_edges = 3,
        .edges_parent = { 2, 2, 2 },
        .edges_child = { 0, 0, 1 },
        .edges_left = { 0, 6, 0 },
        .edges_right = { 4, 10, 10 },
        .indexes_insert = { 0, 2, 1 },
        .indexes_remove = { 0, 2, 1 },
        .sequence_length = 10,
        .size_biased = 1,
        .edges_stats = { { 0, 4 }, { 0, 4 }, { 0, 10 } },
        .mutations_edge = { -1, -1 } },
    /* a mutation at a breakpoint belongs to the tree on its right */
    { .num_nodes = 3,
        .is_sample = { 1, 1, 0 },
        .num_mutations = 2,
        .mutations_node = { 0, 0 },
        .mutations_position = { 4.5, 5 },
        .num_edges = 3,
        .edges_parent = { 2, 2, 2 },
        .edges_child = { 0, 0, 1 },
        .edges_left = { 0, 5, 0 },
        .edges_right = { 5, 10, 10 },
        .indexes_insert = { 0, 2, 1 },
        .indexes_remove = { 0, 2, 1 },
        .sequence_length = 10,
        .size_biased = 0,
        .edges_stats = { { 1, 5 }, { 1, 5 }, { 0, 10 } },
        .mutations_edge = { 0, 1 } },
    { .num_nodes = 3,
        .is_sample = { 1, 1, 0 },
        .num_mutations = 2,
        .mutations_node = { 0, 0 },
        .mutations_position = { 4.5, 5 },
        .num_edges = 3,
        .edges_parent = { 2, 2, 2 },
        .edges_child = { 0, 0, 1 },
        .edges_left = { 0, 5, 0 },
        .edges_right = { 5, 10, 10 },
        .indexes_insert = { 0, 2, 1 },
        .indexes_remove = { 0, 2, 1 },
        .sequence_length = 10,
        .size_biased = 1,
        .edges_stats = { { 1, 5 }, { 1, 5 }, { 0, 10 } },
        .mutations_edge = { 0, 1 } },
    /* counts weighted by the samples below the edge */
    { .num_nodes = 5,
        .is_sample = { 1, 1, 1, 0, 0 },
        .num_mutations = 3,
        .mutations_node = { 0, 3, 3 },
        .mutations_position = { 1, 2, 3 },
        .num_edges = 4,
        .edges_parent = { 3, 3, 4, 4 },
        .edges_child = { 0, 1, 2, 3 },
        .edges_left = { 0, 0, 0, 0 },
        .edges_right = { 10, 10, 10, 10 },
        .indexes_insert = { 0, 1, 2, 3 },
        .indexes_remove = { 3, 2, 1, 0 },
        .sequence_length = 10,
        .size_biased = 0,
        .edges_stats = { { 1, 10 }, { 0, 10 }, { 0, 10 }, { 2, 10 } },
        .mutations_edge = { 0, 3, 3 } },
    { .num_nodes = 5,
        .is_sample = { 1, 1, 1, 0, 0 },
        .num_mutations = 3,
        .mutations_node = { 0, 3, 3 },
        .mutations_position = { 1, 2, 3 },
        .num_edges = 4,
        .edges_parent = { 3, 3, 4, 4 },
        .edges_child = { 0, 1, 2, 3 },
        .edges_left = { 0, 0, 0, 0 },
        .edges_right = { 10, 10, 10, 10 },
        .indexes_insert = { 0, 1, 2, 3 },
        .indexes_remove = { 3, 2, 1, 0 },
        .sequence_length = 10,
        .size_biased = 1,
        .edges_stats = { { 1, 10 }, { 0, 10 }, { 0, 10 }, { 4, 20 } },
        .mutations_edge = { 0, 3, 3 } },
    /* no samples flagged */
    { .num_nodes = 3,
        .is_sample = { 0, 0, 0 },
        .num_mutations = 1,
        .mutations_node = { 0 },
        .mutations_position = { 1 },
        .num_edges = 2,
        .edges_parent = { 2, 2 },
        .edges_child = { 0, 1 },
        .edges_left = { 0, 0 },
        .edges_right = { 10, 10 },
        .indexes_insert = { 0, 1 },
        .indexes_remove = { 1, 0 },
        .sequence_length = 10,
        .size_biased = 1,
        .edges_stats = { { 0, 0 }, { 0, 0 } },
        .mutations_edge = { 0 } },
    /* mutations not in order of position */
    { .num_nodes = 3,
        .is_sample = { 1, 1, 0 },
        .num_mutations = 5,
        .mutations_node = { 1, 0, 2, 1, 0 },
        .mutations_position = { 8, 1, 9, 2, 6 },
        .num_edges = 3,
        .edges_parent = { 2, 2, 2 },
        .edges_child = { 0, 0, 1 },
        .edges_left = { 0, 5, 0 },
        .edges_right = { 5, 10, 10 },
        .indexes_insert = { 0, 2, 1 },
        .indexes_remove = { 0, 2, 1 },
        .sequence_length = 10,
        .size_biased = 0,
        .edges_stats = { { 1, 5 }, { 1, 5 }, { 2, 10 } },
        .mutations_edge = { 2, 0, -1, 2, 1 } },
};

static void
test_count_mutations(void)
{
    const size_t num_rows
        = sizeof(count_mutations_rows) / sizeof(count_mutations_rows[0]);
    const count_mutations_row_t *row;
    double stats[2 * TABLE_MAX_EDGES];
    tsd_id_t mutations_edge[TABLE_MAX_MUTATIONS];
    tsd_size_t k;
    size_t j;
    int ret;

    for (j = 0; j < num_rows; j++) {
        row = count_mutations_rows + j;
        ret = tsd_count_mutations(row->num_nodes, row->is_sample, row->num_mutations,
            row->mutations_node, row->mutations_position, row->num_edges,
            row->edges_parent, row->edges_child, row->edges_left, row->edges_right,
            row->indexes_insert, row->indexes_remove, row->sequence_length,
            row->size_biased, stats, mutations_edge);
        CU_ASSERT_EQUAL(ret, 0);
        assert_array_close(stats, &row->edges_stats[0][0], 2 * row->num_edges);
        for (k = 0; k < row->num_mutations; k++) {
            CU_ASSERT_EQUAL(mutations_edge[k], row->mutations_edge[k]);
        }
    }
}

/* Expected values of the reference implementation */
static const area_row_t area_rows[] = {
    /* a single epoch */
    { .num_nodes = 3,
        .nodes_time = { 0, 0, 1 },
        .num_edges = 2,
        .likelihoods = { { 2, 10 }, { 1, 10 } },
        .edges_parent = { 2, 2 },
        .edges_child = { 0, 1 },
        .num_epochs = 1,
        .counts = { 3 },
        .offset = { 20 },
        .duration = { 1 },
        .nodes_index = { 0, 0, 1 } },
    /* all nodes at time zero: no epoch, and every edge has zero length */
    { .num_nodes = 3,
        .nodes_time = { 0, 0, 0 },
        .num_edges = 2,
        .likelihoods = { { 2, 10 }, { 1, 10 } },
        .edges_parent = { 2, 2 },
        .edges_child = { 0, 1 },
        .num_epochs = 0,
        .nodes_index = { 0, 0, 0 } },
    /* no edges */
    { .num_nodes = 2,
        .nodes_time = { 0, 1 },
        .num_edges = 0,
        .num_epochs = 1,
        .counts = { 0 },
        .offset = { 0 },
        .duration = { 1 },
        .nodes_index = { 0, 1 } },
    /* a single node */
    { .num_nodes = 1,
        .nodes_time = { 0 },
        .num_edges = 0,
        .num_epochs = 0,
        .nodes_index = { 0 } },
    /* no nodes */
    { .num_nodes = 0, .num_edges = 0, .num_epochs = 0 },
    /* an edge of zero length */
    { .num_nodes = 5,
        .nodes_time = { 0, 0, 1, 1, 3 },
        .num_edges = 3,
        .likelihoods = { { 2, 10 }, { 1, 10 }, { 4, 5 } },
        .edges_parent = { 2, 2, 4 },
        .edges_child = { 0, 1, 3 },
        .num_epochs = 2,
        .counts = { 3, 2 },
        .offset = { 20, 5 },
        .duration = { 1, 2 },
        .nodes_index = { 0, 0, 1, 1, 2 } },
    /* an edge whose parent is younger than its child is skipped */
    { .num_nodes = 3,
        .nodes_time = { 0, 1, 0.5 },
        .num_edges = 2,
        .likelihoods = { { 2, 10 }, { 1, 10 } },
        .edges_parent = { 1, 2 },
        .edges_child = { 0, 1 },
        .num_epochs = 2,
        .counts = { 2, 2 },
        .offset = { 10, 10 },
        .duration = { 0.5, 0.5 },
        .nodes_index = { 0, 2, 1 } },
    /* two epochs */
    { .num_nodes = 4,
        .nodes_time = { 0, 0, 1, 3 },
        .num_edges = 2,
        .likelihoods = { { 6, 12 }, { 3, 12 } },
        .edges_parent = { 3, 2 },
        .edges_child = { 0, 1 },
        .num_epochs = 2,
        .counts = { 5, 2 },
        .offset = { 24, 12 },
        .duration = { 1, 2 },
        .nodes_index = { 0, 0, 1, 2 } },
    /* NOTE: current behaviour, possibly a bug: the first epoch always starts at zero,
       even if the youngest node is older */
    { .num_nodes = 2,
        .nodes_time = { 1, 2 },
        .num_edges = 1,
        .likelihoods = { { 2, 10 } },
        .edges_parent = { 1 },
        .edges_child = { 0 },
        .num_epochs = 1,
        .counts = { 2 },
        .offset = { 10 },
        .duration = { 2 },
        .nodes_index = { 0, 1 } },
    /* tied and unsorted times */
    { .num_nodes = 6,
        .nodes_time = { 3, 0, 1, 0, 1, 5 },
        .num_edges = 4,
        .likelihoods = { { 2, 10 }, { 1, 10 }, { 3, 10 }, { 2, 10 } },
        .edges_parent = { 4, 2, 5, 2 },
        .edges_child = { 1, 3, 0, 1 },
        .num_epochs = 3,
        .counts = { 5, 0, 1.5 },
        .offset = { 30, 0, 10 },
        .duration = { 1, 2, 2 },
        .nodes_index = { 2, 0, 1, 0, 1, 3 } },
    /* NaN times are ordered last, whichever their position */
    { .num_nodes = 5,
        .nodes_time = { 0, NAN, 1, NAN, 2 },
        .num_edges = 2,
        .likelihoods = { { 1, 1 }, { 1, 1 } },
        .edges_parent = { 2, 4 },
        .edges_child = { 0, 2 },
        .num_epochs = 2,
        .counts = { 1, 1 },
        .offset = { 1, 1 },
        .duration = { 1, 1 },
        .nodes_index = { 0, 2, 1, 2, 2 } },
};

static void
test_mutational_area(void)
{
    const size_t num_rows = sizeof(area_rows) / sizeof(area_rows[0]);
    const area_row_t *row;
    tsd_epoch_stats_t stats;
    tsd_id_t nodes_index[TABLE_MAX_NODES];
    tsd_size_t k;
    size_t j;
    int ret;

    for (j = 0; j < num_rows; j++) {
        row = area_rows + j;
        ret = tsd_mutational_area(row->num_nodes, row->nodes_time, row->num_edges,
            &row->likelihoods[0][0], row->edges_parent, row->edges_child, &stats,
            nodes_index);
        CU_ASSERT_EQUAL_FATAL(ret, 0);
        CU_ASSERT_EQUAL_FATAL(stats.num_epochs, row->num_epochs);
        assert_array_close(stats.counts, row->counts, row->num_epochs);
        assert_array_close(stats.offset, row->offset, row->num_epochs);
        assert_array_close(stats.duration, row->duration, row->num_epochs);
        for (k = 0; k < row->num_nodes; k++) {
            CU_ASSERT_EQUAL(nodes_index[k], row->nodes_index[k]);
        }
        tsd_epoch_stats_free(&stats);
    }
}

/* Expected values of the reference implementation */
static const timescale_row_t timescale_rows[] = {
    { .num_nodes = 4,
        .nodes_time = { 0, 0, 1, 2 },
        .num_edges = 3,
        .likelihoods = { { 2, 10 }, { 1, 10 }, { 3, 10 } },
        .edges_parent = { 2, 2, 3 },
        .edges_child = { 0, 1, 2 },
        .max_intervals = 1,
        .num_breaks = 2,
        .origin = { 0, 2 },
        .adjust = { 0, 0.4 },
        .ret = 0 },
    { .num_nodes = 4,
        .nodes_time = { 0, 0, 1, 2 },
        .num_edges = 3,
        .likelihoods = { { 2, 10 }, { 1, 10 }, { 3, 10 } },
        .edges_parent = { 2, 2, 3 },
        .edges_child = { 0, 1, 2 },
        .max_intervals = 2,
        .num_breaks = 2,
        .origin = { 0, 2 },
        .adjust = { 0, 0.4 },
        .ret = 0 },
    /* as many intervals as epochs */
    { .num_nodes = 4,
        .nodes_time = { 0, 0, 1, 2 },
        .num_edges = 3,
        .likelihoods = { { 2, 10 }, { 1, 10 }, { 3, 10 } },
        .edges_parent = { 2, 2, 3 },
        .edges_child = { 0, 1, 2 },
        .max_intervals = 3,
        .num_breaks = 3,
        .origin = { 0, 1, 2 },
        .adjust = { 0, 0.15, 0.44999999999999996 },
        .ret = 0 },
    /* more intervals than epochs */
    { .num_nodes = 4,
        .nodes_time = { 0, 0, 1, 2 },
        .num_edges = 3,
        .likelihoods = { { 2, 10 }, { 1, 10 }, { 3, 10 } },
        .edges_parent = { 2, 2, 3 },
        .edges_child = { 0, 1, 2 },
        .max_intervals = 10000,
        .num_breaks = 3,
        .origin = { 0, 1, 2 },
        .adjust = { 0, 0.15, 0.44999999999999996 },
        .ret = 0 },
    { .num_nodes = 4,
        .nodes_time = { 0, 0, 1, 2 },
        .num_edges = 3,
        .likelihoods = { { 2, 10 }, { 1, 10 }, { 3, 10 } },
        .edges_parent = { 2, 2, 3 },
        .edges_child = { 0, 1, 2 },
        .max_intervals = 0,
        .ret = TSD_ERR_NONPOSITIVE_INTERVALS },
    { .num_nodes = 4,
        .nodes_time = { 0, 0, 1, 2 },
        .num_edges = 3,
        .likelihoods = { { 2, 10 }, { 1, 10 }, { 3, 10 } },
        .edges_parent = { 2, 2, 3 },
        .edges_child = { 0, 1, 2 },
        .max_intervals = -1,
        .ret = TSD_ERR_NONPOSITIVE_INTERVALS },
    /* a historical sample */
    { .num_nodes = 4,
        .nodes_time = { 0, 500, 100, 700 },
        .num_edges = 3,
        .likelihoods = { { 2, 10 }, { 1, 10 }, { 3, 10 } },
        .edges_parent = { 2, 3, 3 },
        .edges_child = { 0, 2, 1 },
        .max_intervals = 1,
        .num_breaks = 2,
        .origin = { 0, 700 },
        .adjust = { 0, 0.6708333333333333 },
        .ret = 0 },
    { .num_nodes = 4,
        .nodes_time = { 0, 500, 100, 700 },
        .num_edges = 3,
        .likelihoods = { { 2, 10 }, { 1, 10 }, { 3, 10 } },
        .edges_parent = { 2, 3, 3 },
        .edges_child = { 0, 2, 1 },
        .max_intervals = 3,
        .num_breaks = 4,
        .origin = { 0, 100, 500, 700 },
        .adjust = { 0, 0.2, 0.2666666666666667, 0.4333333333333334 },
        .ret = 0 },
    { .num_nodes = 4,
        .nodes_time = { 0, 500, 100, 700 },
        .num_edges = 3,
        .likelihoods = { { 2, 10 }, { 1, 10 }, { 3, 10 } },
        .edges_parent = { 2, 3, 3 },
        .edges_child = { 0, 2, 1 },
        .max_intervals = 100,
        .num_breaks = 4,
        .origin = { 0, 100, 500, 700 },
        .adjust = { 0, 0.2, 0.2666666666666667, 0.4333333333333334 },
        .ret = 0 },
    /* the top edge has no span, so the last epoch has no weight and is merged */
    { .num_nodes = 4,
        .nodes_time = { 0, 0, 1, 2 },
        .num_edges = 3,
        .likelihoods = { { 2, 10 }, { 1, 10 }, { 3, 0 } },
        .edges_parent = { 2, 2, 3 },
        .edges_child = { 0, 1, 2 },
        .max_intervals = 1,
        .num_breaks = 2,
        .origin = { 0, 2 },
        .adjust = { 0, 0.6 },
        .ret = 0 },
    /* an edge of negative length is skipped from the totals */
    { .num_nodes = 4,
        .nodes_time = { 0, 0, 2, 1 },
        .num_edges = 3,
        .likelihoods = { { 2, 10 }, { 1, 10 }, { 5, 10 } },
        .edges_parent = { 2, 2, 3 },
        .edges_child = { 0, 1, 2 },
        .max_intervals = 2,
        .num_breaks = 3,
        .origin = { 0, 1, 2 },
        .adjust = { 0, 0.075, 0.15 },
        .ret = 0 },
    /* the total weight is zero, so the first changepoint is -1, which counts from the
       end */
    { .num_nodes = 6,
        .nodes_time = { 0, 0, 1, 2, 3, 4 },
        .num_edges = 4,
        .likelihoods = { { 1, 1 }, { 1, 1 }, { 1, -3 }, { 1, 1 } },
        .edges_parent = { 2, 3, 4, 5 },
        .edges_child = { 0, 2, 3, 4 },
        .max_intervals = 2,
        .num_breaks = 2,
        .origin = { 4, 4 },
        .adjust = { 0, 1 },
        .ret = 0 },
    /* NOTE: current behaviour, possibly a bug: every edge over an interval has zero span
       (a zero mutation rate) or there are no edges */
    { .num_nodes = 3,
        .nodes_time = { 0, 0, 1 },
        .num_edges = 2,
        .likelihoods = { { 2, 0 }, { 1, 0 } },
        .edges_parent = { 2, 2 },
        .edges_child = { 0, 1 },
        .max_intervals = 1,
        .ret = TSD_ERR_ZERO_EDGE_SPAN },
    { .num_nodes = 2,
        .nodes_time = { 0, 1 },
        .num_edges = 0,
        .max_intervals = 1,
        .ret = TSD_ERR_ZERO_EDGE_SPAN },
    /* all nodes at time zero: no epochs, and the span is zero */
    { .num_nodes = 3,
        .nodes_time = { 0, 0, 0 },
        .num_edges = 2,
        .likelihoods = { { 1, 10 }, { 1, 10 } },
        .edges_parent = { 2, 2 },
        .edges_child = { 0, 1 },
        .max_intervals = 1,
        .ret = TSD_ERR_ZERO_EDGE_SPAN },
};

static void
test_mutational_timescale(void)
{
    const size_t num_rows = sizeof(timescale_rows) / sizeof(timescale_rows[0]);
    const timescale_row_t *row;
    double *origin = NULL;
    double *adjust = NULL;
    tsd_size_t num_breaks;
    size_t j;
    int ret;

    for (j = 0; j < num_rows; j++) {
        row = timescale_rows + j;
        ret = tsd_mutational_timescale(row->num_nodes, row->nodes_time, row->num_edges,
            &row->likelihoods[0][0], row->edges_parent, row->edges_child,
            row->max_intervals, &num_breaks, &origin, &adjust);
        CU_ASSERT_EQUAL(ret, row->ret);
        if (ret == 0 && row->ret == 0) {
            CU_ASSERT_EQUAL_FATAL(num_breaks, row->num_breaks);
            assert_array_close(origin, row->origin, num_breaks);
            assert_array_close(adjust, row->adjust, num_breaks);
        }
        tsd_safe_free(origin);
        tsd_safe_free(adjust);
        origin = NULL;
        adjust = NULL;
    }
}

/* Expected values of the reference implementation */
static const point_row_t point_rows[] = {
    { .num_points = 5,
        .estimate = { 0, 0.5, 1, 2, 3 },
        .fixed = { 0, 0, 0, 0, 0 },
        .num_breaks = 3,
        .original = { 0, 1, 3 },
        .rescaled = { 0, 2, 3.5 },
        .out = { 0, 1, 2, 2.75, 3.5 },
        .ret = 0 },
    /* the last piece is flat, so everything beyond it maps to the last point */
    { .num_points = 5,
        .estimate = { 3, 4, 10, 1e+300, INFINITY },
        .fixed = { 0, 0, 0, 0, 0 },
        .num_breaks = 3,
        .original = { 0, 1, 3 },
        .rescaled = { 0, 2, 3.5 },
        .out = { 3.5, 3.5, 3.5, 3.5, NAN },
        .ret = 0 },
    /* NOTE: current behaviour, possibly a bug: a time below the first break has search
       index -1, which wraps to the last breakpoint; -inf gives NaN */
    { .num_points = 4,
        .estimate = { -1, -1e-09, 0, -INFINITY },
        .fixed = { 0, 0, 0, 0 },
        .num_breaks = 3,
        .original = { 0, 1, 3 },
        .rescaled = { 0, 2, 3.5 },
        .out = { 3.5, 3.5, 0, NAN },
        .ret = 0 },
    /* NaN, fixed or not */
    { .num_points = 3,
        .estimate = { NAN, 1, NAN },
        .fixed = { 1, 0, 0 },
        .num_breaks = 3,
        .original = { 0, 1, 3 },
        .rescaled = { 0, 2, 3.5 },
        .out = { NAN, 2, NAN },
        .ret = 0 },
    /* some fixed */
    { .num_points = 4,
        .estimate = { 0, 0.5, 3, 4 },
        .fixed = { 1, 0, 1, 1 },
        .num_breaks = 3,
        .original = { 0, 1, 3 },
        .rescaled = { 0, 2, 3.5 },
        .out = { 0, 1, 3, 4 },
        .ret = 0 },
    { .num_points = 0,
        .num_breaks = 3,
        .original = { 0, 1, 3 },
        .rescaled = { 0, 2, 3.5 },
        .ret = 0 },
    /* a single breakpoint */
    { .num_points = 3,
        .estimate = { 0, 0.5, 7 },
        .fixed = { 0, 0, 0 },
        .num_breaks = 1,
        .original = { 0 },
        .rescaled = { 2 },
        .out = { 2, 2, 2 },
        .ret = 0 },
    { .num_points = 3,
        .estimate = { 0, 0.3, 1 },
        .fixed = { 0, 0, 0 },
        .num_breaks = 2,
        .original = { 0, 5 },
        .rescaled = { 0, 5 },
        .out = { 0, 0.3, 1 },
        .ret = 0 },
    { .num_points = 1,
        .estimate = { 0.5 },
        .fixed = { 0 },
        .num_breaks = 0,
        .ret = TSD_ERR_NO_BREAKPOINTS },
    { .num_points = 1,
        .estimate = { 0.5 },
        .fixed = { 0 },
        .num_breaks = 3,
        .original = { 0, 1, 3 },
        .rescaled = { 0, 2, 2 },
        .ret = TSD_ERR_NON_INCREASING_BREAKPOINTS },
    { .num_points = 1,
        .estimate = { 0.5 },
        .fixed = { 0 },
        .num_breaks = 3,
        .original = { 0, 1, 3 },
        .rescaled = { 0, 2, 1 },
        .ret = TSD_ERR_NON_INCREASING_BREAKPOINTS },
    { .num_points = 1,
        .estimate = { 0.5 },
        .fixed = { 0 },
        .num_breaks = 3,
        .original = { 0, 2, 1 },
        .rescaled = { 0, 1, 2 },
        .ret = TSD_ERR_NON_INCREASING_BREAKPOINTS },
    { .num_points = 1,
        .estimate = { 0.5 },
        .fixed = { 0 },
        .num_breaks = 3,
        .original = { 0, 1, NAN },
        .rescaled = { 0, 1, 2 },
        .ret = TSD_ERR_NON_INCREASING_BREAKPOINTS },
    { .num_points = 1,
        .estimate = { 0.5 },
        .fixed = { 0 },
        .num_breaks = 3,
        .original = { 0, 1, 2 },
        .rescaled = { 0, 1, NAN },
        .ret = TSD_ERR_NON_INCREASING_BREAKPOINTS },
};

static void
test_piecewise_scale_point_estimate(void)
{
    const size_t num_rows = sizeof(point_rows) / sizeof(point_rows[0]);
    const point_row_t *row;
    double out[TABLE_MAX_POINTS];
    size_t j;
    int ret;

    for (j = 0; j < num_rows; j++) {
        row = point_rows + j;
        ret = tsd_piecewise_scale_point_estimate(row->num_points, row->estimate,
            row->fixed, row->num_breaks, row->original, row->rescaled, out);
        CU_ASSERT_EQUAL(ret, row->ret);
        if (ret == 0 && row->ret == 0) {
            assert_array_close(out, row->out, row->num_points);
        }
    }
}

/* Expected values of the reference implementation */
static const posterior_row_t posterior_rows[] = {
    { .num_posteriors = 4,
        .posteriors = { { 1, 1 }, { 2, 3 }, { 0, 1 }, { 5, 2 } },
        .fixed = { 0, 0, 0, 0 },
        .num_breaks = 3,
        .original = { 0, 1, 3 },
        .rescaled = { 0, 2, 3.5 },
        .quantile_width = 0.5,
        .max_shape = 1000,
        .out = { { 5.752572529118271, 2.455480919679371 },
            { 3.460836454696765, 2.2304182273483826 },
            { 0.22573432805250393, 0.612867164026252 },
            { 39.903223991686914, 11.686635426196261 } },
        .ret = 0 },
    { .num_posteriors = 4,
        .posteriors = { { 1, 1 }, { 2, 3 }, { 0, 1 }, { 5, 2 } },
        .fixed = { 0, 0, 0, 0 },
        .num_breaks = 3,
        .original = { 0, 1, 3 },
        .rescaled = { 0, 2, 3.5 },
        .quantile_width = 0.001,
        .max_shape = 1000,
        .out = { { 2.471583651492129, 1.2623940550880468 },
            { 3.210845310401922, 2.105422655200961 },
            { 0.23603088036204523, 0.6180154401810226 },
            { 26.04896064141572, 7.72827446897592 } },
        .ret = 0 },
    { .num_posteriors = 4,
        .posteriors = { { 1, 1 }, { 2, 3 }, { 0, 1 }, { 5, 2 } },
        .fixed = { 0, 0, 0, 0 },
        .num_breaks = 3,
        .original = { 0, 1, 3 },
        .rescaled = { 0, 2, 3.5 },
        .quantile_width = 0.999999999,
        .max_shape = 1000,
        .out = { { 6.175396703307324, 2.609235164839027 },
            { 1.9999982566565375, 1.4999991283282688 },
            { -2.6908024519656237e-07, 0.4999998654598774 },
            { 13.715489108570114, 4.204425459591461 } },
        .ret = 0 },
    { .num_posteriors = 4,
        .posteriors = { { 1, 1 }, { 2, 3 }, { 0, 1 }, { 5, 2 } },
        .fixed = { 0, 0, 0, 0 },
        .num_breaks = 3,
        .original = { 0, 1, 3 },
        .rescaled = { 0, 2, 3.5 },
        .quantile_width = 0,
        .max_shape = 1000,
        .ret = TSD_ERR_INVALID_QUANTILE_WIDTH },
    { .num_posteriors = 4,
        .posteriors = { { 1, 1 }, { 2, 3 }, { 0, 1 }, { 5, 2 } },
        .fixed = { 0, 0, 0, 0 },
        .num_breaks = 3,
        .original = { 0, 1, 3 },
        .rescaled = { 0, 2, 3.5 },
        .quantile_width = 1,
        .max_shape = 1000,
        .ret = TSD_ERR_INVALID_QUANTILE_WIDTH },
    { .num_posteriors = 4,
        .posteriors = { { 1, 1 }, { 2, 3 }, { 0, 1 }, { 5, 2 } },
        .fixed = { 0, 0, 0, 0 },
        .num_breaks = 3,
        .original = { 0, 1, 3 },
        .rescaled = { 0, 2, 3.5 },
        .quantile_width = NAN,
        .max_shape = 1000,
        .ret = TSD_ERR_INVALID_QUANTILE_WIDTH },
    { .num_posteriors = 4,
        .posteriors = { { 1, 1 }, { 2, 3 }, { 0, 1 }, { 5, 2 } },
        .fixed = { 1, 0, 1, 0 },
        .num_breaks = 3,
        .original = { 0, 1, 3 },
        .rescaled = { 0, 2, 3.5 },
        .quantile_width = 0.5,
        .max_shape = 1000,
        .out = { { NAN, NAN }, { 3.460836454696765, 2.2304182273483826 }, { NAN, NAN },
            { 39.903223991686914, 11.686635426196261 } },
        .ret = 0 },
    /* fixed rows are not validated */
    { .num_posteriors = 2,
        .posteriors = { { NAN, NAN }, { 1, 1 } },
        .fixed = { 1, 0 },
        .num_breaks = 3,
        .original = { 0, 1, 3 },
        .rescaled = { 0, 2, 3.5 },
        .quantile_width = 0.5,
        .max_shape = 1000,
        .out = { { NAN, NAN }, { 5.752572529118271, 2.455480919679371 } },
        .ret = 0 },
    /* NOTE: current behaviour, possibly a bug: no posteriors is an error */
    { .num_posteriors = 0,
        .num_breaks = 3,
        .original = { 0, 1, 3 },
        .rescaled = { 0, 2, 3.5 },
        .quantile_width = 0.5,
        .max_shape = 1000,
        .ret = TSD_ERR_NO_POSTERIORS },
    { .num_posteriors = 4,
        .posteriors
        = { { -0.99, 1 }, { 1, 1e-300 }, { 1, 1e+300 }, { 1000000, 1000000 } },
        .fixed = { 0, 0, 0, 0 },
        .num_breaks = 3,
        .original = { 0, 1, 3 },
        .rescaled = { 0, 2, 3.5 },
        .quantile_width = 0.5,
        .max_shape = 1000,
        .out = { { -0.99, 0.5 }, { 999, 285.7142857142857 },
            { 1.000000000000004, 5.00000000000001e+299 }, { 999, 499.99981250007033 } },
        .ret = 0 },
    /* NOTE: current behaviour, possibly a bug: for a shape this close to -1 the lower
       quantile underflows to zero and the division by it is an error (IEEE would give
       inf), rather than a failed gamma fit */
    { .num_posteriors = 1,
        .posteriors = { { -0.9999, 1 } },
        .fixed = { 0 },
        .num_breaks = 3,
        .original = { 0, 1, 3 },
        .rescaled = { 0, 2, 3.5 },
        .quantile_width = 0.5,
        .max_shape = 1000,
        .ret = TSD_ERR_ZERO_DIVISION },
    { .num_posteriors = 1,
        .posteriors = { { 1, INFINITY } },
        .fixed = { 0 },
        .num_breaks = 3,
        .original = { 0, 1, 3 },
        .rescaled = { 0, 2, 3.5 },
        .quantile_width = 0.5,
        .max_shape = 1000,
        .ret = TSD_ERR_ZERO_DIVISION },
    /* NOTE: current behaviour, possibly a bug: the error of tsd_approximate_gamma_iqr is
       passed on */
    { .num_posteriors = 1,
        .posteriors = { { INFINITY, 1 } },
        .fixed = { 0 },
        .num_breaks = 3,
        .original = { 0, 1, 3 },
        .rescaled = { 0, 2, 3.5 },
        .quantile_width = 0.5,
        .max_shape = 1000,
        .ret = TSD_ERR_UNSORTED_QUANTILES },
    { .num_posteriors = 4,
        .posteriors = { { 1, 1 }, { 2, 3 }, { 0, 1 }, { -1, 2 } },
        .fixed = { 0, 0, 0, 0 },
        .num_breaks = 3,
        .original = { 0, 1, 3 },
        .rescaled = { 0, 2, 3.5 },
        .quantile_width = 0.5,
        .max_shape = 1000,
        .ret = TSD_ERR_INVALID_POSTERIOR },
    { .num_posteriors = 1,
        .posteriors = { { 1, NAN } },
        .fixed = { 0 },
        .num_breaks = 3,
        .original = { 0, 1, 3 },
        .rescaled = { 0, 2, 3.5 },
        .quantile_width = 0.5,
        .max_shape = 1000,
        .ret = TSD_ERR_INVALID_POSTERIOR },
    /* all quantiles map to one point beyond the last break: zero interquantile range */
    { .num_posteriors = 2,
        .posteriors = { { 1, 0.001 }, { 1, 1e-09 } },
        .fixed = { 0, 0 },
        .num_breaks = 3,
        .original = { 0, 1, 3 },
        .rescaled = { 0, 2, 3.5 },
        .quantile_width = 0.5,
        .max_shape = 1000,
        .out = { { 999, 285.7142857142857 }, { 999, 285.7142857142857 } },
        .ret = 0 },
    { .num_posteriors = 4,
        .posteriors = { { 1, 1 }, { 2, 3 }, { 0, 1 }, { 5, 2 } },
        .fixed = { 0, 0, 0, 0 },
        .num_breaks = 3,
        .original = { 0.5, 1, 3 },
        .rescaled = { 0, 2, 3.5 },
        .quantile_width = 0.5,
        .max_shape = 1000,
        .ret = TSD_ERR_TIME_BELOW_FIRST_BREAKPOINT },
    /* the lower quantile is below the first breakpoint, the mean is not */
    { .num_posteriors = 1,
        .posteriors = { { 1, 1 } },
        .fixed = { 0 },
        .num_breaks = 3,
        .original = { 1.5, 3, 5 },
        .rescaled = { 0, 1, 2 },
        .quantile_width = 0.5,
        .max_shape = 1000,
        .ret = TSD_ERR_TIME_BELOW_FIRST_BREAKPOINT },
    /* the rescaled quantiles of fixed posteriors are zero, and are checked too */
    { .num_posteriors = 1,
        .posteriors = { { 1, 1 } },
        .fixed = { 1 },
        .num_breaks = 3,
        .original = { 0.5, 2, 5 },
        .rescaled = { 0, 1, 2 },
        .quantile_width = 0.5,
        .max_shape = 1000,
        .ret = TSD_ERR_TIME_BELOW_FIRST_BREAKPOINT },
    /* NOTE: current behaviour, possibly a bug: max_shape = 0 gives shape -1 (improper)
       and zero rate */
    { .num_posteriors = 4,
        .posteriors = { { 1, 1 }, { 2, 3 }, { 0, 1 }, { 5, 2 } },
        .fixed = { 0, 0, 0, 0 },
        .num_breaks = 3,
        .original = { 0, 1, 3 },
        .rescaled = { 0, 2, 3.5 },
        .quantile_width = 0.5,
        .max_shape = 0,
        .out = { { -1, 0 }, { -1, 0 }, { -1, 0 }, { -1, 0 } },
        .ret = 0 },
    { .num_posteriors = 4,
        .posteriors = { { 1, 1 }, { 2, 3 }, { 0, 1 }, { 5, 2 } },
        .fixed = { 0, 0, 0, 0 },
        .num_breaks = 3,
        .original = { 0, 1, 3 },
        .rescaled = { 0, 2, 3.5 },
        .quantile_width = 0.5,
        .max_shape = 1,
        .out = { { 0, 0.36363636363636365 }, { 0, 0.5 }, { 0, 0.5 },
            { 0, 0.2857142857142857 } },
        .ret = 0 },
    { .num_posteriors = 4,
        .posteriors = { { 1, 1 }, { 2, 3 }, { 0, 1 }, { 5, 2 } },
        .fixed = { 0, 0, 0, 0 },
        .num_breaks = 3,
        .original = { 0, 1, 3 },
        .rescaled = { 0, 2, 3.5 },
        .quantile_width = 0.5,
        .max_shape = NAN,
        .out = { { 5.752572529118271, 2.455480919679371 },
            { 3.460836454696765, 2.2304182273483826 },
            { 0.22573432805250393, 0.612867164026252 },
            { 39.903223991686914, 11.686635426196261 } },
        .ret = 0 },
    { .num_posteriors = 4,
        .posteriors = { { 1, 1 }, { 2, 3 }, { 0, 1 }, { 5, 2 } },
        .fixed = { 0, 0, 0, 0 },
        .num_breaks = 3,
        .original = { 0, 1, 1 },
        .rescaled = { 0, 2, 3.5 },
        .quantile_width = 0.5,
        .max_shape = 1000,
        .ret = TSD_ERR_NON_INCREASING_BREAKPOINTS },
    { .num_posteriors = 4,
        .posteriors = { { 1, 1 }, { 2, 3 }, { 0, 1 }, { 5, 2 } },
        .fixed = { 0, 0, 0, 0 },
        .num_breaks = 3,
        .original = { 0, 1, 3 },
        .rescaled = { 0, 2, 2 },
        .quantile_width = 0.5,
        .max_shape = 1000,
        .ret = TSD_ERR_NON_INCREASING_BREAKPOINTS },
    /* the breaks are checked before the posteriors */
    { .num_posteriors = 1,
        .posteriors = { { -5, 1 } },
        .fixed = { 0 },
        .num_breaks = 2,
        .original = { 0, 0 },
        .rescaled = { 0, 1 },
        .quantile_width = 0.5,
        .max_shape = 1000,
        .ret = TSD_ERR_NON_INCREASING_BREAKPOINTS },
    /* a single breakpoint at zero: every quantile maps to zero */
    { .num_posteriors = 4,
        .posteriors = { { 1, 1 }, { 2, 3 }, { 0, 1 }, { 5, 2 } },
        .fixed = { 0, 0, 0, 0 },
        .num_breaks = 1,
        .original = { 0 },
        .rescaled = { 0 },
        .quantile_width = 0.5,
        .max_shape = 1000,
        .ret = TSD_ERR_ZERO_DIVISION },
    { .num_posteriors = 4,
        .posteriors = { { 1, 1 }, { 2, 3 }, { 0, 1 }, { 5, 2 } },
        .fixed = { 0, 0, 0, 0 },
        .num_breaks = 1,
        .original = { 0 },
        .rescaled = { 4 },
        .quantile_width = 0.5,
        .max_shape = 1000,
        .out = { { 999, 250 }, { 999, 250 }, { 999, 250 }, { 999, 250 } },
        .ret = 0 },
    { .num_posteriors = 4,
        .posteriors = { { 1, 1 }, { 2, 3 }, { 0, 1 }, { 5, 2 } },
        .fixed = { 0, 0, 0, 0 },
        .num_breaks = 0,
        .quantile_width = 0.5,
        .max_shape = 1000,
        .ret = TSD_ERR_NO_BREAKPOINTS },
};

static void
test_piecewise_scale_posterior(void)
{
    const size_t num_rows = sizeof(posterior_rows) / sizeof(posterior_rows[0]);
    const posterior_row_t *row;
    double out[2 * TABLE_MAX_POSTERIORS];
    size_t j;
    int ret;

    for (j = 0; j < num_rows; j++) {
        row = posterior_rows + j;
        ret = tsd_piecewise_scale_posterior(row->num_posteriors, &row->posteriors[0][0],
            row->fixed, row->num_breaks, row->original, row->rescaled,
            row->quantile_width, row->max_shape, out);
        CU_ASSERT_EQUAL(ret, row->ret);
        if (ret == 0 && row->ret == 0) {
            assert_array_close(out, &row->out[0][0], 2 * row->num_posteriors);
        }
    }
}

static void
handle_cunit_error(void)
{
    fprintf(
        stderr, "CUnit error occurred: %d: %s\n", CU_get_error(), CU_get_error_msg());
    exit(EXIT_FAILURE);
}

/* Nodes 0 and 1 are fixed, at ages 1 and 5; 2 and 3 are free. */
static const double ep_constraints[8] = { 1, 1, 5, 5, 0, INFINITY, 0, INFINITY };
static const double ep_likelihoods[8] = { 1, 0.1, 1, 0.1, 1, 0.1, 1, 0.1 };
static const double ep_huge[8] = { 1, 1e300, 1, 1e300, 1, 1e300, 1, 1e300 };
static const tsd_id_t ep_parent[4] = { 2, 1, 3, 1 };
static const tsd_id_t ep_child[4] = { 0, 3, 2, 0 };
static const tsd_id_t ep_twin_parent[4] = { 2, 1, 3, 2 };
static const tsd_id_t ep_twin_child[4] = { 0, 3, 2, 2 };

/* Four nodes with four edges, and the same four as blocks, with posterior `post`. */
static void
ep_init(tsd_ep_t *ep, const tsd_id_t *parent, const tsd_id_t *child,
    const double *likelihoods, const double *post)
{
    int ret;

    ret = tsd_ep_init(ep, 4, 4, 4, ep_constraints, parent, child, likelihoods, parent,
        child, likelihoods);
    CU_ASSERT_FATAL(ret == 0);
    memcpy(ep->posterior, post, 8 * sizeof(*post));
}

typedef struct {
    double x[2];
    double y[2];
    double s;
    double d;
    int ret;
} damp_row_t;

typedef struct {
    double x[2];
    double s;
    double d;
    int ret;
} rescale_row_t;

/* Expected values of the reference implementation */
static const damp_row_t damp_rows[] = {
    { .x = { 0, 0 }, .y = { 0, 0 }, .s = 0.5, .d = 1, .ret = 0 },
    /* zero x and y: s is not checked */
    { .x = { 0, 0 }, .y = { 0, 0 }, .s = 5, .d = 1, .ret = 0 },
    { .x = { 0, 1 }, .y = { 0, 0 }, .s = 0.5, .d = 1, .ret = 0 },
    { .x = { 1, 2 }, .y = { 0, 0 }, .s = 0.5, .d = 1, .ret = 0 },
    { .x = { 1, 2 }, .y = { 0.9999999999999999, 0 }, .s = 0.5, .d = 1, .ret = 0 },
    { .x = { 1, 2 }, .y = { 1, 0 }, .s = 0.5, .d = 1, .ret = 0 },
    { .x = { 1, 2 },
        .y = { 1.0000000000000002, 0 },
        .s = 0.5,
        .d = 0.9999999999999998,
        .ret = 0 },
    { .x = { 1, 2 }, .y = { 4, 0 }, .s = 0.5, .d = 0.25, .ret = 0 },
    { .x = { 1, 2 }, .y = { -5, 0 }, .s = 0.5, .d = 1, .ret = 0 },
    { .x = { 1, 2 }, .y = { 0, 0.9999999999999999 }, .s = 0.5, .d = 1, .ret = 0 },
    { .x = { 1, 2 }, .y = { 0, 1 }, .s = 0.5, .d = 1, .ret = 0 },
    { .x = { 1, 2 },
        .y = { 0, 1.0000000000000002 },
        .s = 0.5,
        .d = 0.9999999999999998,
        .ret = 0 },
    { .x = { 1, 2 }, .y = { 0, 8 }, .s = 0.5, .d = 0.125, .ret = 0 },
    { .x = { 1, 2 }, .y = { 0, -5 }, .s = 0.5, .d = 1, .ret = 0 },
    { .x = { 1, 2 }, .y = { 4, 2 }, .s = 0.5, .d = 0.25, .ret = 0 },
    { .x = { 1, 2 }, .y = { 2, 8 }, .s = 0.5, .d = 0.125, .ret = 0 },
    { .x = { 1, 2 }, .y = { -1, -1 }, .s = 0.5, .d = 1, .ret = 0 },
    { .x = { 1, 2 }, .y = { 2, 2 }, .s = 5e-324, .d = 1, .ret = 0 },
    { .x = { 1, 2 },
        .y = { 2, 2 },
        .s = 0.9999999999999999,
        .d = 1.1102230246251565e-16,
        .ret = 0 },
    { .x = { -0.9999999999999999, 1 }, .y = { 0, 0 }, .s = 0.5, .d = 1, .ret = 0 },
    { .x = { 1, 1e-300 }, .y = { 1, 4 }, .s = 0.5, .d = 1.25e-301, .ret = 0 },
    /* NOTE: current behaviour, possibly a bug: the minimum swallows a NaN in the second
       component of y, so the damping is silently skipped */
    { .x = { 1, 1 }, .y = { 0, NAN }, .s = 0.5, .d = 1, .ret = 0 },
    { .x = { 1, 1 }, .y = { 8, NAN }, .s = 0.5, .d = 0.125, .ret = 0 },
    { .x = { 1, 1 }, .y = { -INFINITY, 0 }, .s = 0.5, .d = 1, .ret = 0 },
    { .x = { 1, 1 }, .y = { 0, -INFINITY }, .s = 0.5, .d = 1, .ret = 0 },
    { .x = { 1, 1 }, .y = { 0, 0 }, .s = 0, .ret = TSD_ERR_INVALID_MIN_STEP },
    { .x = { 1, 1 }, .y = { 0, 0 }, .s = 1, .ret = TSD_ERR_INVALID_MIN_STEP },
    { .x = { 1, 1 }, .y = { 0, 0 }, .s = NAN, .ret = TSD_ERR_INVALID_MIN_STEP },
    { .x = { -1, 1 }, .y = { 0, 0 }, .s = 0.5, .ret = TSD_ERR_INVALID_NODE_POSTERIOR },
    { .x = { NAN, 1 }, .y = { 0, 0 }, .s = 0.5, .ret = TSD_ERR_INVALID_NODE_POSTERIOR },
    { .x = { 1, 0 }, .y = { 0, 0 }, .s = 0.5, .ret = TSD_ERR_INVALID_NODE_POSTERIOR },
    { .x = { 1, NAN }, .y = { 0, 0 }, .s = 0.5, .ret = TSD_ERR_INVALID_NODE_POSTERIOR },
    { .x = { 0, 0 }, .y = { 1, 1 }, .s = 0.5, .ret = TSD_ERR_INVALID_NODE_POSTERIOR },
    /* d underflows */
    { .x = { 1, 1e-300 }, .y = { 0, 1e+300 }, .s = 0.5, .ret = TSD_ERR_INVALID_DAMPING },
    { .x = { 1, 1 }, .y = { NAN, 0 }, .s = 0.5, .ret = TSD_ERR_INVALID_DAMPING },
    /* d is rounded to just above one */
    { .x = { 24.23005039128842, 15.403466578537675 },
        .y = { 2.409458061921995, 1.471023884360475 },
        .s = 0.9045004657321672,
        .ret = TSD_ERR_INVALID_DAMPING },
    /* an infinite posterior and a zero message: a division by zero */
    { .x = { INFINITY, 1 }, .y = { 0, 0 }, .s = 0.5, .ret = TSD_ERR_ZERO_DIVISION },
    /* an infinite posterior and a zero message: a division by zero */
    { .x = { 1, INFINITY }, .y = { 0, 0 }, .s = 0.5, .ret = TSD_ERR_ZERO_DIVISION },
};

static void
test_damp(void)
{
    const size_t num_rows = sizeof(damp_rows) / sizeof(damp_rows[0]);
    const damp_row_t *row;
    double d;
    size_t j;
    int ret;

    for (j = 0; j < num_rows; j++) {
        row = damp_rows + j;
        ret = tsd_damp(row->x, row->y, row->s, &d);
        CU_ASSERT_EQUAL(ret, row->ret);
        if (ret == 0 && row->ret == 0) {
            assert_close_ulps(d, row->d, ULPS);
        }
    }
}

/* Expected values of the reference implementation */
static const rescale_row_t rescale_rows[] = {
    { .x = { 0, 0 }, .s = 10, .d = 1, .ret = 0 },
    /* zero x: s is not checked */
    { .x = { 0, 0 }, .s = 0, .d = 1, .ret = 0 },
    { .x = { 0, 1 }, .s = 10, .d = 1, .ret = 0 },
    { .x = { 5, 1 }, .s = 10, .d = 1, .ret = 0 },
    { .x = { -0.5, 1 }, .s = 2, .d = 1, .ret = 0 },
    { .x = { 8.999999999999998, 1 }, .s = 10, .d = 1, .ret = 0 },
    { .x = { 9, 1 }, .s = 10, .d = 1, .ret = 0 },
    { .x = { 9.000000000000002, 1 }, .s = 10, .d = 0.9999999999999998, .ret = 0 },
    { .x = { -0.49999999999999994, 1 }, .s = 2, .d = 1, .ret = 0 },
    { .x = { -0.5000000000000001, 1 }, .s = 2, .d = 0.9999999999999998, .ret = 0 },
    { .x = { -0.95, 1 }, .s = 10, .d = 0.9473684210526316, .ret = 0 },
    { .x = { 99, INFINITY }, .s = 10, .d = 0.09090909090909091, .ret = 0 },
    { .x = { INFINITY, 1 }, .s = 10, .d = 0, .ret = 0 },
    /* NOTE: current behaviour, possibly a bug: with max_shape = 1 the factor is 0 and
       wipes out the posterior */
    { .x = { 1, 1 }, .s = 1, .d = 0, .ret = 0 },
    { .x = { -0.5, 1 }, .s = 1, .d = -0, .ret = 0 },
    { .x = { -1, 1 }, .s = 10, .ret = TSD_ERR_INVALID_NODE_POSTERIOR },
    { .x = { NAN, 1 }, .s = 10, .ret = TSD_ERR_INVALID_NODE_POSTERIOR },
    { .x = { 1, 0 }, .s = 10, .ret = TSD_ERR_INVALID_NODE_POSTERIOR },
    { .x = { 1, -1 }, .s = 10, .ret = TSD_ERR_INVALID_NODE_POSTERIOR },
    { .x = { 1, NAN }, .s = 10, .ret = TSD_ERR_INVALID_NODE_POSTERIOR },
    /* a zero shape with a cap below one: a division by zero */
    { .x = { 0, 1 }, .s = 0.5, .ret = TSD_ERR_ZERO_DIVISION },
};

static void
test_rescale(void)
{
    const size_t num_rows = sizeof(rescale_rows) / sizeof(rescale_rows[0]);
    const rescale_row_t *row;
    double d;
    size_t j;
    int ret;

    for (j = 0; j < num_rows; j++) {
        row = rescale_rows + j;
        ret = tsd_rescale(row->x, row->s, &d);
        CU_ASSERT_EQUAL(ret, row->ret);
        if (ret == 0 && row->ret == 0) {
            assert_close_ulps(d, row->d, ULPS);
        }
    }
}

static void
test_ep_init(void)
{
    const tsd_id_t bad[4] = { 2, 1, 3, 4 };
    tsd_ep_t ep;

    CU_ASSERT_EQUAL(tsd_ep_init(&ep, 4, 4, 4, ep_constraints, ep_parent, ep_child,
                        ep_likelihoods, ep_parent, ep_child, ep_likelihoods),
        0);
    CU_ASSERT_EQUAL(ep.scale[3], 1.0);
    CU_ASSERT_EQUAL(ep.posterior[7], 0.0);
    CU_ASSERT_EQUAL(ep.edges_parent[3], 1);
    tsd_ep_free(&ep);
    CU_ASSERT_EQUAL(tsd_ep_init(&ep, 4, 4, 4, ep_constraints, bad, ep_child,
                        ep_likelihoods, ep_parent, ep_child, ep_likelihoods),
        TSD_ERR_NODE_OUT_OF_BOUNDS);
    tsd_ep_free(&ep);
    CU_ASSERT_EQUAL(tsd_ep_init(&ep, 4, 4, 4, ep_constraints, ep_parent, ep_child,
                        ep_likelihoods, ep_parent, bad, ep_likelihoods),
        TSD_ERR_NODE_OUT_OF_BOUNDS);
    tsd_ep_free(&ep);
    CU_ASSERT_EQUAL(
        tsd_ep_init(&ep, 0, 0, 0, NULL, NULL, NULL, NULL, NULL, NULL, NULL), 0);
    tsd_ep_free(&ep);
}

static void
test_rescale_and_assemble_factors(void)
{
    const double zero[8] = { 0 };
    tsd_ep_t ep;
    double posterior[8];
    int j;

    ep_init(&ep, ep_parent, ep_child, ep_likelihoods, zero);
    for (j = 0; j < 16; j++) {
        ep.node[j] = 1.0 + j;
        ep.edge[j] = 1.0 + j;
        ep.block[j] = 0.5 * (1.0 + j);
    }
    for (j = 0; j < 4; j++) {
        ep.scale[j] = 2.0;
    }
    tsd_ep_assemble_factors(&ep, posterior);
    CU_ASSERT_TRUE(posterior[0] > 0);
    tsd_ep_rescale_factors(&ep);
    CU_ASSERT_EQUAL(ep.scale[3], 1.0);
    CU_ASSERT_EQUAL(ep.node[0], 2.0);
    CU_ASSERT_EQUAL(ep.edge[0], 2.0);
    CU_ASSERT_EQUAL(ep.block[0], 1.0);
    tsd_ep_free(&ep);
}

typedef struct {
    tsd_size_t num_nodes;
    double constraints[3][2];
    tsd_size_t num_edges;
    tsd_id_t edges_parent[2];
    tsd_id_t edges_child[2];
    int consistent;
} consistency_row_t;

/* Expected values of the reference implementation */
static const consistency_row_t consistency_rows[] = {
    { .num_nodes = 0, .num_edges = 0, .consistent = 1 },
    /* no edges */
    { .num_nodes = 2,
        .constraints = { { 0, 1 }, { 0, 1 } },
        .num_edges = 0,
        .consistent = 1 },
    { .num_nodes = 2,
        .constraints = { { 0, 0 }, { 0, INFINITY } },
        .num_edges = 1,
        .edges_parent = { 1 },
        .edges_child = { 0 },
        .consistent = 1 },
    { .num_nodes = 2,
        .constraints = { { 0, 5 }, { 3, 9 } },
        .num_edges = 1,
        .edges_parent = { 1 },
        .edges_child = { 0 },
        .consistent = 1 },
    /* the parent's upper bound equals the child's lower bound */
    { .num_nodes = 2,
        .constraints = { { 5, 5 }, { 0, 5 } },
        .num_edges = 1,
        .edges_parent = { 1 },
        .edges_child = { 0 },
        .consistent = 1 },
    { .num_nodes = 2,
        .constraints = { { 5.000000000000001, 6 }, { 0, 5 } },
        .num_edges = 1,
        .edges_parent = { 1 },
        .edges_child = { 0 },
        .consistent = 0 },
    { .num_nodes = 2,
        .constraints = { { 5, 5 }, { 0, 4.999999999999999 } },
        .num_edges = 1,
        .edges_parent = { 1 },
        .edges_child = { 0 },
        .consistent = 0 },
    /* the maximum lower bound propagates up a chain */
    { .num_nodes = 3,
        .constraints = { { 5, 5 }, { 0, 9 }, { 0, 9 } },
        .num_edges = 2,
        .edges_parent = { 1, 2 },
        .edges_child = { 0, 1 },
        .consistent = 1 },
    { .num_nodes = 3,
        .constraints = { { 5, 5 }, { 0, INFINITY }, { 0, 4 } },
        .num_edges = 2,
        .edges_parent = { 1, 2 },
        .edges_child = { 0, 1 },
        .consistent = 0 },
    /* and from every child */
    { .num_nodes = 3,
        .constraints = { { 0, 0 }, { 7, 7 }, { 0, 6 } },
        .num_edges = 2,
        .edges_parent = { 2, 2 },
        .edges_child = { 0, 1 },
        .consistent = 0 },
    { .num_nodes = 2,
        .constraints = { { 10, 10 }, { 0, 0 } },
        .num_edges = 1,
        .edges_parent = { 1 },
        .edges_child = { 0 },
        .consistent = 0 },
    /* NOTE: current behaviour, possibly a bug: the check is a single pass in edge order,
       so edges must be sorted children-first */
    { .num_nodes = 3,
        .constraints = { { 5, 5 }, { 0, INFINITY }, { 0, 4 } },
        .num_edges = 2,
        .edges_parent = { 2, 1 },
        .edges_child = { 1, 0 },
        .consistent = 1 },
    /* NaN bounds are never inconsistent */
    { .num_nodes = 2,
        .constraints = { { NAN, NAN }, { 0, 5 } },
        .num_edges = 1,
        .edges_parent = { 1 },
        .edges_child = { 0 },
        .consistent = 1 },
    { .num_nodes = 2,
        .constraints = { { 5, 5 }, { 0, NAN } },
        .num_edges = 1,
        .edges_parent = { 1 },
        .edges_child = { 0 },
        .consistent = 1 },
};

static void
test_constraints_are_consistent(void)
{
    const size_t num_rows = sizeof(consistency_rows) / sizeof(consistency_rows[0]);
    const consistency_row_t *row;
    int consistent;
    size_t j;
    int ret;

    for (j = 0; j < num_rows; j++) {
        row = consistency_rows + j;
        consistent = -1;
        ret = tsd_constraints_are_consistent(row->num_nodes, row->num_edges,
            &row->constraints[0][0], row->edges_parent, row->edges_child, &consistent);
        CU_ASSERT_EQUAL(ret, 0);
        CU_ASSERT_EQUAL(consistent, row->consistent);
    }
}

/* The branches of an EP pass: fixed child, fixed parent, free, both fixed. */
static void
run_likelihood_pass(int unphased)
{
    const tsd_id_t order[5] = { 0, 1, 2, 3, 2 };
    const double post[8] = { 0, 0, 0, 0, 1, 1, 1, 1 };
    double *lognorm;
    tsd_ep_t ep;
    int ret;

    ep_init(&ep, ep_parent, ep_child, ep_likelihoods, post);
    lognorm = unphased ? ep.blocks_lognorm : ep.edges_lognorm;
    ret = tsd_ep_propagate_likelihood(&ep, 5, order, unphased, 1000.0, 0.1);
    CU_ASSERT_EQUAL(ret, 0);
    CU_ASSERT_TRUE(isfinite(lognorm[2]));
    /* a small scale triggers a rescaling of the factors in mid-pass */
    ep.scale[2] = 1e-200;
    ret = tsd_ep_propagate_likelihood(&ep, 5, order, unphased, 1000.0, 0.1);
    CU_ASSERT_EQUAL(ret, 0);
    CU_ASSERT_EQUAL(ep.scale[0], 1.0);
    tsd_ep_free(&ep);
}

static void
test_propagate_likelihood(void)
{
    const tsd_id_t order[1] = { 3 };
    const tsd_id_t sideways[1] = { 0 };
    const tsd_id_t bad[1] = { 4 };
    const double post[8] = { 0, 0, 0, 0, 1, 1, 1, 1 };
    tsd_ep_t ep;

    run_likelihood_pass(0);
    run_likelihood_pass(1);
    /* a block with a single node: the twin update, only in an unphased pass */
    ep_init(&ep, ep_twin_parent, ep_twin_child, ep_likelihoods, post);
    CU_ASSERT_EQUAL(tsd_ep_propagate_likelihood(&ep, 1, order, 1, 1000.0, 0.1), 0);
    CU_ASSERT_EQUAL(tsd_ep_propagate_likelihood(&ep, 1, order, 0, 1000.0, 0.1),
        TSD_ERR_TWIN_EDGE_IN_PHASED_PASS);
    tsd_ep_free(&ep);
    /* the projection of an edge with a huge mutational target divides by zero */
    ep_init(&ep, ep_parent, ep_child, ep_huge, post);
    CU_ASSERT_EQUAL(tsd_ep_propagate_likelihood(&ep, 1, sideways, 1, 1000.0, 0.1),
        TSD_ERR_ZERO_DIVISION);
    tsd_ep_free(&ep);
    /* an improper posterior, bad parameters and ids out of bounds */
    ep_init(&ep, ep_parent, ep_child, ep_likelihoods, post);
    ep.posterior[4] = -5;
    CU_ASSERT_EQUAL(tsd_ep_propagate_likelihood(&ep, 1, sideways, 0, 1000.0, 0.1),
        TSD_ERR_INVALID_NODE_POSTERIOR);
    CU_ASSERT_EQUAL(tsd_ep_propagate_likelihood(&ep, 1, order, 0, 0.5, 0.1),
        TSD_ERR_INVALID_MAX_SHAPE);
    CU_ASSERT_EQUAL(tsd_ep_propagate_likelihood(&ep, 1, order, 0, 1000.0, 1.0),
        TSD_ERR_INVALID_MIN_STEP);
    CU_ASSERT_EQUAL(tsd_ep_propagate_likelihood(&ep, 1, order, 0, 1000.0, 0.0),
        TSD_ERR_INVALID_MIN_STEP);
    CU_ASSERT_EQUAL(tsd_ep_propagate_likelihood(&ep, 1, bad, 0, 1000.0, 0.1),
        TSD_ERR_EDGE_OUT_OF_BOUNDS);
    CU_ASSERT_EQUAL(tsd_ep_propagate_likelihood(&ep, 1, bad, 1, 1000.0, 0.1),
        TSD_ERR_BLOCK_OUT_OF_BOUNDS);
    tsd_ep_free(&ep);
}

static void
test_propagate_prior(void)
{
    const uint8_t free_nodes[4] = { 0, 0, 1, 1 };
    const uint8_t none[4] = { 0, 0, 0, 0 };
    const double post[8] = { 0, 0, 0, 0, 2, 1, 3, 2 };
    const double improper[8] = { 0, 0, 0, 0, 1, 0, 1, 0 };
    tsd_ep_t ep;

    ep_init(&ep, ep_parent, ep_child, ep_likelihoods, post);
    CU_ASSERT_EQUAL(tsd_ep_propagate_prior(&ep, free_nodes, 1000.0, 10, 1e-8), 0);
    CU_ASSERT_TRUE(ep.node[9] != 0);
    CU_ASSERT_EQUAL(
        tsd_ep_propagate_prior(&ep, none, 0.5, 10, 1e-8), TSD_ERR_INVALID_MAX_SHAPE);
    CU_ASSERT_EQUAL(tsd_ep_propagate_prior(&ep, none, 1000.0, 10, 1e-8), 0);
    /* the fit succeeds, but a posterior that is improper cannot be rescaled */
    ep.posterior[4] = -5;
    ep.posterior[6] = 99;
    CU_ASSERT_EQUAL(tsd_ep_propagate_prior(&ep, free_nodes, 1000.0, 10, 1e-8),
        TSD_ERR_INVALID_NODE_POSTERIOR);
    tsd_ep_free(&ep);
    ep_init(&ep, ep_parent, ep_child, ep_likelihoods, improper);
    CU_ASSERT_EQUAL(tsd_ep_propagate_prior(&ep, free_nodes, 1000.0, 10, 1e-8),
        TSD_ERR_NONPOSITIVE_PENALTY);
    tsd_ep_free(&ep);
}

static void
run_mutations_pass(int unphased)
{
    const tsd_id_t edges[6] = { 0, 1, 2, 3, TSD_NULL, 3 };
    const double post[8] = { 0, 0, 0, 0, 1, 1, 1, 1 };
    double mutations_posterior[12];
    double mutations_phase[6];
    tsd_ep_t ep;
    int ret;

    ep_init(&ep, ep_parent, ep_child, ep_likelihoods, post);
    ret = tsd_ep_propagate_mutations(
        &ep, 6, edges, unphased, mutations_posterior, mutations_phase);
    CU_ASSERT_EQUAL(ret, 0);
    CU_ASSERT_TRUE(isfinite(mutations_phase[2]));
    CU_ASSERT_TRUE(isnan(mutations_posterior[8]));
    CU_ASSERT_EQUAL(mutations_phase[4], 1.0);
    tsd_ep_free(&ep);
    ep_init(&ep, ep_twin_parent, ep_twin_child, ep_likelihoods, post);
    ret = tsd_ep_propagate_mutations(
        &ep, 6, edges, unphased, mutations_posterior, mutations_phase);
    CU_ASSERT_EQUAL(ret, 0);
    tsd_ep_free(&ep);
}

static void
test_propagate_mutations(void)
{
    const tsd_id_t edge[1] = { 0 };
    const tsd_id_t bad[1] = { 4 };
    const double post[8] = { 0, 0, 0, 0, 1, 1, 1, 1 };
    double mutations_posterior[2];
    double mutations_phase[1];
    tsd_ep_t ep;

    run_mutations_pass(0);
    run_mutations_pass(1);
    ep_init(&ep, ep_parent, ep_child, ep_huge, post);
    CU_ASSERT_EQUAL(tsd_ep_propagate_mutations(
                        &ep, 1, edge, 1, mutations_posterior, mutations_phase),
        TSD_ERR_ZERO_DIVISION);
    CU_ASSERT_EQUAL(
        tsd_ep_propagate_mutations(&ep, 1, bad, 0, mutations_posterior, mutations_phase),
        TSD_ERR_EDGE_OUT_OF_BOUNDS);
    CU_ASSERT_EQUAL(
        tsd_ep_propagate_mutations(&ep, 1, bad, 1, mutations_posterior, mutations_phase),
        TSD_ERR_BLOCK_OUT_OF_BOUNDS);
    tsd_ep_free(&ep);
}

/*
 * Tables of expectation propagation passes: the inputs of tsd_ep_init, the state
 * that a case sets after it, and the state (or mutations) expected after the pass.
 * Entries of the arrays beyond the sizes of the inputs are unused.
 */
#define EP_MAX_NODES     8
#define EP_MAX_EDGES     8
#define EP_MAX_ORDER     8
#define EP_MAX_MUTATIONS 4

typedef struct {
    tsd_size_t num_nodes;
    tsd_size_t num_edges;
    tsd_size_t num_blocks;
    double constraints[EP_MAX_NODES][2];
    tsd_id_t edges_parent[EP_MAX_EDGES];
    tsd_id_t edges_child[EP_MAX_EDGES];
    double edges_likelihood[EP_MAX_EDGES][2];
    tsd_id_t blocks_left[EP_MAX_EDGES];
    tsd_id_t blocks_right[EP_MAX_EDGES];
    double blocks_likelihood[EP_MAX_EDGES][2];
} ep_inputs_t;

typedef struct {
    double posterior[EP_MAX_NODES][2];
    double node[EP_MAX_NODES][2][2];
    double edge[EP_MAX_EDGES][2][2];
    double block[EP_MAX_EDGES][2][2];
    double scale[EP_MAX_NODES];
    double edges_lognorm[EP_MAX_EDGES];
    double blocks_lognorm[EP_MAX_EDGES];
} ep_state_t;

/* The expected state `after` is only checked if `ret` is 0. */
typedef struct {
    const ep_inputs_t *inputs;
    const ep_state_t *before;
    tsd_size_t num_order;
    tsd_id_t order[EP_MAX_ORDER];
    int unphased;
    double max_shape;
    double min_step;
    int ret;
    ep_state_t after;
} ep_likelihood_row_t;

typedef struct {
    const ep_inputs_t *inputs;
    const ep_state_t *before;
    ep_state_t after;
} ep_rescale_row_t;

typedef struct {
    const ep_inputs_t *inputs;
    const ep_state_t *before;
    uint8_t free[EP_MAX_NODES];
    double max_shape;
    int64_t em_maxitt;
    double em_reltol;
    int ret;
    ep_state_t after;
} ep_prior_row_t;

typedef struct {
    const ep_inputs_t *inputs;
    const ep_state_t *before;
    tsd_size_t num_mutations;
    tsd_id_t mutations_edge[EP_MAX_MUTATIONS];
    int unphased;
    int ret;
    double posterior[EP_MAX_MUTATIONS][2];
    double phase[EP_MAX_MUTATIONS];
} ep_mutations_row_t;

static void
ep_setup(tsd_ep_t *ep, const ep_inputs_t *in, const ep_state_t *state)
{
    const tsd_size_t n = in->num_nodes;
    const tsd_size_t e = in->num_edges;
    const tsd_size_t b = in->num_blocks;
    int ret;

    ret = tsd_ep_init(ep, n, e, b, &in->constraints[0][0], in->edges_parent,
        in->edges_child, &in->edges_likelihood[0][0], in->blocks_left, in->blocks_right,
        &in->blocks_likelihood[0][0]);
    CU_ASSERT_FATAL(ret == 0);
    memcpy(ep->posterior, state->posterior, 2 * n * sizeof(double));
    memcpy(ep->node, state->node, 4 * n * sizeof(double));
    memcpy(ep->edge, state->edge, 4 * e * sizeof(double));
    memcpy(ep->block, state->block, 4 * b * sizeof(double));
    memcpy(ep->scale, state->scale, n * sizeof(double));
    memcpy(ep->edges_lognorm, state->edges_lognorm, e * sizeof(double));
    memcpy(ep->blocks_lognorm, state->blocks_lognorm, b * sizeof(double));
}

static void
assert_ep_state(const tsd_ep_t *ep, const ep_state_t *expected)
{
    const tsd_size_t n = ep->num_nodes;
    const tsd_size_t e = ep->num_edges;
    const tsd_size_t b = ep->num_blocks;

    assert_array_close(ep->posterior, &expected->posterior[0][0], 2 * n);
    assert_array_close(ep->node, &expected->node[0][0][0], 4 * n);
    assert_array_close(ep->edge, &expected->edge[0][0][0], 4 * e);
    assert_array_close(ep->block, &expected->block[0][0][0], 4 * b);
    assert_array_close(ep->scale, expected->scale, n);
    assert_array_close(ep->edges_lognorm, expected->edges_lognorm, e);
    assert_array_close(ep->blocks_lognorm, expected->blocks_lognorm, b);
}

static const ep_inputs_t ep_inputs_fixed_child = { .num_nodes = 2,
    .num_edges = 1,
    .num_blocks = 1,
    .constraints = { { 0, 0 }, { 0, INFINITY } },
    .edges_parent = { 1 },
    .edges_child = { 0 },
    .edges_likelihood = { { 3, 0.1 } },
    .blocks_left = { 1 },
    .blocks_right = { 0 },
    .blocks_likelihood = { { 3, 0.1 } } };

static const ep_state_t ep_state_edge2_valid = { .posterior = { { 4, 0.6 }, { 4, 0.6 } },
    .edge = { { { 1, 0.1 }, { 1, 0.1 } } },
    .scale = { 1, 1 } };

static const ep_state_t ep_state_edge2_damped
    = { .posterior = { { 23, 2.4499999999999997 }, { 23, 2.4499999999999997 } },
          .edge = { { { 22, 2.4 }, { 22, 2.4 } } },
          .scale = { 1, 1 } };

static const ep_inputs_t ep_inputs_fixed_child_huge_span = { .num_nodes = 2,
    .num_edges = 1,
    .num_blocks = 1,
    .constraints = { { 0, 0 }, { 0, INFINITY } },
    .edges_parent = { 1 },
    .edges_child = { 0 },
    .edges_likelihood = { { 1, 1e+300 } },
    .blocks_left = { 1 },
    .blocks_right = { 0 },
    .blocks_likelihood = { { 1, 1e+300 } } };

static const ep_inputs_t ep_inputs_fixed_parent = { .num_nodes = 2,
    .num_edges = 1,
    .num_blocks = 1,
    .constraints = { { 0, INFINITY }, { 10, 10 } },
    .edges_parent = { 1 },
    .edges_child = { 0 },
    .edges_likelihood = { { 3, 0.1 } },
    .blocks_left = { 1 },
    .blocks_right = { 0 },
    .blocks_likelihood = { { 3, 0.1 } } };

static const ep_inputs_t ep_inputs_fixed_parent_huge_span = { .num_nodes = 2,
    .num_edges = 1,
    .num_blocks = 1,
    .constraints = { { 0, INFINITY }, { 10, 10 } },
    .edges_parent = { 1 },
    .edges_child = { 0 },
    .edges_likelihood = { { 1, 1e+300 } },
    .blocks_left = { 1 },
    .blocks_right = { 0 },
    .blocks_likelihood = { { 1, 1e+300 } } };

static const ep_inputs_t ep_inputs_both_free = { .num_nodes = 2,
    .num_edges = 1,
    .num_blocks = 1,
    .constraints = { { 0, INFINITY }, { 0, INFINITY } },
    .edges_parent = { 1 },
    .edges_child = { 0 },
    .edges_likelihood = { { 3, 0.1 } },
    .blocks_left = { 1 },
    .blocks_right = { 0 },
    .blocks_likelihood = { { 3, 0.1 } } };

static const ep_inputs_t ep_inputs_both_free_huge_span = { .num_nodes = 2,
    .num_edges = 1,
    .num_blocks = 1,
    .constraints = { { 0, INFINITY }, { 0, INFINITY } },
    .edges_parent = { 1 },
    .edges_child = { 0 },
    .edges_likelihood = { { 1, 1e+300 } },
    .blocks_left = { 1 },
    .blocks_right = { 0 },
    .blocks_likelihood = { { 1, 1e+300 } } };

static const ep_inputs_t ep_inputs_both_fixed = { .num_nodes = 2,
    .num_edges = 1,
    .num_blocks = 1,
    .constraints = { { 0, 0 }, { 10, 10 } },
    .edges_parent = { 1 },
    .edges_child = { 0 },
    .edges_likelihood = { { 3, 0.1 } },
    .blocks_left = { 1 },
    .blocks_right = { 0 },
    .blocks_likelihood = { { 3, 0.1 } } };

static const ep_inputs_t ep_inputs_twin = { .num_nodes = 2,
    .num_edges = 1,
    .num_blocks = 1,
    .constraints = { { 0, INFINITY }, { 0, INFINITY } },
    .edges_parent = { 1 },
    .edges_child = { 1 },
    .edges_likelihood = { { 3, 0.1 } },
    .blocks_left = { 1 },
    .blocks_right = { 1 },
    .blocks_likelihood = { { 3, 0.1 } } };

static const ep_state_t ep_state_block2_valid
    = { .posterior = { { 4, 0.6 }, { 4, 0.6 } },
          .block = { { { 1, 0.1 }, { 1, 0.1 } } },
          .scale = { 1, 1 } };

static const ep_state_t ep_state_block2_damped
    = { .posterior = { { 23, 2.4499999999999997 }, { 23, 2.4499999999999997 } },
          .block = { { { 22, 2.4 }, { 22, 2.4 } } },
          .scale = { 1, 1 } };

static const ep_inputs_t ep_inputs_twin_huge_span = { .num_nodes = 2,
    .num_edges = 1,
    .num_blocks = 1,
    .constraints = { { 0, INFINITY }, { 0, INFINITY } },
    .edges_parent = { 1 },
    .edges_child = { 1 },
    .edges_likelihood = { { 1, 1e+300 } },
    .blocks_left = { 1 },
    .blocks_right = { 1 },
    .blocks_likelihood = { { 1, 1e+300 } } };

static const ep_inputs_t ep_inputs_sideways_fixed_parent = { .num_nodes = 3,
    .num_edges = 1,
    .num_blocks = 1,
    .constraints = { { 0, INFINITY }, { 5, 5 }, { 0, INFINITY } },
    .edges_parent = { 1 },
    .edges_child = { 2 },
    .edges_likelihood = { { 3, 0.1 } },
    .blocks_left = { 1 },
    .blocks_right = { 2 },
    .blocks_likelihood = { { 3, 0.1 } } };

static const ep_state_t ep_state_block3_valid
    = { .posterior = { { 4, 0.6 }, { 4, 0.6 }, { 4, 0.6 } },
          .block = { { { 1, 0.1 }, { 1, 0.1 } } },
          .scale = { 1, 1, 1 } };

static const ep_state_t ep_state_block3_damped
    = { .posterior = { { 23, 2.4499999999999997 }, { 23, 2.4499999999999997 },
            { 23, 2.4499999999999997 } },
          .block = { { { 22, 2.4 }, { 22, 2.4 } } },
          .scale = { 1, 1, 1 } };

static const ep_inputs_t ep_inputs_sideways_fixed_parent_huge_span = { .num_nodes = 3,
    .num_edges = 1,
    .num_blocks = 1,
    .constraints = { { 0, INFINITY }, { 5, 5 }, { 0, INFINITY } },
    .edges_parent = { 1 },
    .edges_child = { 2 },
    .edges_likelihood = { { 1, 1e+300 } },
    .blocks_left = { 1 },
    .blocks_right = { 2 },
    .blocks_likelihood = { { 1, 1e+300 } } };

static const ep_inputs_t ep_inputs_sideways_fixed_child = { .num_nodes = 3,
    .num_edges = 1,
    .num_blocks = 1,
    .constraints = { { 0, INFINITY }, { 0, INFINITY }, { 5, 5 } },
    .edges_parent = { 1 },
    .edges_child = { 2 },
    .edges_likelihood = { { 3, 0.1 } },
    .blocks_left = { 1 },
    .blocks_right = { 2 },
    .blocks_likelihood = { { 3, 0.1 } } };

static const ep_inputs_t ep_inputs_sideways_fixed_child_huge_span = { .num_nodes = 3,
    .num_edges = 1,
    .num_blocks = 1,
    .constraints = { { 0, INFINITY }, { 0, INFINITY }, { 5, 5 } },
    .edges_parent = { 1 },
    .edges_child = { 2 },
    .edges_likelihood = { { 1, 1e+300 } },
    .blocks_left = { 1 },
    .blocks_right = { 2 },
    .blocks_likelihood = { { 1, 1e+300 } } };

static const ep_inputs_t ep_inputs_unphased_both_free = { .num_nodes = 3,
    .num_edges = 1,
    .num_blocks = 1,
    .constraints = { { 0, INFINITY }, { 0, INFINITY }, { 0, INFINITY } },
    .edges_parent = { 1 },
    .edges_child = { 2 },
    .edges_likelihood = { { 3, 0.1 } },
    .blocks_left = { 1 },
    .blocks_right = { 2 },
    .blocks_likelihood = { { 3, 0.1 } } };

static const ep_inputs_t ep_inputs_unphased_both_free_huge_span = { .num_nodes = 3,
    .num_edges = 1,
    .num_blocks = 1,
    .constraints = { { 0, INFINITY }, { 0, INFINITY }, { 0, INFINITY } },
    .edges_parent = { 1 },
    .edges_child = { 2 },
    .edges_likelihood = { { 1, 1e+300 } },
    .blocks_left = { 1 },
    .blocks_right = { 2 },
    .blocks_likelihood = { { 1, 1e+300 } } };

static const ep_inputs_t ep_inputs_unphased_both_fixed = { .num_nodes = 3,
    .num_edges = 1,
    .num_blocks = 1,
    .constraints = { { 0, 0 }, { 5, 5 }, { 5, 5 } },
    .edges_parent = { 1 },
    .edges_child = { 2 },
    .edges_likelihood = { { 3, 0.1 } },
    .blocks_left = { 1 },
    .blocks_right = { 2 },
    .blocks_likelihood = { { 3, 0.1 } } };

static const ep_state_t ep_state_block2_improper_posterior
    = { .posterior = { { 4, 0.6 }, { -2, 0.6 } },
          .block = { { { 1, 0.1 }, { 1, 0.1 } } },
          .scale = { 1, 1 } };

static const ep_state_t ep_state_edge2_improper_parent
    = { .posterior = { { 4, 0.6 }, { -2, 0.6 } },
          .edge = { { { 1, 0.1 }, { 1, 0.1 } } },
          .scale = { 1, 1 } };

static const ep_state_t ep_state_edge2_improper_child
    = { .posterior = { { -2, 0.6 }, { 4, 0.6 } },
          .edge = { { { 1, 0.1 }, { 1, 0.1 } } },
          .scale = { 1, 1 } };

static const ep_state_t ep_state_edge2_infinite_rate
    = { .posterior = { { 4, 0.6 }, { 3, INFINITY } },
          .edge = { { { 1, 0 }, { 1, 0.1 } } },
          .scale = { 1, 1 } };

static const ep_state_t ep_state_edge2_nan_parent_message
    = { .posterior = { { 4, 0.6 }, { 4, 0.6 } },
          .edge = { { { 1, NAN }, { 1, 0.1 } } },
          .scale = { 1, 1 } };

static const ep_inputs_t ep_inputs_both_free_huge_count = { .num_nodes = 2,
    .num_edges = 1,
    .num_blocks = 1,
    .constraints = { { 0, INFINITY }, { 0, INFINITY } },
    .edges_parent = { 1 },
    .edges_child = { 0 },
    .edges_likelihood = { { 1e+300, 0.1 } },
    .blocks_left = { 1 },
    .blocks_right = { 0 },
    .blocks_likelihood = { { 1e+300, 0.1 } } };

static const ep_inputs_t ep_inputs_twin_huge_count = { .num_nodes = 2,
    .num_edges = 1,
    .num_blocks = 1,
    .constraints = { { 0, INFINITY }, { 0, INFINITY } },
    .edges_parent = { 1 },
    .edges_child = { 1 },
    .edges_likelihood = { { 1e+300, 0.1 } },
    .blocks_left = { 1 },
    .blocks_right = { 1 },
    .blocks_likelihood = { { 1e+300, 0.1 } } };

static const ep_inputs_t ep_inputs_twin_negative_span = { .num_nodes = 2,
    .num_edges = 1,
    .num_blocks = 1,
    .constraints = { { 0, INFINITY }, { 0, INFINITY } },
    .edges_parent = { 1 },
    .edges_child = { 1 },
    .edges_likelihood = { { 3, -0.25 } },
    .blocks_left = { 1 },
    .blocks_right = { 1 },
    .blocks_likelihood = { { 3, -0.25 } } };

static const ep_inputs_t ep_inputs_small = { .num_nodes = 8,
    .num_edges = 7,
    .num_blocks = 2,
    .constraints = { { 0, 0 }, { 0, 0 }, { 1, 1 }, { 0, INFINITY }, { 0, INFINITY },
        { 3, 3 }, { 0, INFINITY }, { 0, 0 } },
    .edges_parent = { 5, 5, 6, 3, 3, 4, 4 },
    .edges_child = { 2, 6, 7, 0, 1, 3, 5 },
    .edges_likelihood = { { 2, 0.4 }, { 1, 0.3 }, { 3, 0.5 }, { 1, 0.2 }, { 2, 0.6 },
        { 4, 0.5 }, { 0, 0.3 } },
    .blocks_left = { 3, 3 },
    .blocks_right = { 3, 6 },
    .blocks_likelihood = { { 1, 0.1 }, { 2, 0.3 } } };

static const ep_state_t ep_state_small_parent_below = {
    .posterior = { { 0, 0 }, { 0, 0 }, { 0, 0 }, { 2, 0.8636609034779383 },
        { 2, 0.35355763983709054 }, { 0, 0 }, { 1.9999999999999998, 0.6559293439814701 },
        { 0, 0 } },
    .node = { { { 0, 0 }, { 0, 0 } }, { { 0, 0 }, { 0, 0 } }, { { 0, 0 }, { 0, 0 } },
        { { 6.7039039649712985e+159, 3.3519519824856493e+159 }, { 0, 0 } },
        { { 1, 0.11336363853632214 }, { 0, 0 } }, { { 0, 0 }, { 0, 0 } },
        { { 1, 0.5 }, { 0, 0 } }, { { 0, 0 }, { 0, 0 } } },
    .edge
    = { { { 0, 0 }, { 0, 0 } }, { { 0, 0 }, { 10.011642211321844, 6.605287307126262 } },
        { { 21.44384001098297, 3.5739733351638274 }, { 0, 0 } },
        { { 6.703903964971305e+159, 1.3407807929942595e+159 }, { 0, 0 } },
        { { 4.006486409929041e+160, 1.2019459229787126e+160 }, { 0, 0 } },
        { { 6.9437839783179705, 0.524576155586376 },
            { 6.734058618746918e+159, 1.196199483833472e+160 } },
        { { 6.297584371607302, 1.8796324968024116 }, { 0, 0 } } },
    .block = { { { 6.7039039649712985e+159, 1.3407807929942595e+159 }, { 0, 0 } },
        { { 5.135700831885259e+159, 1.0968339446850352e+159 },
            { 0.2842511293334642, 0.058215267444087004 } } },
    .scale = { 1, 1, 1, 1.4916681462400413e-160, 0.14043594343309676, 1,
        0.06108785244275427, 1 },
    .edges_lognorm = { 0, 0.9159334894862647, 1.8171156304288907, 1.6342858852166113,
        1.9098718845981875, 7.151100745299825, 3.5484528956654957 },
    .blocks_lognorm = { 2.265195274615815, -2.300107063753849 }
};

static const ep_state_t ep_state_small_child_below
    = { .posterior = { { 0, 0 }, { 0, 0 }, { 0, 0 }, { 2, 0.8636609034779383 },
            { 2, 0.35355763983709054 }, { 0, 0 },
            { 1.9999999999999998, 0.6559293439814701 }, { 0, 0 } },
          .node = { { { 0, 0 }, { 0, 0 } }, { { 0, 0 }, { 0, 0 } },
              { { 0, 0 }, { 0, 0 } }, { { 1, 0.5 }, { 0, 0 } },
              { { 1, 0.11336363853632214 }, { 0, 0 } }, { { 0, 0 }, { 0, 0 } },
              { { 6.7039039649712985e+159, 3.3519519824856493e+159 }, { 0, 0 } },
              { { 0, 0 }, { 0, 0 } } },
          .edge = { { { 0, 0 }, { 0, 0 } },
              { { 0, 0 }, { 6.711708791635453e+160, 4.428121176801834e+160 } },
              { { 1.4375744407383892e+161, 2.395957401230648e+160 }, { 0, 0 } },
              { { 1.0000000000000009, 0.19999999999999996 }, { 0, 0 } },
              { { 5.9763481560347715, 1.7929044468104316 }, { 0, 0 } },
              { { 6.9437839783179705, 0.524576155586376 },
                  { 1.0044980736497988, 1.7843326665831694 } },
              { { 6.297584371607302, 1.8796324968024116 }, { 0, 0 } } },
          .block = { { { 1, 0.19999999999999996 }, { 0, 0 } },
              { { 0.7660761339541722, 0.16361122570014786 },
                  { 1.90559227298618e+159, 3.9026956224027946e+158 } } },
          .scale = { 1, 1, 1, 0.18609979046343747, 0.14043594343309676, 1,
              1.4916681462400413e-160, 1 },
          .edges_lognorm
          = { 0, 0.9159334894862647, 1.8171156304288907, 1.6342858852166113,
              1.9098718845981875, 7.151100745299825, 3.5484528956654957 },
          .blocks_lognorm = { 2.265195274615815, -2.300107063753849 } };

static const ep_state_t ep_state_small_child_just_below
    = { .posterior = { { 0, 0 }, { 0, 0 }, { 0, 0 }, { 2, 0.8636609034779383 },
            { 2, 0.35355763983709054 }, { 0, 0 },
            { 1.9999999999999998, 0.6559293439814701 }, { 0, 0 } },
          .node = { { { 0, 0 }, { 0, 0 } }, { { 0, 0 }, { 0, 0 } },
              { { 0, 0 }, { 0, 0 } }, { { 1, 0.5 }, { 0, 0 } },
              { { 1, 0.11336363853632214 }, { 0, 0 } }, { { 0, 0 }, { 0, 0 } },
              { { 6.7039039649713e+153, 3.35195198248565e+153 }, { 0, 0 } },
              { { 0, 0 }, { 0, 0 } } },
          .edge = { { { 0, 0 }, { 0, 0 } },
              { { 0, 0 }, { 6.711708791635454e+154, 4.4281211768018344e+154 } },
              { { 1.4375744407383894e+155, 2.395957401230648e+154 }, { 0, 0 } },
              { { 1.0000000000000009, 0.19999999999999996 }, { 0, 0 } },
              { { 5.9763481560347715, 1.7929044468104316 }, { 0, 0 } },
              { { 6.9437839783179705, 0.524576155586376 },
                  { 1.0044980736497988, 1.7843326665831694 } },
              { { 6.297584371607302, 1.8796324968024116 }, { 0, 0 } } },
          .block = { { { 1, 0.19999999999999996 }, { 0, 0 } },
              { { 0.7660761339541722, 0.16361122570014786 },
                  { 1.9055922729861804e+153, 3.9026956224027947e+152 } } },
          .scale = { 1, 1, 1, 0.18609979046343747, 0.14043594343309676, 1,
              1.4916681462400412e-154, 1 },
          .edges_lognorm
          = { 0, 0.9159334894862647, 1.8171156304288907, 1.6342858852166113,
              1.9098718845981875, 7.151100745299825, 3.5484528956654957 },
          .blocks_lognorm = { 2.265195274615815, -2.300107063753849 } };

static const ep_state_t ep_state_small_child_at
    = { .posterior = { { 0, 0 }, { 0, 0 }, { 0, 0 }, { 2, 0.8636609034779383 },
            { 2, 0.35355763983709054 }, { 0, 0 },
            { 1.9999999999999998, 0.6559293439814701 }, { 0, 0 } },
          .node = { { { 0, 0 }, { 0, 0 } }, { { 0, 0 }, { 0, 0 } },
              { { 0, 0 }, { 0, 0 } }, { { 1, 0.5 }, { 0, 0 } },
              { { 1, 0.11336363853632214 }, { 0, 0 } }, { { 0, 0 }, { 0, 0 } },
              { { 6.703903964971299e+153, 3.3519519824856493e+153 }, { 0, 0 } },
              { { 0, 0 }, { 0, 0 } } },
          .edge = { { { 0, 0 }, { 0, 0 } },
              { { 0, 0 }, { 6.711708791635453e+154, 4.428121176801834e+154 } },
              { { 1.4375744407383891e+155, 2.3959574012306478e+154 }, { 0, 0 } },
              { { 1.0000000000000009, 0.19999999999999996 }, { 0, 0 } },
              { { 5.9763481560347715, 1.7929044468104316 }, { 0, 0 } },
              { { 6.9437839783179705, 0.524576155586376 },
                  { 1.0044980736497988, 1.7843326665831694 } },
              { { 6.297584371607302, 1.8796324968024116 }, { 0, 0 } } },
          .block = { { { 1, 0.19999999999999996 }, { 0, 0 } },
              { { 0.7660761339541722, 0.16361122570014786 },
                  { 1.90559227298618e+153, 3.902695622402794e+152 } } },
          .scale = { 1, 1, 1, 0.18609979046343747, 0.14043594343309676, 1,
              1.4916681462400413e-154, 1 },
          .edges_lognorm
          = { 0, 0.9159334894862647, 1.8171156304288907, 1.6342858852166113,
              1.9098718845981875, 7.151100745299825, 3.5484528956654957 },
          .blocks_lognorm = { 2.265195274615815, -2.300107063753849 } };

static const ep_state_t ep_state_small_parent_above = {
    .posterior = { { 0, 0 }, { 0, 0 }, { 0, 0 }, { 2, 0.8636609034779383 },
        { 2, 0.35355763983709054 }, { 0, 0 }, { 1.9999999999999998, 0.6559293439814701 },
        { 0, 0 } },
    .node = { { { 0, 0 }, { 0, 0 } }, { { 0, 0 }, { 0, 0 } }, { { 0, 0 }, { 0, 0 } },
        { { 3.3519519824856493e+153, 1.6759759912428246e+153 }, { 0, 0 } },
        { { 1, 0.11336363853632214 }, { 0, 0 } }, { { 0, 0 }, { 0, 0 } },
        { { 1, 0.5 }, { 0, 0 } }, { { 0, 0 }, { 0, 0 } } },
    .edge
    = { { { 0, 0 }, { 0, 0 } }, { { 0, 0 }, { 10.011642211321844, 6.605287307126262 } },
        { { 21.44384001098297, 3.5739733351638274 }, { 0, 0 } },
        { { 3.351951982485652e+153, 6.703903964971297e+152 }, { 0, 0 } },
        { { 2.0032432049645207e+154, 6.0097296148935625e+153 }, { 0, 0 } },
        { { 6.9437839783179705, 0.524576155586376 },
            { 3.367029309373459e+153, 5.9809974191673596e+153 } },
        { { 6.297584371607302, 1.8796324968024116 }, { 0, 0 } } },
    .block = { { { 3.3519519824856493e+153, 6.703903964971297e+152 }, { 0, 0 } },
        { { 2.5678504159426292e+153, 5.484169723425176e+152 },
            { 0.2842511293334642, 0.058215267444087004 } } },
    .scale = { 1, 1, 1, 2.983336292480083e-154, 0.14043594343309676, 1,
        0.06108785244275427, 1 },
    .edges_lognorm = { 0, 0.9159334894862647, 1.8171156304288907, 1.6342858852166113,
        1.9098718845981875, 7.151100745299825, 3.5484528956654957 },
    .blocks_lognorm = { 2.265195274615815, -2.300107063753849 }
};

static const ep_state_t ep_state_small_skipped_below = {
    .posterior = { { 0, 0 }, { 0, 0 }, { 0, 0 }, { 2, 0.8636609034779383 },
        { 2, 0.35355763983709054 }, { 0, 0 }, { 1.9999999999999998, 0.6559293439814701 },
        { 0, 0 } },
    .node = { { { 0, 0 }, { 0, 0 } }, { { 0, 0 }, { 0, 0 } }, { { 0, 0 }, { 0, 0 } },
        { { 1, 0.5 }, { 0, 0 } }, { { 1, 0.11336363853632214 }, { 0, 0 } },
        { { 0, 0 }, { 0, 0 } }, { { 1, 0.5 }, { 0, 0 } }, { { 0, 0 }, { 0, 0 } } },
    .edge
    = { { { 0, 0 }, { 0, 0 } }, { { 0, 0 }, { 10.011642211321844, 6.605287307126262 } },
        { { 21.44384001098297, 3.5739733351638274 }, { 0, 0 } },
        { { 1.0000000000000009, 0.19999999999999996 }, { 0, 0 } },
        { { 5.9763481560347715, 1.7929044468104316 }, { 0, 0 } },
        { { 6.9437839783179705, 0.524576155586376 },
            { 1.0044980736497988, 1.7843326665831694 } },
        { { 6.297584371607302, 1.8796324968024116 }, { 0, 0 } } },
    .block = { { { 1, 0.19999999999999996 }, { 0, 0 } },
        { { 0.7660761339541722, 0.16361122570014786 },
            { 0.2842511293334642, 0.058215267444087004 } } },
    .scale = { 1, 1, 1.4916681462400413e-160, 0.18609979046343747, 0.14043594343309676,
        1, 0.06108785244275427, 1 },
    .edges_lognorm = { 0, 0.9159334894862647, 1.8171156304288907, 1.6342858852166113,
        1.9098718845981875, 7.151100745299825, 3.5484528956654957 },
    .blocks_lognorm = { 2.265195274615815, -2.300107063753849 }
};

static const ep_state_t ep_state_small_random_scales = {
    .posterior = { { 0, 0 }, { 0, 0 }, { 0, 0 }, { 2, 0.8636609034779383 },
        { 2, 0.35355763983709054 }, { 0, 0 }, { 1.9999999999999998, 0.6559293439814701 },
        { 0, 0 } },
    .node = { { { 0, 0 }, { 0, 0 } }, { { 0, 0 }, { 0, 0 } }, { { 0, 0 }, { 0, 0 } },
        { { 1, 0.5 }, { 0, 0 } }, { { 1, 0.11336363853632214 }, { 0, 0 } },
        { { 0, 0 }, { 0, 0 } }, { { 1, 0.5 }, { 0, 0 } }, { { 0, 0 }, { 0, 0 } } },
    .edge
    = { { { 0, 0 }, { 0, 0 } }, { { 0, 0 }, { 10.011642211321844, 6.605287307126262 } },
        { { 21.44384001098297, 3.5739733351638274 }, { 0, 0 } },
        { { 1.0000000000000009, 0.19999999999999996 }, { 0, 0 } },
        { { 5.9763481560347715, 1.7929044468104316 }, { 0, 0 } },
        { { 6.9437839783179705, 0.524576155586376 },
            { 1.0044980736497988, 1.7843326665831694 } },
        { { 6.297584371607302, 1.8796324968024116 }, { 0, 0 } } },
    .block = { { { 1, 0.19999999999999996 }, { 0, 0 } },
        { { 0.7660761339541722, 0.16361122570014786 },
            { 0.2842511293334642, 0.058215267444087004 } } },
    .scale
    = { 1.4554425309821815, 0.9046800706458055, 0.5614602859042921, 0.5247914532927936,
        1.7199053588004087, 1.8691333659165825, 1.4099536636507697, 1.5942448414759975 },
    .edges_lognorm = { 0, 0.9159334894862647, 1.8171156304288907, 1.6342858852166113,
        1.9098718845981875, 7.151100745299825, 3.5484528956654957 },
    .blocks_lognorm = { 2.265195274615815, -2.300107063753849 }
};

static const ep_state_t ep_state_small_scales_1em160 = {
    .posterior = { { 0, 0 }, { 0, 0 }, { 0, 0 }, { 2, 0.8636609034779383 },
        { 2, 0.35355763983709054 }, { 0, 0 }, { 1.9999999999999998, 0.6559293439814701 },
        { 0, 0 } },
    .node = { { { 0, 0 }, { 0, 0 } }, { { 0, 0 }, { 0, 0 } }, { { 0, 0 }, { 0, 0 } },
        { { 1, 0.5 }, { 0, 0 } }, { { 1, 0.11336363853632214 }, { 0, 0 } },
        { { 0, 0 }, { 0, 0 } }, { { 1, 0.5 }, { 0, 0 } }, { { 0, 0 }, { 0, 0 } } },
    .edge
    = { { { 0, 0 }, { 0, 0 } }, { { 0, 0 }, { 10.011642211321844, 6.605287307126262 } },
        { { 21.44384001098297, 3.5739733351638274 }, { 0, 0 } },
        { { 1.0000000000000009, 0.19999999999999996 }, { 0, 0 } },
        { { 5.9763481560347715, 1.7929044468104316 }, { 0, 0 } },
        { { 6.9437839783179705, 0.524576155586376 },
            { 1.0044980736497988, 1.7843326665831694 } },
        { { 6.297584371607302, 1.8796324968024116 }, { 0, 0 } } },
    .block = { { { 1, 0.19999999999999996 }, { 0, 0 } },
        { { 0.7660761339541722, 0.16361122570014786 },
            { 0.2842511293334642, 0.058215267444087004 } } },
    .scale = { 1e-160, 1e-160, 1e-160, 1e-160, 1e-160, 1e-160, 1e-160, 1e-160 },
    .edges_lognorm = { 0, 0.9159334894862647, 1.8171156304288907, 1.6342858852166113,
        1.9098718845981875, 7.151100745299825, 3.5484528956654957 },
    .blocks_lognorm = { 2.265195274615815, -2.300107063753849 }
};

static const ep_state_t ep_state_small_scales_1e100 = {
    .posterior = { { 0, 0 }, { 0, 0 }, { 0, 0 }, { 2, 0.8636609034779383 },
        { 2, 0.35355763983709054 }, { 0, 0 }, { 1.9999999999999998, 0.6559293439814701 },
        { 0, 0 } },
    .node = { { { 0, 0 }, { 0, 0 } }, { { 0, 0 }, { 0, 0 } }, { { 0, 0 }, { 0, 0 } },
        { { 1, 0.5 }, { 0, 0 } }, { { 1, 0.11336363853632214 }, { 0, 0 } },
        { { 0, 0 }, { 0, 0 } }, { { 1, 0.5 }, { 0, 0 } }, { { 0, 0 }, { 0, 0 } } },
    .edge
    = { { { 0, 0 }, { 0, 0 } }, { { 0, 0 }, { 10.011642211321844, 6.605287307126262 } },
        { { 21.44384001098297, 3.5739733351638274 }, { 0, 0 } },
        { { 1.0000000000000009, 0.19999999999999996 }, { 0, 0 } },
        { { 5.9763481560347715, 1.7929044468104316 }, { 0, 0 } },
        { { 6.9437839783179705, 0.524576155586376 },
            { 1.0044980736497988, 1.7843326665831694 } },
        { { 6.297584371607302, 1.8796324968024116 }, { 0, 0 } } },
    .block = { { { 1, 0.19999999999999996 }, { 0, 0 } },
        { { 0.7660761339541722, 0.16361122570014786 },
            { 0.2842511293334642, 0.058215267444087004 } } },
    .scale = { 1e+100, 1e+100, 1e+100, 1e+100, 1e+100, 1e+100, 1e+100, 1e+100 },
    .edges_lognorm = { 0, 0.9159334894862647, 1.8171156304288907, 1.6342858852166113,
        1.9098718845981875, 7.151100745299825, 3.5484528956654957 },
    .blocks_lognorm = { 2.265195274615815, -2.300107063753849 }
};

static const ep_state_t ep_state_small = {
    .posterior = { { 0, 0 }, { 0, 0 }, { 0, 0 }, { 2, 0.8636609034779383 },
        { 2, 0.35355763983709054 }, { 0, 0 }, { 1.9999999999999998, 0.6559293439814701 },
        { 0, 0 } },
    .node = { { { 0, 0 }, { 0, 0 } }, { { 0, 0 }, { 0, 0 } }, { { 0, 0 }, { 0, 0 } },
        { { 1, 0.5 }, { 0, 0 } }, { { 1, 0.11336363853632214 }, { 0, 0 } },
        { { 0, 0 }, { 0, 0 } }, { { 1, 0.5 }, { 0, 0 } }, { { 0, 0 }, { 0, 0 } } },
    .edge
    = { { { 0, 0 }, { 0, 0 } }, { { 0, 0 }, { 10.011642211321844, 6.605287307126262 } },
        { { 21.44384001098297, 3.5739733351638274 }, { 0, 0 } },
        { { 1.0000000000000009, 0.19999999999999996 }, { 0, 0 } },
        { { 5.9763481560347715, 1.7929044468104316 }, { 0, 0 } },
        { { 6.9437839783179705, 0.524576155586376 },
            { 1.0044980736497988, 1.7843326665831694 } },
        { { 6.297584371607302, 1.8796324968024116 }, { 0, 0 } } },
    .block = { { { 1, 0.19999999999999996 }, { 0, 0 } },
        { { 0.7660761339541722, 0.16361122570014786 },
            { 0.2842511293334642, 0.058215267444087004 } } },
    .scale
    = { 1, 1, 1, 0.18609979046343747, 0.14043594343309676, 1, 0.06108785244275427, 1 },
    .edges_lognorm = { 0, 0.9159334894862647, 1.8171156304288907, 1.6342858852166113,
        1.9098718845981875, 7.151100745299825, 3.5484528956654957 },
    .blocks_lognorm = { 2.265195274615815, -2.300107063753849 }
};

static const ep_state_t ep_state_small_zero = {
    .edge
    = { { { 0, 0 }, { 0, 0 } }, { { 0, 0 }, { 10.011642211321844, 6.605287307126262 } },
        { { 21.44384001098297, 3.5739733351638274 }, { 0, 0 } },
        { { 1.0000000000000009, 0.19999999999999996 }, { 0, 0 } },
        { { 5.9763481560347715, 1.7929044468104316 }, { 0, 0 } },
        { { 6.9437839783179705, 0.524576155586376 },
            { 1.0044980736497988, 1.7843326665831694 } },
        { { 6.297584371607302, 1.8796324968024116 }, { 0, 0 } } },
    .block = { { { 1, 0.19999999999999996 }, { 0, 0 } },
        { { 0.7660761339541722, 0.16361122570014786 },
            { 0.2842511293334642, 0.058215267444087004 } } },
    .scale
    = { 1, 1, 1, 0.18609979046343747, 0.14043594343309676, 1, 0.06108785244275427, 1 },
    .edges_lognorm = { 0, 0.9159334894862647, 1.8171156304288907, 1.6342858852166113,
        1.9098718845981875, 7.151100745299825, 3.5484528956654957 },
    .blocks_lognorm = { 2.265195274615815, -2.300107063753849 }
};

static const ep_state_t ep_state_small_negative_rate = {
    .posterior = { { 0, 0 }, { 0, 0 }, { 0, 0 }, { 2, 0.8636609034779383 },
        { 2, 0.35355763983709054 }, { 0, 0 }, { 1.9999999999999998, 0.6559293439814701 },
        { 0, 0 } },
    .node = { { { 0, 0 }, { 0, 0 } }, { { 0, 0 }, { 0, 0 } }, { { 0, 0 }, { 0, 0 } },
        { { 1, 0.5 }, { 0, 0 } }, { { 1, 9.638256465887746 }, { 0, 0 } },
        { { 0, 0 }, { 0, 0 } }, { { 1, 0.5 }, { 0, 0 } }, { { 0, 0 }, { 0, 0 } } },
    .edge
    = { { { 0, 0 }, { 0, 0 } }, { { 0, 0 }, { 10.011642211321844, 6.605287307126262 } },
        { { 21.44384001098297, 3.5739733351638274 }, { 0, 0 } },
        { { 1.0000000000000009, 0.19999999999999996 }, { 0, 0 } },
        { { 5.9763481560347715, 1.7929044468104316 }, { 0, 0 } },
        { { 6.9437839783179705, 0.524576155586376 },
            { 1.0044980736497988, 1.7843326665831694 } },
        { { 6.297584371607302, 1.8796324968024116 }, { 0, 0 } } },
    .block = { { { 1, 0.19999999999999996 }, { 0, 0 } },
        { { 0.7660761339541722, 0.16361122570014786 },
            { 0.2842511293334642, 0.058215267444087004 } } },
    .scale
    = { 1, 1, 1, 0.18609979046343747, 0.14043594343309676, 1, 0.06108785244275427, 1 },
    .edges_lognorm = { 0, 0.9159334894862647, 1.8171156304288907, 1.6342858852166113,
        1.9098718845981875, 7.151100745299825, 3.5484528956654957 },
    .blocks_lognorm = { 2.265195274615815, -2.300107063753849 }
};

static const ep_inputs_t ep_inputs_both_fixed_equal = { .num_nodes = 2,
    .num_edges = 1,
    .num_blocks = 1,
    .constraints = { { 5, 5 }, { 5, 5 } },
    .edges_parent = { 1 },
    .edges_child = { 0 },
    .edges_likelihood = { { 3, 0.1 } },
    .blocks_left = { 1 },
    .blocks_right = { 0 },
    .blocks_likelihood = { { 3, 0.1 } } };

static const ep_inputs_t ep_inputs_both_fixed_adjacent = { .num_nodes = 2,
    .num_edges = 1,
    .num_blocks = 1,
    .constraints = { { 5, 5 }, { 5.000000000000001, 5.000000000000001 } },
    .edges_parent = { 1 },
    .edges_child = { 0 },
    .edges_likelihood = { { 3, 0.1 } },
    .blocks_left = { 1 },
    .blocks_right = { 0 },
    .blocks_likelihood = { { 3, 0.1 } } };

static const ep_inputs_t ep_inputs_both_fixed_inverted = { .num_nodes = 2,
    .num_edges = 1,
    .num_blocks = 1,
    .constraints = { { 10, 10 }, { 0, 0 } },
    .edges_parent = { 1 },
    .edges_child = { 0 },
    .edges_likelihood = { { 3, 0.1 } },
    .blocks_left = { 1 },
    .blocks_right = { 0 },
    .blocks_likelihood = { { 3, 0.1 } } };

/* One edge (block) in each branch of the likelihood pass, with valid, damped and failing
 * projections, and failures of the damping; expected values are those of the reference
 * implementation, except for the divisions by zero that the library reports and the
 * reference implementation computes in floating point. */
static const ep_likelihood_row_t ep_likelihood_branch_rows[] = {
    /* fixed_child, valid */
    { &ep_inputs_fixed_child, &ep_state_edge2_valid, 1, { 0 }, 0, 1e+300, 0.1, 0,
        { .posterior = { { 4, 0.6 }, { 6.000000000000001, 0.6 } },
            .edge = { { { 3.000000000000001, 0.09999999999999998 }, { 1, 0.1 } } },
            .scale = { 1, 1 },
            .edges_lognorm = { 10.155030578372036 } } },
    /* fixed_child, damped */
    { &ep_inputs_fixed_child, &ep_state_edge2_damped, 1, { 0 }, 0, 1e+300, 0.1, 0,
        { .posterior
            = { { 23, 2.4499999999999997 }, { 5.543750000000004, 0.3368750000000001 } },
            .edge = { { { 4.543750000000002, 0.2868750000000003 }, { 22, 2.4 } } },
            .scale = { 1, 1 },
            .edges_lognorm = { 12.861044274991544 } } },
    /* fixed_child, projection_fails */
    { &ep_inputs_fixed_child_huge_span, &ep_state_edge2_valid, 1, { 0 }, 0, 1e+300, 0.1,
        0,
        { .posterior = { { 4, 0.6 }, { 3, 0.5 } },
            .edge = { { { 0, 0 }, { 1, 0.1 } } },
            .scale = { 1, 1 },
            .edges_lognorm = { NAN } } },
    /* fixed_parent, valid */
    { &ep_inputs_fixed_parent, &ep_state_edge2_valid, 1, { 0 }, 0, 1e+300, 0.1, 0,
        { .posterior = { { 5.237205312432989, 1.5864148365014894 }, { 4, 0.6 } },
            .edge = { { { 1, 0.1 }, { 2.237205312432989, 1.0864148365014894 } } },
            .scale = { 1, 1 },
            .edges_lognorm = { 8.392109396313824 } } },
    /* fixed_parent, damped */
    { &ep_inputs_fixed_parent, &ep_state_edge2_damped, 1, { 0 }, 0, 1e+300, 0.1, 0,
        { .posterior
            = { { 6.2859049125683475, 1.59224630406714 }, { 23, 2.4499999999999997 } },
            .edge = { { { 22, 2.4 }, { 5.285904912568346, 1.5422463040671401 } } },
            .scale = { 1, 1 },
            .edges_lognorm = { 8.819556344925992 } } },
    /* fixed_parent, projection_fails */
    { &ep_inputs_fixed_parent_huge_span, &ep_state_edge2_valid, 1, { 0 }, 0, 1e+300, 0.1,
        0,
        { .posterior = { { 3, 0.5 }, { 4, 0.6 } },
            .edge = { { { 1, 0.1 }, { 0, 0 } } },
            .scale = { 1, 1 },
            .edges_lognorm = { NAN } } },
    /* both_free, valid */
    { &ep_inputs_both_free, &ep_state_edge2_valid, 1, { 0 }, 0, 1e+300, 0.1, 0,
        { .posterior = { { 3.340990263647014, 0.8598058362787732 },
              { 9.208111413904785, 0.6820200768549769 } },
            .edge = { { { 6.208111413904785, 0.1820200768549769 },
                { 0.3409902636470141, 0.35980583627877316 } } },
            .scale = { 1, 1 },
            .edges_lognorm = { 13.088415737608075 } } },
    /* both_free, damped */
    { &ep_inputs_both_free, &ep_state_edge2_damped, 1, { 0 }, 0, 1e+300, 0.1, 0,
        { .posterior = { { 3.191895535259021, 0.4211261303374999 },
              { 8.897091337020257, 0.3785701047221309 } },
            .edge = { { { 7.897091337020255, 0.3285701047221311 },
                { 2.191895535259019, 0.3711261303375002 } } },
            .scale = { 1, 1 },
            .edges_lognorm = { 18.31439947231522 } } },
    /* both_free, projection_fails */
    { &ep_inputs_both_free_huge_span, &ep_state_edge2_valid, 1, { 0 }, 0, 1e+300, 0.1, 0,
        { .posterior = { { 3, 0.5 }, { 3, 0.5 } },
            .scale = { 1, 1 },
            .edges_lognorm = { NAN } } },
    /* both_fixed, valid */
    { &ep_inputs_both_fixed, &ep_state_edge2_valid, 1, { 0 }, 0, 1e+300, 0.1, 0,
        { .posterior = { { 4, 0.6 }, { 4, 0.6 } },
            .edge = { { { 1, 0.1 }, { 1, 0.1 } } },
            .scale = { 1, 1 } } },
    /* twin, valid */
    { &ep_inputs_twin, &ep_state_block2_valid, 1, { 0 }, 1, 1e+300, 0.1, 0,
        { .posterior = { { 4, 0.6 }, { 5.999999999999999, 0.6999999999999998 } },
            .block = { { { 2.999999999999999, 0.19999999999999984 }, { 1, 0.1 } } },
            .scale = { 1, 1 },
            .blocks_lognorm = { 11.155417361261065 } } },
    /* twin, damped */
    { &ep_inputs_twin, &ep_state_block2_damped, 1, { 0 }, 1, 1e+300, 0.1, 0,
        { .posterior
            = { { 23, 2.4499999999999997 }, { 5.543750000000005, 0.4287500000000001 } },
            .block = { { { 4.543750000000003, 0.37875000000000036 }, { 22, 2.4 } } },
            .scale = { 1, 1 },
            .blocks_lognorm = { 13.193426982114381 } } },
    /* twin, projection_fails */
    { &ep_inputs_twin_huge_span, &ep_state_block2_valid, 1, { 0 }, 1, 1e+300, 0.1, 0,
        { .posterior = { { 4, 0.6 }, { 3, 0.5 } },
            .block = { { { 0, 0 }, { 1, 0.1 } } },
            .scale = { 1, 1 },
            .blocks_lognorm = { NAN } } },
    /* sideways_fixed_parent, valid */
    { &ep_inputs_sideways_fixed_parent, &ep_state_block3_valid, 1, { 0 }, 1, 1e+300, 0.1,
        0,
        { .posterior
            = { { 4, 0.6 }, { 4, 0.6 }, { 4.231122949265319, 0.5314156646872705 } },
            .block = { { { 1, 0.1 }, { 1.2311229492653188, 0.03141566468727053 } } },
            .scale = { 1, 1, 1 },
            .blocks_lognorm = { 10.937231963805417 } } },
    /* sideways_fixed_parent, damped */
    { &ep_inputs_sideways_fixed_parent, &ep_state_block3_damped, 1, { 0 }, 1, 1e+300,
        0.1, 0,
        { .posterior = { { 23, 2.4499999999999997 }, { 23, 2.4499999999999997 },
              { 4.3632947068956724, 0.308279227260758 } },
            .block = { { { 22, 2.4 }, { 3.3632947068956707, 0.25827922726075825 } } },
            .scale = { 1, 1, 1 },
            .blocks_lognorm = { 13.159832686323291 } } },
    /* sideways_fixed_parent, projection_fails */
    { &ep_inputs_sideways_fixed_parent_huge_span, &ep_state_block3_valid, 1, { 0 }, 1,
        1e+300, 0.1, TSD_ERR_ZERO_DIVISION, { .scale = { 0 } } },
    /* sideways_fixed_child, valid */
    { &ep_inputs_sideways_fixed_child, &ep_state_block3_valid, 1, { 0 }, 1, 1e+300, 0.1,
        0,
        { .posterior
            = { { 4, 0.6 }, { 4.231122949265319, 0.5314156646872705 }, { 4, 0.6 } },
            .block = { { { 1.2311229492653188, 0.03141566468727053 }, { 1, 0.1 } } },
            .scale = { 1, 1, 1 },
            .blocks_lognorm = { 10.937231963805417 } } },
    /* sideways_fixed_child, damped */
    { &ep_inputs_sideways_fixed_child, &ep_state_block3_damped, 1, { 0 }, 1, 1e+300, 0.1,
        0,
        { .posterior = { { 23, 2.4499999999999997 },
              { 4.3632947068956724, 0.308279227260758 }, { 23, 2.4499999999999997 } },
            .block = { { { 3.3632947068956707, 0.25827922726075825 }, { 22, 2.4 } } },
            .scale = { 1, 1, 1 },
            .blocks_lognorm = { 13.159832686323291 } } },
    /* sideways_fixed_child, projection_fails */
    { &ep_inputs_sideways_fixed_child_huge_span, &ep_state_block3_valid, 1, { 0 }, 1,
        1e+300, 0.1, TSD_ERR_ZERO_DIVISION, { .scale = { 0 } } },
    /* unphased_both_free, valid */
    { &ep_inputs_unphased_both_free, &ep_state_block3_valid, 1, { 0 }, 1, 1e+300, 0.1, 0,
        { .posterior = { { 4, 0.6 }, { 3.7142857142857224, 0.5142857142857151 },
              { 3.7142857142857224, 0.5142857142857151 } },
            .block = { { { 0.7142857142857224, 0.014285714285715123 },
                { 0.7142857142857224, 0.014285714285715123 } } },
            .scale = { 1, 1, 1 },
            .blocks_lognorm = { 15.781852011892113 } } },
    /* unphased_both_free, damped */
    { &ep_inputs_unphased_both_free, &ep_state_block3_damped, 1, { 0 }, 1, 1e+300, 0.1,
        0,
        { .posterior
            = { { 23, 2.4499999999999997 }, { 3.4503846153846256, 0.2902307692307697 },
                { 3.4503846153846256, 0.2902307692307697 } },
            .block = { { { 2.450384615384624, 0.24023076923076994 },
                { 2.450384615384624, 0.24023076923076994 } } },
            .scale = { 1, 1, 1 },
            .blocks_lognorm = { 20.176250719010007 } } },
    /* unphased_both_free, projection_fails */
    { &ep_inputs_unphased_both_free_huge_span, &ep_state_block3_valid, 1, { 0 }, 1,
        1e+300, 0.1, 0,
        { .posterior = { { 4, 0.6 }, { 3, 0.5 }, { 3, 0.5 } },
            .scale = { 1, 1, 1 },
            .blocks_lognorm = { NAN } } },
    /* unphased_both_fixed, valid */
    { &ep_inputs_unphased_both_fixed, &ep_state_block3_valid, 1, { 0 }, 1, 1e+300, 0.1,
        0,
        { .posterior = { { 4, 0.6 }, { 4, 0.6 }, { 4, 0.6 } },
            .block = { { { 1, 0.1 }, { 1, 0.1 } } },
            .scale = { 1, 1, 1 } } },
    /* twin, in a phased pass */
    { &ep_inputs_twin, &ep_state_block2_valid, 1, { 0 }, 0, 1e+300, 0.1,
        TSD_ERR_TWIN_EDGE_IN_PHASED_PASS, { .scale = { 0 } } },
    /* twin, improper posterior */
    { &ep_inputs_twin, &ep_state_block2_improper_posterior, 1, { 0 }, 1, 1e+300, 0.1,
        TSD_ERR_INVALID_NODE_POSTERIOR, { .scale = { 0 } } },
    /* both_free, improper parent posterior */
    { &ep_inputs_both_free, &ep_state_edge2_improper_parent, 1, { 0 }, 0, 1e+300, 0.1,
        TSD_ERR_INVALID_NODE_POSTERIOR, { .scale = { 0 } } },
    /* both_free, improper child posterior */
    { &ep_inputs_both_free, &ep_state_edge2_improper_child, 1, { 0 }, 0, 1e+300, 0.1,
        TSD_ERR_INVALID_NODE_POSTERIOR, { .scale = { 0 } } },
    /* fixed_child, infinite posterior rate */
    { &ep_inputs_fixed_child, &ep_state_edge2_infinite_rate, 1, { 0 }, 0, 1e+300, 0.1,
        TSD_ERR_ZERO_DIVISION, { .scale = { 0 } } },
    /* both_free, NaN rate of the parent message */
    { &ep_inputs_both_free, &ep_state_edge2_nan_parent_message, 1, { 0 }, 0, 1e+300, 0.1,
        TSD_ERR_INVALID_NODE_POSTERIOR, { .scale = { 0 } } },
    /* both_free, huge mutation count */
    { &ep_inputs_both_free_huge_count, &ep_state_edge2_valid, 1, { 0 }, 0, 1e+300, 0.1,
        TSD_ERR_INVALID_HYPERGEO, { .scale = { 0 } } },
    /* twin, huge mutation count */
    { &ep_inputs_twin_huge_count, &ep_state_block2_valid, 1, { 0 }, 1, 1e+300, 0.1, 0,
        { .posterior = { { 4, 0.6 }, { NAN, 0 } },
            .block = { { { INFINITY, 0.19999999999999996 }, { 1, 0.1 } } },
            .scale = { 1, 0 },
            .blocks_lognorm = { 6.908253500227124e+302 } } },
    /* twin, span that cancels the cavity rate */
    { &ep_inputs_twin_negative_span, &ep_state_block2_valid, 1, { 0 }, 1, 1e+300, 0.1,
        TSD_ERR_ZERO_DIVISION, { .scale = { 0 } } },
};

/* A node scale below sqrt(DBL_MIN) rescales the factors mid-pass (even for a skipped
 * edge); expected values are those of the reference implementation. */
static const ep_likelihood_row_t ep_likelihood_tiny_scale_rows[] = {
    /* parent 3, scale far below sqrt(DBL_MIN) */
    { &ep_inputs_small, &ep_state_small_parent_below, 7, { 6, 0, 3, 1, 2, 4, 5 }, 0,
        1000, 0.1, 0,
        { .posterior
            = { { 0, 0 }, { 0, 0 }, { 0, 0 }, { 0.5843090800906015, 0.4141118212927717 },
                { 4.576142402519787, 0.6250950421834373 }, { 0, 0 },
                { 7.309791285287766, 4.2876962848892894 }, { 0, 0 } },
            .node = { { { 0, 0 }, { 0, 0 } }, { { 0, 0 }, { 0, 0 } },
                { { 0, 0 }, { 0, 0 } }, { { 1, 0.5 }, { 0, 0 } },
                { { 0.14043594343309676, 0.015920329528856966 }, { 0, 0 } },
                { { 0, 0 }, { 0, 0 } },
                { { 0.06108785244275427, 0.030543926221377134 }, { 0, 0 } },
                { { 0, 0 }, { 0, 0 } } },
            .edge = { { { 0, 0 }, { 0, 0 } },
                { { 0, 0 }, { 4.231339141799602, 3.7535961130003725 } },
                { { 3.000000000000001, 0.49999999999999956 }, { 0, 0 } },
                { { 0.9999999999999996, 0.19999999999999984 }, { 0, 0 } },
                { { 4.252444351577177, 1.2757333054731532 }, { 0, 0 } },
                { { 2.587238914201411, 0.12364381294339392 },
                    { 1.3127109581979957, 1.8519547257352813 } },
                { { 1.848467544885279, 0.48553089971118635 }, { 0, 0 } } },
            .block = { { { 1, 0.19999999999999996 }, { 0, 0 } },
                { { 0.7660761339541722, 0.16361122570014786 },
                    { 0.01736429104540892, 0.003556245667539863 } } },
            .scale = { 1, 1, 1, 1, 1, 1, 1, 1 },
            .edges_lognorm
            = { 0, 0.8096806220619701, -2.9410830709207385, 1.132872360686421,
                1.2487332284559316, 6.048304970158144, 2.570276182534273 },
            .blocks_lognorm = { 2.265195274615815, -2.300107063753849 } } },
    /* child 6, scale far below sqrt(DBL_MIN) */
    { &ep_inputs_small, &ep_state_small_child_below, 7, { 6, 0, 3, 1, 2, 4, 5 }, 0, 1000,
        0.1, 0,
        { .posterior
            = { { 0, 0 }, { 0, 0 }, { 0, 0 }, { 3.7958421321682865, 1.333492106759158 },
                { 10.065420818105734, 1.1253981016342018 }, { 0, 0 },
                { 0.771929388179595, 2.9935415583239657 }, { 0, 0 } },
            .node
            = { { { 0, 0 }, { 0, 0 } }, { { 0, 0 }, { 0, 0 } }, { { 0, 0 }, { 0, 0 } },
                { { 0.18609979046343747, 0.09304989523171874 }, { 0, 0 } },
                { { 0.14043594343309676, 0.015920329528856966 }, { 0, 0 } },
                { { 0, 0 }, { 0, 0 } }, { { 1, 0.5 }, { 0, 0 } },
                { { 0, 0 }, { 0, 0 } } },
            .edge = { { { 0, 0 }, { 0, 0 } },
                { { 0, 0 }, { 14.85515137621622, 9.954829484254553 } },
                { { 15.37226023426819, 2.562043372378031 }, { 0, 0 } },
                { { 0.9999999999999999, 0.19999999999999973 }, { 0, 0 } },
                { { 1.9999999999999993, 0.5999999999999999 }, { 0, 0 } },
                { { 8.076517329787357, 0.6239468723941586 },
                    { 0.28107594323350105, 0.3727742386144888 } },
                { { 1.848467544885279, 0.48553089971118635 }, { 0, 0 } } },
            .block = { { { 0.18609979046343747, 0.03721995809268749 }, { 0, 0 } },
                { { 0.1425666080079117, 0.030448014820263693 },
                    { 0.2842511293334642, 0.05821526744408701 } } },
            .scale = { 1, 1, 1, 1, 1, 1, 1, 1 },
            .edges_lognorm
            = { 0, 1.3769145175226662, -2.0216361082280465, 1.4634663891508815,
                1.5314734140184885, 8.186688217624864, 2.570276182534273 },
            .blocks_lognorm = { 2.265195274615815, -2.300107063753849 } } },
    /* child 6, scale just below sqrt(DBL_MIN) */
    { &ep_inputs_small, &ep_state_small_child_just_below, 7, { 6, 0, 3, 1, 2, 4, 5 }, 0,
        1000, 0.1, 0,
        { .posterior
            = { { 0, 0 }, { 0, 0 }, { 0, 0 }, { 3.7958421321682865, 1.333492106759158 },
                { 10.065420818105734, 1.1253981016342018 }, { 0, 0 },
                { 0.771929388179595, 2.9935415583239657 }, { 0, 0 } },
            .node
            = { { { 0, 0 }, { 0, 0 } }, { { 0, 0 }, { 0, 0 } }, { { 0, 0 }, { 0, 0 } },
                { { 0.18609979046343747, 0.09304989523171874 }, { 0, 0 } },
                { { 0.14043594343309676, 0.015920329528856966 }, { 0, 0 } },
                { { 0, 0 }, { 0, 0 } }, { { 1, 0.5 }, { 0, 0 } },
                { { 0, 0 }, { 0, 0 } } },
            .edge = { { { 0, 0 }, { 0, 0 } },
                { { 0, 0 }, { 14.85515137621622, 9.954829484254553 } },
                { { 15.37226023426819, 2.562043372378031 }, { 0, 0 } },
                { { 0.9999999999999999, 0.19999999999999973 }, { 0, 0 } },
                { { 1.9999999999999993, 0.5999999999999999 }, { 0, 0 } },
                { { 8.076517329787357, 0.6239468723941586 },
                    { 0.28107594323350105, 0.3727742386144888 } },
                { { 1.848467544885279, 0.48553089971118635 }, { 0, 0 } } },
            .block = { { { 0.18609979046343747, 0.03721995809268749 }, { 0, 0 } },
                { { 0.1425666080079117, 0.030448014820263693 },
                    { 0.2842511293334642, 0.058215267444087004 } } },
            .scale = { 1, 1, 1, 1, 1, 1, 1, 1 },
            .edges_lognorm
            = { 0, 1.3769145175226662, -2.0216361082280465, 1.4634663891508815,
                1.5314734140184885, 8.186688217624864, 2.570276182534273 },
            .blocks_lognorm = { 2.265195274615815, -2.300107063753849 } } },
    /* child 6, scale at sqrt(DBL_MIN) */
    { &ep_inputs_small, &ep_state_small_child_at, 7, { 6, 0, 3, 1, 2, 4, 5 }, 0, 1000,
        0.1, 0,
        { .posterior
            = { { 0, 0 }, { 0, 0 }, { 0, 0 }, { 3.7958421321682865, 1.333492106759158 },
                { 10.065420818105734, 1.1253981016342018 }, { 0, 0 },
                { 0.771929388179595, 2.9935415583239657 }, { 0, 0 } },
            .node = { { { 0, 0 }, { 0, 0 } }, { { 0, 0 }, { 0, 0 } },
                { { 0, 0 }, { 0, 0 } }, { { 1, 0.5 }, { 0, 0 } },
                { { 1, 0.11336363853632214 }, { 0, 0 } }, { { 0, 0 }, { 0, 0 } },
                { { 6.703903964971299e+153, 3.3519519824856493e+153 }, { 0, 0 } },
                { { 0, 0 }, { 0, 0 } } },
            .edge = { { { 0, 0 }, { 0, 0 } },
                { { 0, 0 }, { 9.958750821126476e+154, 6.673622085010729e+154 } },
                { { 1.0305415633508114e+155, 1.717569272251352e+154 }, { 0, 0 } },
                { { 5.37346118181937, 1.0746922363638727 }, { 0, 0 } },
                { { 10.746922363638738, 3.224076709091622 }, { 0, 0 } },
                { { 57.51032913902832, 4.442928620274516 },
                    { 1.510350670108483, 2.003087900777227 } },
                { { 13.162353594796642, 3.457312194028815 }, { 0, 0 } } },
            .block = { { { 1, 0.19999999999999996 }, { 0, 0 } },
                { { 0.7660761339541722, 0.16361122570014786 },
                    { 1.90559227298618e+153, 3.902695622402794e+152 } } },
            .scale = { 1, 1, 1, 0.18609979046343747, 0.14043594343309676, 1,
                1.4916681462400413e-154, 1 },
            .edges_lognorm
            = { 0, 1.3769145175226662, -2.0216361082280465, 1.4634663891508815,
                1.5314734140184885, 8.186688217624864, 2.570276182534273 },
            .blocks_lognorm = { 2.265195274615815, -2.300107063753849 } } },
    /* parent 3, scale above sqrt(DBL_MIN) */
    { &ep_inputs_small, &ep_state_small_parent_above, 7, { 6, 0, 3, 1, 2, 4, 5 }, 0,
        1000, 0.1, 0,
        { .posterior
            = { { 0, 0 }, { 0, 0 }, { 0, 0 }, { 0.584309080090585, 0.41411182129276836 },
                { 4.5761424025197845, 0.6250950421834371 }, { 0, 0 },
                { 7.309791285287766, 4.2876962848892894 }, { 0, 0 } },
            .node
            = { { { 0, 0 }, { 0, 0 } }, { { 0, 0 }, { 0, 0 } }, { { 0, 0 }, { 0, 0 } },
                { { 3.3519519824856493e+153, 1.6759759912428246e+153 }, { 0, 0 } },
                { { 1, 0.11336363853632214 }, { 0, 0 } }, { { 0, 0 }, { 0, 0 } },
                { { 1, 0.5 }, { 0, 0 } }, { { 0, 0 }, { 0, 0 } } },
            .edge = { { { 0, 0 }, { 0, 0 } },
                { { 0, 0 }, { 69.26645761143446, 61.445867924689054 } },
                { { 49.10960002745744, 8.184933337909563 }, { 0, 0 } },
                { { 3.3519519824856478e+153, 6.703903964971293e+152 }, { 0, 0 } },
                { { 1.425398927467902e+154, 4.276196782403706e+153 }, { 0, 0 } },
                { { 18.422911193201482, 0.8804285421580647 },
                    { 4.400144098762354e+153, 6.207663314402032e+153 } },
                { { 13.162353594796642, 3.457312194028815 }, { 0, 0 } } },
            .block = { { { 3.3519519824856493e+153, 6.703903964971297e+152 }, { 0, 0 } },
                { { 2.5678504159426292e+153, 5.484169723425176e+152 },
                    { 0.2842511293334642, 0.058215267444087004 } } },
            .scale = { 1, 1, 1, 2.983336292480083e-154, 0.14043594343309676, 1,
                0.06108785244275427, 1 },
            .edges_lognorm
            = { 0, 0.8096806220619701, -2.9410830709207385, 1.132872360686421,
                1.2487332284559312, 6.048304970158142, 2.570276182534273 },
            .blocks_lognorm = { 2.265195274615815, -2.300107063753849 } } },
    /* child of a skipped edge between fixed nodes 2, scale far below sqrt(DBL_MIN) */
    { &ep_inputs_small, &ep_state_small_skipped_below, 7, { 6, 0, 3, 1, 2, 4, 5 }, 0,
        1000, 0.1, 0,
        { .posterior
            = { { 0, 0 }, { 0, 0 }, { 0, 0 }, { 3.7958421321682865, 1.333492106759158 },
                { 10.065420818105734, 1.1253981016342018 }, { 0, 0 },
                { 7.309791285287766, 4.2876962848892894 }, { 0, 0 } },
            .node
            = { { { 0, 0 }, { 0, 0 } }, { { 0, 0 }, { 0, 0 } }, { { 0, 0 }, { 0, 0 } },
                { { 0.18609979046343747, 0.09304989523171874 }, { 0, 0 } },
                { { 0.14043594343309676, 0.015920329528856966 }, { 0, 0 } },
                { { 0, 0 }, { 0, 0 } },
                { { 0.06108785244275427, 0.030543926221377134 }, { 0, 0 } },
                { { 0, 0 }, { 0, 0 } } },
            .edge = { { { 0, 0 }, { 0, 0 } },
                { { 0, 0 }, { 4.231339141799602, 3.7535961130003725 } },
                { { 3.000000000000001, 0.49999999999999956 }, { 0, 0 } },
                { { 0.9999999999999998, 0.19999999999999973 }, { 0, 0 } },
                { { 1.9999999999999993, 0.5999999999999999 }, { 0, 0 } },
                { { 8.076517329787357, 0.6239468723941586 },
                    { 0.28107594323350105, 0.3727742386144888 } },
                { { 1.848467544885279, 0.48553089971118635 }, { 0, 0 } } },
            .block = { { { 0.18609979046343747, 0.03721995809268749 }, { 0, 0 } },
                { { 0.1425666080079117, 0.030448014820263693 },
                    { 0.01736429104540892, 0.003556245667539863 } } },
            .scale = { 1, 1, 1, 1, 1, 1, 1, 1 },
            .edges_lognorm
            = { 0, 0.8096806220619701, -2.9410830709207385, 1.4634663891508815,
                1.5314734140184885, 8.186688217624864, 2.570276182534273 },
            .blocks_lognorm = { 2.265195274615815, -2.300107063753849 } } },
};

/* Rescaling the factors of the small state; expected values are those of the reference
 * implementation. */
static const ep_rescale_row_t ep_rescale_rows[] = {
    /* random scales */
    { &ep_inputs_small, &ep_state_small_random_scales,
        { .posterior = { { 0, 0 }, { 0, 0 }, { 0, 0 }, { 2, 0.8636609034779383 },
              { 2, 0.35355763983709054 }, { 0, 0 },
              { 1.9999999999999998, 0.6559293439814701 }, { 0, 0 } },
            .node
            = { { { 0, 0 }, { 0, 0 } }, { { 0, 0 }, { 0, 0 } }, { { 0, 0 }, { 0, 0 } },
                { { 0.5247914532927936, 0.2623957266463968 }, { 0, 0 } },
                { { 1.7199053588004087, 0.19497472941173297 }, { 0, 0 } },
                { { 0, 0 }, { 0, 0 } },
                { { 1.4099536636507697, 0.7049768318253848 }, { 0, 0 } },
                { { 0, 0 }, { 0, 0 } } },
            .edge = { { { 0, 0 }, { 0, 0 } },
                { { 0, 0 }, { 14.115951615013927, 9.3131490381486 } },
                { { 30.2348207862264, 5.039136797704399 }, { 0, 0 } },
                { { 0.5247914532927941, 0.1049582906585587 }, { 0, 0 } },
                { { 3.1363364341891953, 0.9409009302567586 }, { 0, 0 } },
                { { 11.942651274661499, 0.9022213410919251 },
                    { 0.5271520039004896, 0.9364025332539873 } },
                { { 10.831249108225103, 3.2327900038258597 }, { 0, 0 } } },
            .block = { { { 0.5247914532927936, 0.1049582906585587 }, { 0, 0 } },
                { { 0.4020302076707349, 0.08586177291019587 },
                    { 0.4007809212005866, 0.08208082961319985 } } },
            .scale = { 1, 1, 1, 1, 1, 1, 1, 1 },
            .edges_lognorm
            = { 0, 0.9159334894862647, 1.8171156304288907, 1.6342858852166113,
                1.9098718845981875, 7.151100745299825, 3.5484528956654957 },
            .blocks_lognorm = { 2.265195274615815, -2.300107063753849 } } },
    /* scales 1e-160 */
    { &ep_inputs_small, &ep_state_small_scales_1em160,
        { .posterior = { { 0, 0 }, { 0, 0 }, { 0, 0 }, { 2, 0.8636609034779383 },
              { 2, 0.35355763983709054 }, { 0, 0 },
              { 1.9999999999999998, 0.6559293439814701 }, { 0, 0 } },
            .node = { { { 0, 0 }, { 0, 0 } }, { { 0, 0 }, { 0, 0 } },
                { { 0, 0 }, { 0, 0 } }, { { 1e-160, 5e-161 }, { 0, 0 } },
                { { 1e-160, 1.1336363853632214e-161 }, { 0, 0 } },
                { { 0, 0 }, { 0, 0 } }, { { 1e-160, 5e-161 }, { 0, 0 } },
                { { 0, 0 }, { 0, 0 } } },
            .edge = { { { 0, 0 }, { 0, 0 } },
                { { 0, 0 }, { 1.0011642211321843e-159, 6.605287307126262e-160 } },
                { { 2.1443840010982971e-159, 3.5739733351638275e-160 }, { 0, 0 } },
                { { 1.000000000000001e-160, 1.9999999999999997e-161 }, { 0, 0 } },
                { { 5.976348156034772e-160, 1.7929044468104316e-160 }, { 0, 0 } },
                { { 6.943783978317971e-160, 5.24576155586376e-161 },
                    { 1.0044980736497989e-160, 1.7843326665831694e-160 } },
                { { 6.297584371607302e-160, 1.8796324968024114e-160 }, { 0, 0 } } },
            .block = { { { 1e-160, 1.9999999999999997e-161 }, { 0, 0 } },
                { { 7.660761339541722e-161, 1.6361122570014785e-161 },
                    { 2.842511293334642e-161, 5.8215267444087e-162 } } },
            .scale = { 1, 1, 1, 1, 1, 1, 1, 1 },
            .edges_lognorm
            = { 0, 0.9159334894862647, 1.8171156304288907, 1.6342858852166113,
                1.9098718845981875, 7.151100745299825, 3.5484528956654957 },
            .blocks_lognorm = { 2.265195274615815, -2.300107063753849 } } },
    /* scales 1e100 */
    { &ep_inputs_small, &ep_state_small_scales_1e100,
        { .posterior = { { 0, 0 }, { 0, 0 }, { 0, 0 }, { 2, 0.8636609034779383 },
              { 2, 0.35355763983709054 }, { 0, 0 },
              { 1.9999999999999998, 0.6559293439814701 }, { 0, 0 } },
            .node = { { { 0, 0 }, { 0, 0 } }, { { 0, 0 }, { 0, 0 } },
                { { 0, 0 }, { 0, 0 } }, { { 1e+100, 5e+99 }, { 0, 0 } },
                { { 1e+100, 1.1336363853632215e+99 }, { 0, 0 } }, { { 0, 0 }, { 0, 0 } },
                { { 1e+100, 5e+99 }, { 0, 0 } }, { { 0, 0 }, { 0, 0 } } },
            .edge = { { { 0, 0 }, { 0, 0 } },
                { { 0, 0 }, { 1.0011642211321844e+101, 6.605287307126262e+100 } },
                { { 2.144384001098297e+101, 3.5739733351638276e+100 }, { 0, 0 } },
                { { 1.000000000000001e+100, 1.9999999999999997e+99 }, { 0, 0 } },
                { { 5.9763481560347715e+100, 1.7929044468104318e+100 }, { 0, 0 } },
                { { 6.9437839783179705e+100, 5.24576155586376e+99 },
                    { 1.0044980736497988e+100, 1.7843326665831695e+100 } },
                { { 6.297584371607302e+100, 1.8796324968024115e+100 }, { 0, 0 } } },
            .block = { { { 1e+100, 1.9999999999999997e+99 }, { 0, 0 } },
                { { 7.660761339541722e+99, 1.6361122570014786e+99 },
                    { 2.842511293334642e+99, 5.8215267444087e+98 } } },
            .scale = { 1, 1, 1, 1, 1, 1, 1, 1 },
            .edges_lognorm
            = { 0, 0.9159334894862647, 1.8171156304288907, 1.6342858852166113,
                1.9098718845981875, 7.151100745299825, 3.5484528956654957 },
            .blocks_lognorm = { 2.265195274615815, -2.300107063753849 } } },
};

/* The prior pass on the small state, with and without EM iterations, and its failures;
 * expected values are those of the reference implementation. */
static const ep_prior_row_t ep_prior_rows[] = {
    /* the root */
    { &ep_inputs_small, &ep_state_small, { 0, 0, 0, 0, 1, 0, 0, 0 }, 1000, 10, 1e-08, 0,
        { .posterior = { { 0, 0 }, { 0, 0 }, { 0, 0 }, { 2, 0.8636609034779383 },
              { 2, 0.5192046941817232 }, { 0, 0 },
              { 1.9999999999999998, 0.6559293439814701 }, { 0, 0 } },
            .node = { { { 0, 0 }, { 0, 0 } }, { { 0, 0 }, { 0, 0 } },
                { { 0, 0 }, { 0, 0 } }, { { 1, 0.5 }, { 0, 0 } },
                { { 1.0000000000000007, 1.2928839970373238 }, { 0, 0 } },
                { { 0, 0 }, { 0, 0 } }, { { 1, 0.5 }, { 0, 0 } },
                { { 0, 0 }, { 0, 0 } } },
            .edge = { { { 0, 0 }, { 0, 0 } },
                { { 0, 0 }, { 10.011642211321844, 6.605287307126262 } },
                { { 21.44384001098297, 3.5739733351638274 }, { 0, 0 } },
                { { 1.0000000000000009, 0.19999999999999996 }, { 0, 0 } },
                { { 5.9763481560347715, 1.7929044468104316 }, { 0, 0 } },
                { { 6.9437839783179705, 0.524576155586376 },
                    { 1.0044980736497988, 1.7843326665831694 } },
                { { 6.297584371607302, 1.8796324968024116 }, { 0, 0 } } },
            .block = { { { 1, 0.19999999999999996 }, { 0, 0 } },
                { { 0.7660761339541722, 0.16361122570014786 },
                    { 0.2842511293334642, 0.058215267444087004 } } },
            .scale = { 1, 1, 1, 0.18609979046343747, 0.14043594343309676, 1,
                0.06108785244275427, 1 },
            .edges_lognorm
            = { 0, 0.9159334894862647, 1.8171156304288907, 1.6342858852166113,
                1.9098718845981875, 7.151100745299825, 3.5484528956654957 },
            .blocks_lognorm = { 2.265195274615815, -2.300107063753849 } } },
    /* free internal nodes */
    { &ep_inputs_small, &ep_state_small, { 0, 0, 0, 1, 1, 0, 1, 0 }, 1000, 10, 1e-08, 0,
        { .posterior = { { 0, 0 }, { 0, 0 }, { 0, 0 }, { 2, 1.057316890860762 },
              { 2, 0.624343192922776 }, { 0, 0 },
              { 1.9999999999999998, 0.9120913003746354 }, { 0, 0 } },
            .node
            = { { { 0, 0 }, { 0, 0 } }, { { 0, 0 }, { 0, 0 } }, { { 0, 0 }, { 0, 0 } },
                { { 0.9999999999999996, 1.5406029308285045 }, { 0, 0 } },
                { { 1.0000000000000007, 2.041542041202067 }, { 0, 0 } },
                { { 0, 0 }, { 0, 0 } },
                { { 1.0000000000000002, 4.693337073573113 }, { 0, 0 } },
                { { 0, 0 }, { 0, 0 } } },
            .edge = { { { 0, 0 }, { 0, 0 } },
                { { 0, 0 }, { 10.011642211321844, 6.605287307126262 } },
                { { 21.44384001098297, 3.5739733351638274 }, { 0, 0 } },
                { { 1.0000000000000009, 0.19999999999999996 }, { 0, 0 } },
                { { 5.9763481560347715, 1.7929044468104316 }, { 0, 0 } },
                { { 6.9437839783179705, 0.524576155586376 },
                    { 1.0044980736497988, 1.7843326665831694 } },
                { { 6.297584371607302, 1.8796324968024116 }, { 0, 0 } } },
            .block = { { { 1, 0.19999999999999996 }, { 0, 0 } },
                { { 0.7660761339541722, 0.16361122570014786 },
                    { 0.2842511293334642, 0.058215267444087004 } } },
            .scale = { 1, 1, 1, 0.18609979046343747, 0.14043594343309676, 1,
                0.06108785244275427, 1 },
            .edges_lognorm
            = { 0, 0.9159334894862647, 1.8171156304288907, 1.6342858852166113,
                1.9098718845981875, 7.151100745299825, 3.5484528956654957 },
            .blocks_lognorm = { 2.265195274615815, -2.300107063753849 } } },
    /* no EM iteration */
    { &ep_inputs_small, &ep_state_small, { 0, 0, 0, 1, 1, 0, 1, 0 }, 1000, -1, 1e-08, 0,
        { .posterior = { { 0, 0 }, { 0, 0 }, { 0, 0 }, { 2, 0.9489678090196197 },
              { 2, 0.5159941110816337 }, { 0, 0 },
              { 1.9999999999999998, 0.803742218533493 }, { 0, 0 } },
            .node
            = { { { 0, 0 }, { 0, 0 } }, { { 0, 0 }, { 0, 0 } }, { { 0, 0 }, { 0, 0 } },
                { { 0.9999999999999996, 0.9583933454693566 }, { 0, 0 } },
                { { 1.0000000000000007, 1.270022448764114 }, { 0, 0 } },
                { { 0, 0 }, { 0, 0 } },
                { { 1.0000000000000002, 2.919677049386196 }, { 0, 0 } },
                { { 0, 0 }, { 0, 0 } } },
            .edge = { { { 0, 0 }, { 0, 0 } },
                { { 0, 0 }, { 10.011642211321844, 6.605287307126262 } },
                { { 21.44384001098297, 3.5739733351638274 }, { 0, 0 } },
                { { 1.0000000000000009, 0.19999999999999996 }, { 0, 0 } },
                { { 5.9763481560347715, 1.7929044468104316 }, { 0, 0 } },
                { { 6.9437839783179705, 0.524576155586376 },
                    { 1.0044980736497988, 1.7843326665831694 } },
                { { 6.297584371607302, 1.8796324968024116 }, { 0, 0 } } },
            .block = { { { 1, 0.19999999999999996 }, { 0, 0 } },
                { { 0.7660761339541722, 0.16361122570014786 },
                    { 0.2842511293334642, 0.058215267444087004 } } },
            .scale = { 1, 1, 1, 0.18609979046343747, 0.14043594343309676, 1,
                0.06108785244275427, 1 },
            .edges_lognorm
            = { 0, 0.9159334894862647, 1.8171156304288907, 1.6342858852166113,
                1.9098718845981875, 7.151100745299825, 3.5484528956654957 },
            .blocks_lognorm = { 2.265195274615815, -2.300107063753849 } } },
    /* EM iterations reach the maximum */
    { &ep_inputs_small, &ep_state_small, { 0, 0, 0, 1, 1, 0, 1, 0 }, 1000, 3, 0, 0,
        { .posterior = { { 0, 0 }, { 0, 0 }, { 0, 0 }, { 2, 1.0553334660423712 },
              { 2, 0.6223597681043852 }, { 0, 0 },
              { 1.9999999999999998, 0.9101078755562446 }, { 0, 0 } },
            .node
            = { { { 0, 0 }, { 0, 0 } }, { { 0, 0 }, { 0, 0 } }, { { 0, 0 }, { 0, 0 } },
                { { 0.9999999999999996, 1.5299450745598246 }, { 0, 0 } },
                { { 1.0000000000000007, 2.0274186994855237 }, { 0, 0 } },
                { { 0, 0 }, { 0, 0 } },
                { { 1.0000000000000002, 4.660868673734545 }, { 0, 0 } },
                { { 0, 0 }, { 0, 0 } } },
            .edge = { { { 0, 0 }, { 0, 0 } },
                { { 0, 0 }, { 10.011642211321844, 6.605287307126262 } },
                { { 21.44384001098297, 3.5739733351638274 }, { 0, 0 } },
                { { 1.0000000000000009, 0.19999999999999996 }, { 0, 0 } },
                { { 5.9763481560347715, 1.7929044468104316 }, { 0, 0 } },
                { { 6.9437839783179705, 0.524576155586376 },
                    { 1.0044980736497988, 1.7843326665831694 } },
                { { 6.297584371607302, 1.8796324968024116 }, { 0, 0 } } },
            .block = { { { 1, 0.19999999999999996 }, { 0, 0 } },
                { { 0.7660761339541722, 0.16361122570014786 },
                    { 0.2842511293334642, 0.058215267444087004 } } },
            .scale = { 1, 1, 1, 0.18609979046343747, 0.14043594343309676, 1,
                0.06108785244275427, 1 },
            .edges_lognorm
            = { 0, 0.9159334894862647, 1.8171156304288907, 1.6342858852166113,
                1.9098718845981875, 7.151100745299825, 3.5484528956654957 },
            .blocks_lognorm = { 2.265195274615815, -2.300107063753849 } } },
    /* NaN tolerance: no EM iteration */
    { &ep_inputs_small, &ep_state_small, { 0, 0, 0, 1, 1, 0, 1, 0 }, 1000, 10, NAN, 0,
        { .posterior = { { 0, 0 }, { 0, 0 }, { 0, 0 }, { 2, 0.9489678090196197 },
              { 2, 0.5159941110816337 }, { 0, 0 },
              { 1.9999999999999998, 0.803742218533493 }, { 0, 0 } },
            .node
            = { { { 0, 0 }, { 0, 0 } }, { { 0, 0 }, { 0, 0 } }, { { 0, 0 }, { 0, 0 } },
                { { 0.9999999999999996, 0.9583933454693566 }, { 0, 0 } },
                { { 1.0000000000000007, 1.270022448764114 }, { 0, 0 } },
                { { 0, 0 }, { 0, 0 } },
                { { 1.0000000000000002, 2.919677049386196 }, { 0, 0 } },
                { { 0, 0 }, { 0, 0 } } },
            .edge = { { { 0, 0 }, { 0, 0 } },
                { { 0, 0 }, { 10.011642211321844, 6.605287307126262 } },
                { { 21.44384001098297, 3.5739733351638274 }, { 0, 0 } },
                { { 1.0000000000000009, 0.19999999999999996 }, { 0, 0 } },
                { { 5.9763481560347715, 1.7929044468104316 }, { 0, 0 } },
                { { 6.9437839783179705, 0.524576155586376 },
                    { 1.0044980736497988, 1.7843326665831694 } },
                { { 6.297584371607302, 1.8796324968024116 }, { 0, 0 } } },
            .block = { { { 1, 0.19999999999999996 }, { 0, 0 } },
                { { 0.7660761339541722, 0.16361122570014786 },
                    { 0.2842511293334642, 0.058215267444087004 } } },
            .scale = { 1, 1, 1, 0.18609979046343747, 0.14043594343309676, 1,
                0.06108785244275427, 1 },
            .edges_lognorm
            = { 0, 0.9159334894862647, 1.8171156304288907, 1.6342858852166113,
                1.9098718845981875, 7.151100745299825, 3.5484528956654957 },
            .blocks_lognorm = { 2.265195274615815, -2.300107063753849 } } },
    /* the cap on the shape binds */
    { &ep_inputs_small, &ep_state_small, { 0, 0, 0, 1, 1, 0, 1, 0 }, 1.5, 10, 1e-08, 0,
        { .posterior = { { 0, 0 }, { 0, 0 }, { 0, 0 }, { 0.5, 0.2643292227151905 },
              { 0.5, 0.156085798230694 }, { 0, 0 }, { 0.5, 0.2280228250936589 },
              { 0, 0 } },
            .node
            = { { { 0, 0 }, { 0, 0 } }, { { 0, 0 }, { 0, 0 } }, { { 0, 0 }, { 0, 0 } },
                { { 0.9999999999999996, 1.5406029308285045 }, { 0, 0 } },
                { { 1.0000000000000007, 2.041542041202067 }, { 0, 0 } },
                { { 0, 0 }, { 0, 0 } },
                { { 1.0000000000000002, 4.693337073573113 }, { 0, 0 } },
                { { 0, 0 }, { 0, 0 } } },
            .edge = { { { 0, 0 }, { 0, 0 } },
                { { 0, 0 }, { 10.011642211321844, 6.605287307126262 } },
                { { 21.44384001098297, 3.5739733351638274 }, { 0, 0 } },
                { { 1.0000000000000009, 0.19999999999999996 }, { 0, 0 } },
                { { 5.9763481560347715, 1.7929044468104316 }, { 0, 0 } },
                { { 6.9437839783179705, 0.524576155586376 },
                    { 1.0044980736497988, 1.7843326665831694 } },
                { { 6.297584371607302, 1.8796324968024116 }, { 0, 0 } } },
            .block = { { { 1, 0.19999999999999996 }, { 0, 0 } },
                { { 0.7660761339541722, 0.16361122570014786 },
                    { 0.2842511293334642, 0.058215267444087004 } } },
            .scale = { 1, 1, 1, 0.04652494761585937, 0.03510898585827419, 1,
                0.01527196311068857, 1 },
            .edges_lognorm
            = { 0, 0.9159334894862647, 1.8171156304288907, 1.6342858852166113,
                1.9098718845981875, 7.151100745299825, 3.5484528956654957 },
            .blocks_lognorm = { 2.265195274615815, -2.300107063753849 } } },
    /* no free nodes */
    { &ep_inputs_small, &ep_state_small, { 0, 0, 0, 0, 0, 0, 0, 0 }, 1000, 10, 1e-08, 0,
        { .posterior = { { 0, 0 }, { 0, 0 }, { 0, 0 }, { 2, 0.8636609034779383 },
              { 2, 0.35355763983709054 }, { 0, 0 },
              { 1.9999999999999998, 0.6559293439814701 }, { 0, 0 } },
            .node = { { { 0, 0 }, { 0, 0 } }, { { 0, 0 }, { 0, 0 } },
                { { 0, 0 }, { 0, 0 } }, { { 1, 0.5 }, { 0, 0 } },
                { { 1, 0.11336363853632214 }, { 0, 0 } }, { { 0, 0 }, { 0, 0 } },
                { { 1, 0.5 }, { 0, 0 } }, { { 0, 0 }, { 0, 0 } } },
            .edge = { { { 0, 0 }, { 0, 0 } },
                { { 0, 0 }, { 10.011642211321844, 6.605287307126262 } },
                { { 21.44384001098297, 3.5739733351638274 }, { 0, 0 } },
                { { 1.0000000000000009, 0.19999999999999996 }, { 0, 0 } },
                { { 5.9763481560347715, 1.7929044468104316 }, { 0, 0 } },
                { { 6.9437839783179705, 0.524576155586376 },
                    { 1.0044980736497988, 1.7843326665831694 } },
                { { 6.297584371607302, 1.8796324968024116 }, { 0, 0 } } },
            .block = { { { 1, 0.19999999999999996 }, { 0, 0 } },
                { { 0.7660761339541722, 0.16361122570014786 },
                    { 0.2842511293334642, 0.058215267444087004 } } },
            .scale = { 1, 1, 1, 0.18609979046343747, 0.14043594343309676, 1,
                0.06108785244275427, 1 },
            .edges_lognorm
            = { 0, 0.9159334894862647, 1.8171156304288907, 1.6342858852166113,
                1.9098718845981875, 7.151100745299825, 3.5484528956654957 },
            .blocks_lognorm = { 2.265195274615815, -2.300107063753849 } } },
    /* zero posteriors */
    { &ep_inputs_small, &ep_state_small_zero, { 0, 0, 0, 0, 1, 0, 0, 0 }, 1000, 10,
        1e-08, TSD_ERR_NONPOSITIVE_PENALTY, { .scale = { 0 } } },
    /* negative cavity rate */
    { &ep_inputs_small, &ep_state_small_negative_rate, { 0, 0, 0, 0, 1, 0, 0, 0 }, 1000,
        10, 1e-08, TSD_ERR_NONPOSITIVE_PENALTY, { .scale = { 0 } } },
    /* max_shape below one */
    { &ep_inputs_small, &ep_state_small, { 0, 0, 0, 0, 1, 0, 0, 0 }, 0.5, 10, 1e-08,
        TSD_ERR_INVALID_MAX_SHAPE, { .scale = { 0 } } },
};

/* The mutations above one edge (block) in each branch, with fixed ages, and on the small
 * state; expected values are those of the reference implementation, except for the
 * divisions by zero that the library reports and the reference implementation computes
 * in floating point. */
static const ep_mutations_row_t ep_mutations_rows[] = {
    /* fixed_child, valid */
    { &ep_inputs_fixed_child, &ep_state_edge2_valid, 1, { 0 }, 0, 0,
        { { 0.9090909090909085, 0.32727272727272716 } }, { 1 } },
    /* fixed_child, projection_fails */
    { &ep_inputs_fixed_child_huge_span, &ep_state_edge2_valid, 1, { 0 }, 0, 0,
        { { NAN, NAN } }, { NAN } },
    /* fixed_parent, valid */
    { &ep_inputs_fixed_parent, &ep_state_edge2_valid, 1, { 0 }, 0, 0,
        { { 11.458120369143545, 1.788464840386457 } }, { 1 } },
    /* fixed_parent, projection_fails */
    { &ep_inputs_fixed_parent_huge_span, &ep_state_edge2_valid, 1, { 0 }, 0, 0,
        { { NAN, NAN } }, { NAN } },
    /* both_free, valid */
    { &ep_inputs_both_free, &ep_state_edge2_valid, 1, { 0 }, 0, 0,
        { { 4.300199693233025, 0.529589210511545 } }, { 1 } },
    /* both_free, projection_fails */
    { &ep_inputs_both_free_huge_span, &ep_state_edge2_valid, 1, { 0 }, 0, 0,
        { { NAN, NAN } }, { NAN } },
    /* both_fixed, valid */
    { &ep_inputs_both_fixed, &ep_state_edge2_valid, 1, { 0 }, 0, 0,
        { { 2.0000000000000004, 0.6000000000000001 } }, { 1 } },
    /* twin, valid */
    { &ep_inputs_twin, &ep_state_block2_valid, 1, { 0 }, 1, 0,
        { { 0.9090909090909081, 0.3818181818181816 } }, { 0.5 } },
    /* twin, projection_fails */
    { &ep_inputs_twin_huge_span, &ep_state_block2_valid, 1, { 0 }, 1, 0,
        { { NAN, NAN } }, { NAN } },
    /* sideways_fixed_parent, valid */
    { &ep_inputs_sideways_fixed_parent, &ep_state_block3_valid, 1, { 0 }, 1, 0,
        { { 0.4631781099265906, 0.3438769242905483 } }, { 0.36382649552564394 } },
    /* sideways_fixed_parent, projection_fails */
    { &ep_inputs_sideways_fixed_parent_huge_span, &ep_state_block3_valid, 1, { 0 }, 1,
        TSD_ERR_ZERO_DIVISION, { { 0 } }, { 0 } },
    /* sideways_fixed_child, valid */
    { &ep_inputs_sideways_fixed_child, &ep_state_block3_valid, 1, { 0 }, 1, 0,
        { { 0.4631781099265906, 0.3438769242905483 } }, { 0.36382649552564394 } },
    /* sideways_fixed_child, projection_fails */
    { &ep_inputs_sideways_fixed_child_huge_span, &ep_state_block3_valid, 1, { 0 }, 1,
        TSD_ERR_ZERO_DIVISION, { { 0 } }, { 0 } },
    /* unphased_both_free, valid */
    { &ep_inputs_unphased_both_free, &ep_state_block3_valid, 1, { 0 }, 1, 0,
        { { 0.7515923566878997, 0.3439490445859875 } }, { 0.5 } },
    /* unphased_both_free, projection_fails */
    { &ep_inputs_unphased_both_free_huge_span, &ep_state_block3_valid, 1, { 0 }, 1, 0,
        { { NAN, NAN } }, { NAN } },
    /* unphased_both_fixed, valid */
    { &ep_inputs_unphased_both_fixed, &ep_state_block3_valid, 1, { 0 }, 1, 0,
        { { 1.9999999999999991, 1.1999999999999997 } }, { 0.5 } },
    /* twin, in a phased pass */
    { &ep_inputs_twin, &ep_state_block2_valid, 1, { 0 }, 0, 0,
        { { 1.0000000000000013, 0.40000000000000024 } }, { 0.5 } },
    /* equal fixed ages */
    { &ep_inputs_both_fixed_equal, &ep_state_edge2_valid, 1, { 0 }, 0, 0,
        { { NAN, NAN } }, { NAN } },
    /* adjacent fixed ages */
    { &ep_inputs_both_fixed_adjacent, &ep_state_edge2_valid, 1, { 0 }, 0, 0,
        { { 3.802951800684688e+32, 7.605903601369376e+31 } }, { 1 } },
    /* inverted fixed ages */
    { &ep_inputs_both_fixed_inverted, &ep_state_edge2_valid, 1, { 0 }, 0, 0,
        { { 2.0000000000000004, 0.6000000000000001 } }, { 1 } },
    /* edges of each branch, and a root */
    { &ep_inputs_small, &ep_state_small, 4, { 3, -1, 1, 6 }, 0, 0,
        { { 0.46427524306562473, 0.788165385679627 }, { NAN, NAN },
            { 17.735867699826013, 8.054195276382863 },
            { 3.653465150153581, 0.9407997952233624 } },
        { 1, 1, 1, 1 } },
    /* blocks, and a root */
    { &ep_inputs_small, &ep_state_small, 3, { 1, -1, 0 }, 1, 0,
        { { 0.5085725565286765, 0.6842153617467468 }, { NAN, NAN },
            { 0.4642752430656252, 0.7881653856796274 } },
        { 0.43985052500362354, 1, 0.5 } },
};

static void
run_likelihood_rows(const ep_likelihood_row_t *rows, size_t num_rows)
{
    const ep_likelihood_row_t *row;
    tsd_ep_t ep;
    size_t j;
    int ret;

    for (j = 0; j < num_rows; j++) {
        row = rows + j;
        ep_setup(&ep, row->inputs, row->before);
        ret = tsd_ep_propagate_likelihood(&ep, row->num_order, row->order, row->unphased,
            row->max_shape, row->min_step);
        CU_ASSERT_EQUAL(ret, row->ret);
        if (ret == 0 && row->ret == 0) {
            assert_ep_state(&ep, &row->after);
        }
        tsd_ep_free(&ep);
    }
}

static void
test_propagate_likelihood_branches(void)
{
    run_likelihood_rows(ep_likelihood_branch_rows,
        sizeof(ep_likelihood_branch_rows) / sizeof(ep_likelihood_branch_rows[0]));
}

static void
test_propagate_likelihood_tiny_scale(void)
{
    run_likelihood_rows(
        ep_likelihood_tiny_scale_rows, sizeof(ep_likelihood_tiny_scale_rows)
                                           / sizeof(ep_likelihood_tiny_scale_rows[0]));
}

static void
test_rescale_factors(void)
{
    const size_t num_rows = sizeof(ep_rescale_rows) / sizeof(ep_rescale_rows[0]);
    const ep_rescale_row_t *row;
    tsd_ep_t ep;
    size_t j;

    for (j = 0; j < num_rows; j++) {
        row = ep_rescale_rows + j;
        ep_setup(&ep, row->inputs, row->before);
        tsd_ep_rescale_factors(&ep);
        assert_ep_state(&ep, &row->after);
        tsd_ep_free(&ep);
    }
}

static void
test_propagate_prior_table(void)
{
    const size_t num_rows = sizeof(ep_prior_rows) / sizeof(ep_prior_rows[0]);
    const ep_prior_row_t *row;
    tsd_ep_t ep;
    size_t j;
    int ret;

    for (j = 0; j < num_rows; j++) {
        row = ep_prior_rows + j;
        ep_setup(&ep, row->inputs, row->before);
        ret = tsd_ep_propagate_prior(
            &ep, row->free, row->max_shape, row->em_maxitt, row->em_reltol);
        CU_ASSERT_EQUAL(ret, row->ret);
        if (ret == 0 && row->ret == 0) {
            assert_ep_state(&ep, &row->after);
        }
        tsd_ep_free(&ep);
    }
}

static void
test_propagate_mutations_table(void)
{
    const size_t num_rows = sizeof(ep_mutations_rows) / sizeof(ep_mutations_rows[0]);
    const ep_mutations_row_t *row;
    double posterior[2 * EP_MAX_MUTATIONS];
    double phase[EP_MAX_MUTATIONS];
    tsd_ep_t ep;
    size_t j;
    int ret;

    for (j = 0; j < num_rows; j++) {
        row = ep_mutations_rows + j;
        ep_setup(&ep, row->inputs, row->before);
        ret = tsd_ep_propagate_mutations(&ep, row->num_mutations, row->mutations_edge,
            row->unphased, posterior, phase);
        CU_ASSERT_EQUAL(ret, row->ret);
        if (ret == 0 && row->ret == 0) {
            assert_array_close(posterior, &row->posterior[0][0], 2 * row->num_mutations);
            assert_array_close(phase, row->phase, row->num_mutations);
        }
        tsd_ep_free(&ep);
    }
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
        { "test_digamma", test_digamma },
        { "test_trigamma", test_trigamma },
        { "test_betaln", test_betaln },
        { "test_gammainc_der", test_gammainc_der },
        { "test_hyperu_laplace", test_hyperu_laplace },
        { "test_hyp1f1_laplace", test_hyp1f1_laplace },
        { "test_hyp2f1_laplace", test_hyp2f1_laplace },
        { "test_gammaincinv", test_gammaincinv },
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
        { "test_check_arguments", test_check_arguments },
        { "test_bad_tree_arguments", test_bad_tree_arguments },
        { "test_marginalize_over_ancestors", test_marginalize_over_ancestors },
        { "test_conditional_coalescent_variance", test_conditional_coalescent_variance },
        { "test_fixed_changepoints", test_fixed_changepoints },
        { "test_count_mutations", test_count_mutations },
        { "test_mutational_area", test_mutational_area },
        { "test_mutational_timescale", test_mutational_timescale },
        { "test_piecewise_scale_point_estimate", test_piecewise_scale_point_estimate },
        { "test_piecewise_scale_posterior", test_piecewise_scale_posterior },
        { "test_damp", test_damp },
        { "test_rescale", test_rescale },
        { "test_ep_init", test_ep_init },
        { "test_rescale_and_assemble_factors", test_rescale_and_assemble_factors },
        { "test_constraints_are_consistent", test_constraints_are_consistent },
        { "test_propagate_likelihood", test_propagate_likelihood },
        { "test_propagate_prior", test_propagate_prior },
        { "test_propagate_mutations", test_propagate_mutations },
        { "test_propagate_likelihood_branches", test_propagate_likelihood_branches },
        { "test_propagate_likelihood_tiny_scale", test_propagate_likelihood_tiny_scale },
        { "test_rescale_factors", test_rescale_factors },
        { "test_propagate_prior_table", test_propagate_prior_table },
        { "test_propagate_mutations_table", test_propagate_mutations_table },
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
