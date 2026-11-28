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
 * Moments of node ages under the coalescent, conditional on the number of descendant
 * leaves. Operation order and libm calls match the oracles in tests/oracle, which the
 * tests compare exactly.
 */

#include <math.h>
#include <stdlib.h>

#include "mathutil.h"
#include "tsdate.h"

/*
 * Integrate an expectation over counts of extant ancestors. In a tree with "n" tips,
 * the probability that there are "a" extant ancestors when a subtree of size "k"
 * coalesces is hypergeometric-ish (Wuif & Donnelly 1998), and may be calculated
 * recursively over increasing "a" and decreasing "k" (e.g. using recursive
 * relationships for binomial coefficients).
 *
 * `val` has `num_tips` (at least two) rows and `num_moments` columns; `out` has
 * `num_tips + 1` rows and is written in full.
 */
int
tsd_marginalize_over_ancestors(
    tsd_size_t num_tips, tsd_size_t num_moments, const double *val, double *out)
{
    int ret = 0;
    double *pr_a_ln = NULL; /* log Pr(a | k, n) */
    const tsd_size_t n = num_tips;
    const tsd_size_t N = num_moments;
    tsd_size_t k, a, j;
    double dk, dnk, dnak, weight, constant;

    if (num_tips < 2) {
        ret = tsd_trace_error(TSD_ERR_TOO_FEW_TIPS);
        goto out;
    }
    pr_a_ln = malloc((n + 1) * sizeof(*pr_a_ln));
    if (pr_a_ln == NULL) {
        ret = tsd_trace_error(TSD_ERR_NO_MEMORY);
        goto out;
    }
    for (j = 0; j < (n + 1) * N; j++) {
        out[j] = 0.0;
    }
    pr_a_ln[0] = NAN;
    pr_a_ln[1] = NAN;
    pr_a_ln[2] = 0.0;
    for (k = n - 1; k > 1; k--) {
        dk = (double) k;
        dnk = (double) (n - k);
        /* log(0) at k == 2 is -infinity, and the value is unused then */
        constant = log(dnk) + log((double) (k - 2)) - log(dk + 1);
        for (a = 2; a < n - k + 2; a++) {
            weight = exp(pr_a_ln[a]);
            for (j = 0; j < N; j++) {
                out[k * N + j] += weight * val[a * N + j];
            }
            if (k > 2) { /* Pr(a | k, n) to Pr(a | k - 1, n) */
                dnak = (double) (n - a - k + 2);
                pr_a_ln[a] += constant - log(dnak);
            }
        }
        if (k > 2) { /* Pr(n - k + 1 | k - 1, n) to Pr(n - k + 2 | k - 1, n) */
            pr_a_ln[n - k + 2] = pr_a_ln[n - k + 1] + log((double) (n - k + 2))
                                 - log(dk + 1) - constant;
        }
    }
    for (j = 0; j < N; j++) {
        out[n * N + j] = val[N + j];
    }
out:
    tsd_safe_free(pr_a_ln);
    return ret;
}

/*
 * Variance of node age conditional on the number of descendant leaves, under the
 * standard coalescent. Writes `num_tips + 1` values, indexed by the number of
 * descendant leaves. For a single tip the variances are zero.
 */
int
tsd_conditional_coalescent_variance(tsd_size_t num_tips, double *variance_out)
{
    int ret = 0;
    double *mean = NULL;
    double *variance = NULL;
    double *val = NULL;
    double *moments = NULL;
    tsd_size_t i, n;
    double rate;

    if (num_tips < 1) {
        ret = tsd_trace_error(TSD_ERR_TOO_FEW_TIPS);
        goto out;
    }
    n = num_tips;
    if (n == 1) {
        variance_out[0] = 0.0;
        variance_out[1] = 0.0;
        goto out;
    }
    mean = malloc(n * sizeof(*mean));
    variance = malloc(n * sizeof(*variance));
    val = malloc(2 * n * sizeof(*val));
    moments = malloc(2 * (n + 1) * sizeof(*moments));
    if (mean == NULL || variance == NULL || val == NULL || moments == NULL) {
        ret = tsd_trace_error(TSD_ERR_NO_MEMORY);
        goto out;
    }
    /* coalescence rates, indexed by the number of lineages minus one */
    for (i = 0; i < n; i++) {
        rate = 0.0;
        if (i > 0) {
            rate = 2.0 / (double) ((i + 1) * i);
        }
        mean[i] = rate;
        variance[i] = rate * rate;
    }
    /* hypoexponential mean and variance; e.g. conditional on the number of extant
     * ancestors when the node coalesces, the expected time of coalescence is the sum
     * of exponential RVs (Wuif and Donnelly 1998) */
    for (i = n - 2; i > 0; i--) {
        mean[i] += mean[i + 1];
        variance[i] += variance[i + 1];
    }
    for (i = 0; i < n; i++) {
        val[2 * i] = mean[i];
        val[2 * i + 1] = variance[i] + mean[i] * mean[i];
    }
    ret = tsd_marginalize_over_ancestors(n, 2, val, moments);
    if (ret != 0) {
        goto out;
    }
    for (i = 0; i <= n; i++) {
        variance_out[i] = moments[2 * i + 1] - moments[2 * i] * moments[2 * i];
    }
out:
    tsd_safe_free(mean);
    tsd_safe_free(variance);
    tsd_safe_free(val);
    tsd_safe_free(moments);
    return ret;
}
