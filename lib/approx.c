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
 * Gamma approximations and the moments of the EP updates. Operation order and libm
 * calls match the oracles in tests/oracle, which the tests compare exactly.
 *
 * A float division by zero returns TSD_ERR_ZERO_DIVISION at the divisions that the
 * tests show to be reachable (where the oracle raises ZeroDivisionError). The other
 * divisions follow IEEE semantics (infinity or NaN).
 */

#include <math.h>

#include "mathutil.h"
#include "tsdate.h"

/* The convergence tolerance of the Newton iterations: sqrt(DBL_EPSILON), exactly. */
#define KLMIN_RELTOL 1.4901161193847656e-08
#define KLMIN_MAXITT 100

/* ------------------------------------------------------------------------- *
 * Helpers: gamma approximations and validity checks
 * ------------------------------------------------------------------------- */

/*
 * Approximate log moments via a second-order Taylor series expansion around the mean:
 * E[f(x)] ~ f(mean) + variance * f''(mean) / 2. Returns E[log x], E[x log x] and
 * E[(log x)^2].
 */
int
tsd_approximate_log_moments(
    double mean, double variance, double *logx, double *xlogx, double *logx2)
{
    int ret = 0;
    double log_mean, mean_sq;

    if (!(mean > 0.0) || !(variance > 0.0)) {
        ret = tsd_trace_error(TSD_ERR_NONPOSITIVE_MOMENTS);
        goto out;
    }
    log_mean = log(mean);
    mean_sq = tsd_ipow(mean, 2);
    if (mean_sq == 0.0) {
        ret = tsd_trace_error(TSD_ERR_ZERO_DIVISION);
        goto out;
    }
    *logx = log_mean - 0.5 * variance / mean_sq;
    *xlogx = mean * log_mean + 0.5 * variance / mean;
    *logx2 = tsd_ipow(log_mean, 2) + (1 - log_mean) * variance / mean_sq;
out:
    return ret;
}

/*
 * Newton root finding for the gamma natural parameters matching E[x] and E[log x],
 * minimising KL divergence. The initial condition uses the upper bound
 * digamma(x) <= log(x) - 1/2x.
 */
int
tsd_approximate_gamma_kl(double x, double logx, double out[2])
{
    int ret = 0;
    int itt = 0;
    double log_x, alpha, delta, dg, tg, denom;

    if (x <= 0.0 || isinf(logx)) {
        ret = tsd_trace_error(TSD_ERR_NONPOSITIVE_OR_NONFINITE_MOMENTS);
        goto out;
    }
    log_x = log(x);
    if (!(log_x > logx)) {
        ret = tsd_trace_error(TSD_ERR_JENSEN_VIOLATION);
        goto out;
    }
    alpha = 0.5 / (log_x - logx); /* lower bound on alpha */
    if (alpha == 0.0) {
        ret = tsd_trace_error(TSD_ERR_ZERO_DIVISION);
        goto out;
    }
    /* asymptotically the lower bound becomes sharp */
    if (1.0 / alpha < 1e-4) {
        out[0] = alpha - 1.0;
        out[1] = alpha / x;
        goto out;
    }
    /* converged when the change in alpha falls below the square root of machine
     * precision */
    delta = INFINITY;
    while (fabs(delta) > fabs(alpha) * KLMIN_RELTOL) {
        if (itt > KLMIN_MAXITT) {
            ret = tsd_trace_error(TSD_ERR_KL_MAX_ITERATIONS);
            goto out;
        }
        ret = tsd_digamma(alpha, &dg);
        if (ret != 0) {
            goto out;
        }
        delta = dg - log(alpha) + log_x - logx;
        ret = tsd_trigamma(alpha, &tg);
        if (ret != 0) {
            goto out;
        }
        /* alpha is not zero (the polygamma functions would have failed) */
        denom = tg - 1 / alpha;
        delta /= denom;
        alpha -= delta;
        itt++;
    }
    if (!isfinite(alpha) || alpha <= 0) {
        ret = tsd_trace_error(TSD_ERR_KL_INVALID_SHAPE);
        goto out;
    }
    out[0] = alpha - 1.0;
    out[1] = alpha / x;
out:
    return ret;
}

/*
 * The method of moments: natural parameters of the gamma with the given mean and
 * variance.
 */
int
tsd_approximate_gamma_mom(double mean, double variance, double out[2])
{
    int ret = 0;
    double shape, rate;

    if (!(mean > 0.0 && variance > 0.0)) {
        ret = tsd_trace_error(TSD_ERR_NONPOSITIVE_CENTRAL_MOMENTS);
        goto out;
    }
    shape = tsd_ipow(mean, 2) / variance;
    rate = mean / variance;
    out[0] = shape - 1.0;
    out[1] = rate;
out:
    return ret;
}

/* The gamma with shape `max_shape` whose q-quantile is x. */
static int
gamma_iqr_upper_bound(
    double q, double x, double max_shape, tsd_gammaincinv_fn gammaincinv, double out[2])
{
    int ret = 0;

    if (x == 0.0) {
        ret = tsd_trace_error(TSD_ERR_ZERO_DIVISION);
        goto out;
    }
    out[0] = max_shape - 1;
    out[1] = gammaincinv(max_shape, q) / x;
out:
    return ret;
}

/*
 * One Newton step on the shape of a gamma whose quantiles are in the given ratio.
 * Updates alpha and writes the step to delta.
 */
static int
gamma_iqr_newton_step(double q1, double q2, double x1, double x2,
    tsd_gammaincinv_fn gammaincinv, double *alpha, double *delta)
{
    int ret = 0;
    double y1, y2, obj, inv_1, inv_2, der_1, der_2, gra_1, gra_2, y1_sq, gra;

    y1 = gammaincinv(*alpha, q1);
    y2 = gammaincinv(*alpha, q2);
    if (y1 == 0.0) {
        ret = tsd_trace_error(TSD_ERR_ZERO_DIVISION);
        goto out;
    }
    obj = y2 / y1 - x2 / x1;
    inv_1 = -exp(y1 + log(y1) * (1 - *alpha) + lgamma(*alpha));
    inv_2 = -exp(y2 + log(y2) * (1 - *alpha) + lgamma(*alpha));
    ret = tsd_gammainc_der(*alpha, y1, &der_1);
    if (ret != 0) {
        goto out;
    }
    gra_1 = der_1 * inv_1;
    ret = tsd_gammainc_der(*alpha, y2, &der_2);
    if (ret != 0) {
        goto out;
    }
    gra_2 = der_2 * inv_2;
    y1_sq = tsd_ipow(y1, 2);
    if (y1_sq == 0.0) {
        ret = tsd_trace_error(TSD_ERR_ZERO_DIVISION);
        goto out;
    }
    gra = (gra_2 * y1 - gra_1 * y2) / y1_sq;
    if (gra == 0.0) {
        ret = tsd_trace_error(TSD_ERR_ZERO_DIVISION);
        goto out;
    }
    *delta = -obj / gra;
    *alpha += *delta;
out:
    return ret;
}

/*
 * Natural parameters of a gamma matching two quantiles: q1, q2 are probabilities and
 * x1, x2 the corresponding quantiles. The shape is capped at max_shape.
 */
int
tsd_approximate_gamma_iqr(double q1, double q2, double x1, double x2, double max_shape,
    tsd_gammaincinv_fn gammaincinv, double out[2])
{
    int ret = 0;
    int itt = 0;
    double alpha, delta, ratio_q, ratio_x;

    if (x2 == x1) {
        ret = gamma_iqr_upper_bound(q1, x1, max_shape, gammaincinv, out);
        goto out;
    }
    if (!(q2 > q1 && x2 > x1)) {
        ret = tsd_trace_error(TSD_ERR_UNSORTED_QUANTILES);
        goto out;
    }
    if (q1 == 0.0 || x1 == 0.0) {
        ret = tsd_trace_error(TSD_ERR_ZERO_DIVISION);
        goto out;
    }
    ratio_q = q2 / q1;
    ratio_x = x2 / x1;
    alpha = log(ratio_q) / log(ratio_x); /* lower bound */
    if (alpha > max_shape) {
        ret = gamma_iqr_upper_bound(q1, x1, max_shape, gammaincinv, out);
        goto out;
    }
    /* refine with Newton iteration */
    delta = INFINITY;
    while (fabs(delta) > fabs(alpha) * KLMIN_RELTOL) {
        if (itt > KLMIN_MAXITT) {
            ret = tsd_trace_error(TSD_ERR_QUANTILE_MAX_ITERATIONS);
            goto out;
        }
        ret = gamma_iqr_newton_step(q1, q2, x1, x2, gammaincinv, &alpha, &delta);
        if (ret != 0) {
            goto out;
        }
        itt++;
    }
    if (!(alpha > 0)) {
        ret = tsd_trace_error(TSD_ERR_NEGATIVE_SHAPE);
        goto out;
    }
    if (alpha > max_shape) {
        ret = gamma_iqr_upper_bound(q1, x1, max_shape, gammaincinv, out);
        goto out;
    }
    out[0] = alpha - 1;
    out[1] = gammaincinv(alpha, q1) / x1;
out:
    return ret;
}

int
tsd_average_gammas(
    tsd_size_t num_gammas, const double *alpha, const double *beta, double out[2])
{
    int ret = 0;
    tsd_size_t j;
    double shape, rate, dg;
    double avg_x = 0.0;
    double avg_logx = 0.0;

    if (num_gammas == 0) {
        ret = tsd_trace_error(TSD_ERR_ZERO_DIVISION);
        goto out;
    }
    for (j = 0; j < num_gammas; j++) {
        shape = alpha[j] + 1.0;
        rate = beta[j];
        ret = tsd_digamma(shape, &dg);
        if (ret != 0) {
            goto out;
        }
        avg_logx += dg - log(rate);
        if (rate == 0.0) {
            ret = tsd_trace_error(TSD_ERR_ZERO_DIVISION);
            goto out;
        }
        avg_x += shape / rate;
    }
    avg_x /= (double) num_gammas;
    avg_logx /= (double) num_gammas;
    ret = tsd_approximate_gamma_kl(avg_x, avg_logx, out);
out:
    return ret;
}

int
tsd_valid_moments(double mn, double va)
{
    if (!(isfinite(mn) && isfinite(va))) {
        return 0;
    }
    if (!(mn > 0.0 && va > 0.0)) {
        return 0;
    }
    return 1;
}

int
tsd_valid_gamma(double s, double r)
{
    if (!(isfinite(s) && isfinite(r))) {
        return 0;
    }
    if (s <= 0.0 || r <= 0.0) {
        return 0;
    }
    return 1;
}

int
tsd_valid_hyp1f1(double a, double b, double z)
{
    if (!(isfinite(a) && isfinite(b) && isfinite(z))) {
        return 0;
    }
    if (!(b >= a && a > 0.0)) {
        return 0;
    }
    return 1;
}

int
tsd_valid_hyperu(double a, double b, double z)
{
    if (!(isfinite(a) && isfinite(b) && isfinite(z))) {
        return 0;
    }
    if (z <= 0.0) {
        return 0;
    }
    if (!(b > a && a > 0.0)) {
        return 0;
    }
    return 1;
}

int
tsd_valid_hyp2f1(double a, double b, double c, double z)
{
    if (!(isfinite(a) && isfinite(b) && isfinite(c))) {
        return 0;
    }
    if (!isfinite(z) || z >= 1 || z / (z - 1) >= 1) {
        return 0;
    }
    if (!(a > 0 && b > 0 && c > 0)) {
        return 0;
    }
    return 1;
}

/* ------------------------------------------------------------------------- *
 * Node moments
 * ------------------------------------------------------------------------- */

/* The moments of an edge between two free nodes are unusable: all NaN. */
static void
set_nan_two_node(double *logl, double *mn_i, double *va_i, double *mn_j, double *va_j)
{
    *logl = NAN;
    *mn_i = NAN;
    *va_i = NAN;
    *mn_j = NAN;
    *va_j = NAN;
}

static void
set_nan_one_node(double *logl, double *mn, double *va)
{
    *logl = NAN;
    *mn = NAN;
    *va = NAN;
}

/*
 * log p(t_i, t_j) = log(t_i - t_j) * y_ij - mu_ij * (t_i - t_j)
 *     + log(t_i) * (a_i - 1) - b_i * t_i + log(t_j) * (a_j - 1) - b_j * t_j
 * Writes the normalising constant, E[t_i], V[t_i], E[t_j], V[t_j].
 */
int
tsd_moments(double a_i, double b_i, double a_j, double b_j, double y_ij, double mu_ij,
    double *logl, double *mn_i, double *va_i, double *mn_j, double *va_j)
{
    int ret = 0;
    double a, b, c, t, z, f0, f1, f2, s1, s2, d1, d2;
    double sq_i, sq_j, mn_i_, mn_j_, va_i_, va_j_, logl_;

    a = a_j;
    b = a_i + a_j + y_ij;
    c = a_j + y_ij + 1;
    t = mu_ij + b_i;
    z = t > 0 ? (mu_ij - b_j) / t : NAN;

    if (!tsd_valid_hyp2f1(a, b, c, z)) {
        set_nan_two_node(logl, mn_i, va_i, mn_j, va_j);
        goto out;
    }

    ret = tsd_hyp2f1_laplace(a, b, c, z, &f0);
    if (ret != 0) {
        goto out;
    }
    ret = tsd_hyp2f1_laplace(a + 1, b + 1, c + 1, z, &f1);
    if (ret != 0) {
        goto out;
    }
    ret = tsd_hyp2f1_laplace(a + 2, b + 2, c + 2, z, &f2);
    if (ret != 0) {
        goto out;
    }
    s1 = a * b / c;
    s2 = s1 * (a + 1) * (b + 1) / (c + 1);
    d1 = s1 * exp(f1 - f0);
    d2 = s2 * exp(f2 - f0);

    logl_ = f0 + tsd_betaln(y_ij + 1, a) + lgamma(b) - b * log(t);

    mn_j_ = d1 / t;
    sq_j = d2 / (t * t);
    va_j_ = sq_j - mn_j_ * mn_j_;

    mn_i_ = mn_j_ * z + b / t;
    sq_i = sq_j * (z * z) + (b + 1) * (mn_i_ + mn_j_ * z) / t;
    va_i_ = sq_i - mn_i_ * mn_i_;

    *logl = logl_;
    *mn_i = mn_i_;
    *va_i = va_i_;
    *mn_j = mn_j_;
    *va_j = va_j_;
out:
    return ret;
}

/*
 * log p(t_i) = log(t_i - t_j) * y_ij - mu_ij * (t_i - t_j)
 *     + log(t_i) * (a_i - 1) - b_i * t_i
 * for a fixed child age t_j >= 0. Writes the normalising constant, E[t_i], V[t_i].
 */
int
tsd_rootward_moments(double t_j, double a_i, double b_i, double y_ij, double mu_ij,
    double *logl, double *mn_i, double *va_i)
{
    int ret = 0;
    double s, r, a, b, z, f0, d0, f1, d1;

    if (!(t_j >= 0.0)) {
        ret = tsd_trace_error(TSD_ERR_INVALID_FIXED_AGE);
        goto out;
    }

    s = a_i + y_ij;
    r = mu_ij + b_i;

    if (!tsd_valid_gamma(s, r)) {
        set_nan_one_node(logl, mn_i, va_i);
        goto out;
    }

    if (t_j == 0.0) {
        *logl = lgamma(s) - s * log(r);
        *mn_i = s / r;
        *va_i = s / (r * r);
        goto out;
    }

    a = y_ij + 1;
    b = s + 1;
    z = t_j * r;

    if (!tsd_valid_hyperu(a, b, z)) {
        set_nan_one_node(logl, mn_i, va_i);
        goto out;
    }

    ret = tsd_hyperu_laplace(a, b, z, &f0, &d0);
    if (ret != 0) {
        goto out;
    }
    ret = tsd_hyperu_laplace(a + 1, b + 1, z, &f1, &d1);
    if (ret != 0) {
        goto out;
    }

    *logl = f0 - b_i * t_j + (b - 1) * log(t_j) + lgamma(a);
    *mn_i = t_j * (1 - d0);
    *va_i = (t_j * t_j) * d0 * (d1 - d0);
out:
    return ret;
}

/*
 * log p(t_j) = log(t_i - t_j) * y_ij - mu_ij * (t_i - t_j)
 *     + log(t_j) * (a_j - 1) - b_j * t_j
 * for a fixed parent age t_i > 0. Writes the normalising constant, E[t_j], V[t_j].
 */
int
tsd_leafward_moments(double t_i, double a_j, double b_j, double y_ij, double mu_ij,
    double *logl, double *mn_j, double *va_j)
{
    int ret = 0;
    double a, b, z, f0, f1, f2, d1, d2, sq_j, mn_j_, logl_;

    if (!(t_i > 0.0)) {
        ret = tsd_trace_error(TSD_ERR_INVALID_FIXED_AGE);
        goto out;
    }

    a = a_j;
    b = a_j + y_ij + 1;
    z = t_i * (mu_ij - b_j);

    if (!tsd_valid_hyp1f1(a, b, z)) {
        set_nan_one_node(logl, mn_j, va_j);
        goto out;
    }

    ret = tsd_hyp1f1_laplace(a, b, z, &f0);
    if (ret != 0) {
        goto out;
    }
    ret = tsd_hyp1f1_laplace(a + 1, b + 1, z, &f1);
    if (ret != 0) {
        goto out;
    }
    ret = tsd_hyp1f1_laplace(a + 2, b + 2, z, &f2);
    if (ret != 0) {
        goto out;
    }
    d1 = a / b * exp(f1 - f0);
    d2 = a / b * (a + 1) / (b + 1) * exp(f2 - f0);

    logl_ = f0 - mu_ij * t_i + (b - 1) * log(t_i) + tsd_betaln(a, b - a);
    mn_j_ = t_i * d1;
    sq_j = (t_i * t_i) * d2;

    *logl = logl_;
    *mn_j = mn_j_;
    *va_j = sq_j - mn_j_ * mn_j_;
out:
    return ret;
}

/*
 * log p(t_i, t_j) = log(t_i + t_j) * y_ij - mu_ij * (t_i + t_j)
 *     + log(t_i) * (a_i - 1) - b_i * t_i + log(t_j) * (a_j - 1) - b_j * t_j
 * Writes the normalising constant, E[t_i], V[t_i], E[t_j], V[t_j].
 */
int
tsd_unphased_moments(double a_i, double b_i, double a_j, double b_j, double y_ij,
    double mu_ij, double *logl, double *mn_i, double *va_i, double *mn_j, double *va_j)
{
    int ret = 0;
    double a, b, c, t, z, f0, f1, f2, s1, s2, d1, d2;
    double sq_i, sq_j, mn_i_, mn_j_, va_i_, va_j_, logl_;

    a = a_j;
    b = a_i + a_j + y_ij;
    c = a_j + a_i;
    t = mu_ij + b_i;
    z = t > 0 ? (mu_ij + b_j) / t : NAN;

    if (!tsd_valid_hyp2f1(a, b, c, 1 - z)) {
        set_nan_two_node(logl, mn_i, va_i, mn_j, va_j);
        goto out;
    }

    ret = tsd_hyp2f1_laplace(a, b, c, 1 - z, &f0);
    if (ret != 0) {
        goto out;
    }
    ret = tsd_hyp2f1_laplace(a + 1, b + 1, c + 1, 1 - z, &f1);
    if (ret != 0) {
        goto out;
    }
    ret = tsd_hyp2f1_laplace(a + 2, b + 2, c + 2, 1 - z, &f2);
    if (ret != 0) {
        goto out;
    }
    s1 = a * b / c;
    s2 = s1 * (a + 1) * (b + 1) / (c + 1);
    d1 = s1 * exp(f1 - f0);
    d2 = s2 * exp(f2 - f0);

    logl_ = f0 + tsd_betaln(a_j, a_i) + lgamma(b) - b * log(t);

    mn_j_ = d1 / t;
    sq_j = d2 / (t * t);
    va_j_ = sq_j - mn_j_ * mn_j_;

    mn_i_ = b / t - mn_j_ * z;
    sq_i = sq_j * (z * z) + (b + 1) * (mn_i_ - mn_j_ * z) / t;
    va_i_ = sq_i - mn_i_ * mn_i_;

    *logl = logl_;
    *mn_i = mn_i_;
    *va_i = va_i_;
    *mn_j = mn_j_;
    *va_j = va_j_;
out:
    return ret;
}

/*
 * log p(t_i) = log(2 t_i) * y_ij - mu_ij * (2 t_i) + log(t_i) * (a_i - 1) - b_i * t_i
 * for the two edges of an unphased individual to the same parent. Writes the
 * normalising constant, E[t_i], V[t_i]. A zero rate r (or one whose square
 * underflows) returns TSD_ERR_ZERO_DIVISION.
 */
int
tsd_twin_moments(double a_i, double b_i, double y_ij, double mu_ij, double *logl,
    double *mn_i, double *va_i)
{
    int ret = 0;
    double s, r, r_sq;

    s = a_i + y_ij;
    r = b_i + 2 * mu_ij;
    r_sq = r * r;
    if (r == 0.0 || r_sq == 0.0) {
        ret = tsd_trace_error(TSD_ERR_ZERO_DIVISION);
        goto out;
    }

    *logl = log(2.0) * y_ij + lgamma(s) - log(r) * s;
    *mn_i = s / r;
    *va_i = s / r_sq;
out:
    return ret;
}

/*
 * log p(t_j) = log(t_i + t_j) * y_ij - mu_ij * (t_i + t_j)
 *     + log(t_j) * (a_j - 1) - b_j * t_j
 * for a fixed age t_i > 0. Writes the normalising constant, E[t_j], V[t_j].
 */
int
tsd_sideways_moments(double t_i, double a_j, double b_j, double y_ij, double mu_ij,
    double *logl, double *mn_j, double *va_j)
{
    int ret = 0;
    double a, b, z, f0, d0, f1, d1;

    if (!(t_i > 0.0)) {
        ret = tsd_trace_error(TSD_ERR_INVALID_FIXED_AGE);
        goto out;
    }

    a = a_j;
    b = a_j + y_ij + 1;
    z = t_i * (mu_ij + b_j);

    if (!tsd_valid_hyperu(a, b, z)) {
        set_nan_one_node(logl, mn_j, va_j);
        goto out;
    }

    ret = tsd_hyperu_laplace(a, b, z, &f0, &d0);
    if (ret != 0) {
        goto out;
    }
    ret = tsd_hyperu_laplace(a + 1, b + 1, z, &f1, &d1);
    if (ret != 0) {
        goto out;
    }

    *logl = f0 - mu_ij * t_i + (b - 1) * log(t_i) + lgamma(a);
    *mn_j = -t_i * d0;
    *va_j = (t_i * t_i) * d0 * (d1 - d0);
out:
    return ret;
}

/* ------------------------------------------------------------------------- *
 * Node projections
 * ------------------------------------------------------------------------- */

/*
 * Project the moments of one marginal onto a gamma, writing its natural parameters to
 * `proj`; if the moments are unusable, write a copy of `pars` instead and set
 * `*usable` to 0.
 */
static int
project_moments(double mn, double va, const double pars[2], double proj[2], int *usable)
{
    int ret = 0;

    *usable = tsd_valid_moments(mn, va);
    if (!*usable) {
        proj[0] = pars[0];
        proj[1] = pars[1];
        goto out;
    }
    ret = tsd_approximate_gamma_mom(mn, va, proj);
out:
    return ret;
}

/* The projection of the two marginals of a two-parameter update; both are checked
 * before either is projected. */
static int
project_pair(double mn_i, double va_i, double mn_j, double va_j, const double pars_i[2],
    const double pars_j[2], double *logl, double logl_value, double proj_i[2],
    double proj_j[2])
{
    int ret = 0;
    int valid_i = tsd_valid_moments(mn_i, va_i);
    int valid_j = tsd_valid_moments(mn_j, va_j);

    if (!(valid_i && valid_j)) {
        *logl = NAN;
        proj_i[0] = pars_i[0];
        proj_i[1] = pars_i[1];
        proj_j[0] = pars_j[0];
        proj_j[1] = pars_j[1];
        goto out;
    }
    ret = tsd_approximate_gamma_mom(mn_i, va_i, proj_i);
    if (ret != 0) {
        goto out;
    }
    ret = tsd_approximate_gamma_mom(mn_j, va_j, proj_j);
    if (ret != 0) {
        goto out;
    }
    *logl = logl_value;
out:
    return ret;
}

/* Project a single marginal; on unusable moments `*logl` is NaN. */
static int
project_single(double mn, double va, const double pars[2], double logl_value,
    double *logl, double proj[2])
{
    int ret = 0;
    int usable;

    ret = project_moments(mn, va, pars, proj, &usable);
    if (ret != 0) {
        goto out;
    }
    *logl = usable ? logl_value : NAN;
out:
    return ret;
}

int
tsd_gamma_projection(const double pars_i[2], const double pars_j[2],
    const double pars_ij[2], double *logl, double proj_i[2], double proj_j[2])
{
    int ret = 0;
    double a_i = pars_i[0] + 1;
    double b_i = pars_i[1];
    double a_j = pars_j[0] + 1;
    double b_j = pars_j[1];
    double y_ij = pars_ij[0];
    double mu_ij = pars_ij[1];
    double logl_value, mn_i, va_i, mn_j, va_j;

    ret = tsd_moments(
        a_i, b_i, a_j, b_j, y_ij, mu_ij, &logl_value, &mn_i, &va_i, &mn_j, &va_j);
    if (ret != 0) {
        goto out;
    }
    ret = project_pair(
        mn_i, va_i, mn_j, va_j, pars_i, pars_j, logl, logl_value, proj_i, proj_j);
out:
    return ret;
}

int
tsd_leafward_projection(double t_i, const double pars_j[2], const double pars_ij[2],
    double *logl, double proj_j[2])
{
    int ret = 0;
    double a_j = pars_j[0] + 1;
    double b_j = pars_j[1];
    double y_ij = pars_ij[0];
    double mu_ij = pars_ij[1];
    double logl_value, mn_j, va_j;

    ret = tsd_leafward_moments(t_i, a_j, b_j, y_ij, mu_ij, &logl_value, &mn_j, &va_j);
    if (ret != 0) {
        goto out;
    }
    ret = project_single(mn_j, va_j, pars_j, logl_value, logl, proj_j);
out:
    return ret;
}

int
tsd_rootward_projection(double t_j, const double pars_i[2], const double pars_ij[2],
    double *logl, double proj_i[2])
{
    int ret = 0;
    double a_i = pars_i[0] + 1;
    double b_i = pars_i[1];
    double y_ij = pars_ij[0];
    double mu_ij = pars_ij[1];
    double logl_value, mn_i, va_i;

    ret = tsd_rootward_moments(t_j, a_i, b_i, y_ij, mu_ij, &logl_value, &mn_i, &va_i);
    if (ret != 0) {
        goto out;
    }
    ret = project_single(mn_i, va_i, pars_i, logl_value, logl, proj_i);
out:
    return ret;
}

int
tsd_unphased_projection(const double pars_i[2], const double pars_j[2],
    const double pars_ij[2], double *logl, double proj_i[2], double proj_j[2])
{
    int ret = 0;
    double a_i = pars_i[0] + 1;
    double b_i = pars_i[1];
    double a_j = pars_j[0] + 1;
    double b_j = pars_j[1];
    double y_ij = pars_ij[0];
    double mu_ij = pars_ij[1];
    double logl_value, mn_i, va_i, mn_j, va_j;

    ret = tsd_unphased_moments(
        a_i, b_i, a_j, b_j, y_ij, mu_ij, &logl_value, &mn_i, &va_i, &mn_j, &va_j);
    if (ret != 0) {
        goto out;
    }
    ret = project_pair(
        mn_i, va_i, mn_j, va_j, pars_i, pars_j, logl, logl_value, proj_i, proj_j);
out:
    return ret;
}

int
tsd_twin_projection(
    const double pars_i[2], const double pars_ij[2], double *logl, double proj_i[2])
{
    int ret = 0;
    double a_i = pars_i[0] + 1;
    double b_i = pars_i[1];
    double y_ij = pars_ij[0];
    double mu_ij = pars_ij[1];
    double logl_value, mn_i, va_i;

    ret = tsd_twin_moments(a_i, b_i, y_ij, mu_ij, &logl_value, &mn_i, &va_i);
    if (ret != 0) {
        goto out;
    }
    ret = project_single(mn_i, va_i, pars_i, logl_value, logl, proj_i);
out:
    return ret;
}

int
tsd_sideways_projection(double t_i, const double pars_j[2], const double pars_ij[2],
    double *logl, double proj_j[2])
{
    int ret = 0;
    double a_j = pars_j[0] + 1;
    double b_j = pars_j[1];
    double y_ij = pars_ij[0];
    double mu_ij = pars_ij[1];
    double logl_value, mn_j, va_j;

    ret = tsd_sideways_moments(t_i, a_j, b_j, y_ij, mu_ij, &logl_value, &mn_j, &va_j);
    if (ret != 0) {
        goto out;
    }
    ret = project_single(mn_j, va_j, pars_j, logl_value, logl, proj_j);
out:
    return ret;
}

/* ------------------------------------------------------------------------- *
 * Mutation moments
 * ------------------------------------------------------------------------- */

/*
 * Moments of the mutation age t_m. Conditional on the node ages, t_m is uniform on
 * (t_j, t_i) for a phased edge; the moments are those of the EP surrogate density
 * integrated over the node ages, in terms of hypergeometric functions. Arguments outside
 * the domain give NaN, not an error.
 */
int
tsd_mutation_moments(double a_i, double b_i, double a_j, double b_j, double y_ij,
    double mu_ij, double *mn_m, double *va_m)
{
    int ret = 0;
    double a, b, c, t, z, s1, d1, d2, d3, sq_m;
    double f000, f020, f111, f121, f222;

    a = a_j;
    b = a_i + a_j + y_ij;
    c = a_j + y_ij + 1;
    t = mu_ij + b_i;
    z = t > 0 ? (mu_ij - b_j) / t : NAN;

    if (!tsd_valid_hyp2f1(a, b, c, z)) {
        *mn_m = NAN;
        *va_m = NAN;
        goto out;
    }

    ret = tsd_hyp2f1_laplace(a + 0, b + 0, c + 0, z, &f000);
    if (ret != 0) {
        goto out;
    }
    ret = tsd_hyp2f1_laplace(a + 0, b + 2, c + 0, z, &f020);
    if (ret != 0) {
        goto out;
    }
    ret = tsd_hyp2f1_laplace(a + 1, b + 1, c + 1, z, &f111);
    if (ret != 0) {
        goto out;
    }
    ret = tsd_hyp2f1_laplace(a + 1, b + 2, c + 1, z, &f121);
    if (ret != 0) {
        goto out;
    }
    ret = tsd_hyp2f1_laplace(a + 2, b + 2, c + 2, z, &f222);
    if (ret != 0) {
        goto out;
    }

    s1 = a * b / c;
    d1 = b * (b + 1) / tsd_ipow(t, 2);
    d2 = d1 * a / c;
    d3 = d2 * (a + 1) / (c + 1);

    *mn_m = s1 * exp(f111 - f000) / t / 2 * (1 + z) + b / t / 2;
    sq_m = d1 * exp(f020 - f000) / 3 + d2 * exp(f121 - f000) / 3
           + d3 * exp(f222 - f000) / 3;
    *va_m = sq_m - tsd_ipow(*mn_m, 2);
out:
    return ret;
}

/* Moments of t_m given a fixed child age t_j: uniform on (t_j, t_i). */
int
tsd_mutation_rootward_moments(double t_j, double a_i, double b_i, double y_ij,
    double mu_ij, double *mn_m, double *va_m)
{
    int ret = 0;
    double logl, mn_i, va_i, sq_m;

    ret = tsd_rootward_moments(t_j, a_i, b_i, y_ij, mu_ij, &logl, &mn_i, &va_i);
    if (ret != 0) {
        goto out;
    }
    *mn_m = mn_i / 2 + t_j / 2;
    sq_m = (va_i + tsd_ipow(mn_i, 2) + mn_i * t_j + tsd_ipow(t_j, 2)) / 3;
    *va_m = sq_m - tsd_ipow(*mn_m, 2);
out:
    return ret;
}

/* Moments of t_m given a fixed parent age t_i: uniform on (t_j, t_i). */
int
tsd_mutation_leafward_moments(double t_i, double a_j, double b_j, double y_ij,
    double mu_ij, double *mn_m, double *va_m)
{
    int ret = 0;
    double logl, mn_j, va_j, sq_m;

    ret = tsd_leafward_moments(t_i, a_j, b_j, y_ij, mu_ij, &logl, &mn_j, &va_j);
    if (ret != 0) {
        goto out;
    }
    *mn_m = mn_j / 2 + t_i / 2;
    sq_m = (va_j + tsd_ipow(mn_j, 2) + mn_j * t_i + tsd_ipow(t_i, 2)) / 3;
    *va_m = sq_m - tsd_ipow(*mn_m, 2);
out:
    return ret;
}

/*
 * Unphased edge: the mutation is under node i with probability t_i / (t_i + t_j), and
 * uniform on (0, t_i) or (0, t_j). Integrating over the EP surrogate density gives
 * hypergeometric functions with integer perturbations of a_i, a_j and y_ij.
 */
int
tsd_mutation_unphased_moments(double a_i, double b_i, double a_j, double b_j,
    double y_ij, double mu_ij, double *pr_m, double *mn_m, double *va_m)
{
    int ret = 0;
    double a, b, c, t, z, s0, s1, s2, d0, d1, d2, sq_m;
    double f000, f001, f012, f023, f212, f323;

    a = a_j;
    b = a_j + a_i + y_ij;
    c = a_j + a_i;
    t = mu_ij + b_i;
    z = t > 0 ? (mu_ij + b_j) / t : NAN;

    if (!tsd_valid_hyp2f1(a, b, c, 1 - z)) {
        *pr_m = NAN;
        *mn_m = NAN;
        *va_m = NAN;
        goto out;
    }

    ret = tsd_hyp2f1_laplace(a + 0, b + 0, c + 0, 1 - z, &f000);
    if (ret != 0) {
        goto out;
    }
    ret = tsd_hyp2f1_laplace(a + 0, b + 0, c + 1, 1 - z, &f001);
    if (ret != 0) {
        goto out;
    }
    ret = tsd_hyp2f1_laplace(a + 0, b + 1, c + 2, 1 - z, &f012);
    if (ret != 0) {
        goto out;
    }
    ret = tsd_hyp2f1_laplace(a + 0, b + 2, c + 3, 1 - z, &f023);
    if (ret != 0) {
        goto out;
    }
    ret = tsd_hyp2f1_laplace(a + 2, b + 1, c + 2, 1 - z, &f212);
    if (ret != 0) {
        goto out;
    }
    ret = tsd_hyp2f1_laplace(a + 3, b + 2, c + 3, 1 - z, &f323);
    if (ret != 0) {
        goto out;
    }

    s0 = b / t / c / (c + 1);
    s1 = (c - a) * (c - a + 1);
    s2 = a * (a + 1);
    d0 = s0 * (b + 1) / t / (c + 2);
    d1 = s1 * (c - a + 2);
    d2 = s2 * (a + 2);

    *mn_m = s0 * s1 * exp(f012 - f000) / 2 + s0 * s2 * exp(f212 - f000) / 2;
    sq_m = d0 * d1 * exp(f023 - f000) / 3 + d0 * d2 * exp(f323 - f000) / 3;
    *va_m = sq_m - tsd_ipow(*mn_m, 2);
    *pr_m = (c - a) / c * exp(f001 - f000);
out:
    return ret;
}

/* Twin nodes (both parents are the same node): a closed form. */
int
tsd_mutation_twin_moments(double a_i, double b_i, double y_ij, double mu_ij,
    double *pr_m, double *mn_m, double *va_m)
{
    int ret = 0;
    double s, r, r_sq, sq_m;

    s = a_i + y_ij;
    r = b_i + 2 * mu_ij;
    r_sq = tsd_ipow(r, 2);
    if (r_sq == 0.0) {
        /* r is zero or its square underflows */
        ret = tsd_trace_error(TSD_ERR_ZERO_DIVISION);
        goto out;
    }
    *pr_m = 0.5;
    *mn_m = s / r / 2;
    sq_m = (s + 1) * s / 3 / r_sq;
    *va_m = sq_m - tsd_ipow(*mn_m, 2);
out:
    return ret;
}

/* Unphased edge with a fixed parent age t_i, in terms of Tricomi functions. */
int
tsd_mutation_sideways_moments(double t_i, double a_j, double b_j, double y_ij,
    double mu_ij, double *pr_m, double *mn_m, double *va_m)
{
    int ret = 0;
    double a, b, z, sq_m;
    double f00, d00, f10, d10, f21, d21, f32, d32;

    if (!(t_i > 0)) {
        ret = tsd_trace_error(TSD_ERR_INVALID_FIXED_AGE);
        goto out;
    }

    a = a_j;
    b = a_j + y_ij + 1;
    z = t_i * (mu_ij + b_j);

    if (!tsd_valid_hyperu(a, b, z)) {
        *pr_m = NAN;
        *mn_m = NAN;
        *va_m = NAN;
        goto out;
    }

    ret = tsd_hyperu_laplace(a + 0, b + 0, z, &f00, &d00);
    if (ret != 0) {
        goto out;
    }
    ret = tsd_hyperu_laplace(a + 1, b + 0, z, &f10, &d10);
    if (ret != 0) {
        goto out;
    }
    ret = tsd_hyperu_laplace(a + 2, b + 1, z, &f21, &d21);
    if (ret != 0) {
        goto out;
    }
    ret = tsd_hyperu_laplace(a + 3, b + 2, z, &f32, &d32);
    if (ret != 0) {
        goto out;
    }

    *pr_m = 1.0 - exp(f10 - f00) * a;
    *mn_m = *pr_m * t_i / 2 + t_i * exp(f21 - f00) * a * (a + 1) / 2;
    sq_m = *pr_m * tsd_ipow(t_i, 2) / 3
           + tsd_ipow(t_i, 2) * exp(f32 - f00) * a * (a + 1) * (a + 2) / 3;
    *va_m = sq_m - tsd_ipow(*mn_m, 2);
out:
    return ret;
}

/* Both node ages fixed and phased: t_m is uniform on (t_j, t_i). */
int
tsd_mutation_edge_moments(double t_i, double t_j, double *mn_m, double *va_m)
{
    *mn_m = 1.0 / 2 * (t_i + t_j);
    *va_m = 1.0 / 12 * tsd_ipow(t_i - t_j, 2);
    return 0;
}

/* Both node ages fixed and unphased: t_m is uniform on (0, t_i) or (0, t_j). */
int
tsd_mutation_block_moments(
    double t_i, double t_j, double *pr_m, double *mn_m, double *va_m)
{
    int ret = 0;
    double sq_m;

    if (!(t_i > 0) || !(t_j > 0)) {
        ret = tsd_trace_error(TSD_ERR_INVALID_FIXED_AGE);
        goto out;
    }
    *pr_m = t_i / (t_i + t_j);
    *mn_m = *pr_m * t_i / 2 + (1 - *pr_m) * t_j / 2;
    sq_m = *pr_m * tsd_ipow(t_i, 2) / 3 + (1 - *pr_m) * tsd_ipow(t_j, 2) / 3;
    *va_m = sq_m - tsd_ipow(*mn_m, 2);
out:
    return ret;
}

/* ------------------------------------------------------------------------- *
 * Mutation projections
 * ------------------------------------------------------------------------- */

/* Set the outputs of a projection whose moments are unusable: NaN, in-band. */
static void
mutation_projection_invalid(double *pr_m, double proj_m[2])
{
    *pr_m = NAN;
    proj_m[0] = NAN;
    proj_m[1] = NAN;
}

/*
 * The gamma matching the mean and variance of the mutation age, with phase
 * probability `pr`; NaN outputs if the moments are not usable. Phase probabilities
 * outside [0, 1] (or NaN) are unusable when `check_pr` is set.
 */
static int
mutation_project_moments(
    double pr, int check_pr, double mn_m, double va_m, double *pr_m, double proj_m[2])
{
    int ret = 0;
    int usable = tsd_valid_moments(mn_m, va_m);

    if (check_pr && !(pr >= 0.0 && pr <= 1.0)) {
        usable = 0;
    }
    if (!usable) {
        mutation_projection_invalid(pr_m, proj_m);
        goto out;
    }
    ret = tsd_approximate_gamma_mom(mn_m, va_m, proj_m);
    if (ret != 0) {
        goto out;
    }
    *pr_m = pr;
out:
    return ret;
}

int
tsd_mutation_gamma_projection(const double pars_i[2], const double pars_j[2],
    const double pars_ij[2], double *pr_m, double proj_m[2])
{
    int ret = 0;
    double mn_m, va_m;
    const double a_i = pars_i[0] + 1;
    const double a_j = pars_j[0] + 1;

    ret = tsd_mutation_moments(
        a_i, pars_i[1], a_j, pars_j[1], pars_ij[0], pars_ij[1], &mn_m, &va_m);
    if (ret != 0) {
        goto out;
    }
    ret = mutation_project_moments(1.0, 0, mn_m, va_m, pr_m, proj_m);
out:
    return ret;
}

int
tsd_mutation_leafward_projection(double t_i, const double pars_j[2],
    const double pars_ij[2], double *pr_m, double proj_m[2])
{
    int ret = 0;
    double mn_m, va_m;
    const double a_j = pars_j[0] + 1;

    ret = tsd_mutation_leafward_moments(
        t_i, a_j, pars_j[1], pars_ij[0], pars_ij[1], &mn_m, &va_m);
    if (ret != 0) {
        goto out;
    }
    ret = mutation_project_moments(1.0, 0, mn_m, va_m, pr_m, proj_m);
out:
    return ret;
}

int
tsd_mutation_rootward_projection(double t_j, const double pars_i[2],
    const double pars_ij[2], double *pr_m, double proj_m[2])
{
    int ret = 0;
    double mn_m, va_m;
    const double a_i = pars_i[0] + 1;

    ret = tsd_mutation_rootward_moments(
        t_j, a_i, pars_i[1], pars_ij[0], pars_ij[1], &mn_m, &va_m);
    if (ret != 0) {
        goto out;
    }
    ret = mutation_project_moments(1.0, 0, mn_m, va_m, pr_m, proj_m);
out:
    return ret;
}

int
tsd_mutation_edge_projection(double t_i, double t_j, double *pr_m, double proj_m[2])
{
    int ret = 0;
    double mn_m, va_m;

    ret = tsd_mutation_edge_moments(t_i, t_j, &mn_m, &va_m);
    if (ret != 0) {
        goto out;
    }
    ret = mutation_project_moments(1.0, 0, mn_m, va_m, pr_m, proj_m);
out:
    return ret;
}

int
tsd_mutation_unphased_projection(const double pars_i[2], const double pars_j[2],
    const double pars_ij[2], double *pr_m, double proj_m[2])
{
    int ret = 0;
    double pr, mn_m, va_m;
    const double a_i = pars_i[0] + 1;
    const double a_j = pars_j[0] + 1;

    ret = tsd_mutation_unphased_moments(
        a_i, pars_i[1], a_j, pars_j[1], pars_ij[0], pars_ij[1], &pr, &mn_m, &va_m);
    if (ret != 0) {
        goto out;
    }
    ret = mutation_project_moments(pr, 1, mn_m, va_m, pr_m, proj_m);
out:
    return ret;
}

int
tsd_mutation_twin_projection(
    const double pars_i[2], const double pars_ij[2], double *pr_m, double proj_m[2])
{
    int ret = 0;
    double pr, mn_m, va_m;
    const double a_i = pars_i[0] + 1;

    ret = tsd_mutation_twin_moments(
        a_i, pars_i[1], pars_ij[0], pars_ij[1], &pr, &mn_m, &va_m);
    if (ret != 0) {
        goto out;
    }
    ret = mutation_project_moments(pr, 1, mn_m, va_m, pr_m, proj_m);
out:
    return ret;
}

int
tsd_mutation_sideways_projection(double t_i, const double pars_j[2],
    const double pars_ij[2], double *pr_m, double proj_m[2])
{
    int ret = 0;
    double pr, mn_m, va_m;
    const double a_j = pars_j[0] + 1;

    ret = tsd_mutation_sideways_moments(
        t_i, a_j, pars_j[1], pars_ij[0], pars_ij[1], &pr, &mn_m, &va_m);
    if (ret != 0) {
        goto out;
    }
    ret = mutation_project_moments(pr, 1, mn_m, va_m, pr_m, proj_m);
out:
    return ret;
}

int
tsd_mutation_block_projection(double t_i, double t_j, double *pr_m, double proj_m[2])
{
    int ret = 0;
    double pr, mn_m, va_m;

    ret = tsd_mutation_block_moments(t_i, t_j, &pr, &mn_m, &va_m);
    if (ret != 0) {
        goto out;
    }
    ret = mutation_project_moments(pr, 1, mn_m, va_m, pr_m, proj_m);
out:
    return ret;
}
