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
 * Numerically stable implementations of the Gauss hypergeometric function and related
 * special functions, ported from tsdate/hypergeo.py (numba). Operation order and
 * libm calls follow the numba code so that results agree to the last bit or two.
 *
 * In numba, a float division by zero raises ZeroDivisionError; here it returns
 * TSD_ERR_ZERO_DIVISION, but only at the divisions that the tests show to be
 * reachable (those of the poles of the polygamma functions and of the Laplace
 * roots). The other divisions follow IEEE semantics (infinity or NaN).
 */

#include <math.h>

#include "mathutil.h"
#include "tsdate.h"

/* Digamma function for x > 0 (or NaN), by recurrence and asymptotic series. */
static double
digamma_positive(double x)
{
    double xpm2, value;

    if (x <= 1e-5) {
        return -TSD_EULER_GAMMA - (1 / x);
    }
    if (x < 8.5) {
        return digamma_positive(1 + x) - 1 / x;
    }
    xpm2 = 1 / tsd_ipow(x, 2);
    value = log(x);
    value -= 0.5 / x;
    value -= 0.083333333333333333 * xpm2;
    value += 0.008333333333333333 * tsd_ipow(xpm2, 2);
    value -= 0.003968253968253968 * tsd_ipow(xpm2, 3);
    value += 0.004166666666666667 * tsd_ipow(xpm2, 4);
    value -= 0.007575757575757576 * tsd_ipow(xpm2, 5);
    value += 0.021092796092796094 * tsd_ipow(xpm2, 6);
    return value;
}

/* Digamma (psi) function, from asymptotic series expansion. */
int
tsd_digamma(double x, double *out)
{
    int ret = 0;
    double reflected, tan_pix;

    if (x <= 0.0) {
        /* Reflection formula; 1 - x >= 1 */
        reflected = digamma_positive(1 - x);
        tan_pix = tan(TSD_PI * x);
        if (tan_pix == 0.0) {
            ret = tsd_trace_error(TSD_ERR_ZERO_DIVISION);
            goto out;
        }
        *out = reflected - TSD_PI / tan_pix;
    } else {
        *out = digamma_positive(x);
    }
out:
    return ret;
}

/* Trigamma function for x > 1e-4 (or NaN), by recurrence and asymptotic series. */
static double
trigamma_large(double x)
{
    double x_sq, xpm1, xpm2, value;

    x_sq = tsd_ipow(x, 2);
    if (x < 5) {
        return trigamma_large(1 + x) + 1 / x_sq;
    }
    xpm1 = 1 / x;
    xpm2 = 1 / x_sq;
    value = 1.000000000000000000;
    value += 0.500000000000000000 * xpm1;
    value += 0.166666666666666667 * pow(xpm2, 1);
    value -= 0.033333333333333333 * pow(xpm2, 2);
    value += 0.023809523809523808 * pow(xpm2, 3);
    value -= 0.033333333333333333 * pow(xpm2, 4);
    value += 0.075757575757575756 * pow(xpm2, 5);
    value -= 0.253113553113553102 * pow(xpm2, 6);
    value += 1.166666666666666741 * pow(xpm2, 7);
    return xpm1 * value;
}

/* Trigamma function, from asymptotic series expansion. */
int
tsd_trigamma(double x, double *out)
{
    int ret = 0;
    double reflected, sin_pix, sin_sq, x_sq;

    if (x <= 0.0) {
        /* Reflection formula; 1 - x >= 1 */
        reflected = trigamma_large(1 - x);
        sin_pix = sin(TSD_PI * x);
        sin_sq = tsd_ipow(sin_pix, 2);
        if (sin_sq == 0.0) {
            ret = tsd_trace_error(TSD_ERR_ZERO_DIVISION);
            goto out;
        }
        *out = -reflected + tsd_ipow(TSD_PI, 2) / sin_sq;
    } else if (x <= 1e-4) {
        x_sq = tsd_ipow(x, 2);
        if (x_sq == 0.0) {
            ret = tsd_trace_error(TSD_ERR_ZERO_DIVISION);
            goto out;
        }
        *out = 1 / x_sq;
    } else {
        *out = trigamma_large(x);
    }
out:
    return ret;
}

/* Log of the beta function. */
double
tsd_betaln(double p, double q)
{
    return lgamma(p) + lgamma(q) - lgamma(p + q);
}

/*
 * Approximate Tricomi's confluent hypergeometric function with real arguments, using
 * Laplace's method with a calibration point. Also returns the derivative wrt x.
 */
int
tsd_hyperu_laplace(double a, double b, double x, double *value, double *dvalue)
{
    int ret = 0;
    double t, denom, u, r, g, w, v, dr, dg, du;

    if (!(b >= a && a > 0.0)) {
        ret = tsd_trace_error(TSD_ERR_INVALID_HYPERGEO);
        goto out;
    }
    if (!(x > 0.0)) {
        ret = tsd_trace_error(TSD_ERR_INVALID_HYPERGEO);
        goto out;
    }

    t = b - x - 1;
    denom = sqrt(tsd_ipow(t, 2) + 4 * x * a) - t;
    if (denom == 0.0) {
        ret = tsd_trace_error(TSD_ERR_ZERO_DIVISION);
        goto out;
    }
    u = 2 * a / denom;
    r = (b - a - 1) * tsd_ipow(u, 2) + a * tsd_ipow(1 + u, 2);
    g = (b - a) * log(1 + u);
    g += a * (1 + log(u));
    g -= x * u;
    g -= log(a) * (a - 0.5);

    w = (b - 1) * u + a;
    v = u * (1 + u);
    dr = w / (w * u + a * (1 + u));
    if (v == 0.0) {
        ret = tsd_trace_error(TSD_ERR_ZERO_DIVISION);
        goto out;
    }
    dg = (w - x * v + u) / v;
    du = -u * v / (w - x * u + a);

    *value = g - log(r) / 2;
    *dvalue = (dg - dr) * du - u;
out:
    return ret;
}

/*
 * Approximate Kummer's confluent hypergeometric function with real arguments, using
 * Laplace's method with a calibration point.
 */
int
tsd_hyp1f1_laplace(double a, double b, double x, double *out)
{
    int ret = 0;
    double t, denom, u, r, g;

    if (!(b > a && a > 0.0)) {
        ret = tsd_trace_error(TSD_ERR_INVALID_HYPERGEO);
        goto out;
    }

    if (x == 0.0) {
        *out = 0.0;
        goto out;
    }

    t = x - b;
    denom = sqrt(tsd_ipow(t, 2) + 4 * x * a) - t;
    if (denom == 0.0) {
        ret = tsd_trace_error(TSD_ERR_ZERO_DIVISION);
        goto out;
    }
    u = 2 * a / denom;
    r = tsd_ipow(u, 2) * b / a + tsd_ipow(1 - u, 2) * b / (b - a);
    g = (b - a) * log(1 - u);
    g += a * log(u);
    g += b * log(b);
    g -= (b - a) * log(b - a);
    g -= a * log(a);
    g += x * u;

    *out = g - log(r) / 2;
out:
    return ret;
}

/*
 * Gauss hypergeometric function when `x` is near unity. See limits in DLMF 15.4.
 *
 * TODO: this works in practice, but when (c - a - b) is close to zero the limits don't
 * converge. A good reference is Buhring 2003 "Partial sums of hypergeometric series of
 * unit argument"
 */
static int
hyp2f1_unity(double a, double b, double c, double x, double *out)
{
    int ret = 0;
    double g, value;

    tsd_bug_assert(tsd_isclose(x, 1.0));
    if (!(x < 1.0)) {
        ret = tsd_trace_error(TSD_ERR_INVALID_HYPERGEO);
        goto out;
    }

    g = c - a - b;

    if (g < 0.0) {
        value = lgamma(c) + lgamma(-g) - lgamma(a) - lgamma(b) + g * log(1 - x);
    } else if (g > 0.0) {
        /* will only occur when a_i + a_j < 1 */
        value = lgamma(c) + lgamma(g) - lgamma(c - a) - lgamma(c - b);
    } else {
        /* will only occur when a_i + a_j == 1 */
        value = log(-log(1 - x)) + lgamma(a + b) - lgamma(a) - lgamma(b);
    }
    *out = value;
out:
    return ret;
}

/*
 * Approximate a Gaussian hypergeometric function with real arguments, using Laplace's
 * method with a calibration point.
 */
int
tsd_hyp2f1_laplace(double a, double b, double c, double x, double *out)
{
    int ret = 0;
    double s, t, u, y, yy, my, ymy, r, f, unity, denom;

    /* TODO: simplify, we can safely assume a,b > 0? */
    if (!(c > 0.0 && a >= 0.0 && b >= 0.0 && c >= a && x < 1.0)) {
        ret = tsd_trace_error(TSD_ERR_INVALID_HYPERGEO);
        goto out;
    }

    if (x == 0.0) {
        *out = 0.0;
        goto out;
    }

    s = 0.0;
    if (x < 0.0) {
        s = -b * log(1 - x);
        a = c - a;
        x = x / (x - 1);
    }

    if (tsd_isclose(x, 1.0)) {
        ret = hyp2f1_unity(a, b, c, x, &unity);
        if (ret != 0) {
            goto out;
        }
        *out = s + unity;
        goto out;
    }

    t = x * (b - a) - c;
    u = sqrt(tsd_ipow(t, 2) - 4 * a * x * (c - b)) - t;
    if (u == 0.0) {
        ret = tsd_trace_error(TSD_ERR_ZERO_DIVISION);
        goto out;
    }
    y = 2 * a / u;

    /* Root on boundary */
    if (!(0 < y && y < 1)) {
        ret = tsd_trace_error(TSD_ERR_INVALID_HYPERGEO);
        goto out;
    }

    yy = tsd_ipow(y, 2) / a;
    denom = c - a;
    if (denom == 0.0) {
        ret = tsd_trace_error(TSD_ERR_ZERO_DIVISION);
        goto out;
    }
    my = tsd_ipow(1 - y, 2) / denom;
    ymy = tsd_ipow(x, 2) * b * yy * my / tsd_ipow(1 - x * y, 2);
    r = yy + my - ymy;
    f = (c - 0.5) * log(c);
    f += a * (log(y) - log(a));
    f += (c - a) * (log(1 - y) - log(c - a));
    f -= b * log(1 - x * y);

    *out = f - log(r) / 2 + s;
out:
    return ret;
}

/* Gaussian approximation of the derivative of the lower incomplete gamma function. */
static double
gammainc_der_gaussian(double p, double x)
{
    double pn1, grad, cbrt_ratio;

    cbrt_ratio = pow(x / p, 1.0 / 3.0);
    pn1 = 3 * sqrt(p) * (cbrt_ratio + 1 / (9 * p) - 1);
    grad = pn1 / (2 * p) - 1 / sqrt(p) * (cbrt_ratio + 1 / (3 * p));
    grad *= 1 / sqrt(2 * TSD_PI) * exp(-(tsd_ipow(pn1, 2)) / 2);
    return grad;
}

/* Derivative of the lower incomplete gamma function by series expansion. */
static double
gammainc_der_series(double p, double x, double tol, double elimit)
{
    double arg, value, grad, c, dc, a, darg;

    arg = p * log(x) - x - lgamma(p + 1);
    value = 1.0;
    grad = 0.0;
    c = 1.0;
    dc = 0.0;
    a = p;
    while (1) {
        a += 1.0;
        dc = -c * x / tsd_ipow(a, 2) + dc * x / a;
        c *= x / a;
        value += c;
        grad += dc;
        if (c <= tol) {
            break;
        }
    }
    darg = exp(arg) * (log(x) - digamma_positive(p + 1));
    grad = grad * exp(arg) + value * darg;
    arg += log(value);
    if (!(arg >= elimit)) {
        grad = 0.0;
    }
    return grad;
}

/* Derivative of the upper incomplete gamma function by continued fraction. */
static double
gammainc_der_fraction(double p, double x, double tol, double elimit, double oflo)
{
    double arg, a, b, c, an, value, grad, rn, darg;
    double pn1, pn2, pn3, pn4, pn5, pn6;
    double dpn1, dpn2, dpn3, dpn4, dpn5, dpn6;

    arg = p * log(x) - x - lgamma(p);
    a = 1.0 - p;
    b = a + x + 1.0;
    c = 0.0;
    pn1 = 1.0;
    pn2 = x;
    pn3 = x + 1.0;
    pn4 = x * b;
    dpn1 = 0.0;
    dpn2 = 0.0;
    dpn3 = 0.0;
    dpn4 = -x;
    value = pn3 / pn4;
    grad = 0.0;
    while (1) {
        a += 1.0;
        b += 2.0;
        c += 1.0;
        an = a * c;
        pn5 = b * pn3 - an * pn1;
        pn6 = b * pn4 - an * pn2;
        dpn5 = b * dpn3 - pn3 - an * dpn1 + c * pn1;
        dpn6 = b * dpn4 - pn4 - an * dpn2 + c * pn2;
        if (pn6 != 0.0) {
            rn = pn5 / pn6;
            grad = (dpn5 * pn6 - pn5 * dpn6) / tsd_ipow(pn6, 2);
            if (fabs(value - rn) <= tsd_min(tol, tol * rn)) {
                break;
            }
            value = rn;
        }
        pn1 = pn3;
        pn2 = pn4;
        pn3 = pn5;
        pn4 = pn6;
        dpn1 = dpn3;
        dpn2 = dpn4;
        dpn3 = dpn5;
        dpn4 = dpn6;
        if (oflo <= fabs(pn5)) {
            pn1 /= oflo;
            pn2 /= oflo;
            pn3 /= oflo;
            pn4 /= oflo;
            dpn1 /= oflo;
            dpn2 /= oflo;
            dpn3 /= oflo;
            dpn4 /= oflo;
        }
    }
    darg = exp(arg) * (log(x) - digamma_positive(p));
    grad = grad * exp(arg) + value * darg;
    arg += log(value);
    if (arg >= elimit) {
        grad *= -1.0;
    } else {
        grad = 0.0;
    }
    return grad;
}

/*
 * Derivative of the lower incomplete gamma function with regards to `p`.
 *
 * Based on Shea B (1988) "Algorithm AS 239" Applied Statistics 37: 466-473
 *
 * Adapted from https://people.math.sc.edu/Burkardt/cpp_src/asa239/asa239.cpp
 */
int
tsd_gammainc_der(double p, double x, double *out)
{
    int ret = 0;
    const double elimit = -88.0;
    const double oflo = 1.0e37;
    const double plimit = 1.0e3;
    const double tol = 1.0e-14;
    const double xbig = 1.0e8;

    if (!(x >= 0.0 && p > 0.0)) {
        ret = tsd_trace_error(TSD_ERR_INVALID_HYPERGEO);
        goto out;
    }

    if (x == 0) {
        *out = 0.0;
    } else if (x > xbig) {
        *out = 0.0;
    } else if (p > plimit) {
        *out = gammainc_der_gaussian(p, x);
    } else if (x <= 1 || x < p) {
        *out = gammainc_der_series(p, x, tol, elimit);
    } else {
        *out = gammainc_der_fraction(p, x, tol, elimit, oflo);
    }
out:
    return ret;
}
