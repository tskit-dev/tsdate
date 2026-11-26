"""
Oracles for the special functions (``lib/hypergeo.c``).
"""

from math import pi

import _tsdate
import numpy as np

# Plain `math` raises on log(0), exp overflow etc., where C gives -inf, inf or
# nan; `_libm` gives the C results.
# C computes `x ** n` for integer constants `n` by repeated squaring, whereas
# CPython calls libm pow, which differs by an ulp in a few percent of cases; `ipow`
# squares (see `_libm.ipow`).
# NumPy ufuncs on Python floats return `np.float64`, so a following division
# by zero (e.g. `pi / tan(0)`) gives inf with a warning instead of the
# ZeroDivisionError that stands for the library's TSD_ERR_ZERO_DIVISION; and NumPy's
# SIMD loops may differ from libm. Logs, tangents, sines and powers are the `_libm`
# versions (libm functions on Python floats).
from ._libm import erf, exp, ipow, lgamma, log, pow, sin, sqrt, tan  # noqa: A004

# gammaincinv is a primitive of the oracles, as the libm functions are: the C library's
# port of `scipy.special.gammaincinv`, which `TestGammaincInv` compares with SciPy.
gammainc_inv = _tsdate.gammainc_inv


def digamma(x):
    """
    Digamma (psi) function, from asymptotic series expansion.
    """
    if x <= 0.0:
        return digamma(1 - x) - np.pi / tan(np.pi * x)
    if x <= 1e-5:
        return -np.euler_gamma - (1 / x)
    if x < 8.5:
        return digamma(1 + x) - 1 / x
    xpm2 = 1 / ipow(x, 2)
    return (
        log(x)
        - 0.5 / x
        - 0.083333333333333333 * xpm2
        + 0.008333333333333333 * ipow(xpm2, 2)
        - 0.003968253968253968 * ipow(xpm2, 3)
        + 0.004166666666666667 * ipow(xpm2, 4)
        - 0.007575757575757576 * ipow(xpm2, 5)
        + 0.021092796092796094 * ipow(xpm2, 6)
    )


def trigamma(x):
    """
    Trigamma function, from asymptotic series expansion
    """
    if x <= 0.0:
        return -trigamma(1 - x) + ipow(np.pi, 2) / ipow(sin(np.pi * x), 2)
    if x <= 1e-4:
        return 1 / ipow(x, 2)
    if x < 5:
        return trigamma(1 + x) + 1 / ipow(x, 2)
    xpm1 = 1 / x
    xpm2 = 1 / ipow(x, 2)
    return xpm1 * (
        1.000000000000000000
        + 0.500000000000000000 * xpm1
        + 0.166666666666666667 * pow(xpm2, 1)
        - 0.033333333333333333 * pow(xpm2, 2)
        + 0.023809523809523808 * pow(xpm2, 3)
        - 0.033333333333333333 * pow(xpm2, 4)
        + 0.075757575757575756 * pow(xpm2, 5)
        - 0.253113553113553102 * pow(xpm2, 6)
        + 1.166666666666666741 * pow(xpm2, 7)
    )


def betaln(p, q):
    """
    Log of the beta function.
    """
    return lgamma(p) + lgamma(q) - lgamma(p + q)


def hyperu_laplace(a, b, x):
    """
    Approximate Tricomi's confluent hypergeometric function with real
    arguments, using Laplace's method with a calibration point.

    Also returns derivative wrt x.

    TODO: details
    """

    assert b >= a > 0.0
    assert x > 0.0

    t = b - x - 1
    u = 2 * a / (sqrt(ipow(t, 2) + 4 * x * a) - t)
    r = (b - a - 1) * ipow(u, 2) + a * ipow(1 + u, 2)
    g = (b - a) * log(1 + u) + a * (1 + log(u)) - x * u - log(a) * (a - 1 / 2)

    w = (b - 1) * u + a
    v = u * (1 + u)
    dr = w / (w * u + a * (1 + u))
    dg = (w - x * v + u) / v
    du = -u * v / (w - x * u + a)

    return g - log(r) / 2, (dg - dr) * du - u


def hyp1f1_laplace(a, b, x):
    """
    Approximate Kummer's confluent hypergeometric function with real arguments,
    using Laplace's method with a calibration point.

    TODO: details
    """

    assert b > a > 0.0

    if x == 0.0:
        return 0.0

    t = x - b
    u = 2 * a / (sqrt(ipow(t, 2) + 4 * x * a) - t)
    r = ipow(u, 2) * b / a + ipow(1 - u, 2) * b / (b - a)
    g = (
        (b - a) * log(1 - u)
        + a * log(u)
        + b * log(b)
        - (b - a) * log(b - a)
        - a * log(a)
        + x * u
    )

    return g - log(r) / 2


def hyp2f1_laplace(a, b, c, x):
    r"""
    Approximate a Gaussian hypergeometric function with real arguments,
    using Laplace's method with a calibration point.

    TODO: details
    """

    def _hyp2f1_unity(a, b, c, x):
        """
        Gauss hypergeometric function when `x` is near unity

        See limits in DLMF 15.4.

        TODO: this works in practice, but when (c - a - b) is close to zero the
        limits don't converge. A good reference is Buhring 2003 "Partial sums of
        hypergeometric series of unit argument"
        """
        assert np.isclose(x, 1.0)
        assert x < 1.0

        g = c - a - b

        if g < 0.0:
            return lgamma(c) + lgamma(-g) - lgamma(a) - lgamma(b) + g * log(1 - x)
        elif g > 0.0:
            # will only occur when a_i + a_j < 1
            return lgamma(c) + lgamma(g) - lgamma(c - a) - lgamma(c - b)
        else:
            # will only occur when a_i + a_j == 1
            return log(-log(1 - x)) + lgamma(a + b) - lgamma(a) - lgamma(b)

    # TODO: simplify, we can safely assume a,b > 0?
    assert c > 0.0
    assert a >= 0.0
    assert b >= 0.0
    assert c >= a
    assert x < 1.0

    if x == 0.0:
        return 0.0

    s = 0.0
    if x < 0.0:
        s = -b * log(1 - x)
        a = c - a
        x = x / (x - 1)

    if np.isclose(x, 1.0):
        return s + _hyp2f1_unity(a, b, c, x)

    t = x * (b - a) - c
    u = sqrt(ipow(t, 2) - 4 * a * x * (c - b)) - t
    y = 2 * a / u

    assert 0 < y < 1, "Root on boundary"

    yy = ipow(y, 2) / a
    my = ipow(1 - y, 2) / (c - a)
    ymy = ipow(x, 2) * b * yy * my / ipow(1 - x * y, 2)
    r = yy + my - ymy
    f = (
        (c - 1 / 2) * log(c)
        + a * (log(y) - log(a))
        + (c - a) * (log(1 - y) - log(c - a))
        - b * log(1 - x * y)
    )

    return f - log(r) / 2 + s


def gammainc_der(p, x):
    """
    Derivative of lower incomplete gamma function with regards to `p`.

    Based on Shea B (1988) "Algorithm AS 239" Applied Statistics 37: 466-473

    Adapted from https://people.math.sc.edu/Burkardt/cpp_src/asa239/asa239.cpp
    """

    elimit = -88.0
    oflo = 1.0e37
    plimit = 1.0e3
    tol = 1.0e-14
    xbig = 1.0e8

    assert x >= 0.0
    assert p > 0.0

    if x == 0:
        value, grad = 0.0, 0.0
        return grad

    if x > xbig:
        value, grad = 1.0, 0.0
        return grad

    if p > plimit:  # gaussian approximation
        pn1 = 3 * sqrt(p) * (pow(x / p, 1 / 3) + 1 / (9 * p) - 1)
        grad = pn1 / (2 * p) - 1 / sqrt(p) * (pow(x / p, 1 / 3) + 1 / (3 * p))
        grad *= 1 / sqrt(2 * pi) * exp(-(ipow(pn1, 2)) / 2)
        value = (1 + erf(pn1 / sqrt(2))) / 2
        return grad

    if x <= 1 or x < p:  # series expansion
        arg = p * log(x) - x - lgamma(p + 1)
        value, grad = 1.0, 0.0
        c, dc = 1.0, 0.0
        a = p
        while True:
            a += 1.0
            dc = -c * x / ipow(a, 2) + dc * x / a
            c *= x / a
            value += c
            grad += dc
            if c <= tol:
                break
        darg = exp(arg) * (log(x) - digamma(p + 1))
        grad = grad * exp(arg) + value * darg
        arg += log(value)
        if arg >= elimit:
            value = exp(arg)
        else:
            grad = 0.0
            value = 0.0
        return grad
    else:  # continued fraction
        arg = p * log(x) - x - lgamma(p)
        a = 1.0 - p
        b = a + x + 1.0
        c = 0.0
        pn1, pn2, pn3, pn4 = 1.0, x, x + 1.0, x * b
        dpn1, dpn2, dpn3, dpn4 = 0.0, 0.0, 0.0, -x
        value, grad = pn3 / pn4, 0.0
        while True:
            a += 1.0
            b += 2.0
            c += 1.0
            an = a * c
            pn5 = b * pn3 - an * pn1
            pn6 = b * pn4 - an * pn2
            dpn5 = b * dpn3 - pn3 - an * dpn1 + c * pn1
            dpn6 = b * dpn4 - pn4 - an * dpn2 + c * pn2
            if pn6 != 0.0:
                rn = pn5 / pn6
                grad = (dpn5 * pn6 - pn5 * dpn6) / ipow(pn6, 2)
                if abs(value - rn) <= min(tol, tol * rn):
                    break
                value = rn
            pn1, pn2, pn3, pn4 = pn3, pn4, pn5, pn6
            dpn1, dpn2, dpn3, dpn4 = dpn3, dpn4, dpn5, dpn6
            if oflo <= abs(pn5):
                pn1 /= oflo
                pn2 /= oflo
                pn3 /= oflo
                pn4 /= oflo
                dpn1 /= oflo
                dpn2 /= oflo
                dpn3 /= oflo
                dpn4 /= oflo
        darg = exp(arg) * (log(x) - digamma(p))
        grad = grad * exp(arg) + value * darg
        arg += log(value)
        if arg >= elimit:
            grad *= -1.0
            value = 1.0 - exp(arg)
        else:
            grad = 0.0
            value = 1.0
        return grad
