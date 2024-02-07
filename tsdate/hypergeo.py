# MIT License
#
# Copyright (c) 2021-23 Tskit Developers
# Copyright (c) 2020-21 University of Oxford
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""
Numerically stable implementations of the Gauss hypergeometric function with numba.
"""
import ctypes
from math import log
from math import sqrt

import numba
import numpy as np
from numba.extending import get_cython_function_address

_HYP2F1_TOL = 1e-10
_HYP2F1_MAXTERM = int(1e6)


class Invalid2F1(Exception):
    pass


# --- numba bindings for scipy cython interface --- #

_dbl = ctypes.c_double

# gammaln
_gammaln_addr = get_cython_function_address("scipy.special.cython_special", "gammaln")
_gammaln_functype = ctypes.CFUNCTYPE(_dbl, _dbl)
_gammaln_f8 = _gammaln_functype(_gammaln_addr)

# gammainc
_gammainc_addr = get_cython_function_address("scipy.special.cython_special", "gammainc")
_gammainc_functype = ctypes.CFUNCTYPE(_dbl, _dbl, _dbl)
_gammainc_f8 = _gammainc_functype(_gammainc_addr)


@numba.cfunc("f8(f8)")
def _gammaln(x):
    """scipy.special.cython_special.gammaln"""
    return _gammaln_f8(x)


@numba.cfunc("f8(f8, f8)")
def _gammainc(a, x):
    """scipy.special.cython_special.gammainc"""
    return _gammainc_f8(a, x)


@numba.njit("f8(f8)")
def _digamma(x):
    """
    Digamma (psi) function, from asymptotic series expansion.
    """
    if x <= 0.0:
        return _digamma(1 - x) - np.pi / np.tan(np.pi * x)
    if x <= 1e-5:
        return -np.euler_gamma - (1 / x)
    if x < 8.5:
        return _digamma(1 + x) - 1 / x
    xpm2 = 1 / x**2
    return (
        np.log(x)
        - 0.5 / x
        - 0.083333333333333333 * xpm2
        + 0.008333333333333333 * xpm2**2
        - 0.003968253968253968 * xpm2**3
        + 0.004166666666666667 * xpm2**4
        - 0.007575757575757576 * xpm2**5
        + 0.021092796092796094 * xpm2**6
    )


@numba.njit("f8(f8)")
def _trigamma(x):
    """
    Trigamma function, from asymptotic series expansion
    """
    if x <= 0.0:
        return -_trigamma(1 - x) + np.pi**2 / np.sin(np.pi * x) ** 2
    if x <= 1e-4:
        return 1 / x**2
    if x < 5:
        return _trigamma(1 + x) + 1 / x**2
    xpm1 = 1 / x
    xpm2 = 1 / x**2
    return xpm1 * (
        1.000000000000000000
        + 0.500000000000000000 * xpm1
        + 0.166666666666666667 * np.power(xpm2, 1)
        - 0.033333333333333333 * np.power(xpm2, 2)
        + 0.023809523809523808 * np.power(xpm2, 3)
        - 0.033333333333333333 * np.power(xpm2, 4)
        + 0.075757575757575756 * np.power(xpm2, 5)
        - 0.253113553113553102 * np.power(xpm2, 6)
        + 1.166666666666666741 * np.power(xpm2, 7)
    )


@numba.njit("f8(f8, f8)")
def _betaln(p, q):
    return _gammaln(p) + _gammaln(q) - _gammaln(p + q)


@numba.njit("UniTuple(f8, 2)(f8, f8, f8)")
def _hyperu_laplace(a, b, x):
    """
    Approximate Tricomi's confluent hypergeometric function with real
    arguments, using Laplace's method with a calibration point.

    Also returns derivative wrt x.

    TODO: details
    """

    assert b > a > 0.0
    assert x > 0.0

    t = b - x - 1
    u = 2 * a / (sqrt(t**2 + 4 * x * a) - t)
    r = (b - a - 1) * u**2 + a * (1 + u) ** 2
    g = (b - a) * log(1 + u) + a * (1 + log(u)) - x * u - log(a) * (a - 1 / 2)

    w = (b - 1) * u + a
    v = u * (1 + u)
    dr = w / (w * u + a * (1 + u))
    dg = (w - x * v + u) / v
    du = -u * v / (w - x * u + a)

    return g - log(r) / 2, (dg - dr) * du - u


@numba.njit("f8(f8, f8, f8)")
def _hyp1f1_laplace(a, b, x):
    """
    Approximate Kummer's confluent hypergeometric function with real arguments,
    using Laplace's method with a calibration point.

    TODO: details
    """

    assert b > a > 0.0

    if x == 0.0:
        return 0.0

    t = x - b
    u = 2 * a / (sqrt(t**2 + 4 * x * a) - t)
    r = u**2 * b / a + (1 - u) ** 2 * b / (b - a)
    g = (
        (b - a) * log(1 - u)
        + a * log(u)
        + b * log(b)
        - (b - a) * log(b - a)
        - a * log(a)
        + x * u
    )

    return g - log(r) / 2


@numba.njit("f8(f8, f8, f8, f8)")
def _hyp2f1_unity(a, b, c, x):
    """
    Gauss hypergeometric function when `x` is near unity

    See limits in DLMF 15.4.

    TODO: this works in practice, but when (c - a - b) is close to zero the
    limits don't converge. A good reference is Buhring 2003 "Partial sums of
    hypergeometric series of unit argument"
    """
    assert np.isclose(x, 1.0) and x < 1.0

    g = c - a - b

    if g < 0.0:
        return _gammaln(c) + _gammaln(-g) - _gammaln(a) - _gammaln(b) + g * log(1 - x)
    elif g > 0.0:
        # will only occur when a_i + a_j < 1
        return _gammaln(c) + _gammaln(g) - _gammaln(c - a) - _gammaln(c - b)
    else:
        # will only occur when a_i + a_j == 1
        return log(-log(1 - x)) + _gammaln(a + b) - _gammaln(a) - _gammaln(b)


@numba.njit("f8(f8, f8, f8, f8)")
def _hyp2f1_laplace(a, b, c, x):
    r"""
    Approximate a Gaussian hypergeometric function with real arguments,
    using Laplace's method with a calibration point.

    TODO: details
    """

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
    u = sqrt(t**2 - 4 * a * x * (c - b)) - t
    y = 2 * a / u

    assert 0 < y < 1, "Root on boundary"

    yy = y**2 / a
    my = (1 - y) ** 2 / (c - a)
    ymy = x**2 * b * yy * my / (1 - x * y) ** 2
    r = yy + my - ymy
    f = (
        (c - 1 / 2) * log(c)
        + a * (log(y) - log(a))
        + (c - a) * (log(1 - y) - log(c - a))
        - b * log(1 - x * y)
    )

    return f - log(r) / 2 + s
