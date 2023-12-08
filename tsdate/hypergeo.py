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

import numba
import numpy as np
from numba.extending import get_cython_function_address

_HYP2F1_TOL = 1e-10
_HYP2F1_MAXTERM = int(1e6)

_PTR = ctypes.POINTER
_dbl = ctypes.c_double
_ptr_dbl = _PTR(_dbl)
_gammaln_addr = get_cython_function_address("scipy.special.cython_special", "gammaln")
_gammaln_functype = ctypes.CFUNCTYPE(_dbl, _dbl)
_gammaln_f8 = _gammaln_functype(_gammaln_addr)


class Invalid2F1(Exception):
    pass


@numba.njit("f8(f8)")
def _gammaln(x):
    return _gammaln_f8(x)


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


@numba.njit("b1(f8, f8, f8, f8, f8, f8, f8)")
def _is_valid_2f1(f1, f2, a, b, c, z, tol):
    """
    Use the contiguous relation between the Gauss hypergeometric function and
    its first and second derivatives to check its numerical accuracy. The first
    and second derivatives are assumed to be normalized by the function value.

    See Eq. 6 in https://doi.org/10.1016/j.cpc.2007.11.007
    """
    if z == 0.0:
        return np.abs(f1 - a * b / c) < tol
    u = c - (a + b + 1) * z
    v = a * b
    w = z * (1 - z)
    denom = np.abs(f1) + np.abs(f2) + 1.0
    if z == 1.0:
        numer = np.abs(u * f1 - v)
    else:
        numer = np.abs(f2 + u / w * f1 - v / w)
    return numer / denom < tol


@numba.njit("UniTuple(f8, 5)(f8, f8, f8, f8)")
def _hyp2f1_taylor_series(a, b, c, z):
    """
    Evaluate a Gaussian hypergeometric function, via its Taylor series at the
    origin.  Also returns the gradient with regard to `a`, `b`, `c`, and `z`.
    Requires that :math:`1 > z >= 0`, :math:`a,b,c >= 0`.

    To avoid overflow, returns the function value on a log scale and the
    derivatives divided by the (unlogged) function value.
    """
    assert 1.0 > z >= 0.0
    if not (a >= 0.0 and b >= 0.0 and c > 0.0):
        raise Invalid2F1("Negative parameters in Taylor series")

    if z == 0.0:
        val = 0.0
        da = 0.0
        db = 0.0
        dc = 0.0
        dz = a * b / c
        d2z = dz * (a + 1) * (b + 1) / (c + 1)
    else:
        k = 1
        val = 1.0
        ltol = np.log(_HYP2F1_TOL)
        zk = 0.0  # multiplicative increment for parameter (log)
        iz = 0.0  # additive increment for parameter
        dz = 0.0  # partial derivative of parameter
        d2z = 0.0  # second order derivative of parameter
        arg = np.array([a, b, c])
        argk = np.zeros(3)  # multiplicative increment for args (log)
        argd = np.zeros(3)  # additive increment for args
        argp = np.zeros(3)  # partial derivatives of args
        offset = 0.0  # maximum increment encountered
        while k < _HYP2F1_MAXTERM:
            argk += np.log(arg + k - 1)
            argd += 1 / (arg + k - 1)
            zk += np.log(z) - np.log(k)
            iz += 1 / z
            weight = argk[0] + argk[1] - argk[2] + zk
            norm = np.exp(weight - offset)
            if weight < offset:
                val += norm
                argp += argd * norm
                dz += iz * norm
                d2z += iz * (iz - 1 / z) * norm
            else:
                val = 1.0 + val / norm
                argp = argd + argp / norm
                dz = iz + dz / norm
                d2z = iz * (iz - 1 / z) + d2z / norm
                offset = weight
            if not np.isfinite(val):
                raise Invalid2F1("Nonfinite function value in Taylor series")
            if weight < ltol + np.log(val) and _is_valid_2f1(
                dz / val, d2z / val, a, b, c, z, _HYP2F1_TOL
            ):
                break
            k += 1
        argp /= val
        da = argp[0]
        db = argp[1]
        dc = -argp[2]
        dz /= val
        d2z /= val
        val = np.log(val) + offset
        if k >= _HYP2F1_MAXTERM:
            raise Invalid2F1("Maximum terms reached in Taylor series")
    return val, da, db, dc, dz


@numba.njit("UniTuple(f8, 5)(f8, f8, f8, f8)")
def _hyp2f1_laplace_approx(a, b, c, x):
    """
    Approximate a Gaussian hypergeometric function, using Laplace's method
    as per Butler & Wood 2002 Annals of Statistics. Also returns the gradient
    with regard to `a`, `b`, `c`, and `x`. Requires that :math:`1 > x > 0`,
    :math:`c > a > 0` and :math:`b >= 0`.

    To avoid overflow, returns the function value on a log scale and the
    derivatives divided by the (unlogged) function value.
    """
    assert c > 0.0
    assert a >= 0.0
    assert b >= 0.0
    assert c >= a
    assert 1.0 > x >= 0.0

    if x == 0.0:
        return 0.0, 0.0, 0.0, 0.0, a * b / c

    # Equations 19, 24, 25 in Butler & Wood
    t = x * (b - a) - c
    u = np.sqrt(t**2 - 4 * a * x * (c - b)) - t
    y = 2 * a / u
    yy = y**2 / a
    my = (1 - y) ** 2 / (c - a)
    ymy = x**2 * b * yy * my / (1 - x * y) ** 2
    r = yy + my - ymy
    f = (
        (c - 1 / 2) * log(c)
        - log(r) / 2
        + a * (log(y) - log(a))
        + (c - a) * (log(1 - y) - log(c - a))
        - b * log(1 - x * y)
    )

    # Eq. 24, derivatives
    df_dr = -1 / (2 * r)
    df_da = log(y) - log(a) - log(1 - y) + log(c - a)
    df_db = -log(1 - x * y)
    df_dc = log(c) - 0.5 / c + log(1 - y) - log(c - a)
    df_dx = y * b / (1 - x * y)

    # Eq. 25, derivatives
    df_dy = (
        yy * 2 / y
        - my * 2 / (1 - y)
        - ymy * (2 / y - 2 / (1 - y) + 2 * x / (1 - x * y))
    ) * df_dr
    df_da += (my * a - yy * (c - a) + ymy * (c - 2 * a)) / (a * (c - a)) * df_dr
    df_db += -ymy / b * df_dr
    df_dc += (ymy - my) / (c - a) * df_dr
    df_dx += -ymy * (2 / x + 2 * y / (1 - x * y)) * df_dr

    # Eq. 19, derivatives
    uy = y**2 / (u + t)
    df_dt = uy / y * df_dy
    df_da += uy * (x * (c - b) + u / y + t / y) / a * df_dy
    df_db += -uy * x * df_dy
    df_dc += uy * x * df_dy
    df_dx += uy * (c - b) * df_dy

    # Finally,
    df_da += -x * df_dt
    df_db += x * df_dt
    df_dc += -df_dt
    df_dx += (b - a) * df_dt

    return f, df_da, df_db, df_dc, df_dx


@numba.njit("f8(f8, f8, f8, f8)")
def _hyp2f1_fast(a, b, c, x):
    """
    Approximate a Gaussian hypergeometric function, using Laplace's method
    as per Butler & Wood 2002 Annals of Statistics.

    Shortcut bypassing the lengthly derivative computation.
    """

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

    t = x * (b - a) - c
    u = np.sqrt(t**2 - 4 * a * x * (c - b)) - t
    y = 2 * a / u
    yy = y**2 / a
    my = (1 - y) ** 2 / (c - a)
    ymy = x**2 * b * yy * my / (1 - x * y) ** 2
    r = yy + my - ymy
    f = (
        +(c - 1 / 2) * log(c)
        - log(r) / 2
        + a * (log(y) - log(a))
        + (c - a) * (log(1 - y) - log(c - a))
        - b * log(1 - x * y)
    )

    return f + s


# @numba.njit("UniTuple(f8, 5)(f8, f8, f8, f8)")
# def _hyp2f1_laplace_recurrence(a, b, c, x):
#    """
#    Use contiguous relations to stabilize the calculation of 2F1
#    """
#
#    if x == 0.0:
#        return 0.0, 0.0, 0.0, 0.0, a * b / c
#
#    A = x * (1 - x) * (a + 1) * (b + 1) / (c * (c + 1))
#    dA_da = A / (a + 1)
#    dA_db = A / (b + 1)
#    dA_dc = -A / (c * (c + 1)) * (2 * c + 1)
#
#    B = 1 - (a + b + 1) * x / c
#    dB_da = -x / c
#    dB_db = -x / c
#    dB_dc = (a + b + 1) * x / c**2
#
#    Af, dAf_da, dAf_db, dAf_dc, _ = _hyp2f1_laplace_approx(a + 2, b + 2, c + 2, x)
#    Bf, dBf_da, dBf_db, dBf_dc, _ = _hyp2f1_laplace_approx(a + 1, b + 1, c + 1, x)
#    Mx = max(Af, Bf)
#    Cf = np.log(A * np.exp(Af - Mx) + B * np.exp(Bf - Mx)) + Mx
#
#    sA = np.exp(Af - Cf)
#    sB = np.exp(Bf - Cf)
#
#    v = Cf
#    da = sA * (dA_da + A * dAf_da) + sB * (dB_da + B * dBf_da)
#    db = sA * (dA_db + A * dAf_db) + sB * (dB_db + B * dBf_db)
#    dc = sA * (dA_dc + A * dAf_dc) + sB * (dB_dc + B * dBf_dc)
#    dx = sB * a * b / c
#
#    return v, da, db, dc, dx


@numba.njit("UniTuple(f8, 5)(f8, f8, f8, f8, f8, f8)")
def _hyp2f1_dlmf1581(a_i, b_i, a_j, b_j, y, mu):
    """
    DLMF 15.8.1, series expansion with Pfaff transformation
    """

    a = y + 1
    b = a_i + a_j + y
    c = a_j + y + 1
    z = (b_j - mu) / (b_i + b_j)
    s = (mu - b_j) / (mu + b_i)
    scale = -b * np.log(1 - s)

    # 2F1(y+1, b; c; z) via series expansion
    val, _, db, dc, dz = _hyp2f1_laplace_approx(a, b, c, z)

    # map gradient to parameters
    da_i = db - np.log(1 - s)
    da_j = db + dc - np.log(1 - s)
    db_i = z * (b / (mu + b_i) - dz / (b_i + b_j))
    db_j = (z - 1) * (b / (mu + b_i) - dz / (b_i + b_j))

    val += scale

    return val, da_i, db_i, da_j, db_j


@numba.njit("UniTuple(f8, 5)(f8, f8, f8, f8, f8, f8)")
def _hyp2f1_dlmf1521(a_i, b_i, a_j, b_j, y, mu):
    """
    DLMF 15.2.1, series expansion without transformation
    """
    a = a_j
    b = a_i + a_j + y
    c = a_j + y + 1
    z = (mu - b_j) / (mu + b_i)

    # 2F1(a, y+1; c; z) via series expansion
    val, da, db, dc, dz = _hyp2f1_laplace_approx(a, b, c, z)

    # map gradient to parameters
    da_i = db
    da_j = da + db + dc
    db_i = -dz * z / (mu + b_i)
    db_j = -dz / (mu + b_i)

    return val, da_i, db_i, da_j, db_j


@numba.njit("UniTuple(f8, 5)(f8, f8, f8, f8, f8, f8)")
def _hyp2f1(a_i, b_i, a_j, b_j, y, mu):
    """
    Evaluates:

    ..math:

        {2_}F{1_}(a_j, a_i+a_j+y; a_j+y+1; (\\mu-b_j)/(\\mu+b_i))

    where the age of parent :math:`i` and child :math:`j` have marginal shape
    and rate :math:`a` and :math:`b`; and edge :math:`(i, j)` has :math:`y`
    observed mutations with span-weighted mutation rate span :math:`\\mu`.

    Returns overflow-protected values of:

      \\log f, df/da_i, df/db_i, d2f/(da_i db_i), df/da_j, df/db_j, d2f/(da_j db_j)

    Overflow protection entails log-transforming the function value,
    and dividing the gradient by the function value.
    """
    z = (mu - b_j) / (mu + b_i)
    assert z < 1.0, "Invalid argument"
    if z > 0.0:
        return _hyp2f1_dlmf1521(a_i, b_i, a_j, b_j, y, mu)
    else:
        return _hyp2f1_dlmf1581(a_i, b_i, a_j, b_j, y, mu)
