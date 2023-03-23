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

import numba
import numpy as np
from numba.extending import get_cython_function_address

_HYP2F1_TOL = np.sqrt(np.finfo(np.float64).eps)
_HYP2F1_CHECK = np.sqrt(_HYP2F1_TOL)
_HYP2F1_MAXTERM = int(1e6)

_PTR = ctypes.POINTER
_dbl = ctypes.c_double
_ptr_dbl = _PTR(_dbl)
_gammaln_addr = get_cython_function_address("scipy.special.cython_special", "gammaln")
_gammaln_functype = ctypes.CFUNCTYPE(_dbl, _dbl)
_gammaln_float64 = _gammaln_functype(_gammaln_addr)


class Invalid2F1(Exception):
    pass


@numba.njit("float64(float64)")
def _gammaln(x):
    return _gammaln_float64(x)


@numba.njit("float64(float64)")
def _digamma(x):
    """
    digamma (psi) function, from asymptotic series expansion
    """
    if x <= 0.0:
        return np.nan
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


@numba.njit("float64(float64)")
def _trigamma(x):
    """
    trigamma function, from asymptotic series expansion
    """
    if x <= 0.0:
        return np.nan
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


@numba.njit("float64(float64, float64)")
def _betaln(p, q):
    return _gammaln(p) + _gammaln(q) - _gammaln(p + q)


@numba.njit("boolean(float64, float64, float64, float64, float64, float64)")
def _is_valid_2f1(f1, f2, a, b, c, z):
    """
    Use the contiguous relation between the Gauss hypergeometric function and
    its first and second derivatives to check its numerical accuracy. The first
    and second derivatives are assumed to be normalized by the function value.

    See Eq. 6 in https://doi.org/10.1016/j.cpc.2007.11.007
    """
    if z == 0.0:
        return np.abs(f1 - a * b / c) < _HYP2F1_CHECK
    u = c - (a + b + 1) * z
    v = a * b
    w = z * (1 - z)
    denom = np.abs(f1) + np.abs(f2) + 1.0
    if z == 1.0:
        numer = np.abs(u * f1 - v)
    else:
        numer = np.abs(f2 + u / w * f1 - v / w)
    return numer / denom < _HYP2F1_CHECK


@numba.njit("UniTuple(float64, 7)(float64, float64, float64, float64)")
def _hyp2f1_taylor_series(a, b, c, z):
    """
    Evaluate a Gaussian hypergeometric function, via its Taylor series at the
    origin.  Also returns the gradient with regard to `a`, `b`, and `c`; and
    first and second partial derivatives wrt `z`. Requires :math:`1 > z >= 0`.

    To avoid overflow, returns the function value on a log scale and the
    derivatives divided by the (unlogged) function value.
    """
    assert 1.0 > z >= 0.0
    assert a >= 0.0
    assert b >= 0.0
    assert c > 0.0

    if z == 0.0:
        sign = 1.0
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
                raise Invalid2F1("Hypergeometric series did not converge")
            if weight < ltol + np.log(val) and _is_valid_2f1(
                dz / val, d2z / val, a, b, c, z
            ):
                break
            k += 1
        argp /= val
        da = argp[0]
        db = argp[1]
        dc = -argp[2]
        dz /= val
        d2z /= val
        sign = 1.0
        val = np.log(val) + offset
    if k >= _HYP2F1_MAXTERM:
        raise Invalid2F1("Hypergeometric series did not converge")
    return val, sign, da, db, dc, dz, d2z


@numba.njit("UniTuple(float64, 7)(float64, float64, float64, float64)")
def _hyp2f1_recurrence(a, b, c, z):
    """
    Evaluate 2F1(a, -b; c; z) where b is an integer using (0, -1, 0) recurrence.
    See https://doi.org/10.48550/arXiv.1909.10765

    Returns log function value, sign, gradient, and second derivative wrt z. The
    derivatives are divided by the function value.
    """
    # TODO
    # fails with (200.0, 101.0, 401.6, 1.1)
    assert b % 1.0 == 0.0 and b >= 0
    assert np.abs(c) >= np.abs(a)
    assert 2.0 > z > 1.0  # TODO: generalize
    f0 = 1.0
    f1 = 1 - a * z / c
    s0 = 1.0
    s1 = np.sign(f1)
    g0 = np.zeros(4)  # df/da df/db df/dc df/dz
    g1 = np.array([-z / c, 0.0, a * z / c**2, -a / c]) / f1
    p0 = 0.0  # d2f/dz2
    p1 = 0.0
    f0 = np.log(np.abs(f0))
    f1 = np.log(np.abs(f1))
    if b == 0:
        return f0, s0, g0[0], g0[1], g0[2], g0[3], p0
    if b == 1:
        return f1, s1, g1[0], g1[1], g1[2], g1[3], p1
    for n in range(1, int(b)):
        ak = n * (z - 1) / (c + n)
        dak = np.array([0.0, 0.0, -ak / (c + n), ak / (z - 1)])
        bk = (2 * n + c - z * (a + n)) / (c + n)
        dbk = np.array([-z / (c + n), 0.0, (1 - bk) / (c + n), -(a + n) / (c + n)])
        u = s0 * np.exp(f0 - f1)
        v = s1 * bk + u * ak
        s = np.sign(v)
        f = np.log(np.abs(v)) + f1
        g = (g1 * bk * s1 + g0 * u * ak + dbk * s1 + dak * u) / v
        p = (
            p1 * bk * s1
            + p0 * u * ak
            + 2 / (c + n) * (u * g0[3] * n - s1 * g1[3] * (a + n))
        ) / v
        f1, f0 = f, f1
        s1, s0 = s, s1
        g1, g0 = g, g1
        p1, p0 = p, p1
    if not _is_valid_2f1(g[3], p, a, -b, c, z):
        raise Invalid2F1("Hypergeometric series did not converge")
    da, db, dc, dz = g
    return f, s, da, db, dc, dz, p


@numba.njit(
    "UniTuple(float64, 6)(float64, float64, float64, float64, float64, float64)"
)
def _hyp2f1_dlmf1583_first(a_i, b_i, a_j, b_j, y, mu):
    """
    DLMF 15.8.3, first term

    F(a_j, 1 - a_i; 1 - a_i - y; 1/(1 - z))

    Use Pfaff transform,

    (1 - 1 / (1 - z))**(-a_j) F(a_j, -y; 1 - a_i - y; 1 / z)

    Then rearrange terms in series to get,

    (1 - a_i - a_j - y)_y / (1 - a_i - y)_y \\times
      (1 - 1/(1-z))^(-a_j) \\times
        F(a_j, -y; a_j + a_i; 1 - 1/z)
    """

    a = a_j
    c = a_j + a_i
    s = (b_j - mu) / (mu + b_i)
    z = (b_j + b_i) / (b_j - mu)
    scale = (
        -a_j * np.log(s)
        + _gammaln(a_j + y + 1)
        - _gammaln(y + 1)
        + _gammaln(a_i)
        - _gammaln(a_i + a_j)
    )

    # 2F1(a, -y; c; z) via backwards recurrence
    val, sign, da, _, dc, dz, _ = _hyp2f1_recurrence(a, y, c, z)

    # map gradient to parameters
    da_i = dc - _digamma(a_i + a_j) + _digamma(a_i)
    da_j = da + dc - np.log(s) + _digamma(a_j + y + 1) - _digamma(a_i + a_j)
    db_i = dz / (b_j - mu) + a_j / (mu + b_i)
    db_j = dz * (1 - z) / (b_j - mu) - a_j / s / (mu + b_i)

    val += scale

    return val, sign, da_i, db_i, da_j, db_j


@numba.njit(
    "UniTuple(float64, 6)(float64, float64, float64, float64, float64, float64)"
)
def _hyp2f1_dlmf1583_second(a_i, b_i, a_j, b_j, y, mu):
    """
    DLMF 15.8.3, second term
    """
    y = int(y)
    a = a_i + a_j + y
    c = a_i + y + 1
    z = (b_i + mu) / (b_i + b_j)
    scale = (
        _gammaln(a_j + y + 1)
        - _gammaln(a_j)
        + _gammaln(a_i)
        - _gammaln(a_i + y + 1)
        + (a_i + a_j + y) * np.log(z)
    )

    # 2F1(a, y+1; c; z) via series expansion
    val, sign, da, _, dc, dz, _ = _hyp2f1_taylor_series(a, y + 1, c, z)

    # map gradient to parameters
    da_i = da + np.log(z) + dc + _digamma(a_i) - _digamma(a_i + y + 1)
    da_j = da + np.log(z) + _digamma(a_j + y + 1) - _digamma(a_j)
    db_i = (1 - z) * (dz + a / z) / (b_i + b_j)
    db_j = -z * (dz + a / z) / (b_i + b_j)

    sign *= (-1) ** (y + 1)
    val += scale

    return val, sign, da_i, db_i, da_j, db_j


@numba.njit(
    "UniTuple(float64, 6)(float64, float64, float64, float64, float64, float64)"
)
def _hyp2f1_dlmf1583(a_i, b_i, a_j, b_j, y, mu):
    """
    DLMF 15.8.3, sum of recurrence and series expansion
    """
    assert b_i > 0
    assert 0 < mu <= b_j
    assert y >= 0 and y % 1.0 == 0.0

    f_1, s_1, da_i_1, db_i_1, da_j_1, db_j_1 = _hyp2f1_dlmf1583_first(
        a_i, b_i, a_j, b_j, y, mu
    )

    f_2, s_2, da_i_2, db_i_2, da_j_2, db_j_2 = _hyp2f1_dlmf1583_second(
        a_i, b_i, a_j, b_j, y, mu
    )

    if np.abs(f_1 - f_2) < _HYP2F1_TOL:
        # TODO: detect a priori if this will occur
        raise Invalid2F1("Singular hypergeometric function")

    f_0 = max(f_1, f_2)
    f_1 = np.exp(f_1 - f_0) * s_1
    f_2 = np.exp(f_2 - f_0) * s_2
    f = f_1 + f_2

    da_i = (da_i_1 * f_1 + da_i_2 * f_2) / f
    db_i = (db_i_1 * f_1 + db_i_2 * f_2) / f
    da_j = (da_j_1 * f_1 + da_j_2 * f_2) / f
    db_j = (db_j_1 * f_1 + db_j_2 * f_2) / f

    sign = np.sign(f)
    val = np.log(np.abs(f)) + f_0

    return val, sign, da_i, db_i, da_j, db_j


@numba.njit(
    "UniTuple(float64, 6)(float64, float64, float64, float64, float64, float64)"
)
def _hyp2f1_dlmf1521(a_i, b_i, a_j, b_j, y, mu):
    """
    DLMF 15.2.1, series expansion without transformation
    """
    assert b_i > 0
    assert mu >= b_j > 0
    assert y >= 0 and y % 1 == 0.0

    y = int(y)
    a = a_j
    b = a_i + a_j + y
    c = a_j + y + 1
    z = (mu - b_j) / (mu + b_i)

    # 2F1(a, y+1; c; z) via series expansion
    val, sign, da, db, dc, dz, _ = _hyp2f1_taylor_series(a, b, c, z)

    # map gradient to parameters
    da_i = db
    da_j = da + db + dc
    db_i = -dz * z / (mu + b_i)
    db_j = -dz / (mu + b_i)

    return val, sign, da_i, db_i, da_j, db_j


@numba.njit(
    "UniTuple(float64, 6)(float64, float64, float64, float64, float64, float64)"
)
def _hyp2f1_dlmf1581(a_i, b_i, a_j, b_j, y, mu):
    """
    DLMF 15.8.1, series expansion with Pfaff transformation
    """
    assert b_i > 0
    assert 0 < mu <= b_j
    assert y >= 0 and y % 1 == 0.0

    y = int(y)
    a = a_i + a_j + y
    c = a_j + y + 1
    z = (b_j - mu) / (b_i + b_j)
    scale = -a * np.log(1 - z / (z - 1))

    # 2F1(a, y+1; c; z) via series expansion
    val, sign, da, _, dc, dz, _ = _hyp2f1_taylor_series(a, y + 1, c, z)

    # map gradient to parameters
    da_i = da - np.log(1 - z / (z - 1))
    da_j = da + dc - np.log(1 - z / (z - 1))
    db_i = z * (a / (mu + b_i) - dz / (b_i + b_j))
    db_j = (z - 1) * (a / (mu + b_i) - dz / (b_i + b_j))

    val += scale

    return val, sign, da_i, db_i, da_j, db_j


@numba.njit(
    "UniTuple(float64, 6)(float64, float64, float64, float64, float64, float64)"
)
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
    assert z < 1.0  # TODO: allow z == 1.0 for improper prior
    if 0.0 <= z < 1.0:
        return _hyp2f1_dlmf1521(a_i, b_i, a_j, b_j, y, mu)
    elif -1.0 < z < 0.0:
        return _hyp2f1_dlmf1581(a_i, b_i, a_j, b_j, y, mu)
    else:
        try:
            # if posterior marginals are too incompatible (e.g. parent age
            # younger than child age) then this transform will diverge
            return _hyp2f1_dlmf1583(a_i, b_i, a_j, b_j, y, mu)
        except:  # noqa: E722,B001
            return _hyp2f1_dlmf1581(a_i, b_i, a_j, b_j, y, mu)
