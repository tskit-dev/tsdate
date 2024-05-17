"""
Moments for EP updates using exact hypergeometric evaluations rather than
a Laplace approximation; intended for testing and accuracy benchmarking.
"""

import mpmath
import numpy as np
from scipy.special import betaln, gammaln
from math import log, exp


def moments(a_i, b_i, a_j, b_j, y_ij, mu_ij):
    """
    log p(t_i, t_j) := \
        log(t_i - t_j) * y_ij - mu_ij * (t_i - t_j) + \
        log(t_i) * (a_i - 1) - mu_ij * t_i + \
        log(t_j) * (a_j - 1) - mu_ij * t_j
    """
    a = a_j
    b = a_i + a_j + y_ij
    c = a_j + y_ij + 1
    t = mu_ij + b_i
    z = (mu_ij - b_j) / t
    f0 = float(mpmath.log(mpmath.hyp2f1(a + 0, b + 0, c + 0, z)))
    f1 = float(mpmath.log(mpmath.hyp2f1(a + 1, b + 1, c + 1, z)))
    f2 = float(mpmath.log(mpmath.hyp2f1(a + 2, b + 2, c + 2, z)))
    logl = f0 + betaln(y_ij + 1, a) + gammaln(b) - b * log(t)
    mn_j = d1 / t
    sq_j = d2 / t**2
    va_j = sq_j - mn_j**2
    mn_i = mn_j * z + b / t
    sq_i = sq_j * z**2 + (b + 1) * (mn_i + mn_j * z) / t
    va_i = sq_i - mn_i**2
    return logl, mn_i, va_i, mn_j, va_j


def rootward_moments(t_j, a_i, b_i, y_ij, mu_ij):
    r"""
    log p(t_i) := \
        log(t_i - t_j) * y_ij - mu_ij * (t_i - t_j) + \
        log(t_i) * (a_i - 1) - mu_ij * t_i 
    """
    assert t_j >= 0.0
    s = a_i + y_ij
    r = mu_ij + b_i
    a = y_ij + 1
    b = s + 1
    z = t_j * r
    if t_j == 0.0:
        logl = gammaln(s) - s * log(r)
        mn_i = s / r
        va_i = s / r**2
        return logl, mn_i, va_i
    f0 = float(mpmath.log(mpmath.hyperu(a + 0, b + 0, z)))
    f1 = float(mpmath.log(mpmath.hyperu(a + 1, b + 1, z)))
    f2 = float(mpmath.log(mpmath.hyperu(a + 2, b + 2, z)))
    d0 = -a * f1 / f0
    d1 = -(a + 1) * f2 / f1
    logl = f0 - b_i * t_j + (b - 1) * log(t_j) + gammaln(a)
    mn_i = t_j * (1 - d0)
    va_i = t_j**2 * d0 * (d1 - d0)
    return logl, mn_i, va_i


def leafward_moments(t_i, a_j, b_j, y_ij, mu_ij):
    r"""
    log p(t_j) := \
        log(t_i - t_j) * y_ij - mu_ij * (t_i - t_j) + \
        log(t_j) * (a_j - 1) - mu_ij * t_j
    """
    assert t_i > 0.0
    a = a_j
    b = a_j + y_ij + 1
    z = t_i * (mu_ij - b_j)
    f0 = float(mpmath.log(mpmath.hyp1f1(a + 0, b + 0, z)))
    f1 = float(mpmath.log(mpmath.hyp1f1(a + 1, b + 1, z)))
    f2 = float(mpmath.log(mpmath.hyp1f1(a + 2, b + 2, z)))
    d1 = a / b * exp(f1 - f0)
    d2 = a / b * (a + 1) / (b + 1) * exp(f2 - f0)
    logl = f0 - mu_ij * t_i + (b - 1) * log(t_i) + betaln(a, b - a)
    mn_j = t_i * d1
    sq_j = t_i**2 * d2
    va_j = sq_j - mn_j**2
    return logl, mn_j, va_j


def unphased_moments(a_i, b_i, a_j, b_j, y_ij, mu_ij):
    r"""
    log p(t_i, t_j) := \
        log(t_i + t_j) * y_ij - mu_ij * (t_i + t_j) + \
        log(t_i) * (a_i - 1) - mu_ij * t_i + \
        log(t_j) * (a_j - 1) - mu_ij * t_j
    """
    a = a_j
    b = a_i + a_j + y_ij
    c = a_j + a_i
    t = mu_ij + b_i
    z = (mu_ij + b_j) / t
    f0 = float(mpmath.log(mpmath.hyp2f1(a + 0, b + 0, c + 0, 1 - z)))
    f1 = float(mpmath.log(mpmath.hyp2f1(a + 1, b + 1, c + 1, 1 - z)))
    f2 = float(mpmath.log(mpmath.hyp2f1(a + 2, b + 2, c + 2, 1 - z)))
    s1 = a * b / c
    s2 = s1 * (a + 1) * (b + 1) / (c + 1)
    d1 = s1 * exp(f1 - f0)
    d2 = s2 * exp(f2 - f0)
    logl = f0 + _betaln(a_j, a_i) + _gammaln(b) - b * log(t)
    mn_j = d1 / t
    sq_j = d2 / t**2
    va_j = sq_j - mn_j**2
    mn_i = b / t - mn_j * z
    sq_i = sq_j * z**2 + (b + 1) * (mn_i - mn_j * z) / t
    va_i = sq_i - mn_i**2
    return logl, mn_i, va_i, mn_j, va_j


def unphased_rightward_moments(t_i, a_j, b_j, y_ij, mu_ij):
    r"""
    log p(t_i, t_j) := \
        log(t_i + t_j) * y_ij - mu_ij * (t_i + t_j) + \
        log(t_i) * (a_i - 1) - mu_ij * t_i + \
        log(t_j) * (a_j - 1) - mu_ij * t_j
    """
    assert t_i > 0.0
    a = a_j
    b = a_j + y_ij + 1
    z = t_i * (mu_ij + b_j)
    f0, d0 = float(mpmath.log(mpmath.hyperu(a + 0, b + 0, z)))
    f1, d1 = float(mpmath.log(mpmath.hyperu(a + 1, b + 1, z)))
    logl = f0 - mu_ij * t_i + (b - 1) * log(t_i) + gammaln(a)
    mn_j = -t_i * d0
    va_j = t_i**2 * d0 * (d1 - d0)
    return logl, mn_j, va_j


def mutation_moments(a_i, b_i, a_j, b_j, y_ij, mu_ij):
    r"""
    log p(t_m, t_i, t_j) := \
        log(t_i - t_j) * y_ij - mu_ij * (t_i - t_j) + \
        log(t_i) * (a_i - 1) - mu_ij * t_i + \
        log(t_j) * (a_j - 1) - mu_ij * t_j - \
        log(t_i - t_j) * int(t_j < t_m < t_i)
    """
    a = a_j
    b = a_i + a_j + y_ij
    c = a_j + y_ij + 1
    t = mu_ij + b_i
    z = (mu_ij - b_j) / t
    f000 = float(mpmath.log(mpmath.hyp2f1(a + 0, b + 0, c + 0, z)))
    f020 = float(mpmath.log(mpmath.hyp2f1(a + 0, b + 2, c + 0, z)))
    f111 = float(mpmath.log(mpmath.hyp2f1(a + 1, b + 1, c + 1, z)))
    f121 = float(mpmath.log(mpmath.hyp2f1(a + 1, b + 2, c + 1, z)))
    f222 = float(mpmath.log(mpmath.hyp2f1(a + 2, b + 2, c + 2, z)))
    s1 = a * b / c
    s2 = s1 * (a + 1) * (b + 1) / (c + 1)
    d1 = b * (b + 1) / t ** 2
    d2 = d1 * a / c
    d3 = d2 * (a + 1) / (c + 1)
    mn_m = s1 * exp(f111 - f000) / t / 2 * (1 + z) + b / t / 2
    sq_m = d1 * exp(f020 - f000) / 3 + d2 * exp(f121 - f000) / 3 + d3 * exp(f222 - f000) / 3
    va_m = sq_m - mn_m**2
    return mn_m, va_m


def mutation_rootward_moments(t_j, a_i, b_i, y_ij, mu_ij):
    r"""
    log p(t_m, t_i) := \
        log(t_i - t_j) * y_ij - mu_ij * (t_i - t_j) + \
        log(t_i) * (a_i - 1) - mu_ij * t_i + \
        log(t_i - t_j) * int(t_j < t_m < t_i)
    """
    logl, mn_i, va_i = rootward_moments(t_j, a_i, b_i, y_ij, mu_ij)
    mn_m = mn_i / 2 + t_j / 2
    sq_m = (va_i + mn_i**2 + mn_i * t_j + t_j**2) / 3
    va_m = sq_m - mn_m**2
    return mn_m, va_m


def mutation_leafward_moments(t_i, a_j, b_j, y_ij, mu_ij):
    r"""
    log p(t_m, t_j) := \
        log(t_i - t_j) * y_ij - mu_ij * (t_i - t_j) + \
        log(t_j) * (a_j - 1) - mu_ij * t_j - \
        log(t_i - t_j) * int(t_j < t_m < t_i)
    """
    logl, mn_j, va_j = leafward_moments(t_i, a_j, b_j, y_ij, mu_ij)
    mn_m = mn_j / 2 + t_i / 2
    sq_m = (va_j + mn_j**2 + mn_j * t_i + t_i**2) / 3
    va_m = sq_m - mn_m**2
    return mn_m, va_m
