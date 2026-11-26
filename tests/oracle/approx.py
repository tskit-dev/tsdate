"""
Oracles for the gamma approximations (``lib/approx.c``).
"""

from math import inf, nan

import numpy as np

from . import hypergeo

# Plain `math` raises on log(0), exp overflow etc., where C gives -inf, inf or
# nan; `_libm` gives the C results.
# C computes `x ** n` for integer constants `n` by repeated squaring, whereas
# CPython calls libm pow, which differs by an ulp in a few percent of cases; `ipow`
# squares (see `_libm.ipow`).
# `np.log` of a Python float returns an `np.float64`, so that a following
# division by zero gives inf (with a warning) instead of the ZeroDivisionError that
# stands for the library's TSD_ERR_ZERO_DIVISION; and NumPy's SIMD log may differ
# from libm by an ulp. Logs are the `_libm.log` of a Python float.
# For the same reason, elements unpacked from a NumPy array are converted with
# `float`.
from ._libm import exp, ipow, lgamma, log

# the tolerances of the KL minimisation, as in the library
_KLMIN_MAXITT = 100
_KLMIN_RELTOL = np.sqrt(np.finfo(np.float64).eps)


class KLMinimizationFailedError(Exception):
    pass


def approximate_log_moments(mean, variance):
    """
    Approximate log moments via a second-order Taylor series expansion around
    the mean, e.g.:

      E[f(x)] \\approx E[f(mean)] + variance * f''(mean)/2

    Returns approximations to E[log x], E[x log x], E[(log x)^2]
    """
    assert mean > 0
    assert variance > 0
    logx = log(mean) - 0.5 * variance / ipow(mean, 2)
    xlogx = mean * log(mean) + 0.5 * variance / mean
    logx2 = ipow(log(mean), 2) + (1 - log(mean)) * variance / ipow(mean, 2)
    return logx, xlogx, logx2


def approximate_gamma_kl(x, logx):
    """
    Use Newton root finding to get gamma natural parameters matching the sufficient
    statistics :math:`E[x]` and :math:`E[\\log x]`, minimizing KL divergence.

    The initial condition uses the upper bound :math:`digamma(x) \\leq log(x) - 1/2x`.

    Returns the shape and rate of the approximating gamma.
    """
    if x <= 0.0 or np.isinf(logx):
        raise KLMinimizationFailedError("Nonpositive or nonfinite moments")
    if not log(x) > logx:
        raise KLMinimizationFailedError(
            "log E[t] <= E[log t] violates Jensen's inequality"
        )
    alpha = 0.5 / (log(x) - logx)  # lower bound on alpha
    # asymptotically the lower bound becomes sharp
    if 1.0 / alpha < 1e-4:
        return alpha - 1.0, alpha / x
    itt = 0
    delta = np.inf
    # determine convergence when the change in alpha falls below
    # some small value (e.g. square root of machine precision)
    while np.abs(delta) > np.abs(alpha) * _KLMIN_RELTOL:
        if itt > _KLMIN_MAXITT:
            raise KLMinimizationFailedError(
                "Maximum iterations reached in KL minimization"
            )
        delta = hypergeo._digamma(alpha) - log(alpha) + log(x) - logx
        delta /= hypergeo._trigamma(alpha) - 1 / alpha
        alpha -= delta
        itt += 1
    if not np.isfinite(alpha) or alpha <= 0:
        raise KLMinimizationFailedError("Invalid shape parameter in KL minimization")
    return alpha - 1.0, alpha / x


def approximate_gamma_mom(mean, variance):
    """
    Use the method of moments to approximate a distribution with a gamma of the
    same mean and variance, returning natural parameters
    """
    if not (mean > 0.0 and variance > 0.0):
        raise KLMinimizationFailedError("Nonpositive central moments")
    shape = ipow(mean, 2) / variance
    rate = mean / variance
    return shape - 1.0, rate


def approximate_gamma_iqr(q1, q2, x1, x2, max_shape):
    """
    Find gamma natural parameters that match empirical quantiles
    """

    def upper_bound(q, x):  # cap shape parameter
        beta = hypergeo._gammainc_inv(max_shape, q) / x
        return max_shape - 1, beta

    if x2 == x1:
        return upper_bound(q1, x1)
    if not (q2 > q1 and x2 > x1):
        raise KLMinimizationFailedError("Quantiles must be sorted")
    alpha = log(q2 / q1) / log(x2 / x1)  # lower bound
    if alpha > max_shape:
        return upper_bound(q1, x1)

    # refine with newton iteration
    delta = inf
    itt = 0
    while abs(delta) > abs(alpha) * _KLMIN_RELTOL:
        if itt > _KLMIN_MAXITT:
            raise KLMinimizationFailedError(
                "Maximum iterations reached in quantile matching"
            )
        y1 = hypergeo._gammainc_inv(alpha, q1)
        y2 = hypergeo._gammainc_inv(alpha, q2)
        obj = y2 / y1 - x2 / x1
        inv_1 = -exp(y1 + log(y1) * (1 - alpha) + lgamma(alpha))
        inv_2 = -exp(y2 + log(y2) * (1 - alpha) + lgamma(alpha))
        # print(itt, alpha, y1, y2) #DEBUG
        gra_1 = hypergeo._gammainc_der(alpha, y1) * inv_1
        gra_2 = hypergeo._gammainc_der(alpha, y2) * inv_2
        gra = (gra_2 * y1 - gra_1 * y2) / ipow(y1, 2)
        delta = -obj / gra
        alpha += delta
        itt += 1

    if not alpha > 0:
        raise KLMinimizationFailedError("Negative shape parameter")
    if alpha > max_shape:
        return upper_bound(q1, x1)
    beta = hypergeo._gammainc_inv(alpha, q1) / x1
    return alpha - 1, beta


def average_gammas(alpha, beta):
    """
    Given natural parameters for a set of gammas, average sufficient
    statistics so as to get a "global" gamma, returning natural
    parameters
    """
    assert alpha.size == beta.size, "Array sizes are not equal"
    avg_x = 0.0
    avg_logx = 0.0
    for shape, rate in zip(map(float, alpha + 1.0), map(float, beta)):
        avg_logx += hypergeo._digamma(shape) - log(rate)
        avg_x += shape / rate
    avg_x /= alpha.size
    avg_logx /= alpha.size
    return approximate_gamma_kl(avg_x, avg_logx)


def _valid_moments(mn, va):
    """
    Whether a mean and variance are finite and positive.
    """
    if not (np.isfinite(mn) and np.isfinite(va)):
        return False
    if not (mn > 0.0 and va > 0.0):
        return False
    return True


def _valid_gamma(s, r):
    """
    Whether a gamma shape and rate are finite and positive.
    """
    if not (np.isfinite(s) and np.isfinite(r)):
        return False
    if s <= 0.0 or r <= 0.0:
        return False
    return True


def _valid_hyp1f1(a, b, z):
    """
    Whether the arguments are in the domain of `_hyp1f1_laplace`.
    """
    if not (np.isfinite(a) and np.isfinite(b) and np.isfinite(z)):
        return False
    if not (b >= a > 0.0):
        return False
    return True


def _valid_hyperu(a, b, z):
    """
    Whether the arguments are in the domain of `_hyperu_laplace`.
    """
    if not (np.isfinite(a) and np.isfinite(b) and np.isfinite(z)):
        return False
    if z <= 0.0:
        return False
    if not (b > a > 0.0):
        return False
    return True


def _valid_hyp2f1(a, b, c, z):
    """
    Whether the arguments are in the domain of `_hyp2f1_laplace`.
    """
    if not (np.isfinite(a) and np.isfinite(b) and np.isfinite(c)):
        return False
    if not np.isfinite(z) or z >= 1 or z / (z - 1) >= 1:
        return False
    if not (a > 0 and b > 0 and c > 0):
        return False
    return True


def moments(a_i, b_i, a_j, b_j, y_ij, mu_ij):
    r"""
    log p(t_i, t_j) := \
        log(t_i - t_j) * y_ij - mu_ij * (t_i - t_j) + \
        log(t_i) * (a_i - 1) - b_i * t_i + \
        log(t_j) * (a_j - 1) - b_j * t_j

    Returns normalizing constant, E[t_i], V[t_i], E[t_j], V[t_j].
    """

    a = a_j
    b = a_i + a_j + y_ij
    c = a_j + y_ij + 1
    t = mu_ij + b_i
    z = (mu_ij - b_j) / t if t > 0 else nan

    if not _valid_hyp2f1(a, b, c, z):
        return nan, nan, nan, nan, nan

    hyp2f1 = hypergeo._hyp2f1_laplace
    f0 = hyp2f1(a + 0, b + 0, c + 0, z)
    f1 = hyp2f1(a + 1, b + 1, c + 1, z)
    f2 = hyp2f1(a + 2, b + 2, c + 2, z)
    s1 = a * b / c
    s2 = s1 * (a + 1) * (b + 1) / (c + 1)
    d1 = s1 * exp(f1 - f0)
    d2 = s2 * exp(f2 - f0)

    logl = f0 + hypergeo._betaln(y_ij + 1, a) + lgamma(b) - b * log(t)

    mn_j = d1 / t
    sq_j = d2 / ipow(t, 2)
    va_j = sq_j - ipow(mn_j, 2)

    mn_i = mn_j * z + b / t
    sq_i = sq_j * ipow(z, 2) + (b + 1) * (mn_i + mn_j * z) / t
    va_i = sq_i - ipow(mn_i, 2)

    return logl, mn_i, va_i, mn_j, va_j


def rootward_moments(t_j, a_i, b_i, y_ij, mu_ij):
    r"""
    log p(t_i) := \
        log(t_i - t_j) * y_ij - mu_ij * (t_i - t_j) + \
        log(t_i) * (a_i - 1) - b_i * t_i

    Returns normalizing constant, E[t_i], V[t_i].
    """

    assert t_j >= 0.0

    s = a_i + y_ij
    r = mu_ij + b_i

    if not _valid_gamma(s, r):
        return nan, nan, nan

    if t_j == 0.0:
        logl = lgamma(s) - s * log(r)
        mn_i = s / r
        va_i = s / ipow(r, 2)
        return logl, mn_i, va_i

    a = y_ij + 1
    b = s + 1
    z = t_j * r

    if not _valid_hyperu(a, b, z):
        return nan, nan, nan

    hyperu = hypergeo._hyperu_laplace
    f0, d0 = hyperu(a + 0, b + 0, z)
    f1, d1 = hyperu(a + 1, b + 1, z)

    logl = f0 - b_i * t_j + (b - 1) * log(t_j) + lgamma(a)
    mn_i = t_j * (1 - d0)
    va_i = ipow(t_j, 2) * d0 * (d1 - d0)

    return logl, mn_i, va_i


def leafward_moments(t_i, a_j, b_j, y_ij, mu_ij):
    r"""
    log p(t_j) := \
        log(t_i - t_j) * y_ij - mu_ij * (t_i - t_j) + \
        log(t_j) * (a_j - 1) - b_j * t_j

    Returns normalizing constant, E[t_j], V[t_j].
    """

    assert t_i > 0.0

    a = a_j
    b = a_j + y_ij + 1
    z = t_i * (mu_ij - b_j)

    if not _valid_hyp1f1(a, b, z):
        return nan, nan, nan

    hyp1f1 = hypergeo._hyp1f1_laplace
    f0 = hyp1f1(a + 0, b + 0, z)
    f1 = hyp1f1(a + 1, b + 1, z)
    f2 = hyp1f1(a + 2, b + 2, z)
    d1 = a / b * exp(f1 - f0)
    d2 = a / b * (a + 1) / (b + 1) * exp(f2 - f0)

    logl = f0 - mu_ij * t_i + (b - 1) * log(t_i) + hypergeo._betaln(a, b - a)
    mn_j = t_i * d1
    sq_j = ipow(t_i, 2) * d2
    va_j = sq_j - ipow(mn_j, 2)

    return logl, mn_j, va_j


def unphased_moments(a_i, b_i, a_j, b_j, y_ij, mu_ij):
    r"""
    log p(t_i, t_j) := \
        log(t_i + t_j) * y_ij - mu_ij * (t_i + t_j) + \
        log(t_i) * (a_i - 1) - b_i * t_i + \
        log(t_j) * (a_j - 1) - b_j * t_j

    Returns normalizing constant, E[t_i], V[t_i], E[t_j], V[t_j].
    """

    a = a_j
    b = a_i + a_j + y_ij
    c = a_j + a_i
    t = mu_ij + b_i
    z = (mu_ij + b_j) / t if t > 0 else nan

    if not _valid_hyp2f1(a, b, c, 1 - z):
        return nan, nan, nan, nan, nan

    hyp2f1 = hypergeo._hyp2f1_laplace
    f0 = hyp2f1(a + 0, b + 0, c + 0, 1 - z)
    f1 = hyp2f1(a + 1, b + 1, c + 1, 1 - z)
    f2 = hyp2f1(a + 2, b + 2, c + 2, 1 - z)
    s1 = a * b / c
    s2 = s1 * (a + 1) * (b + 1) / (c + 1)
    d1 = s1 * exp(f1 - f0)
    d2 = s2 * exp(f2 - f0)

    logl = f0 + hypergeo._betaln(a_j, a_i) + lgamma(b) - b * log(t)

    mn_j = d1 / t
    sq_j = d2 / ipow(t, 2)
    va_j = sq_j - ipow(mn_j, 2)

    mn_i = b / t - mn_j * z
    sq_i = sq_j * ipow(z, 2) + (b + 1) * (mn_i - mn_j * z) / t
    va_i = sq_i - ipow(mn_i, 2)

    return logl, mn_i, va_i, mn_j, va_j


def twin_moments(a_i, b_i, y_ij, mu_ij):
    r"""
    log p(t_i) := \
        log(2 * t_i) * y_ij - mu_ij * (2 * t_i) + \
        log(t_i) * (a_i - 1) - b_i * t_i

    Returns normalizing constant, E[t_i], V[t_i].
    """
    s = a_i + y_ij
    r = b_i + 2 * mu_ij
    logl = log(2) * y_ij + lgamma(s) - log(r) * s
    mn_i = s / r
    va_i = s / ipow(r, 2)
    return logl, mn_i, va_i


def sideways_moments(t_i, a_j, b_j, y_ij, mu_ij):
    r"""
    log p(t_j) := \
        log(t_i + t_j) * y_ij - mu_ij * (t_i + t_j) + \
        log(t_j) * (a_j - 1) - b_j * t_j

    Returns normalizing constant, E[t_j], V[t_j].
    """

    assert t_i > 0.0

    a = a_j
    b = a_j + y_ij + 1
    z = t_i * (mu_ij + b_j)

    if not _valid_hyperu(a, b, z):
        return nan, nan, nan

    hyperu = hypergeo._hyperu_laplace
    f0, d0 = hyperu(a + 0, b + 0, z)
    f1, d1 = hyperu(a + 1, b + 1, z)

    logl = f0 - mu_ij * t_i + (b - 1) * log(t_i) + lgamma(a)
    mn_j = -t_i * d0
    va_j = ipow(t_i, 2) * d0 * (d1 - d0)

    return logl, mn_j, va_j


def mutation_moments(a_i, b_i, a_j, b_j, y_ij, mu_ij):
    r"""
    log p(t_m, t_i, t_j) = \
        log(t_i - t_j) * y_ij - mu_ij * (t_i - t_j) + \
        log(t_i) * (a_i - 1) - b_i * t_i + \
        log(t_j) * (a_j - 1) - b_j * t_j - \
        log(t_i - t_j) + log(int(t_j < t_m < t_i))

    Returns E[t_m], V[t_m].
    """

    a = a_j
    b = a_i + a_j + y_ij
    c = a_j + y_ij + 1
    t = mu_ij + b_i
    z = (mu_ij - b_j) / t if t > 0 else nan

    if not _valid_hyp2f1(a, b, c, z):
        return nan, nan

    hyp2f1 = hypergeo._hyp2f1_laplace
    f000 = hyp2f1(a + 0, b + 0, c + 0, z)
    f020 = hyp2f1(a + 0, b + 2, c + 0, z)
    f111 = hyp2f1(a + 1, b + 1, c + 1, z)
    f121 = hyp2f1(a + 1, b + 2, c + 1, z)
    f222 = hyp2f1(a + 2, b + 2, c + 2, z)

    s1 = a * b / c
    d1 = b * (b + 1) / ipow(t, 2)
    d2 = d1 * a / c
    d3 = d2 * (a + 1) / (c + 1)

    mn_m = s1 * exp(f111 - f000) / t / 2 * (1 + z) + b / t / 2
    sq_m = (
        d1 * exp(f020 - f000) / 3 + d2 * exp(f121 - f000) / 3 + d3 * exp(f222 - f000) / 3
    )
    va_m = sq_m - ipow(mn_m, 2)

    return mn_m, va_m


def mutation_rootward_moments(t_j, a_i, b_i, y_ij, mu_ij):
    r"""
    log p(t_m, t_i) := \
        log(t_i - t_j) * y_ij - mu_ij * (t_i - t_j) + \
        log(t_i) * (a_i - 1) - b_i * t_i - \
        log(t_i - t_j) + log(int(t_j < t_m < t_i))

    Returns E[t_m], V[t_m].
    """

    logl, mn_i, va_i = rootward_moments(t_j, a_i, b_i, y_ij, mu_ij)
    mn_m = mn_i / 2 + t_j / 2
    sq_m = (va_i + ipow(mn_i, 2) + mn_i * t_j + ipow(t_j, 2)) / 3
    va_m = sq_m - ipow(mn_m, 2)

    return mn_m, va_m


def mutation_leafward_moments(t_i, a_j, b_j, y_ij, mu_ij):
    r"""
    log p(t_m, t_j) := \
        log(t_i - t_j) * y_ij - mu_ij * (t_i - t_j) + \
        log(t_j) * (a_j - 1) - b_j * t_j - \
        log(t_i - t_j) + log(int(t_j < t_m < t_i))

    Returns E[t_m], V[t_m].
    """

    logl, mn_j, va_j = leafward_moments(t_i, a_j, b_j, y_ij, mu_ij)
    mn_m = mn_j / 2 + t_i / 2
    sq_m = (va_j + ipow(mn_j, 2) + mn_j * t_i + ipow(t_i, 2)) / 3
    va_m = sq_m - ipow(mn_m, 2)

    return mn_m, va_m


def mutation_unphased_moments(a_i, b_i, a_j, b_j, y_ij, mu_ij):
    r"""
    log p(t_m, t_i, t_j) := \
        log(t_i + t_j) * y_ij - mu_ij * (t_i + t_j) + \
        log(t_i) * (a_i - 1) - b_i * t_i + \
        log(t_j) * (a_j - 1) - b_j * t_j + \
        log(t_i / (t_i + t_j) * int(0 < t_m < t_i) / t_i + \
            t_j / (t_i + t_j) * int(0 < t_m < t_j) / t_j)

    Returns P[m under i], E[t_m], V[t_m].
    """

    # Conditioning on ages of parents:
    #   P[x under i | t_i, t_j] = t_i / (t_i + t_j)
    #   E[x | x under t_i, t_i] = t_i / 2
    #   E[x^2 | x under t_i, t_i] = t_i**2 / 3
    # and equivalently for t_j. Integrating these moments over the EP surrogate
    # density leads to hypergeometric functions similar to the node case, but
    # with integer perturbations of a_i, a_j, y_ij.

    a = a_j
    b = a_j + a_i + y_ij
    c = a_j + a_i
    t = mu_ij + b_i
    z = (mu_ij + b_j) / t if t > 0 else nan

    if not _valid_hyp2f1(a, b, c, 1 - z):
        return nan, nan, nan

    hyp2f1 = hypergeo._hyp2f1_laplace
    f000 = hyp2f1(a + 0, b + 0, c + 0, 1 - z)
    f001 = hyp2f1(a + 0, b + 0, c + 1, 1 - z)
    f012 = hyp2f1(a + 0, b + 1, c + 2, 1 - z)
    f023 = hyp2f1(a + 0, b + 2, c + 3, 1 - z)
    f212 = hyp2f1(a + 2, b + 1, c + 2, 1 - z)
    f323 = hyp2f1(a + 3, b + 2, c + 3, 1 - z)

    s0 = b / t / c / (c + 1)
    s1 = (c - a) * (c - a + 1)
    s2 = a * (a + 1)
    d0 = s0 * (b + 1) / t / (c + 2)
    d1 = s1 * (c - a + 2)
    d2 = s2 * (a + 2)

    mn_m = s0 * s1 * exp(f012 - f000) / 2 + s0 * s2 * exp(f212 - f000) / 2
    sq_m = d0 * d1 * exp(f023 - f000) / 3 + d0 * d2 * exp(f323 - f000) / 3
    va_m = sq_m - ipow(mn_m, 2)
    pr_m = (c - a) / c * exp(f001 - f000)

    return pr_m, mn_m, va_m


def mutation_twin_moments(a_i, b_i, y_ij, mu_ij):
    r"""
    log p(t_m, t_i) := \
        log(2 * t_i) * y_ij - mu_ij * (2 * t_i) + \
        log(t_i) * (a_i - 1) - b_i * t_i + \
        log(int(0 < t_m < t_i) / t_i)
    """

    s = a_i + y_ij
    r = b_i + 2 * mu_ij
    pr_m = 0.5
    mn_m = s / r / 2
    sq_m = (s + 1) * s / 3 / ipow(r, 2)
    va_m = sq_m - ipow(mn_m, 2)

    return pr_m, mn_m, va_m


def mutation_sideways_moments(t_i, a_j, b_j, y_ij, mu_ij):
    r"""
    log p(t_m, t_j) := \
        log(t_i + t_j) * y_ij - mu_ij * (t_i + t_j) + \
        log(t_j) * (a_j - 1) - b_j * t_j + \
        log(t_i / (t_i + t_j) * int(0 < t_m < t_i) / t_i + \
            t_j / (t_i + t_j) * int(0 < t_m < t_j) / t_j)

    Returns P[m under i], E[t_m], V[t_m].
    """

    assert t_i > 0

    # Conditioning on ages of parents:
    #   P[x under i | t_i, t_j] = t_i / (t_i + t_j)
    #   E[x | x under t_i, t_i] = t_i / 2
    #   E[x^2 | x under t_i, t_i] = t_i**2 / 3
    # and equivalently for t_j. Integrating these moments over the EP surrogate
    # density leads to Tricomi functions similar to the node case, but
    # with integer perturbations of a_j, y_ij.

    a = a_j
    b = a_j + y_ij + 1
    z = t_i * (mu_ij + b_j)

    if not _valid_hyperu(a, b, z):
        return nan, nan, nan

    # direct but unstable:
    hyperu = hypergeo._hyperu_laplace
    f00, d00 = hyperu(a + 0, b + 0, z)
    f10, d10 = hyperu(a + 1, b + 0, z)
    f21, d21 = hyperu(a + 2, b + 1, z)
    f32, d32 = hyperu(a + 3, b + 2, z)
    pr_m = 1.0 - exp(f10 - f00) * a
    mn_m = pr_m * t_i / 2 + t_i * exp(f21 - f00) * a * (a + 1) / 2
    sq_m = (
        pr_m * ipow(t_i, 2) / 3
        + ipow(t_i, 2) * exp(f32 - f00) * a * (a + 1) * (a + 2) / 3
    )

    # TODO: use a stabler approach with derivatives
    # note that exp(f10 - f00) = (a + z * d00) / (a - b + 1)
    # however the denominator is 0 if y_ij is 0
    # note that when y_ij == 0 then a == b + 1 and f00 = z**(-a)

    va_m = sq_m - ipow(mn_m, 2)

    return pr_m, mn_m, va_m


def mutation_edge_moments(t_i, t_j):
    r"""
    log p(t_m) := \
        log(t_i - t_j) + log(int(t_j < t_m < t_i))

    Returns E[t_m], V[t_m].
    """

    mn_m = 1 / 2 * (t_i + t_j)
    va_m = 1 / 12 * ipow(t_i - t_j, 2)

    return mn_m, va_m


def mutation_block_moments(t_i, t_j):
    r"""
    log p(t_m) := \
        log(t_i / (t_i + t_j) * int(0 < t_m < t_i) / t_i + \
            t_j / (t_i + t_j) * int(0 < t_m < t_j) / t_j)

    Returns P[m under i], E[t_m], V[t_m].
    """

    assert t_i > 0
    assert t_j > 0

    pr_m = t_i / (t_i + t_j)
    mn_m = pr_m * t_i / 2 + (1 - pr_m) * t_j / 2
    sq_m = pr_m * ipow(t_i, 2) / 3 + (1 - pr_m) * ipow(t_j, 2) / 3
    va_m = sq_m - ipow(mn_m, 2)

    return pr_m, mn_m, va_m


def gamma_projection(pars_i, pars_j, pars_ij):
    r"""
    log p(t_i, t_j) := \
        log(t_i - t_j) * y_ij - mu_ij * (t_i - t_j) + \
        log(t_i) * (a_i - 1) - b_i * t_i + \
        log(t_j) * (a_j - 1) - b_j * t_j

    Returns normalizing constant, gamma natural parameters for parent and child ages
    """
    a_i, b_i = map(float, pars_i)
    a_j, b_j = map(float, pars_j)
    y_ij, mu_ij = map(float, pars_ij)
    a_i += 1
    a_j += 1

    logl, mn_i, va_i, mn_j, va_j = moments(a_i, b_i, a_j, b_j, y_ij, mu_ij)

    if not (_valid_moments(mn_i, va_i) and _valid_moments(mn_j, va_j)):
        return np.nan, pars_i, pars_j

    proj_i = approximate_gamma_mom(mn_i, va_i)
    proj_j = approximate_gamma_mom(mn_j, va_j)

    return logl, np.array(proj_i), np.array(proj_j)


def leafward_projection(t_i, pars_j, pars_ij):
    r"""
    log p(t_j) := \
        log(t_i - t_j) * y_ij - mu_ij * (t_i - t_j) + \
        log(t_j) * (a_j - 1) - b_j * t_j

    Returns normalizing constant, gamma natural parameters for child age
    """
    a_j, b_j = map(float, pars_j)
    y_ij, mu_ij = map(float, pars_ij)
    a_j += 1

    logl, mn_j, va_j = leafward_moments(t_i, a_j, b_j, y_ij, mu_ij)

    if not _valid_moments(mn_j, va_j):
        return np.nan, pars_j

    proj_j = approximate_gamma_mom(mn_j, va_j)

    return logl, np.array(proj_j)


def rootward_projection(t_j, pars_i, pars_ij):
    r"""
    log p(t_i) := \
        log(t_i - t_j) * y_ij - mu_ij * (t_i - t_j) + \
        log(t_i) * (a_i - 1) - b_i * t_i

    Returns normalizing constant, gamma natural parameters for parent age
    """
    a_i, b_i = map(float, pars_i)
    y_ij, mu_ij = map(float, pars_ij)
    a_i += 1

    logl, mn_i, va_i = rootward_moments(t_j, a_i, b_i, y_ij, mu_ij)

    if not _valid_moments(mn_i, va_i):
        return np.nan, pars_i

    proj_i = approximate_gamma_mom(mn_i, va_i)

    return logl, np.array(proj_i)


def unphased_projection(pars_i, pars_j, pars_ij):
    r"""
    log p(t_i, t_j) := \
        log(t_i + t_j) * y_ij - mu_ij * (t_i + t_j) + \
        log(t_i) * (a_i - 1) - b_i * t_i + \
        log(t_j) * (a_j - 1) - b_j * t_j

    Returns normalizing constant, gamma natural parameters for parent ages
    """
    a_i, b_i = map(float, pars_i)
    a_j, b_j = map(float, pars_j)
    y_ij, mu_ij = map(float, pars_ij)
    a_i += 1
    a_j += 1

    logl, mn_i, va_i, mn_j, va_j = unphased_moments(a_i, b_i, a_j, b_j, y_ij, mu_ij)

    if not _valid_moments(mn_i, va_i) or not _valid_moments(mn_j, va_j):
        return np.nan, pars_i, pars_j

    proj_i = approximate_gamma_mom(mn_i, va_i)
    proj_j = approximate_gamma_mom(mn_j, va_j)

    return logl, np.array(proj_i), np.array(proj_j)


def twin_projection(pars_i, pars_ij):
    r"""
    log p(t_i) := \
        log(2 * t_i) * y_ij - mu_ij * (2 * t_i) + \
        log(t_i) * (a_i - 1) - b_i * t_i

    Returns normalizing constant, gamma natural parameters for parent ages
    """
    a_i, b_i = map(float, pars_i)
    y_ij, mu_ij = map(float, pars_ij)
    a_i += 1

    logl, mn_i, va_i = twin_moments(a_i, b_i, y_ij, mu_ij)

    if not _valid_moments(mn_i, va_i):
        return np.nan, pars_i

    proj_i = approximate_gamma_mom(mn_i, va_i)

    return logl, np.array(proj_i)


def sideways_projection(t_i, pars_j, pars_ij):
    r"""
    log p(t_j) := \
        log(t_i + t_j) * y_ij - mu_ij * (t_i + t_j) + \
        log(t_j) * (a_j - 1) - b_j * t_j

    Returns normalizing constant, gamma natural parameters for nonfixed parent age
    """
    a_j, b_j = map(float, pars_j)
    y_ij, mu_ij = map(float, pars_ij)
    a_j += 1

    logl, mn_j, va_j = sideways_moments(t_i, a_j, b_j, y_ij, mu_ij)

    if not _valid_moments(mn_j, va_j):
        return np.nan, pars_j

    proj_j = approximate_gamma_mom(mn_j, va_j)

    return logl, np.array(proj_j)


def mutation_gamma_projection(pars_i, pars_j, pars_ij):
    r"""
    log p(t_m, t_i, t_j) = \
        log(t_i - t_j) * y_ij - mu_ij * (t_i - t_j) + \
        log(t_i) * (a_i - 1) - b_i * t_i + \
        log(t_j) * (a_j - 1) - b_j * t_j - \
        log(t_i - t_j) + log(int(t_j < t_m < t_i))

    Returns phase probability, gamma natural parameters for mutation age
    """
    a_i, b_i = map(float, pars_i)
    a_j, b_j = map(float, pars_j)
    y_ij, mu_ij = map(float, pars_ij)
    a_i += 1
    a_j += 1

    mn_m, va_m = mutation_moments(a_i, b_i, a_j, b_j, y_ij, mu_ij)

    if not _valid_moments(mn_m, va_m):
        return np.nan, np.full(2, np.nan)

    proj_m = approximate_gamma_mom(mn_m, va_m)

    return 1.0, np.array(proj_m)


def mutation_leafward_projection(t_i, pars_j, pars_ij):
    r"""
    log p(t_m, t_j) := \
        log(t_i - t_j) * y_ij - mu_ij * (t_i - t_j) + \
        log(t_j) * (a_j - 1) - b_j * t_j - \
        log(t_i - t_j) + log(int(t_j < t_m < t_i))

    Returns phase probability, gamma natural parameters for mutation age
    """
    a_j, b_j = map(float, pars_j)
    y_ij, mu_ij = map(float, pars_ij)
    a_j += 1

    mn_m, va_m = mutation_leafward_moments(t_i, a_j, b_j, y_ij, mu_ij)

    if not _valid_moments(mn_m, va_m):
        return np.nan, np.full(2, np.nan)

    proj_m = approximate_gamma_mom(mn_m, va_m)

    return 1.0, np.array(proj_m)


def mutation_rootward_projection(t_j, pars_i, pars_ij):
    r"""
    log p(t_m, t_i) := \
        log(t_i - t_j) * y_ij - mu_ij * (t_i - t_j) + \
        log(t_i) * (a_i - 1) - b_i * t_i - \
        log(t_i - t_j) + log(int(t_j < t_m < t_i))

    Returns phase probability, gamma natural parameters for mutation age
    """
    a_i, b_i = map(float, pars_i)
    y_ij, mu_ij = map(float, pars_ij)
    a_i += 1

    mn_m, va_m = mutation_rootward_moments(t_j, a_i, b_i, y_ij, mu_ij)

    if not _valid_moments(mn_m, va_m):
        return np.nan, np.full(2, np.nan)

    proj_m = approximate_gamma_mom(mn_m, va_m)

    return 1.0, np.array(proj_m)


def mutation_edge_projection(t_i, t_j):
    r"""
    log p(t_m) := \
        log(t_i - t_j) + log(int(t_j < t_m < t_i))

    Returns phase probability, gamma natural parameters for mutation age
    """
    mn_m, va_m = mutation_edge_moments(t_i, t_j)

    if not _valid_moments(mn_m, va_m):
        return np.nan, np.full(2, np.nan)

    proj_m = approximate_gamma_mom(mn_m, va_m)

    return 1.0, np.array(proj_m)


def mutation_unphased_projection(pars_i, pars_j, pars_ij):
    r"""
    log p(t_m, t_i, t_j) := \
        log(t_i + t_j) * y_ij - mu_ij * (t_i + t_j) + \
        log(t_i) * (a_i - 1) - b_i * t_i + \
        log(t_j) * (a_j - 1) - b_j * t_j + \
        log(t_i / (t_i + t_j) * int(0 < t_m < t_i) + \
            t_j / (t_i + t_j) * int(0 < t_m < t_j))

    Returns phase probability, gamma natural parameters for mutation age
    """
    a_i, b_i = map(float, pars_i)
    a_j, b_j = map(float, pars_j)
    y_ij, mu_ij = map(float, pars_ij)
    a_i += 1
    a_j += 1

    pr_m, mn_m, va_m = mutation_unphased_moments(a_i, b_i, a_j, b_j, y_ij, mu_ij)

    if not _valid_moments(mn_m, va_m) or not (0 <= pr_m <= 1):
        return np.nan, np.full(2, np.nan)

    proj_m = approximate_gamma_mom(mn_m, va_m)

    return pr_m, np.array(proj_m)


def mutation_twin_projection(pars_i, pars_ij):
    r"""
    log p(t_m, t_i) := \
        log(2 * t_i) * y_ij - mu_ij * (2 * t_i) + \
        log(t_i) * (a_i - 1) - b_i * t_i + \
        log(int(0 < t_m < t_i) / t_i)

    Returns phase probability, gamma natural parameters for mutation age
    """
    a_i, b_i = map(float, pars_i)
    y_ij, mu_ij = map(float, pars_ij)
    a_i += 1

    pr_m, mn_m, va_m = mutation_twin_moments(a_i, b_i, y_ij, mu_ij)

    if not _valid_moments(mn_m, va_m) or not (0 <= pr_m <= 1):
        return np.nan, np.full(2, np.nan)

    proj_m = approximate_gamma_mom(mn_m, va_m)

    return pr_m, np.array(proj_m)


def mutation_sideways_projection(t_i, pars_j, pars_ij):
    r"""
    log p(t_m, t_j) := \
        log(t_i + t_j) * y_ij - mu_ij * (t_i + t_j) + \
        log(t_j) * (a_j - 1) - b_j * t_j + \
        log(t_i / (t_i + t_j) * int(0 < t_m < t_i) + \
            t_j / (t_i + t_j) * int(0 < t_m < t_j))

    Returns phase probability, gamma natural parameters for mutation age
    """
    a_j, b_j = map(float, pars_j)
    y_ij, mu_ij = map(float, pars_ij)
    a_j += 1

    pr_m, mn_m, va_m = mutation_sideways_moments(t_i, a_j, b_j, y_ij, mu_ij)

    if not _valid_moments(mn_m, va_m) or not (0 <= pr_m <= 1):
        return np.nan, np.full(2, np.nan)

    proj_m = approximate_gamma_mom(mn_m, va_m)

    return pr_m, np.array(proj_m)


def mutation_block_projection(t_i, t_j):
    r"""
    log p(t_m) := \
        log(t_i / (t_i + t_j) * int(0 < t_m < t_i) + \
            t_j / (t_i + t_j) * int(0 < t_m < t_j))

    Returns phase probability, gamma natural parameters for mutation age
    """
    pr_m, mn_m, va_m = mutation_block_moments(t_i, t_j)

    if not _valid_moments(mn_m, va_m) or not (0 <= pr_m <= 1):
        return np.nan, np.full(2, np.nan)

    proj_m = approximate_gamma_mom(mn_m, va_m)

    return pr_m, np.array(proj_m)
