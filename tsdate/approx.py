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
Tools for approximating combinations of Gamma variates with Gamma distributions
"""
import logging

import mpmath
import numba
import numpy as np

from . import hypergeo

# TODO: these are reasonable defaults but could
# be set via a control dict
_KLMIN_MAXITER = 100
_KLMIN_TOL = np.sqrt(np.finfo(np.float64).eps)


class KLMinimizationFailed(Exception):
    pass


@numba.njit("UniTuple(float64, 3)(float64, float64)")
def approximate_log_moments(mean, variance):
    """
    Approximate log moments via a second-order Taylor series expansion around
    the mean, e.g.:

      E[f(x)] \\approx E[f(mean)] + variance * f''(mean)/2

    Returns approximations to E[log x], E[x log x], E[(log x)^2]
    """
    assert mean > 0
    assert variance > 0
    logx = np.log(mean) - 0.5 * variance / mean**2
    xlogx = mean * np.log(mean) + 0.5 * variance / mean
    logx2 = np.log(mean) ** 2 + (1 - np.log(mean)) * variance / mean**2
    return logx, xlogx, logx2


@numba.njit("UniTuple(float64, 2)(float64, float64)")
def approximate_gamma_kl(x, logx):
    """
    Use Newton root finding to get gamma parameters matching the sufficient
    statistics :math:`E[x]` and :math:`E[\\log x]`, minimizing KL divergence.

    The initial condition uses the upper bound :math:`digamma(x) \\leq log(x) - 1/2x`.

    Returns the shape and rate of the approximating gamma.
    """
    assert np.isfinite(x) and np.isfinite(logx)
    alpha = 0.5 / (np.log(x) - logx)  # lower bound on alpha
    if not alpha > 0:
        raise KLMinimizationFailed(
            "Sufficient statistics don't satisfy Jensen's inequality"
        )
    # asymptotically the lower bound becomes sharp
    if 1.0 / alpha < 1e-4:
        return alpha, alpha / x
    itt = 0
    delta = np.inf
    # determine convergence when the change in alpha falls below
    # some small value (e.g. square root of machine precision)
    while np.abs(delta) > alpha * _KLMIN_TOL:
        if itt > _KLMIN_MAXITER:
            raise KLMinimizationFailed("Maximum iterations reached in KL minimization")
        delta = hypergeo._digamma(alpha) - np.log(alpha) + np.log(x) - logx
        delta /= hypergeo._trigamma(alpha) - 1 / alpha
        alpha -= delta
        itt += 1
    if not np.isfinite(alpha) or alpha <= 0:
        raise KLMinimizationFailed("Invalid shape parameter in KL minimization")
    return alpha, alpha / x


@numba.njit("UniTuple(float64, 2)(float64, float64)")
def approximate_gamma_mom(mean, variance):
    """
    Use the method of moments to approximate a distribution with a gamma of the
    same mean and variance
    """
    assert mean > 0
    assert variance > 0
    alpha = mean**2 / variance
    beta = mean / variance
    return alpha, beta


# @numba.njit("UniTuple(float64, 2)(float64[:], float64[:])")
def rescale_gammas(posterior, edges_in, edges_out, new_shape):
    """
    Given a factorization of gamma parameters in `posterior` into additive
    terms `edges_in` and `edges_out` and a prior, rescale so that the posterior
    shape has a fixed value.
    """

    in_shape, in_rate = edges_in[:, 0], edges_in[:, 1]
    out_shape, out_rate = edges_out[:, 0], edges_out[:, 1]
    post_shape, post_rate = posterior[0], posterior[1]

    assert post_shape > 0 and post_rate > 0

    # new posterior parameters
    new_rate = new_shape * post_rate / post_shape

    # rescale messages to match desired shape
    shape_scale = (new_shape - 1) / (post_shape - 1)
    rate_scale = new_rate / post_rate
    in_shape = (in_shape - 1) * shape_scale + 1
    out_shape = (out_shape - 1) * shape_scale + 1
    in_rate = in_rate * rate_scale
    out_rate = out_rate * rate_scale

    return (
        np.array([new_shape, new_rate]),
        np.column_stack([in_shape, in_rate]),
        np.column_stack([out_shape, out_rate]),
    )


@numba.njit("UniTuple(float64, 2)(float64[:], float64[:])")
def average_gammas(shape, rate):
    """
    Given shape and rate parameters for a set of gammas, average sufficient
    statistics so as to get a "global" gamma
    """
    assert shape.size == rate.size, "Array sizes are not equal"
    avg_x = 0.0
    avg_logx = 0.0
    for a, b in zip(shape, rate):
        avg_logx += hypergeo._digamma(a) - np.log(b)
        avg_x += a / b
    avg_x /= shape.size
    avg_logx /= shape.size
    return approximate_gamma_kl(avg_x, avg_logx)


@numba.njit(
    "UniTuple(float64, 5)(float64, float64, float64, float64, float64, float64)"
)
def sufficient_statistics(a_i, b_i, a_j, b_j, y_ij, mu_ij):
    """
    Calculate gamma sufficient statistics for the PDF proportional to
    :math:`Ga(t_j | a_j, b_j) Ga(t_i | a_i, b_i) Po(y_{ij} |
    \\mu_{ij} t_i - t_j)`, where :math:`i` is the parent and :math:`j` is
    the child.

    :param float a_i: the shape parameter of the cavity distribution for the parent
    :param float b_i: the rate parameter of the cavity distribution for the parent
    :param float a_j: the shape parameter of the cavity distribution for the child
    :param float b_j: the rate parameter of the cavity distribution for the child
    :param float y_ij: the number of mutations on the edge
    :param float mu_ij: the span-weighted mutation rate of the edge

    :return: normalizing constant, E[t_i], E[log t_i], E[t_j], E[log t_j]
    """

    a = a_i + a_j + y_ij
    b = a_j
    c = a_j + y_ij + 1
    t = mu_ij + b_i

    assert a > 0
    assert b > 0
    assert c > 0
    assert t > 0

    log_f, sign_f, da_i, db_i, da_j, db_j = hypergeo._hyp2f1(
        a_i, b_i, a_j, b_j, y_ij, mu_ij
    )

    logconst = (
        log_f + hypergeo._betaln(y_ij + 1, b) + hypergeo._gammaln(a) - a * np.log(t)
    )

    t_i = -db_i + a / t
    t_j = -db_j
    ln_t_i = da_i - np.log(t) + hypergeo._digamma(a)
    ln_t_j = (
        da_j
        - np.log(t)
        + hypergeo._digamma(a)
        + hypergeo._digamma(b)
        - hypergeo._digamma(c)
    )

    return logconst, t_i, ln_t_i, t_j, ln_t_j


def mean_and_variance(a_i, b_i, a_j, b_j, y_ij, mu_ij, dps=100, maxterms=1e7):
    """
    Calculate mean and variance for the PDF proportional to
    :math:`Ga(t_j | a_j, b_j) Ga(t_i | a_i, b_i) Po(y_{ij} |
    \\mu_{ij} t_i - t_j)`, where :math:`i` is the parent and :math:`j` is
    the child.

    This is intended to provide a stable approximation when calculation of
    gamma sufficient statistics fails (e.g. when the log-normalizer is close to
    singular). Calculations are done at arbitrary precision and are slow.

    :param float a_i: the shape parameter of the cavity distribution for the parent
    :param float b_i: the rate parameter of the cavity distribution for the parent
    :param float a_j: the shape parameter of the cavity distribution for the child
    :param float b_j: the rate parameter of the cavity distribution for the child
    :param float y_ij: the number of mutations on the edge
    :param float mu_ij: the span-weighted mutation rate of the edge
    :param int dps: decimal places for multiprecision computations

    :return: normalizing constant, E[t_i], V[t_i], E[t_j], V[t_j]
    """

    a = a_i + a_j + y_ij
    b = a_j
    c = a_j + y_ij + 1
    t = mu_ij + b_i
    z = (mu_ij - b_j) / t

    assert a > 0
    assert b > 0
    assert c > 0
    assert t > 0

    # 2F1 and first/second derivatives of argument, in arbitrary precision
    with mpmath.workdps(dps):
        s0 = a * b / c
        s1 = s0 * (a + 1) * (b + 1) / (c + 1)
        v0 = mpmath.hyp2f1(a, b, c, z, maxterms=maxterms)
        v1 = s0 * (mpmath.hyp2f1(a + 1, b + 1, c + 1, z, maxterms=maxterms) / v0)
        v2 = s1 * (mpmath.hyp2f1(a + 2, b + 2, c + 2, z, maxterms=maxterms) / v0)
        logconst = float(mpmath.log(v0))
        dz = float(v1)
        d2z = float(v2)

    # mean / variance of child and parent age
    logconst += hypergeo._betaln(y_ij + 1, b) + hypergeo._gammaln(a) - a * np.log(t)
    t_i = dz * z / t + a / t
    va_t_i = z / t**2 * (d2z * z + 2 * dz * (1 + a)) + a * (1 + a) / t**2 - t_i**2
    t_j = dz / t
    va_t_j = d2z / t**2 - t_j**2

    return logconst, t_i, va_t_i, t_j, va_t_j


def gamma_projection(a_i, b_i, a_j, b_j, y_ij, mu_ij):
    """
    Match a pair of gamma distributions to the potential function
    :math:`Ga(t_j | a_j, b_j) Ga(t_i | a_i, b_i) Po(y_{ij} |
    \\mu_{ij} t_i - t_j)`, where :math:`i` is the parent and :math:`j` is
    the child, by minimizing KL divergence.

    :param float a_i: the shape parameter of the cavity distribution for the parent
    :param float b_i: the rate parameter of the cavity distribution for the parent
    :param float a_j: the shape parameter of the cavity distribution for the child
    :param float b_j: the rate parameter of the cavity distribution for the child
    :param float y_ij: the number of mutations on the edge
    :param float mu_ij: the span-weighted mutation rate of the edge

    :return: gamma parameters for parent and child
    """
    try:
        logconst, t_i, ln_t_i, t_j, ln_t_j = sufficient_statistics(
            a_i, b_i, a_j, b_j, y_ij, mu_ij
        )
        proj_i = approximate_gamma_kl(t_i, ln_t_i)
        proj_j = approximate_gamma_kl(t_j, ln_t_j)
    except (hypergeo.Invalid2F1, KLMinimizationFailed):
        logging.warning(
            f"Matching sufficient statistics failed with parameters: "
            f"{a_i} {b_i} {a_j} {b_j} {y_ij} {mu_ij},"
            f"matching mean and variance instead"
        )
        logconst, t_i, va_t_i, t_j, va_t_j = mean_and_variance(
            a_i, b_i, a_j, b_j, y_ij, mu_ij
        )
        proj_i = approximate_gamma_mom(t_i, va_t_i)
        proj_j = approximate_gamma_mom(t_j, va_t_j)

    return logconst, np.array(proj_i), np.array(proj_j)
