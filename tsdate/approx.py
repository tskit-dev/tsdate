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
import numba
import numpy as np

from . import hypergeo


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
    assert alpha > 0, "kl-min: bad initial condition"
    last = np.inf
    itt = 0
    while np.abs(alpha - last) > alpha * 1e-8:
        last = alpha
        numer = hypergeo._digamma(alpha) - np.log(alpha) - logx + np.log(x)
        denom = hypergeo._trigamma(alpha) - 1 / alpha
        alpha -= numer / denom
        itt += 1
    assert np.isfinite(alpha) and alpha > 0, "kl-min: failed"
    beta = alpha / x
    return alpha, beta


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
    assert a_i > 0 and b_i > 0, "Invalid parent parameters"
    assert a_j > 0 and b_j > 0, "Invalid child parameters"
    assert y_ij >= 0 and mu_ij > 0, "Invalid edge parameters"

    a = a_i + a_j + y_ij
    b = a_j
    c = a_j + y_ij + 1
    t = mu_ij + b_i

    log_f, sign_f, da_i, db_i, da_j, db_j = hypergeo._hyp2f1(
        a_i, b_i, a_j, b_j, y_ij, mu_ij
    )

    if sign_f <= 0:
        raise hypergeo.Invalid2F1("Singular hypergeometric function")

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


@numba.njit(
    "Tuple((float64, float64[:], float64[:]))"
    "(float64, float64, float64, float64, float64, float64)"
)
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
    logconst, t_i, ln_t_i, t_j, ln_t_j = sufficient_statistics(
        a_i, b_i, a_j, b_j, y_ij, mu_ij
    )

    alpha_i, beta_i = approximate_gamma_kl(t_i, ln_t_i)
    alpha_j, beta_j = approximate_gamma_kl(t_j, ln_t_j)

    return logconst, np.array([alpha_i, beta_i]), np.array([alpha_j, beta_j])
