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

# TODO: these are reasonable defaults but could
# be set via a control dict
_KLMIN_MAXITER = 100
_KLMIN_TOL = np.sqrt(np.finfo(np.float64).eps)


class KLMinimizationFailed(Exception):
    pass


@numba.njit("UniTuple(f8, 3)(f8, f8)")
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


@numba.njit("UniTuple(f8, 2)(f8, f8)")
def approximate_gamma_kl(x, logx):
    """
    Use Newton root finding to get gamma natural parameters matching the sufficient
    statistics :math:`E[x]` and :math:`E[\\log x]`, minimizing KL divergence.

    The initial condition uses the upper bound :math:`digamma(x) \\leq log(x) - 1/2x`.

    Returns the shape and rate of the approximating gamma.
    """
    if x <= 0.0 or np.isinf(logx):
        raise KLMinimizationFailed("Nonpositive or nonfinite moments")
    if not np.log(x) > logx:
        raise KLMinimizationFailed("log E[t] <= E[log t] violates Jensen's inequality")
    alpha = 0.5 / (np.log(x) - logx)  # lower bound on alpha
    # asymptotically the lower bound becomes sharp
    if 1.0 / alpha < 1e-4:
        return alpha - 1.0, alpha / x
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
    return alpha - 1.0, alpha / x


@numba.njit("UniTuple(f8, 2)(f8, f8)")
def approximate_gamma_mom(mean, variance):
    """
    Use the method of moments to approximate a distribution with a gamma of the
    same mean and variance, returning natural parameters
    """
    assert mean > 0
    assert variance > 0
    shape = mean**2 / variance
    rate = mean / variance
    return shape - 1.0, rate


@numba.njit("UniTuple(f8, 2)(f8[:], f8[:])")
def average_gammas(alpha, beta):
    """
    Given natural parameters for a set of gammas, average sufficient
    statistics so as to get a "global" gamma, returning natural
    parameters
    """
    assert alpha.size == beta.size, "Array sizes are not equal"
    avg_x = 0.0
    avg_logx = 0.0
    for shape, rate in zip(alpha + 1.0, beta):
        avg_logx += hypergeo._digamma(shape) - np.log(rate)
        avg_x += shape / rate
    avg_x /= alpha.size
    avg_logx /= alpha.size
    return approximate_gamma_kl(avg_x, avg_logx)


@numba.njit("UniTuple(f8, 5)(f8, f8, f8, f8, f8, f8)")
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

    log_f, da_i, db_i, da_j, db_j = hypergeo._hyp2f1(a_i, b_i, a_j, b_j, y_ij, mu_ij)

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


@numba.njit("UniTuple(f8, 7)(f8, f8, f8, f8, f8, f8)")
def taylor_approximation(a_i, b_i, a_j, b_j, y_ij, mu_ij):
    """
    Calculate sufficient statistics for the PDF proportional to
    :math:`Ga(t_j | a_j, b_j) Ga(t_i | a_i, b_i) Po(y_{ij} |
    \\mu_{ij} t_i - t_j)`, where :math:`i` is the parent and :math:`j` is
    the child. The logarithmic moments are approximated via a Taylor
    expansion around the mean.

    :param float a_i: the shape parameter of the cavity distribution for the parent
    :param float b_i: the rate parameter of the cavity distribution for the parent
    :param float a_j: the shape parameter of the cavity distribution for the child
    :param float b_j: the rate parameter of the cavity distribution for the child
    :param float y_ij: the number of mutations on the edge
    :param float mu_ij: the span-weighted mutation rate of the edge

    :return: normalizing constant, E[t_i], E[log t_i], V[t_i],
        E[t_j], E[log t_j], V[t_j]
    """

    a = a_j
    b = a_i + a_j + y_ij
    c = a_j + y_ij + 1
    t = mu_ij + b_i
    z = (mu_ij - b_j) / t

    f0 = hypergeo._hyp2f1_fast(a, b, c, z)
    f1 = hypergeo._hyp2f1_fast(a + 1, b + 1, c + 1, z)
    f2 = hypergeo._hyp2f1_fast(a + 2, b + 2, c + 2, z)
    s1 = a * b / c
    s2 = s1 * (a + 1) * (b + 1) / (c + 1)
    d1 = s1 * np.exp(f1 - f0)
    d2 = s2 * np.exp(f2 - f0)

    logl = f0 + hypergeo._betaln(y_ij + 1, a) + hypergeo._gammaln(b) - b * np.log(t)

    mn_j = d1 / t
    sq_j = d2 / t**2
    va_j = sq_j - mn_j**2
    ln_j = np.log(mn_j) - va_j / 2 / mn_j**2

    mn_i = mn_j * z + b / t
    sq_i = sq_j * z**2 + (b + 1) * (mn_i + mn_j * z) / t
    va_i = sq_i - mn_i**2
    ln_i = np.log(mn_i) - va_i / 2 / mn_i**2

    return logl, mn_i, ln_i, va_i, mn_j, ln_j, va_j


@numba.njit("b1(f8, f8, f8, f8)")
def _valid_sufficient_statistics(t_i, ln_t_i, t_j, ln_t_j):
    if t_i <= 0:
        return False
    if t_j <= 0:
        return False
    if np.isinf(ln_t_i):
        return False
    if np.isinf(ln_t_j):
        return False
    if np.log(t_i) <= ln_t_i:
        return False
    if np.log(t_j) <= ln_t_j:
        return False
    return True


@numba.njit("b1(f8, f8, f8, f8, f8, f8)")
def _valid_parameterization(a_i, b_i, a_j, b_j, y, mu):
    """Uses shape / rate parameterization"""
    a = a_j
    b = a_i + a_j + y
    c = a_j + y + 1
    s = mu - b_j
    t = mu + b_i
    # check that 2F1 argument is not unity under some transformation
    if t <= 0.0:
        return False
    z = s / t
    if z >= 1.0:
        return False
    if z / (z - 1) >= 1.0:
        return False
    # check that 2F1 is positive
    if a <= 0:
        return False
    if b <= 0:
        return False
    if c <= 0:
        return False
    return True


@numba.njit("Tuple((f8, f8[:], f8[:]))(f8[:], f8[:], f8[:], b1)")
def gamma_projection(pars_i, pars_j, pars_ij, min_kl):
    """
    Match a pair of gamma distributions to the potential function
    :math:`Ga(t_j | a_j + 1, b_j) Ga(t_i | a_i + 1, b_i) Po(y_{ij} |
    \\mu_{ij} t_i - t_j)`, where :math:`i` is the parent and :math:`j` is
    the child, by minimizing KL divergence.

    :param float a_i: the first parameter of the cavity distribution for the parent
    :param float b_i: the second parameter of the cavity distribution for the parent
    :param float a_j: the first parameter of the cavity distribution for the child
    :param float b_j: the second parameter of the cavity distribution for the child
    :param float y_ij: the number of mutations on the edge
    :param float mu_ij: the span-weighted mutation rate of the edge
    :param bool min_kl: minimize KL divergence (match central moments if False)

    :return: gamma natural parameters for parent and child
    """

    # switch from natural to canonical parameterization
    a_i, b_i = pars_i
    a_j, b_j = pars_j
    y_ij, mu_ij = pars_ij
    a_i += 1
    a_j += 1

    # skip update, zeroing out message
    if not _valid_parameterization(a_i, b_i, a_j, b_j, y_ij, mu_ij):
        return np.nan, pars_i, pars_j

    # if min_kl:
    #    logconst, t_i, ln_t_i, t_j, ln_t_j = sufficient_statistics(
    #        a_i, b_i, a_j, b_j, y_ij, mu_ij
    #    )
    #    if not _valid_sufficient_statistics(t_i, ln_t_i, t_j, ln_t_j):
    #        logconst, t_i, ln_t_i, _, t_j, ln_t_j, _ = taylor_approximation(
    #            a_i, b_i, a_j, b_j, y_ij, mu_ij
    #        )
    #    proj_i = approximate_gamma_kl(t_i, ln_t_i)
    #    proj_j = approximate_gamma_kl(t_j, ln_t_j)

    if min_kl:
        logconst, t_i, ln_t_i, _, t_j, ln_t_j, _ = taylor_approximation(
            a_i, b_i, a_j, b_j, y_ij, mu_ij
        )
        proj_i = approximate_gamma_kl(t_i, ln_t_i)
        proj_j = approximate_gamma_kl(t_j, ln_t_j)
    else:
        logconst, t_i, _, va_t_i, t_j, _, va_t_j = taylor_approximation(
            a_i, b_i, a_j, b_j, y_ij, mu_ij
        )
        proj_i = approximate_gamma_mom(t_i, va_t_i)
        proj_j = approximate_gamma_mom(t_j, va_t_j)

    return logconst, np.array(proj_i), np.array(proj_j)
