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
from math import exp
from math import inf
from math import lgamma
from math import log

import mpmath
import numba
import numpy as np
from numba.types import Tuple as _tuple
from numba.types import UniTuple as _unituple

from . import hypergeo

# TODO: these are reasonable defaults but could
# be set via a control dict
_KLMIN_MAXITT = 100
_KLMIN_RELTOL = np.sqrt(np.finfo(np.float64).eps)


# shorthand for numba readonly array types, [type][dimension][constness]
# type is one of "i" (int32), "f" (float64), "b" (boolean)
# constness is one of "r" (read-only) or "w" (writable)
_f = numba.types.float64
_i = numba.types.int32
_b = numba.types.bool_
_f1w = numba.types.Array(_f, 1, "C", readonly=False)
_f1r = numba.types.Array(_f, 1, "C", readonly=True)
_f2w = numba.types.Array(_f, 2, "C", readonly=False)
_f2r = numba.types.Array(_f, 2, "C", readonly=True)
_f3w = numba.types.Array(_f, 3, "C", readonly=False)
_f3r = numba.types.Array(_f, 3, "C", readonly=True)
_i1w = numba.types.Array(_i, 1, "C", readonly=False)
_i1r = numba.types.Array(_i, 1, "C", readonly=True)
_b1w = numba.types.Array(_b, 1, "C", readonly=False)
_b1r = numba.types.Array(_b, 1, "C", readonly=True)


class KLMinimizationFailed(Exception):
    pass


@numba.njit(_unituple(_f, 3)(_f, _f))
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


@numba.njit(_unituple(_f, 2)(_f, _f))
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
    while np.abs(delta) > np.abs(alpha) * _KLMIN_RELTOL:
        if itt > _KLMIN_MAXITT:
            raise KLMinimizationFailed("Maximum iterations reached in KL minimization")
        delta = hypergeo._digamma(alpha) - np.log(alpha) + np.log(x) - logx
        delta /= hypergeo._trigamma(alpha) - 1 / alpha
        alpha -= delta
        itt += 1
    if not np.isfinite(alpha) or alpha <= 0:
        raise KLMinimizationFailed("Invalid shape parameter in KL minimization")
    return alpha - 1.0, alpha / x


@numba.njit(_unituple(_f, 2)(_f, _f))
def approximate_gamma_mom(mean, variance):
    """
    Use the method of moments to approximate a distribution with a gamma of the
    same mean and variance, returning natural parameters
    """
    if not (mean > 0.0 and variance > 0.0):
        raise KLMinimizationFailed("Nonpositive central moments")
    shape = mean**2 / variance
    rate = mean / variance
    return shape - 1.0, rate


@numba.njit(_unituple(_f, 2)(_f, _f, _f, _f))
def approximate_gamma_iqr(q1, q2, x1, x2):
    """Find gamma natural parameters that match empirical quantiles"""
    if not (q2 > q1 and x2 > x1):
        raise KLMinimizationFailed("Quantiles must be sorted")
    # find starting value from asymptotic solutions
    # if x2 / x1 < log(1 - q2) / log(1 - q1):
    #    y1 = hypergeo._erf_inv(2 * q1 - 1) * sqrt(2)
    #    y2 = hypergeo._erf_inv(2 * q2 - 1) * sqrt(2)
    #    alpha = (y1 * x2 - y2 * x1) ** 2 / (x1 - x2) ** 2
    # else:
    alpha = log(q2 / q1) / log(x2 / x1)
    # refine with newton iteration
    delta = inf
    itt = 0
    while abs(delta) > abs(alpha) * _KLMIN_RELTOL:
        if itt > _KLMIN_MAXITT:
            raise KLMinimizationFailed(
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
        gra = (gra_2 * y1 - gra_1 * y2) / y1**2
        delta = -obj / gra
        alpha += delta
        itt += 1
    if not alpha > 0:
        raise KLMinimizationFailed("Negative shape parameter")
    beta = hypergeo._gammainc_inv(alpha, q1) / x1
    return alpha - 1, beta


@numba.njit(_unituple(_f, 2)(_f1r, _f1r))
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


@numba.njit(_b(_f, _f))
def _valid_moments(mn, va):
    if not (np.isfinite(mn) and np.isfinite(va)):
        return False
    if not (mn > 0.0 and va > 0.0):
        return False
    return True


@numba.njit(_b(_f, _f, _f))
def _valid_hyp1f1(a, b, z):
    if not (b >= a > 0.0):
        return False
    return True


@numba.njit(_b(_f, _f, _f))
def _valid_hyperu(a, b, z):
    if z <= 0.0:
        return False
    if not (b > a > 0.0):
        return False
    return True


@numba.njit(_b(_f, _f, _f, _f))
def _valid_hyp2f1(a, b, c, z):
    if z >= 1 or z / (z - 1) >= 1.0:
        return False
    if a <= 0 or b <= 0 or c <= 0:
        return False
    return True


# --- node posteriors --- #

@numba.njit(_unituple(_f, 5)(_f, _f, _f, _f, _f, _f))
def moments(a_i, b_i, a_j, b_j, y_ij, mu_ij):
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

    hyp2f1 = hypergeo._hyp2f1_laplace
    f0 = hyp2f1(a + 0, b + 0, c + 0, z)
    f1 = hyp2f1(a + 1, b + 1, c + 1, z)
    f2 = hyp2f1(a + 2, b + 2, c + 2, z)
    s1 = a * b / c
    s2 = s1 * (a + 1) * (b + 1) / (c + 1)
    d1 = s1 * np.exp(f1 - f0)
    d2 = s2 * np.exp(f2 - f0)

    logl = f0 + hypergeo._betaln(y_ij + 1, a) + hypergeo._gammaln(b) - b * log(t)

    mn_j = d1 / t
    sq_j = d2 / t**2
    va_j = sq_j - mn_j**2

    mn_i = mn_j * z + b / t
    sq_i = sq_j * z**2 + (b + 1) * (mn_i + mn_j * z) / t
    va_i = sq_i - mn_i**2

    return logl, mn_i, va_i, mn_j, va_j


@numba.njit(_unituple(_f, 3)(_f, _f, _f, _f, _f))
def rootward_moments(t_j, a_i, b_i, y_ij, mu_ij):
    """
    Calculate sufficient statistics for the PDF proportional to
    :math:`Ga(t_i | a_i, b_i) Po(y_{ij} | \\mu_{ij} t_i - t_j)`, where
    :math:`i` is the parent and :math:`j` is the child. The logarithmic moments
    are approximated via a Taylor expansion around the mean.

    :param float t_j: the age of the child
    :param float a_i: the shape parameter of the cavity distribution for the parent
    :param float b_i: the rate parameter of the cavity distribution for the parent
    :param float y_ij: the number of mutations on the edge
    :param float mu_ij: the span-weighted mutation rate of the edge

    :return: normalizing constant, E[t_i], E[log t_i], V[t_i]
    """

    assert t_j >= 0.0

    s = a_i + y_ij
    r = mu_ij + b_i
    a = y_ij + 1
    b = s + 1
    z = t_j * r

    if t_j == 0.0:
        logl = hypergeo._gammaln(s) - s * log(r)
        mn_i = s / r
        va_i = s / r**2
        return logl, mn_i, va_i

    hyperu = hypergeo._hyperu_laplace
    f0, d0 = hyperu(a + 0, b + 0, z)
    f1, d1 = hyperu(a + 1, b + 1, z)

    logl = f0 - b_i * t_j + (b - 1) * log(t_j) + hypergeo._gammaln(a)
    mn_i = t_j * (1 - d0)
    va_i = t_j**2 * d0 * (d1 - d0)

    return logl, mn_i, va_i


@numba.njit(_unituple(_f, 3)(_f, _f, _f, _f, _f))
def leafward_moments(t_i, a_j, b_j, y_ij, mu_ij):
    """
    Calculate sufficient statistics for the PDF proportional to
    :math:`Ga(t_j | a_j, b_j) Po(y_{ij} | \\mu_{ij} t_i - t_j)`, where
    :math:`i` is the parent and :math:`j` is the child. The logarithmic moments
    are approximated via a Taylor expansion around the mean.

    :param float t_i: the age of the parent
    :param float a_j: the shape parameter of the cavity distribution for the child
    :param float b_j: the rate parameter of the cavity distribution for the child
    :param float y_ij: the number of mutations on the edge
    :param float mu_ij: the span-weighted mutation rate of the edge

    :return: normalizing constant, E[t_j], E[log t_j], V[t_j]
    """

    assert t_i > 0.0

    a = a_j
    b = a_j + y_ij + 1
    z = t_i * (mu_ij - b_j)

    hyp1f1 = hypergeo._hyp1f1_laplace
    f0 = hyp1f1(a + 0, b + 0, z)
    f1 = hyp1f1(a + 1, b + 1, z)
    f2 = hyp1f1(a + 2, b + 2, z)
    d1 = a / b * exp(f1 - f0)
    d2 = a / b * (a + 1) / (b + 1) * exp(f2 - f0)

    logl = f0 - mu_ij * t_i + (b - 1) * log(t_i) + hypergeo._betaln(a, b - a)
    mn_j = t_i * d1
    sq_j = t_i**2 * d2
    va_j = sq_j - mn_j**2

    return logl, mn_j, va_j


@numba.njit(_b(_f, _f, _f, _f, _f, _f))
def _hyp2f1_valid_parameterization(a_i, b_i, a_j, b_j, y, mu):
    """Uses shape / rate parameterization"""
    a = a_j
    b = a_i + a_j + y
    c = a_j + y + 1
    s = mu - b_j
    t = mu + b_i
    # check that 2F1 argument is less than unity
    if t <= 0.0:
        return False
    z = s / t
    if z >= 1.0 or z / (z - 1) >= 1.0:
        return False
    # check that 2F1 is positive
    if a <= 0:
        return False
    if b <= 0:
        return False
    if c <= 0:
        return False
    return True


@numba.njit(_b(_f, _f, _f, _f, _f))
def _hyp1f1_valid_parameterization(t_i, a_j, b_j, y, mu):
    """Uses shape / rate parameterization"""
    a = a_j
    b = a_j + y + 1
    if not (b > a > 0.0):
        return False
    return True


@numba.njit(_b(_f, _f, _f, _f, _f))
def _hyperu_valid_parameterization(t_j, a_i, b_i, y, mu):
    """Uses shape / rate parameterization"""
    a = y + 1
    b = a_i + y + 1
    if t_j < 0.0:
        return False
    if mu + b_i <= 0.0:
        return False
    if not (b > a > 0.0):
        return False
    return True

@numba.njit(_tuple((_f, _f1r, _f1r))(_f1r, _f1r, _f1r))
def gamma_projection(pars_i, pars_j, pars_ij):
    """
    Match a pair of gamma distributions to the potential function
    :math:`Ga(t_j | a_j + 1, b_j) Ga(t_i | a_i + 1, b_i) Po(y_{ij} |
    \\mu_{ij} t_i - t_j)`, where :math:`i` is the parent and :math:`j` is
    the child, by minimizing KL divergence.

    :param float pars_i: gamma natural parameters for the parent cavity distribution
    :param float pars_j: gamma natural parameters for the child cavity distribution
    :param float pars_ij: gamma natural parameters for the edge likelihood

    :return: normalizing constant, gamma natural parameters for parent and child
    """

    a_i, b_i = pars_i
    a_j, b_j = pars_j
    y_ij, mu_ij = pars_ij
    a_i += 1
    a_j += 1

    if not _hyp2f1_valid_parameterization(a_i, b_i, a_j, b_j, y_ij, mu_ij):
        return np.nan, pars_i, pars_j

    logl, mn_i, va_i, mn_j, va_j = moments(a_i, b_i, a_j, b_j, y_ij, mu_ij)

    if not _valid_moments(mn_i, va_i) or not _valid_moments(mn_j, va_j):
        return np.nan, pars_i, pars_j

    proj_i = approximate_gamma_mom(mn_i, va_i)
    proj_j = approximate_gamma_mom(mn_j, va_j)

    return logl, np.array(proj_i), np.array(proj_j)


@numba.njit(_tuple((_f, _f1r))(_f, _f1r, _f1r))
def leafward_projection(t_i, pars_j, pars_ij):
    r"""
    Match a gamma distributions to the potential function :math:`Ga(t_j | a_j +
    1, b_j) Po(y_{ij} | \mu_{ij} t_i - t_j)`, where :math:`i` is the parent and
    :math:`j` is the child, by minimizing KL divergence.

    :param float t_i: the age of the parent
    :param float pars_j: gamma natural parameters for the child cavity distribution
    :param float pars_ij: gamma natural parameters for the edge likelihood

    :return: normalizing constant, gamma natural parameters for child
    """

    a_j, b_j = pars_j
    y_ij, mu_ij = pars_ij
    a_j += 1

    if not _hyp1f1_valid_parameterization(t_i, a_j, b_j, y_ij, mu_ij):
        return np.nan, pars_j

    logl, mn_j, va_j = leafward_moments(t_i, a_j, b_j, y_ij, mu_ij)

    if not _valid_moments(mn_j, va_j):
        return np.nan, pars_j

    proj_j = approximate_gamma_mom(mn_j, va_j)

    return logl, np.array(proj_j)


@numba.njit(_tuple((_f, _f1r))(_f, _f1r, _f1r))
def rootward_projection(t_j, pars_i, pars_ij):
    r"""
    Match a gamma distributions to the potential function :math:`Ga(t_i | a_i +
    1, b_i) Po(y_{ij} | \mu_{ij} t_i - t_j)`, where :math:`i` is the parent and
    :math:`j` is the child, by minimizing KL divergence.

    :param float t_j: the age of the child
    :param float pars_i: gamma natural parameters for the parent cavity distribution
    :param float pars_ij: gamma natural parameters for the edge likelihood

    :return: normalizing constant, gamma natural parameters for child
    """

    a_i, b_i = pars_i
    y_ij, mu_ij = pars_ij
    a_i += 1

    if not _hyperu_valid_parameterization(t_j, a_i, b_i, y_ij, mu_ij):
        return np.nan, pars_i

    logl, mn_i, va_i = rootward_moments(t_j, a_i, b_i, y_ij, mu_ij)

    if not _valid_moments(mn_i, va_i):
        return np.nan, pars_i

    proj_i = approximate_gamma_mom(mn_i, va_i)

    return logl, np.array(proj_i)


# --- mutation posteriors from node posteriors --- #

@numba.njit(_unituple(_f, 2)(_f, _f, _f, _f, _f, _f))
def mutation_moments(a_i, b_i, a_j, b_j, y_ij, mu_ij):
    r"""
    log p(t_m, t_i, t_j) = \
        log(t_i - t_j) * y_ij - mu_ij * (t_i - t_j) + \
        log(t_i) * (a_i - 1) - mu_ij * t_i + \
        log(t_j) * (a_j - 1) - mu_ij * t_j - \
        log(t_i - t_j) * int(t_j < t_m < t_i)

    Returns E[t_m], V[t_m].
    """

    a = a_j
    b = a_i + a_j + y_ij
    c = a_j + y_ij + 1
    t = mu_ij + b_i
    z = (mu_ij - b_j) / t

    hyp2f1 = hypergeo._hyp2f1_laplace
    f000 = hyp2f1(a + 0, b + 0, c + 0, z)
    f020 = hyp2f1(a + 0, b + 2, c + 0, z)
    f111 = hyp2f1(a + 1, b + 1, c + 1, z)
    f121 = hyp2f1(a + 1, b + 2, c + 1, z)
    f222 = hyp2f1(a + 2, b + 2, c + 2, z)

    s1 = a * b / c
    s2 = s1 * (a + 1) * (b + 1) / (c + 1)
    d1 = b * (b + 1) / t ** 2
    d2 = d1 * a / c
    d3 = d2 * (a + 1) / (c + 1)

    mn_m = s1 * exp(f111 - f000) / t / 2 * (1 + z) + b / t / 2
    sq_m = d1 * exp(f020 - f000) / 3 + d2 * exp(f121 - f000) / 3 + d3 * exp(f222 - f000) / 3
    va_m = sq_m - mn_m**2

    return mn_m, va_m


@numba.njit(_unituple(_f, 2)(_f, _f, _f, _f, _f))
def mutation_rootward_moments(t_j, a_i, b_i, y_ij, mu_ij):
    r"""
    Calculate gamma sufficient statistics for the PDF proportional to:

    ..math::

        p(x) = \int_{t_j}^\infty Unif(x | t_i, t_j)
        Ga(t_i | a_i, b_i) Po(y | \mu_ij (t_i - t_j)) dt_i

    which models the time :math:`x` of a mutation uniformly distributed between
    parent age :math:`t_i` and child age :math:`t_j`, on a branch with
    :math:`y_{ij}` mutations and total mutation rate :math:`\mu_{ij}`.

    Returns E[x], E[\log x], V[x].
    """

    _, mn_i, va_i = rootward_moments(t_j, a_i, b_i, y_ij, mu_ij)
    mn_m = mn_i / 2 + t_j / 2
    sq_m = (va_i + mn_i**2 + mn_i * t_j + t_j**2) / 3
    va_m = sq_m - mn_m**2

    return mn_m, va_m


@numba.njit(_unituple(_f, 2)(_f, _f, _f, _f, _f))
def mutation_leafward_moments(t_i, a_j, b_j, y_ij, mu_ij):
    r"""
    Calculate gamma sufficient statistics for the PDF proportional to:

    ..math::

        p(x) = \int_0^{t_i} Unif(x | t_i, t_j)
        Ga(t_j | a_j, b_j) Po(y | \mu_ij (t_i - t_j)) dt_j

    which models the time :math:`x` of a mutation uniformly distributed between
    parent age :math:`t_i` and child age :math:`t_j`, on a branch with
    :math:`y_{ij}` mutations and total mutation rate :math:`\mu_{ij}`.

    Returns E[x], E[\log x], V[x].
    """

    _, mn_j, va_j = leafward_moments(t_i, a_j, b_j, y_ij, mu_ij)
    mn_m = mn_j / 2 + t_i / 2
    sq_m = (va_j + mn_j**2 + mn_j * t_i + t_i**2) / 3
    va_m = sq_m - mn_m**2

    return mn_m, va_m


@numba.njit(_f1r(_f1r, _f1r, _f1r))
def mutation_gamma_projection(pars_i, pars_j, pars_ij):
    r"""
    Match a gamma distribution via KL minimization to the potential function

    ..math::

        p(x) = \int_0^\infty \int_0^{t_i} Unif(x | t_i, t_j)
        Ga(t_i | a_i, b_i) Ga(t_j | a_j b_j) Po(y | \mu_ij (t_i - t_j)) dt_j dt_i

    which models the time :math:`x` of a mutation uniformly distributed between
    parent age :math:`t_i` and child age :math:`t_j`, on a branch with
    :math:`y_{ij}` mutations and total mutation rate :math:`\mu_{ij}`.

    TODO: params

    :return: gamma parameters for mutation age
    """

    a_i, b_i = pars_i
    a_j, b_j = pars_j
    y_ij, mu_ij = pars_ij
    a_i += 1
    a_j += 1

    if not _hyp2f1_valid_parameterization(a_i, b_i, a_j, b_j, y_ij, mu_ij):
        return np.full(2, np.nan)

    mn_m, va_m = mutation_moments(a_i, b_i, a_j, b_j, y_ij, mu_ij)

    if not _valid_moments(mn_m, va_m):
        return np.full(2, np.nan)

    proj_m = approximate_gamma_mom(mn_m, va_m)

    return np.array(proj_m)


@numba.njit(_f1r(_f, _f1r, _f1r))
def mutation_leafward_projection(t_i, pars_j, pars_ij):
    r"""
    Match a gamma distribution via KL minimization to the potential function

    ..math::

        p(x) = \int_0^{t_i} Unif(x | t_i, t_j)
        Ga(t_j | a_j, b_j) Po(y | \mu_ij (t_i - t_j)) dt_j

    which models the time :math:`x` of a mutation uniformly distributed between
    parent age :math:`t_i` and child age :math:`t_j`, on a branch with
    :math:`y_{ij}` mutations and total mutation rate :math:`\mu_{ij}`.

    TODO

    :return: gamma parameters for mutation age
    """

    a_j, b_j = pars_j
    y_ij, mu_ij = pars_ij
    a_j += 1

    if not _hyp1f1_valid_parameterization(t_i, a_j, b_j, y_ij, mu_ij):
        return np.full(2, np.nan)

    mn_m, va_m = mutation_leafward_moments(t_i, a_j, b_j, y_ij, mu_ij)

    if not _valid_moments(mn_m, va_m):
        return np.full(2, np.nan)

    proj_m = approximate_gamma_mom(mn_m, va_m)

    return np.array(proj_m)


@numba.njit(_f1r(_f, _f1r, _f1r))
def mutation_rootward_projection(t_j, pars_i, pars_ij):
    r"""
    Match a gamma distribution via KL minimization to the potential function

    ..math::

        p(x) = \int_{t_j}^{\infty} Unif(x | t_i, t_j)
        Ga(t_i | a_i, b_i) Po(y | \mu_ij (t_i - t_j)) dt_i

    which models the time :math:`x` of a mutation uniformly distributed between
    parent age :math:`t_i` and child age :math:`t_j`, on a branch with
    :math:`y_{ij}` mutations and total mutation rate :math:`\mu_{ij}`.

    TODO

    :return: gamma parameters for mutation age
    """

    a_i, b_i = pars_i
    y_ij, mu_ij = pars_ij
    a_i += 1

    if not _hyperu_valid_parameterization(t_j, a_i, b_i, y_ij, mu_ij):
        return np.full(2, np.nan)

    mn_m, va_m = mutation_rootward_moments(t_j, a_i, b_i, y_ij, mu_ij)

    if not _valid_moments(mn_m, va_m):
        return np.full(2, np.nan)

    proj_m = approximate_gamma_mom(mn_m, va_m)

    return np.array(proj_m)


# --- unphased node posteriors --- #

@numba.njit(_unituple(_f, 5)(_f, _f, _f, _f, _f, _f))
def unphased_moments(a_i, b_i, a_j, b_j, y_ij, mu_ij):
    """
    Calculate sufficient statistics for the PDF proportional to :math:`Ga(t_j |
    a_j, b_j) Ga(t_i | a_i, b_i) Po(y_{ij} | \\mu_{ij} t_i + t_j)`, where
    :math:`i` and :math:`j` are parents of the same individual (assumed to be at
    time zero). The logarithmic moments are approximated via a Taylor expansion
    around the mean.

    :param float a_i: the shape parameter of the cavity distribution for the first parent
    :param float b_i: the rate parameter of the cavity distribution for the first parent
    :param float a_j: the shape parameter of the cavity distribution for the second parent
    :param float b_j: the rate parameter of the cavity distribution for the second parent
    :param float y_ij: the number of mutations on the singleton edge pair
    :param float mu_ij: the span-weighted mutation rate of the singleton edge pair

    :return: normalizing constant, E[t_i], E[log t_i], V[t_i],
        E[t_j], E[log t_j], V[t_j]
    """

    a = a_j
    b = a_i + a_j + y_ij
    c = a_j + a_i
    t = mu_ij + b_i
    z = (mu_ij + b_j) / t

    hyp2f1 = hypergeo._hyp2f1_laplace
    f0 = hyp2f1(a + 0, b + 0, c + 0, 1 - z)
    f1 = hyp2f1(a + 1, b + 1, c + 1, 1 - z)
    f2 = hyp2f1(a + 2, b + 2, c + 2, 1 - z)
    s1 = a * b / c
    s2 = s1 * (a + 1) * (b + 1) / (c + 1)
    d1 = s1 * exp(f1 - f0)
    d2 = s2 * exp(f2 - f0)

    logl = f0 + hypergeo._betaln(a_j, a_i) + hypergeo._gammaln(b) - b * log(t)

    mn_j = d1 / t
    sq_j = d2 / t**2
    va_j = sq_j - mn_j**2

    mn_i = b / t - mn_j * z
    sq_i = sq_j * z**2 + (b + 1) * (mn_i - mn_j * z) / t
    va_i = sq_i - mn_i**2

    return logl, mn_i, va_i, mn_j, va_j


@numba.njit(_unituple(_f, 3)(_f, _f, _f, _f, _f))
def unphased_rightward_moments(t_i, a_j, b_j, y_ij, mu_ij):
    """
    Calculate sufficient statistics for the PDF proportional to :math:`Ga(t_j |
    a_j, b_j) Po(y_{ij} | \\mu_{ij} t_i + t_j)`, where :math:`i` and :math:`j`
    are parents of the same individual (assumed to be at time zero). The
    logarithmic moments are approximated via a Taylor expansion around the
    mean.

    :param float t_i: the age of the first parent
    :param float a_j: the shape parameter of the cavity distribution for the second parent
    :param float b_j: the rate parameter of the cavity distribution for the second parent
    :param float y_ij: the number of mutations on the singleton edge pair
    :param float mu_ij: the span-weighted mutation rate of the singleton edge pair

    :return: normalizing constant, E[t_j], E[log t_j], V[t_j]
    """

    assert t_i > 0.0

    a = a_j
    b = a_j + y_ij + 1
    z = t_i * (mu_ij + b_j)

    hyperu = hypergeo._hyperu_laplace
    f0, d0 = hyperu(a + 0, b + 0, z)
    f1, d1 = hyperu(a + 1, b + 1, z)

    logl = f0 - mu_ij * t_i + (b - 1) * log(t_i) + hypergeo._gammaln(a)
    mn_j = -t_i * d0
    va_j = t_i**2 * d0 * (d1 - d0)

    return logl, mn_j, va_j


@numba.njit(_unituple(_f, 3)(_f, _f, _f, _f, _f, _f))
def unphased_mutation_moments(a_i, b_i, a_j, b_j, y_ij, mu_ij):
    r"""
    Calculate gamma sufficient statistics for the PDF proportional to:

    ..math::

        p(x) = \int_0^\infty \int_0^\infty (Unif(x | 0, t_i) + Unif(x | 0, t_j))
        Ga(t_i | a_i, b_i) Ga(t_j | a_j b_j) Po(y | \mu_ij (t_i + t_j)) dt_j dt_i

    which models the time :math:`x` of a mutation uniformly distributed between
    zero and one of two parents with ages :math:`t_i` and :math:`t_j`, where
    the mutation count on both branches is :math:`y_{ij}` with total mutation
    rate :math:`\mu_{ij}`.

    Returns log P[x under i], E[x], E[\log x], V[x].
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
    z = (mu_ij + b_j) / t

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
    va_m = sq_m - mn_m**2
    pr_m = (c - a) / c * exp(f001 - f000)

    return pr_m, mn_m, va_m


@numba.njit(_unituple(_f, 3)(_f, _f, _f, _f, _f))
def unphased_mutation_rightward_moments(t_i, a_j, b_j, y_ij, mu_ij):
    r"""
    Calculate gamma sufficient statistics for the PDF proportional to:

    ..math::

        p(x) = \int_0^\infty (Unif(x | 0, t_i) + Unif(x | 0, t_j))
        Ga(t_j | a_j b_j) Po(y | \mu_ij (t_i + t_j)) dt_j

    which models the time :math:`x` of a mutation uniformly distributed between
    zero and one of two parents with ages :math:`t_i` and :math:`t_j`, where
    the mutation count on both branches is :math:`y_{ij}` with total mutation
    rate :math:`\mu_{ij}`.

    Returns log P[x under i], E[x], E[\log x], V[x].
    """

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

    #with numba.objmode(f00='f8', f10='f8', f21='f8', f32='f8'):
    #   f00 = float(mpmath.log(mpmath.hyperu(a + 0, b + 0, z)))
    #   f10 = float(mpmath.log(mpmath.hyperu(a + 1, b + 0, z)))
    #   f21 = float(mpmath.log(mpmath.hyperu(a + 2, b + 1, z)))
    #   f32 = float(mpmath.log(mpmath.hyperu(a + 3, b + 2, z)))

    # direct but unstable:
    hyperu = hypergeo._hyperu_laplace
    f00, d00 = hyperu(a + 0, b + 0, z)
    f10, d10 = hyperu(a + 1, b + 0, z) 
    f21, d21 = hyperu(a + 2, b + 1, z) 
    f32, d32 = hyperu(a + 3, b + 2, z)
    pr_m = 1.0 - exp(f10 - f00) * a
    mn_m = pr_m * t_i / 2 + t_i * exp(f21 - f00) * a * (a + 1) / 2
    sq_m = pr_m * t_i ** 2 / 3 + t_i ** 2 * exp(f32 - f00) * a * (a + 1) * (a + 2) / 3

    # TODO: use a stabler approach with derivatives
    # note that exp(f10 - f00) = (a + z * d00) / (a - b + 1)
    # however the denominator is 0 if y_ij is 0
    # note that when y_ij == 0 then a == b + 1 and f00 = z**(-a)

    va_m = sq_m - mn_m**2

    return pr_m, mn_m, va_m


@numba.njit(_tuple((_f, _f1r, _f1r))(_f1r, _f1r, _f1r))
def unphased_projection(pars_i, pars_j, pars_ij):
    a_i, b_i = pars_i
    a_j, b_j = pars_j
    y_ij, mu_ij = pars_ij
    a_i += 1
    a_j += 1
    logl, mn_i, va_i, mn_j, va_j = unphased_moments(a_i, b_i, a_j, b_j, y_ij, mu_ij)
    if not _valid_moments(mn_i, va_i) or not _valid_moments(mn_j, va_j):
        return np.nan, pars_i, pars_j
    proj_i = approximate_gamma_mom(mn_i, va_i)
    proj_j = approximate_gamma_mom(mn_j, va_j)
    return logl, np.array(proj_i), np.array(proj_j)
