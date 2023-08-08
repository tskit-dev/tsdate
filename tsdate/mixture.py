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
Mixture of gamma distributions that may be fit via EM to distribution-valued observations
"""
import numba
import numpy as np

from . import approx
from . import hypergeo


@numba.njit("UniTuple(f8[:], 4)(f8[:], f8[:], f8[:], f8, f8)")
def _conditional_posterior(prior_logweight, prior_alpha, prior_beta, alpha, beta):
    r"""
    Return expectations of node age :math:`t` from the mixture model,

    ..math::

        Ga(t | a, b) \sum_j \pi_j w_j Ga(t | \alpha_j, \beta_j)

    where :math:`a` and :math:`b` are variational parameters,
    and :math:`\pi_j, \alpha_j, \beta_j` are prior weights and
    parameters for a gamma mixture; and :math:`w_j` are fixed,
    observation-specific weights. We use natural parameterization,
    so that the shape parameter is :math:`\alpha + 1`.

    TODO:
    The normalizing constants of the prior are assumed to have already
    been integrated into `prior_weight`.

    Returns the contribution from each component to the
    posterior expectations of :math:`E[1]`, :math:`E[t]`, :math:`E[log t]`,
    and :math:`E[t log t]`.

    Note that :math:`E[1]` is *unnormalized* and *log-transformed*.
    """

    dim = prior_logweight.size
    E = np.full(dim, -np.inf)  # E[1] (e.g. normalizing constant)
    E_t = np.zeros(dim)  # E[t]
    E_logt = np.zeros(dim)  # E[log(t)]
    E_tlogt = np.zeros(dim)  # E[t * log(t)]
    C = (alpha + 1) * np.log(beta) - hypergeo._gammaln(alpha + 1) if beta > 0 else 0.0
    for i in range(dim):
        post_alpha = prior_alpha[i] + alpha
        post_beta = prior_beta[i] + beta
        if (post_alpha <= -1) or (post_beta <= 0):  # skip node if divergent
            E[:] = -np.inf
            break
        E[i] = C + (
            +hypergeo._gammaln(post_alpha + 1)
            - (post_alpha + 1) * np.log(post_beta)
            + prior_logweight[i]
        )
        assert np.isfinite(E[i])
        # TODO: option to use moments instead of sufficient statistics?
        E_t[i] = (post_alpha + 1) / post_beta
        E_logt[i] = hypergeo._digamma(post_alpha + 1) - np.log(post_beta)
        E_tlogt[i] = E_t[i] * E_logt[i] + E_t[i] / (post_alpha + 1)

    return E, E_t, E_logt, E_tlogt


@numba.njit("f8(f8[:], f8[:], f8[:], f8[:], f8[:])")
def _em_update(prior_weight, prior_alpha, prior_beta, alpha, beta):
    """
    Perform an expectation maximization step for parameters of mixture components,
    given variational parameters `alpha`, `beta` for each node.

    The maximization step is performed using Ye & Chen (2017) "Closed form
    estimators for the gamma distribution ..."

    ``prior_weight``, ``prior_alpha``, ``prior_beta`` are updated in place.
    """
    assert alpha.size == beta.size

    dim = prior_weight.size
    n = np.zeros(dim)
    t = np.zeros(dim)
    logt = np.zeros(dim)
    tlogt = np.zeros(dim)

    # incorporate prior normalizing constants into weights
    prior_logweight = np.log(prior_weight)
    for k in range(dim):
        prior_logweight[k] += (prior_alpha[k] + 1) * np.log(
            prior_beta[k]
        ) - hypergeo._gammaln(prior_alpha[k] + 1)

    # expectation step:
    loglik = 0.0
    for a, b in zip(alpha, beta):
        E, E_t, E_logt, E_tlogt = _conditional_posterior(
            prior_logweight, prior_alpha, prior_beta, a, b
        )

        # skip if posterior is improper
        if np.any(np.isinf(E)):
            continue

        # convert evidence to posterior weights
        norm_const = np.log(np.sum(np.exp(E - np.max(E)))) + np.max(E)
        weight = np.exp(E - norm_const)

        # weighted contributions to sufficient statistics
        loglik += norm_const
        n += weight
        t += E_t * weight
        logt += E_logt * weight
        tlogt += E_tlogt * weight

    # maximization step: update parameters in place
    prior_weight[:] = n / np.sum(n)
    prior_beta[:] = n**2 / (n * tlogt - t * logt)
    prior_alpha[:] = n * t / (n * tlogt - t * logt) - 1.0

    return loglik


@numba.njit("f8[:](f8[:], f8[:], f8[:], f8[:], f8[:])")
def _gamma_projection(prior_weight, prior_alpha, prior_beta, alpha, beta):
    """
    Given variational approximation to posterior: multiply by exact prior,
    calculate sufficient statistics, and moment match to get new
    approximate posterior.

    Updates ``alpha`` and ``beta`` in-place.
    """
    assert alpha.size == beta.size

    dim = prior_weight.size

    # incorporate prior normalizing constants into weights
    prior_logweight = np.log(prior_weight)
    for k in range(dim):
        prior_logweight[k] += (prior_alpha[k] + 1) * np.log(
            prior_beta[k]
        ) - hypergeo._gammaln(prior_alpha[k] + 1)

    log_const = np.full(alpha.size, -np.inf)
    for i in range(alpha.size):
        E, E_t, E_logt, E_tlogt = _conditional_posterior(
            prior_logweight, prior_alpha, prior_beta, alpha[i], beta[i]
        )

        # skip if posterior is improper for all components
        if np.any(np.isinf(E)):
            continue

        norm = np.log(np.sum(np.exp(E - np.max(E)))) + np.max(E)
        weight = np.exp(E - norm)
        t = np.sum(weight * E_t)
        logt = np.sum(weight * E_logt)
        # tlogt = np.sum(weight * E_tlogt)
        log_const[i] = norm
        alpha[i], beta[i] = approx.approximate_gamma_kl(t, logt)
        # beta[i] = 1 / (tlogt - t * logt)
        # alpha[i] = t * beta[i] - 1.0

    return log_const


@numba.njit("Tuple((f8[:,:], f8[:,:]))(f8[:,:], f8[:,:], i4, f8, b1)")
def fit_gamma_mixture(mixture, observations, max_iterations, tolerance, verbose):
    """
    Run EM until relative tolerance or maximum number of iterations is
    reached.  Then, perform expectation-propagation update and return new
    variational parameters for the posterior approximation.
    """

    assert mixture.shape[1] == 3
    assert observations.shape[1] == 2

    mix_weight, mix_alpha, mix_beta = mixture.T
    alpha, beta = observations.T

    last = np.inf
    for itt in range(max_iterations):
        loglik = _em_update(mix_weight, mix_alpha, mix_beta, alpha, beta)
        loglik /= float(alpha.size)
        update = np.abs(loglik - last)
        last = loglik
        if verbose:
            print("EM iteration:", itt, "mean(loglik):", np.round(loglik, 5))
            print("  -> weights:", mix_weight)
            print("  -> alpha:", mix_alpha)
            print("  -> beta:", mix_beta)
        if update < np.abs(loglik) * tolerance:
            break

    # conditional posteriors for each observation
    # log_const = _gamma_projection(mix_weight, mix_alpha, mix_beta, alpha, beta)
    _gamma_projection(mix_weight, mix_alpha, mix_beta, alpha, beta)

    new_mixture = np.zeros(mixture.shape)
    new_observations = np.zeros(observations.shape)
    new_observations[:, 0] = alpha
    new_observations[:, 1] = beta
    new_mixture[:, 0] = mix_weight
    new_mixture[:, 1] = mix_alpha
    new_mixture[:, 2] = mix_beta

    return new_mixture, new_observations


def initialize_mixture(parameters, num_components):
    """initialize clusters by dividing nodes into equal groups"""
    global_prior = np.empty((num_components, 3))
    num_nodes = parameters.shape[0]
    age_classes = np.tile(np.arange(num_components), num_nodes // num_components + 1)[
        :num_nodes
    ]
    for k in range(num_components):
        indices = np.equal(age_classes, k)
        alpha, beta = approx.average_gammas(
            parameters[indices, 0] - 1.0, parameters[indices, 1]
        )
        global_prior[k] = [1.0 / num_components, alpha, beta]
    return global_prior
