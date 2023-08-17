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
import logging

import numba
import numpy as np
import scipy.stats

from . import approx
from . import hypergeo


@numba.njit("UniTuple(f8[:], 4)(f8[:], f8[:], f8[:], f8, f8)")
def _conditional_posterior(prior_weight, prior_shape, prior_rate, shape, rate):
    r"""
    Return expectations of node age :math:`t` from the mixture model,

    ..math::

        Ga(t | a, b) \sum_j \pi_j w_j Ga(t | \alpha_j, \beta_j)

    where :math:`a` and :math:`b` are variational parameters,
    and :math:`\pi_j, \alpha_j, \beta_j` are prior weights and
    parameters for a gamma mixture; and :math:`w_j` are fixed,
    observation-specific weights.

    Returns the contribution from each component to the
    posterior expectations of :math:`E[1]`, :math:`E[t]`, :math:`E[log t]`,
    and :math:`E[t log t]`.

    Note that :math:`E[1]` is *unnormalized* and *log-transformed*.
    """

    dim = prior_weight.size
    E = np.full(dim, -np.inf)  # E[1] (e.g. normalizing constant)
    E_t = np.zeros(dim)  # E[t]
    E_logt = np.zeros(dim)  # E[log(t)]
    E_tlogt = np.zeros(dim)  # E[t * log(t)]
    for i in range(dim):
        post_shape = prior_shape[i] + shape - 1
        post_rate = prior_rate[i] + rate
        # TODO: if observation shape parameters are too small, negative shape
        # parameters will occur. Could skip observation when doing updates, but
        # will need a workaround for projection.
        assert post_shape > 0 and post_rate > 0
        E[i] = (
            prior_shape[i] * np.log(prior_rate[i])
            - hypergeo._gammaln(prior_shape[i])
            + shape * np.log(rate)
            - hypergeo._gammaln(shape)
            + hypergeo._gammaln(post_shape)
            - post_shape * np.log(post_rate)
            + np.log(prior_weight[i])
        )
        E_t[i] = post_shape / post_rate
        E_logt[i] = hypergeo._digamma(post_shape) - np.log(post_rate)
        E_tlogt[i] = E_t[i] * E_logt[i] + E_t[i] / post_shape

    return E, E_t, E_logt, E_tlogt


@numba.njit("Tuple((f8, f8[:], f8[:], f8[:]))(f8[:], f8[:], f8[:], f8[:], f8[:])")
def _em_update(prior_weight, prior_shape, prior_rate, shape, rate):
    """
    Perform an expectation maximization step for parameters of mixture components,
    given variational parameters (shape and rate) for each node.

    The maximization step is performed using Ye & Chen (2017) "Closed form
    estimators for the gamma distribution ..."
    """
    assert np.all(prior_weight > 0)
    assert shape.size == rate.size

    dim = prior_weight.size
    n = np.zeros(dim)
    t = np.zeros(dim)
    logt = np.zeros(dim)
    tlogt = np.zeros(dim)

    # expectation step:
    loglik = 0.0
    for alpha, beta in zip(shape, rate):
        E, E_t, E_logt, E_tlogt = _conditional_posterior(
            prior_weight, prior_shape, prior_rate, alpha, beta
        )

        # convert evidence to posterior weights
        norm_const = np.log(np.sum(np.exp(E - np.max(E)))) + np.max(E)
        weight = np.exp(E - norm_const)

        # weighted contributions to sufficient statistics
        loglik += norm_const
        n += weight
        t += E_t * weight
        logt += E_logt * weight
        tlogt += E_tlogt * weight

    # maximization step:
    new_weight = n / np.sum(n)
    new_rate = n**2 / (n * tlogt - t * logt)
    new_shape = n * t / (n * tlogt - t * logt)

    return loglik, new_weight, new_shape, new_rate


@numba.njit("UniTuple(f8[:], 3)(f8[:], f8[:], f8[:], f8[:], f8[:])")
def _gamma_projection(prior_weight, prior_shape, prior_rate, shape, rate):
    """
    Given variational approximation to posterior: multiply by exact prior,
    calculate sufficient statistics, and moment match to get new
    approximate posterior.
    """
    assert shape.size == rate.size

    post_shape = np.zeros(shape.size)
    post_rate = np.zeros(rate.size)
    log_const = np.zeros(shape.size)
    for i, (alpha, beta) in enumerate(zip(shape, rate)):
        E, E_t, E_logt, E_tlogt = _conditional_posterior(
            prior_weight, prior_shape, prior_rate, alpha, beta
        )
        norm = np.log(np.sum(np.exp(E - np.max(E)))) + np.max(E)
        weight = np.exp(E - norm)
        t = np.sum(weight * E_t)
        logt = np.sum(weight * E_logt)
        log_const[i] = norm
        post_shape[i], post_rate[i] = approx.approximate_gamma_kl(t, logt)

    return log_const, post_shape, post_rate


class GammaMixture:
    """
    TODO
    """

    def __init__(self, weight, shape, rate):
        """
        TODO set fixed=False to indicate no optimization
        """
        assert weight.ndim == shape.ndim == rate.ndim == 1
        assert rate.size == shape.size == weight.size
        assert np.all(weight > 0) and np.all(shape > 0) and np.all(rate > 0)
        self.dim = weight.size
        self.weight = weight / np.sum(weight)
        self.shape = shape
        self.rate = rate

    def as_dict(self):
        return {
            "weight": list(self.weight),
            "shape": list(self.shape),
            "rate": list(self.rate),
            # "fixed": self.fixed,
        }

    def pdf(self, x):
        logpdf = scipy.stats.gamma.logpdf(x, self.shape, scale=1.0 / self.rate)
        logpdf += np.log(self.weight)
        norm = logpdf.max()
        logpdf = np.log(np.sum(np.exp(logpdf - norm))) + norm
        return np.exp(logpdf)

    def propagate(self, observations, max_iterations=100, tolerance=1e-6):
        """
        Run EM until relative tolerance or maximum number of iterations is
        reached.  Then, perform expectation-propagation update and return new
        shape, rate parameters for the posterior approximation.
        """

        shape, rate = observations[:, 0], observations[:, 1]

        # max_iterations *= 1 - int(self.fixed)

        last = np.inf
        for itt in range(max_iterations):
            loglik, self.weight, self.shape, self.rate = _em_update(
                self.weight, self.shape, self.rate, shape, rate
            )
            loglik /= float(shape.size)
            delta = np.abs(loglik - last)
            last = loglik
            logging.info(f"EM: Iteration {itt}, loglikelihood {loglik:.3f}")
            if delta < np.abs(loglik) * tolerance:
                logging.info("EM: Converged")
                break
        if itt + 1 == max_iterations:
            logging.info("EM: Reached maximum number of iterations")

        log_const, post_shape, post_rate = _gamma_projection(
            self.weight, self.shape, self.rate, shape, rate
        )

        return log_const, np.column_stack([post_shape, post_rate])
