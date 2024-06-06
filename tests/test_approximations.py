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
# all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""
Test cases for the gamma-variational approximations in tsdate
"""

from math import sqrt

import numpy as np
import pytest
import scipy.integrate
import scipy.special
import scipy.stats
from exact_moments import (
    leafward_moments,
    moments,
    mutation_block_moments,
    mutation_edge_moments,
    mutation_leafward_moments,
    mutation_moments,
    mutation_rootward_moments,
    mutation_sideways_moments,
    mutation_twin_moments,
    mutation_unphased_moments,
    rootward_moments,
    sideways_moments,
    twin_moments,
    unphased_moments,
)

from tsdate import approx, hypergeo

# TODO: better test set?
_gamma_trio_test_cases = [  # [shape1, rate1, shape2, rate2, muts, rate]
    [2.0, 0.0005, 1.5, 0.005, 0.0, 0.001],
    [2.0, 0.0005, 1.5, 0.005, 1.0, 0.001],
    [2.0, 0.0005, 1.5, 0.005, 2.0, 0.001],
    [2.0, 0.0005, 1.5, 0.005, 3.0, 0.001],
]


@pytest.mark.parametrize("pars", _gamma_trio_test_cases)
class TestPosteriorMomentMatching:
    """
    Test Laplace approximation of pairwise joint distributions for EP updates
    """

    def test_moments(self, pars):
        """
        Test mean and variance when ages of both nodes are free
        """
        rtol = 1e-2
        ll, mn_i, va_i, mn_j, va_j = approx.moments(*pars)
        ck_ll, ck_mn_i, ck_va_i, ck_mn_j, ck_va_j = moments(*pars)
        assert np.isclose(ck_ll, ll, rtol=rtol)
        assert np.isclose(ck_mn_i, mn_i, rtol=rtol)
        assert np.isclose(ck_mn_j, mn_j, rtol=rtol)
        assert np.isclose(sqrt(ck_va_i), sqrt(va_i), rtol=rtol)
        assert np.isclose(sqrt(ck_va_j), sqrt(va_j), rtol=rtol)

    def test_rootward_moments(self, pars):
        """
        Test mean and variance of parent age when child age is fixed to a nonzero value
        """
        rtol = 2e-2
        a_i, b_i, a_j, b_j, y, mu = pars
        pars_redux = (a_i, b_i, y, mu)
        mn_j = a_j / b_j
        for t_j in [0.0, mn_j]:
            ll, mn_i, va_i = approx.rootward_moments(t_j, *pars_redux)
            ck_ll, ck_mn_i, ck_va_i = rootward_moments(t_j, *pars_redux)
            assert np.isclose(ck_ll, ll, rtol=rtol)
            assert np.isclose(ck_mn_i, mn_i, rtol=rtol)
            assert np.isclose(sqrt(ck_va_i), sqrt(va_i), rtol=rtol)

    def test_leafward_moments(self, pars):
        """
        Test mean and variance of child age when parent age is fixed to a nonzero value
        """
        rtol = 1e-2
        a_i, b_i, a_j, b_j, y, mu = pars
        t_i = a_i / b_i
        pars_redux = (a_j, b_j, y, mu)
        ll, mn_j, va_j = approx.leafward_moments(t_i, *pars_redux)
        ck_ll, ck_mn_j, ck_va_j = leafward_moments(t_i, *pars_redux)
        assert np.isclose(ck_ll, ll, rtol=rtol)
        assert np.isclose(ck_mn_j, mn_j, rtol=rtol)
        assert np.isclose(sqrt(ck_va_j), sqrt(va_j), rtol=rtol)

    def test_unphased_moments(self, pars):
        """
        Parent ages for an singleton nodes above an unphased individual
        """
        rtol = 1e-2
        ll, mn_i, va_i, mn_j, va_j = approx.unphased_moments(*pars)
        ck_ll, ck_mn_i, ck_va_i, ck_mn_j, ck_va_j = unphased_moments(*pars)
        assert np.isclose(ck_ll, ll, rtol=rtol)
        assert np.isclose(ck_mn_i, mn_i, rtol=rtol)
        assert np.isclose(ck_mn_j, mn_j, rtol=rtol)
        assert np.isclose(sqrt(ck_va_i), sqrt(va_i), rtol=rtol)
        assert np.isclose(sqrt(ck_va_j), sqrt(va_j), rtol=rtol)

    def test_twin_moments(self, pars):
        """
        Parent age for a singleton node above both edges of an unphased
        individual
        """
        a_i, b_i, a_j, b_j, y_ij, mu_ij = pars
        pars_redux = (a_i, b_i, y_ij, mu_ij)
        ll, mn_i, va_i = approx.twin_moments(*pars_redux)
        ck_ll, ck_mn_i, ck_va_i = twin_moments(*pars_redux)
        assert np.isclose(ck_ll, ll)
        assert np.isclose(ck_mn_i, mn_i)
        assert np.isclose(sqrt(ck_va_i), sqrt(va_i))

    def test_sideways_moments(self, pars):
        """
        Parent ages for an singleton nodes above an unphased individual, where
        second parent is fixed to a particular time
        """
        rtol = 1e-2
        a_i, b_i, a_j, b_j, y, mu = pars
        pars_redux = (a_j, b_j, y, mu)
        t_i = a_i / b_i
        ll, mn_j, va_j = approx.sideways_moments(t_i, *pars_redux)
        ck_ll, ck_mn_j, ck_va_j = sideways_moments(t_i, *pars_redux)
        assert np.isclose(ck_ll, ll, rtol=rtol)
        assert np.isclose(ck_mn_j, mn_j, rtol=rtol)
        assert np.isclose(sqrt(ck_va_j), sqrt(va_j), rtol=rtol)

    def test_mutation_moments(self, pars):
        """
        Mutation mapped to a single branch with both nodes free
        """
        rtol = 2e-2
        mn, va = approx.mutation_moments(*pars)
        ck_mn, ck_va = mutation_moments(*pars)
        assert np.isclose(ck_mn, mn, rtol=rtol)
        assert np.isclose(sqrt(ck_va), sqrt(va), rtol=rtol)

    def test_mutation_rootward_moments(self, pars):
        """
        Mutation mapped to a single branch with child node fixed
        """
        rtol = 1e-2
        a_i, b_i, a_j, b_j, y, mu = pars
        pars_redux = (a_i, b_i, y, mu)
        mn_j = a_j / b_j
        for t_j in [0.0, mn_j]:
            mn, va = approx.mutation_rootward_moments(t_j, *pars_redux)
            ck_mn, ck_va = mutation_rootward_moments(t_j, *pars_redux)
            assert np.isclose(ck_mn, mn, rtol=rtol)
            assert np.isclose(sqrt(ck_va), sqrt(va), rtol=rtol)

    def test_mutation_leafward_moments(self, pars):
        """
        Mutation mapped to a single branch with parent node fixed
        """
        rtol = 1e-2
        a_i, b_i, a_j, b_j, y, mu = pars
        t_i = a_i / b_i
        pars_redux = (a_j, b_j, y, mu)
        mn, va = approx.mutation_leafward_moments(t_i, *pars_redux)
        ck_mn, ck_va = mutation_leafward_moments(t_i, *pars_redux)
        assert np.isclose(ck_mn, mn, rtol=rtol)
        assert np.isclose(sqrt(ck_va), sqrt(va), rtol=rtol)

    def test_mutation_unphased_moments(self, pars):
        """
        Mutation mapped to two singleton branches with children fixed to time zero
        """
        rtol = 1e-2
        pr, mn, va = approx.mutation_unphased_moments(*pars)
        ck_pr, ck_mn, ck_va = mutation_unphased_moments(*pars)
        assert np.isclose(ck_pr, pr, rtol=rtol)
        assert np.isclose(ck_mn, mn, rtol=rtol)
        assert np.isclose(sqrt(ck_va), sqrt(va), rtol=rtol)

    def test_mutation_twin_moments(self, pars):
        """
        Mutation mapped to two singleton branches with children fixed to time zero
        and the same parent
        """
        a_i, b_i, a_j, b_j, y_ij, mu_ij = pars
        pars_redux = (a_i, b_i, y_ij, mu_ij)
        pr, mn, va = approx.mutation_twin_moments(*pars_redux)
        ck_pr, ck_mn, ck_va = mutation_twin_moments(*pars_redux)
        assert np.isclose(ck_pr, pr)
        assert np.isclose(ck_mn, mn)
        assert np.isclose(sqrt(ck_va), sqrt(va))

    def test_mutation_sideways_moments(self, pars):
        """
        Mutation mapped to two branches with children fixed to time zero, and
        left parent (i) fixed
        """
        rtol = 1e-2
        a_i, b_i, a_j, b_j, y, mu = pars
        t_i = a_i / b_i
        pars_redux = (a_j, b_j, y, mu)
        pr, mn, va = approx.mutation_sideways_moments(t_i, *pars_redux)
        ck_pr, ck_mn, ck_va = mutation_sideways_moments(t_i, *pars_redux)
        assert np.isclose(ck_pr, pr, rtol=rtol)
        assert np.isclose(ck_mn, mn, rtol=rtol)
        assert np.isclose(sqrt(ck_va), sqrt(va), rtol=rtol)

    def test_mutation_edge_moments(self, pars):
        """
        Mutation mapped to a single edge with parent and child fixed
        """
        a_i, b_i, a_j, b_j, y, mu = pars
        t_i = a_i / b_i
        t_j = a_j / b_j
        mn, va = approx.mutation_edge_moments(t_i, t_j)
        ck_mn, ck_va = mutation_edge_moments(t_i, t_j)
        assert np.isclose(ck_mn, mn)
        assert np.isclose(sqrt(ck_va), sqrt(va))

    def test_mutation_block_moments(self, pars):
        """
        Mutation mapped to two branches with children fixed to time zero, and
        both parents fixed
        """
        a_i, b_i, a_j, b_j, y, mu = pars
        t_i = a_i / b_i
        t_j = a_j / b_j
        pr, mn, va = approx.mutation_block_moments(t_i, t_j)
        ck_pr, ck_mn, ck_va = mutation_block_moments(t_i, t_j)
        assert np.isclose(ck_pr, pr)
        assert np.isclose(ck_mn, mn)
        assert np.isclose(sqrt(ck_va), sqrt(va))

    def test_approximate_gamma_kl(self, pars):
        a_i, b_i, a_j, b_j, y, mu = pars
        t_i = a_i / b_i
        ln_t_i = hypergeo._digamma(a_i) - np.log(b_i)
        t_j = a_j / b_j
        ln_t_j = hypergeo._digamma(a_j) - np.log(b_j)
        alpha_i, beta_i = approx.approximate_gamma_kl(t_i, ln_t_i)
        alpha_j, beta_j = approx.approximate_gamma_kl(t_j, ln_t_j)
        ck_t_i = (alpha_i + 1) / beta_i
        assert np.isclose(t_i, ck_t_i)
        ck_t_j = (alpha_j + 1) / beta_j
        assert np.isclose(t_j, ck_t_j)
        ck_ln_t_i = hypergeo._digamma(alpha_i + 1) - np.log(beta_i)
        assert np.isclose(ln_t_i, ck_ln_t_i)
        ck_ln_t_j = hypergeo._digamma(alpha_j + 1) - np.log(beta_j)
        assert np.isclose(ln_t_j, ck_ln_t_j)

    def test_approximate_gamma_mom(self, pars):
        _, t_i, va_t_i, t_j, va_t_j = approx.moments(*pars)
        alpha_i, beta_i = approx.approximate_gamma_mom(t_i, va_t_i)
        alpha_j, beta_j = approx.approximate_gamma_mom(t_j, va_t_j)
        ck_t_i = (alpha_i + 1) / beta_i
        assert np.isclose(t_i, ck_t_i)
        ck_t_j = (alpha_j + 1) / beta_j
        assert np.isclose(t_j, ck_t_j)
        ck_va_t_i = (alpha_i + 1) / beta_i**2
        assert np.isclose(va_t_i, ck_va_t_i)
        ck_va_t_j = (alpha_j + 1) / beta_j**2
        assert np.isclose(va_t_j, ck_va_t_j)


class TestGammaFactorization:
    """
    Test various functions for manipulating factorizations of gamma distributions
    """

    def test_average_gammas(self):
        # E[x] = shape/rate
        # E[log x] = digamma(shape) - log(rate)
        shape = np.array([0.5, 1.5])
        rate = np.array([1.0, 1.0])
        avg_shape, avg_rate = approx.average_gammas(shape, rate)
        E_x = np.mean(shape + 1)
        E_logx = np.mean(scipy.special.digamma(shape + 1))
        assert np.isclose(E_x, (avg_shape + 1) / avg_rate)
        assert np.isclose(E_logx, scipy.special.digamma(avg_shape + 1) - np.log(avg_rate))


class TestKLMinimizationFailed:
    """
    Test errors in KL minimization
    """

    def test_violates_jensen(self):
        with pytest.raises(approx.KLMinimizationFailedError, match="violates Jensen's"):
            approx.approximate_gamma_kl(1, 0)

    def test_asymptotic_bound(self):
        # check that bound is returned over threshold (rather than optimization)
        logx = -0.000001
        alpha, _ = approx.approximate_gamma_kl(1, logx)
        alpha += 1
        alpha_bound = -0.5 / logx
        assert alpha == alpha_bound
        assert alpha > 1e4
        # check that bound matches optimization result just under threshold
        logx = -0.000051
        alpha, _ = approx.approximate_gamma_kl(1, logx)
        alpha += 1
        alpha_bound = -0.5 / logx
        assert np.abs(alpha - alpha_bound) < 1
        assert alpha < 1e4
