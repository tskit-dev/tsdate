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
import numpy as np
import pytest
import scipy.integrate
import scipy.special
import scipy.stats

from tsdate import approx
from tsdate import hypergeo
from tsdate import prior

# TODO: better test set?
# TODO: test special case where child is fixed to age 0
_gamma_trio_test_cases = [  # [shape1, rate1, shape2, rate2, muts, rate]
    [2.0, 0.0005, 1.5, 0.005, 0.0, 0.001],
    [2.0, 0.0005, 1.5, 0.005, 1.0, 0.001],
    [2.0, 0.0005, 1.5, 0.005, 2.0, 0.001],
    [2.0, 0.0005, 1.5, 0.005, 3.0, 0.001],
]


@pytest.mark.parametrize("pars", _gamma_trio_test_cases)
class TestPosteriorMomentMatching:
    """
    Test approximation of marginal pairwise joint distributions by a gamma via
    moment matching of sufficient statistics
    """

    @staticmethod
    def pdf(t_i, t_j, a_i, b_i, a_j, b_j, y, mu):
        """
        Target joint (pair) distribution, proportional to the parent/child
        marginals (gamma) and a Poisson mutation likelihood
        """
        assert 0 < t_j < t_i
        return (
            t_i ** (a_i - 1)
            * np.exp(-t_i * b_i)
            * t_j ** (a_j - 1)
            * np.exp(-t_j * b_j)
            * (t_i - t_j) ** y
            * np.exp(-(t_i - t_j) * mu)
        )

    @staticmethod
    def pdf_rootward(t_i, t_j, a_i, b_i, y, mu):
        """
        Target conditional distribution, proportional to the parent
        marginals (gamma) and a Poisson mutation likelihood at a
        fixed child age
        """
        assert 0 <= t_j < t_i
        return (
            t_i ** (a_i - 1)
            * np.exp(-t_i * b_i)
            * (t_i - t_j) ** y
            * np.exp(-(t_i - t_j) * mu)
        )

    @staticmethod
    def pdf_leafward(t_i, t_j, a_j, b_j, y, mu):
        """
        Target conditional distribution, proportional to the child
        marginals (gamma) and a Poisson mutation likelihood at a
        fixed parent age
        """
        assert 0 < t_j < t_i
        return (
            t_j ** (a_j - 1)
            * np.exp(-t_j * b_j)
            * (t_i - t_j) ** y
            * np.exp(-(t_i - t_j) * mu)
        )

    @staticmethod
    def pdf_unphased(t_i, t_j, a_i, b_i, a_j, b_j, y, mu):
        """
        Target joint (pair) distribution, proportional to the parent
        marginals (gamma) and a Poisson mutation likelihood over the
        two branches leading from (present-day) individual to parents
        """
        assert t_i > 0 and t_j > 0
        return (
            t_i ** (a_i - 1)
            * np.exp(-t_i * b_i)
            * t_j ** (a_j - 1)
            * np.exp(-t_j * b_j)
            * (t_i + t_j) ** y
            * np.exp(-(t_i + t_j) * mu)
        )

    @staticmethod
    def pdf_unphased_rightward(t_i, t_j, a_j, b_j, y, mu):
        """
        Target joint (pair) distribution, proportional to the parent
        marginals (gamma) and a Poisson mutation likelihood over the
        two branches leading from (present-day) individual to parents,
        with left parent fixed to t_i
        """
        assert t_i > 0 and t_j > 0
        return (
            t_j ** (a_j - 1)
            * np.exp(-t_j * b_j)
            * (t_i + t_j) ** y
            * np.exp(-(t_i + t_j) * mu)
        )

    def test_moments(self, pars):
        """
        Test mean and variance when ages of both nodes are free
        """
        logconst, t_i, var_t_i, t_j, var_t_j = approx.moments(*pars)
        ck_normconst = scipy.integrate.dblquad(
            lambda t_i, t_j: self.pdf(t_i, t_j, *pars),
            0,
            np.inf,
            lambda t_j: t_j,
            np.inf,
            epsabs=0,
        )[0]
        assert np.isclose(logconst, np.log(ck_normconst), rtol=2e-2)
        ck_t_i = scipy.integrate.dblquad(
            lambda t_i, t_j: t_i * self.pdf(t_i, t_j, *pars) / ck_normconst,
            0,
            np.inf,
            lambda t_j: t_j,
            np.inf,
            epsabs=0,
        )[0]
        assert np.isclose(t_i, ck_t_i, rtol=2e-2)
        ck_t_j = scipy.integrate.dblquad(
            lambda t_i, t_j: t_j * self.pdf(t_i, t_j, *pars) / ck_normconst,
            0,
            np.inf,
            lambda t_j: t_j,
            np.inf,
            epsabs=0,
        )[0]
        assert np.isclose(t_j, ck_t_j, rtol=2e-2)
        ck_var_t_i = (
            scipy.integrate.dblquad(
                lambda t_i, t_j: t_i**2 * self.pdf(t_i, t_j, *pars) / ck_normconst,
                0,
                np.inf,
                lambda t_j: t_j,
                np.inf,
                epsabs=0,
            )[0]
            - ck_t_i**2
        )
        assert np.isclose(var_t_i, ck_var_t_i, rtol=2e-2)
        ck_var_t_j = (
            scipy.integrate.dblquad(
                lambda t_i, t_j: t_j**2 * self.pdf(t_i, t_j, *pars) / ck_normconst,
                0,
                np.inf,
                lambda t_j: t_j,
                np.inf,
                epsabs=0,
            )[0]
            - ck_t_j**2
        )
        assert np.isclose(var_t_j, ck_var_t_j, rtol=2e-2)

    def test_rootward_moments(self, pars):
        """
        Test mean and variance of parent age when child age is fixed to a nonzero value
        """
        a_i, b_i, a_j, b_j, y, mu = pars
        pars_redux = (a_i, b_i, y, mu)
        mn_j = a_j / b_j  # point "estimate" for child
        for t_j in [0.0, mn_j]:
            logconst, t_i, var_t_i = approx.rootward_moments(t_j, *pars_redux)
            ck_normconst = scipy.integrate.quad(
                lambda t_i: self.pdf_rootward(t_i, t_j, *pars_redux),
                t_j,
                np.inf,
                epsabs=0,
            )[0]
            assert np.isclose(logconst, np.log(ck_normconst), rtol=2e-2)
            ck_t_i = scipy.integrate.quad(
                lambda t_i: t_i * self.pdf_rootward(t_i, t_j, *pars_redux) / ck_normconst,
                t_j,
                np.inf,
                epsabs=0,
            )[0]
            assert np.isclose(t_i, ck_t_i, rtol=2e-2)
            ck_var_t_i = (
                scipy.integrate.quad(
                    lambda t_i: t_i**2
                    * self.pdf_rootward(t_i, t_j, *pars_redux)
                    / ck_normconst,
                    t_j,
                    np.inf,
                    epsabs=0,
                )[0]
                - ck_t_i**2
            )
            assert np.isclose(var_t_i, ck_var_t_i, rtol=2e-2)

    def test_leafward_moments(self, pars):
        """
        Test mean and variance of child age when parent age is fixed to a nonzero value
        """
        a_i, b_i, a_j, b_j, y, mu = pars
        t_i = a_i / b_i  # point "estimate" for parent
        pars_redux = (a_j, b_j, y, mu)
        logconst, t_j, var_t_j = approx.leafward_moments(t_i, *pars_redux)
        ck_normconst = scipy.integrate.quad(
            lambda t_j: self.pdf_leafward(t_i, t_j, *pars_redux),
            0,
            t_i,
            epsabs=0,
        )[0]
        assert np.isclose(logconst, np.log(ck_normconst), rtol=2e-2)
        ck_t_j = scipy.integrate.quad(
            lambda t_j: t_j * self.pdf_leafward(t_i, t_j, *pars_redux) / ck_normconst,
            0,
            t_i,
            epsabs=0,
        )[0]
        assert np.isclose(t_j, ck_t_j, rtol=2e-2)
        ck_var_t_j = (
            scipy.integrate.quad(
                lambda t_j: t_j**2
                * self.pdf_leafward(t_i, t_j, *pars_redux)
                / ck_normconst,
                0,
                t_i,
                epsabs=0,
            )[0]
            - ck_t_j**2
        )
        assert np.isclose(var_t_j, ck_var_t_j, rtol=2e-2)

    def test_unphased_moments(self, pars):
        """
        Parent ages for an singleton nodes above an unphased individual
        """
        logconst, t_i, var_t_i, t_j, var_t_j = approx.unphased_moments(*pars)
        ck_normconst = scipy.integrate.dblquad(
            lambda t_i, t_j: self.pdf_unphased(t_i, t_j, *pars),
            0,
            np.inf,
            0,
            np.inf,
            epsabs=0,
        )[0]
        assert np.isclose(logconst, np.log(ck_normconst), rtol=2e-2)
        ck_t_i = scipy.integrate.dblquad(
            lambda t_i, t_j: t_i * self.pdf_unphased(t_i, t_j, *pars) / ck_normconst,
            0,
            np.inf,
            0,
            np.inf,
            epsabs=0,
        )[0]
        assert np.isclose(t_i, ck_t_i, rtol=2e-2)
        ck_t_j = scipy.integrate.dblquad(
            lambda t_i, t_j: t_j * self.pdf_unphased(t_i, t_j, *pars) / ck_normconst,
            0,
            np.inf,
            0,
            np.inf,
            epsabs=0,
        )[0]
        assert np.isclose(t_j, ck_t_j, rtol=2e-2)
        ck_var_t_i = (
            scipy.integrate.dblquad(
                lambda t_i, t_j: t_i**2 * self.pdf_unphased(t_i, t_j, *pars) / ck_normconst,
                0,
                np.inf,
                0,
                np.inf,
                epsabs=0,
            )[0]
            - ck_t_i**2
        )
        assert np.isclose(var_t_i, ck_var_t_i, rtol=2e-2)
        ck_var_t_j = (
            scipy.integrate.dblquad(
                lambda t_i, t_j: t_j**2 * self.pdf_unphased(t_i, t_j, *pars) / ck_normconst,
                0,
                np.inf,
                0,
                np.inf,
                epsabs=0,
            )[0]
            - ck_t_j**2
        )
        assert np.isclose(var_t_j, ck_var_t_j, rtol=2e-2)

    def test_unphased_rightward_moments(self, pars):
        """
        Parent ages for an singleton nodes above an unphased individual, where
        second parent is fixed to a particular time
        """
        a_i, b_i, a_j, b_j, y, mu = pars
        pars_redux = (a_j, b_j, y, mu)
        t_i = a_i / b_i  # point "estimate" for left parent
        nc, mn, va = approx.unphased_rightward_moments(t_i, *pars_redux)
        ck_nc = scipy.integrate.quad(
            lambda t_j: self.pdf_unphased_rightward(t_i, t_j, *pars_redux),
            0,
            np.inf,
        )[0]
        assert np.isclose(np.exp(nc), ck_nc, rtol=2e-2)
        ck_mn = scipy.integrate.quad(
            lambda t_j: t_j * self.pdf_unphased_rightward(t_i, t_j, *pars_redux),
            0,
            np.inf,
        )[0] / ck_nc
        assert np.isclose(mn, ck_mn, rtol=2e-2)
        ck_va = scipy.integrate.quad(
            lambda t_j: t_j**2 * self.pdf_unphased_rightward(t_i, t_j, *pars_redux),
            0,
            np.inf,
        )[0] / ck_nc - ck_mn**2
        assert np.isclose(va, ck_va, rtol=2e-2)

    def test_mutation_moments(self, pars):
        """
        Mutation mapped to a single branch with both nodes free
        """
        def f(t_i, t_j):
            assert t_j < t_i
            mn = t_i / 2 + t_j / 2
            sq = (t_i**2 + t_i*t_j + t_j**2) / 3
            return mn, sq
        mn, va = approx.mutation_moments(*pars)
        nc = scipy.integrate.dblquad(
            lambda t_i, t_j: self.pdf(t_i, t_j, *pars),
            0,
            np.inf,
            lambda t_j: t_j,
            np.inf,
            epsabs=0,
        )[0]
        ck_mn = scipy.integrate.dblquad(
            lambda t_i, t_j: f(t_i, t_j)[0] * self.pdf(t_i, t_j, *pars),
            0,
            np.inf,
            lambda t_j: t_j,
            np.inf,
            epsabs=0,
        )[0] / nc
        assert np.isclose(mn, ck_mn, rtol=2e-2)
        ck_va = scipy.integrate.dblquad(
            lambda t_i, t_j: f(t_i, t_j)[1] * self.pdf(t_i, t_j, *pars),
            0,
            np.inf,
            lambda t_j: t_j,
            np.inf,
            epsabs=0,
        )[0] / nc - ck_mn**2
        assert np.isclose(va, ck_va, rtol=5e-2)

    def test_mutation_rootward_moments(self, pars):
        """
        Mutation mapped to a single branch with child node fixed
        """
        def f(t_i, t_j): # conditional moments
            assert t_j < t_i
            mn = t_i / 2 + t_j / 2
            sq = (t_i**2 + t_i*t_j + t_j**2) / 3
            return mn, sq
        a_i, b_i, a_j, b_j, y, mu = pars
        pars_redux = (a_i, b_i, y, mu)
        mn_j = a_j / b_j  # point "estimate" for child
        for t_j in [0.0, mn_j]:
            mn, va = approx.mutation_rootward_moments(t_j, *pars_redux)
            nc = scipy.integrate.quad(
                lambda t_i: self.pdf_rootward(t_i, t_j, *pars_redux),
                t_j,
                np.inf,
            )[0]
            ck_mn = scipy.integrate.quad(
                lambda t_i: f(t_i, t_j)[0] * self.pdf_rootward(t_i, t_j, *pars_redux),
                t_j,
                np.inf,
            )[0] / nc
            assert np.isclose(mn, ck_mn, rtol=2e-2)
            ck_va = scipy.integrate.quad(
                lambda t_i: f(t_i, t_j)[1] * self.pdf_rootward(t_i, t_j, *pars_redux),
                t_j,
                np.inf,
            )[0] / nc - ck_mn**2
            assert np.isclose(va, ck_va, rtol=2e-2)

    def test_mutation_leafward_moments(self, pars):
        """
        Mutation mapped to a single branch with parent node fixed
        """
        def f(t_i, t_j):
            assert t_j < t_i
            mn = t_i / 2 + t_j / 2
            sq = (t_i**2 + t_i*t_j + t_j**2) / 3
            return mn, sq
        a_i, b_i, a_j, b_j, y, mu = pars
        t_i = a_i / b_i  # point "estimate" for parent
        pars_redux = (a_j, b_j, y, mu)
        mn, va = approx.mutation_leafward_moments(t_i, *pars_redux)
        nc = scipy.integrate.quad(
            lambda t_j: self.pdf_leafward(t_i, t_j, *pars_redux),
            0,
            t_i,
        )[0]
        ck_mn = scipy.integrate.quad(
            lambda t_j: f(t_i, t_j)[0] * self.pdf_leafward(t_i, t_j, *pars_redux),
            0,
            t_i,
        )[0] / nc
        assert np.isclose(mn, ck_mn, rtol=2e-2)
        ck_va = scipy.integrate.quad(
            lambda t_j: f(t_i, t_j)[1] * self.pdf_leafward(t_i, t_j, *pars_redux),
            0,
            t_i,
        )[0] / nc - ck_mn**2
        assert np.isclose(va, ck_va, rtol=2e-2)

    def test_unphased_mutation_moments(self, pars):
        """
        Mutation mapped to two singleton branches with children fixed to time zero
        """
        def f(t_i, t_j): # conditional moments
            pr = t_i / (t_i + t_j)
            mn = pr * t_i / 2 + (1 - pr) * t_j / 2
            sq = pr * t_i**2 / 3 + (1 - pr) * t_j**2 / 3
            return pr, mn, sq
        pr, mn, va = approx.unphased_mutation_moments(*pars)
        nc = scipy.integrate.dblquad(
            lambda t_i, t_j: self.pdf_unphased(t_i, t_j, *pars),
            0,
            np.inf,
            0,
            np.inf,
            epsabs=0,
        )[0]
        ck_pr = scipy.integrate.dblquad(
            lambda t_i, t_j: f(t_i, t_j)[0] * self.pdf_unphased(t_i, t_j, *pars),
            0,
            np.inf,
            0,
            np.inf,
            epsabs=0,
        )[0] / nc
        assert np.isclose(pr, ck_pr, rtol=2e-2)
        ck_mn = scipy.integrate.dblquad(
            lambda t_i, t_j: f(t_i, t_j)[1] * self.pdf_unphased(t_i, t_j, *pars),
            0,
            np.inf,
            0,
            np.inf,
            epsabs=0,
        )[0] / nc
        assert np.isclose(mn, ck_mn, rtol=2e-2)
        ck_va = scipy.integrate.dblquad(
            lambda t_i, t_j: f(t_i, t_j)[2] * self.pdf_unphased(t_i, t_j, *pars),
            0,
            np.inf,
            0,
            np.inf,
            epsabs=0,
        )[0] / nc - ck_mn**2
        assert np.isclose(va, ck_va, rtol=2e-2)

    def test_unphased_mutation_rightward_moments(self, pars):
        """
        Mutation mapped to two branches with children fixed to time zero, and
        left parent (i) fixed
        """
        def f(t_i, t_j):  # conditional moments
            pr = t_i / (t_i + t_j)
            mn = pr * t_i / 2 + (1 - pr) * t_j / 2
            sq = pr * t_i**2 / 3 + (1 - pr) * t_j**2 / 3
            return pr, mn, sq
        a_i, b_i, a_j, b_j, y, mu = pars
        t_i = a_i / b_i  # point "estimate" for left parent
        pars_redux = (a_j, b_j, y, mu)
        pr, mn, va = approx.unphased_mutation_rightward_moments(t_i, *pars_redux)
        nc = scipy.integrate.quad(
            lambda t_j: self.pdf_unphased_rightward(t_i, t_j, *pars_redux),
            0,
            np.inf,
        )[0]
        ck_pr = scipy.integrate.quad(
            lambda t_j: f(t_i, t_j)[0] * self.pdf_unphased_rightward(t_i, t_j, *pars_redux),
            0,
            np.inf,
        )[0] / nc
        assert np.isclose(pr, ck_pr, rtol=2e-2)
        ck_mn = scipy.integrate.quad(
            lambda t_j: f(t_i, t_j)[1] * self.pdf_unphased_rightward(t_i, t_j, *pars_redux),
            0,
            np.inf,
        )[0] / nc
        assert np.isclose(mn, ck_mn, rtol=2e-2)
        ck_va = scipy.integrate.quad(
            lambda t_j: f(t_i, t_j)[2] * self.pdf_unphased_rightward(t_i, t_j, *pars_redux),
            0,
            np.inf,
        )[0] / nc - ck_mn**2
        assert np.isclose(va, ck_va, rtol=2e-2)

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
        assert np.isclose(
            E_logx, scipy.special.digamma(avg_shape + 1) - np.log(avg_rate)
        )


class TestKLMinimizationFailed:
    """
    Test errors in KL minimization
    """

    def test_violates_jensen(self):
        with pytest.raises(approx.KLMinimizationFailed, match="violates Jensen's"):
            approx.approximate_gamma_kl(1, 0)

    def test_asymptotic_bound(self):
        # check that bound is returned over threshold (rather than optimization)
        logx = -0.000001
        alpha, _ = approx.approximate_gamma_kl(1, logx)
        alpha += 1
        alpha_bound = -0.5 / logx
        assert alpha == alpha_bound and alpha > 1e4
        # check that bound matches optimization result just under threshold
        logx = -0.000051
        alpha, _ = approx.approximate_gamma_kl(1, logx)
        alpha += 1
        alpha_bound = -0.5 / logx
        assert np.abs(alpha - alpha_bound) < 1 and alpha < 1e4
