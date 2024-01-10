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
from distribution_functions import conditional_coalescent_pdf
from distribution_functions import kl_divergence

from tsdate import approx
from tsdate import hypergeo
from tsdate import prior

# TODO: better test set?
# TODO: test special case where child is fixed to age 0
_gamma_trio_test_cases = [  # [shape1, rate1, shape2, rate2, muts, rate]
    [2.0, 0.0005, 2.0, 0.005, 0.0, 0.001],
    [2.0, 0.0005, 2.0, 0.005, 1.0, 0.001],
    [2.0, 0.0005, 2.0, 0.005, 2.0, 0.001],
    [2.0, 0.0005, 2.0, 0.005, 3.0, 0.001],
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
        if t_i < t_j:
            return 0.0
        else:
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
        if t_i < t_j:
            return 0.0
        else:
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
        if t_i < t_j:
            return 0.0
        else:
            return (
                t_j ** (a_j - 1)
                * np.exp(-t_j * b_j)
                * (t_i - t_j) ** y
                * np.exp(-(t_i - t_j) * mu)
            )

    @staticmethod
    def pdf_truncated(t_i, low, upp, a_i, b_i):
        """
        Target proportional to the node marginals (gamma) and an indicator
        function
        """
        if low < t_i < upp:
            return np.exp(
                np.log(t_i) * (a_i - 1)
                - t_i * b_i
                - scipy.special.gammaln(a_i)
                + np.log(b_i) * a_i
            )
        else:
            return 0.0

    def test_moments(self, pars):
        """
        Test mean and variance when ages of both nodes are free
        """
        logconst, t_i, _, var_t_i, t_j, _, var_t_j = approx.moments(*pars)
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
        t_j = a_j / b_j  # point "estimate" for child
        pars_redux = (a_i, b_i, y, mu)
        logconst, t_i, _, var_t_i = approx.rootward_moments(t_j, *pars_redux)
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
        logconst, t_j, _, var_t_j = approx.leafward_moments(t_i, *pars_redux)
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

    def test_truncated_moments(self, pars):
        """
        Test mean and variance of child age when parent age is fixed to a nonzero value
        """
        a_i, b_i, *_ = pars
        upp = a_i / b_i * 2
        low = a_i / b_i / 2
        pars_redux = (low, upp, a_i, b_i)
        logconst, t_i, _, var_t_i = approx.truncated_moments(*pars_redux)
        ck_normconst = scipy.integrate.quad(
            lambda t_i: self.pdf_truncated(t_i, *pars_redux),
            low,
            upp,
            epsabs=0,
        )[0]
        assert np.isclose(logconst, np.log(ck_normconst), rtol=1e-4)
        ck_t_i = scipy.integrate.quad(
            lambda t_i: t_i * self.pdf_truncated(t_i, *pars_redux) / ck_normconst,
            low,
            upp,
            epsabs=0,
        )[0]
        assert np.isclose(t_i, ck_t_i, rtol=1e-4)
        ck_var_t_i = (
            scipy.integrate.quad(
                lambda t_i: t_i**2
                * self.pdf_truncated(t_i, *pars_redux)
                / ck_normconst,
                low,
                upp,
                epsabs=0,
            )[0]
            - ck_t_i**2
        )
        assert np.isclose(var_t_i, ck_var_t_i, rtol=1e-4)

    def test_approximate_gamma_kl(self, pars):
        _, t_i, ln_t_i, _, t_j, ln_t_j, _ = approx.moments(*pars)
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
        _, t_i, _, va_t_i, t_j, _, va_t_j = approx.moments(*pars)
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


class TestPriorMomentMatching:
    """
    Test approximation of the conditional coalescent prior via
    moment matching to a gamma distribution
    """

    n = 10
    priors = prior.ConditionalCoalescentTimes(False)
    priors.add(n)

    @pytest.mark.parametrize("k", np.arange(2, 10))
    def test_conditional_coalescent_pdf(self, k):
        """
        Check that the utility function matches the implementation in
        `tsdate.prior`
        """
        mean, _ = scipy.integrate.quad(
            lambda x: x * conditional_coalescent_pdf(x, self.n, k), 0, np.inf
        )
        var, _ = scipy.integrate.quad(
            lambda x: x**2 * conditional_coalescent_pdf(x, self.n, k), 0, np.inf
        )
        var -= mean**2
        mean_column = prior.PriorParams.field_index("mean")
        var_column = prior.PriorParams.field_index("var")
        assert np.isclose(mean, self.priors[self.n][k][mean_column])
        assert np.isclose(var, self.priors[self.n][k][var_column])

    @pytest.mark.parametrize("k", np.arange(2, 10))
    def test_approximate_gamma(self, k):
        """
        Test that matching gamma to Taylor-series-approximated sufficient
        statistics will result in lower KL divergence than matching to
        mean/variance
        """
        mean_column = prior.PriorParams.field_index("mean")
        var_column = prior.PriorParams.field_index("var")
        x = self.priors[self.n][k][mean_column]
        xvar = self.priors[self.n][k][var_column]
        # match mean/variance
        alpha_0, beta_0 = approx.approximate_gamma_mom(x, xvar)
        ck_x = (alpha_0 + 1) / beta_0
        ck_xvar = (alpha_0 + 1) / beta_0**2
        assert np.isclose(x, ck_x)
        assert np.isclose(xvar, ck_xvar)
        # match approximate sufficient statistics
        logx, _, _ = approx.approximate_log_moments(x, xvar)
        alpha_1, beta_1 = approx.approximate_gamma_kl(x, logx)
        ck_x = (alpha_1 + 1) / beta_1
        ck_logx = hypergeo._digamma(alpha_1 + 1) - np.log(beta_1)
        assert np.isclose(x, ck_x)
        assert np.isclose(logx, ck_logx)
        # compare KL divergence between strategies
        kl_0 = kl_divergence(
            lambda x: conditional_coalescent_pdf(x, self.n, k),
            lambda x: scipy.stats.gamma.logpdf(x, alpha_0 + 1, scale=1 / beta_0),
        )
        kl_1 = kl_divergence(
            lambda x: conditional_coalescent_pdf(x, self.n, k),
            lambda x: scipy.stats.gamma.logpdf(x, alpha_1 + 1, scale=1 / beta_1),
        )
        assert kl_1 < kl_0


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
