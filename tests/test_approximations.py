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
import scipy.stats
from distribution_functions import conditional_coalescent_pdf
from distribution_functions import kl_divergence

from tsdate import approx
from tsdate import hypergeo
from tsdate import prior

_gamma_trio_test_cases = [  # [shape1, rate1, shape2, rate2, muts, rate]
    [10.541, 0.0005, 10.552, 0.005, 1.0, 0.0151],
    [10.541, 0.0065, 10.552, 0.005, 1.0, 0.0101],
    [10.541, 0.0065, 10.552, 0.022, 1.0, 0.0051],
    [10.541, 0.0265, 10.552, 0.022, 1.0, 0.0051],
    [4, 4, 4, 4, 4, 4],
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

    def test_sufficient_statistics(self, pars):
        logconst, t_i, ln_t_i, t_j, ln_t_j = approx.sufficient_statistics(*pars)
        ck_normconst = scipy.integrate.dblquad(
            lambda ti, tj: self.pdf(ti, tj, *pars),
            0,
            np.inf,
            lambda tj: tj,
            np.inf,
            epsabs=0,
        )[0]
        assert np.isclose(logconst, np.log(ck_normconst), rtol=1e-3)
        ck_t_i = scipy.integrate.dblquad(
            lambda ti, tj: ti * self.pdf(ti, tj, *pars) / ck_normconst,
            0,
            np.inf,
            lambda tj: tj,
            np.inf,
            epsabs=0,
        )[0]
        assert np.isclose(t_i, ck_t_i, rtol=1e-3)
        ck_t_j = scipy.integrate.dblquad(
            lambda ti, tj: tj * self.pdf(ti, tj, *pars) / ck_normconst,
            0,
            np.inf,
            lambda tj: tj,
            np.inf,
            epsabs=0,
        )[0]
        assert np.isclose(t_j, ck_t_j, rtol=1e-3)
        ck_ln_t_i = scipy.integrate.dblquad(
            lambda ti, tj: np.log(ti) * self.pdf(ti, tj, *pars) / ck_normconst,
            0,
            np.inf,
            lambda tj: tj,
            np.inf,
            epsabs=0,
        )[0]
        assert np.isclose(ln_t_i, ck_ln_t_i, rtol=1e-3)
        ck_ln_t_j = scipy.integrate.dblquad(
            lambda ti, tj: np.log(tj) * self.pdf(ti, tj, *pars) / ck_normconst,
            0,
            np.inf,
            lambda tj: tj,
            np.inf,
            epsabs=0,
        )[0]
        assert np.isclose(ln_t_j, ck_ln_t_j, rtol=1e-3)

    def test_mean_and_variance(self, pars):
        logconst, t_i, var_t_i, t_j, var_t_j = approx.mean_and_variance(*pars)
        ck_normconst = scipy.integrate.dblquad(
            lambda ti, tj: self.pdf(ti, tj, *pars),
            0,
            np.inf,
            lambda tj: tj,
            np.inf,
            epsabs=0,
        )[0]
        assert np.isclose(logconst, np.log(ck_normconst), rtol=1e-3)
        ck_t_i = scipy.integrate.dblquad(
            lambda ti, tj: ti * self.pdf(ti, tj, *pars) / ck_normconst,
            0,
            np.inf,
            lambda tj: tj,
            np.inf,
            epsabs=0,
        )[0]
        assert np.isclose(t_i, ck_t_i, rtol=1e-3)
        ck_t_j = scipy.integrate.dblquad(
            lambda ti, tj: tj * self.pdf(ti, tj, *pars) / ck_normconst,
            0,
            np.inf,
            lambda tj: tj,
            np.inf,
            epsabs=0,
        )[0]
        assert np.isclose(t_j, ck_t_j, rtol=1e-3)
        ck_var_t_i = (
            scipy.integrate.dblquad(
                lambda ti, tj: ti**2 * self.pdf(ti, tj, *pars) / ck_normconst,
                0,
                np.inf,
                lambda tj: tj,
                np.inf,
                epsabs=0,
            )[0]
            - ck_t_i**2
        )
        assert np.isclose(var_t_i, ck_var_t_i, rtol=1e-3)
        ck_var_t_j = (
            scipy.integrate.dblquad(
                lambda ti, tj: tj**2 * self.pdf(ti, tj, *pars) / ck_normconst,
                0,
                np.inf,
                lambda tj: tj,
                np.inf,
                epsabs=0,
            )[0]
            - ck_t_j**2
        )
        assert np.isclose(var_t_j, ck_var_t_j, rtol=1e-3)

    def test_approximate_gamma(self, pars):
        _, t_i, ln_t_i, t_j, ln_t_j = approx.sufficient_statistics(*pars)
        alpha_i, beta_i = approx.approximate_gamma_kl(t_i, ln_t_i)
        alpha_j, beta_j = approx.approximate_gamma_kl(t_j, ln_t_j)
        ck_t_i = alpha_i / beta_i
        assert np.isclose(t_i, ck_t_i)
        ck_t_j = alpha_j / beta_j
        assert np.isclose(t_j, ck_t_j)
        ck_ln_t_i = hypergeo._digamma(alpha_i) - np.log(beta_i)
        assert np.isclose(ln_t_i, ck_ln_t_i)
        ck_ln_t_j = hypergeo._digamma(alpha_j) - np.log(beta_j)
        assert np.isclose(ln_t_j, ck_ln_t_j)


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
        ck_x = alpha_0 / beta_0
        ck_xvar = alpha_0 / beta_0**2
        assert np.isclose(x, ck_x)
        assert np.isclose(xvar, ck_xvar)
        # match approximate sufficient statistics
        logx, _, _ = approx.approximate_log_moments(x, xvar)
        alpha_1, beta_1 = approx.approximate_gamma_kl(x, logx)
        ck_x = alpha_1 / beta_1
        ck_logx = hypergeo._digamma(alpha_1) - np.log(beta_1)
        assert np.isclose(x, ck_x)
        assert np.isclose(logx, ck_logx)
        # compare KL divergence between strategies
        kl_0 = kl_divergence(
            lambda x: conditional_coalescent_pdf(x, self.n, k),
            lambda x: scipy.stats.gamma.logpdf(x, alpha_0, scale=1 / beta_0),
        )
        kl_1 = kl_divergence(
            lambda x: conditional_coalescent_pdf(x, self.n, k),
            lambda x: scipy.stats.gamma.logpdf(x, alpha_1, scale=1 / beta_1),
        )
        assert kl_1 < kl_0


@pytest.mark.parametrize(
    "pars",
    [
        # taken from issue tsdate/290
        # TODO: strangely, the sufficient statistic calculation works if the
        # second argument is rounded to the next decimal place: so it'd be good
        # to use a more problematic example
        [34.64243, 0.0017707277, 662123.70387, 16.3125724739, 2.0, 0.00275263],
    ],
)
class TestSingular2F1:
    """
    Test approximation of marginal pairwise joint distributions by a gamma via
    arbitrary precision mean/variance matching, when sufficient statistics
    calculation fails
    """

    def test_sufficient_statistics_throws_exception(self, pars):
        with pytest.raises(Exception, match="did not converge"):
            approx.sufficient_statistics(*pars)

    def test_exception_uses_mean_and_variance(self, pars):
        _, t_i, va_t_i, t_j, va_t_j = approx.mean_and_variance(*pars)
        ai1, bi1 = approx.approximate_gamma_mom(t_i, va_t_i)
        aj1, bj1 = approx.approximate_gamma_mom(t_j, va_t_j)
        _, par_i, par_j = approx.gamma_projection(*pars)
        ai2, bi2 = par_i
        aj2, bj2 = par_j
        assert np.isclose(ai1, ai2)
        assert np.isclose(bi1, bi2)
        assert np.isclose(aj1, aj2)
        assert np.isclose(bj1, bj2)
