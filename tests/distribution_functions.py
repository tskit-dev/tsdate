# MIT License
#
# Copyright (C) 2023 University of Oxford
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
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
Utility functions to construct distributions used in variational inference,
for testing purposes
"""
import mpmath
import numpy as np
import scipy.integrate
import scipy.special

from tsdate import approx
from tsdate import hypergeo


def kl_divergence(p, logq):
    """
    KL(p||q) divergence for fixed p
    """
    val, _ = scipy.integrate.quad(lambda x: -p(x) * logq(x), 0, np.inf)
    return val


def conditional_coalescent_pdf(t, n, k):
    """
    Evaluate the PDF of the conditional coalescent, using results from Wiuf and
    Donnelly (1999).
    """

    def pr_t_bar_a(t, a, n):
        """
        Phase-type distribution for sum of waiting times during which
        there are n, n-1, ..., a+1 ancestors.
        """
        lineages = np.arange(n, a, -1)
        rate = (lineages * (lineages - 1)) / 2
        weight = []
        for i in range(len(rate)):
            weight.append(1)
            for j in range(len(rate)):
                if i != j:
                    weight[-1] *= rate[j] / (rate[j] - rate[i])
        weight = np.array(weight)
        val = np.sum(weight * rate * np.exp(-rate * t))
        return val

    def pr_a(a, n, k):
        """
        Hypergeometric-ish distribution for number of ancestors "a" after
        subsample of size "k" has coalesced in tree of size "n"
        """
        return np.exp(
            np.log(n + 1)
            + scipy.special.betaln(n - k, k + 2)
            - np.log(n - a)
            - scipy.special.betaln(n - a - k + 2, k - 1)
            - np.log(a + 1)
            - scipy.special.betaln(a - 1, 3)
        )

    if n == k:
        return pr_t_bar_a(t, 1)
    else:
        return np.sum(
            [pr_a(a, n, k) * pr_t_bar_a(t, a, n) for a in range(2, n - k + 2)]
        )


class TiltedGammaDiff:
    r"""
    The distribution of the difference of two gamma RVs restricted to the
    positive reals, tilted by an exponential function.

    Specifically, for p(x; a, b) = b^a/Gamma(a) x^{a-1} e^{-xb}, this is

        p(u; a1, a2, a3, b1, b2, b3) \propto p(u, a3, b3) \times
            \int_u^inf p(x; a1, b1) p(x - u; a2, b2) dx, u \geq 0
    """

    @staticmethod
    def _2F1(a, b, c, z):
        """
        log of |Re| for Gaussian hypergeometric function
        """
        val = mpmath.log(mpmath.hyp2f1(a, b, c, z))
        return float(val)

    @staticmethod
    def _U(a, b, z):
        """
        log of |Re| for confluent hypergeometric function of the second kind
        """
        val = mpmath.log(mpmath.hyperu(a, b, z))
        return float(val)

    def __init__(self, shape1, shape2, shape3, rate1, rate2, rate3, reorder=True):
        assert shape1 > 0 and shape2 > 0 and shape3 > 0
        assert rate1 >= 0 and rate2 > 0 and rate3 >= 0
        # for convergence of 2F1, we need rate2 > rate3. Invariant
        # transformations of 2F1 allow us to switch arguments, with
        # appropriate rescaling
        self.reparametrize = rate3 > rate2
        self.shape1 = shape1
        self.shape2 = shape2
        self.shape3 = shape3
        self.rate1 = rate1
        self.rate2 = rate2
        self.rate3 = rate3
        mu = (1.0 - shape1 - shape2) / 2
        nu = (shape3 - 1) + (shape1 + shape2) / 2
        zeta = 1 / 2 - (rate2 - rate3) / (rate1 + rate2)
        assert zeta > -1 / 2, "Invalid parameters"
        assert nu + 1 / 2 > abs(mu), "Invalid parameters"

    def pdf(self, x):
        r"""
        Probability density function, derived using equation 3.383.4 from
        Gradshteyn & Ryzhik (2000) "Table of Integrals, Series, and Products"

        That is:

        u^{a3 - 1} e^{(b2 - b3) u} \times
          \int_u^inf x^{a1-1} (x-u)^{a2-1} e^{-x(b1+b2)} dx =
        u^{a3 - 1} e^{(b2 - b3) u} (b1+b2)^{-(a1+a2)/2} Gamma(a2) \times
          u^{(a1+a2)/2-1} e^{-(b1+b2)/2 u} W((a1-a2)/2, (1-a1-a2)/2, (b1+b2)u) =
        (b1+b2)^{-(a1+a2)/2} Gamma(a2) u^{a3 + (a1+a2)/2 - 2} \times
          e^{-u(b1+b2)(1/2+(b3 - b2)/(b1 + b2))} \times
          W((a1-a2)/2, (1-a1-a2)/2, (b1+b2)u)

        s.t. a2 > 0, (b1+b2)u > 0

        Change variables, t = (b1+b2)*u and integrate t using 13.23.4 of DLMF,
        to find the normalizing constant:

        (b1+b2)^{-(a1+a2)/2} Gamma(a2) (b1+b2)^{-a3-(a1+a2)/2+2} \times
          \int_0^inf t^{a3+(a1+a2)/2-2} e^{-t (1/2+(b3-b2)/(b1+b2))} \times
          W((a1-a2)/2, (1-a1-a2)/2, t) 1/(b1+b2) dt =
        (b1+b2)^{-a1-a2-a3+1} Gamma(a1+a2+a3-1) Beta(a2,a3) \times
            2F1(a1+a2+a3-1, a3, a3+a2, (b2-b3)/(b1+b2))

        Note that 2F1 is regularized in this equation, e.g. 2F1(a,b;c;z) =
        2F1(a,b;c;z)/Gamma(c)

        Transform W to U, and use Kummer's transformations:

          W((a1-a2)/2, (1-a1-a2)/2, (b1+b2)u) =
          exp(-u(b1+b2)/2) (b1+b2)^(1-a1/2-a2/2) \times
            u^(1-a1/2-a2/2) U(1-a1, 2-a1-a2, (b1+b2)u) =
          exp(-u(b1+b2)/2) (b1+b2)^(a1/2+a2/2) \times
            u^(a1/2+a2/2) U(a2, a1+a2, (b1+b2)u)

        so that the kernel becomes,

          Gamma(a2) u^(a3 + a1 + a2 - 2) \times
            e^(-u(b1+b3)) U(a2, a1+a2, (b1+b2)u)

        In the special case where b1 == b3 == 0 and a1 == a3 == 1, the integral
        becomes (via 3.382.2 GR2000):

            e^{u b2} \int_u^inf (x-u)^{a2-1} e^{-x b2} dx = Gamma(a2) b2^{-a2}

        e.g. a constant (this should never happen if the edge update order is
        correct and samples are fixed to the present day, or if a proper prior
        is used)
        """
        assert x >= 0.0, "PDF is only defined for non-negative reals"
        A = self.shape1 + self.shape2 + self.shape3 - 1
        B = self.shape3
        C = self.shape2 + self.shape3
        S = self.rate2 - self.rate3
        T = self.rate2 + self.rate1
        if self.reparametrize:
            val = (
                scipy.stats.gamma.logpdf(x, A, scale=1 / (self.rate1 + self.rate3))
                + self._U(
                    self.shape2,
                    self.shape1 + self.shape2,
                    (self.rate1 + self.rate2) * x,
                )
                + scipy.special.loggamma(self.shape2 + self.shape3)
                - scipy.special.loggamma(self.shape3)
                - self._2F1(A, C - B, C, S / (S - T))
            )
        else:
            val = (
                scipy.stats.gamma.logpdf(x, A, scale=1 / (self.rate1 + self.rate3))
                + self._U(
                    self.shape2,
                    self.shape1 + self.shape2,
                    (self.rate1 + self.rate2) * x,
                )
                + scipy.special.loggamma(self.shape2 + self.shape3)
                - scipy.special.loggamma(self.shape3)
                + A * np.log(T / (T - S))
                - self._2F1(A, B, C, S / T)
            )
        return np.exp(val)

    def laplace(self, s):
        """
        Laplace transform, derived using 13.23.4 in DLMF
        """
        # TODO: assert ROC
        A = self.shape1 + self.shape2 + self.shape3 - 1
        B = self.shape3
        C = self.shape2 + self.shape3
        S = self.rate2 - self.rate3
        T = self.rate1 + self.rate2
        if self.reparametrize:
            val = (
                -A * np.log(T - S + s)
                + A * np.log(T - S)
                + self._2F1(A, C - B, C, (s - S) / (T - S + s))
                + -self._2F1(A, C - B, C, S / (S - T))
            )
        else:
            val = self._2F1(A, B, C, (S - s) / T) - self._2F1(A, B, C, S / T)
        return np.exp(val)

    def moments(self):
        r"""
        Returns the first two raw moments.

        Derived from differentiating the Laplace transform, and using the relation

            d^n 2F1(A, B, C, z) / dz^n = a_n b_n / c_n 2F1(A + 1, B + 1, C + 1, z)

        where x_n = x (x + 1) \dots (x + n - 1) is the rising factorial and 2F1 is the
        Gaussian hypergeometric function. See 15.5.2 in DLMF.
        """
        A = self.shape1 + self.shape2 + self.shape3 - 1
        B = self.shape3
        C = self.shape2 + self.shape3
        S = self.rate2 - self.rate3
        T = self.rate1 + self.rate2
        numer = 0
        denom = 0
        moments = []
        if self.reparametrize:
            F0 = self._2F1(A, C - B, C, S / (S - T)) - A * np.log(1 - S / T)
            for i in range(2):
                numer += np.log(A + i) + np.log(B + i) - np.log(C + i) - np.log(T)
                denom = (
                    F0
                    - self._2F1(A + i + 1, C - B, C + i + 1, S / (S - T))
                    + (A + i + 1) * np.log(1 - S / T)
                )
                moments.append(np.exp(numer - denom))
        else:
            F0 = self._2F1(A, B, C, S / T)
            for i in range(2):
                numer += np.log(A + i) + np.log(B + i) - np.log(C + i) - np.log(T)
                denom = F0 - self._2F1(A + i + 1, B + i + 1, C + i + 1, S / T)
                moments.append(np.exp(numer - denom))
        return moments[0], moments[1]

    def sufficient_statistics(self):
        r"""
        Return E[x], E[x^2], and E[log x].

        To get raw moments use the Laplace transform E[e^{-su}]:

        (b1+b2)^{-a1-a2-a3+1} Gamma(a1+a2+a3-1) Beta(a2,a3) \times
            2F1(a1+a2+a3-1, a3, a3+a2, (b2-b3-s)/(b1+b2))

        To get log moments use the Mellin transform E[u^s]:

        (b1+b2)^{-a1-a2-a3-s+1} Gamma(a1+a2+a3+s-1) Beta(a2,a3+s) \times
            2F1(a1+a2+a3+s-1, a3+s, a3+s+a2, (b2-b3)/(b1+b2))

        In both cases: differentiate wrt s, evaluate at zero, and normalize
        """
        A = self.shape1 + self.shape2 + self.shape3 - 1
        B = self.shape3
        C = self.shape2 + self.shape3
        S = self.rate2 - self.rate3
        T = self.rate1 + self.rate2
        if self.reparametrize:
            # underflow protection is used, so the actual derivatives are
            # np.exp(F) * dF_dz, etc.
            F, dF_da, _, dF_dc, dF_dz, d2F_dz2 = hypergeo._hyp2f1_series(
                A,
                C - B,
                C,
                S / (S - T),
            )
            logconst = (
                F
                + scipy.special.betaln(self.shape2, self.shape3)
                + scipy.special.loggamma(A)
                - A * np.log(T - S)
            )
            x = -dF_dz * T / (S - T) ** 2 - A / (S - T)
            xsq = (
                d2F_dz2 * T**2 / (S - T) ** 4
                + A * (A + 1) / (T - S) ** 2
                + 2 * dF_dz * (1 + A) * T / (S - T) ** (3)
            )
            logx = (
                (dF_da + dF_dc)
                + -np.log(1 - S / T)
                + scipy.special.digamma(A)
                + scipy.special.digamma(B)
                - scipy.special.digamma(C)
                - np.log(T)
            )
        else:
            F, dF_da, dF_db, dF_dc, dF_dz, d2F_dz2 = hypergeo._hyp2f1_series(
                A,
                B,
                C,
                S / T,
            )
            logconst = (
                F
                + scipy.special.betaln(self.shape2, self.shape3)
                + scipy.special.loggamma(A)
                - A * np.log(T)
            )
            x = dF_dz / T
            xsq = d2F_dz2 / T**2
            logx = (
                (dF_da + dF_db + dF_dc)
                + scipy.special.digamma(A)
                + scipy.special.digamma(B)
                - scipy.special.digamma(C)
                - np.log(T)
            )
        return logconst, x, xsq, logx

    def to_gamma(self, minimize_kl=True):
        """
        Return the shape and rate parameters of a gamma distribution with the
        same expected sufficient statistics (if ``minimize_kl`` is ``True``), and
        otherwise one with the same mean and variance.
        """
        logconst, x, xsq, logx = self.sufficient_statistics()
        if minimize_kl:
            _, xlogx, _ = approx.approximate_log_moments(x, xsq)
            alpha, beta = approx.approximate_gamma_kl(x, logx, xlogx)
        else:
            alpha, beta = approx.approximate_gamma_mom(x, xsq)
        return logconst, alpha, beta


class TiltedGammaSum:
    r"""
    The distribution of the sum of two gamma RVs, tilted by an exponential function.

    Specifically, for p(x; a, b) = b^a/Gamma(a) x^{a-1} e^{-xb}, this is

        p(u; a1, a2, a3, b1, b2, b3) \propto p(u; a3, b3) \times
            \int_0^u p(x; a1, b1) p(u - x; a2, b2) dx, u \geq 0
    """

    @staticmethod
    def _2F1(a, b, c, z):
        """
        log of |Re| for Gaussian hypergeometric function
        """
        val = mpmath.log(mpmath.hyp2f1(a, b, c, z))
        return float(val)

    @staticmethod
    def _M(a, b, x):
        """
        log of |Re| for confluent hypergeometric function (aka 1F1)
        """
        val = mpmath.log(mpmath.hyp1f1(a, b, x))
        return float(val)

    def __init__(self, shape1, shape2, shape3, rate1, rate2, rate3):
        assert shape1 > 0 and shape2 > 0 and shape3 > 0
        assert rate1 >= 0 and rate2 > 0 and rate3 >= 0
        # for numeric stability of hypergeometric we need rate2 > rate1
        # as this is a convolution, the order of (1) and (2) don't matter
        self.reparametrize = rate1 > rate2
        self.shape1 = shape2 if self.reparametrize else shape1
        self.shape2 = shape1 if self.reparametrize else shape2
        self.shape3 = shape3
        self.rate1 = rate2 if self.reparametrize else rate1
        self.rate2 = rate1 if self.reparametrize else rate2
        self.rate3 = rate3
        assert shape1 + shape2 + shape3 - 1 > 0, "Invalid parameters"
        assert rate3 + rate2 > max(0, rate2 - rate1), "Invalid parameters"

    def pdf(self, x):
        r"""
        Probability density function, derived using equation 3.383.1 from
        Gradshteyn & Ryzhik (2000) "Table of Integrals, Series, and Products"

        u^{a3-1} e^{-u b3} \times
            \int_0^u x^{a1-1} (u-x)^{a2-1} e^{-x b1} e^{-(u-x) b2} dx =
        u^{a3-1} e^{-u (b2 + b3)} \times
            \int_0^u x^{a1-1} (u-x)^{a2-1} e^{x (b2-b1)} dx =
        Beta(a2, a1) u^{a1 + a2 + a3 - 2} e^{-u (b2+b3)} M(a1, a1+a2, (b2-b1) u)

        s.t. a1 > 0, a2 > 0

        Then integrate u using 13.10.3 of DLMF -- note that this equation is
        regularized, so that we have M(a,c,kt)/Gamma(c) and
        2F1(a,b;c;z)/Gamma(c)

        \int_0^\infty e^{-u (b2+b3)} u^{a1+a2+a3-2} M(a1, a1+a2, (b2-b1) u) du =
            Gamma(a1+a2+a3-1) (b2+b3)^{-a1-a2-a3+1} \times
              2F1(a1, a1+a2+a3-1, a1+a2, (b2-b1)/(b2+b3))

        s.t. a1+a2 > 1
        s.t. b2 > max(b2-b1, 0)

        In the special case where b1 == b3 == 0 and a1 == a3 == 1, the integrals
        become (via 3.382.1 in GR2000):

        e^{-u b2} \int_0^u (u-x)^{a2-1} e^{x b2} dx = b2^(-a2) gamma(a2, b2 u)

        which is improper (this special case should never happen if edge order
        is correct or priors are proper)
        """
        assert x >= 0.0, "PDF is only defined for non-negative reals"
        A = self.shape1
        B = self.shape1 + self.shape2 + self.shape3 - 1
        C = self.shape1 + self.shape2
        T = self.rate2 - self.rate1
        S = self.rate2 + self.rate3
        val = (
            scipy.stats.gamma.logpdf(x, B, scale=1 / S)
            + self._M(self.shape1, self.shape1 + self.shape2, T * x)
            - self._2F1(A, B, C, T / S)
        )
        return np.exp(val)

    def laplace(self, s):
        """
        Laplace transform, derived using 13.10.3 in DLMF
        """
        # TODO: assert ROC
        A = self.shape1
        B = self.shape1 + self.shape2 + self.shape3 - 1
        C = self.shape1 + self.shape2
        T = self.rate2 - self.rate1
        S = self.rate2 + self.rate3
        val = (
            self._2F1(A, B, C, T / (S + s))
            - self._2F1(A, B, C, T / S)
            + B * (np.log(S) - np.log(S + s))
        )
        return np.exp(val)

    def moments(self):
        r"""
        First two raw moments, from differentiating the Laplace transform
        """
        A = self.shape1
        B = self.shape1 + self.shape2 + self.shape3 - 1
        C = self.shape1 + self.shape2
        T = self.rate2 - self.rate1
        S = self.rate2 + self.rate3
        F0 = self._2F1(A, B, C, T / S)
        Z1 = np.log(A) + np.log(B) - np.log(C)
        F1 = self._2F1(A + 1, B + 1, C + 1, T / S)
        Z2 = Z1 + np.log(A + 1) + np.log(B + 1) - np.log(C + 1)
        F2 = self._2F1(A + 2, B + 2, C + 2, T / S)
        x = np.exp(F1 - F0 + Z1) * T / S**2 + B / S
        xsq = (
            B * (B + 1) / S**2
            + 2 * (B + 1) * T / S**3 * np.exp(Z1 + F1 - F0)
            + T**2 / S**4 * np.exp(Z2 + F2 - F0)
        )
        return x, xsq

    def sufficient_statistics(self):
        r"""
        Return E[x], E[x^2], and E[log x].

        To get raw moments use the Laplace transform E[e^{-su}]:

        Gamma(a1+a2+a3-1) (b2+b3+s)^{-a1-a2-a3+1} \times
          2F1(a1, a1+a2+a3-1, a1+a2, (b2-b1)/(b2+b3+s))

        To get log moments use the Mellin transform E[u^s]:

        Gamma(a1+a2+a3+s-1) (b2+b3)^{-a1-a2-a3+s+1} \times
          2F1(a1, a1+a2+a3+s-1, a1+a2, (b2-b1)/(b2+b3))

        In both cases: differentiate wrt s, evaluate at zero, and normalize
        """
        A = self.shape1
        B = self.shape1 + self.shape2 + self.shape3 - 1
        C = self.shape1 + self.shape2
        T = self.rate2 - self.rate1
        S = self.rate2 + self.rate3
        # underflow protection is used, so the actual derivatives are
        # np.exp(F) * dF_dz, etc.
        F, _, dF_db, _, dF_dz, d2F_dz2 = hypergeo._hyp2f1_series(A, B, C, T / S)
        logconst = (
            F
            + scipy.special.loggamma(B)
            - B * np.log(S)
            + scipy.special.betaln(self.shape1, self.shape2)
        )
        x = dF_dz * T / S**2 + B / S
        xsq = (
            d2F_dz2 * T**2 / S**4
            + B * (B + 1) / S**2
            + 2 * dF_dz * (1 + B) * T / S**3
        )
        logx = dF_db + scipy.special.digamma(B) - np.log(S)
        return logconst, x, xsq, logx

    def to_gamma(self, minimize_kl=True):
        """
        Return the shape and rate parameters of a gamma distribution with the
        same expected sufficient statistics (if ``minimize_kl`` is ``True``), and
        otherwise one with the same mean and variance.
        """
        logconst, x, xsq, logx = self.sufficient_statistics()
        if minimize_kl:
            _, xlogx, _ = approx.approximate_log_moments(x, xsq)
            alpha, beta = approx.approximate_gamma_kl(x, logx, xlogx)
        else:
            alpha, beta = approx.approximate_gamma_mom(x, xsq)
        return logconst, alpha, beta
