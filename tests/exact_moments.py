# flake8: noqa
"""
Moments for EP updates using exact hypergeometric evaluations rather than
a Laplace approximation; intended for testing and accuracy benchmarking.
"""

from math import exp
from math import log

import mpmath
import numpy as np
import scipy
from scipy.special import betaln
from math import lgamma


def moments(a_i, b_i, a_j, b_j, y_ij, mu_ij):
    """
    log p(t_i, t_j) := \
        log(t_i - t_j) * y_ij - mu_ij * (t_i - t_j) + \
        log(t_i) * (a_i - 1) - b_i * t_i + \
        log(t_j) * (a_j - 1) - b_j * t_j
    """
    a = a_j
    b = a_i + a_j + y_ij
    c = a_j + y_ij + 1
    t = mu_ij + b_i
    z = (mu_ij - b_j) / t
    f0 = float(mpmath.log(mpmath.hyp2f1(a + 0, b + 0, c + 0, z)))
    f1 = float(mpmath.log(mpmath.hyp2f1(a + 1, b + 1, c + 1, z)))
    f2 = float(mpmath.log(mpmath.hyp2f1(a + 2, b + 2, c + 2, z)))
    s1 = a * b / c
    s2 = s1 * (a + 1) * (b + 1) / (c + 1)
    d1 = s1 * exp(f1 - f0)
    d2 = s2 * exp(f2 - f0)
    logl = f0 + betaln(y_ij + 1, a) + lgamma(b) - b * log(t)
    mn_j = d1 / t
    sq_j = d2 / t**2
    va_j = sq_j - mn_j**2
    mn_i = mn_j * z + b / t
    sq_i = sq_j * z**2 + (b + 1) * (mn_i + mn_j * z) / t
    va_i = sq_i - mn_i**2
    return logl, mn_i, va_i, mn_j, va_j


def rootward_moments(t_j, a_i, b_i, y_ij, mu_ij):
    """
    log p(t_i) := \
        log(t_i - t_j) * y_ij - mu_ij * (t_i - t_j) + \
        log(t_i) * (a_i - 1) - b_i * t_i 
    """
    assert t_j >= 0.0
    s = a_i + y_ij
    r = mu_ij + b_i
    a = y_ij + 1
    b = s + 1
    z = t_j * r
    if t_j == 0.0:
        logl = lgamma(s) - s * log(r)
        mn_i = s / r
        va_i = s / r**2
        return logl, mn_i, va_i
    f0 = float(mpmath.log(mpmath.hyperu(a + 0, b + 0, z)))
    f1 = float(mpmath.log(mpmath.hyperu(a + 1, b + 1, z)))
    f2 = float(mpmath.log(mpmath.hyperu(a + 2, b + 2, z)))
    d0 = -a * exp(f1 - f0)
    d1 = -(a + 1) * exp(f2 - f1)
    logl = f0 - b_i * t_j + (b - 1) * log(t_j) + lgamma(a)
    mn_i = t_j * (1 - d0)
    va_i = t_j**2 * d0 * (d1 - d0)
    return logl, mn_i, va_i


def leafward_moments(t_i, a_j, b_j, y_ij, mu_ij):
    """
    log p(t_j) := \
        log(t_i - t_j) * y_ij - mu_ij * (t_i - t_j) + \
        log(t_j) * (a_j - 1) - b_j * t_j
    """
    assert t_i > 0.0
    a = a_j
    b = a_j + y_ij + 1
    z = t_i * (mu_ij - b_j)
    f0 = float(mpmath.log(mpmath.hyp1f1(a + 0, b + 0, z)))
    f1 = float(mpmath.log(mpmath.hyp1f1(a + 1, b + 1, z)))
    f2 = float(mpmath.log(mpmath.hyp1f1(a + 2, b + 2, z)))
    d1 = a / b * exp(f1 - f0)
    d2 = a / b * (a + 1) / (b + 1) * exp(f2 - f0)
    logl = f0 - mu_ij * t_i + (b - 1) * log(t_i) + betaln(a, b - a)
    mn_j = t_i * d1
    sq_j = t_i**2 * d2
    va_j = sq_j - mn_j**2
    return logl, mn_j, va_j


def unphased_moments(a_i, b_i, a_j, b_j, y_ij, mu_ij):
    """
    log p(t_i, t_j) := \
        log(t_i + t_j) * y_ij - mu_ij * (t_i + t_j) + \
        log(t_i) * (a_i - 1) - b_i * t_i + \
        log(t_j) * (a_j - 1) - b_j * t_j
    """
    a = a_j
    b = a_i + a_j + y_ij
    c = a_j + a_i
    t = mu_ij + b_i
    z = (mu_ij + b_j) / t
    f0 = float(mpmath.log(mpmath.hyp2f1(a + 0, b + 0, c + 0, 1 - z)))
    f1 = float(mpmath.log(mpmath.hyp2f1(a + 1, b + 1, c + 1, 1 - z)))
    f2 = float(mpmath.log(mpmath.hyp2f1(a + 2, b + 2, c + 2, 1 - z)))
    s1 = a * b / c
    s2 = s1 * (a + 1) * (b + 1) / (c + 1)
    d1 = s1 * exp(f1 - f0)
    d2 = s2 * exp(f2 - f0)
    logl = f0 + betaln(a_j, a_i) + lgamma(b) - b * log(t)
    mn_j = d1 / t
    sq_j = d2 / t**2
    va_j = sq_j - mn_j**2
    mn_i = b / t - mn_j * z
    sq_i = sq_j * z**2 + (b + 1) * (mn_i - mn_j * z) / t
    va_i = sq_i - mn_i**2
    return logl, mn_i, va_i, mn_j, va_j


def twin_moments(a_i, b_i, y_ij, mu_ij):
    """
    log p(t_i) := \
        log(2 * t_i) * y_ij - mu_ij * (2 * t_i) + \
        log(t_i) * (a_i - 1) - b_i * t_i
    """
    s = a_i + y_ij
    r = b_i + 2 * mu_ij
    logl = log(2) * y_ij + lgamma(s) - log(r) * s
    mn_i = s / r
    va_i = s / r**2
    return logl, mn_i, va_i


def sideways_moments(t_i, a_j, b_j, y_ij, mu_ij):
    """
    log p(t_j) := \
        log(t_i + t_j) * y_ij - mu_ij * (t_i + t_j) + \
        log(t_j) * (a_j - 1) - b_j * t_j
    """
    assert t_i > 0.0
    a = a_j
    b = a_j + y_ij + 1
    z = t_i * (mu_ij + b_j)
    f0 = float(mpmath.log(mpmath.hyperu(a + 0, b + 0, z)))
    f1 = float(mpmath.log(mpmath.hyperu(a + 1, b + 1, z)))
    f2 = float(mpmath.log(mpmath.hyperu(a + 2, b + 2, z)))
    d0 = -a * exp(f1 - f0)
    d1 = -(a + 1) * exp(f2 - f1)
    logl = f0 - mu_ij * t_i + (b - 1) * log(t_i) + lgamma(a)
    mn_j = -t_i * d0
    va_j = t_i**2 * d0 * (d1 - d0)
    return logl, mn_j, va_j


def mutation_moments(a_i, b_i, a_j, b_j, y_ij, mu_ij):
    """
    log p(t_m, t_i, t_j) := \
        log(t_i - t_j) * y_ij - mu_ij * (t_i - t_j) + \
        log(t_i) * (a_i - 1) - b_i * t_i + \
        log(t_j) * (a_j - 1) - b_j * t_j + \
        log(int(t_j < t_m < t_i) / (t_i - t_j))
    """
    a = a_j
    b = a_i + a_j + y_ij
    c = a_j + y_ij + 1
    t = mu_ij + b_i
    z = (mu_ij - b_j) / t
    f000 = float(mpmath.log(mpmath.hyp2f1(a + 0, b + 0, c + 0, z)))
    f020 = float(mpmath.log(mpmath.hyp2f1(a + 0, b + 2, c + 0, z)))
    f111 = float(mpmath.log(mpmath.hyp2f1(a + 1, b + 1, c + 1, z)))
    f121 = float(mpmath.log(mpmath.hyp2f1(a + 1, b + 2, c + 1, z)))
    f222 = float(mpmath.log(mpmath.hyp2f1(a + 2, b + 2, c + 2, z)))
    s1 = a * b / c
    s2 = s1 * (a + 1) * (b + 1) / (c + 1)
    d1 = b * (b + 1) / t**2
    d2 = d1 * a / c
    d3 = d2 * (a + 1) / (c + 1)
    mn_m = s1 * exp(f111 - f000) / t / 2 * (1 + z) + b / t / 2
    sq_m = (
        d1 * exp(f020 - f000) / 3 + d2 * exp(f121 - f000) / 3 + d3 * exp(f222 - f000) / 3
    )
    va_m = sq_m - mn_m**2
    return mn_m, va_m


def mutation_rootward_moments(t_j, a_i, b_i, y_ij, mu_ij):
    """
    log p(t_m, t_i) := \
        log(t_i - t_j) * y_ij - mu_ij * (t_i - t_j) + \
        log(t_i) * (a_i - 1) - b_i * t_i + \
        log(int(t_j < t_m < t_i) / (t_i - t_j))
    """
    logl, mn_i, va_i = rootward_moments(t_j, a_i, b_i, y_ij, mu_ij)
    mn_m = mn_i / 2 + t_j / 2
    sq_m = (va_i + mn_i**2 + mn_i * t_j + t_j**2) / 3
    va_m = sq_m - mn_m**2
    return mn_m, va_m


def mutation_leafward_moments(t_i, a_j, b_j, y_ij, mu_ij):
    """
    log p(t_m, t_j) := \
        log(t_i - t_j) * y_ij - mu_ij * (t_i - t_j) + \
        log(t_j) * (a_j - 1) - b_j * t_j + \
        log(int(t_j < t_m < t_i) / (t_i - t_j))
    """
    logl, mn_j, va_j = leafward_moments(t_i, a_j, b_j, y_ij, mu_ij)
    mn_m = mn_j / 2 + t_i / 2
    sq_m = (va_j + mn_j**2 + mn_j * t_i + t_i**2) / 3
    va_m = sq_m - mn_m**2
    return mn_m, va_m


def mutation_unphased_moments(a_i, b_i, a_j, b_j, y_ij, mu_ij):
    """
    log p(t_m, t_i, t_j) := \
        log(t_i + t_j) * y_ij - mu_ij * (t_i + t_j) + \
        log(t_i) * (a_i - 1) - b_i * t_i + \
        log(t_j) * (a_j - 1) - b_j * t_j + \
        log(t_i / (t_i + t_j) * int(0 < t_m < t_i) / t_i + \
            t_j / (t_i + t_j) * int(0 < t_m < t_j) / t_j)
    """
    a = a_j
    b = a_j + a_i + y_ij
    c = a_j + a_i
    t = mu_ij + b_i
    z = (mu_ij + b_j) / t
    f000 = float(mpmath.log(mpmath.hyp2f1(a + 0, b + 0, c + 0, 1 - z)))
    f001 = float(mpmath.log(mpmath.hyp2f1(a + 0, b + 0, c + 1, 1 - z)))
    f012 = float(mpmath.log(mpmath.hyp2f1(a + 0, b + 1, c + 2, 1 - z)))
    f023 = float(mpmath.log(mpmath.hyp2f1(a + 0, b + 2, c + 3, 1 - z)))
    f212 = float(mpmath.log(mpmath.hyp2f1(a + 2, b + 1, c + 2, 1 - z)))
    f323 = float(mpmath.log(mpmath.hyp2f1(a + 3, b + 2, c + 3, 1 - z)))
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


def mutation_twin_moments(a_i, b_i, y_ij, mu_ij):
    """
    log p(t_m, t_i) := \
        log(int(0 < t_m < t_i) / t_i) + \
        log(2 * t_i) * y_ij - mu_ij * (2 * t_i) + \
        log(t_i) * (a_i - 1) - b_i * t_i
    """
    s = a_i + y_ij
    r = b_i + 2 * mu_ij
    pr_m = 0.5
    mn_m = s / r / 2
    sq_m = (s + 1) * s / 3 / r**2
    va_m = sq_m - mn_m**2
    return pr_m, mn_m, va_m


def mutation_sideways_moments(t_i, a_j, b_j, y_ij, mu_ij):
    """
    log p(t_m, t_j) := \
        log(t_i + t_j) * y_ij - mu_ij * (t_i + t_j) + \
        log(t_j) * (a_j - 1) - b_j * t_j + \
        log(t_i / (t_i + t_j) * int(0 < t_m < t_i) / t_i + \
            t_j / (t_i + t_j) * int(0 < t_m < t_j) / t_j)
    """
    a = a_j
    b = a_j + y_ij + 1
    z = t_i * (mu_ij + b_j)
    f00 = float(mpmath.log(mpmath.hyperu(a + 0, b + 0, z)))
    f10 = float(mpmath.log(mpmath.hyperu(a + 1, b + 0, z)))
    f21 = float(mpmath.log(mpmath.hyperu(a + 2, b + 1, z)))
    f32 = float(mpmath.log(mpmath.hyperu(a + 3, b + 2, z)))
    pr_m = 1.0 - exp(f10 - f00) * a
    mn_m = pr_m * t_i / 2 + t_i * exp(f21 - f00) * a * (a + 1) / 2
    sq_m = pr_m * t_i**2 / 3 + t_i**2 * exp(f32 - f00) * a * (a + 1) * (a + 2) / 3
    va_m = sq_m - mn_m**2
    return pr_m, mn_m, va_m


def mutation_edge_moments(t_i, t_j):
    """
    log p(t_m) := int(t_j < t_m < t_i) / (t_i - t_j)
    """
    mn_m = t_i / 2 + t_j / 2
    va_m = (t_i - t_j) ** 2 / 12
    return mn_m, va_m


def mutation_block_moments(t_i, t_j):
    """
    log p(t_m) := \
        log(t_i / (t_i + t_j) * int(0 < t_m < t_i) / t_i + \
            t_j / (t_i + t_j) * int(0 < t_m < t_j) / t_j)
    """
    pr_m = t_i / (t_i + t_j)
    mn_m = pr_m * t_i / 2 + (1 - pr_m) * t_j / 2
    sq_m = pr_m * t_i**2 / 3 + (1 - pr_m) * t_j**2 / 3
    va_m = sq_m - mn_m**2
    return pr_m, mn_m, va_m


# --- verify exact solutions with quadrature --- #


class TestExactMoments:
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
    def pdf_twin(t_i, a_i, b_i, y, mu):
        """
        Target marginal distribution, proportional to the parent
        marginal (gamma) and a Poisson mutation likelihood over the two
        branches leading from (present-day) individual to a *single* parent
        """
        assert t_i > 0
        return (
            t_i ** (a_i - 1)
            * np.exp(-t_i * b_i)
            * (t_i + t_i) ** y
            * np.exp(-(t_i + t_i) * mu)
        )

    @staticmethod
    def pdf_sideways(t_i, t_j, a_j, b_j, y, mu):
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

    @staticmethod
    def pdf_edge(x, t_i, t_j):
        """
        Mutation uniformly distributed between child and parent
        """
        assert t_i > 0 and t_j > 0
        return int(t_j < x < t_i) / (t_i - t_j)

    @staticmethod
    def pdf_block(x, t_i, t_j):
        """
        Mutation uniformly distributed between child at time zero and one of
        two parents with fixed ages
        """
        assert t_i > 0 and t_j > 0
        return (int(0 < x < t_i) + int(0 < x < t_j)) / (t_i + t_j)

    def test_moments(self, pars):
        """
        Test mean and variance when ages of both nodes are free
        """
        logconst, t_i, var_t_i, t_j, var_t_j = moments(*pars)
        ck_normconst = scipy.integrate.dblquad(
            lambda t_i, t_j: self.pdf(t_i, t_j, *pars),
            0,
            np.inf,
            lambda t_j: t_j,
            np.inf,
            epsabs=0,
        )[0]
        assert np.isclose(logconst, np.log(ck_normconst))
        ck_t_i = scipy.integrate.dblquad(
            lambda t_i, t_j: t_i * self.pdf(t_i, t_j, *pars) / ck_normconst,
            0,
            np.inf,
            lambda t_j: t_j,
            np.inf,
            epsabs=0,
        )[0]
        assert np.isclose(t_i, ck_t_i)
        ck_t_j = scipy.integrate.dblquad(
            lambda t_i, t_j: t_j * self.pdf(t_i, t_j, *pars) / ck_normconst,
            0,
            np.inf,
            lambda t_j: t_j,
            np.inf,
            epsabs=0,
        )[0]
        assert np.isclose(t_j, ck_t_j)
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
        assert np.isclose(var_t_i, ck_var_t_i)
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
        assert np.isclose(var_t_j, ck_var_t_j)

    def test_rootward_moments(self, pars):
        """
        Test mean and variance of parent age when child age is fixed to a nonzero value
        """
        a_i, b_i, a_j, b_j, y, mu = pars
        pars_redux = (a_i, b_i, y, mu)
        mn_j = a_j / b_j  # point "estimate" for child
        for t_j in [0.0, mn_j]:
            logconst, t_i, var_t_i = rootward_moments(t_j, *pars_redux)
            ck_normconst = scipy.integrate.quad(
                lambda t_i: self.pdf_rootward(t_i, t_j, *pars_redux),
                t_j,
                np.inf,
                epsabs=0,
            )[0]
            assert np.isclose(logconst, np.log(ck_normconst))
            ck_t_i = scipy.integrate.quad(
                lambda t_i: t_i * self.pdf_rootward(t_i, t_j, *pars_redux) / ck_normconst,
                t_j,
                np.inf,
                epsabs=0,
            )[0]
            assert np.isclose(t_i, ck_t_i)
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
            assert np.isclose(var_t_i, ck_var_t_i)

    def test_leafward_moments(self, pars):
        """
        Test mean and variance of child age when parent age is fixed to a nonzero value
        """
        a_i, b_i, a_j, b_j, y, mu = pars
        t_i = a_i / b_i  # point "estimate" for parent
        pars_redux = (a_j, b_j, y, mu)
        logconst, t_j, var_t_j = leafward_moments(t_i, *pars_redux)
        ck_normconst = scipy.integrate.quad(
            lambda t_j: self.pdf_leafward(t_i, t_j, *pars_redux),
            0,
            t_i,
            epsabs=0,
        )[0]
        assert np.isclose(logconst, np.log(ck_normconst))
        ck_t_j = scipy.integrate.quad(
            lambda t_j: t_j * self.pdf_leafward(t_i, t_j, *pars_redux) / ck_normconst,
            0,
            t_i,
            epsabs=0,
        )[0]
        assert np.isclose(t_j, ck_t_j)
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
        assert np.isclose(var_t_j, ck_var_t_j)

    def test_unphased_moments(self, pars):
        """
        Parent ages for an singleton nodes above an unphased individual
        """
        logconst, t_i, var_t_i, t_j, var_t_j = unphased_moments(*pars)
        ck_normconst = scipy.integrate.dblquad(
            lambda t_i, t_j: self.pdf_unphased(t_i, t_j, *pars),
            0,
            np.inf,
            0,
            np.inf,
            epsabs=0,
        )[0]
        assert np.isclose(logconst, np.log(ck_normconst))
        ck_t_i = scipy.integrate.dblquad(
            lambda t_i, t_j: t_i * self.pdf_unphased(t_i, t_j, *pars) / ck_normconst,
            0,
            np.inf,
            0,
            np.inf,
            epsabs=0,
        )[0]
        assert np.isclose(t_i, ck_t_i)
        ck_t_j = scipy.integrate.dblquad(
            lambda t_i, t_j: t_j * self.pdf_unphased(t_i, t_j, *pars) / ck_normconst,
            0,
            np.inf,
            0,
            np.inf,
            epsabs=0,
        )[0]
        assert np.isclose(t_j, ck_t_j)
        ck_var_t_i = (
            scipy.integrate.dblquad(
                lambda t_i, t_j: t_i**2
                * self.pdf_unphased(t_i, t_j, *pars)
                / ck_normconst,
                0,
                np.inf,
                0,
                np.inf,
                epsabs=0,
            )[0]
            - ck_t_i**2
        )
        assert np.isclose(var_t_i, ck_var_t_i)
        ck_var_t_j = (
            scipy.integrate.dblquad(
                lambda t_i, t_j: t_j**2
                * self.pdf_unphased(t_i, t_j, *pars)
                / ck_normconst,
                0,
                np.inf,
                0,
                np.inf,
                epsabs=0,
            )[0]
            - ck_t_j**2
        )
        assert np.isclose(var_t_j, ck_var_t_j)

    def test_twin_moments(self, pars):
        """
        Parent age for one singleton node above an unphased individual
        """
        a_i, b_i, a_j, b_j, y_ij, mu_ij = pars
        pars_redux = (a_i, b_i, y_ij, mu_ij)
        logconst, t_i, var_t_i = twin_moments(*pars_redux)
        ck_normconst = scipy.integrate.quad(
            lambda t_i: self.pdf_twin(t_i, *pars_redux),
            0,
            np.inf,
            epsabs=0,
        )[0]
        assert np.isclose(logconst, np.log(ck_normconst))
        ck_t_i = scipy.integrate.quad(
            lambda t_i: t_i * self.pdf_twin(t_i, *pars_redux) / ck_normconst,
            0,
            np.inf,
            epsabs=0,
        )[0]
        assert np.isclose(t_i, ck_t_i)
        ck_var_t_i = (
            scipy.integrate.quad(
                lambda t_i: t_i**2 * self.pdf_twin(t_i, *pars_redux) / ck_normconst,
                0,
                np.inf,
                epsabs=0,
            )[0]
            - ck_t_i**2
        )
        assert np.isclose(var_t_i, ck_var_t_i)

    def test_sideways_moments(self, pars):
        """
        Parent ages for an singleton nodes above an unphased individual, where
        second parent is fixed to a particular time
        """
        a_i, b_i, a_j, b_j, y, mu = pars
        pars_redux = (a_j, b_j, y, mu)
        t_i = a_i / b_i  # point "estimate" for left parent
        nc, mn, va = sideways_moments(t_i, *pars_redux)
        ck_nc = scipy.integrate.quad(
            lambda t_j: self.pdf_sideways(t_i, t_j, *pars_redux),
            0,
            np.inf,
        )[0]
        assert np.isclose(np.exp(nc), ck_nc)
        ck_mn = (
            scipy.integrate.quad(
                lambda t_j: t_j * self.pdf_sideways(t_i, t_j, *pars_redux),
                0,
                np.inf,
            )[0]
            / ck_nc
        )
        assert np.isclose(mn, ck_mn)
        ck_va = (
            scipy.integrate.quad(
                lambda t_j: t_j**2 * self.pdf_sideways(t_i, t_j, *pars_redux),
                0,
                np.inf,
            )[0]
            / ck_nc
            - ck_mn**2
        )
        assert np.isclose(va, ck_va)

    def test_mutation_moments(self, pars):
        """
        Mutation mapped to a single branch with both nodes free
        """

        def f(t_i, t_j):
            assert t_j < t_i
            mn = t_i / 2 + t_j / 2
            sq = (t_i**2 + t_i * t_j + t_j**2) / 3
            return mn, sq

        mn, va = mutation_moments(*pars)
        nc = scipy.integrate.dblquad(
            lambda t_i, t_j: self.pdf(t_i, t_j, *pars),
            0,
            np.inf,
            lambda t_j: t_j,
            np.inf,
            epsabs=0,
        )[0]
        ck_mn = (
            scipy.integrate.dblquad(
                lambda t_i, t_j: f(t_i, t_j)[0] * self.pdf(t_i, t_j, *pars),
                0,
                np.inf,
                lambda t_j: t_j,
                np.inf,
                epsabs=0,
            )[0]
            / nc
        )
        assert np.isclose(mn, ck_mn)
        ck_va = (
            scipy.integrate.dblquad(
                lambda t_i, t_j: f(t_i, t_j)[1] * self.pdf(t_i, t_j, *pars),
                0,
                np.inf,
                lambda t_j: t_j,
                np.inf,
                epsabs=0,
            )[0]
            / nc
            - ck_mn**2
        )
        assert np.isclose(va, ck_va)

    def test_mutation_rootward_moments(self, pars):
        """
        Mutation mapped to a single branch with child node fixed
        """

        def f(t_i, t_j):  # conditional moments
            assert t_j < t_i
            mn = t_i / 2 + t_j / 2
            sq = (t_i**2 + t_i * t_j + t_j**2) / 3
            return mn, sq

        a_i, b_i, a_j, b_j, y, mu = pars
        pars_redux = (a_i, b_i, y, mu)
        mn_j = a_j / b_j  # point "estimate" for child
        for t_j in [0.0, mn_j]:
            mn, va = mutation_rootward_moments(t_j, *pars_redux)
            nc = scipy.integrate.quad(
                lambda t_i: self.pdf_rootward(t_i, t_j, *pars_redux),
                t_j,
                np.inf,
            )[0]
            ck_mn = (
                scipy.integrate.quad(
                    lambda t_i: f(t_i, t_j)[0] * self.pdf_rootward(t_i, t_j, *pars_redux),
                    t_j,
                    np.inf,
                )[0]
                / nc
            )
            assert np.isclose(mn, ck_mn)
            ck_va = (
                scipy.integrate.quad(
                    lambda t_i: f(t_i, t_j)[1] * self.pdf_rootward(t_i, t_j, *pars_redux),
                    t_j,
                    np.inf,
                )[0]
                / nc
                - ck_mn**2
            )
            assert np.isclose(va, ck_va)

    def test_mutation_leafward_moments(self, pars):
        """
        Mutation mapped to a single branch with parent node fixed
        """

        def f(t_i, t_j):
            assert t_j < t_i
            mn = t_i / 2 + t_j / 2
            sq = (t_i**2 + t_i * t_j + t_j**2) / 3
            return mn, sq

        a_i, b_i, a_j, b_j, y, mu = pars
        t_i = a_i / b_i  # point "estimate" for parent
        pars_redux = (a_j, b_j, y, mu)
        mn, va = mutation_leafward_moments(t_i, *pars_redux)
        nc = scipy.integrate.quad(
            lambda t_j: self.pdf_leafward(t_i, t_j, *pars_redux),
            0,
            t_i,
        )[0]
        ck_mn = (
            scipy.integrate.quad(
                lambda t_j: f(t_i, t_j)[0] * self.pdf_leafward(t_i, t_j, *pars_redux),
                0,
                t_i,
            )[0]
            / nc
        )
        assert np.isclose(mn, ck_mn)
        ck_va = (
            scipy.integrate.quad(
                lambda t_j: f(t_i, t_j)[1] * self.pdf_leafward(t_i, t_j, *pars_redux),
                0,
                t_i,
            )[0]
            / nc
            - ck_mn**2
        )
        assert np.isclose(va, ck_va)

    def test_mutation_unphased_moments(self, pars):
        """
        Mutation mapped to two singleton branches with children fixed to time zero
        """

        def f(t_i, t_j):  # conditional moments
            pr = t_i / (t_i + t_j)
            mn = pr * t_i / 2 + (1 - pr) * t_j / 2
            sq = pr * t_i**2 / 3 + (1 - pr) * t_j**2 / 3
            return pr, mn, sq

        pr, mn, va = mutation_unphased_moments(*pars)
        nc = scipy.integrate.dblquad(
            lambda t_i, t_j: self.pdf_unphased(t_i, t_j, *pars),
            0,
            np.inf,
            0,
            np.inf,
            epsabs=0,
        )[0]
        ck_pr = (
            scipy.integrate.dblquad(
                lambda t_i, t_j: f(t_i, t_j)[0] * self.pdf_unphased(t_i, t_j, *pars),
                0,
                np.inf,
                0,
                np.inf,
                epsabs=0,
            )[0]
            / nc
        )
        assert np.isclose(pr, ck_pr)
        ck_mn = (
            scipy.integrate.dblquad(
                lambda t_i, t_j: f(t_i, t_j)[1] * self.pdf_unphased(t_i, t_j, *pars),
                0,
                np.inf,
                0,
                np.inf,
                epsabs=0,
            )[0]
            / nc
        )
        assert np.isclose(mn, ck_mn)
        ck_va = (
            scipy.integrate.dblquad(
                lambda t_i, t_j: f(t_i, t_j)[2] * self.pdf_unphased(t_i, t_j, *pars),
                0,
                np.inf,
                0,
                np.inf,
                epsabs=0,
            )[0]
            / nc
            - ck_mn**2
        )
        assert np.isclose(va, ck_va)

    def test_mutation_twin_moments(self, pars):
        """
        Mutation mapped to two singleton branches with children fixed to time zero
        and a single parent node
        """

        def f(t_i):  # conditional moments
            pr = 0.5
            mn = t_i / 2
            sq = t_i**2 / 3
            return pr, mn, sq

        a_i, b_i, a_j, b_j, y_ij, mu_ij = pars
        pars_redux = (a_i, b_i, y_ij, mu_ij)
        pr, mn, va = mutation_twin_moments(*pars_redux)
        nc = scipy.integrate.quad(
            lambda t_i: self.pdf_twin(t_i, *pars_redux),
            0,
            np.inf,
            epsabs=0,
        )[0]
        ck_pr = (
            scipy.integrate.quad(
                lambda t_i: f(t_i)[0] * self.pdf_twin(t_i, *pars_redux),
                0,
                np.inf,
                epsabs=0,
            )[0]
            / nc
        )
        assert np.isclose(pr, ck_pr)
        ck_mn = (
            scipy.integrate.quad(
                lambda t_i: f(t_i)[1] * self.pdf_twin(t_i, *pars_redux),
                0,
                np.inf,
                epsabs=0,
            )[0]
            / nc
        )
        assert np.isclose(mn, ck_mn)
        ck_va = (
            scipy.integrate.quad(
                lambda t_i: f(t_i)[2] * self.pdf_twin(t_i, *pars_redux),
                0,
                np.inf,
                epsabs=0,
            )[0]
            / nc
            - ck_mn**2
        )
        assert np.isclose(va, ck_va)

    def test_mutation_sideways_moments(self, pars):
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
        pars_redux = a_j, b_j, y, mu
        t_i = a_i / b_i  # point "estimate" for left parent
        pr, mn, va = mutation_sideways_moments(t_i, *pars_redux)
        nc = scipy.integrate.quad(
            lambda t_j: self.pdf_sideways(t_i, t_j, *pars_redux),
            0,
            np.inf,
        )[0]
        ck_pr = (
            scipy.integrate.quad(
                lambda t_j: f(t_i, t_j)[0] * self.pdf_sideways(t_i, t_j, *pars_redux),
                0,
                np.inf,
            )[0]
            / nc
        )
        assert np.isclose(pr, ck_pr)
        ck_mn = (
            scipy.integrate.quad(
                lambda t_j: f(t_i, t_j)[1] * self.pdf_sideways(t_i, t_j, *pars_redux),
                0,
                np.inf,
            )[0]
            / nc
        )
        assert np.isclose(mn, ck_mn)
        ck_va = (
            scipy.integrate.quad(
                lambda t_j: f(t_i, t_j)[2] * self.pdf_sideways(t_i, t_j, *pars_redux),
                0,
                np.inf,
            )[0]
            / nc
            - ck_mn**2
        )
        assert np.isclose(va, ck_va)

    def test_mutation_edge_moments(self, pars):
        """
        Mutation mapped to two branches with children fixed to time zero, and
        both parents fixed
        """
        a_i, b_i, a_j, b_j, y, mu = pars
        t_i = a_i / b_i
        t_j = a_j / b_j
        mn, va = mutation_edge_moments(t_i, t_j)
        ck_mn = scipy.integrate.quad(
            lambda x: x * self.pdf_edge(x, t_i, t_j),
            0,
            max(t_i, t_j),
        )[0]
        assert np.isclose(mn, ck_mn)
        ck_va = (
            scipy.integrate.quad(
                lambda x: x**2 * self.pdf_edge(x, t_i, t_j),
                0,
                max(t_i, t_j),
            )[0]
            - ck_mn**2
        )
        assert np.isclose(va, ck_va)

    def test_mutation_block_moments(self, pars):
        """
        Mutation mapped to two branches with children fixed to time zero, and
        both parents fixed
        """
        a_i, b_i, a_j, b_j, y, mu = pars
        t_i = a_i / b_i
        t_j = a_j / b_j
        pars_redux = (a_j, b_j, y, mu)
        pr, mn, va = mutation_block_moments(t_i, t_j)
        ck_pr = t_i / (t_i + t_j)
        assert np.isclose(pr, ck_pr)
        ck_mn = scipy.integrate.quad(
            lambda x: x * self.pdf_block(x, t_i, t_j),
            0,
            max(t_i, t_j),
        )[0]
        assert np.isclose(mn, ck_mn)
        ck_va = (
            scipy.integrate.quad(
                lambda x: x**2 * self.pdf_block(x, t_i, t_j),
                0,
                max(t_i, t_j),
            )[0]
            - ck_mn**2
        )
        assert np.isclose(va, ck_va)


def validate():
    tests = TestExactMoments()
    test_names = [f for f in dir(tests) if f.startswith("test")]
    test_cases = [  # [shape1, rate1, shape2, rate2, muts, rate]
        [2.0, 0.0005, 1.5, 0.005, 0.0, 0.001],
        [2.0, 0.0005, 1.5, 0.005, 1.0, 0.001],
        [2.0, 0.0005, 1.5, 0.005, 2.0, 0.001],
        [2.0, 0.0005, 1.5, 0.005, 3.0, 0.001],
    ]
    for pars in test_cases:
        for test in test_names:
            getattr(tests, test)(pars)
