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
Test cases for numba-fied hypergeometric functions
"""
import itertools

import mpmath
import numdifftools as nd
import numpy as np
import pytest

from tsdate import hypergeo


@pytest.mark.parametrize("x", [-0.3, 1e-10, 1e-6, 1e-2, 1e1, 1e2, 1e3, 1e5, 1e10])
class TestPolygamma:
    """
    Test numba-fied gamma functions
    """

    def test_gammaln(self, x):
        assert np.isclose(hypergeo._gammaln(x), float(mpmath.re(mpmath.loggamma(x))))

    def test_digamma(self, x):
        assert np.isclose(hypergeo._digamma(x), float(mpmath.psi(0, x)))

    def test_trigamma(self, x):
        assert np.isclose(hypergeo._trigamma(x), float(mpmath.psi(1, x)))

    def test_betaln(self, x):
        assert np.isclose(
            hypergeo._betaln(x, 2 * x),
            float(mpmath.re(mpmath.log(mpmath.beta(x, 2 * x)))),
        )


@pytest.mark.parametrize("a_i", [1.0, 10.0, 100.0, 1000.0])
@pytest.mark.parametrize("b_i", [0.001, 0.01, 0.1, 1.0])
@pytest.mark.parametrize("a_j", [1.0, 10.0, 100.0, 1000.0])
@pytest.mark.parametrize("b_j", [0.001, 0.01, 0.1, 1.0])
@pytest.mark.parametrize("y", [0.0, 1.0, 10.0, 1000.0])
@pytest.mark.parametrize("mu", [0.005, 0.05, 0.5, 5.0])
class TestLaplaceApprox:
    """
    Test that Laplace approximation to 2F1 returns reasonable answers
    """

    @staticmethod
    def _2f1_validate(a_i, b_i, a_j, b_j, y, mu, offset=1.0):
        A = a_j
        B = a_i + a_j + y
        C = a_j + y + 1
        z = (mu - b_j) / (mu + b_i)
        val = mpmath.re(mpmath.hyp2f1(A, B, C, z, maxterms=1e6))
        return val / offset

    def test_2f1(self, a_i, b_i, a_j, b_j, y, mu):
        pars = [a_i, b_i, a_j, b_j, y, mu]
        A = a_j
        B = a_i + a_j + y
        C = a_j + y + 1
        z = (mu - b_j) / (mu + b_i)
        f, *_ = hypergeo._hyp2f1(*pars)
        ff = hypergeo._hyp2f1_fast(A, B, C, z)
        check = float(mpmath.log(self._2f1_validate(*pars)))
        assert np.isclose(f, ff)
        assert np.isclose(f, check, rtol=2e-2)

    def test_grad(self, a_i, b_i, a_j, b_j, y, mu):
        pars = [a_i, b_i, a_j, b_j, y, mu]
        _, *grad = hypergeo._hyp2f1(*pars)
        da_i = nd.Derivative(
            lambda a_i: hypergeo._hyp2f1(a_i, b_i, a_j, b_j, y, mu)[0], step=1e-3
        )
        db_i = nd.Derivative(
            lambda b_i: hypergeo._hyp2f1(a_i, b_i, a_j, b_j, y, mu)[0], step=1e-5
        )
        da_j = nd.Derivative(
            lambda a_j: hypergeo._hyp2f1(a_i, b_i, a_j, b_j, y, mu)[0], step=1e-3
        )
        db_j = nd.Derivative(
            lambda b_j: hypergeo._hyp2f1(a_i, b_i, a_j, b_j, y, mu)[0], step=1e-5
        )
        check = [da_i(a_i), db_i(b_i), da_j(a_j), db_j(b_j)]
        assert np.allclose(grad, check, rtol=1e-3)


# ------------------------------------------------- #
# The routines below aren't used in tsdate anymore, #
# but may be useful in the future                   #
# ------------------------------------------------- #


@pytest.mark.parametrize(
    "pars",
    list(
        itertools.product(
            [0.8, 20.0, 200.0],
            [1.9, 90.3, 900.3],
            [1.6, 30.7, 300.7],
            [0.0, 0.1, 0.45],
        )
    ),
)
class TestTaylorSeries:
    """
    Test Taylor series expansions of 2F1
    """

    @staticmethod
    def _2f1_validate(a, b, c, z, offset=1.0):
        val = mpmath.re(mpmath.hyp2f1(a, b, c, z))
        return val / offset

    @staticmethod
    def _2f1_grad_validate(a, b, c, z, offset=1.0):
        p = [a, b, c, z]
        grad = nd.Gradient(
            lambda x: float(TestTaylorSeries._2f1_validate(*x, offset=offset)),
            step=1e-7,
            richardson_terms=4,
        )
        return grad(p)

    def test_2f1(self, pars):
        f, *_ = hypergeo._hyp2f1_taylor_series(*pars)
        check = self._2f1_validate(*pars)
        assert np.isclose(f, float(mpmath.log(mpmath.fabs(check))))

    def test_2f1_grad(self, pars):
        _, *grad = hypergeo._hyp2f1_taylor_series(*pars)
        offset = self._2f1_validate(*pars)
        check = self._2f1_grad_validate(*pars, offset=offset)
        assert np.allclose(grad, check)


@pytest.mark.parametrize(
    "pars",
    list(
        itertools.product(
            [-130.1, 20.0, 200.0],
            [-20.2, 90.3, 900.3],
            [-1.6, 30.7, 300.7],
            [-3.0, -0.5, 0.0, 0.1, 0.9],
        )
    ),
)
class TestCheckValid2F1:
    """
    Test the check for 2F1 series convergence, that plugs derivatives into the
    differential equation that defines 2F1
    """

    @staticmethod
    def _2f1(a, b, c, z):
        val = mpmath.re(mpmath.hyp2f1(a, b, c, z))
        dz = a * b / c * mpmath.re(mpmath.hyp2f1(a + 1, b + 1, c + 1, z))
        d2z = (
            a
            * b
            / c
            * (a + 1)
            * (b + 1)
            / (c + 1)
            * mpmath.re(mpmath.hyp2f1(a + 2, b + 2, c + 2, z))
        )
        return float(dz / val), float(d2z / val)

    def test_is_valid_2f1(self, pars):
        dz, d2z = self._2f1(*pars)
        assert hypergeo._is_valid_2f1(dz, d2z, *pars, 1e-10)
        # perturb solution to differential equation
        dz *= 1 + 1e-3
        d2z *= 1 - 1e-3
        assert not hypergeo._is_valid_2f1(dz, d2z, *pars, 1e-10)
