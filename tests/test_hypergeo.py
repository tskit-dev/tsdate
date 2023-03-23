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


@pytest.mark.parametrize("x", [1e-10, 1e-6, 1e-2, 1e1, 1e2, 1e3, 1e5, 1e10])
class TestPolygamma:
    """
    Test numba-fied gamma functions
    """

    def test_gammaln(self, x):
        assert np.isclose(hypergeo._gammaln(x), float(mpmath.loggamma(x)))

    def test_digamma(self, x):
        assert np.isclose(hypergeo._digamma(x), float(mpmath.psi(0, x)))

    def test_trigamma(self, x):
        assert np.isclose(hypergeo._trigamma(x), float(mpmath.psi(1, x)))

    def test_betaln(self, x):
        assert np.isclose(
            hypergeo._betaln(x, 2 * x), float(mpmath.log(mpmath.beta(x, 2 * x)))
        )


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
        f, s, *_ = hypergeo._hyp2f1_taylor_series(*pars)
        check = self._2f1_validate(*pars)
        assert s == mpmath.sign(check)
        print(f, mpmath.log(mpmath.fabs(check)))
        assert np.isclose(f, float(mpmath.log(mpmath.fabs(check))))

    def test_2f1_grad(self, pars):
        _, _, *grad = hypergeo._hyp2f1_taylor_series(*pars)
        grad = grad[:-1]
        offset = self._2f1_validate(*pars)
        check = self._2f1_grad_validate(*pars, offset=offset)
        print(grad, check)
        assert np.allclose(grad, check)


@pytest.mark.parametrize(
    "pars",
    list(
        itertools.product(
            [0.8, 20.3, 200.2],
            [0.0, 1.0, 10.0, 51.0],
            [1.6, 30.5, 300.7],
            [1.1, 1.5, 1.9],
        )
    ),
)
class TestRecurrence:
    """
    Test recurrence for 2F1 when one parameter is a negative integer
    """

    @staticmethod
    def _transform_pars(a, b, c, z):
        return a, b, c + a, z

    @staticmethod
    def _2f1_validate(a, b, c, z, offset=1.0):
        val = mpmath.re(mpmath.hyp2f1(a, -b, c, z))
        return val / offset

    @staticmethod
    def _2f1_grad_validate(a, b, c, z, offset=1.0):
        p = [a, b, c, z]
        grad = nd.Gradient(
            lambda x: float(TestRecurrence._2f1_validate(*x, offset=offset)),
            step=1e-6,
            richardson_terms=4,
        )
        return grad(p)

    def test_2f1(self, pars):
        pars = self._transform_pars(*pars)
        f, s, *_ = hypergeo._hyp2f1_recurrence(*pars)
        check = self._2f1_validate(*pars)
        assert s == mpmath.sign(check)
        print(f, mpmath.log(mpmath.fabs(check)))
        assert np.isclose(f, float(mpmath.log(mpmath.fabs(check))))

    def test_2f1_grad(self, pars):
        pars = self._transform_pars(*pars)
        _, _, *grad = hypergeo._hyp2f1_recurrence(*pars)
        grad = grad[:-1]
        offset = self._2f1_validate(*pars)
        check = self._2f1_grad_validate(*pars, offset=offset)
        check[1] = 0.0  # integer parameter has no gradient
        print(grad, check)
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
        assert hypergeo._is_valid_2f1(dz, d2z, *pars)
        # perturb solution to differential equation
        dz *= 1 + 1e-3
        d2z *= 1 - 1e-3
        assert not hypergeo._is_valid_2f1(dz, d2z, *pars)


@pytest.mark.parametrize("muts", [0.0, 1.0, 5.0, 10.0])
@pytest.mark.parametrize(
    "hyp2f1_func, pars",
    [
        (hypergeo._hyp2f1_dlmf1521, [1.4, 0.018, 2.34, 2.3e-05, 0.0, 0.0395]),
        (hypergeo._hyp2f1_dlmf1581, [1.4, 0.018, 20.3, 0.04, 0.0, 2.3e-05]),
        (hypergeo._hyp2f1_dlmf1583, [5.4, 0.018, 10.34, 0.04, 0.0, 2.3e-05]),
    ],
)
class TestTransforms:
    """
    Test numerically stable transformations of hypergeometric functions
    """

    @staticmethod
    def _2f1_validate(a_i, b_i, a_j, b_j, y, mu, offset=1.0):
        A = a_j
        B = a_i + a_j + y
        C = a_j + y + 1
        z = (mu - b_j) / (mu + b_i)
        val = mpmath.re(mpmath.hyp2f1(A, B, C, z, maxterms=1e6))
        return val / offset

    @staticmethod
    def _2f1_grad_validate(a_i, b_i, a_j, b_j, y, mu, offset=1.0):
        p = [a_i, b_i, a_j, b_j]
        grad = nd.Gradient(
            lambda x: float(TestTransforms._2f1_validate(*x, y, mu, offset=offset)),
            step=1e-6,
            richardson_terms=4,
        )
        return grad(p)

    def test_2f1(self, muts, hyp2f1_func, pars):
        pars[4] = muts
        f, s, *_ = hyp2f1_func(*pars)
        assert s > 0
        check = float(mpmath.log(self._2f1_validate(*pars)))
        print(f, check)
        assert np.isclose(f, check)

    def test_2f1_grad(self, muts, hyp2f1_func, pars):
        pars[4] = muts
        _, s, *grad = hyp2f1_func(*pars)
        assert s > 0
        offset = self._2f1_validate(*pars)
        check = self._2f1_grad_validate(*pars, offset=offset)
        print(grad, check)
        assert np.allclose(grad, check)
