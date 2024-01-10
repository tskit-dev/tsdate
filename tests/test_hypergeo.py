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
Test cases for numba-fied hypergeometric functions
"""
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


@pytest.mark.parametrize(
    "pars",
    [
        [969.5, 0.1062, 984.72, 0.1215, 0.0, 0.000239],
        [897.8, 0.0954, 936.03, 0.1042, 1.0, 0.000840],
        [37.94, 0.0087, 413.54, 0.1122, 0.0, 2.70e-05],
        [748.1, 0.0857, 544.65, 0.0621, 0.0, 0.000511],
        [21.18, 0.0226, 21.898, 0.0292, 0.0, 0.000618],
        [7.931, 0.0010, 1000.0, 0.1675, 0.0, 0.000290],
        [77.15, 0.0484, 42.007, 0.0348, 0.0, 0.000277],
        [990.5, 0.1318, 130.68, 0.0205, 1.0, 0.000245],
        [678.9, 0.1184, 121.71, 0.0958, 2.0, 0.000208],
    ],
)
class TestLaplaceApprox2F1:
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

    def test_2f1(self, pars):
        a_i, b_i, a_j, b_j, y, mu = pars
        A = a_j
        B = a_i + a_j + y
        C = a_j + y + 1
        z = (mu - b_j) / (mu + b_i)
        f = hypergeo._hyp2f1_laplace(A, B, C, z)
        check = float(mpmath.log(self._2f1_validate(*pars)))
        assert np.isclose(f, check, rtol=2e-2)


@pytest.mark.parametrize(
    "pars",
    [
        [202.865, 0.0153, 0.0012724, 101353547.00, 0.0, 2.404560e-05],
        [999.998, 0.1041, 0.0012612, 14749022.000, 2.0, 7.830300e-05],
        [157.778, 0.0182, 0.0011993, 20861958.000, 0.0, 4.128000e-07],
        [1000.00, 0.1431, 0.0009718, 22638776.000, 0.0, 5.881110e-05],
        [405.482, 0.1484, 0.0009711, 41053809.000, 0.0, 6.229410e-05],
        [115.773, 0.7506, 0.0015549, 23315696857.0, 0.0, 3.601680e-05],
        [73.4909, 1.0600, 0.0010028, 56254280956.0, 0.0, 0.0002302392],
        [18.9981, 0.1315, 0.0015053, 23443723634.0, 1.0, 2.470348e-05],
    ],
)
class Test2F1Unity:
    """
    As TestLaplaceApprox2F1 but in the argument-close-to-unity limit
    """

    @staticmethod
    def _2f1_validate(a_i, b_i, a_j, b_j, y, mu, offset=1.0):
        A = a_j
        B = a_i + a_j + y
        C = a_j + y + 1
        z = (mu - b_j) / (mu + b_i)
        val = mpmath.re(mpmath.hyp2f1(A, B, C, z, maxterms=1e7))
        return val / offset

    def test_2f1(self, pars):
        a_i, b_i, a_j, b_j, y, mu = pars
        A = a_j
        B = a_i + a_j + y
        C = a_j + y + 1
        z = (mu - b_j) / (mu + b_i)
        s = 0.0
        if z < 0.0:
            s = -B * np.log(1 - z)
            A = C - A
            z = z / (z - 1)
        f = hypergeo._hyp2f1_unity(A, B, C, z) + s
        check = float(mpmath.log(self._2f1_validate(*pars)))
        assert np.isclose(f, check, rtol=1e-2)


@pytest.mark.parametrize(
    "pars",
    [
        [92.301156, 3.338, 0.008172, 0.0, 0.00022691],
        [92.301156, 3.678, 0.005831, 0.0, 0.00250921],
        [2998.6825, 3.999, 0.000750, 1.0, 0.00098396],
        [11985.278, 5.251, 0.000193, 8.0, 1.9537e-05],
        [11985.278, 14.46, 0.000822, 3.0, 0.00011632],
        [12957.347, 0.999, 9.999e-5, 1.0, 0.00019150],
    ],
)
class TestLaplaceApproxU:
    """
    Test that Laplace approximation to Tricomi's confluent hypergeometric
    function returns reasonable answers
    """

    @staticmethod
    def _hyperu_validate(t_j, a_i, b_i, y, mu, offset=1.0):
        A = y + 1
        B = a_i + y + 1
        z = t_j * (mu + b_i)
        val = mpmath.re(mpmath.hyperu(A, B, z, maxterms=1e6))
        return val / offset

    def test_hyperu(self, pars):
        t_j, a_i, b_i, y, mu = pars
        A = y + 1
        B = a_i + y + 1
        z = t_j * (mu + b_i)
        f, df = hypergeo._hyperu_laplace(A, B, z)
        der = nd.Derivative(
            lambda z: hypergeo._hyperu_laplace(A, B, z)[0], n=1, step=1e-4
        )
        check_f = float(mpmath.log(self._hyperu_validate(*pars)))
        check_df = der(z)
        assert np.isclose(f, check_f, rtol=2e-2)
        assert np.isclose(df, check_df, rtol=2e-2)


@pytest.mark.parametrize(
    "pars",
    [
        [25128.46, 132.48, 0.006849, 0.0, 4.4562e-05],
        [15900.89, 997.09, 0.093978, 1.0, 0.00010288],
        [10062.80, 978.24, 0.102753, 0.0, 0.00022747],
        [11659.45, 713.52, 0.064035, 8.0, 0.00064625],
        [40888.01, 999.99, 0.071372, 3.0, 3.7925e-05],
        [11659.45, 989.08, 0.096664, 2.0, 0.00010425],
        [22333.94, 28.546, 0.001817, 0.0, 9.6225e-05],
        [30613.25, 30.956, 0.000942, 0.0, 1.1250e-07],
    ],
)
class TestLaplaceApproxM:
    """
    Test that Laplace approximation to Kummer's confluent hypergeometric
    function returns reasonable answers
    """

    @staticmethod
    def _hyp1f1_validate(t_i, a_j, b_j, y, mu, offset=1.0):
        A = a_j
        B = a_j + y + 1
        z = t_i * (b_j - mu)
        val = mpmath.re(mpmath.hyp1f1(A, B, z, maxterms=1e6))
        return val / offset

    def test_hyp1f1(self, pars):
        t_i, a_j, b_j, y, mu = pars
        A = a_j
        B = a_j + y + 1
        z = t_i * (b_j - mu)
        f = hypergeo._hyp1f1_laplace(A, B, z)
        check = float(mpmath.log(self._hyp1f1_validate(*pars)))
        assert np.isclose(f, check, rtol=2e-2)
