"""
The oracle's ``ipow`` (``oracle/_libm.py``) against numba's ``x ** n``, for the small
integer constants ``n`` used in tsdate.

This is the only test module that imports numba. It checks the oracle against the
numba behaviour it imitates, so it is deleted together with numba (step E).
"""

import numba
import numpy as np
import pytest
from oracle._libm import ipow
from oracle.helpers import INF, NAN


def compile_power(n):
    @numba.njit("f8[:](f8[:])")
    def power(x):
        out = np.empty_like(x)
        for i in range(x.size):
            out[i] = x[i] ** n
        return out

    return power


class TestIpow:
    @pytest.mark.parametrize("seed", [1, 2])
    @pytest.mark.parametrize("n", [2, 3, 4, 5, 6, 7])
    def test_matches_numba(self, n, seed):
        rng = np.random.default_rng(seed)
        x = np.concatenate(
            [
                10.0 ** rng.uniform(-3, 3, 2000),
                -(10.0 ** rng.uniform(-100, 100, 500)),
                [0.0, -0.0, 1.0, -1.0, 5e-324, 1e-170, 1e170, INF, -INF, NAN],
            ]
        )
        actual = compile_power(n)(x)
        with np.errstate(all="ignore"):
            expected = np.array([ipow(xi, n) for xi in x.tolist()])
        np.testing.assert_array_equal(np.isnan(actual), np.isnan(expected))
        finite = ~np.isnan(expected)
        # bit for bit, including the sign of zero
        np.testing.assert_array_equal(
            actual[finite].view(np.int64), expected[finite].view(np.int64)
        )
