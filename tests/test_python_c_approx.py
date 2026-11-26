"""
Tests of the low-level _tsdate module for the approx functions: error messages,
argument checking of the array-level function, and the test adapters. Numerical values
are tested against the oracles in test_oracle_approx.py.
"""

import re

import _tsdate
import numpy as np
import pytest
from oracle import approx
from oracle.helpers import compare

# error code: the message of the oracle's KLMinimizationFailedError (or assertion)
APPROX_ERRORS = {
    -6: "Nonpositive or nonfinite moments",
    -7: "log E[t] <= E[log t] violates Jensen's inequality",
    -8: "Maximum iterations reached in KL minimization",
    -9: "Invalid shape parameter in KL minimization",
    -10: "Nonpositive central moments",
    -11: "Quantiles must be sorted",
    -12: "Maximum iterations reached in quantile matching",
    -13: "Negative shape parameter",
    -14: "The mean and variance must be positive",
    -15: "The age of a fixed node is outside the domain of the function",
}


class TestApproxErrors:
    @pytest.mark.parametrize(("code", "message"), APPROX_ERRORS.items())
    def test_messages(self, code, message):
        assert _tsdate.strerror(code) == message
        with pytest.raises(_tsdate.LibraryError, match=re.escape(message)):
            _tsdate.raise_library_error(code)


class TestAverageGammas:
    def test_returns_floats(self):
        alpha = np.array([0.5, 3.0])
        beta = np.array([1.0, 2.0])
        result = _tsdate.average_gammas(alpha, beta)
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert all(isinstance(x, float) for x in result)

    def test_accepts_lists_and_integers(self):
        assert _tsdate.average_gammas([1, 2], [1, 2]) == _tsdate.average_gammas(
            np.array([1.0, 2.0]), np.array([1.0, 2.0])
        )

    def test_size_mismatch(self):
        with pytest.raises(ValueError, match="Array sizes are not equal"):
            _tsdate.average_gammas(np.ones(2), np.ones(3))

    @pytest.mark.parametrize("shape", [(), (2, 2)])
    def test_wrong_dimension(self, shape):
        with pytest.raises(ValueError):
            _tsdate.average_gammas(np.ones(shape), np.ones(shape))

    def test_wrong_dimension_second_argument(self):
        with pytest.raises(ValueError):
            _tsdate.average_gammas(np.ones(2), np.ones((2, 2)))

    def test_not_numeric(self):
        with pytest.raises(ValueError):
            _tsdate.average_gammas(["a"], [1.0])

    def test_wrong_number_of_arguments(self):
        with pytest.raises(TypeError):
            _tsdate.average_gammas(np.ones(2))

    def test_library_error(self):
        with pytest.raises(_tsdate.LibraryError, match="division by zero"):
            _tsdate.average_gammas(np.array([]), np.array([]))


# Failures of the quantile fit (found by random search), each at a different step of
# the Newton iteration: the lower quantile underflows to zero, its square underflows
# to zero, and the iteration does not converge.
GAMMA_IQR_FAILURES = {
    "lower quantile is zero": (1e-300, 1e-299, 1.0, 1e30, 50.0),
    "lower quantile squared is zero": (
        0.25,
        0.75,
        5.256386560310826e-218,
        9.53e-42,
        50.0,
    ),
    "no convergence": (
        4.195119806947257e-42,
        1.3790748660114965e-37,
        1.656873535271179e-06,
        8.871129030923311e-06,
        1e7,
    ),
}


class TestGammaIqrFailures:
    @pytest.mark.parametrize("label", list(GAMMA_IQR_FAILURES))
    def test_failures(self, label):
        args = GAMMA_IQR_FAILURES[label]
        expected = compare(
            _tsdate.approximate_gamma_iqr, approx.approximate_gamma_iqr, *args
        )
        assert expected.exception is not None
