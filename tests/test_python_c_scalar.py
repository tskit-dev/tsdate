"""
Tests of the generic scalar-function wrapper of the low-level _tsdate module: argument
handling, return types, and the mapping of library errors to exceptions. Numerical
values are tested against the oracles in test_oracle_*.py.
"""

import math
import subprocess
import sys

import _tsdate
import numpy as np
import pytest
import scipy.special

SCALAR_FUNCTIONS = {
    # name: (number of arguments, number of results, valid arguments)
    "digamma": (1, 1, (1.5,)),
    "trigamma": (1, 1, (1.5,)),
    "betaln": (2, 1, (2.0, 3.0)),
    "gammainc_der": (2, 1, (2.0, 1.5)),
    "gammainc_inv": (2, 1, (2.0, 0.3)),
    "hyperu_laplace": (3, 2, (1.0, 2.0, 1.0)),
    "hyp1f1_laplace": (3, 1, (1.0, 2.0, 1.0)),
    "hyp2f1_laplace": (4, 1, (1.0, 2.0, 3.0, 0.5)),
}


class TestScalarFunctions:
    @pytest.mark.parametrize("name", SCALAR_FUNCTIONS)
    def test_result_type(self, name):
        _, num_results, args = SCALAR_FUNCTIONS[name]
        result = getattr(_tsdate, name)(*args)
        if num_results == 1:
            assert isinstance(result, float)
        else:
            assert isinstance(result, tuple)
            assert len(result) == num_results
            assert all(isinstance(x, float) for x in result)

    def test_integers_accepted(self):
        assert _tsdate.betaln(2, 3) == _tsdate.betaln(2.0, 3.0)

    @pytest.mark.parametrize("name", SCALAR_FUNCTIONS)
    def test_wrong_number_of_arguments(self, name):
        num_args, _, args = SCALAR_FUNCTIONS[name]
        func = getattr(_tsdate, name)
        with pytest.raises(TypeError, match=f"{name}.*{num_args} arguments"):
            func()
        with pytest.raises(TypeError, match=f"{name}.*{num_args} arguments"):
            func(*args, 1.0)

    @pytest.mark.parametrize("name", SCALAR_FUNCTIONS)
    def test_non_numeric_argument(self, name):
        _, _, args = SCALAR_FUNCTIONS[name]
        func = getattr(_tsdate, name)
        with pytest.raises(TypeError):
            func("x", *args[1:])
        with pytest.raises(TypeError):
            func(*args[:-1], None)

    @pytest.mark.parametrize("name", SCALAR_FUNCTIONS)
    def test_keyword_arguments_rejected(self, name):
        _, _, args = SCALAR_FUNCTIONS[name]
        with pytest.raises(TypeError):
            getattr(_tsdate, name)(*args[:-1], x=args[-1])


class TestScalarErrors:
    def test_zero_division(self):
        with pytest.raises(_tsdate.LibraryError, match="division by zero"):
            _tsdate.digamma(0.0)

    def test_invalid_hypergeo(self):
        with pytest.raises(_tsdate.LibraryError, match="domain of the hypergeometric"):
            _tsdate.hyp2f1_laplace(1.0, 2.0, 3.0, 1.0)

    def test_messages_match_strerror(self):
        with pytest.raises(_tsdate.LibraryError) as exc_info:
            _tsdate.gammainc_der(-1.0, 1.0)
        assert str(exc_info.value) == _tsdate.strerror(-5)

    def test_nan_is_not_an_error(self):
        assert math.isnan(_tsdate.gammainc_inv(math.nan, 0.5))


def gammainc_inv_grid():
    inf, nan = math.inf, math.nan
    shapes = [nan, -inf, -1e300, -1e5 - 0.5, -100.5, -40.7, -40.0, -35.7, -35.0, -34.5]
    shapes += [-33.5, -1.5, -1.0, -0.5, -1e-10, 3e305]
    shapes += [-0.0, 0.0, 5e-324, 1e-300, 1e-100, 1e-10, 1e-3, 0.01, 0.1, 0.2, 0.25]
    shapes += [0.3, 0.5, 0.9, 1.0, 1.1, 2.0, 5.0, 10.0, 20.0, 21.0, 50.0, 150.0]
    shapes += [
        25.0,
        30.0,
        40.0,
        60.0,
        100.0,
        170.0,
        171.7,
        199.0,
        200.0,
        201.0,
        500.0,
        1e3,
        1e4,
        1e5,
        1e6,
        1e8,
    ]
    shapes += [1e10, 1e15, 1e300, inf]
    quantiles = [nan, -inf, -0.1, -0.0, 0.0, 5e-324, 1e-300, 1e-100, 1e-20, 1e-10]
    quantiles += [1e-5, 1e-3, 0.01, 0.1, 0.3, 0.5, 0.7, 0.9, np.nextafter(0.9, 1)]
    quantiles += [0.95, 0.99, 1 - 1e-10, np.nextafter(1.0, 0), 1.0]
    quantiles += [np.nextafter(1.0, 2), 1.1, inf]
    return [(a, q) for a in shapes for q in quantiles]


def assert_identical(actual, expected):
    """Equal values (including the sign of zero), or both NaN"""
    both_nan = np.isnan(actual) & np.isnan(expected)
    same = (actual == expected) & (np.signbit(actual) == np.signbit(expected))
    assert np.all(both_nan | same)


class TestGammaincInv:
    """
    ``gammainc_inv`` calls SciPy's ``gammaincinv`` through its Cython C API, so it is
    identical to ``scipy.special.gammaincinv``, including outside the domain.
    """

    def test_grid(self):
        a, q = np.array(gammainc_inv_grid()).T
        actual = np.array([_tsdate.gammainc_inv(*x) for x in zip(a, q)])
        assert_identical(actual, scipy.special.gammaincinv(a, q))

    @pytest.mark.parametrize("seed", [1, 2])
    def test_random_draws(self, seed):
        rng = np.random.default_rng(seed)
        a = 10 ** rng.uniform(-6, 6, 5000)
        q = np.where(
            rng.random(5000) < 0.5, rng.random(5000), 10 ** rng.uniform(-12, 0, 5000)
        )
        actual = np.array([_tsdate.gammainc_inv(*x) for x in zip(a, q)])
        assert_identical(actual, scipy.special.gammaincinv(a, q))


class TestScipyDependency:
    def test_import_fails_without_scipy_cython_api(self):
        code = (
            "import sys; sys.modules['scipy.special.cython_special'] = None\n"
            "import _tsdate"
        )
        result = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True
        )
        assert result.returncode != 0
        assert "ImportError" in result.stderr
        assert "cython_special" in result.stderr
