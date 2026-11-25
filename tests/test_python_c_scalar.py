"""
Tests of the generic scalar-function wrapper of the low-level _tsdate module: argument
handling, return types, and the mapping of library errors to exceptions. Numerical
values are tested against the oracles in test_oracle_*.py.
"""

import math
import subprocess
import sys

import _tsdate
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


class TestGammaincInv:
    """
    ``gammainc_inv`` calls SciPy's ``gammaincinv`` through its Cython C API, so it is
    identical to ``scipy.special.gammaincinv``, including outside the domain.
    """

    @pytest.mark.parametrize(
        ("a", "q"),
        [
            (0.5, 0.3),
            (2.0, 0.9),
            (1e5, 1e-10),
            (1e-3, 0.99),
            (2.0, 0.0),
            (2.0, 1.0),
            (-1.0, 0.5),
            (2.0, 1.5),
            (math.nan, 0.5),
        ],
    )
    def test_matches_scipy(self, a, q):
        actual = _tsdate.gammainc_inv(a, q)
        expected = scipy.special.gammaincinv(a, q)
        assert actual == expected or (math.isnan(actual) and math.isnan(expected))


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
