"""
Tests of the generic scalar-function wrapper of the low-level _tsdate module: argument
handling and return types. Numerical values and library errors are tested against the
oracles in test_oracle_*.py.
"""

import math
import subprocess
import sys

import _tsdate
import pytest

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
    def test_nan_is_not_an_error(self):
        assert math.isnan(_tsdate.gammainc_inv(math.nan, 0.5))


class TestScipyDependency:
    def test_import_without_scipy(self):
        code = (
            "import sys; sys.modules['scipy'] = None\n"
            "import _tsdate; print(_tsdate.gammainc_inv(2.0, 0.5))"
        )
        result = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True
        )
        assert result.returncode == 0, result.stderr
        assert float(result.stdout) == _tsdate.gammainc_inv(2.0, 0.5)
