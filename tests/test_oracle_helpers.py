"""
The comparison helpers in ``tests/oracle/helpers.py`` detect disagreements: a helper
that passed everything would make every parity test vacuous.
"""

import _tsdate
import numpy as np
import pytest
from oracle import errors
from oracle.errors import library_error
from oracle.helpers import (
    INF,
    NAN,
    assert_outcome,
    assert_same_outcome,
    assert_same_values,
    compare,
    errors_match,
    kind,
    outcome,
)


def returns(value, modify=None):
    def func(x):
        if modify is not None:
            x[0] = modify
        return value

    return func


def raises(exception):
    def func(x):
        raise exception

    return func


class TestAssertSameValues:
    @pytest.mark.parametrize(
        ("actual", "desired", "match"),
        [
            pytest.param(np.nextafter(1.0, 2.0), 1.0, "differ", id="one_ulp"),
            pytest.param(
                np.array([1.0, NAN]), np.array([NAN, 1.0]), "differ", id="nan_place"
            ),
            pytest.param(1.0, NAN, "differ", id="nan_vs_number"),
            pytest.param(INF, -INF, "differ", id="inf_sign"),
            pytest.param((1.0, 2.0), (1.0, 2.0, 3.0), "structure", id="structure"),
            pytest.param(np.zeros(3), np.zeros(4), "shape", id="shape"),
        ],
    )
    def test_fails(self, actual, desired, match):
        with pytest.raises(AssertionError, match=match):
            assert_same_values(actual, desired)


class TestAssertSameOutcome:
    @pytest.mark.parametrize(
        ("impl", "oracle", "match"),
        [
            pytest.param(returns(1.0), returns(2.0), "differ", id="value"),
            pytest.param(
                raises(ValueError("a")), returns(1.0), "implementation", id="impl_raises"
            ),
            pytest.param(
                returns(1.0), raises(ValueError("a")), "oracle", id="oracle_raises"
            ),
            pytest.param(
                raises(KeyError("a")),
                raises(ValueError("a")),
                "implementation",
                id="type",
            ),
            pytest.param(
                raises(ValueError("a")),
                raises(ValueError("b")),
                "implementation",
                id="message",
            ),
            pytest.param(
                returns(1.0, modify=1.0),
                returns(1.0, modify=2.0),
                "argument 0",
                id="inplace",
            ),
        ],
    )
    def test_fails(self, impl, oracle, match):
        x = np.zeros(2)
        with pytest.raises(AssertionError, match=match):
            assert_same_outcome(outcome(impl, x), outcome(oracle, x))

    def test_compare_copies_arguments(self):
        x = np.zeros(2)
        compare(returns(1.0, modify=3.0), returns(1.0, modify=3.0), x)
        assert x[0] == 0.0

    def test_wrong_kind(self):
        with pytest.raises(AssertionError):
            assert_outcome(returns(1.0), returns(1.0), (np.zeros(1),), "nan")


class TestErrorsMatch:
    """
    Exceptions match if they have the same type and message; an oracle
    ``ZeroDivisionError`` matches the library's error for a float division by zero.
    """

    def test_same_library_error(self):
        error = library_error(errors.INVALID_HYPERGEO)
        assert errors_match(library_error(errors.INVALID_HYPERGEO), error)

    def test_library_error_code_differs(self):
        error = library_error(errors.INVALID_HYPERGEO)
        assert not errors_match(library_error(errors.ZERO_DIVISION), error)

    def test_type_differs(self):
        message = _tsdate.strerror(errors.INVALID_MAX_SHAPE)
        assert not errors_match(_tsdate.LibraryError(message), ValueError(message))
        assert not errors_match(ValueError(message), _tsdate.LibraryError(message))

    def test_value_error_message_differs(self):
        assert not errors_match(ValueError("b"), ValueError("a"))

    def test_zero_division(self):
        actual = library_error(errors.ZERO_DIVISION)
        assert errors_match(actual, ZeroDivisionError("division by zero"))

    @pytest.mark.parametrize(
        "actual",
        [
            library_error(errors.INVALID_HYPERGEO),
            ValueError(_tsdate.strerror(errors.ZERO_DIVISION)),
            ZeroDivisionError("division by zero"),
        ],
        ids=repr,
    )
    def test_zero_division_rejects(self, actual):
        assert not errors_match(actual, ZeroDivisionError("division by zero"))


class TestErrors:
    """
    The error codes of ``oracle.errors`` are those of the library.
    """

    def test_every_code_has_a_distinct_message(self):
        messages = {_tsdate.strerror(code) for code in errors.CODES}
        assert len(messages) == len(errors.CODES)
        assert "Unknown error" not in messages

    def test_no_further_codes(self):
        code = errors.CODES[0] - 1
        assert _tsdate.strerror(code) == "Unknown error"

    def test_kind_is_the_code(self):
        result = outcome(raises(library_error(errors.NO_POSTERIORS)), np.zeros(1))
        assert kind(result) == errors.NO_POSTERIORS
