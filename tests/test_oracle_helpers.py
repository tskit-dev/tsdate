"""
The comparison helpers in ``tests/oracle/helpers.py`` detect disagreements: a helper
that passed everything would make every parity test vacuous.
"""

import _tsdate
import numpy as np
import pytest
from oracle.helpers import (
    INF,
    NAN,
    assert_outcome,
    assert_same_outcome,
    assert_ulp_close,
    compare,
    errors_match,
    outcome,
)

KLFailed = type("KLMinimizationFailedError", (Exception,), {})


def ulps_up(x, n):
    for _ in range(n):
        x = np.nextafter(x, np.inf)
    return x


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


class TestAssertUlpClose:
    @pytest.mark.parametrize(
        ("actual", "desired", "match"),
        [
            pytest.param(ulps_up(1.0, 5), 1.0, "ulp", id="five_ulp"),
            pytest.param(
                np.array([1.0, NAN]), np.array([NAN, 1.0]), "NaN", id="nan_place"
            ),
            pytest.param(1.0, NAN, "NaN", id="nan_vs_number"),
            pytest.param(INF, -INF, "infinity", id="inf_sign"),
            pytest.param((1.0, 2.0), (1.0, 2.0, 3.0), "structure", id="structure"),
            pytest.param(np.zeros(3), np.zeros(4), "shape", id="shape"),
        ],
    )
    def test_fails(self, actual, desired, match):
        with pytest.raises(AssertionError, match=match):
            assert_ulp_close(actual, desired)


class TestAssertSameOutcome:
    @pytest.mark.parametrize(
        ("impl", "oracle", "match"),
        [
            pytest.param(returns(1.0), returns(2.0), "ulp", id="value"),
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


class TestLibraryError:
    """
    The C library reports the oracle's ZeroDivisionError, AssertionError,
    KLMinimizationFailedError and ValueError as ``_tsdate.LibraryError``; messages are
    compared only for the last two.
    """

    @pytest.mark.parametrize("oracle_type", [ValueError, KLFailed])
    def test_messages_compared(self, oracle_type):
        assert errors_match(_tsdate.LibraryError("a"), oracle_type("a"))
        assert not errors_match(_tsdate.LibraryError("b"), oracle_type("a"))

    @pytest.mark.parametrize("oracle_exc", [KeyError("a"), IndexError("a")], ids=type)
    def test_other_oracle_errors_rejected(self, oracle_exc):
        assert not errors_match(_tsdate.LibraryError("a"), oracle_exc)

    def test_value_error_rejected_for_zero_division(self):
        assert not errors_match(ValueError("a"), ZeroDivisionError("a"))
