"""Tests of the comparison helpers in ``tests/oracle/helpers.py``."""

import numpy as np
import pytest
from oracle.helpers import (
    INF,
    NAN,
    Outcome,
    assert_inplace_close,
    assert_outcome,
    assert_same_outcome,
    assert_ulp_close,
    both,
    call,
    compare,
    errors_match,
    expected_error,
    kind,
    outcome,
)


def one_ulp_up(x, n=1):
    for _ in range(n):
        x = np.nextafter(x, np.inf)
    return x


class TestAssertUlpClose:
    @pytest.mark.parametrize(
        ("actual", "desired"),
        [
            pytest.param(1.5, 1.5, id="scalar"),
            pytest.param(1.0, one_ulp_up(1.0), id="one_ulp"),
            pytest.param(one_ulp_up(1.0, 4), 1.0, id="four_ulp"),
            pytest.param(np.array([1.0, NAN]), np.array([1.0, NAN]), id="nan"),
            pytest.param(INF, INF, id="inf"),
            pytest.param(-INF, -INF, id="minus_inf"),
            pytest.param(
                (1.0, np.arange(2.0), (np.zeros(1), NAN)),
                (1.0, np.arange(2.0), (np.zeros(1), NAN)),
                id="nested",
            ),
        ],
    )
    def test_passes(self, actual, desired):
        assert_ulp_close(actual, desired)

    @pytest.mark.parametrize(
        ("actual", "desired", "match"),
        [
            pytest.param(one_ulp_up(1.0, 5), 1.0, "ulp", id="five_ulp"),
            pytest.param(
                np.array([1.0, NAN]), np.array([NAN, 1.0]), "NaN", id="nan_place"
            ),
            pytest.param(1.0, NAN, "NaN", id="nan_vs_number"),
            pytest.param(INF, -INF, "infinity", id="inf_sign"),
            pytest.param(INF, 1e308, "infinity", id="inf_vs_huge"),
            pytest.param((1.0, 2.0), (1.0, 2.0, 3.0), "structure", id="structure"),
            pytest.param(np.zeros(3), np.zeros(4), "shape", id="shape"),
            pytest.param(np.zeros((2, 2)), np.zeros(4), "shape", id="ndim"),
            pytest.param(
                (1.0, np.array([1.0, 2.0])),
                (1.0, np.array([1.0, 2.5])),
                r"\[1\]",
                id="path",
            ),
        ],
    )
    def test_fails(self, actual, desired, match):
        with pytest.raises(AssertionError, match=match):
            assert_ulp_close(actual, desired)

    def test_maxulp(self):
        x = one_ulp_up(1.0, 10)
        with pytest.raises(AssertionError, match="ulp"):
            assert_ulp_close(x, 1.0, maxulp=4)
        assert_ulp_close(x, 1.0, maxulp=10)


class TestAssertInplaceClose:
    def test_agree(self):
        assert_inplace_close((np.ones(3), 2.0), (np.ones(3), 2.0))

    def test_differ(self):
        b = np.ones(3)
        b[1] = 2.0
        with pytest.raises(AssertionError, match="argument 0"):
            assert_inplace_close((np.ones(3), 2.0), (b, 2.0))


class TestOutcome:
    def test_return_value(self):
        result = outcome(lambda x: x + 1, 1.0)
        assert result.value == 2.0
        assert result.exception is None
        assert result.error is None

    def test_exception(self):
        def fail():
            raise ValueError("bad")

        result = outcome(fail)
        assert result.value is None
        assert result.error is ValueError
        assert str(result.exception) == "bad"

    def test_arrays_are_copied(self):
        def modify(x):
            x[0] = 5.0

        x = np.zeros(2)
        result = outcome(modify, x)
        assert x[0] == 0.0
        assert result.args[0][0] == 5.0

    def test_numpy_scalar_becomes_float(self):
        result = outcome(lambda x: type(x), np.float64(1.0))
        assert result.value is float

    def test_warnings_are_suppressed(self):
        with np.errstate(all="raise"):
            result = outcome(lambda: np.float64(1.0) / np.zeros(1))
        assert result.exception is None

    def test_keyword_arguments(self):
        assert outcome(lambda x, y=0: x + y, 1, y=2).value == 3


class TestCall:
    def test_plain_function(self):
        assert call(lambda x: x + 1, 1) == 2

    def test_ctypes_interface(self):
        class CFunc:
            ctypes = staticmethod(lambda x: x * 2)

        assert call(CFunc(), 2) == 4


class TestErrorsMatch:
    def test_same_type_and_message(self):
        assert errors_match(ValueError("a"), ValueError("a"))

    def test_subclass_accepted(self):
        assert errors_match(FloatingPointError("a"), ArithmeticError("a"))

    def test_type_mismatch(self):
        assert not errors_match(KeyError("a"), ValueError("a"))

    def test_message_mismatch(self):
        assert not errors_match(ValueError("a"), ValueError("b"))
        assert errors_match(ValueError("a"), ValueError("b"), check_message=False)

    def test_zero_division_message_ignored(self):
        actual = ZeroDivisionError("division by zero")
        assert errors_match(actual, ZeroDivisionError("float division by zero"))

    def test_expected_error(self):
        assert ValueError in expected_error(ValueError)


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


class TestAssertSameOutcome:
    def test_agree(self):
        x = np.zeros(2)
        a = outcome(returns(np.ones(2)), x)
        b = outcome(returns(np.ones(2)), x)
        assert_same_outcome(a, b)

    def test_value_mismatch(self):
        x = np.zeros(2)
        a = outcome(returns(np.ones(2)), x)
        b = outcome(returns(np.full(2, 2.0)), x)
        with pytest.raises(AssertionError, match="ulp"):
            assert_same_outcome(a, b)

    def test_implementation_raises(self):
        x = np.zeros(2)
        a = outcome(raises(ValueError("a")), x)
        b = outcome(returns(1.0), x)
        with pytest.raises(AssertionError, match="implementation raised"):
            assert_same_outcome(a, b)

    def test_oracle_raises(self):
        x = np.zeros(2)
        a = outcome(returns(1.0), x)
        b = outcome(raises(ValueError("a")), x)
        with pytest.raises(AssertionError, match="oracle raised"):
            assert_same_outcome(a, b)

    def test_exceptions_agree(self):
        x = np.zeros(2)
        a = outcome(raises(ValueError("a")), x)
        b = outcome(raises(ValueError("a")), x)
        assert_same_outcome(a, b)

    def test_exception_type_mismatch(self):
        x = np.zeros(2)
        a = outcome(raises(KeyError("a")), x)
        b = outcome(raises(ValueError("a")), x)
        with pytest.raises(AssertionError, match="implementation raised"):
            assert_same_outcome(a, b)

    def test_exception_message_mismatch(self):
        x = np.zeros(2)
        a = outcome(raises(ValueError("a")), x)
        b = outcome(raises(ValueError("b")), x)
        with pytest.raises(AssertionError, match="implementation raised"):
            assert_same_outcome(a, b)
        assert_same_outcome(a, b, check_message=False)

    def test_inplace_mismatch(self):
        x = np.zeros(2)
        a = outcome(returns(1.0, modify=1.0), x)
        b = outcome(returns(1.0, modify=2.0), x)
        with pytest.raises(AssertionError, match="argument 0"):
            assert_same_outcome(a, b)

    def test_maxulp(self):
        a = Outcome(value=one_ulp_up(1.0, 6))
        b = Outcome(value=1.0)
        with pytest.raises(AssertionError, match="ulp"):
            assert_same_outcome(a, b)
        assert_same_outcome(a, b, maxulp=6)


class TestCompare:
    def test_returns_oracle_outcome(self):
        result = compare(returns(1.0), returns(1.0), np.zeros(2))
        assert result.value == 1.0

    def test_arguments_are_copied(self):
        x = np.zeros(2)
        compare(returns(1.0, modify=3.0), returns(1.0, modify=3.0), x)
        assert x[0] == 0.0

    def test_returns_oracle_args(self):
        result = compare(returns(1.0, modify=3.0), returns(1.0, modify=3.0), np.zeros(2))
        assert result.args[0][0] == 3.0

    def test_mismatch(self):
        with pytest.raises(AssertionError):
            compare(returns(1.0), returns(2.0), np.zeros(2))

    def test_oracle_raises_implementation_returns(self):
        with pytest.raises(AssertionError, match="oracle raised"):
            compare(returns(1.0), raises(ValueError("a")), np.zeros(2))

    def test_check_message(self):
        impl, oracle = raises(ValueError("a")), raises(ValueError("b"))
        with pytest.raises(AssertionError):
            compare(impl, oracle, np.zeros(2))
        compare(impl, oracle, np.zeros(2), check_message=False)

    def test_keyword_arguments(self):
        compare(lambda x, y=0: x + y, lambda x, y=0: x + y, 1.0, y=2.0)


class TestKind:
    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            pytest.param(1.0, "finite", id="finite"),
            pytest.param(np.zeros(0), "finite", id="empty"),
            pytest.param(NAN, "nan", id="nan"),
            pytest.param(np.array([1.0, NAN]), "nan", id="array_with_nan"),
            pytest.param(INF, "nonfinite", id="inf"),
            pytest.param(np.array([-INF, 1.0]), "nonfinite", id="array_with_inf"),
            pytest.param((1.0, np.array([INF, NAN])), "nan", id="nan_beats_inf"),
            pytest.param((np.ones(2), (1.0, INF)), "nonfinite", id="nested"),
            pytest.param((np.ones((2, 3)), np.ones(4)), "finite", id="mixed_ndim"),
            pytest.param(None, None, id="none"),
        ],
    )
    def test_value(self, value, expected):
        assert kind(Outcome(value=value)) == expected

    def test_exception(self):
        assert kind(Outcome(exception=KeyError("a"))) is KeyError


class TestAssertOutcome:
    def test_expected_kind(self):
        assert_outcome(returns(1.0), returns(1.0), (np.zeros(1),), "finite")

    def test_wrong_kind(self):
        with pytest.raises(AssertionError):
            assert_outcome(returns(1.0), returns(1.0), (np.zeros(1),), "nan")

    def test_expected_exception(self):
        assert_outcome(
            raises(KeyError("a")), raises(KeyError("a")), (np.zeros(1),), KeyError
        )


class TestBoth:
    def test_returns_oracle_value(self):
        assert both(returns(1.0), returns(1.0), np.zeros(1)) == 1.0

    def test_reraises_oracle_exception(self):
        with pytest.raises(ValueError, match="a"):
            both(raises(ValueError("a")), raises(ValueError("a")), np.zeros(1))
