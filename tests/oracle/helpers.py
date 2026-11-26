"""
Comparison helpers shared by the oracle equivalence tests.
"""

import numpy as np
import tskit
from _tsdate import LibraryError

from . import errors
from .errors import library_error


def _flatten(x, path=""):
    """
    Yield (path, array) for each leaf of a nested tuple/list of scalars/arrays.
    """
    if isinstance(x, (tuple, list)):
        for i, item in enumerate(x):
            yield from _flatten(item, f"{path}[{i}]")
    else:
        yield path, np.asarray(x, dtype=np.float64)


def assert_same_values(actual, desired):
    """
    Assert that two scalars, arrays, or (nested) tuples of them are equal, with NaNs in
    the same places. Shapes and tuple structure must match.
    """
    a_leaves = list(_flatten(actual))
    d_leaves = list(_flatten(desired))
    assert len(a_leaves) == len(d_leaves), (
        f"structure mismatch: {len(a_leaves)} vs {len(d_leaves)} values"
    )
    for (path, a), (_, d) in zip(a_leaves, d_leaves):
        assert a.shape == d.shape, (
            f"shape mismatch at {path or 'value'}: {a.shape} vs {d.shape}"
        )
        assert np.array_equal(a, d, equal_nan=True), (
            f"values differ at {path or 'value'}:\nactual:  {a}\ndesired: {d}"
        )


INF = float("inf")
NAN = float("nan")


def log_uniform(rng, lo, hi, size=None):
    """
    ``10 ** U(lo, hi)``: a scalar float if ``size`` is None, else an array.
    """
    x = 10.0 ** rng.uniform(lo, hi, size)
    return float(x) if size is None else x


class Outcome:
    """
    What calling a function did: ``value`` (its return value) or ``exception``, and
    ``args``, the (copied) arguments after the call, to compare in-place updates.
    """

    def __init__(self, value=None, exception=None, args=()):
        self.value = value
        self.exception = exception
        self.args = args

    @property
    def error(self):
        """
        ``None`` if the call returned, the error code (an `errors` constant) if it
        raised a ``_tsdate.LibraryError``, else the class of the exception.
        """
        if self.exception is None:
            return None
        if isinstance(self.exception, LibraryError):
            return errors.code_of(self.exception)
        return type(self.exception)

    def __repr__(self):
        if self.exception is None:
            return f"Outcome(value={self.value!r})"
        return f"Outcome(exception={self.exception!r})"


def _copy_arg(x):
    if isinstance(x, np.ndarray):
        return x.copy()
    if isinstance(x, np.floating):
        # the oracles expect Python floats (division by zero raises)
        return float(x)
    return x


def outcome(func, *args, **kwargs):
    """
    Call ``func`` on copies of ``args`` and return an `Outcome`. NumPy floating
    scalars are converted to ``float``; NumPy warnings (which the oracle emits where
    C is silent) are suppressed.
    """
    copies = [_copy_arg(x) for x in args]
    try:
        with np.errstate(all="ignore"):
            value = func(*copies, **kwargs)
    except Exception as e:  # noqa: BLE001
        return Outcome(exception=e, args=copies)
    return Outcome(value=value, args=copies)


def errors_match(actual, expected):
    """
    Whether exception ``actual`` (from an implementation) matches ``expected`` (from
    the oracle): the same type and message. An oracle ``ZeroDivisionError`` stands for
    the library's ``TSD_ERR_ZERO_DIVISION``, and matches the ``_tsdate.LibraryError``
    that reports it.
    """
    if isinstance(expected, ZeroDivisionError):
        expected = library_error(errors.ZERO_DIVISION)
    return type(actual) is type(expected) and str(actual) == str(expected)


def assert_same_outcome(actual, expected):
    """
    Assert that two `Outcome` objects agree: both raise matching exceptions, or both
    return the same values and leave array arguments the same.
    """
    if expected.exception is not None:
        assert actual.exception is not None, (
            f"implementation returned {actual.value!r}, oracle raised "
            f"{expected.exception!r}"
        )
        assert errors_match(actual.exception, expected.exception), (
            f"implementation raised {actual.exception!r}, oracle raised "
            f"{expected.exception!r}"
        )
        return
    assert actual.exception is None, (
        f"implementation raised {actual.exception!r}, oracle returned {expected.value!r}"
    )
    assert_same_values(actual.value, expected.value)
    for i, (a, d) in enumerate(zip(actual.args, expected.args)):
        if isinstance(a, np.ndarray):
            try:
                assert_same_values(a, d)
            except AssertionError as e:
                raise AssertionError(
                    f"argument {i} differs after the call: {e}"
                ) from None


def compare(impl_func, oracle_func, *args, **kwargs):
    """
    Call the implementation and the oracle on copies of ``args``, assert that their
    outcomes agree (`assert_same_outcome`), and return the oracle's `Outcome`.
    """
    expected = outcome(oracle_func, *args, **kwargs)
    actual = outcome(impl_func, *args, **kwargs)
    assert_same_outcome(actual, expected)
    return expected


def kind(result):
    """
    Classify an `Outcome`: its `Outcome.error` if it raised, ``None`` if it returned
    ``None`` (an in-place update), else ``"nan"`` if any returned value is NaN,
    ``"finite"`` if all are finite, otherwise ``"nonfinite"``.
    """
    if result.exception is not None:
        return result.error
    if result.value is None:
        return None
    values = [leaf for _, leaf in _flatten(result.value)]
    values = np.concatenate([v.ravel() for v in values]) if values else []
    if np.isnan(values).any():
        return "nan"
    return "finite" if np.isfinite(values).all() else "nonfinite"


def assert_outcome(impl_func, oracle_func, args, expected_kind, **kwargs):
    """
    `compare`, then assert the `kind` of the outcome (for edge-case tables).
    """
    result = compare(impl_func, oracle_func, *args, **kwargs)
    assert kind(result) == expected_kind, f"{args}: {result!r}"
    return result


def make_ts(nodes, edges, mutations, length, num_individuals=0):
    """
    Tree sequence from ``nodes`` ``(is_sample, time)`` or ``(is_sample, time,
    individual)``, ``edges`` ``(left, right, parent, child)`` and ``mutations``
    ``(position, node)``, with ``num_individuals`` empty individuals.
    """
    tables = tskit.TableCollection(length)
    for _ in range(num_individuals):
        tables.individuals.add_row()
    for is_sample, time, *individual in nodes:
        tables.nodes.add_row(
            flags=int(is_sample),
            time=time,
            individual=individual[0] if individual else -1,
        )
    for left, right, parent, child in edges:
        tables.edges.add_row(left, right, parent, child)
    for i, (position, node) in enumerate(mutations):
        tables.sites.add_row(position, "0")
        tables.mutations.add_row(i, node, "1")
    tables.sort()
    return tables.tree_sequence()


def i32(*x):
    return np.array(x, dtype=np.int32)


def f64(*x):
    return np.array(x, dtype=np.float64)


def b1(*x):
    return np.array(x, dtype=bool)


def shrink(args, index, size):
    """
    Copy of the argument list with ``args[index]`` truncated to ``size``.
    """
    args = list(args)
    args[index] = args[index][:size]
    return args
