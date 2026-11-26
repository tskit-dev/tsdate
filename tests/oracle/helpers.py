"""Comparison helpers shared by the oracle equivalence tests."""

import numpy as np
import tskit

try:
    from _tsdate import LibraryError
except ImportError:  # the extension has not been built
    LibraryError = None


def _flatten(x, path=""):
    """Yield (path, array) for each leaf of a nested tuple/list of scalars/arrays."""
    if isinstance(x, (tuple, list)):
        for i, item in enumerate(x):
            yield from _flatten(item, f"{path}[{i}]")
    else:
        yield path, np.asarray(x, dtype=np.float64)


def assert_ulp_close(actual, desired, maxulp=4):
    """
    Assert that two scalars, arrays, or (nested) tuples of them agree to within
    ``maxulp`` units in the last place. NaNs must occur in the same places, and
    equal infinities compare equal. Shapes and tuple structure must match.
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
        nan_a, nan_d = np.isnan(a), np.isnan(d)
        assert np.array_equal(nan_a, nan_d), (
            f"NaN pattern mismatch at {path or 'value'}:\n{a}\nvs\n{d}"
        )
        inf_mask = np.isinf(a) | np.isinf(d)
        assert np.array_equal(a[inf_mask], d[inf_mask]), (
            f"infinity mismatch at {path or 'value'}:\n{a}\nvs\n{d}"
        )
        ok = ~(nan_a | inf_mask)
        if ok.any():
            try:
                np.testing.assert_array_max_ulp(a[ok], d[ok], maxulp=maxulp)
            except AssertionError as e:
                raise AssertionError(
                    f"values differ by more than {maxulp} ulp at {path or 'value'}:"
                    f"\nactual:  {a}\ndesired: {d}\n{e}"
                ) from None


def assert_inplace_close(args_actual, args_desired, maxulp=4):
    """
    Compare sequences of arguments after both implementations have modified them
    in place (e.g. posterior arrays, ``EPFactors`` fields). Each element is
    compared with `assert_ulp_close`; use ``(a.node, a.edge, ...)`` to pass fields.
    """
    assert len(args_actual) == len(args_desired)
    for i, (a, d) in enumerate(zip(args_actual, args_desired)):
        try:
            assert_ulp_close(a, d, maxulp=maxulp)
        except AssertionError as e:
            raise AssertionError(
                f"argument {i} differs after in-place update: {e}"
            ) from None


INF = float("inf")
NAN = float("nan")


def log_uniform(rng, lo, hi, size=None):
    """``10 ** U(lo, hi)``: a scalar float if ``size`` is None, else an array."""
    x = 10.0 ** rng.uniform(lo, hi, size)
    return float(x) if size is None else x


# Exceptions of the oracle that the C library reports as a LibraryError.
_LIBRARY_ERROR_NAMES = (
    "ZeroDivisionError",
    "AssertionError",
    "KLMinimizationFailedError",
    "ValueError",
)


def expected_error(exc_type):
    """
    The exceptions an implementation may raise where the oracle raises ``exc_type``
    (``ZeroDivisionError``, ``AssertionError``, ...), for ``pytest.raises``. The C
    library reports these as ``_tsdate.LibraryError``, and its argument checks raise
    ``ValueError`` where the oracle asserts on array shapes and sizes.
    """
    allowed = (exc_type,)
    if LibraryError is not None and exc_type.__name__ in _LIBRARY_ERROR_NAMES:
        allowed += (LibraryError,)
    if exc_type is AssertionError:
        allowed += (ValueError,)
    return allowed


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
        return None if self.exception is None else type(self.exception)

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


def errors_match(actual, expected, check_message=True):
    """
    Whether exception ``actual`` (from an implementation) matches ``expected`` (from
    the oracle): the same type and, if ``check_message``, the same message (never for
    a ``ZeroDivisionError``). A ``_tsdate.LibraryError`` (or, for an
    ``AssertionError``, a ``ValueError`` from the module's shape checks) is accepted
    in place of the exceptions of `expected_error`; its message is compared only with
    those of the tsdate-authored, user-facing errors (``KLMinimizationFailedError``
    and ``ValueError``), as the messages of the others (assertions, divisions) are not
    part of the interface.
    """
    if not isinstance(actual, expected_error(type(expected))):
        return False
    if isinstance(expected, ZeroDivisionError):
        return True
    if isinstance(expected, AssertionError) and not isinstance(actual, AssertionError):
        return True
    return not check_message or str(actual) == str(expected)


def assert_same_outcome(actual, expected, maxulp=4, check_message=True):
    """
    Assert that two `Outcome` objects agree: both raise matching exceptions, or both
    return values (and leave array arguments) within ``maxulp`` ulp.
    """
    if expected.exception is not None:
        assert actual.exception is not None, (
            f"implementation returned {actual.value!r}, oracle raised "
            f"{expected.exception!r}"
        )
        assert errors_match(actual.exception, expected.exception, check_message), (
            f"implementation raised {actual.exception!r}, oracle raised "
            f"{expected.exception!r}"
        )
        return
    assert actual.exception is None, (
        f"implementation raised {actual.exception!r}, oracle returned {expected.value!r}"
    )
    assert_ulp_close(actual.value, expected.value, maxulp=maxulp)
    for i, (a, d) in enumerate(zip(actual.args, expected.args)):
        if isinstance(a, np.ndarray):
            try:
                assert_ulp_close(a, d, maxulp=maxulp)
            except AssertionError as e:
                raise AssertionError(
                    f"argument {i} differs after the call: {e}"
                ) from None


def compare(impl_func, oracle_func, *args, maxulp=4, check_message=True, **kwargs):
    """
    Call the implementation and the oracle on copies of ``args``, assert that their
    outcomes agree (`assert_same_outcome`), and return the oracle's `Outcome`.
    """
    expected = outcome(oracle_func, *args, **kwargs)
    actual = outcome(impl_func, *args, **kwargs)
    assert_same_outcome(actual, expected, maxulp=maxulp, check_message=check_message)
    return expected


def kind(result):
    """
    Classify an `Outcome`: the exception class if it raised, ``None`` if it returned
    ``None`` (an in-place update), else ``"nan"`` if any returned value is NaN,
    ``"finite"`` if all are finite, otherwise ``"nonfinite"``.
    """
    if result.exception is not None:
        return type(result.exception)
    if result.value is None:
        return None
    values = [leaf for _, leaf in _flatten(result.value)]
    values = np.concatenate([v.ravel() for v in values]) if values else []
    if np.isnan(values).any():
        return "nan"
    return "finite" if np.isfinite(values).all() else "nonfinite"


def assert_outcome(impl_func, oracle_func, args, expected_kind, **kwargs):
    """`compare`, then assert the `kind` of the outcome (for edge-case tables)."""
    result = compare(impl_func, oracle_func, *args, **kwargs)
    assert kind(result) == expected_kind, f"{args}: {result!r}"
    return result


def both(impl_func, oracle_func, *args, maxulp=4):
    """
    `compare`, returning the oracle's value; if both raise, re-raise the oracle's
    exception (for use with ``pytest.raises``).
    """
    result = compare(impl_func, oracle_func, *args, maxulp=maxulp)
    if result.exception is not None:
        raise result.exception
    return result.value


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
    """Copy of the argument list with ``args[index]`` truncated to ``size``."""
    args = list(args)
    args[index] = args[index][:size]
    return args
