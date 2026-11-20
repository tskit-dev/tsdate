"""
C-semantics versions of the `math` functions used by the oracles.

Compiled numba calls libm directly, so e.g. ``log(0.0) == -inf``, ``exp(1e4) == inf``
and ``sqrt(-1.0)`` is NaN. Python's `math` module raises ``ValueError`` or
``OverflowError`` in these cases. These wrappers return the C results instead and
otherwise defer to `math`, so values agree with numba's libm calls. `lgamma` goes
to libm itself (see `_load_libm_lgamma`).
"""

import ctypes
import ctypes.util
import math
import sys

import numpy as np

_inf = math.inf
_nan = math.nan


def log(x):
    try:
        return math.log(x)
    except ValueError:
        return -_inf if x == 0.0 else _nan


def exp(x):
    try:
        return math.exp(x)
    except OverflowError:
        return _inf


def sqrt(x):
    try:
        return math.sqrt(x)
    except ValueError:
        return _nan


def sin(x):
    try:
        return math.sin(x)
    except ValueError:  # sin(+-inf)
        return _nan


def tan(x):
    try:
        return math.tan(x)
    except ValueError:  # tan(+-inf)
        return _nan


def _load_libm_lgamma():
    """
    The platform C runtime's ``lgamma``, as used by compiled numba (and by the C
    library): libm on Linux and macOS, the Universal CRT (``ucrtbase``) on Windows.
    CPython's `math.lgamma` is its own Lanczos implementation, which differs from
    the C runtime by up to thousands of ulp where lgamma is near zero (x near 1 or
    2), so there is deliberately no fallback to it: if the C runtime can't be
    loaded, the import fails rather than the oracle silently drifting.
    """
    name = ctypes.util.find_library("m")
    if name is None and sys.platform == "win32":
        name = "ucrtbase"
    if name is None:
        raise ImportError("could not locate the C runtime library providing lgamma")
    func = ctypes.CDLL(name).lgamma
    func.restype = ctypes.c_double
    func.argtypes = [ctypes.c_double]
    assert math.isclose(func(5.0), math.log(24.0), rel_tol=1e-12)
    return func


_libm_lgamma = _load_libm_lgamma()


def lgamma(x):
    return _libm_lgamma(x)


def erf(x):
    return math.erf(x)


def pow(x, y):  # noqa: A001
    try:
        return math.pow(x, y)
    except OverflowError:
        return _inf
    except ValueError:
        # 0 ** negative is a pole; negative ** non-integer is NaN
        return _inf if x == 0.0 else _nan


def ipow(x, n):
    """
    ``x ** n`` for a small positive integer constant ``n``, by repeated squaring.

    Numba lowers a float raised to an integer constant this way (``llvm.powi``), so
    ``x ** 2 == x * x``. CPython calls libm ``pow`` instead, which differs from
    this by an ulp in a small fraction of cases. Compiled code (and the C port) has
    the repeated-squaring behaviour, and it matters where results cancel
    (e.g. ``va = sq - mn ** 2``).
    """
    result = 1.0
    while True:
        if n & 1:
            result *= x
        n >>= 1
        if n == 0:
            return result
        x *= x


def seq_sum(x):
    """
    Sum of a 1d array, accumulating strictly left to right, as numba (and a C loop)
    does. ``np.sum`` uses pairwise summation, which differs in the last bits.
    """
    x = np.asarray(x)
    return x.dtype.type(0) if x.size == 0 else np.cumsum(x)[-1]


def seq_mean(x):
    """Mean of a 1d array using `seq_sum`, as ``np.mean`` is compiled by numba."""
    return seq_sum(x) / np.size(x)
