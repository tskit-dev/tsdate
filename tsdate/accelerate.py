import os
from typing import Callable

from numba import jit

# By default we disable the numba cache. See
_DISABLE_CACHE = os.environ.get("TSDATE_DISABLE_NUMBA_CACHE", "1")

try:
    CACHE_NUMBA = {"0": True, "1": False}[_DISABLE_CACHE]
except KeyError as e:  # pragma: no cover
    raise KeyError(
        "Environment variable 'TSDATE_DISABLE_NUMBA_CACHE' must be '0' or '1'"
    ) from e


DEFAULT_NUMBA_ARGS = {
    "nopython": True,
    "cache": CACHE_NUMBA,
}


def numba_jit(*args, **kwargs) -> Callable:  # pragma: no cover
    kwargs_ = DEFAULT_NUMBA_ARGS.copy()
    kwargs_.update(kwargs)
    return jit(*args, **kwargs_)
