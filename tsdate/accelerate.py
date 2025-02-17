import os
from typing import Callable

from numba import jit

# By default we disable the numba cache. See e.g.
# https://github.com/sgkit-dev/sgkit/blob/main/sgkit/accelerate.py
_ENABLE_CACHE = os.environ.get("TSDATE_ENABLE_NUMBA_CACHE", "0")

try:
    CACHE_NUMBA = {"0": False, "1": True}[_ENABLE_CACHE]
except KeyError as e:  # pragma: no cover
    raise KeyError(
        "Environment variable 'TSDATE_ENABLE_NUMBA_CACHE' must be '0' or '1'"
    ) from e


DEFAULT_NUMBA_ARGS = {
    "nopython": True,
    "cache": CACHE_NUMBA,
}


def numba_jit(*args, **kwargs) -> Callable:  # pragma: no cover
    kwargs_ = DEFAULT_NUMBA_ARGS.copy()
    kwargs_.update(kwargs)
    return jit(*args, **kwargs_)
