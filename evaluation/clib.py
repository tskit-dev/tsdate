"""
The scalar functions of the C library (``lib/``), called with ctypes from the shared
library that meson builds (``cd lib && meson setup build && ninja -C build``), to
measure their accuracy by hand. They are not part of the ``_tsdate`` module, and the
signatures below must match ``lib/tsdate.h``: a mismatch is not detected and can
crash the interpreter.

Each function is named as in C without the ``tsd_`` prefix and takes numbers: a
two-element array argument (natural parameters) is given as two numbers. It returns
its outputs as a float, or a tuple of floats with two-element outputs flattened, and
raises ``LibraryError`` for an error code.
"""

import ctypes
import pathlib

BUILD = pathlib.Path(__file__).parents[1] / "lib" / "build"

# Arguments: "d" a double, "p" a two-element array of doubles; outputs: "o" a double,
# "O" a two-element array. Results: "error" (an error code; the outputs are the
# result), "double" or "flag" (the return value is the result; no outputs).
SIGNATURES = {
    "digamma": ("d", "o", "error"),
    "trigamma": ("d", "o", "error"),
    "betaln": ("dd", "", "double"),
    "gammainc_der": ("dd", "o", "error"),
    "gammaincinv": ("dd", "", "double"),
    "hyperu_laplace": ("ddd", "oo", "error"),
    "hyp1f1_laplace": ("ddd", "o", "error"),
    "hyp2f1_laplace": ("dddd", "o", "error"),
    "approximate_log_moments": ("dd", "ooo", "error"),
    "approximate_gamma_kl": ("dd", "O", "error"),
    "approximate_gamma_mom": ("dd", "O", "error"),
    "approximate_gamma_iqr": ("ddddd", "O", "error"),
    "valid_moments": ("dd", "", "flag"),
    "valid_gamma": ("dd", "", "flag"),
    "valid_hyp1f1": ("ddd", "", "flag"),
    "valid_hyperu": ("ddd", "", "flag"),
    "valid_hyp2f1": ("dddd", "", "flag"),
    "moments": ("dddddd", "ooooo", "error"),
    "rootward_moments": ("ddddd", "ooo", "error"),
    "leafward_moments": ("ddddd", "ooo", "error"),
    "unphased_moments": ("dddddd", "ooooo", "error"),
    "twin_moments": ("dddd", "ooo", "error"),
    "sideways_moments": ("ddddd", "ooo", "error"),
    "gamma_projection": ("ppp", "oOO", "error"),
    "leafward_projection": ("dpp", "oO", "error"),
    "rootward_projection": ("dpp", "oO", "error"),
    "unphased_projection": ("ppp", "oOO", "error"),
    "twin_projection": ("pp", "oO", "error"),
    "sideways_projection": ("dpp", "oO", "error"),
    "mutation_moments": ("dddddd", "oo", "error"),
    "mutation_rootward_moments": ("ddddd", "oo", "error"),
    "mutation_leafward_moments": ("ddddd", "oo", "error"),
    "mutation_unphased_moments": ("dddddd", "ooo", "error"),
    "mutation_twin_moments": ("dddd", "ooo", "error"),
    "mutation_sideways_moments": ("ddddd", "ooo", "error"),
    "mutation_edge_moments": ("dd", "oo", "error"),
    "mutation_block_moments": ("dd", "ooo", "error"),
    "mutation_gamma_projection": ("ppp", "oO", "error"),
    "mutation_leafward_projection": ("dpp", "oO", "error"),
    "mutation_rootward_projection": ("dpp", "oO", "error"),
    "mutation_edge_projection": ("dd", "oO", "error"),
    "mutation_unphased_projection": ("ppp", "oO", "error"),
    "mutation_twin_projection": ("pp", "oO", "error"),
    "mutation_sideways_projection": ("dpp", "oO", "error"),
    "mutation_block_projection": ("dd", "oO", "error"),
    "damp": ("ppd", "o", "error"),
    "rescale": ("pd", "o", "error"),
}

PAIR = ctypes.c_double * 2
CTYPES = {
    "d": ctypes.c_double,
    "p": PAIR,
    "o": ctypes.POINTER(ctypes.c_double),
    "O": PAIR,
}
RESTYPES = {"error": ctypes.c_int, "double": ctypes.c_double, "flag": ctypes.c_int}


class LibraryError(Exception):
    pass


def _load():
    for name in ["libtsdate.so", "libtsdate.dylib"]:
        path = BUILD / name
        if path.exists():
            return ctypes.CDLL(str(path))
    raise FileNotFoundError(f"No shared library in {BUILD}: build it with meson")


_lib = _load()
_lib.tsd_strerror.argtypes = [ctypes.c_int]
_lib.tsd_strerror.restype = ctypes.c_char_p


def _function(name, inputs, outputs, result):
    func = getattr(_lib, "tsd_" + name)
    func.argtypes = [CTYPES[kind] for kind in inputs + outputs]
    func.restype = RESTYPES[result]
    num_numbers = inputs.count("d") + 2 * inputs.count("p")

    def call(*numbers):
        if len(numbers) != num_numbers:
            raise TypeError(f"{name} takes {num_numbers} numbers")
        numbers = [float(x) for x in numbers]
        args = []
        for kind in inputs:
            if kind == "d":
                args.append(numbers.pop(0))
            else:
                args.append(PAIR(numbers.pop(0), numbers.pop(0)))
        outs = [ctypes.c_double() if kind == "o" else PAIR() for kind in outputs]
        refs = [ctypes.byref(x) if kind == "o" else x for x, kind in zip(outs, outputs)]
        ret = func(*args, *refs)
        if result != "error":
            return float(ret)
        if ret != 0:
            raise LibraryError(_lib.tsd_strerror(ret).decode())
        values = []
        for x, kind in zip(outs, outputs):
            values.extend([x.value] if kind == "o" else list(x))
        return values[0] if len(values) == 1 else tuple(values)

    call.__name__ = name
    return call


for _name, _signature in SIGNATURES.items():
    globals()[_name] = _function(_name, *_signature)
