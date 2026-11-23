import os
import platform

import numpy
from setuptools import Extension, setup

IS_WINDOWS = platform.system() == "Windows"

libdir = "lib"
includes = [libdir]

tsd_source_files = [
    "err.c",
    "hypergeo.c",
    "approx.c",
    "rescaling.c",
    "phasing.c",
    "util.c",
    "prior.c",
]

sources = ["_tsdatemodule.c"] + [os.path.join(libdir, f) for f in tsd_source_files]

# Unlike tsinfer we do not undefine NDEBUG: assert() is for expensive checks and
# is compiled out of the Python module; tsd_bug_assert is always on.
# -ffp-contract=off stops clang fusing multiply-adds, which would change results.
extra_compile_args = [] if IS_WINDOWS else ["-std=c11", "-ffp-contract=off"]

_tsdate_module = Extension(
    "_tsdate",
    sources=sources,
    extra_compile_args=extra_compile_args,
    include_dirs=includes + [numpy.get_include()],
)

setup(
    ext_modules=[_tsdate_module],
)
