---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.12
    jupytext_version: 1.9.1
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

:::{currentmodule} tsdate
:::

(sec_installation)=

# Installation

To install `tsdate` simply run:

    $ python3 -m pip install tsdate --user

Python 3.8, or a more recent version, is required. The software has been tested
on MacOSX and Linux.

Once installed, `tsdate`'s {ref}`Command Line Interface (CLI) <sec_cli>` can be accessed via:

    $ python3 -m tsdate

or

    $ tsdate

Alternatively, the {ref}`Python API <sec_python_api>` allows more fine-grained control
of the inference process.

(sec_installation_testing)=

## Testing

Unit tests can be run from a clone of the
[Github repository](https://github.com/tskit-dev/tsdate) by running pytest
at the top level of the repository

    $python -m pytest

_Tsdate_ makes extensive use of [numba](https://numba.pydata.org)'s
"just in time" (jit) compilation to speed up time-consuming numerical functions.
Because of the need to compile these functions, loading the tsdate package can take
tens of seconds. To speed up loading time, you can set the environment variable

    TSDATE_ENABLE_NUMBA_CACHE=1

The compiled code is not cached by default as it can be problematic when
e.g. running the same installation on different CPU types in a cluster,
and can occassionally lead to unexpected crashes.