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

    $ python -m pip install tsdate

Once installed, `tsdate`'s {ref}`Command Line Interface (CLI) <sec_cli>` can be accessed via:

    $ python -m tsdate

or

    $ tsdate

Alternatively, the {ref}`Python API <sec_python_api>` allows more fine-grained control
of the inference process.

(sec_installation_testing)=

## Development

See the [tskit developer documentation](https://tskit.dev/tskit/docs/stable/development.html)
for the general development workflow (git, prek, testing, documentation).
Install development dependencies with `uv sync` and run the tests with:

    $ uv run pytest

_Tsdate_ makes extensive use of [numba](https://numba.pydata.org)'s
"just in time" (jit) compilation to speed up time-consuming numerical functions.
Because of the need to compile these functions, loading the tsdate package can take
tens of seconds. To speed up loading time, you can set the environment variable

    TSDATE_ENABLE_NUMBA_CACHE=1

The compiled code is not cached by default as it can be problematic when
e.g. running the same installation on different CPU types in a cluster,
and can occassionally lead to unexpected crashes.
