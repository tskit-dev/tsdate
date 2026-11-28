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

Binary wheels are provided for the common platforms. Building from source (e.g.
from the sdist, or a development checkout) requires a C compiler, as `tsdate`
includes a C extension module (`_tsdate`).

Alternatively, the {ref}`Python API <sec_python_api>` allows more fine-grained control
of the inference process.

(sec_installation_testing)=

## Development

See the [tskit developer documentation](https://tskit.dev/tskit/docs/stable/development.html)
for the general development workflow (git, prek, testing, documentation).
Install development dependencies with `uv sync` and run the tests with:

    $ uv run pytest

`uv sync` also (re)builds the C extension. After editing C code, run `uv sync`
(or `make`) again to rebuild it.
