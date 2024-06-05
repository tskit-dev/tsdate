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

(sec_cli)=

# Command line interface

This page provides formal documentation for the command-line interface to _tsdate_.

## Example

For a mutation rate of 1e-8 and an effective diploid population size of 20,000, the following command-line
command will create dated tree sequence named `output.trees`.

```{code} bash
tsdate date -m 1e-8 input.trees output.trees 20000
```

Alternatively, the following is more reliable, especially if you have multiple versions of Python installed or
if the {command}`tsdate` executable is not installed on your path:

```{code} bash
python3 -m tsdate date -m 1e-8 input.trees output.trees 20000
```

For long-running dating processes, the `--progress` option may also be useful.
Additional `tsdate` commands, notably {func}`preprocess_ts` are also available, see below:

## Argument details

```{eval-rst}
.. argparse::
    :module: tsdate.cli
    :func: tsdate_cli_parser
    :prog: tsdate
    :nodefault:
```