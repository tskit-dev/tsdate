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

(sec_welcome)=

# Welcome to _tsdate_

This is the documentation for _tsdate_, a method for efficiently inferring the
age of ancestors in a genetic genealogy or "[ARG](https://tskit.dev/tutorials/args.html)".

Basic usage is as simple as running the following python command

```{code-cell} ipython3
:tags: [remove-cell]
import tskit
input_ts = tskit.load("data/basic_example.trees")
```

```{code-cell} ipython3
import tsdate
output_ts = tsdate.date(input_ts, mutation_rate=1e-8)
```

The rest of this documentation is organised into the following sections:

```{tableofcontents}
```

## Source code

_Tsdate_ is open source software released under the liberal MIT licence. The code is
freely available on [GitHub](https://github.com/tskit-dev/tsdate).
Bug reports and suggestions for improvements can be made by opening an issue on that repository:
we suggest also checking the [discussions list](https://github.com/tskit-dev/tsdate/discussions).
Pull requests are welcome: we largely follow the
[tskit development workflow](https://tskit.dev/tskit/docs/latest/development.html#workflow).

## Citing

The algorithm for the `inside_outside` and `maximization` methods is described 
in [our Science paper](https://www.science.org/doi/10.1126/science.abi8264) (citation below,
preprint [here](https://www.biorxiv.org/content/10.1101/2021.02.16.431497v2)).
[Another repository](https://github.com/awohns/unified_genealogy_paper) provides
code to reproduce evaluations of the accuracy and computational requirements of these methods.
The default `variational_gamma` method has not yet been described in print. For the moment,
please cite this github repository if you need a citable reference.

The original _tsdate_ algorithm, which you should cite in published work, is published in:

> Anthony Wilder Wohns, Yan Wong, Ben Jeffery, Ali Akbari, Swapan Mallick, Ron Pinhasi, Nick Patterson, David Reich, Jerome Kelleher, and Gil McVean (2022) *A unified genealogy of modern and ancient genomes*. Science **375**: eabi8264; doi: https://doi.org/10.1126/science.abi8264

