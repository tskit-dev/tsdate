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

# Welcome to tsdate

This is the documentation for `tsdate`, a method for efficiently inferring the
age of ancestors in a genetic genealogy or "[ARG](https://tskit.dev/tutorials/args.html)".

Basic usage is as simple as running the following python command

```{code-cell} ipython3
:tags: [remove-cell]
import tskit
input_ts = tskit.load("data/basic_example.trees")
```

```{code-cell} ipython3
import tsdate
output_ts = tsdate.date(input_ts, population_size=100, mutation_rate=1e-8)
```

The rest of this documentation is organised into the following sections:

```{tableofcontents}
```

## Citing

The algorithm is described in [our Science paper](https://www.science.org/doi/10.1126/science.abi8264)
(citation below, preprint [here](https://www.biorxiv.org/content/10.1101/2021.02.16.431497v2)). We also provide
evaluations of the accuracy and computational requirements of the method using both simulated and real
data; the code to reproduce these results can be found in
[another repository](https://github.com/awohns/unified_genealogy_paper).

Please cite the following paper if you use `tsdate` in published work:

> Anthony Wilder Wohns, Yan Wong, Ben Jeffery, Ali Akbari, Swapan Mallick, Ron Pinhasi, Nick Patterson, David Reich, Jerome Kelleher, and Gil McVean (2022) *A unified genealogy of modern and ancient genomes*. Science **375**: eabi8264; doi: https://doi.org/10.1126/science.abi8264

