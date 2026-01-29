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

(sec_citation)=

# Citation

The default `variational_gamma` method is described in a
[preprint](https://www.biorxiv.org/content/10.64898/2026.01.07.698223v1), and
should be cited by published work using tsdate versions 0.2 or higher:

> Nathaniel S. Pope, Sam Tallman, Ben Jeffery, Duncan Robertson, Yan Wong, Savita Karthikeyan, Peter L. Ralph, and Jerome Kelleher (2026) _Tracing the evolutionary histories of ultra-rare variants using variational dating of large ancestral recombination graphs_. bioRxiv: 2026.01.07.698223; doi: https://doi.org/10.64898/2026.01.07.698223

The algorithm for the `inside_outside` and `maximization` methods (the defaults
prior to version 0.2) is described
in [our Science paper](https://www.science.org/doi/10.1126/science.abi8264) (citation below,
preprint [here](https://www.biorxiv.org/content/10.1101/2021.02.16.431497v2)).
[Another repository](https://github.com/awohns/unified_genealogy_paper) provides
code to reproduce evaluations of the accuracy and computational requirements of these methods.

> Anthony Wilder Wohns, Yan Wong, Ben Jeffery, Ali Akbari, Swapan Mallick, Ron Pinhasi, Nick Patterson, David Reich, Jerome Kelleher, and Gil McVean (2022) *A unified genealogy of modern and ancient genomes*. Science **375**: eabi8264; doi: https://doi.org/10.1126/science.abi8264
