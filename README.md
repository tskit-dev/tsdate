# tsdate <img align="right" width="145" height="90" src="https://github.com/tskit-dev/tsdate/blob/main/docs/tsdate_logo.svg">

[![License](https://img.shields.io/github/license/tskit-dev/tsdate)](https://github.com/tskit-dev/tsdate/blob/main/LICENSE) [![PyPI version](https://img.shields.io/pypi/v/tsdate.svg)](https://pypi.org/project/tsdate/) [![Supported Python Versions](https://img.shields.io/pypi/pyversions/tsdate.svg)](https://pypi.org/project/tsdate/) [![Docs Build](https://github.com/tskit-dev/tsdate/actions/workflows/docs.yml/badge.svg)](https://github.com/tskit-dev/tsdate/actions/workflows/docs.yml) [![Binary wheels](https://github.com/tskit-dev/tsdate/actions/workflows/wheels.yml/badge.svg)](https://github.com/tskit-dev/tsdate/actions/workflows/wheels.yml) [![Tests](https://github.com/tskit-dev/tsdate/actions/workflows/tests.yml/badge.svg)](https://github.com/tskit-dev/tsdate/actions/workflows/tests.yml) [![codecov](https://codecov.io/gh/tskit-dev/tsdate/branch/main/graph/badge.svg)](https://codecov.io/gh/tskit-dev/tsdate)

``tsdate`` is a scalable method for estimating the age of ancestral nodes in a 
[tree sequence](https://tskit.dev/tutorials/what_is.html). The method uses a coalescent prior and updates node times on the basis of the number of mutations along each edge of the tree sequence (i.e. using the "molecular clock").

The method is frequently combined with the [tsinfer](https://tskit.dev/tsinfer/docs/stable/) algorithm, which efficiently infers tree sequence *topologies* from large genetic datasets.

Please refer to the documentation ([stable](https://tskit.dev/tsdate/docs/stable/) • [latest](https://tskit.dev/tsdate/docs/latest/)) for information on installing and using the software.

## Installation

```bash
python -m pip install tsdate
```

If using the `variational_gamma` algorithm (the default from version 0.2 onwards), please cite the [preprint](https://www.biorxiv.org/content/10.64898/2026.01.07.698223v1):

> Nathaniel S. Pope, Sam Tallman, Ben Jeffery, Duncan Robertson, Yan Wong, Savita Karthikeyan, Peter L. Ralph, and Jerome Kelleher (2026) _Tracing the evolutionary histories of ultra-rare variants using variational dating of large ancestral recombination graphs_. bioRxiv: 2026.01.07.698223; doi: https://doi.org/10.64898/2026.01.07.698223

The algorithms for the original `inside_outside` and `maximization` [methods](https://tskit.dev/tsdate/docs/stable/methods.html) (the default prior to version 0.2) are described [in this Science paper](https://www.science.org/doi/10.1126/science.abi8264) (citation below, preprint [here](https://www.biorxiv.org/content/10.1101/2021.02.16.431497v2), evaluations in [another repository](https://github.com/awohns/unified_genealogy_paper)). 

The citation to use for the original tsdate algorithms is:

> Anthony Wilder Wohns, Yan Wong, Ben Jeffery, Ali Akbari, Swapan Mallick, Ron Pinhasi, Nick Patterson, David Reich, Jerome Kelleher, and Gil McVean (2022) _A unified genealogy of modern and ancient genomes_. Science **375**: eabi8264; doi: https://doi.org/10.1126/science.abi8264
