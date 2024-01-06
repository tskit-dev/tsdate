# tsdate <img align="right" width="145" height="90" src="https://github.com/tskit-dev/tsdate/blob/main/docs/tsdate_logo.svg">

[![CircleCI](https://circleci.com/gh/tskit-dev/tsdate.svg?style=svg)](https://circleci.com/gh/tskit-dev/tsdate)
[![codecov](https://codecov.io/gh/tskit-dev/tsdate/branch/master/graph/badge.svg)](https://codecov.io/gh/tskit-dev/tsdate)

``tsdate`` is a scalable method for estimating the age of ancestral nodes in a 
[tree sequence](https://tskit.dev/tutorials/what_is.html). The method uses a coalescent prior and updates node times on the basis of the number of mutations along each edge of the tree sequence (i.e. using the "molecular clock").

The method is frequently combined with the [tsinfer](https://tsinfer.readthedocs.io/en/latest/) algorithm, which efficiently infers tree sequence *topologies* from large genetic datasets.

Please refer to the [documentation](https://tskit.dev/tsdate/docs/latest/) for information on installing and using the software.

The discrete-time estimation methods are described [in this Science paper](https://www.science.org/doi/10.1126/science.abi8264) (preprint [here](https://www.biorxiv.org/content/10.1101/2021.02.16.431497v2)). We also provide evaluations of the accuracy and computational requirements of the method using both simulated and real data. The code to reproduce these results can be found in [another repository](https://github.com/awohns/unified_genealogy_paper).

> Anthony Wilder Wohns, Yan Wong, Ben Jeffery, Ali Akbari, Swapan Mallick, Ron Pinhasi, Nick Patterson, David Reich, Jerome Kelleher, and Gil McVean (2022) _A unified genealogy of modern and ancient genomes_. Science **375**: eabi8264; doi: https://doi.org/10.1126/science.abi8264

Please cite this paper if you use ``tsdate`` in published work.
