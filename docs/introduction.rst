.. _sec_introduction:

============
Introduction
============

``tsdate`` is a scalable method for estimating the age of ancestral nodes in a 
`tree sequence <https://tskit.dev/tutorials/what_is.html>`_. The method uses a coalescent prior and updates node times on the basis of the number of mutations along each edge of the tree sequence (i.e. using the "molecular clock").

The method is designed to operate on the output of `tsinfer <https://tsinfer.readthedocs.io/en/latest/>`_, which efficiently infers tree sequence *topologies* from large genetic datasets. ``tsdate`` and  ``tsinfer`` are scalable to the largest genomic datasets currently available.

The algorithm is described `in this Science paper <https://www.science.org/doi/10.1126/science.abi8264>`_
(preprint `here <https://www.biorxiv.org/content/10.1101/2021.02.16.431497v2>`_). We also provide evaluations of the accuracy and computational requirements of the method using both simulated and real data; the code to reproduce these results can be found in `another repository <https://github.com/awohns/unified_genealogy_paper>`_.

Please cite this paper if you use ``tsdate`` in published work:

> Anthony Wilder Wohns, Yan Wong, Ben Jeffery, Ali Akbari, Swapan Mallick, Ron Pinhasi, Nick Patterson, David Reich, Jerome Kelleher, and Gil McVean (2022) *A unified genealogy of modern and ancient genomes*. Science **375**: eabi8264; doi: https://doi.org/10.1126/science.abi826421

