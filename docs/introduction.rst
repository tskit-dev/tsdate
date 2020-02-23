.. _sec_introduction:

============
Introduction
============

``tsdate`` is a scalable method for estimating the age of ancestral nodes in a 
`tree sequence <https://www.youtube.com/watch?v=X1GEuQrF1jQ>`_). The method uses a prior distribution of node times under the coalescent, and updates the times on the basis of the number of mutations along each edge or branch of the tree sequence (i.e. using the "molecular clock").

The method is designed to operate on the output of
`tsinfer <https://tsinfer.readthedocs.io/en/latest/>`_), which is an efficient method 
for inferring tree sequence *topologies from large genetic datasets. ``tsdate`` and 
``tsinfer`` are scalable to the largest genomic datasets currently available.

.. note:: This documentation is currently under development. Please wait to use
``tsdate`` in your published work until the release of our preprint.