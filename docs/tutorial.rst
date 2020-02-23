``tsdate`` is compatible with `simplified tree sequences <https://tskit.readthedocs.io/en/latest/python-api.html#tskit.TreeSequence.simplify>`_

.. currentmodule:: tskit
.. _sec_tutorial:

========
Tutorial
========

.. _sec_tutorial_tsdate:

*********************
Dating Tree Sequences
*********************

To illustrate the typical use case of the Python API of ``tsdate``, we will first create
sample data with `msprime <https://github.com/tskit-dev/msprime>`_ and infer a tree
sequence from this data using `tsinfer <https://tsinfer.readthedocs.io/en/latest/>`_).
We will then run ``tsdate`` on the inferred tree sequence.

Let's start by creating some sample data with ``msprime`` using human-like parameters. 

.. code-block:: python

    import msprime

    sample_tree_sequence = msprime.simulate(sample_size=10, Ne=10000, length=1000,
    										mutation_rate=1e-8, recombination_rate=1e-8,
    										random_seed=2)
    print(sample_tree_sequence.num_trees, sample_tree_sequence.num_nodes)
::
    2 20

We take this simulated tree sequence and turn it into a `tsinfer` SampleData object as 
documented `here <https://tsinfer.readthedocs.io/en/latest/api.html#tsinfer.SampleData.from_tree_sequence>`_, and then infer a tree sequence from
the data

	sample_data = tsinfer.SampleData.from_tree_sequence(ts)
	inferred_ts = tsinfer.infer(sample_data)

Next, we run `tsdate` to estimate the ages of nodes and mutations in the inferred tree
sequence:

	import tsdate
	dated_ts = tsdate.date(inferred_ts, Ne=10000, mutation_rate=1e-8)

All we need to run ``tsdate`` with its default parameters is the inferred tree sequence
object, the *estimated effective population size, and *estimated mutation rate.

.. _sec_tutorial_specify_prior:

+++++++++++++++++++
 Specifying a Prior
+++++++++++++++++++

The above example shows the basic use of ``tsdate``, using default parameters. The software has parameters the user can access through the tsdate.build_prior_grid() function which may affect the runtime, accuracy, and numerical stability of the algorithm.

.. _sec_tutorial_inside_outside_v_maximization:

++++++++++++++++++++++++++++++
Inside Outside vs Maximization
++++++++++++++++++++++++++++++

.. _command_line_interface:

++++++++++++++++++++++++++++++
Command Line Interface Example
++++++++++++++++++++++++++++++

