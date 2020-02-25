.. currentmodule:: tskit
.. _sec_tutorial:

========
Tutorial
========

.. _sec_tutorial_tsdate:

*********************
Dating Tree Sequences
*********************

To illustrate the typical use case of ``tsdate``'s Python API, we will first create
sample data with `msprime <https://github.com/tskit-dev/msprime>`_ and infer a tree
sequence from this data using `tsinfer <https://tsinfer.readthedocs.io/en/latest/>`_.
We will then run ``tsdate`` on the inferred tree sequence.

Let's start by creating some sample data with ``msprime`` using human-like parameters. 

.. code-block:: python

    import msprime

    sample_ts = msprime.simulate(sample_size=10, Ne=10000, 
                                            length=1e4,
                                            mutation_rate=1e-8,
                                            recombination_rate=1e-8,
                                            random_seed=2)
    print(sample_ts.num_trees,
          sample_ts.num_nodes)

The output of this code is:

.. code-block:: python

    12 29

We take this simulated tree sequence and turn it into a `tsinfer` SampleData object as 
documented `here <https://tsinfer.readthedocs.io/en/latest/api.html#tsinfer.SampleData.from_tree_sequence>`_, and then infer a tree sequence from
the data

.. code-block:: python

    import tsinfer
    
    sample_data = tsinfer.SampleData.from_tree_sequence(sample_ts) 
    inferred_ts = tsinfer.infer(sample_data)

.. note:: ``tsdate`` works best with `simplified tree sequences 
    <https://tskit.readthedocs.io/en/latest/python-api.html#tskit.TreeSequence.simplify>`_ (``tsinfer``'s documentation provides)
    details on how to simplify an inferred tree sequence. This should not be an issue
    when working with tree sequences simulated using ``msprime``.

Next, we run `tsdate` to estimate the ages of nodes and mutations in the inferred tree
sequence:

.. code-block:: python

	import tsdate
	dated_ts = tsdate.date(inferred_ts, Ne=10000, mutation_rate=1e-8)

All we need to run ``tsdate`` (with its default parameters) is the inferred tree sequence
object, the *estimated* effective population size, and *estimated* mutation rate.

.. _sec_tutorial_specify_prior:

+++++++++++++++++++
 Specifying a Prior
+++++++++++++++++++

The above example shows the basic use of ``tsdate``, using default parameters. The software has parameters the user can access through the :meth:`tsdate.build_prior_grid()`
function which may affect the runtime and accuracy of the algorithm.


.. _sec_tutorial_inside_outside_v_maximization:

++++++++++++++++++++++++++++++
Inside Outside vs Maximization
++++++++++++++++++++++++++++++

One of the most important parameters to consider is whether ``tsdate`` should use the 
inside-outside or the maximization algorithms to perform inference. A detailed
description of the algorithms will be presented in our preprint, but from the users
perspective, the inside-outside approach performs better empirically but has issues with
numeric stability, while the maximization approach is slightly less accurate
empirically, but is numerically stable.

.. _command_line_interface:

++++++++++++++++++++++++++++++
Command Line Interface Example
++++++++++++++++++++++++++++++

``tsdate`` also offers a convenient :ref:`command line interface (CLI) <sec_cli>` for 
accessing the core functionality of the algorithm. 

For a simple example of CLI, we'll first save the inferred tree sequence we created in
:ref:`the section above <sec_tutorial_tsdate>` as a file.

.. code-block:: python

    import tskit

    inferred_ts.dump("inferred_ts.trees")

Now we use the CLI to again date the inferred tree sequence and output the resulting
dated tree sequence to ``dated_ts.trees`` file:

::

    $ tsdate date inferred_ts.trees dated_ts.trees 10000 1e-8 --progress

The first two arguments are the input and output tree sequence file names, the third is the estimated effective population size, and the fourth is the estimated mutation rate. We also add the ``--progress`` option to keep track of ``tsdate``'s progress.
