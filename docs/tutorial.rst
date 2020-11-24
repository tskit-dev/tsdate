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
object, the *estimated* effective population size, and *estimated* mutation rate. Here we
have provided a human mutation rate per base pair per generation, so the nodes dates in the
resulting tree sequence should be interpreted as generations.

.. _sec_tutorial_specify_prior:

+++++++++++++++++++
 Specifying a Prior
+++++++++++++++++++

The above example shows the basic use of ``tsdate``, using default parameters. The 
software has parameters the user can access through the :meth:`tsdate.build_prior_grid()`
function which may affect the runtime and accuracy of the algorithm.


.. _sec_tutorial_inside_outside_v_maximization:

++++++++++++++++++++++++++++++
Inside Outside vs Maximization
++++++++++++++++++++++++++++++

One of the most important parameters to consider is whether ``tsdate`` should use the 
inside-outside or the maximization algorithms to perform inference. A detailed
description of the algorithms will be presented in our preprint, but from the users
perspective, the inside-outside approach performs better empirically but has issues with
numerical stability, while the maximization approach is slightly less accurate
empirically, but is numerically stable.

.. _command_line_interface:

++++++++++++++++++++++
Troubleshooting tsdate
++++++++++++++++++++++

If numerical stability issues are encountered when attempting to date
tree sequences using the Inside-Outside algorithm, it may be necessary to remove 
large sections of the tree which do not have any variable sites using 
:meth:`tsdate.preprocess_ts()` method.

.. _troubleshooting:

*******************************************************************
Inferring and Dating Tree Sequences with Historic (Ancient) Samples
*******************************************************************

``tsdate`` and ``tsinfer`` can be used together to infer tree sequences from both
modern and historic samples. The following recipe shows how this is accomplished
with a few lines of Python. The only requirement is a tsinfer.SampleData file with
modern and historic samples (the latter are specified using the `individuals_time`
array in a tsinfer.SampleData file).

.. code-block:: python

   import msprime
   import tsdate
   import tsinfer
   import tskit

   import numpy as np


   def make_historical_samples():
       samples = [
           msprime.Sample(population=0, time=0),
           msprime.Sample(0, 0),
           msprime.Sample(0, 0),
           msprime.Sample(0, 0),
           msprime.Sample(0, 1.0),
           msprime.Sample(0, 1.0)
       ]
       sim = msprime.simulate(samples=samples, mutation_rate=1, length=100)
       # Get the SampleData file from the simulated tree sequence
       # Retain the individuals times and ignore the sites times.
       samples = tsinfer.SampleData.from_tree_sequence(
         sim, use_sites_time=False, use_individuals_time=True)
       return samples

   def infer_historic_ts(samples, Ne=1, mutation_rate=1):
      """
      Input is tsinfer.SampleData file with modern and historic samples.
      """
      modern_samples = samples.subset(np.where(samples.individuals_time[:] == 0)[0])
      inferred_ts = tsinfer.infer(modern_samples) # Infer tree seq from modern samples
      # Removes unary nodes (currently required in tsdate), keeps historic-only sites
      inferred_ts = tsdate.preprocess_ts(inferred_ts, filter_sites=False)
      dated_ts = tsdate.date(inferred_ts, Ne=Ne, mutation_rate=mutation_rate) # Date tree seq
      sites_time = tsdate.sites_time_from_ts(dated_ts)  # Get tsdate site age estimates
      dated_samples = tsdate.add_sampledata_times(
         samples, sites_time) # Get SampleData file with time estimates assigned to sites
      ancestors = tsinfer.generate_ancestors(dated_samples)
      ancestors_w_proxy = ancestors.insert_proxy_samples(
         dated_samples, allow_mutation=True)
      ancestors_ts = tsinfer.match_ancestors(dated_samples, ancestors_w_proxy)
      return tsinfer.match_samples(
         dated_samples, ancestors_ts, force_sample_times=True) 

   samples = make_historical_samples()
   inferred_ts = infer_historic_ts(samples)
   
We simulate a tree sequence with six sample chromosomes, four modern and
two historic. We then infer and date a tree sequence using only the modern
samples. Next, we find derived alleles which are carried by the historic samples and use
the age of the historic samples to constrain the ages of these alleles. Finally, we
reinfer the tree sequence, using the date estimates from tsdate and the historic 
constraints rather than the frequency of the alleles to order mutations in ``tsinfer``.
Historic samples are added to the ancestors tree sequence as `proxy nodes, in addition
to being used as samples <https://tsinfer.readthedocs.io/en/latest/api.html?highlight=proxy#tsinfer.AncestorData.insert_proxy_samples>`_.

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
