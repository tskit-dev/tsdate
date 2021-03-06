--------------------
[0.1.3] - 2021-02-15
--------------------

**Features**

- Two new methods, ``tsdate.sites_time_from_ts`` and ``tsdate.add_sampledata_times``, 
  support inference of tree sequences from non-contemporaneous samples.
- New tutorial on inferring tree sequences from modern and historic/ancient samples 
  explains how to use these functions in conjunction with ``tsinfer``.
- ``tsdate.preprocess_ts`` supports dating inferred tree sequences which include large, 
  uninformative stretches (i.e. centromeres and telomeres). Simply run this function 
  on the tree sequence before dating it.
- ``ignore_outside`` is a new parameter in the outside pass which tells ``tsdate`` to 
  ignore edges from oldest root (these edges are often of low quality in ``tsinfer``
  inferred tree sequences)
- Development environment is now equivalent to other ``tskit-dev`` projects


--------------------
[0.1.2] - 2020-02-28
--------------------

- Improve user experience with more progress bars and logging.
- Slightly change traversal method in outside and outside maximization algorithms,
  this should only affect inference on inferred tree sequences with large numbers 
  of nodes at the same frequency.
- Improve reporting of current project version
- Use appdirs for default caching location
- Prevent dating tree sequences with dangling nodes


--------------------
[0.1.1] - 2020-02-25
--------------------

Bugfix release: resolve issue with precalculating prior values.


--------------------
[0.1.0] - 2020-02-24
--------------------

Early Alpha release made available via PyPI for community testing and evaluation.

Please don't use this version in published works.


