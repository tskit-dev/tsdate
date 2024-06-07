# Changelog

## [0.2.0] - 2024-06-10

**Bugfixes**

- Variational gamma uses a rescaling approach which helps considerably if e.g.
  population sizes vary over time

- Variational gamma does not use mutational area of branches, but average path
  length, which reduces bias in tree sequences containing polytomies

**Breaking changes**

- The default method has been changed to `variational_gamma`.

- Variational gamma uses an improper (flat) prior, and therefore
  no longer needs `population_size` specifying.

- The standalone `preprocess_ts` function also applies the `split_disjoint_nodes`
  method, which creates extra nodes but improves dating accuracy.

- Json metadata for mean time and variance in the mutation and node tables is now saved
  with a suitable schema. This means `json.loads()` is no longer needed to read it.

- The `mutation_rate` and `population_size` parameters are now keyword-only, and
  therefore these parameter names need to be explicitly typed out.

- The `ignore-oldest` option has been removed from the command-line interface,
  as it is no longer very helpful with new _tsinfer_ output, which has the root
  node split. The option is still accessible from the Python API.


## [0.1.7] - 2024-01-11

**Bugfixes**

- In variational gamma, rescale messages at end of each iteration to avoid numerical
  instability.

## [0.1.6] - 2024-01-07

**Breaking changes**

- The standalone `preprocess_ts` function now defaults to not removing unreferenced
  individuals, populations, or sites, aiming to change the tree sequence tables as
  little as possible.

- `get_dates` (previously undocumented) has been removed, as posteriors can be
  obtained using `return_posterior`. The `normalize` terminology previously used
  in `get_dates` is changed to `standardize` to better reflect the fact that the
  maximum (not sum) is one, and exposed via the `outside_standardize` parameter.

- The `Ne` argument to `date` has been deprecated (although it is
  still in the API for backward compatibility).  The equivalent argument
  `population_size` should be used instead.

- The CLI `-verbosity` flag no longer takes a number, but uses
  `action="count"`, so `-v` turns verbosity to INFO level,
  whereas `-vv` turns verbosity to DEBUG level.

- The `return_posteriors=True` option with `method="inside_outside"`
  previously returned a dict that included keys `start_time` and `end_time`,
  giving the impression that the posterior for node age is discretized over
  time slices in this algorithm. In actuality, the posterior is discretized
  atomically over time points, so `start_time` and `end_time` have been
  replaced by a single key `time`.

- The `return_posteriors=True` option with `method="maximization"` is no
  longer accepted (previously simply returned `None`)

- Python 3.7 is no longer supported.

**Features**

- A new continuous-time method, `"variational_gamma"` has been introduced, which
  uses an iterative expectation propagation approach. Tests show this increases
  accuracy, especially at older times. A Laplace approximation and damping are
  used to ensure numerical stability. After testing, the node priors used in this
  method are based on a global mixture prior, which can be refined during iteration.
  Future releases may switch to using this as the default method.

- Priors may be calculated using a piecewise-constant effective population trajectory,
  which is implemented in the `demography.PopulationSizeHistory` class. The
  `population_size` argument to `date` accepts either a single scalar effective
  population size, or a `PopulationSizeHistory` instance.

- Added support and wheels for Python 3.11

- The `.date()` function is now a wrapper for the individual dating methods
  (accessible using `tsdate.core.dating_methods`), which can be called independently.
  (e.g. `tsdate.inside_outside(...)`). This makes it easier to document method-specific
  options. The API docs have been revised accordingly. Provenance is now saved with the
  name of the method used as the celled command, rather than `"command": "date"`.

- Major re-write of documentation (now at
  [https://tskit.dev/tsdate/docs/](https://tskit.dev/tsdate/docs/)), to use the
  standard tskit jupyterbook framework.

**Bugfixes**

- The returned posteriors when `return_posteriors=True` now return actual
  probabilities (scaled so that they sum to one) rather than standardized
  "probabilities" whose maximum value is one.

- The population size is saved in provenance metadata (as a dictionary if
  it is a `PopulationSizeHistory` instance)

- `preprocess_ts` always records provenance as being from the `preprocess_ts`
  command, even if no gaps are removed. The command now has a (rarely used)
  `delete_intervals` parameter, which is normally filled out and saved in provenance
  (as it was before). If no gap deletion is done, the param is saved as `[]`


## [0.1.5] - 2022-06-07

**Features**

- Added the `time_units` parameter to `tsdate.date`, allowing users to specify
  the time units of the dated tree sequence. Default is `"generations"`.
- Added the `return_posteriors` parameter to `tsdate.date`. If True, the function
  returns a tuple of `(dated_ts, posteriors)`.
- `mutation_rate` is now a required argument in `tsdate.date` and `tsdate.get_dates`
- tsdate returns an error if users attempt to date an unsimplified tree sequence.
- Updated tsdate citation information to cite the recent Science paper
- Built wheel on Python 3.10


## [0.1.4] - 2021-06-30

**Features**

- The algorithm now operates completely in unscaled time (in units of generations) under
  the hood, which means that `tsdate.build_prior_grid` now requires the parameter
  `Ne`.
- Users now have access to the marginal posterior distributions on node age by running 
  `tsdate.get_dates`, though this is undocumented for now.

**Bugfixes**

- A fix to the way likelihoods are added should eliminate numerical errors that are
  sometimes encountered when dating very large tree sequences.


## [0.1.3] - 2021-02-15

**Features**

- Two new methods, `tsdate.sites_time_from_ts` and `tsdate.add_sampledata_times`, 
  support inference of tree sequences from non-contemporaneous samples.
- New tutorial on inferring tree sequences from modern and historic/ancient samples 
  explains how to use these functions in conjunction with `tsinfer`.
- `tsdate.preprocess_ts` supports dating inferred tree sequences which include large, 
  uninformative stretches (i.e. centromeres and telomeres). Simply run this function 
  on the tree sequence before dating it.
- `ignore_outside` is a new parameter in the outside pass which tells `tsdate` to 
  ignore edges from oldest root (these edges are often of low quality in `tsinfer`
  inferred tree sequences)
- Development environment is now equivalent to other `tskit-dev` projects


## [0.1.2] - 2020-02-28

- Improve user experience with more progress bars and logging.
- Slightly change traversal method in outside and outside maximization algorithms,
  this should only affect inference on inferred tree sequences with large numbers 
  of nodes at the same frequency.
- Improve reporting of current project version
- Use appdirs for default caching location
- Prevent dating tree sequences with dangling nodes



## [0.1.1] - 2020-02-25

- Bugfix release: resolve issue with precalculating prior values.


€# [0.1.0] - 2020-02-24

Early Alpha release made available via PyPI for community testing and evaluation.

Please don't use this version in published works.


