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

(sec_usage)=

# Usage

We'll first generate a few "undated" tree sequences for later use:

```{code-cell} ipython3
:tags: [hide-input]
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)  # don't display FutureWarnings for stdpopsim
import msprime
import numpy as np
import stdpopsim
import tsinfer
import tskit

# Use msprime to create a simulated tree sequence with mutations, for demonstration
n = 10
Ne = 100
mu = 1e-6
ts = msprime.sim_ancestry(n, population_size=Ne, sequence_length=1e6, random_seed=123)
ts = msprime.sim_mutations(ts, rate=mu, random_seed=123, discrete_genome=False)
# Remove time information
tables = ts.dump_tables()
tables.nodes.time = np.where(tables.nodes.flags & tskit.NODE_IS_SAMPLE, 0, np.arange(ts.num_nodes, dtype=float))
tables.mutations.time = np.full(ts.num_mutations, tskit.UNKNOWN_TIME)
tables.time_units = tskit.TIME_UNITS_UNCALIBRATED
sim_ts = tables.tree_sequence()

# Use stdpopsim to create simulated genomes for inference
species = stdpopsim.get_species("HomSap")
model = species.get_demographic_model("AmericanAdmixture_4B11")
contig = species.get_contig("chr1", left=1e7, right=1.1e7, mutation_rate=model.mutation_rate)
samples = {"AFR": 5, "EUR": 5, "ASIA": 5, "ADMIX": 5}
# Create DNA sequences, stored in the tsinfer SampleData format
stdpopsim_ts = stdpopsim.get_engine("msprime").simulate(model, contig, samples, seed=123)
sample_data = tsinfer.SampleData.from_tree_sequence(stdpopsim_ts)
inf_ts = tsinfer.infer(sample_data)
if inf_ts.table_metadata_schemas.node.schema is None:
    # Only needed because of tsinfer bug that doesn't set schemas: remove when fixed
    tables = inf_ts.dump_tables()
    tables.nodes.metadata_schema = tskit.MetadataSchema.permissive_json()
    inf_ts = tables.tree_sequence()

print(f"* Simulated `sim_ts` ({2*n} genomes from a popn of {Ne} diploids, mut_rate={mu} /bp/gen)")
print(f"* Inferred `inf_ts` using tsinfer ({stdpopsim_ts.num_samples} samples of human {contig.origin})")

```
(sec_usage_basic_example)=

## Quickstart

Given a known genetic genealogy in the form of a tree sequence, _tsdate_ simply
re-estimates the node times based on the mutations on each edge. Usage is as
simple as calling the {func}`date` function with an estimated per base pair
mutation rate.

```{code-cell} ipython3
import tsdate
# Running `tsdate` is usually a single function call, as follows:
mu_per_bp_per_generation = 1e-6
redated_ts = tsdate.date(sim_ts, mutation_rate=mu_per_bp_per_generation)
```

This simple example has no recombination, infinite sites mutation,
a high mutation rate, and a known genealogy, so we would expect that the node times
as estimated by _tsdate_ from the mutations would be very close to the actual node times,
as indeed they seem to be:

```{code-cell} ipython3
:tags: [hide-input]
from matplotlib import pyplot as plt
import numpy as np
def plot_real_vs_tsdate_times(x, y, ts_x=None, ts_y=None, delta = 0.1, **kwargs):
    x, y = np.array(x), np.array(y)
    plt.scatter(x + delta, y + delta, **kwargs)
    plt.xscale('log')
    plt.xlabel(f'Real time' + ('' if ts_x is None else f' ({ts_x.time_units})'))
    plt.yscale('log')
    plt.ylabel(f'Estimated time from tsdate' + ('' if ts_y is None else f' ({ts_y.time_units})'))
    line_pts = np.logspace(np.log10(delta), np.log10(x.max()), 10)
    plt.plot(line_pts, line_pts, linestyle=":")

plot_real_vs_tsdate_times(ts.nodes_time, redated_ts.nodes_time, ts, redated_ts)
```

:::{note}
By default time is assumed to be measured in "generations", but this can be changed by
using the `time_units` parameter. If you are using the alternative discrete-time methods 
you will also need to adjust the prior, as described in the
[Timescale adjustment](sec_usage_timescale) section.
:::

(sec_usage_inferred_example)=

## Inferred topologies

A more typical use-case is where the genealogy has been inferred from DNA sequence data,
for example by {ref}`tsinfer<tsinfer:sec_introduction>` or
[Relate](https://myersgroup.github.io/relate/). Below we will demonstrate with `tsinfer`
output based on DNA sequences generated by a more realistic simulation.

With real data, especially from `tsinfer` you may want to {func}`preprocess<preprocess_ts>`
the tree sequence before dating. This removes regions with no variable sites,
also simplifies to remove locally unary portions of nodes, and splits disjoint nodes
into separate datable nodes (see the
{ref}`sec_usage_real_data_stability` section below for more details)

```{code-cell} ipython3
import tsdate
simplified_ts = tsdate.preprocess_ts(inf_ts)
dated_ts = tsdate.date(simplified_ts, mutation_rate=model.mutation_rate)
print(
    f"Dated `inf_ts` (inferred from {inf_ts.num_sites} variants under the {model.id}",
    f"stdpopsim model, mutation rate = {model.mutation_rate} /bp/gen)"
)
```
:::{note}
In simulated data you may not have missing data regions, and you may be able to
pass `remove_telomeres=False` to the `preprocess_ts` function.
:::

The inference in this case is much more noisy (as illustrated using the original
and inferred times of the node under each mutation):

```{code-cell} ipython3
:tags: [hide-input]
# If there are multiple mutations at a site, we simply pick the first one

plot_real_vs_tsdate_times(
    [s.mutations[0].time for s in stdpopsim_ts.sites()],
    [s.mutations[0].metadata['mn'] for s in dated_ts.sites()],
    delta=100,
    alpha=0.1,
)
```

(sec_usage_posterior)=

## Results and posteriors

The default output of _tsdate_ is a new, dated tree sequence,
created with node times changed to the posterior mean time for each node,
and mutation times set halfway between the node above and below.

However, these tree-sequence time arrays do not capture all the information
about the distribution of times. They do not contain information
about the variance in the time estimates, and even the mean times must be constrained
to create a valid tree sequence. For example, a mutation time must be bounded between the
time of the node above and the node below, and if the mean time of a child node is
older than the mean time of all of its parents, a small value, $\epsilon$, is added
to the parent time to ensure a valid tree sequence.

For this reason, there are two ways to get additional, unconstrained dates when
running _tsdate_:

1. The nodes and mutations in the tree sequence will usually contain
    {ref}`tskit:sec_metadata` specifying true mean time and the variance in mean time.
    The metadata values (currently saved as `mn` and `vr`) need not be constrained by
    the topology of the tree sequence, and should be used in preference
    e.g. to `nodes_time` and `mutations_time` when evaluating the accuracy of _tsdate_. 
2. The `return_fit` parameter can be used when calling {func}`tsdate.date`, which
    then returns both the dated tree sequence and a fit object. This object can then be
    queried for the unconstrained posterior distributions using e.g. `.node_posteriors()`
    which can be read in to a [pandas](https://pandas.pydata.org) dataframe, as below:

```{code-cell} ipython3
import pandas as pd
redated_ts, fit = tsdate.date(
    sim_ts, mutation_rate=1e-6, return_fit=True)
posteriors_df = pd.DataFrame(fit.node_posteriors())  # mutation_posteriors() also available
posteriors_df.tail()  # Show the dataframe
```

<!--
Since we are using a {ref}`sec_methods_discrete_time` method, each node
(numbered column of the dataframe) is associated with a vector of probabilities
that sum to one: each cell gives the probability that the time of the node
whose ID is given by the column header lies at the specific timepoint
given by the `time` column.

For the continuous-time `variational_gamma` method, the posterior for
each node is represented by the shape and rate parameter of the gamma approximation,
as described by the `parameter` column.
-->


(sec_usage_timescale)=

### Timescale adjustment

The default _tsdate_ timescale is "generations". Changing this can be as simple as:

```{code-cell} ipython3
mu_per_bp_per_gen = 1e-8  # per generation
ts_generations = tsdate.date(ts, mutation_rate=mu_per_bp_per_gen)

mu_per_bp_per_year = 3.4e-10  # Human generation time ~ 29 years
ts_years = tsdate.date(ts, mutation_rate=mu_per_bp_per_year, time_units="years")
```

However, if you are specifying a node-specific prior, e.g. because you are using a
discrete-time method, you will also need to change the scale of the prior. In particular,
if you are setting the prior using the `population_size` argument, you will also need to
modify that by multiplying it by the generation time. For example:

```{code-cell} ipython3
Ne = 100  # Diploid population size
mu_per_bp_per_gen = 1e-8  # per generation
ts_generations = tsdate.inside_outside(ts, mutation_rate=mu_per_bp_per_gen, population_size=Ne)

# To infer dates in years, adjust both the rates and the population size:
generation_time = 29  # Years
mu_per_bp_per_year = mu_per_bp_per_gen / generation_time
ts_years = tsdate.inside_outside(
    ts,
    mutation_rate=mu_per_bp_per_year,
    population_size=Ne * generation_time,
    time_units="years"
)

# Check that the inferred node times are identical, just on different scales
assert np.allclose(ts_generations.nodes_time, ts_years.nodes_time / generation_time, 5)
```

(sec_usage_real_data)=

## Real data

Real world data is likely to consist of larger datasets than in the example, and
may exhibit issues that are not present in simulated data and can e.g. cause numerical
instability and other problems. In addition to {func}`preprocessing<preprocess_ts>`
below we detail some other solutions to problems common in real datasets.

(sec_usage_real_data_scaling)=

### Memory and run time

_Tsdate_ can be run on most modern computers: large tree sequences of millions or
tens of millions of edges will take of the order of hours, and use
tens of GB of RAM (e.g. 24 GB / 1 hour on a 2022-era laptop
for a tree sequence of 5 million edges covering
60 megabases of 7500 samples of human chromosome 6 from {cite:t}`wohns2022unified`).


:::{todo}
Add some scaling plots. Some real-world examples: a dataset of 10K samples of half a
million sites (~4M edges) on one node of a
2023 Intel Platinum cluster takes ~30 mins (20GB max memory) for the `inside-outside`
method and ~10 mins (1.5GB max memory) using the `variational_gamma` method.
:::

Running the dating algorithm is linear in the number of edges in the tree sequence.
This makes _tsdate_ usable even for vary large tree sequences (e.g. millions of samples).
For large instances, if you are running _tsdate_ interactively, it can be useful to
specify the `progress` option to display a progress bar telling you how long
different stages of dating will take.

The time taken to date a tree sequence using _tsdate_ is only a fraction of that
required to infer the initial tree sequence, therefore the core _tsdate_ algorithm
has not been parallelised to allow running on many CPU cores. 


#### Continuous time optimisations

The [expectation progagation](sec_methods_expectation_propagation) approach
in the `variational_gamma` method involves iteratively refining the
local time estimates. The number of rounds of iteration can be
set via the `max_iterations` parameter. Reducing this will speed up _tsdate_
inference, but may produce worse date estimation. Future updates to _tsdate_ may
optimise this so that iterations are halted after an appropriate convergence criterion
has been reached.

#### Continuous time optimisations

If the {ref}`method<sec_methods>` used for dating involves discrete time slices, _tsdate_ scales
quadratically in the number of time slices chosen. For greater temporal resolution,
you are thus advised to keep with the `variational_gamma` method, which does not discretise time.

Some precalculation of regularly reused mutational likelihoods *can* be parallelised
easily: this step can be sped up by specifying the `num_threads` parameter to 
{func}`date` (however, this behaviour is subject to change in future versions).

For discrete-time methods, before the dating algorithm is run the conditional
coalescent prior distribution must be calculated for each node.
Although this is roughly linear in the number of nodes,
it can still take some time if there are millions of nodes.
To speed up this process an approximation to the conditional coalescent
is used for nodes which have a large number of descendants
(the exact number can be modified by {ref}`making a prior<sec_priors_timegrid>`
using the `approx_prior_size` parameter).
The results of this approximation are cached, so that _tsdate_ will run slightly faster
the second time it is run.

### CLI use

Computationally-intensive uses of _tsdate_ are likely to involve
running the program non-interactively, e.g. as part of an
automated pipeline. In this case, it may be useful to use the
command-line interface. See {ref}`sec_cli` for more details.

(sec_usage_real_data_stability)=

### Numerical stability and preprocessing

Numerical stability issues will manifest themselves by raising an error when dating.
They are usually caused by "bad" tree sequences (i.e.
pathological combinations of topologies and mutations). These can be caused,
for example, by long deep branches with very few mutations, such as samples attaching directly
to a local root. These can often be fixed by removing bad regions of the tree sequence,
e.g. regions that have no variation because they are unmappable or have been
removed by a QC filter, such as at the centromere.

In addition, the `tsinfer` algorithm produces can overestimate the span of genome
covered by some nodes, especially those at ancient times. This is particularly
seen when nodes contain "locally unary" sections (i.e. where the node only has
one child in some trees), or where a node is "disjoint", disappearing from one tree
to reappear in a tree in a more distant region of the genome. This extended node
span can be another source of numerical instability and inaccuracy. Locally unary
sections of nodes can be removed by simplification, and disjoint nodes can be split
using {func}`tsdate.util.split_disjoint_nodes`.

The {func}`tsdate.preprocess_ts()` function carries out all these procedures in
a single function call: it removes data-poor regions, simplifies, and splits
disjoint nodes. See the documentation for that function for details on how to
increase or decrease its stringency.

:::{note}
If unary regions are *correctly* estimated, they can help improve dating slightly.
You can set the `allow_unary=True` option to run tsdate on such tree sequences.
:::
