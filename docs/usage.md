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

We'll first generate a few example "undated" tree sequences:

```{code-cell} ipython3
:tags: [hide-input]
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
model = species.get_demographic_model("AmericanAdmixture_4B18")
contig = species.get_contig("chr1", left=1e7, right=1.1e7, mutation_rate=model.mutation_rate)
samples = {"AFR": 5, "EUR": 5, "ASIA": 5, "ADMIX": 5}
# Create DNA sequences, stored in the tsinfer SampleData format
stdpopsim_ts = stdpopsim.get_engine("msprime").simulate(model, contig, samples, seed=123)
sample_data = tsinfer.SampleData.from_tree_sequence(stdpopsim_ts)
inf_ts = tsinfer.infer(sample_data)

print(f"* Simulated `sim_ts` ({2*n} genomes from a popn of {Ne} diploids, mut_rate={mu} /bp/gen)")
print(f"* Inferred `inf_ts` using tsinfer ({stdpopsim_ts.num_samples} samples of human {contig.origin})")

```

(sec_usage_basic_example)=

## Quickstart

Given a known genetic genealogy in the form of a tree sequence, _tsdate_ simply
re-estimates the node times based on the mutations on each edge. Usage is as
simple as calling the {func}`date` function with an
estimated per base pair per generation mutation rate.

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
import scipy.stats as stats

import numpy as np
def plot_real_vs_tsdate_times(
    x, y, ax=None, ts_x=None, ts_y=None, y_variance=None, delta=0.1, **kwargs
):
    if ax is None:
        ax=plt.gca()
    x, y = np.array(x), np.array(y)
    line_pts = np.logspace(np.log10(delta), np.log10(x.max()), 10)
    ax.plot(line_pts, line_pts, linestyle=":", c="black")
    if y_variance is None:
        ax.scatter(x + delta, y + delta, **kwargs)
    else:
        # variance is of a gamma distribution. The sample nodes have zero mean, so ignore those divisions
        with np.errstate(invalid='ignore'):
            scale = y_variance / y
            shape = y / scale
            lower = y - stats.gamma.ppf(0.025, shape, scale=scale)
            upper = stats.gamma.ppf(0.975, shape, scale=scale) - y
            err = np.array([lower, upper])
            ax.errorbar(x + delta, y + delta, err + delta, fmt='_', ecolor='lightgrey', mec='tab:blue')
    ax.set_xscale('log')
    ax.set_xlabel(f'Real time' + ('' if ts_x is None else f' ({ts_x.time_units})'))
    ax.set_yscale('log')
    ax.set_ylabel(f'Estimated time from tsdate' + ('' if ts_y is None else f' ({ts_y.time_units})'))

fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharex=True, sharey=True)
plot_real_vs_tsdate_times(ts.nodes_time, redated_ts.nodes_time, axes[0], ts, redated_ts)
mean, var = np.array([[n.metadata["mn"], n.metadata["vr"]] for n in redated_ts.nodes()]).T
plot_real_vs_tsdate_times(ts.nodes_time, mean, axes[1], ts, redated_ts, y_variance=var)
axes[1].set_ylabel("");
```

The left hand plot shows the redated tree sequence node times. The right hand plot is
essentially identical but with 95% confidence intervals (technically, it shows the
unconstrained {ref}`posterior node times<sec_usage_posterior>`, with confidence intervals
based on the [gamma distribution](https://en.wikipedia.org/wiki/Gamma_distribution)
fitted by the default _tsdate_ {ref}`method<sec_methods>`).

:::{note}
By default time is assumed to be measured in "generations", but this can be changed by
using the `time_units` parameter: see the [Timescale adjustment](sec_usage_timescale) section.
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
{ref}`sec_real_data_preprocessing` section for more details)

```{code-cell} ipython3
import tsdate
simplified_ts = tsdate.preprocess_ts(
    inf_ts,
    remove_telomeres=False  # Simulated example, so no flanking regions / telomeres exist
)
dated_ts = tsdate.date(simplified_ts, mutation_rate=model.mutation_rate)
print(
    f"Dated `inf_ts` (inferred from {inf_ts.num_sites} variants under the {model.id}",
    f"stdpopsim model, mutation rate = {model.mutation_rate} /bp/gen)"
)
```
:::{note}
In simulated data you may not have missing data regions, and you may be able to
pass `erase_flanks=False` to the `preprocess_ts` function.
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

Note that if a bifurcating topology in the original genealogies has been collapsed
into a *polytomy* (multifurcation) in the inferred version, then the default node
times output by _tsdate_ correspond to the time of the oldest bifurcating node.
If you wish to consider diversity or coalescence over time, you should therefore
consider explicitly setting the `match_segregating_sites` parameter to `True`
(see the {ref}`sec_real_data_rescaling` section).


(sec_usage_posterior)=

## Results and posteriors

The default output of _tsdate_ is a new, dated tree sequence,
created with node times set to the (constrained) posterior mean time for each node,
and mutation times set halfway between the time of the node above and
the time of the node below the mutation.

However, these tree-sequence time arrays do not capture all the information
about the distribution of times. They do not contain information about the
variance in the time estimates, and if necessary, mean times are _constrained_
to create a valid tree sequence (i.e. if the mean time of a child node is
older than the mean time of all of its parents, a small value, $\epsilon$, is added
to the parent time to ensure a valid tree sequence).

For this reason, there are two ways to get variances and unconstrained dates when
running _tsdate_:

1. The nodes and mutations in the tree sequence will usually contain
    {ref}`tskit:sec_metadata` specifying mean time and its variance.
    These metadata values (currently saved as `mn` and `vr`) are not constrained by
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

(sec_usage_timescale)=

### Timescale adjustment

The default _tsdate_ timescale is "generations". Changing this can be as simple as
providing a `time_units` argument:

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

(sec_usage_memory_time)=

## Memory and run time

_Tsdate_ can be run on most modern computers. Using the default `variational_gamma`
{ref}`method<sec_methods>`, large tree sequences of millions or
tens of millions of edges take tens of minutes and gigabytes of RAM (e.g. 10 GB / 50 mins
on a 2024-era Apple M2 laptop for a tree sequence of 65 million edges covering
81 megabases of 2.85 million samples of human chromosome 17 from
{cite:t}`anderson-trocme2023genes`).


```{code-cell} ipython3
:"tags": ["remove-input"]
# This cell deliberately removed (not just hidden via a toggle) as it's not helpful
# for understanding tskit code (it's merely plotting code)
import matplotlib_inline
import matplotlib.pyplot as plt
import numpy as np
%matplotlib inline
matplotlib_inline.backend_inline.set_matplotlib_formats('svg')

data = np.genfromtxt("data/perf_sim_MEMTIME_data+42.csv", delimiter="\t", names=True)

def perf_mem_time_plot(ax, df, colours, log=True):
    ax.plot(df["sample_size"], np.array(df["time"])/60, "o-", c=colours[1])
    if log:
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title("Log scale")
        ax.set_xlim(20, 4e6)
    else:
        ax.set_title("Linear scale")
        ax.set_xlim(20, 3e6)
    ax.set_ylabel("Time (mins)", c=colours[1])
    ax.set_ylim(0.2, 55)
    ax.set_xlabel("Number of sampled genomes ($n$)")
    r_ax = ax.twinx()
    r_ax.plot(df["sample_size"], np.array(df["mem"])/1000/1000/1000, "o-", c=colours[0])
    r_ax.set_ylim(0.2, 55)
    if log:
        r_ax.set_yscale("log")
        ticks = [0.3, 1, 3, 10, 30]
        ax.set_yticks(ticks)
        ax.set_yticklabels([f'{tick:g}' for tick in ticks])
        r_ax.set_yticks(ticks)
        r_ax.set_yticklabels([f'{tick:g}' for tick in ticks])
    r_ax.set_ylabel("Memory (Gb)", c=colours[0])

fig, ax = plt.subplots(1, 2, figsize=(10, 3), gridspec_kw={'wspace': 0.5})
perf_mem_time_plot(ax[0], data, ("tab:orange", "tab:blue"), log=False)
perf_mem_time_plot(ax[1], data, ("tab:orange", "tab:blue"))
```

Running the dating algorithm is linear in the number of edges in the tree sequence.
This makes _tsdate_ usable even for vary large tree sequences (e.g. millions of samples).
For large instances, if you are running _tsdate_ interactively, it can be useful to
specify the `progress` option to display a progress bar telling you how long
different stages of dating will take.

As the time taken to date a tree sequence using _tsdate_ is only a fraction of that
required to infer the initial tree sequence, the core _tsdate_ algorithm
has not been parallelised to allow running on many CPU cores. 

:::{note}
Extensive use of just-in-time compilation via [numba](https://numba.pydata.org)
means that it can take tens of seconds to load the _tsdate_ module into python.
See {ref}`here <sec_installation_testing>` for a workaround if this is causing you problems.
:::

## CLI use

Computationally-intensive uses of _tsdate_ are likely to involve
running the program non-interactively, e.g. as part of an
automated pipeline. In this case, it may be useful to use the
command-line interface. See {ref}`sec_cli` for more details.

