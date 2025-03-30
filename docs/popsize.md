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

(sec_popsize)=

# Population size

The rate of coalescence events over time is determined by demographics,
population structure, and selection. In the {func}`tsdate.variational_gamma` method 
with `match_segregating_sites=True`, the  {ref}`rescaling<sec_real_data_rescaling>`
step attempts to distribute coalescences so that the mutation rate over time is
reasonably constant. Assuming this is a good approximation, and coupled with
the absence of a strong informative prior on internal nodes, results from _tsdate_
should be robust to deviations from neutrality and panmixia. This means that,
for example, the inverse coalescence rate in a dated tree sequence should
reflect historical processes.

To illustrate this, we will generate data from a population that was large
in recent history, but had a small bottleneck of only 100 individuals
10 thousand generations ago, lasting for (say) 80 generations,
with a medium-sized population before that:

```{code-cell} ipython3
import msprime
import demesdraw
from matplotlib import pyplot as plt

bottleneck_time = 10000
demography = msprime.Demography()
demography.add_population(name="Population", initial_size=5e4)
demography.add_population_parameters_change(time=bottleneck_time, initial_size=100)
demography.add_population_parameters_change(time=bottleneck_time + 80, initial_size=2e3)

mutation_rate = 1e-8
# Simulate a short tree sequence with a population size history.
ts = msprime.sim_ancestry(
    10, sequence_length=2e6, recombination_rate=2e-8, demography=demography, random_seed=321)
ts = msprime.sim_mutations(ts, rate=mutation_rate, random_seed=321)
fig, ax = plt.subplots(1, 1, figsize=(4, 6))
demesdraw.tubes(demography.to_demes(), ax, scale_bar=True)
ax.annotate(
  "bottleneck",
  xy=(0, bottleneck_time),
  xytext=(1e4, bottleneck_time * 1.04),
  arrowprops=dict(arrowstyle="->"))
```

To test how well tsdate does in this situation, we can redate the known (true) tree sequence topology,
which replaces the true node and mutation times with estimates from the dating algorithm. 

```{code-cell} ipython3
import tsdate
redated_ts = tsdate.date(ts, mutation_rate=mutation_rate, match_segregating_sites=True)
```

If we plot true node time against tsdate-estimated node time for
each node in the tree sequence, we can see that the _tsdate_ estimates
are pretty much in line with the truth, although there is a a clear band
which is difficult to date at 10,000 generations, corresponding to the
instantaneous change in population size at that time.

```{code-cell} ipython3
:tags: [hide-input]
import numpy as np

fig, axes = plt.subplots(1, 2, figsize=(10, 5))
def plot_real_vs_tsdate_times(ax, x, y, ts_x=None, ts_y=None, plot_zeros=False, title=None, **kwargs):
    x, y = np.array(x), np.array(y)
    if plot_zeros is False:
        x, y = x[(x * y) > 0], y[(x * y) > 0]
    ax.scatter(x, y, **kwargs)
    ax.set_xscale('log')
    ax.set_xlabel(f'Real time' + ('' if ts_x is None else f' ({ts_x.time_units})'))
    ax.set_yscale('log')
    ax.set_ylabel(f'Estimated time from tsdate'+ ('' if ts_y is None else f' ({ts_y.time_units})'))
    line_pts = np.logspace(np.log10(x.min()), np.log10(x.max()), 10)
    ax.plot(line_pts, line_pts, linestyle=":")
    if title:
      ax.set_title(title)

unconstr_times = [nd.metadata.get("mn", nd.time) for nd in redated_ts.nodes()]
plot_real_vs_tsdate_times(
  axes[0], ts.nodes_time, unconstr_times, ts, redated_ts, alpha=0.1, title="With rescaling")
redated_unscaled_ts = tsdate.date(ts, mutation_rate=mutation_rate, rescaling_intervals=0)
unconstr_times2 = [nd.metadata.get("mn", nd.time) for nd in redated_unscaled_ts.nodes()]
plot_real_vs_tsdate_times(
  axes[1], ts.nodes_time, unconstr_times2, ts, redated_ts, alpha=0.1, title="Without rescaling")
```

The plot on the right is from running _tsdate_ without the
{ref}`rescaling<sec_real_data_rescaling>` step. It is clear that for populations with
variable sizes over time, rescaling can help considerably in obtaining correct date
estimations.

<!--

While not as rigorous as statistical approaches such as SINGER, we can also
look at the rate of coalescence over time in our dated tree sequence. Note, however, that
this does not account either for the concetration of coalescences caused by polytomies,
or the _variation_ in coalescence times {ref}`estimated by _tsdate_<sec_usage_posterior>`.
Both of these are major contributors to coalescence rate variation, and accounting for
them is needed if some measure of confidence in rate is required.
The inverse of the instantaeous coalescence rate (IICR) in a panmictic population is
a measure of effective population size. We can compare
this to the actual population size in the simulation, to see how well we can infer
historical population sizes. Note that for small sample sizes, there are very few
coalescence points in recent time, meaning that the estimates of IICR in recent time
are likely to be highly variable.

```{code-cell} ipython3
fig, ax = plt.subplots(1, 1, figsize=(15, 3))
demesdraw.size_history(demography.to_demes(), ax, log_size=True, inf_ratio=0.2)
ax.set_ylabel("Population size", rotation=90);

#TODO: add inverse coalescence rate and its inverse (estimate of Ne)
```
-->


## Misspecified priors

:::{note}
Functionality described below applies only to the non-default,
{ref}`sec_methods_discrete_time` methods, and is preliminary and subject to
change in the future. For this reason, classes and methods may not form part
of the publicly available API and may not be fully documented yet.
:::

Approaches such as the `inside_outside` method that use a coalescent prior
based on a fixed population size. perform very poorly on data where
demography has been variable over time.

```{code-cell} ipython3
import tsdate
est_pop_size = ts.diversity() / (4 * mutation_rate)  # calculate av Ne from data
redated_ts = tsdate.inside_outside(ts, mutation_rate=mutation_rate, population_size=est_pop_size)
unconstr_times = [nd.metadata.get("mn", nd.time) for nd in redated_ts.nodes()]
fig, ax = plt.subplots(1, 1, figsize=(5, 5))
title = "inside_outside method; prior based on fixed Ne"
plot_real_vs_tsdate_times(ax, ts.nodes_time, unconstr_times, ts, redated_ts, alpha=0.1, title=title)
```

If you cannot use the `variational_gamma` method, 
the discrete time methods also allow `population_size` to be either
a single number, specifying the "effective population size",
or a piecewise constant function of time, specifying a set of fixed population sizes
over a number of contiguous time intervals. Functions of this sort are captured by the
{class}`~demography.PopulationSizeHistory` class: see the {ref}`sec_popsize` page
for its use and interpretation.

### Estimating Ne for parameter specification

In the example above, in the absence of an expected effective population size for use in the
`inside_outside` method, we used a value approximated from the data. The standard way to do so
is to use the (sitewise) genetic diversity divided by four-times the mutation rate:

```{code-cell} ipython3
print("A rough estimate of the effective population size is", ts.diversity() / (4 * 1e-6))
```


<!--

## Setting variable population sizes

The {class}`demography.PopulationSizeHistory` class can be used to define a population size that
changes in a piecewise-constant manner over time (that is, the population size is constant between
specified time intervals). This can them be used to create a prior, via the {func}`build_prior_grid`
function (see {ref}`sec_priors`).

For example, the following code defines a population that is of effective size
1 million in the last 50,000 generations, only two hundred for a period of 10 generations 50,000 generations ago, then
of size 10,000 for all generations before that, exactly matching the simulated bottleneck

```{code-cell} ipython3
popsize = tsdate.demography.PopulationSizeHistory(population_size=[1e6, 2e2, 1e4], time_breaks=[50_000, 50_010])
```

We can then use this to create a prior for dating, rather than specifying a constant population size. This
gives a much better fit to the true times:

```{code-cell} ipython3
prior = tsdate.build_prior_grid(ts, popsize)
redated_ts = tsdate.inside_outside(ts, mutation_rate=mutation_rate, priors=prior)
fig, ax = plt.subplots(1, 1, figsize=(15, 3))
plot_real_vs_tsdate_times(ax, ts.nodes_time, redated_ts.nodes_time, ts, redated_ts, alpha=0.1)
```

## Estimating population size

If you don't know the population size, it is possible to use _tsdate_ to
*estimate* changes in population size over time, by first estimating the rate
of coalescence in different time intervals, and then re-estimating the dates.
However, this approach has not been fully tested or documented.

If you are interested in doing this, see
[GitHub issue #237](https://github.com/tskit-dev/tsdate/issues/237#issuecomment-1785655708)
for an example.
-->
