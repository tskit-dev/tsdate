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

(sec_variable_popsize)=

# Variable population sizes

:::{note}
Functionality described on this page is preliminary and subject to change in the future. For this reason,
classes and methods may not form part of the publicly available API and may not be fully documented yet.
:::

The population size determines the prior probability of coalescence events over time.
When setting the `tsdate` prior, a reasonable first order approximation is to assume a constant
population size. However, better results will be gained if the prior more accurately reflects
the true history. To illustrate this, we will generate data from a population that was large
in recent history, but had a small bottleneck of only 200 individuals
50,000 generations ago, with a medium-sized population before that:

```{code-cell} ipython3
:tags: [hide-input]
from matplotlib import pyplot as plt
import numpy as np
def plot_real_vs_tsdate_times(x, y, ts_x=None, ts_y=None, delta = 0.1, **kwargs):
    plt.scatter(x + delta, y + delta, **kwargs)
    plt.xscale('log')
    plt.xlabel(f'Real time' + ('' if ts_x is None else f' ({ts_x.time_units})'))
    plt.yscale('log')
    plt.ylabel(f'Estimated time from tsdate'+ ('' if ts_y is None else f' ({ts_y.time_units})'))
    line_pts = np.logspace(np.log10(delta), np.log10(x.max()), 10)
    plt.plot(line_pts, line_pts, linestyle=":")
```

```{code-cell} ipython3
import msprime

demography = msprime.Demography()
demography.add_population(name="A", initial_size=1e6)
demography.add_population_parameters_change(time=50000, initial_size=2e2)
demography.add_population_parameters_change(time=50010, initial_size=1e4)

mutation_rate = 1e-8
# Simulate a tree sequence with a population size history.
ts = msprime.sim_ancestry(
    10, sequence_length=1e5, recombination_rate=1e-8, demography=demography, random_seed=123)
ts = msprime.sim_mutations(ts, rate=mutation_rate, random_seed=123)
```

If we redate this tree sequence with a constant population size of (say) the ancestral size of 10,000, we
get a poor fit to the true times:

```{code-cell} ipython3
import tsdate
redated_ts = tsdate.date(ts, mutation_rate, population_size=1e4)
plot_real_vs_tsdate_times(ts.nodes_time, redated_ts.nodes_time, ts, redated_ts, delta=1000, alpha=0.1)
```

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
redated_ts = tsdate.date(ts, mutation_rate, priors=prior)
plot_real_vs_tsdate_times(ts.nodes_time, redated_ts.nodes_time, ts, redated_ts, delta=1000, alpha=0.1)
```

## Estimating population size

If you don't know the population size, it is possible to use `tsdate` to
*estimate* changes in population size over time, by first estimating the rate
of coalescence in different time intervals, and then re-estimating the dates.
However, this approach has not been fully tested or documented.

If you are interested in doing this, see
[GitHub issue #237](https://github.com/tskit-dev/tsdate/issues/237#issuecomment-1785655708)
for an example.
