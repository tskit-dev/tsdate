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

(sec_priors)=

# Specifying a Prior

## Basic usage

The {func}`build_prior_grid` and {func}`build_parameter_grid` functions allow you to create a bespoke prior
for the {ref}`sec_methods_discrete_time` and {ref}`sec_methods_continuous_time`, respectively.
This can be passed in to {func}`date` using the `priors` argument. It provides
a tuneable alternative to passing the {ref}`population size<sec_usage_popsize>`
directly to the {func}`date` function.

Along with adjusting the {ref}`method<sec_methods>`,
this is the recommended way to carry out more sophisticated
analyses, tweaking the `tsdate` algorithm to alter its runtime and accuracy.

```{code-cell} ipython3
import tskit
import tsdate
ts = tskit.load("data/basic_example.trees")
mu = 1e-8  # mutation rate
N = 100  # effective population size

prior1 = tsdate.build_prior_grid(ts, population_size=N)
prior2 = tsdate.build_prior_grid(ts, population_size=N, timepoints=40)
prior3 = tsdate.build_prior_grid(ts, population_size=N, prior_distribution="gamma")

ts1 = tsdate.date(ts, mu, priors=prior1)  # Identical to tsdate.date(ts, mu, population_size=N)
ts2 = tsdate.date(ts, mu, priors=prior2)  # Uses a denser timegrid
ts3 = tsdate.date(ts, mu, priors=prior3)  # Approximates the discrete-time priors with a gamma
```

See below for more explanation of the interpretation of the parameters passed to
{func}`build_prior_grid`.

## Prior shape

For {ref}`sec_methods_discrete_time` methods, it is possible to switch from the
(default) lognormal approximation to a gamma distribution, used when building a
mixture prior for nodes that have variable numbers of children in different
genomic regions. The discretised prior is then constructed by evaluating the
lognormal (or gamma) distribution across a grid of fixed times. Tests have shown that the
lognormal is usually a better fit to the true prior in most cases.

For {ref}`sec_methods_continuous_time` methods, only the gamma distribution is available.

(sec_priors_timegrid)=

## Setting the timegrid

For {ref}`sec_methods_discrete_time` methods, a grid of timepoints is created. An explicit
timegrid can be passed via the `timepoints` parameter. Alternatively, if a single integer is
passed, a nonuniform time grid will be chosen based on quantiles of the
lognormal (or gamma) approximation of the mixture prior.

Note that if an integer is given this is *not* the number of timepoints, but the number of
quantiles used as a basis for generating timepoints.  The actual number of timepoints will
be larger than this number. For instance

```{code-cell} ipython3
timepoints = 10
prior = tsdate.build_prior_grid(ts, population_size=N, timepoints=timepoints)
dated_ts = tsdate.date(ts, mu, priors=prior)

print(
    f"`timepoints`={timepoints}, producing a total of {len(prior.timepoints)}",
    "timepoints in the timegrid, at these times"
)
print(prior.timepoints)
```

For {ref}`sec_methods_continuous_time` methods, a grid of variational parameters is
created (e.g. shape and rate parameters of gamma distributions for each node), which
may be modified manually. 
Currently, node-specific priors are combined to generate a global i.i.d. prior
(although this behaviour will be changed in future releases to provide more
flexibility.)


(sec_priors_conditional_coalescent)=

## The conditional coalescent

Currently, priors are based on the [conditional coalescent](http://dx.doi.org/10.1006/tpbi.1998.1411).
Specifically, in a tree sequence of `s` samples, the distribution of times for a node that always
has `n` descendant samples is taken from the theoretical distribution of times
for a node with `n` descendant tips averaged over all coalescent trees of `s` total
tips (note that this assumes that all the samples are at time 0)

In most tree sequences, a node will not always have the same number of
descendant samples in all regions of the genome. For such nodes, an approximate prior
is constructed by averaging the mean and variance of the times, and then constructing
a mixture prior based on a lognormal (default) or gamma distribution with the same mean
and variance, weighted by the span of the node in each local tree.

It is unclear how well this approximation works in practice, as there are clear
correlations between the span of a node and the number of children it has. Testing
indicates that using a single prior for all nodes may provide better accuracy; this
may also be a result of too strongly constraining each node prior to a fixed topology
before dating. Currently the `variational_gamma` method using a single ("global") prior
for all nodes. The best prior to use for different methods is a current topic of 
investigation, and may be subject to change in future versions of `tsdate`.
