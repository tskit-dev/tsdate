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

(sec_real_data)=

# Real data

Real world data is likely to consist of larger datasets than in the {ref}`sec_usage`
examples, and may exhibit issues that are not present in simulated data such as
{ref}`numerical instability<sec_real_data_numerical_stability>` and other problems.

In particular, two issues can have a large influence when dating tree sequences
inferred by tools such as {ref}`tsinfer<tsinfer:sec_introduction>`:

1. An inferred tree sequence may have too many edge constraints, for instance
if multiple ancestors in the true genealogy are wrongly combined into a
single node in the inferred tree sequence, or if a parent-child relationship
(i.e. an edge) is wrongly inferred between two nodes.
2. Inferred tree sequences may contain *polytomies* (multifurcations) indicating
uncertainty about the order of branching in the original genealogy. In this case, 
it is possible to choose the polytomy date to represent either the oldest of the
contributing branch points, or the expected pairwise tMRCA between the lineages
contributing to the polytomy. The first provides a more accurate estimate of
both mutation dates and the date of nodes under each mutation. The second provides
a better estimate of coalescence times and edge areas / total branch lengths.

Issue 1 can be ameliorated by fully simplifying the tree sequence, and breaking up
nodes (e.g. by splitting nodes into separate ancestors: see
{ref}`sec_real_data_preprocessing`). Using the default `variational_gamma` method,
issue 2 can be addressed during the rescaling step by explicitly setting `match_segregating_sites` 
to either `True` or `False` (see {ref}`sec_real_data_rescaling` below)

(sec_real_data_preprocessing)=

## Preprocessing

For real datasets, you are advised to run {func}`preprocess_ts` on your inferred tree
sequence before dating. This removes regions with no variable sites, simplifies the
tree sequence to remove locally unary portions of nodes, and splits disjoint nodes into
separate datable nodes. Not only can this improve the accuracy of dating, but it
also increases {ref}`sec_real_data_numerical_stability`. See the documentation for
{func}`tsdate.preprocess_ts()` for details on how to increase or decrease its stringency.

In particular, the `tsinfer` algorithm can overestimate the span of genome
covered by some nodes, especially those at ancient times. This is particularly
seen when nodes contain "locally unary" sections (i.e. where the node only has
one child in some trees), or where a node is "disjoint", disappearing from one tree
to reappear in a tree in a more distant region of the genome. This extended node
span can cause problems by tying together distant parts of the genealogy. Preprocessing
removes locally unary sections of nodes via simplification, and splits disjoint nodes
using the {func}`tsdate.util.split_disjoint_nodes` functionality.

:::{note}
If unary regions are *correctly* estimated, they can help improve dating slightly.
You can set the `allow_unary=True` option to run tsdate on such tree sequences.
:::

## Continuous time considerations

In general, when analysing real data with _tsdate_, we recommended using
the default {func}`tsdate.variational_gamma` {ref}`method<sec_methods>`. This
has several parameters which can be adjusted. The `max_iterations` parameter simply
adjusts the number of rounds of {ref}`sec_methods_expectation_propagation`, and
is usually sufficiently large to give reasonable results on real data. The
other parameters concern rescaling.

(sec_real_data_rescaling)=

### Rescaling details

`Tsdate` uses a modified version of the rescaling algorithm introduced in
[SINGER](https://github.com/popgenmethods/SINGER) {cite}`deng2024robust`
that works on gamma natural parameters rather than point estimates.
Rescaling can help to deal with the effects of variable population size
though time (see {ref}`sec_popsize`). Currently, three parameters can be set:

* `match_segregating_sites`. This has the largest effect if tree sequences contain
  polytomies.
  1. When explicitly set to `True`, times within intervals are scaled such that the
     average mutational distance between pairs of samples best matches the mutation rate.
     This is similar to the approach in SINGER, and is appropriate if, for example,
     you are interested in estimating coalescence rates over time, or average
     divergent times (MRCAs) between samples. In the resulting dated tree sequence,
     polytomy dates will correspond to the average distance between the pairs of
     lineages contributing to the polytomy.

  2. By default, `match_segregating_sites=False`, meaning that times within intervals
     are simultaneously scaled such that the expected density of mutations along each
     path from a sample to the root best matches the mutational density predicted
     from the user-provided mutation rate. This is most suitable if you wish to estimate
     dates of mutations or dates of identifiable common ancestors. In the resulting
     dated tree sequence, polytomy dates will correspond to the oldest ancestor that
     contributes to the polytomy.

* `rescaling_intervals`. This sets the number of time intervals over which rescaling
  takes place. To turn off rescaling entirely, a value of `0` can be provided;
  however, resulting dates may be less accurately estimated if the dataset comes
  from a set of samples with a complex demographic history (see {ref}`sec_popsize`)

* `rescaling_iterations`: several rounds of rescaling are carried out. Normally very
  few iterations are required to reach a stable set of rescaled intervals.

:::{todo}
Describe the rescaling step in more detail. Could also link to [the population size docs](sec_popsize)
:::


## Discrete time considerations

A few parameters can be set to speed up discrete-time methods.

:::{note}
For {ref}`sec_methods_discrete_time` methods, _tsdate_ scales
quadratically in the number of time slices. To increase speed or temporal resolution,
you are thus advised to keep with the {ref}`sec_methods_continuous_time`
`variational_gamma` method.
:::

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

Regularly reused mutational likelihoods are precalculated in the dicrete-time methods.
This precalculation can be parallelised by specifying the `num_threads` parameter to 
{func}`tsdate.inside_outside` and {func}`tsdate.maximization`. However, this
behaviour is subject to change in future versions.

(sec_real_data_numerical_stability)=

## Numerical stability

When passing messages between nodes, it is possible that the node time updates are
wildly incompatible with each other, for instance if a focal node is simultaneoulsy
attached to one small edge with many mutations and another large edge with few
mutations. The `variational_gamma` method incorporates a form of "damping" which
encourages gradual convergence to reasonable node times, but it may still be the case
that the tree sequence topology contains, for example, long deep branches with very few mutations, such as samples attaching directly to a local root.

Such "bad" tree sequences (caused by pathological combinations of topologies and
mutations) can result in _tsdate_ raising an error when dating. Issues of this
nature are collected in
[this set of GitHub issues](https://github.com/tskit-dev/tsdate/issues?q=label%3A%22numerical%20stability%22).
They can often be fixed by removing bad regions of the tree sequence,
e.g. regions that have no variation because they are unmappable or have been
removed by a QC filter, such as at the centromere. {ref}`sec_real_data_preprocessing`
the tree sequence can remove such regions, as well
as cutting ties between nodes by removing locally unary regions and
splitting disjoint nodes, which can also cause stability problems.

If numerical issues still persist, this is likely to be a sign that the
tree sequence topology has been poorly inferred, and you are encouraged
to examine it in detail before proceeding. Running the
{func}`tsdate.maximization` method should always work, but may not give
accurate results.

It is also possible for _tsdate_ to have issues when
{ref}`rescaling <sec_real_data_rescaling>`, e.g. if there is not enough
information within a rescaling interval. Setting the `rescaling_intervals`
parameter to a smaller value, or omitting rescaling entirely, should
allow _tsdate_ to run to completion.