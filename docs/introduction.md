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

(sec_introduction)=

# Introduction

The _tsdate_ program {cite}`wohns2022unified` infers dates for nodes in a
genetic genealogy, sometimes loosely known as an ancestral recombination graph
or ARG {cite}`wong2023general`. More precisely, it takes a genealogy in 
[tree sequence](https://tskit.dev/tutorials/what_is.html) format as an input
and returns a copy of that tree sequence with altered node and mutation times. These
times have been estimated on the basis of the number of mutations
along the edges connecting genomes in the genealogy (i.e. using the "molecular clock").

## Technical details

Methodologically, the genealogy is treated as a interconnected graph, and a
Bayesian network approach is used to update the probability distribution
of times for each node, given the time distribution on connected nodes and
the mutations on connected edges. This results in a posterior distribution of
times (which can be {ref}`output separately<sec_usage_posterior>`). This
{ref}`scales well<sec_usage_real_data_scaling>` to large genetic genealogies.
The input tree sequence can come from any source: e.g. from simulation or from
a variety of inference programs, such as [tsinfer](https://tskit.dev/).

`Tsdate` provides several methods for assigning probabilities to different times,
and updating information through the genealogy. These include continuous-time
(default) and discrete-time methods, see {ref}`sec_methods` for more details.

The output of _tsdate_ is a new tree sequence with altered
{attr}`node<tskit:tskit.TreeSequence.nodes_time>`and
{attr}`mutation<tskit:tskit.TreeSequence.mutations_time>` times,
as well as extra node and mutation {ref}`sec_metadata`.
Optionally, a posterior distribution of node times can be generated
(see {ref}`sec_usage_posterior`).

Since the method is Bayesian, technically it requires each node to have a
prior distribution of times. The default `variational_gamma` method currently
uses an improper (flat) prior which does not need any user input. However,
the alternative discrete-time methods currently require
the prior to be explicitly provided, either via providing an estimated
effective population size (which is then used in the
[conditional coalescent](http://dx.doi.org/10.1006/tpbi.1998.1411)), or
{ref}`directly<sec_priors>`. 

The ultimate source of technical detail for _tsdate_ is the source code on our
[GitHub repository](http://github.com/tskit-dev/tsdate).

## Sources of genealogies

Input genealogies can come from any source,
but _tsdate_ is often coupled with [_tsinfer_](https://tskit.dev/software/tsinfer.html)
{cite}`kelleher2019inferring`, which estimates the tree sequence topology but
not the tree sequence node times.

```{code-cell} ipython3
:tags: [remove-input]
import msprime
import tsinfer
import tskit
import tsdate
import numpy as np

from IPython.display import HTML
ts = msprime.sim_ancestry(5, population_size=1000, recombination_rate=1e-7, sequence_length=1000, random_seed=2)
ts = msprime.sim_mutations(ts, rate=2e-6, random_seed=7)
inferred = tsinfer.infer(tsinfer.SampleData.from_tree_sequence(ts)).simplify()
svg1 = inferred.draw_svg(
    size=(260, 300), mutation_labels={}, node_labels={}, y_axis=True, y_ticks=[0, 1],
    style=".x-axis .ticks .lab {font-size: 0.7em} .y-axis .ticks .lab {text-anchor: middle; transform: rotate(-90deg) translateY(-10px)}",
    symbol_size=4,
    )
tables = tsdate.variational_gamma(inferred, mutation_rate=2e-6).dump_tables()
# remove the mutation times so they appear nicely spaced out
tables.mutations.time = np.full_like(tables.mutations.time, tskit.UNKNOWN_TIME)
dated_ts = tables.tree_sequence()
svg2 = dated_ts.draw_svg(
    size=(260, 300), mutation_labels={}, node_labels={}, y_axis=True,
        y_ticks=[0, 2000, 4000, 6000],
    style=".x-axis .ticks .lab {font-size: 0.7em} .y-axis .ticks .lab {text-anchor: middle; transform: rotate(-90deg) translateY(-10px)}",
    symbol_size=4,
)
haps = "<br>".join(ts.haplotypes())
cen = 'style="text-align: center; padding: 0.5em 0"'
HTML(f"""<table>
    <caption style="padding: 0 4em">An example of using <em>tsinfer</em> followed by <em>tsdate</em> on some DNA sequence data, illustrating  that <em>tsdate</em> sets a timescale and changes node times so that mutations (red crosses)
    are more evenly distributed over edges of the genealogy. The modified genealogy also shows
    an increase in recent coalescences, as expected from theory.</caption>
    <t style=""><td {cen}><div style="font-size: 0.6em">{haps}</div></td><td></td><td {cen} width="300">{svg1}</td><td></td><td {cen} width="300">{svg2}</td></tr>
    <tr style="font-size: 2em"><td {cen}">DNA sequence</td><td {cen}>→</td><td {cen}>tsinfer</td><td {cen}>→</td><td {cen}>tsdate</td></tr>
    </table>"""
)
```

Together, _tsdate_ and  _tsinfer_ scale to analyses of millions of genomes, the largest genomic datasets currently available.
