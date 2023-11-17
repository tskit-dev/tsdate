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

`tsdate` {cite}`wohns2022unified` infers dates for nodes in a genetic genealogy,
sometimes loosely known as an ancestral recombination graph or ARG
{cite}`wong2023general`. More precisely, it takes a genealogy in 
[tree sequence](https://tskit.dev/tutorials/what_is.html) format as an input
and returns a copy of that tree sequence with altered node times. These
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

As the approach is Bayesian, it requires a
{ref}`prior distribution<sec_priors>` to be defined
for each of the nodes to date. By default, `tsdate` calculates priors from the
[conditional coalescent](http://dx.doi.org/10.1006/tpbi.1998.1411), although
alternative prior distributions can also be specified.

`tsdate` provides several methods for assigning probabilities to different times,
and updating information through the genealogy. These include discrete-time and
continuous-time methods, see {ref}`sec_methods` for more details.

The output of tsdate is a new tree sequence with altered
{attr}`node times<tskit:tskit.TreeSequence.nodes_time>`, extra node {ref}`sec_metadata`, and
(optionally) a posterior distribution of node times
(see {ref}`sec_usage_posterior`).

## Sources of genealogies

The input genealogies to date can come from any source,
but `tsdate` is often coupled with [tsinfer](https://tskit.dev/software/tsinfer.html)
{cite}`kelleher2019inferring`, which estimates the tree sequence topology but
not the tree sequence node times.

```{code-cell} ipython3
:tags: [remove-input]
import msprime
import tsinfer
import tsdate

from IPython.display import HTML
ts = msprime.sim_ancestry(5, population_size=1000, recombination_rate=1e-7, sequence_length=1000, random_seed=321)
ts = msprime.sim_mutations(ts, rate=4e-6, random_seed=4321)
inferred = tsinfer.infer(tsinfer.SampleData.from_tree_sequence(ts)).simplify()
svg1 = inferred.draw_svg(
    size=(260, 300), mutation_labels={}, node_labels={}, y_axis=True, y_ticks=[0, 1],
    style=".x-axis .ticks .lab {font-size: 0.7em} .y-axis .ticks .lab {text-anchor: middle; transform: rotate(-90deg) translateY(-10px)}",
    symbol_size=4,
    )
svg2 = tsdate.date(inferred, population_size=1000, mutation_rate=4e-6).draw_svg(
    size=(260, 300), mutation_labels={}, node_labels={}, y_axis=True,
        y_ticks=[0, 1000, 2000, 3000],
    style=".x-axis .ticks .lab {font-size: 0.7em} .y-axis .ticks .lab {text-anchor: middle; transform: rotate(-90deg) translateY(-10px)}",
    symbol_size=4,
)
haps = "<br>".join(ts.haplotypes())
cen = 'style="text-align: center; padding: 0.5em 0"'
HTML(f"""<table>
    <caption style="padding: 0 4em">An example of using `tsinfer` followed by `tsdate` on some DNA sequence data.
    You can see that tsdate sets a timescale and changes node times so that mutations (red crosses)
    are more evenly distributed over edges of the genealogy. This results in more realistic local trees
    (with coalescences clustered, as expected from theory, at recent times)</caption>
    <t style=""><td {cen}><div style="font-size: 0.6em">{haps}</div></td><td></td><td {cen} width="300">{svg1}</td><td></td><td {cen} width="300">{svg2}</td></tr>
    <tr style="font-size: 2em"><td {cen}">DNA sequence</td><td {cen}>→</td><td {cen}>tsinfer</td><td {cen}>→</td><td {cen}>tsdate</td></tr>
    </table>"""
)
```

Together, `tsdate` and  `tsinfer` scale to analyses of millions of genomes, the largest genomic datasets currently available.
