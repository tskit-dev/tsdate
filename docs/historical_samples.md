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


(sec_historical_samples)=

# Historical (Ancient) Samples

Sometimes you may wish to infer and date a genetic genealogy from
data which includes *historical samples*,
whose time is older that the current generation (i.e. sample nodes with
times > 0).

The output of [`tsinfer`](https://tskit.dev/tsinfer/) is valid regardless
of the inclusion of historical samples, but *dating* such a tree sequence
is more complicated. This is because the time scale of a tsinferred
tree sequence is uncalibrated, so it is unclear where in time to
place any historical samples.

:::{note}
We are currently working on methods to make dating historical samples
much more convenient.
:::

## The 2 step approach

Currently, the best way to date such tree sequences is 
to perform a two step process, in  which inference and dating is first
performed on the modern samples in order to establish a timescale, followed by
adding the historical samples and re-inferring using the dated timescale as a
basis for site ages in the `tsinfer` algorithm. The re-inferred tree sequence
can then be redated. The following recipe shows how this is accomplished
with a few lines of Python. The only requirement is a tsinfer.SampleData file with
modern and historical samples (the latter are specified using the
`individuals_time` array in a tsinfer.SampleData file).

:::{note}
While `tsinfer` does not make assumptions about the
time of sample nodes, _tsdate_ requires a [prior](sec_priors)
which is calculated from the conditional coalescent on the assumption that all
samples are at time 0. Although this is not the case if there are historical samples,
testing shows that it is still a close enough approximation for the method to
work well. 
:::

```{code-cell} ipython3
import logging

import msprime
import tsinfer
import tsdate

Ne = 1e4
mu = 1e-7  # mutation rate

def make_historical_data(num_modern, num_historical, random_seed, historical_time = 10000):
    # Returns two SampleData files, one with only moderns and one with everyone
    samples = [
        msprime.SampleSet(num_modern),
        msprime.SampleSet(num_historical, time=historical_time)
    ]
    
    ts = msprime.sim_ancestry(
        samples=samples,
        population_size=Ne,
        recombination_rate=1e-7, 
        sequence_length=1e5,
        random_seed=random_seed
    )
    ts = msprime.sim_mutations(ts, rate=mu, random_seed=random_seed)
    ## Make the SampleData files directly from the tree sequence rather than going via VCF
    all_samples = tsinfer.SampleData.from_tree_sequence(
        ts, use_sites_time=True, use_individuals_time=True)
    modern_samples = tsinfer.SampleData.from_tree_sequence(
        ts.simplify(ts.samples(time=0), filter_sites=False))

    # NB: if importing from a VCF you would create the modern_samples by subsetting all_samples:
    # modern_samples = all_samples.subset(np.where(all_samples.individuals_time[:] == 0)[0])
    return modern_samples, all_samples

def infer_and_date_modern_and_ancient(modern_samples, all_samples):
    # Date the moderns
    inferred_ts = tsinfer.infer(modern_samples)

    tsdate_logger = logging.getLogger('tsdate')  # temporarily disable the warning
    tsdate_logger.setLevel(logging.CRITICAL)
    inferred_ts = tsdate.preprocess_ts(inferred_ts)
    tsdate_logger.setLevel(logging.WARNING)

    dated_modern_ts = tsdate.date(inferred_ts,  mutation_rate=mu)
    
    # Find the dates for each site and use those rather than frequency
    sites_time = tsdate.sites_time_from_ts(dated_modern_ts)  # Get tsdate site age estimates
    dated_samples = tsdate.add_sampledata_times(all_samples, sites_time)
    ancestors = tsinfer.generate_ancestors(dated_samples)
    ancestors_w_proxy = ancestors.insert_proxy_samples(
        dated_samples, allow_mutation=True)
    ancestors_ts = tsinfer.match_ancestors(dated_samples, ancestors_w_proxy)
    return tsinfer.match_samples(dated_samples, ancestors_ts, force_sample_times=True)

dated_ts = infer_and_date_modern_and_ancient(*make_historical_data(2, 1, random_seed=1234))
```

In the code above, we simulate a tree sequence with six sample chromosomes, four modern and
two historical. We then infer and date a tree sequence using only the modern
samples. Next, we find derived alleles which are carried by the historical samples and
use the age of the historical samples to constrain the ages of these alleles. Finally,
we reinfer the tree sequence, using the date estimates from _tsdate_ and the historical 
constraints rather than the frequency of the alleles to order mutations in `tsinfer`.
Historical samples are added to the ancestors tree sequence as
{meth}`proxy nodes, in addition to being used as samples<tsinfer:tsinfer.AncestorData.insert_proxy_samples>`.