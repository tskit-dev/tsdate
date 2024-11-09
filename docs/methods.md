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

(sec_methods)=

# Methods

The methods available for _tsdate_ inference can be divided into  *continuous-time*
and *discrete-time*  approaches. 
Both approaches iteratively propagate information between nodes to
construct an approximation of the marginal posterior distribution for the
age of each node, given the mutational information in the tree sequence.
Discrete-time approaches approximate the posterior across a grid of discrete
timepoints (e.g. assign a probability to each node being at each timepoint),
and propagate exact information about the probability of each node being in
these states ("belief propagation"). Continuous-time approaches approximate
the posterior by a continuous univariate distribution (e.g. a gamma distribution),
and propagate summaries of these probabilities between nodes ("expectation
propagation").

In tests, we find that the continuous-time `variational_gamma` approach is the
most accurate.  The discrete-time `inside_outside` approach is slightly less
accurate, especially for older times, but is slightly more numerically robust
and also allows each node to have an arbitrary (discretised) probability distribution.
The discrete-time `maximization` approach is always stable but is the least
accurate.

Changing the method is very simple:

```{code-cell} ipython3
import tskit
import tsdate

input_ts = tskit.load("data/basic_example.trees")
ts = tsdate.date(input_ts, method="inside_outside", mutation_rate=1e-8, population_size=1000)
```

Alternatively each method can be called directly as a separate function:

```{code-cell} ipython3
ts = tsdate.inside_outside(input_ts, mutation_rate=1e-8, population_size=1000)
```

The available method names and functions are also available via the
{data}`tsdate.core.estimation_methods` variable.

(sec_methods_continuous_time)=

## Continuous-time

The only continuous-time algorithm currently implemented is the `variational_gamma`
method.

Pros
: Time estimates are precise, and not affected by choice of timepoints.
: No need to define (possibly arbitrary) timegrid, and no quadratic scaling
    with number of timepoints
: Old nodes do not suffer from time-discretisation issues caused by forcing
    bounds on the oldest times
: Iterative updating properly accounts for cycles in the genealogy
: No need to specify node-specific priors; a mixture "prior" (fit by
    expectation-maximization) is used to regularise the roots, and the prior
    on non-root nodes is only informed via their connection to roots. This
    means that the posterior node dates are not strongly influenced by any
    prior model such as the coalescent, and should therefore be robust to
    changes in underlying demography or selection.
: Can account for variable population sizes using rescaling

Cons
: Assumes posterior times can be reasonably modelled by gamma distributions
    (e.g. they are not bimodal)
: The "expectation propagation" algorithm used to fit the posterior requires
    multiple rounds of iteration until convergence.
: Numerical stability issues are more common (but often indicate pathological
    of otherwise problematic tree sequences)

(sec_methods_continuous_time_vgamma)=

### The variational gamma method

The `variational_gamma` method approximates times by fitting separate gamma
distributions for each node, in a similar spirit to {cite:t}`schweiger2023ultra`.
The directed graph that represents the genealogy can (in its undirected form) contain
cycles, so a technique called "expectation propagation" (EP) is used, in which
local estimates to each gamma distribution are iteratively refined until
they converge to a stable solution.  This comes under a class of approaches
sometimes known as "loopy belief propagation".

(sec_methods_expectation_propagation)=
#### Expectation propagation

We are in the process of writing a formal description of the algorithm, but in
summary, this approach uses an expectation propagation ("message passing")
approach to update the gamma distribution for each node based on the ages of connected
nodes and the mutations on connected edges. Updates take the form of moment matching
against an iteratively refined approximation to the posterior, which makes this method
very fast.

The iterative expectation propagation should converge to a fixed
point that approximately minimizes Kullback-Leibler divergence between the true posterior
distribution and the approximation {cite}`minka2001expectation`.
At the moment a relatively large number of iterations are used (which testing indicates is
more than enough, but which can be also changed by using the `max_iterations` parameter);
however, convergence is not formally checked.
A stopping criterion will be implemented in future releases.

Progress through iterations can be output using the progress bar:

```{code-cell} ipython3
ts = tsdate.date(input_ts, mutation_rate=1e-8, progress=True)
```

(sec_rescaling)=
#### Rescaling

The `variational_gamma` method implements a further
process called *rescaling*, and which can help to deal with the effects of
variable population size though time. This is based on an algorithm introduced
by the ARG inference software
[SINGER](https://doi.org/10.1101/2024.03.16.585351) (Deng et al 2024) that
rescales node ages by matching observed and expected segregating sites within
time windows.
Basically, time is broken up into a number of intervals, and times within
intervals are simultaneously scaled such that the expected density of mutations
along each path from a sample to the root best matches the mutational density
predicted from the user-provided mutation rate. The number of intervals can be
specified using the `rescaling_intervals` parameter. If set to 0, no rescaling
is performed; this means that dates may be inaccurately estimated if the
dataset comes from a set of samples with a complex demographic history.
`tsdate` uses a modified version of Deng et al's algorithm that works on gamma
natural parameters rather than point estimates, and that is not biased by the
artefactual polytomies introduced by `tsinfer` for the sake of compression.

:::{todo}
Describe the rescaling step in more detail. Could also link to [the population size docs](sec_popsize)
:::


(sec_methods_discrete_time)=

## Discrete-time

The available discrete-time algorithms are the `inside_outside` and `maximization` methods.
For historical reasons, these approaches use an informative (node-specific) prior,
the [conditional coalescent prior](sec_priors_conditional_coalescent),
which means that you either need to provide them with an estimated effective
population size, or a [priors](sec_priors) object. Future improvements may
allow Empirical Bayes priors to be set in discrete time methods, and coalescent priors
to be set in continuous time methods.

The _tsdate_ discrete time methods have the following advantages and disadvantages:

Pros
: Methods allow any shape for the distributions of times
: Currently require just a single upwards and downward pass through the edges

Cons
: Choice of grid timepoints is somewhat arbitrary (but reasonable defaults are picked
    based on the conditional coalescent)
: Inferred times are imprecise due to discretization: a denser timegrid can increase
    precision, but also increases computational cost (quadratic with number of timepoints)
: In particular, the oldest/youngest nodes can suffer from poor dating, as time into the past
    is an unbounded value, but a single oldest/youngest timepoint must be chosen.
: Currently, discrete-time methods use a conditional coalescent prior, and so are more
    inaccurate where data deviates from patterns expected from the neutral coalescent.
    Moreover, we have found that weighting coalescent priors by node span introduces
    extra bias in the prior, due to a correlation between span and age.

### Inside Outside vs Maximization

The `inside_outside` approach has been shown to perform better empirically than the
`maximization` approach, but in theory it does not properly account for cycles in
the underlying genealogical network when updating posterior probabilities
(a potential solution would be to implement a "loopy belief propagation" algorithm
as in the continuous-time [`variational_gamma`](sec_methods_continuous_time_vgamma)
method, above). Occasionally the `inside_outside` method also
has issues with numerical stability, although this is commonly indicative
of pathological combinations of tree sequence topology and mutation patterns.
Problems like this are often caused by long regions of the genome that
have no mapped mutations (e.g. in the centromere), which can be removed by
{ref}`preprocessing<sec_usage_real_data_stability>`.

The `maximization` approach is slightly less accurate empirically,
and will not return true posteriors, but is theoretically robust and
additionally is always numerically stable.
