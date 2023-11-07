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

The methods available for `tsdate` inference can be divided into _discrete-time_
and _continuous-time_ approaches. Discrete-time approaches define time grid based on
discrete timepoints, and assign a probability to each node being at
each timepoint. Continuous-time
approaches approximate the probability distribution of times by a continuous
mathematical function (e.g. a gamma distribution).

In tests, we find that the continuous-time `variational_gamma` approach is
the most accurate (but can suffer from numerical stability). The discrete-time
`inside_outside` approach is slightly less accurate, especially for older times,
but is more numerically robust, and the discrete-time `maximization` approach is
always stable but is the least accurate.

Changing the method is very simple:


```{code-cell} ipython3
import tskit
import tsdate

input_ts = tskit.load("data/basic_example.trees")
ts = tsdate.date(input_ts, method="variational_gamma", population_size=100, mutation_rate=1e-8)
```

Currently the default is `inside_outside`, but this may change in future releases.


(sec_methods_discrete_time)=

## Discrete-time

The `inside_outside` and `maximization` methods both implement discrete-time
algorithms. These have the following advantages and disadvantages:

Pros
: allows any shape for the distribution of times
: Currently require just a single upwards and downward pass through the edges

Cons
: Choice of grid timpoints is somewhat arbitrary (but reasonable defaults picked
    based on the conditional coalescent)
: Inferred times are imprecise due to discretization: a denser timegrid can increase
    precision, but also increases computational cost (quadratic with number of timepoints)
: In particular, the oldest nodes can suffer from poor dating, as time into the past
    is an unbounded value, but a single oldest timepoint must be chosen.

### Inside Outside vs Maximization

The `inside_outside` approach has been shown to perform better empirically, but
suffers from the theoretical problem of "loopy belief propagation". Occasionally
it also has issues with numerical stability, although this is commonly indicative
of pathological combinations of tree sequence topology and mutation patterns.
Issues like this are often caused by long regions of the genome that
have no mapped mutations (e.g. in the centromere), which can be removed by
{ref}`preprocessing<sec_usage_real_data_stability>`.

The `maximization` approach is slightly less accurate empirically,
and will not return true posteriors, but is theoretically robust and
additionally is always numerically stable.

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

Cons
: Assumes posterior times can be reasonably modelled by a gamma distribution
    (e.g. they are not bimodal)
: The "Expectation propagation" approach requires multiple rounds of iteration
    until convergence.
: Numerical stability issues are more common (but often indicate pathological
    of otherwise problematic tree sequences)

### The variational gamma method

The `variational_gamma` method approximates times by fitting a separate gamma
distribution for each node. Iteration is required to converge
to a stable solution.

We are in the process of writing a formal description of the algorithm, but in
summary, this approach uses an expectation propagation ("message passing")
approach to update the gamma distribution for each node based on the times of connected
nodes and the mutations on connected edges. Updates take the form of moment matching
against an iteratively refined approximation to the posterior, which makes this method
very fast.

The iterative expectation propagation should converge to a fixed
point that approximately minimizes KL divergence between the true posterior
distribution and the approximation {cite}`minka2001expectation`.
At the moment, when `method="variational_gamma"`,
a relatively large number of iterations is used (which testing indicates is
more than enough) but convergence is not formally checked.
A stopping criterion will be implemented in future releases.

Progress through iterations can be output using the progress bar:

```{code-cell} ipython3
ts = tsdate.date(
    input_ts,
    method="variational_gamma",
    progress=True,
    population_size=100,
    mutation_rate=1e-8)
```