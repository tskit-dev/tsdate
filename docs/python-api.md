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

(sec_python_api)=

# Python API

This page provides formal documentation for the _tsdate_ Python API.


## Running tsdate

```{eval-rst}
.. autofunction:: tsdate.date
.. autodata:: tsdate.core.estimation_methods
   :no-value:
.. autofunction:: tsdate.variational_gamma
.. autofunction:: tsdate.inside_outside
.. autofunction:: tsdate.maximization
```

## Prior and Time Discretisation Options

```{eval-rst}
.. autofunction:: tsdate.build_prior_grid
.. autofunction:: tsdate.build_parameter_grid
.. autoclass:: tsdate.base.NodeGridValues
.. autodata:: tsdate.base.DEFAULT_APPROX_PRIOR_SIZE
```

## Variable population sizes

```{eval-rst}
.. autoclass:: tsdate.demography.PopulationSizeHistory
```

## Preprocessing Tree Sequences

```{eval-rst}
.. autofunction:: tsdate.preprocess_ts
.. autofunction:: tsdate.util.split_disjoint_nodes
```

# Functions for Inferring Tree Sequences with Historical Samples

```{eval-rst}
.. autofunction:: tsdate.sites_time_from_ts
.. autofunction:: tsdate.add_sampledata_times
```

# Constants

```{eval-rst}
.. automodule:: tsdate
   :members:
```
