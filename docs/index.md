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

(sec_welcome)=

# Welcome to _tsdate_

This is the documentation for _tsdate_, a method for efficiently inferring the
age of ancestors in a genetic genealogy or "[ARG](https://tskit.dev/tutorials/args.html)".

Basic usage is as simple as running the following python command

```{code-cell} ipython3
:tags: [remove-cell]
import tskit
input_ts = tskit.load("data/basic_example.trees")
```

```{code-cell} ipython3
import tsdate
output_ts = tsdate.date(input_ts, mutation_rate=1e-8)
```

The rest of this documentation is organised into the following sections:

```{tableofcontents}
```

## Source code

_Tsdate_ is open source software released under the liberal MIT licence. The code is
freely available on [GitHub](https://github.com/tskit-dev/tsdate).
Bug reports and suggestions for improvements can be made by opening an issue on that repository:
we suggest also checking the [discussions list](https://github.com/tskit-dev/tsdate/discussions).
Pull requests are welcome: we largely follow the
[tskit development workflow](https://tskit.dev/tskit/docs/latest/development.html#workflow).

## Citing
To cite tsdate, please consult the citation manual at: [https://tskit.dev/tsdate/docs/stable/citation](https://tskit.dev/tsdate/docs/stable/citation)
