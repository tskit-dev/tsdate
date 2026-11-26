"""
Oracles for the prior functions (``lib/prior.c``).
"""

import numpy as np

# NumPy's SIMD log and exp can differ from libm in the last bit on some CPUs; `_libm`
# gives the C library's results.
from ._libm import exp, log


def _marginalize_over_ancestors(val):
    """
    Integrate an expectation over counts of extant ancestors. In a tree with
    "n" tips, the probability that there are "a" extent ancestors when a
    subtree of size "k" coalesces is hypergeometric-ish (Wuif & Donnelly 1998),
    and may be calculated recursively over increasing "a" and decreasing "k"
    (e.g. using recursive relationships for binomial coefficients).
    """
    n, N = val.shape  # number of tips, number of moments
    pr_a_ln = [np.nan, np.nan, 0.0]  # log Pr(a | k, n)
    out = np.zeros((n + 1, N))
    for k in range(n - 1, 1, -1):
        # log(0) at k == 2 is -inf; the value is unused for k == 2
        const = log(n - k) + log(k - 2) - log(k + 1)
        for a in range(2, n - k + 2):
            out[k] += exp(pr_a_ln[a]) * val[a]
            if k > 2:  # Pr(a | k, n) to Pr(a | k - 1, n)
                pr_a_ln[a] += const - log(n - a - k + 2)
        if k > 2:  # Pr(n - k + 1 | k - 1, n) to Pr(n - k + 2 | k - 1, n)
            pr_a_ln.append(pr_a_ln[-1] + log(n - k + 2) - log(k + 1) - const)
    out[n] = val[1]
    return out


def conditional_coalescent_variance(num_tips):
    """
    Variance of node age conditional on the number of descendant leaves, under
    the standard coalescent. Returns array indexed by number of descendant
    leaves.
    """

    coal_rates = np.array(
        [2 / (i * (i - 1)) if i > 1 else 0.0 for i in range(1, num_tips + 1)]
    )

    # hypoexponential mean and variance; e.g. conditional on the number of
    # extant ancestors when the node coalesces, the expected time of
    # coalescence is the sum of exponential RVs (Wuif and Donnelly 1998)
    mean = coal_rates.copy()
    variance = coal_rates.copy() ** 2
    for i in range(coal_rates.size - 2, 0, -1):
        mean[i] += mean[i + 1]
        variance[i] += variance[i + 1]

    # marginalize over number of ancestors using recursive algorithm
    moments = _marginalize_over_ancestors(np.stack((mean, variance + mean**2), 1))

    return moments[:, 1] - moments[:, 0] ** 2
