"""Oracles for the jitted functions in ``tsdate.discrete``."""

import numpy as np


def logsumexp(X):
    alpha = -np.inf
    r = 0.0
    for x in X:
        if x != -np.inf:
            if x <= alpha:
                r += np.exp(x - alpha)
            else:
                r *= np.exp(alpha - x)
                r += 1.0
                alpha = x
    return -np.inf if r == 0 else np.log(r) + alpha
