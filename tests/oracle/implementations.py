"""
Registry of implementations to test against the oracles.

``IMPLEMENTATIONS`` maps a name to a namespace with one attribute per tsdate module
(``hypergeo``, ``approx``, ``variational``, ``rescaling``, ``phasing``, ``util``,
``prior``, ``discrete``), exposing the jitted functions under the same names and
signatures as the oracle modules. To add the C extension, add an entry here; to
drop numba, delete the "numba" entry.
"""

import types

import tsdate.approx
import tsdate.hypergeo
import tsdate.phasing
import tsdate.prior
import tsdate.rescaling
import tsdate.util
import tsdate.variational
from tsdate.discrete import LogLikelihoods

_EP = tsdate.variational.ExpectationPropagation

# Functions that are staticmethods of ExpectationPropagation, exposed as plain names.
_EP_STATIC = (
    "_check_valid_constraints",
    "propagate_likelihood",
    "propagate_prior",
    "propagate_mutations",
)


def _numba_variational():
    """
    Namespace with ``EPFactors`` and the module-level helpers of
    ``tsdate.variational`` plus the jitted staticmethods of
    ``ExpectationPropagation`` as module-level names. Accessing a staticmethod on
    the class returns the underlying dispatcher.
    """
    ns = types.SimpleNamespace(
        EPFactors=tsdate.variational.EPFactors,
        _rescale_factors=tsdate.variational._rescale_factors,
        _assemble_factors=tsdate.variational._assemble_factors,
        _damp=tsdate.variational._damp,
        _rescale=tsdate.variational._rescale,
    )
    for name in _EP_STATIC:
        setattr(ns, name, getattr(_EP, name))
    return ns


def _numba_discrete():
    """
    ``logsumexp`` is a jitted staticmethod of ``LogLikelihoods``; accessing it on
    the class gives a callable ``logsumexp(X)``, so it is exposed under that name.
    """
    return types.SimpleNamespace(logsumexp=LogLikelihoods.logsumexp)


def _numba_namespace():
    return types.SimpleNamespace(
        hypergeo=tsdate.hypergeo,
        approx=tsdate.approx,
        variational=_numba_variational(),
        rescaling=tsdate.rescaling,
        phasing=tsdate.phasing,
        util=tsdate.util,
        prior=tsdate.prior,
        discrete=_numba_discrete(),
    )


IMPLEMENTATIONS: dict[str, types.SimpleNamespace] = {
    "numba": _numba_namespace(),
}
