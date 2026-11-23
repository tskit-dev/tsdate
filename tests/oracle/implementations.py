"""
Registry of implementations to test against the oracles.

``IMPLEMENTATIONS`` maps a name to a namespace with one attribute per tsdate module
(``hypergeo``, ``approx``, ``variational``, ``rescaling``, ``phasing``, ``util``,
``prior``, ``discrete``), exposing the jitted functions under the same names and
signatures as the oracle modules. To drop numba, delete the "numba" entry.

The "c" entry exists if the ``_tsdate`` extension can be imported. Its namespaces
have the attributes of the functions that have been ported (as thin adapters where the
oracle's signature or return type differs from the module's); other names skip the test
that uses them.
"""

import types

import pytest

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


class _PortedNamespace(types.SimpleNamespace):
    """
    A namespace of the functions ported to C: using a name that has not been ported yet
    skips the test. Add ported functions as attributes in `_c_namespace`.
    """

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        pytest.skip("not yet ported to C")


def _c_hypergeo(_tsdate):
    return _PortedNamespace(
        _digamma=_tsdate.digamma,
        _trigamma=_tsdate.trigamma,
        _betaln=_tsdate.betaln,
        _gammainc_der=_tsdate.gammainc_der,
        _gammainc_inv=_tsdate.gammainc_inv,
        _hyperu_laplace=_tsdate.hyperu_laplace,
        _hyp1f1_laplace=_tsdate.hyp1f1_laplace,
        _hyp2f1_laplace=_tsdate.hyp2f1_laplace,
    )


def _c_namespace():
    import _tsdate

    return types.SimpleNamespace(
        hypergeo=_c_hypergeo(_tsdate),
        approx=_PortedNamespace(),
        variational=_PortedNamespace(),
        rescaling=_PortedNamespace(),
        phasing=_PortedNamespace(),
        util=_PortedNamespace(),
        prior=_PortedNamespace(),
        discrete=_PortedNamespace(),
    )


IMPLEMENTATIONS: dict[str, types.SimpleNamespace] = {
    "numba": _numba_namespace(),
}

try:
    IMPLEMENTATIONS["c"] = _c_namespace()
except ImportError:  # the extension has not been built
    pass
