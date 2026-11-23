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

import numpy as np
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


def _c_valid(func):
    """The `_valid_*` checks return a bool in numba; the C function a float."""

    def valid(*args):
        return bool(func(*args))

    return valid


def _c_projection(func):
    """
    Adapter for a projection. The C function takes flat arguments (the ages, and the
    entries of the parameter arrays) and returns a flat tuple: the first result, then
    the pairs of natural parameters; the oracle takes arrays and returns the first
    result and one array per pair.
    """

    def projection(*args):
        flat = [float(x) for arg in args for x in np.atleast_1d(arg)]
        first, *rest = func(*flat)
        pairs = [np.array(rest[i : i + 2]) for i in range(0, len(rest), 2)]
        return (first, *pairs)

    return projection


def _c_approx(_tsdate):
    # The helpers; each commented section is owned by the agent that ports it.
    return _PortedNamespace(
        approximate_log_moments=_tsdate.approximate_log_moments,
        approximate_gamma_kl=_tsdate.approximate_gamma_kl,
        approximate_gamma_mom=_tsdate.approximate_gamma_mom,
        approximate_gamma_iqr=_tsdate.approximate_gamma_iqr,
        average_gammas=_tsdate.average_gammas,
        _valid_moments=_c_valid(_tsdate.valid_moments),
        _valid_gamma=_c_valid(_tsdate.valid_gamma),
        _valid_hyp1f1=_c_valid(_tsdate.valid_hyp1f1),
        _valid_hyperu=_c_valid(_tsdate.valid_hyperu),
        _valid_hyp2f1=_c_valid(_tsdate.valid_hyp2f1),
        # approx: node moments
        moments=_tsdate.moments,
        rootward_moments=_tsdate.rootward_moments,
        leafward_moments=_tsdate.leafward_moments,
        unphased_moments=_tsdate.unphased_moments,
        twin_moments=_tsdate.twin_moments,
        sideways_moments=_tsdate.sideways_moments,
        # approx: node projections
        gamma_projection=_c_projection(_tsdate.gamma_projection),
        leafward_projection=_c_projection(_tsdate.leafward_projection),
        rootward_projection=_c_projection(_tsdate.rootward_projection),
        unphased_projection=_c_projection(_tsdate.unphased_projection),
        twin_projection=_c_projection(_tsdate.twin_projection),
        sideways_projection=_c_projection(_tsdate.sideways_projection),
        # approx: mutation moments
        mutation_moments=_tsdate.mutation_moments,
        mutation_rootward_moments=_tsdate.mutation_rootward_moments,
        mutation_leafward_moments=_tsdate.mutation_leafward_moments,
        mutation_unphased_moments=_tsdate.mutation_unphased_moments,
        mutation_twin_moments=_tsdate.mutation_twin_moments,
        mutation_sideways_moments=_tsdate.mutation_sideways_moments,
        mutation_edge_moments=_tsdate.mutation_edge_moments,
        mutation_block_moments=_tsdate.mutation_block_moments,
        # approx: mutation projections
        mutation_gamma_projection=_c_projection(_tsdate.mutation_gamma_projection),
        mutation_leafward_projection=_c_projection(_tsdate.mutation_leafward_projection),
        mutation_rootward_projection=_c_projection(_tsdate.mutation_rootward_projection),
        mutation_edge_projection=_c_projection(_tsdate.mutation_edge_projection),
        mutation_unphased_projection=_c_projection(_tsdate.mutation_unphased_projection),
        mutation_twin_projection=_c_projection(_tsdate.mutation_twin_projection),
        mutation_sideways_projection=_c_projection(_tsdate.mutation_sideways_projection),
        mutation_block_projection=_c_projection(_tsdate.mutation_block_projection),
    )


def _c_phasing(_tsdate):
    return _PortedNamespace(
        _block_singletons=_tsdate.block_singletons,
        reallocate_unphased=_tsdate.reallocate_unphased,
        _mutation_frequency=_tsdate.mutation_frequency,
    )


def _c_util(_tsdate):
    return _PortedNamespace(
        _split_disjoint_nodes=_tsdate.split_disjoint_nodes,
        _relabel_mutations_node=_tsdate.relabel_mutations_node,
        _constrain_ages=_tsdate.constrain_ages,
        _contains_unary_nodes=_tsdate.contains_unary_nodes,
    )


def _c_prior(_tsdate):
    return _PortedNamespace(
        _marginalize_over_ancestors=_tsdate.marginalize_over_ancestors,
        conditional_coalescent_variance=_tsdate.conditional_coalescent_variance,
    )


def _c_rescaling(_tsdate):
    return _PortedNamespace(
        _fixed_changepoints=_tsdate.fixed_changepoints,
        _poisson_changepoints=_tsdate.poisson_changepoints,
        _count_mutations=_tsdate.count_mutations,
        mutational_area=_tsdate.mutational_area,
        mutational_timescale=_tsdate.mutational_timescale,
        piecewise_scale_point_estimate=_tsdate.piecewise_scale_point_estimate,
        piecewise_scale_posterior=_tsdate.piecewise_scale_posterior,
    )


def _c_namespace():
    import _tsdate

    return types.SimpleNamespace(
        hypergeo=_c_hypergeo(_tsdate),
        approx=_c_approx(_tsdate),
        variational=_PortedNamespace(),
        rescaling=_c_rescaling(_tsdate),
        phasing=_c_phasing(_tsdate),
        util=_c_util(_tsdate),
        prior=_c_prior(_tsdate),
        discrete=_PortedNamespace(),
    )


IMPLEMENTATIONS: dict[str, types.SimpleNamespace] = {
    "numba": _numba_namespace(),
}

try:
    IMPLEMENTATIONS["c"] = _c_namespace()
except ImportError:  # the extension has not been built
    pass
