"""Oracles for the jitted functions in ``tsdate.variational``."""

import numpy as np
import tskit

from tsdate.variational import (
    CONSTRNT,
    LEAFWARD,
    LOWER,
    MIXPRIOR,
    ROOTWARD,
    TINY,
    UPPER,
)

from . import approx

# oracle: numba's np.mean accumulates sequentially, whereas numpy sums pairwise,
# which differs in the last bits; `seq_mean` sums sequentially.
from ._libm import seq_mean


# oracle: the numba version is a `jitclass`; this is a plain class with the same
# fields (arrays are created with the same dtypes) and constructor.
class EPFactors:
    """
    Mutable state of the EP algorithm, that is the factors associated
    with each edge likelihood / block likelihood / prior factor, that
    together form the posterior.

    Because of constraints on numba's AOT compilation of jitclass,
    this is a dataclass (no intrinsic methods).
    """

    def __init__(
        self,
        node_constraints,
        edge_parents,
        edge_children,
        block_left,
        block_right,
    ):
        assert edge_parents.size == edge_children.size
        assert block_left.size == block_right.size
        num_nodes = node_constraints.shape[0]
        num_edges = edge_parents.size
        num_blocks = block_left.size
        self.node = np.zeros((num_nodes, 2, 2))
        self.edge = np.zeros((num_edges, 2, 2))
        self.block = np.zeros((num_blocks, 2, 2))
        self.scale = np.ones(num_nodes)
        self._p, self._c = edge_parents, edge_children
        self._j, self._k = block_left, block_right


def _rescale_factors(factors):
    """
    At each update, all factors associated with a node need to be rescaled.
    This is expensive, so we keep track of the scaling in a separate vector.
    As updates progress, the factors will drift upward in magnitude while
    the scaling drifts downward. To avoid underflow, we may need to
    cancel out the scaling.
    """
    factors.edge[:, ROOTWARD] *= factors.scale[factors._p, np.newaxis]
    factors.edge[:, LEAFWARD] *= factors.scale[factors._c, np.newaxis]
    factors.block[:, ROOTWARD] *= factors.scale[factors._j, np.newaxis]
    factors.block[:, LEAFWARD] *= factors.scale[factors._k, np.newaxis]
    factors.node[:, MIXPRIOR] *= factors.scale[:, np.newaxis]
    factors.node[:, CONSTRNT] *= factors.scale[:, np.newaxis]
    factors.scale[:] = 1.0


def _assemble_factors(factors):
    """
    Transform factors into posteriors, intended for debugging purposes only.
    """
    posterior = np.zeros((factors.scale.size, 2))
    for i, (p, c) in enumerate(zip(factors._p, factors._c)):
        posterior[p] += factors.edge[i, ROOTWARD]
        posterior[c] += factors.edge[i, LEAFWARD]
    for i, (j, k) in enumerate(zip(factors._j, factors._k)):
        posterior[j] += factors.block[i, ROOTWARD]
        posterior[k] += factors.block[i, LEAFWARD]
    posterior += factors.node[:, MIXPRIOR]
    posterior += factors.node[:, CONSTRNT]
    return posterior


def _damp(x, y, s):
    """
    If `x - y` is too small, find `d` so that `x - d*y` is large enough:

        x[0] - d*y[0] + 1 >= (x[0] + 1)*s
        x[1] - d*y[1] >= y[1]*s

    for `0 < s < 1`.
    """
    assert x.size == y.size == 2
    if np.all(y == 0.0) and np.all(x == 0.0):
        return 1.0
    assert 0 < s < 1
    assert 0.0 < x[0] + 1
    assert 0.0 < x[1]
    a = 1.0 if (1 + x[0] - y[0] > (1 + x[0]) * s) else (1 - s) * (1 + x[0]) / y[0]
    b = 1.0 if (x[1] - y[1] > x[1] * s) else (1 - s) * x[1] / y[1]
    d = min(a, b)
    assert 0.0 < d <= 1.0
    return d


def _rescale(x, s):
    """
    Find `d` so that `d*x[0] + 1 <= s[0]` or `d*x[0] + 1 >= 1/s[0]`
    """
    assert x.size == 2
    if np.all(x == 0.0):
        return 1.0
    assert 0 < x[0] + 1
    assert 0 < x[1]
    if 1 + x[0] > s:
        return (s - 1) / x[0]
    elif 1 + x[0] < 1 / s:
        return (1 / s - 1) / x[0]
    return 1.0


def _check_valid_constraints(constraints, edges_parent, edges_child):
    # Check that upper-bound on node age is greater than maximum lower-bound
    # for ages of descendants
    lower_max = constraints[:, LOWER].copy()
    for p, c in zip(edges_parent, edges_child):
        lower_max[p] = max(lower_max[c], lower_max[p])
    if np.any(lower_max > constraints[:, UPPER]):
        raise ValueError(
            "Node age constraints are inconsistent, some descendants have"
            " lower bounds that exceed upper bounds of their ancestors."
        )


def propagate_likelihood(
    edge_order,
    edges_parent,
    edges_child,
    likelihoods,
    constraints,
    posterior,
    factors,
    lognorm,
    max_shape,
    min_step,
    unphased,
):
    # Update approximating factors for Poisson mutation likelihoods on edges.
    #
    # :param numpy.ndarray edges_parent: integer array of parent ids per edge
    # :param numpy.ndarray edges_child: integer array of child ids per edge
    # :param numpy.ndarray likelihoods: array of dimension `[num_edges, 2]`
    #     containing mutation count and mutational target size per edge.
    # :param numpy.ndarray constraints: array of dimension `[num_nodes, 2]`
    #     containing lower and upper bounds for each node.
    # :param numpy.ndarray posterior: array of dimension `[num_nodes, 2]`
    #     containing natural parameters for each node, updated in-place.
    # :param EPFactors factors: object containing parent and child factors
    #     (natural parameters) for each edge, updated in-place.
    # :param numpy.ndarray lognorm: array of dimension `[num_edges]`
    #     containing the approximate normalizing constants per edge,
    #     updated in-place.
    # :param float max_shape: the maximum allowed shape for node posteriors.
    # :param float min_step: the minimum allowed step size in (0, 1).
    # :param bool unphased: if True, edges are treated as blocks of unphased
    #     singletons in contemporary individuals

    assert constraints.shape == posterior.shape
    assert edges_child.size == edges_parent.size
    assert likelihoods.shape == (edges_parent.size, 2)
    assert max_shape >= 1.0
    assert 0.0 < min_step < 1.0

    def cavity_damping(x, y):
        return _damp(x, y, min_step)

    def posterior_damping(x):
        return _rescale(x, max_shape)

    def leafward_projection(x, y, z):
        if unphased:
            return approx.sideways_projection(x, y, z)
        return approx.leafward_projection(x, y, z)

    def rootward_projection(x, y, z):
        if unphased:
            return approx.sideways_projection(x, y, z)
        return approx.rootward_projection(x, y, z)

    def gamma_projection(x, y, z):
        if unphased:
            return approx.unphased_projection(x, y, z)
        return approx.gamma_projection(x, y, z)

    def twin_projection(x, y):
        assert unphased, "Invalid update"
        return approx.twin_projection(x, y)

    fixed = constraints[:, LOWER] == constraints[:, UPPER]

    scale = factors.scale
    factor = factors.block if unphased else factors.edge
    assert factor.shape == (edges_parent.size, 2, 2)

    for i in edge_order:
        p, c = edges_parent[i], edges_child[i]
        if scale[p] < TINY or scale[c] < TINY:
            # modifies `scale` and `factor` by reference
            _rescale_factors(factors)
        if fixed[p] and fixed[c]:
            continue
        elif fixed[p] and not fixed[c]:
            # in practice this should only occur if a sample is the
            # ancestor of another sample
            child_message = factor[i, LEAFWARD] * scale[c]
            child_delta = cavity_damping(posterior[c], child_message)
            child_cavity = posterior[c] - child_delta * child_message
            edge_likelihood = child_delta * likelihoods[i]
            parent_age = constraints[p, LOWER]
            lognorm[i], posterior[c] = leafward_projection(
                parent_age,
                child_cavity,
                edge_likelihood,
            )
            factor[i, LEAFWARD] *= 1.0 - child_delta
            factor[i, LEAFWARD] += (posterior[c] - child_cavity) / scale[c]
            child_eta = posterior_damping(posterior[c])
            posterior[c] *= child_eta
            scale[c] *= child_eta
        elif fixed[c] and not fixed[p]:
            # in practice this should only occur if a sample has age
            # greater than zero
            parent_message = factor[i, ROOTWARD] * scale[p]
            parent_delta = cavity_damping(posterior[p], parent_message)
            parent_cavity = posterior[p] - parent_delta * parent_message
            edge_likelihood = parent_delta * likelihoods[i]
            child_age = constraints[c, LOWER]
            lognorm[i], posterior[p] = rootward_projection(
                child_age,
                parent_cavity,
                edge_likelihood,
            )
            factor[i, ROOTWARD] *= 1.0 - parent_delta
            factor[i, ROOTWARD] += (posterior[p] - parent_cavity) / scale[p]
            parent_eta = posterior_damping(posterior[p])
            posterior[p] *= parent_eta
            scale[p] *= parent_eta
        else:
            if p == c:  # singleton block with single parent
                parent_message = factor[i, ROOTWARD] * scale[p]
                parent_delta = cavity_damping(posterior[p], parent_message)
                parent_cavity = posterior[p] - parent_delta * parent_message
                edge_likelihood = parent_delta * likelihoods[i]
                child_age = constraints[c, LOWER]
                lognorm[i], posterior[p] = \
                    twin_projection(parent_cavity, edge_likelihood)  # fmt: skip
                factor[i, ROOTWARD] *= 1.0 - parent_delta
                factor[i, ROOTWARD] += (posterior[p] - parent_cavity) / scale[p]
                parent_eta = posterior_damping(posterior[p])
                posterior[p] *= parent_eta
                scale[p] *= parent_eta
            else:
                # lower-bound cavity
                parent_message = factor[i, ROOTWARD] * scale[p]
                child_message = factor[i, LEAFWARD] * scale[c]
                parent_delta = cavity_damping(posterior[p], parent_message)
                child_delta = cavity_damping(posterior[c], child_message)
                delta = min(parent_delta, child_delta)

                parent_cavity = posterior[p] - delta * parent_message
                child_cavity = posterior[c] - delta * child_message
                edge_likelihood = delta * likelihoods[i]

                # match moments and update factors
                lognorm[i], posterior[p], posterior[c] = gamma_projection(
                    parent_cavity,
                    child_cavity,
                    edge_likelihood,
                )
                factor[i, ROOTWARD] *= 1.0 - delta
                factor[i, ROOTWARD] += (posterior[p] - parent_cavity) / scale[p]
                factor[i, LEAFWARD] *= 1.0 - delta
                factor[i, LEAFWARD] += (posterior[c] - child_cavity) / scale[c]

                # upper-bound posterior
                parent_eta = posterior_damping(posterior[p])
                child_eta = posterior_damping(posterior[c])
                posterior[p] *= parent_eta
                posterior[c] *= child_eta
                scale[p] *= parent_eta
                scale[c] *= child_eta


def propagate_prior(free, posterior, factors, max_shape, em_maxitt, em_reltol):
    # Update approximating factors for global prior.
    #
    # :param ndarray free: boolean array for if prior should be applied to node
    # :param ndarray penalty: initial value for regularisation penalty
    # :param ndarray posterior: rows are nodes, columns are first and
    #     second natural parameters of gamma posteriors. Updated in place.
    # :param EPFactors factors: object containing EP messages for nodes.
    #     Updated in place.
    # :param float max_shape: the maximum allowed shape for node posteriors.
    # :param int em_maxitt: the maximum number of EM iterations to use when
    #     fitting the regularisation.
    # :param int em_reltol: the termination criterion for relative change in
    #     log-likelihood.

    assert free.size == posterior.shape[0]
    assert max_shape >= 1.0

    def posterior_damping(x):
        return _rescale(x, max_shape)

    if not np.any(free):
        return

    factor = factors.node
    scale = factors.scale
    assert factor.shape == (free.size, 2, 2)
    assert scale.size == free.size

    # fit an exponential to cavity distributions for unconstrained nodes
    cavity = posterior - factor[:, MIXPRIOR] * scale[:, np.newaxis]
    shape, rate = cavity[free, 0] + 1, cavity[free, 1]
    penalty = 1 / seq_mean(shape / rate)  # oracle: seq_mean
    itt, delta = 0, np.inf
    while abs(delta) > abs(penalty) * em_reltol:
        if itt > em_maxitt:
            break
        delta = 1 / seq_mean(shape / (rate + penalty)) - penalty  # oracle: seq_mean
        penalty += delta
        itt += 1
    assert penalty > 0

    # update posteriors and rescale to keep shape bounded
    posterior[free, 1] = cavity[free, 1] + penalty
    factor[free, MIXPRIOR] = \
        (posterior[free] - cavity[free]) / scale[free, np.newaxis]  # fmt: skip
    for i in np.flatnonzero(free):
        eta = posterior_damping(posterior[i])
        posterior[i] *= eta
        scale[i] *= eta


def propagate_mutations(
    mutations_order,
    mutations_posterior,
    mutations_phase,
    mutations_edge,
    edges_parent,
    edges_child,
    likelihoods,
    constraints,
    posterior,
    factors,
    unphased,
):
    # Calculate posteriors for mutations. (publicly undocumented)
    #
    # :param ndarray mutations_order: integer array giving order in
    #     which to traverse mutations
    # :param ndarray mutations_posterior: array of dimension `(num_mutations, 2)`
    #     containing natural parameters for each mutation, modified in place
    # :param ndarray mutations_phase: array of dimension `(num_mutations, )`
    #     containing mutation phase, modified in place
    # :param ndarray mutations_edge: integer array giving edge for each mutation
    # :param ndarray edges_parent: integer array of parent ids per edge
    # :param ndarray edges_child: integer array of child ids per edge
    # :param ndarray likelihoods: array of dimension `[num_edges, 2]`
    #     containing mutation count and mutational target size per edge.
    # :param ndarray constraints: array of dimension `[num_nodes, 2]`
    #     containing lower and upper bounds for each node.
    # :param ndarray posterior: array of dimension `[num_nodes, 2]`
    #     containing natural parameters for each node
    # :param EPFactors factors: object containing
    #     containing parent and child factors (natural parameters) for each edge
    # :param bool unphased: if True, edges are treated as blocks of unphased
    #     singletons in contemporary individuals

    # TODO: scale should be 1.0, can we delete
    # TODO: we don't seem to need to damp?

    # TODO: assert more stuff here?
    assert mutations_phase.size == mutations_edge.size
    assert mutations_posterior.shape == (mutations_phase.size, 2)
    assert constraints.shape == posterior.shape
    assert edges_child.size == edges_parent.size
    assert likelihoods.shape == (edges_parent.size, 2)

    def leafward_projection(x, y, z):
        if unphased:
            return approx.mutation_sideways_projection(x, y, z)
        return approx.mutation_leafward_projection(x, y, z)

    def rootward_projection(x, y, z):
        if unphased:
            return approx.mutation_sideways_projection(x, y, z)
        return approx.mutation_rootward_projection(x, y, z)

    def gamma_projection(x, y, z):
        if unphased:
            return approx.mutation_unphased_projection(x, y, z)
        return approx.mutation_gamma_projection(x, y, z)

    def fixed_projection(x, y):
        if unphased:
            return approx.mutation_block_projection(x, y)
        return approx.mutation_edge_projection(x, y)

    twin_projection = approx.mutation_twin_projection

    fixed = constraints[:, LOWER] == constraints[:, UPPER]

    scale = factors.scale
    factor = factors.block if unphased else factors.edge
    assert factor.shape == (edges_parent.size, 2, 2)

    for m in mutations_order:
        i = mutations_edge[m]
        if i == tskit.NULL:  # skip mutations above root
            continue
        p, c = edges_parent[i], edges_child[i]
        if fixed[p] and fixed[c]:
            child_age = constraints[c, 0]
            parent_age = constraints[p, 0]
            mutations_phase[m], mutations_posterior[m] = \
                fixed_projection(parent_age, child_age)  # fmt: skip
        elif fixed[p] and not fixed[c]:
            child_message = factor[i, LEAFWARD] * scale[c]
            child_delta = 1.0  # hopefully we don't need to damp
            child_cavity = posterior[c] - child_delta * child_message
            edge_likelihood = child_delta * likelihoods[i]
            parent_age = constraints[p, LOWER]
            mutations_phase[m], mutations_posterior[m] = leafward_projection(
                parent_age,
                child_cavity,
                edge_likelihood,
            )
        elif fixed[c] and not fixed[p]:
            parent_message = factor[i, ROOTWARD] * scale[p]
            parent_delta = 1.0  # hopefully we don't need to damp
            parent_cavity = posterior[p] - parent_delta * parent_message
            edge_likelihood = parent_delta * likelihoods[i]
            child_age = constraints[c, LOWER]
            mutations_phase[m], mutations_posterior[m] = rootward_projection(
                child_age,
                parent_cavity,
                edge_likelihood,
            )
        else:
            if p == c:  # singleton block with single parent
                parent_message = factor[i, ROOTWARD] * scale[p]
                parent_delta = 1.0  # hopefully we don't need to damp
                parent_cavity = posterior[p] - parent_delta * parent_message
                edge_likelihood = parent_delta * likelihoods[i]
                child_age = constraints[c, LOWER]
                mutations_phase[m], mutations_posterior[m] = \
                    twin_projection(parent_cavity, edge_likelihood)  # fmt: skip
            else:
                parent_message = factor[i, ROOTWARD] * scale[p]
                child_message = factor[i, LEAFWARD] * scale[c]
                parent_delta = 1.0  # hopefully we don't need to damp
                child_delta = 1.0  # hopefully we don't need to damp
                delta = min(parent_delta, child_delta)
                parent_cavity = posterior[p] - delta * parent_message
                child_cavity = posterior[c] - delta * child_message
                edge_likelihood = delta * likelihoods[i]
                mutations_phase[m], mutations_posterior[m] = gamma_projection(
                    parent_cavity,
                    child_cavity,
                    edge_likelihood,
                )
