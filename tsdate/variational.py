# MIT License
#
# Copyright (c) 2021-23 Tskit Developers
# Copyright (c) 2020-21 University of Oxford
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""
Expectation propagation implementation
"""

import logging
import time

import numba
import numpy as np
import tskit
from numba.types import void as _void
from tqdm.auto import tqdm

from . import approx
from .accelerate import numba_jit
from .approx import _b, _b1r, _f, _f1r, _f1w, _f2r, _f2w, _f3w, _i, _i1r
from .phasing import block_singletons, reallocate_unphased
from .rescaling import (
    count_mutations,
    mutational_timescale,
    piecewise_scale_point_estimate,
    piecewise_scale_posterior,
)
from .util import contains_unary_nodes

logger = logging.getLogger(__name__)

# columns for edge_factors
ROOTWARD = 0  # edge likelihood to parent
LEAFWARD = 1  # edge likelihood to child

# columns for unphased_factors
NODEONE = 0  # block likelihood to first parent
NODETWO = 1  # block likelihood to second parent

# columns for node_factors
MIXPRIOR = 0  # mixture prior to node
CONSTRNT = 1  # bounds on node ages

# columns for constraints
LOWER = 0  # lower bound on node
UPPER = 1  # upper bound on node

# named flags for unphased updates
USE_EDGE_LIKELIHOOD = False
USE_BLOCK_LIKELIHOOD = True

TINY = np.sqrt(np.finfo(np.float64).tiny)  # DEBUG

# dataclass for factors
_EPFactors = [
    ("node", _f3w),
    ("edge", _f3w),
    ("block", _f3w),
    ("scale", _f1w),
    ("_p", _i1r),
    ("_c", _i1r),
    ("_j", _i1r),
    ("_k", _i1r),
]


@numba.experimental.jitclass(_EPFactors)
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


# bypass bug in https://github.com/numba/numba/issues/7808
_fac = numba.deferred_type() if numba.config.DISABLE_JIT \
    else EPFactors.class_type.instance_type  # fmt: skip


# helper functions for numerical stability
@numba_jit(_void(_fac))
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


@numba_jit(_f2w(_fac))
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


@numba_jit(_f(_f1r, _f1r, _f))
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


@numba_jit(_f(_f1r, _f))
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


# main algorithm
class ExpectationPropagation:
    r"""
    The class that encapsulates running the variational gamma approach to
    tsdate fitting. This contains the Expectation propagation (EP) algorithm
    to infer approximate marginal distributions for node ages.

    The probability model has the form,

    .. math::

        \prod_{i \in \mathcal{N}} f(t_i | \theta_i)
        \prod_{(i,j) \in \mathcal{E}} g(y_ij | t_i - t_j)

    where :math:`f(.)` is a prior distribution on node ages with parameters
    :math:`\\theta` and :math:`g(.)` are Poisson likelihoods per edge. The
    EP approximation to the posterior has the form,

    .. math::

        \prod_{i \in \mathcal{N}} q(t_i | \eta_i)
        \prod_{(i,j) \in \mathcal{E}} q(t_i | \gamma_{ij}) q(t_j | \kappa_{ij})

    where :math:`q(.)` are pseudo-gamma distributions (termed 'factors'), and
    :math:`\eta, \gamma, \kappa` are variational parameters that reflect to
    prior, inside (leaf-to-root), and outside (root-to-edge) information.

    Thus, the EP approximation results in gamma-distribution marginals.  The
    factors :math:`q(.)` do not need to be valid distributions (e.g. the
    shape/rate parameters may be negative), as long as the marginals are valid
    distributions.  For details on how the variational parameters are
    optimized, see Minka (2002) "Expectation Propagation for Approximate
    Bayesian Inference"
    """

    @staticmethod
    @numba_jit(_void(_f2r, _i1r, _i1r))
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

    @staticmethod
    def _check_valid_inputs(ts, mutation_rate, allow_unary):
        if not mutation_rate > 0.0:
            raise ValueError("Mutation rate must be positive")
        if not allow_unary and contains_unary_nodes(ts):
            raise ValueError("Tree sequence contains unary nodes, simplify first")

    @staticmethod
    def _check_valid_state(
        posterior,
        factors,
    ):
        # Check that the messages sum to the posterior (debugging only)
        posterior_check = _assemble_factors(factors)
        np.testing.assert_allclose(posterior_check, posterior)

    def __init__(self, ts, *, mutation_rate, allow_unary=False, singletons_phased=True):
        """
        Initialize an expectation propagation algorithm for dating nodes
        in a tree sequence.

        :param ~tskit.TreeSequence ts: a tree sequence containing the partial
            ordering of nodes.
        :param ~float mutation_rate: the expected per-base mutation rate per
            time unit.
        :param ~bool allow_unary: if False, then error out if the tree sequence
            contains non-sample unary nodes.
        :param ~bool singletons_phased: if False, use an algorithm that
            treats singleton phase as unknown.
        """

        self._check_valid_inputs(ts, mutation_rate, allow_unary)
        self.edge_parents = ts.edges_parent
        self.edge_children = ts.edges_child

        # lower and upper bounds on node ages
        fixed_nodes = np.array(list(ts.samples()))
        self.node_constraints = np.zeros((ts.num_nodes, 2))
        self.node_constraints[:, 1] = np.inf
        self.node_constraints[fixed_nodes, :] = ts.nodes_time[fixed_nodes, np.newaxis]
        self._check_valid_constraints(
            self.node_constraints, self.edge_parents, self.edge_children
        )

        # count mutations on edges
        count_timing = time.time()
        self.edge_likelihoods, self.mutation_edges = count_mutations(ts)
        self.edge_likelihoods[:, 1] *= mutation_rate
        self.sizebiased_likelihoods, _ = count_mutations(ts, size_biased=True)
        self.sizebiased_likelihoods[:, 1] *= mutation_rate
        count_timing -= time.time()
        logger.debug(f"Extracted mutations in {abs(count_timing):.2f} seconds")

        # count mutations in singleton blocks
        phase_timing = time.time()
        individual_phased = np.full(ts.num_individuals, singletons_phased)
        self.block_likelihoods, self.block_edges, self.mutation_blocks = \
            block_singletons(ts, ~individual_phased)  # fmt: skip
        self.block_likelihoods[:, 1] *= mutation_rate
        num_blocks = self.block_likelihoods.shape[0]
        self.block_nodes = np.full((2, num_blocks), tskit.NULL, dtype=np.int32)
        self.block_nodes[0] = self.edge_parents[self.block_edges[:, 0]]
        self.block_nodes[1] = self.edge_parents[self.block_edges[:, 1]]
        num_unphased = np.sum(self.mutation_blocks != tskit.NULL)
        phase_timing -= time.time()
        logger.info(f"Found {num_unphased} unphased singleton mutations")
        logger.info(f"Split unphased singleton edges into {num_blocks} blocks")
        logger.debug(f"Phased singletons in {abs(phase_timing):.2f} seconds")

        # mutable
        self.factors = EPFactors(
            self.node_constraints,
            self.edge_parents,
            self.edge_children,
            self.block_nodes[0],
            self.block_nodes[1],
        )
        self.node_posterior = np.zeros((ts.num_nodes, 2))
        self.mutation_posterior = np.full((ts.num_mutations, 2), np.nan)
        self.mutation_phase = np.ones(ts.num_mutations)
        self.mutation_nodes = ts.mutations_node.copy()
        self.edge_logconst = np.zeros(ts.num_edges)
        self.block_logconst = np.zeros(num_blocks)

        # terminal nodes
        has_parent = np.full(ts.num_nodes, False)
        has_child = np.full(ts.num_nodes, False)
        has_parent[self.edge_children] = True
        has_child[self.edge_parents] = True
        self.roots = np.logical_and(has_child, ~has_parent)
        self.leaves = np.logical_and(~has_child, has_parent)
        if np.any(np.logical_and(~has_child, ~has_parent)):
            raise ValueError("Tree sequence contains disconnected nodes")
        # TODO: we don't need to store all these similar vectors
        self.unconstrained_roots = self.roots.copy()
        self.unconstrained_roots[fixed_nodes] = False

        # edge traversal order
        edge_unphased = np.full(ts.num_edges, False)
        edge_unphased[self.block_edges[:, 0]] = True
        edge_unphased[self.block_edges[:, 1]] = True
        edges = np.arange(ts.num_edges, dtype=np.int32)[~edge_unphased]
        self.edge_order = np.concatenate((edges[:-1], np.flip(edges)))
        self.block_order = np.arange(num_blocks, dtype=np.int32)
        self.mutation_order = np.arange(ts.num_mutations, dtype=np.int32)

    @staticmethod
    @numba_jit(_void(_i1r, _i1r, _i1r, _f2r, _f2r, _f2w, _fac, _f1w, _f, _f, _b))
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

    @staticmethod
    @numba_jit(_void(_b1r, _f2w, _fac, _f, _i, _f))
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
        penalty = 1 / np.mean(shape / rate)
        itt, delta = 0, np.inf
        while abs(delta) > abs(penalty) * em_reltol:
            if itt > em_maxitt:
                break
            delta = 1 / np.mean(shape / (rate + penalty)) - penalty
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

    @staticmethod
    @numba_jit(_void(_i1r, _f2w, _f1w, _i1r, _i1r, _i1r, _f2r, _f2r, _f2r, _fac, _b))
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

    def iterate(
        self,
        *,
        max_shape=1000,
        min_step=0.1,
        em_maxitt=10,
        em_reltol=1e-8,
        regularise=True,
        check_valid=False,  # for debugging
    ):
        logger.debug("Passing through singleton blocks")
        self.propagate_likelihood(
            self.block_order,
            self.block_nodes[ROOTWARD],
            self.block_nodes[LEAFWARD],
            self.block_likelihoods,
            self.node_constraints,
            self.node_posterior,
            self.factors,
            self.block_logconst,
            max_shape,
            min_step,
            USE_BLOCK_LIKELIHOOD,
        )

        logger.debug("Rootward + leafward pass through edges")
        self.propagate_likelihood(
            self.edge_order,
            self.edge_parents,
            self.edge_children,
            self.edge_likelihoods,
            self.node_constraints,
            self.node_posterior,
            self.factors,
            self.edge_logconst,
            max_shape,
            min_step,
            USE_EDGE_LIKELIHOOD,
        )

        if regularise:
            logger.debug("Exponential regularization on roots")
            self.propagate_prior(
                self.unconstrained_roots,
                self.node_posterior,
                self.factors,
                max_shape,
                em_maxitt,
                em_reltol,
            )

        logger.debug("Absorbing scaling term into the factors")
        _rescale_factors(self.factors)

        if check_valid:  # for debugging
            self._check_valid_state(self.node_posterior, self.factors)

    def rescale(
        self,
        *,
        rescale_intervals=1000,
        rescale_segsites=False,
        rescale_iterations=10,
        quantile_width=0.5,
        max_shape=1000,
        progress=False,
    ):
        # Normalise posteriors so that empirical mutation rate is constant
        likelihoods = self.edge_likelihoods if rescale_segsites \
            else self.sizebiased_likelihoods  # fmt: skip
        reallocate_unphased(  # correct mutation counts for unphased singletons
            likelihoods,
            self.mutation_phase,
            self.mutation_blocks,
            self.block_edges,
        )
        nodes_fixed = self.node_constraints[:, 0] == self.node_constraints[:, 1]
        mutations_fixed = np.isnan(self.mutation_posterior[:, 0])
        nodes_time, _ = self.node_moments()
        rescaled_nodes_time = nodes_time.copy()
        for _ in np.arange(rescale_iterations):  # estimate time rescaling
            original_breaks, rescaled_breaks = mutational_timescale(
                rescaled_nodes_time,
                likelihoods,
                nodes_fixed,
                self.edge_parents,
                self.edge_children,
                rescale_intervals,
            )
            rescaled_nodes_time = piecewise_scale_point_estimate(
                rescaled_nodes_time,
                nodes_fixed,
                original_breaks,
                rescaled_breaks,
            )
            assert np.allclose(rescaled_nodes_time[nodes_fixed], nodes_time[nodes_fixed])
        # TODO: clean up
        _, unique = np.unique(rescaled_nodes_time[~nodes_fixed], return_index=True)
        original_breaks = piecewise_scale_point_estimate(  # recover original breakpoints
            rescaled_breaks,
            np.full(rescaled_breaks.size, False),
            np.append(0, rescaled_nodes_time[~nodes_fixed][unique]),
            np.append(0, nodes_time[~nodes_fixed][unique]),
        )
        # /TODO
        self.node_posterior[:] = piecewise_scale_posterior(
            self.node_posterior,
            nodes_fixed,
            original_breaks,
            rescaled_breaks,
            quantile_width,
            max_shape,
        )
        self.mutation_posterior[:] = piecewise_scale_posterior(
            self.mutation_posterior,
            mutations_fixed,
            original_breaks,
            rescaled_breaks,
            quantile_width,
            max_shape,
        )

    # TODO change to `date`
    def infer(
        self,
        *,
        ep_iterations,
        max_shape,
        rescale_intervals,
        rescale_iterations,
        regularise,
        rescale_segsites,
        min_step=0.1,
        progress=None,
    ):
        # Run multiple rounds of expectation propagation, and return stats
        self.mean_edge_logconst = []  # Undocumented: can be used to assess convergence
        nodes_timing = time.time()
        for _ in tqdm(
            np.arange(ep_iterations),
            desc="Expectation Propagation",
            disable=not progress,
        ):
            self.iterate(
                max_shape=max_shape,
                min_step=min_step,
                regularise=regularise,
            )
            self.mean_edge_logconst.append(np.mean(self.edge_logconst))

        nodes_timing -= time.time()
        skipped_edges = np.sum(np.isnan(self.edge_logconst))
        logger.info(f"Skipped {skipped_edges} edges with invalid factors")
        logger.info(f"Calculated node posteriors in {abs(nodes_timing):.2f} seconds")

        muts_timing = time.time()
        mutations_phased = self.mutation_blocks == tskit.NULL
        logger.debug("Passing through unphased singletons")
        self.propagate_mutations(  # unphased singletons
            self.mutation_order[~mutations_phased],
            self.mutation_posterior,
            self.mutation_phase,
            self.mutation_blocks,
            self.block_nodes[ROOTWARD],
            self.block_nodes[LEAFWARD],
            self.block_likelihoods,
            self.node_constraints,
            self.node_posterior,
            self.factors,
            USE_BLOCK_LIKELIHOOD,
        )
        logger.debug("Passing through phased mutations")
        self.propagate_mutations(  # phased mutations
            self.mutation_order[mutations_phased],
            self.mutation_posterior,
            self.mutation_phase,
            self.mutation_edges,
            self.edge_parents,
            self.edge_children,
            self.edge_likelihoods,
            self.node_constraints,
            self.node_posterior,
            self.factors,
            USE_EDGE_LIKELIHOOD,
        )
        muts_timing -= time.time()
        skipped_muts = np.sum(np.isnan(self.mutation_posterior[:, 0]))
        logger.info(f"Skipped {skipped_muts} mutations with invalid posteriors")
        logger.info(f"Calculated mutation posteriors in {abs(muts_timing):.2f} seconds")

        singletons = self.mutation_blocks != tskit.NULL
        switched_blocks = self.mutation_blocks[singletons]
        switched_edges = np.where(
            self.mutation_phase[singletons] < 0.5,
            self.block_edges[switched_blocks, 1],
            self.block_edges[switched_blocks, 0],
        )
        self.mutation_edges[singletons] = switched_edges
        self.mutation_nodes[singletons] = self.edge_children[switched_edges]
        switched = self.mutation_phase < 0.5
        self.mutation_phase[switched] = 1 - self.mutation_phase[switched]
        logger.info(f"Switched phase of {np.sum(switched)} singletons")

        if rescale_intervals > 0 and rescale_iterations > 0:
            rescale_timing = time.time()
            self.rescale(
                rescale_intervals=rescale_intervals,
                rescale_iterations=rescale_iterations,
                rescale_segsites=rescale_segsites,
                max_shape=max_shape,
                progress=progress,
            )
            rescale_timing -= time.time()
            logger.info(f"Timescale rescaled in {abs(rescale_timing):.2f} seconds")

    def node_moments(self):
        # Posterior mean and variance of node ages (equivalent to node_posteriors)
        alpha, beta = self.node_posterior.T
        nodes_mn = np.ascontiguousarray(self.node_constraints[:, 0])
        nodes_va = np.zeros(nodes_mn.size)
        free = self.node_constraints[:, 0] != self.node_constraints[:, 1]
        nodes_mn[free] = (alpha[free] + 1) / beta[free]
        nodes_va[free] = nodes_mn[free] / beta[free]
        return nodes_mn, nodes_va

    def mutation_moments(self):
        # Posterior mean and variance of mutation ages
        alpha, beta = self.mutation_posterior.T
        muts_mn = np.full(alpha.size, np.nan)
        muts_va = np.full(alpha.size, np.nan)
        free = np.isfinite(alpha)
        muts_mn[free] = (alpha[free] + 1) / beta[free]
        muts_va[free] = muts_mn[free] / beta[free]
        return muts_mn, muts_va

    def mutation_mapping(self):
        # Map from mutations to edges and subtended node, using estimated singleton
        # phase (if singletons were unphased)

        # TODO: should these be copies? Should members be readonly?
        return self.mutation_nodes

    def marginal_likelihood(self):
        # Return the marginal likelihood of the data given the model

        # TODO: implement
        return None

    def node_posteriors(self):
        """
        Return parameters specifying the inferred posterior distribution of node
        times which can be e.g. read into a ``pandas.DataFrame`` for further analysis.
        The mean times are not strictly constrained by topology, so unlike the
        ``nodes_time`` attribute of a tree sequence, the mean time of a parent node
        may occasionally be less than that of one of its children.

        :return: The distribution of posterior node times as a structured array of
            mean and variance. Row ``i`` gives the mean and variance of inferred
            node times for node ``i``.
        :rtype: numpy.ndarray
        """
        node_mn, node_va = self.node_moments()
        dtype = [("mean", node_mn.dtype), ("variance", node_va.dtype)]
        data = np.empty(node_mn.size, dtype=dtype)
        data["mean"] = node_mn
        data["variance"] = node_va
        return data

    def mutation_posteriors(self):
        """
        Returns parameters specifying the inferred posterior distribution of mutation
        times which can be e.g. read into a ``pandas.DataFrame`` for further analysis.
        These are calculated as the midpoint distribution of the posterior node time
        distributions of the node above and below the mutation. Note that this means
        it is possible for a mean mutation time not to lie between the mean values of
        its parent and child nodes.

        .. note::
            For unphased singletons, the posterior mutation time is integrated over
            the two possible haploid genomes on which the singleton could be placed,
            accounting for the relative branch lengths above each genome.

        :return: The distribution of posterior mutation times as a structured array
            of mean and variance. Row ``i`` gives the mean and variance of inferred
            mutations times for mutation ``i``.
        :rtype: numpy.ndarray
        """
        mut_mn, mut_va = self.mutation_moments()
        dtype = [("mean", mut_mn.dtype), ("variance", mut_va.dtype)]
        data = np.empty(mut_mn.size, dtype=dtype)
        data["mean"] = mut_mn
        data["variance"] = mut_va
        return data


# NB: used for debugging
# def date(
#     ts,
#     *,
#     mutation_rate,
#     singletons_phased=True,
#     max_iterations=10,
#     rescaling_intervals=1000,
#     rescaling_iterations=10,
#     match_segregating_sites=False,
#     regularise_roots=True,
#     constr_iterations=0,
#     progress=True,
# ):
#     """
#     Date a tree sequence with expectation propagation. Returns dated tree
#     sequence and converged ExpectationPropagation object.
#     """
#
#     posterior = variational.ExpectationPropagation(
#         ts,
#         mutation_rate=mutation_rate,
#         singletons_phased=singletons_phased,
#     )
#     posterior.infer(
#         ep_maxitt=max_iterations,
#         max_shape=max_shape,
#         rescale_intervals=rescaling_intervals,
#         rescale_iterations=rescaling_iterations,
#         regularise=regularise_roots,
#         rescale_segsites=match_segregating_sites,
#         progress=progress,
#     )
#
#     node_mn, node_va = posterior.node_moments()
#     mutation_mn, mutation_va = posterior.mutation_moments()
#     mutation_edge, mutation_node = posterior.mutation_mapping()
#
#     tables = ts.dump_tables()
#     tables.nodes.time = \
#        constrain_ages(ts, node_mn, constr_iterations=constr_iterations)
#     tables.mutations.node = mutation_node
#     tables.sort()
#
#     return tables.tree_sequence(), posterior
