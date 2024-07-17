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
from .approx import _b
from .approx import _b1r
from .approx import _f
from .approx import _f1r
from .approx import _f1w
from .approx import _f2r
from .approx import _f2w
from .approx import _f3r
from .approx import _f3w
from .approx import _i
from .approx import _i1r
from .approx import _i2r
from .hypergeo import _gammainc_inv as gammainc_inv
from .phasing import block_singletons
from .phasing import reallocate_unphased
from .rescaling import count_mutations
from .rescaling import edge_sampling_weight
from .rescaling import mutational_timescale
from .rescaling import piecewise_scale_posterior
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


@numba.njit(_f(_f1r, _f1r, _f))
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


@numba.njit(_f(_f1r, _f))
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


class ExpectationPropagation:
    r"""
    Expectation propagation (EP) algorithm to infer approximate marginal
    distributions for node ages.

    The probability model has the form,

    .. math::

        \prod_{i \in \mathcal{N} f(t_i | \theta_i)
        \prod_{(i,j) \in \mathcal{E}} g(y_ij | t_i - t_j)

    where :math:`f(.)` is a prior distribution on node ages with parameters
    :math:`\\theta` and :math:`g(.)` are Poisson likelihoods per edge. The
    EP approximation to the posterior has the form,

    .. math::

        \prod_{i \in \mathcal{N} q(t_i | \eta_i)
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
    @numba.njit(_void(_f2r, _i1r, _i1r))
    def _check_valid_constraints(constraints, edges_parent, edges_child):
        """
        Check that upper-bound on node age is greater than maximum lower-bound
        for ages of descendants
        """
        lower_max = constraints[:, LOWER].copy()
        for p, c in zip(edges_parent, edges_child):
            lower_max[p] = max(lower_max[c], lower_max[p])
        if np.any(lower_max > constraints[:, UPPER]):
            raise ValueError(
                "Node age constraints are inconsistent, some descendants have"
                " lower bounds that exceed upper bounds of their ancestors."
            )

    @staticmethod
    def _check_valid_inputs(ts, mutation_rate):
        if not mutation_rate > 0.0:
            raise ValueError("Mutation rate must be positive")
        if contains_unary_nodes(ts):
            raise ValueError("Tree sequence contains unary nodes, simplify first")

    @staticmethod
    def _check_valid_state(
        edges_parent,
        edges_child,
        block_nodes,
        posterior,
        node_factors,
        edge_factors,
        block_factors,
    ):
        """Check that the messages sum to the posterior (debugging only)"""
        block_one, block_two = block_nodes
        posterior_check = np.zeros(posterior.shape)
        for i, (p, c) in enumerate(zip(edges_parent, edges_child)):
            posterior_check[p] += edge_factors[i, ROOTWARD]
            posterior_check[c] += edge_factors[i, LEAFWARD]
        for i, (j, k) in enumerate(zip(block_one, block_two)):
            posterior_check[j] += block_factors[i, ROOTWARD]
            posterior_check[k] += block_factors[i, LEAFWARD]
        posterior_check += node_factors[:, MIXPRIOR]
        posterior_check += node_factors[:, CONSTRNT]
        np.testing.assert_allclose(posterior_check, posterior)

    @staticmethod
    @numba.njit(_f1w(_f2r, _f2r, _b))
    def _point_estimate(posteriors, constraints, median):
        assert posteriors.shape == constraints.shape
        fixed = constraints[:, 0] == constraints[:, 1]
        point_estimate = np.zeros(posteriors.shape[0])
        for i in np.flatnonzero(~fixed):
            alpha, beta = posteriors[i]
            point_estimate[i] = gammainc_inv(alpha + 1, 0.5) \
                if median else (alpha + 1)  # fmt: skip
            point_estimate[i] /= beta
        point_estimate[fixed] = constraints[fixed, 0]
        return point_estimate

    def __init__(self, ts, *, mutation_rate, singletons_phased=True):
        """
        Initialize an expectation propagation algorithm for dating nodes
        in a tree sequence.

        .. note:: Each row of ``likelihoods`` corresponds to an edge in the
            tree sequence. The entries are the outcome and rate of a Poisson
            distribution for mutations on the edge. When both entries are zero,
            this degenerates to an indicator function :math:`I[child_age <
            parent_age]`.

        :param ~tskit.TreeSequence ts: a tree sequence containing the partial
            ordering of nodes.
        :param ~float mutation_rate: the expected per-base mutation rate per
            time unit.
        """

        self._check_valid_inputs(ts, mutation_rate)
        self.edge_parents = ts.edges_parent
        self.edge_children = ts.edges_child

        # lower and upper bounds on node ages
        samples = list(ts.samples())
        self.node_constraints = np.zeros((ts.num_nodes, 2))
        self.node_constraints[:, 1] = np.inf
        self.node_constraints[samples, :] = ts.nodes_time[samples, np.newaxis]
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
        logger.info(f"Extracted mutations in {abs(count_timing)} seconds")

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
        logger.info(f"Phased singletons in {abs(phase_timing)} seconds")

        # mutable
        self.node_factors = np.zeros((ts.num_nodes, 2, 2))
        self.edge_factors = np.zeros((ts.num_edges, 2, 2))
        self.block_factors = np.zeros((num_blocks, 2, 2))
        self.node_posterior = np.zeros((ts.num_nodes, 2))
        self.mutation_posterior = np.full((ts.num_mutations, 2), np.nan)
        self.mutation_phase = np.ones(ts.num_mutations)
        self.mutation_nodes = ts.mutations_node.copy()
        self.edge_logconst = np.zeros(ts.num_edges)
        self.block_logconst = np.zeros(num_blocks)
        self.node_scale = np.ones(ts.num_nodes)

        # terminal nodes
        has_parent = np.full(ts.num_nodes, False)
        has_child = np.full(ts.num_nodes, False)
        has_parent[self.edge_children] = True
        has_child[self.edge_parents] = True
        self.roots = np.logical_and(has_child, ~has_parent)
        self.leaves = np.logical_and(~has_child, has_parent)
        if np.any(np.logical_and(~has_child, ~has_parent)):
            raise ValueError("Tree sequence contains disconnected nodes")

        # edge traversal order
        edge_unphased = np.full(ts.num_edges, False)
        edge_unphased[self.block_edges[:, 0]] = True
        edge_unphased[self.block_edges[:, 1]] = True
        edges = np.arange(ts.num_edges, dtype=np.int32)[~edge_unphased]
        self.edge_order = np.concatenate((edges[:-1], np.flip(edges)))
        self.edge_weights = edge_sampling_weight(
            self.leaves,
            self.edge_parents,
            self.edge_children,
            ts.edges_left,
            ts.edges_right,
            ts.indexes_edge_insertion_order,
            ts.indexes_edge_removal_order,
        )
        self.block_order = np.arange(num_blocks, dtype=np.int32)
        self.mutation_order = np.arange(ts.num_mutations, dtype=np.int32)

    @staticmethod
    @numba.njit(_f(_i1r, _i1r, _i1r, _f2r, _f2r, _f2w, _f3w, _f1w, _f1w, _f, _f, _b))
    def propagate_likelihood(
        edge_order,
        edges_parent,
        edges_child,
        likelihoods,
        constraints,
        posterior,
        factors,
        lognorm,
        scale,
        max_shape,
        min_step,
        unphased,
    ):
        """
        Update approximating factors for Poisson mutation likelihoods on edges.

        :param ndarray edges_parent: integer array of parent ids per edge
        :param ndarray edges_child: integer array of child ids per edge
        :param ndarray likelihoods: array of dimension `[num_edges, 2]`
            containing mutation count and mutational target size per edge.
        :param ndarray constraints: array of dimension `[num_nodes, 2]`
            containing lower and upper bounds for each node.
        :param ndarray posterior: array of dimension `[num_nodes, 2]`
            containing natural parameters for each node, updated in-place.
        :param ndarray factors: array of dimension `[num_edges, 2, 2]`
            containing parent and child factors (natural parameters) for each
            edge, updated in-place.
        :param ndarray lognorm: array of dimension `[num_edges]`
            containing the approximate normalizing constants per edge,
            updated in-place.
        :param ndarray scale: array of dimension `[num_nodes]` containing a
            scaling factor for the posteriors, updated in-place.
        :param float max_shape: the maximum allowed shape for node posteriors.
        :param float min_step: the minimum allowed step size in (0, 1).
        :param bool unphased: if True, edges are treated as blocks of unphased
            singletons in contemporary individuals
        """

        assert constraints.shape == posterior.shape
        assert edges_child.size == edges_parent.size
        assert factors.shape == (edges_parent.size, 2, 2)
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

        for i in edge_order:
            p, c = edges_parent[i], edges_child[i]
            if fixed[p] and fixed[c]:
                continue
            elif fixed[p] and not fixed[c]:
                # in practice this should only occur if a sample is the
                # ancestor of another sample
                child_message = factors[i, LEAFWARD] * scale[c]
                child_delta = cavity_damping(posterior[c], child_message)
                child_cavity = posterior[c] - child_delta * child_message
                edge_likelihood = child_delta * likelihoods[i]
                parent_age = constraints[p, LOWER]
                lognorm[i], posterior[c] = leafward_projection(
                    parent_age,
                    child_cavity,
                    edge_likelihood,
                )
                factors[i, LEAFWARD] *= 1.0 - child_delta
                factors[i, LEAFWARD] += (posterior[c] - child_cavity) / scale[c]
                child_eta = posterior_damping(posterior[c])
                posterior[c] *= child_eta
                scale[c] *= child_eta
            elif fixed[c] and not fixed[p]:
                # in practice this should only occur if a sample has age
                # greater than zero
                parent_message = factors[i, ROOTWARD] * scale[p]
                parent_delta = cavity_damping(posterior[p], parent_message)
                parent_cavity = posterior[p] - parent_delta * parent_message
                edge_likelihood = parent_delta * likelihoods[i]
                child_age = constraints[c, LOWER]
                lognorm[i], posterior[p] = rootward_projection(
                    child_age,
                    parent_cavity,
                    edge_likelihood,
                )
                factors[i, ROOTWARD] *= 1.0 - parent_delta
                factors[i, ROOTWARD] += (posterior[p] - parent_cavity) / scale[p]
                parent_eta = posterior_damping(posterior[p])
                posterior[p] *= parent_eta
                scale[p] *= parent_eta
            else:
                if p == c:  # singleton block with single parent
                    parent_message = factors[i, ROOTWARD] * scale[p]
                    parent_delta = cavity_damping(posterior[p], parent_message)
                    parent_cavity = posterior[p] - parent_delta * parent_message
                    edge_likelihood = parent_delta * likelihoods[i]
                    child_age = constraints[c, LOWER]
                    lognorm[i], posterior[p] = \
                        twin_projection(parent_cavity, edge_likelihood)  # fmt: skip
                    factors[i, ROOTWARD] *= 1.0 - parent_delta
                    factors[i, ROOTWARD] += (posterior[p] - parent_cavity) / scale[p]
                    parent_eta = posterior_damping(posterior[p])
                    posterior[p] *= parent_eta
                    scale[p] *= parent_eta
                else:
                    # lower-bound cavity
                    parent_message = factors[i, ROOTWARD] * scale[p]
                    child_message = factors[i, LEAFWARD] * scale[c]
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
                    factors[i, ROOTWARD] *= 1.0 - delta
                    factors[i, ROOTWARD] += (posterior[p] - parent_cavity) / scale[p]
                    factors[i, LEAFWARD] *= 1.0 - delta
                    factors[i, LEAFWARD] += (posterior[c] - child_cavity) / scale[c]

                    # upper-bound posterior
                    parent_eta = posterior_damping(posterior[p])
                    child_eta = posterior_damping(posterior[c])
                    posterior[p] *= parent_eta
                    posterior[c] *= child_eta
                    scale[p] *= parent_eta
                    scale[c] *= child_eta

        return np.nan

    @staticmethod
    @numba.njit(_f(_b1r, _f2w, _f3w, _f1w, _f, _i, _f))
    def propagate_prior(
        free, posterior, factors, scale, max_shape, em_maxitt, em_reltol
    ):
        """
        Update approximating factors for global prior.

        :param ndarray free: boolean array indicating if prior should be
            applied to node
        :param ndarray penalty: initial value for regularisation penalty
        :param ndarray posterior: rows are nodes, columns are first and
            second natural parameters of gamma posteriors. Updated in
            place.
        :param ndarray factors: rows are nodes, columns index different
            types of updates. Updated in place.
        :param ndarray scale: array of dimension `[num_nodes]` containing a
            scaling factor for the posteriors, updated in-place.
        :param float max_shape: the maximum allowed shape for node posteriors.
        :param int em_maxitt: the maximum number of EM iterations to use when
            fitting the regularisation.
        :param int em_reltol: the termination criterion for relative change in
            log-likelihood.
        """

        assert free.size == posterior.shape[0]
        assert factors.shape == (free.size, 2, 2)
        assert scale.size == free.size
        assert max_shape >= 1.0

        def posterior_damping(x):
            return _rescale(x, max_shape)

        # fit an exponential to cavity distributions for unconstrained nodes
        cavity = posterior - factors[:, MIXPRIOR] * scale[:, np.newaxis]
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
        factors[free, MIXPRIOR] = \
            (posterior[free] - cavity[free]) / scale[free, np.newaxis]  # fmt: skip
        for i in np.flatnonzero(free):
            eta = posterior_damping(posterior[i])
            posterior[i] *= eta
            scale[i] *= eta

        return np.nan

    @staticmethod
    @numba.njit(
        _f(_i1r, _f2w, _f1w, _i1r, _i1r, _i1r, _f2r, _f2r, _f2r, _f3r, _f1r, _b)
    )
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
        scale,
        unphased,
    ):
        """
        Calculate posteriors for mutations.

        :param ndarray mutations_order: integer array giving order in
            which to traverse mutations
        :param ndarray mutations_posterior: array of dimension `(num_mutations, 2)`
            containing natural parameters for each mutation, modified in place
        :param ndarray mutations_phase: array of dimension `(num_mutations, )`
            containing mutation phase, modified in place
        :param ndarray mutations_edge: integer array giving edge for each
            mutation
        :param ndarray edges_parent: integer array of parent ids per edge
        :param ndarray edges_child: integer array of child ids per edge
        :param ndarray likelihoods: array of dimension `[num_edges, 2]`
            containing mutation count and mutational target size per edge.
        :param ndarray constraints: array of dimension `[num_nodes, 2]`
            containing lower and upper bounds for each node.
        :param ndarray posterior: array of dimension `[num_nodes, 2]`
            containing natural parameters for each node
        :param ndarray factors: array of dimension `[num_edges, 2, 2]`
            containing parent and child factors (natural parameters) for each
            edge
        :param ndarray scale: array of dimension `(num_nodes, )` containing a
            scaling factor for the posteriors
        :param bool unphased: if True, edges are treated as blocks of unphased
            singletons in contemporary individuals
        """

        # TODO: scale should be 1.0, can we delete
        # TODO: we don't seem to need to damp?

        # TODO: assert more stuff here?
        assert mutations_phase.size == mutations_edge.size
        assert mutations_posterior.shape == (mutations_phase.size, 2)
        assert constraints.shape == posterior.shape
        assert edges_child.size == edges_parent.size
        assert factors.shape == (edges_parent.size, 2, 2)
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
                child_message = factors[i, LEAFWARD] * scale[c]
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
                parent_message = factors[i, ROOTWARD] * scale[p]
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
                    parent_message = factors[i, ROOTWARD] * scale[p]
                    parent_delta = 1.0  # hopefully we don't need to damp
                    parent_cavity = posterior[p] - parent_delta * parent_message
                    edge_likelihood = parent_delta * likelihoods[i]
                    child_age = constraints[c, LOWER]
                    mutations_phase[m], mutations_posterior[m] = \
                        twin_projection(parent_cavity, edge_likelihood)  # fmt: skip
                else:
                    parent_message = factors[i, ROOTWARD] * scale[p]
                    child_message = factors[i, LEAFWARD] * scale[c]
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

        return np.nan

    @staticmethod
    @numba.njit(_void(_i1r, _i1r, _i2r, _f3w, _f3w, _f3w, _f1w))
    def rescale_factors(
        edges_parent,
        edges_child,
        block_nodes,
        node_factors,
        edge_factors,
        block_factors,
        scale,
    ):
        """Incorporate scaling term into factors and reset"""
        p, c = edges_parent, edges_child
        j, k = block_nodes
        edge_factors[:, ROOTWARD] *= scale[p, np.newaxis]
        edge_factors[:, LEAFWARD] *= scale[c, np.newaxis]
        block_factors[:, ROOTWARD] *= scale[j, np.newaxis]
        block_factors[:, LEAFWARD] *= scale[k, np.newaxis]
        node_factors[:, MIXPRIOR] *= scale[:, np.newaxis]
        node_factors[:, CONSTRNT] *= scale[:, np.newaxis]
        scale[:] = 1.0

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
        # pass through singleton blocks
        self.propagate_likelihood(
            self.block_order,
            self.block_nodes[ROOTWARD],
            self.block_nodes[LEAFWARD],
            self.block_likelihoods,
            self.node_constraints,
            self.node_posterior,
            self.block_factors,
            self.block_logconst,
            self.node_scale,
            max_shape,
            min_step,
            USE_BLOCK_LIKELIHOOD,
        )

        # rootward + leafward pass through edges
        self.propagate_likelihood(
            self.edge_order,
            self.edge_parents,
            self.edge_children,
            self.edge_likelihoods,
            self.node_constraints,
            self.node_posterior,
            self.edge_factors,
            self.edge_logconst,
            self.node_scale,
            max_shape,
            min_step,
            USE_EDGE_LIKELIHOOD,
        )

        # exponential regularization on roots
        if regularise:
            self.propagate_prior(
                self.roots,
                self.node_posterior,
                self.node_factors,
                self.node_scale,
                max_shape,
                em_maxitt,
                em_reltol,
            )

        # absorb the scaling term into the factors
        self.rescale_factors(
            self.edge_parents,
            self.edge_children,
            self.block_nodes,
            self.node_factors,
            self.edge_factors,
            self.block_factors,
            self.node_scale,
        )

        if check_valid:  # for debugging
            self._check_valid_state(
                self.edge_parents,
                self.edge_children,
                self.block_nodes,
                self.node_posterior,
                self.node_factors,
                self.edge_factors,
                self.block_factors,
            )

        return np.nan  # TODO: placeholder for marginal likelihood

    def rescale(
        self,
        *,
        rescale_intervals=1000,
        rescale_segsites=False,
        use_median=False,
        quantile_width=0.5,
    ):
        """Normalise posteriors so that empirical mutation rate is constant"""
        likelihoods = self.edge_likelihoods if rescale_segsites \
            else self.sizebiased_likelihoods  # fmt: skip
        nodes_time = self._point_estimate(
            self.node_posterior, self.node_constraints, use_median
        )
        reallocate_unphased(  # correct mutation counts for unphased singletons
            likelihoods,
            self.mutation_phase,
            self.mutation_blocks,
            self.block_edges,
        )
        original_breaks, rescaled_breaks = mutational_timescale(
            nodes_time,
            likelihoods,
            self.node_constraints,
            self.edge_parents,
            self.edge_children,
            rescale_intervals,
        )
        self.node_posterior[:] = piecewise_scale_posterior(
            self.node_posterior,
            original_breaks,
            rescaled_breaks,
            quantile_width,
            use_median,
        )
        self.mutation_posterior[:] = piecewise_scale_posterior(
            self.mutation_posterior,
            original_breaks,
            rescaled_breaks,
            quantile_width,
            use_median,
        )

    def run(
        self,
        *,
        ep_iterations=10,
        max_shape=1000,
        min_step=0.1,
        rescale_intervals=1000,
        rescale_segsites=False,
        rescale_iterations=5,
        regularise=True,
        progress=None,
    ):
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
        nodes_timing -= time.time()
        skipped_edges = np.sum(np.isnan(self.edge_logconst))
        logger.info(f"Skipped {skipped_edges} edges with invalid factors")
        logger.info(f"Calculated node posteriors in {abs(nodes_timing)} seconds")

        muts_timing = time.time()
        mutations_phased = self.mutation_blocks == tskit.NULL
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
            self.block_factors,
            self.node_scale,
            USE_BLOCK_LIKELIHOOD,
        )
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
            self.edge_factors,
            self.node_scale,
            USE_EDGE_LIKELIHOOD,
        )
        muts_timing -= time.time()
        skipped_muts = np.sum(np.isnan(self.mutation_posterior[:, 0]))
        logger.info(f"Skipped {skipped_muts} mutations with invalid posteriors")
        logger.info(f"Calculated mutation posteriors in {abs(muts_timing)} seconds")

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
            for _ in tqdm(
                np.arange(rescale_iterations),
                desc="Path Rescaling",
                disable=not progress,
            ):
                self.rescale(
                    rescale_intervals=rescale_intervals,
                    rescale_segsites=rescale_segsites,
                )
            rescale_timing -= time.time()
            logger.info(f"Timescale rescaled in {abs(rescale_timing)} seconds")

    def node_moments(self):
        """
        Posterior mean and variance of node ages
        """
        alpha, beta = self.node_posterior.T
        nodes_mn = np.ascontiguousarray(self.node_constraints[:, 0])
        nodes_va = np.zeros(nodes_mn.size)
        free = self.node_constraints[:, 0] != self.node_constraints[:, 1]
        nodes_mn[free] = (alpha[free] + 1) / beta[free]
        nodes_va[free] = nodes_mn[free] / beta[free]
        return nodes_mn, nodes_va

    def mutation_moments(self):
        """
        Posterior mean and variance of mutation ages
        """
        alpha, beta = self.mutation_posterior.T
        muts_mn = np.full(alpha.size, np.nan)
        muts_va = np.full(alpha.size, np.nan)
        free = np.isfinite(alpha)
        muts_mn[free] = (alpha[free] + 1) / beta[free]
        muts_va[free] = muts_mn[free] / beta[free]
        return muts_mn, muts_va

    def mutation_mapping(self):
        """
        Map from mutations to edges and subtended node, using estimated singleton
        phase (if singletons were unphased)
        """
        # TODO: should these be copies? Should members be readonly?
        return self.mutation_edges, self.mutation_nodes


# def date(
#     ts,
#     *,
#     mutation_rate,
#     singletons_phased=True,
#     max_iterations=10,
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
#     posterior.run(
#         ep_maxitt=max_iterations,
#         max_shape=max_shape,
#         rescale_intervals=rescaling_intervals,
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
#     tables.nodes.time = constrain_ages(
#       ts, node_mn, constr_iterations=constr_iterations)
#     tables.mutations.node = mutation_node
#     tables.sort()
#
#     return tables.tree_sequence(), posterior
