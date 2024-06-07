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
from .hypergeo import _gammainc_inv as gammainc_inv
from .rescaling import edge_sampling_weight
from .rescaling import mutational_timescale
from .rescaling import piecewise_scale_posterior
from .util import contains_unary_nodes


# columns for edge_factors
ROOTWARD = 0  # edge likelihood to parent
LEAFWARD = 1  # edge likelihood to child

# columns for node_factors
MIXPRIOR = 0  # mixture prior to node
CONSTRNT = 1  # bounds on node ages

# columns for constraints
LOWER = 0  # lower bound on node
UPPER = 1  # upper bound on node


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
    def _check_valid_inputs(ts, likelihoods, constraints, mutations_edge):
        if contains_unary_nodes(ts):
            raise ValueError(
                "Tree sequence contains unary nodes, simplify before dating"
            )
        if likelihoods.shape != (ts.num_edges, 2):
            raise ValueError("Edge likelihoods are the wrong shape")
        if constraints.shape != (ts.num_nodes, 2):
            raise ValueError("Node age constraints are the wrong shape")
        if np.any(likelihoods < 0.0):
            raise ValueError("Edge likelihoods contains negative values")
        if np.any(constraints < 0.0):
            raise ValueError("Node age constraints contain negative values")
        if mutations_edge.size > 0 and mutations_edge.max() >= ts.num_edges:
            raise ValueError("Mutation edge indices are out-of-bounds")
        ExpectationPropagation._check_valid_constraints(
            constraints, ts.edges_parent, ts.edges_child
        )

    @staticmethod
    def _check_valid_state(
        edges_parent, edges_child, posterior, node_factors, edge_factors
    ):
        """Check that the messages sum to the posterior (debugging only)"""
        posterior_check = np.zeros(posterior.shape)
        for i, (p, c) in enumerate(zip(edges_parent, edges_child)):
            posterior_check[p] += edge_factors[i, ROOTWARD]
            posterior_check[c] += edge_factors[i, LEAFWARD]
        posterior_check += node_factors[:, MIXPRIOR]
        posterior_check += node_factors[:, CONSTRNT]
        return np.allclose(posterior_check, posterior)

    @staticmethod
    @numba.njit(_f1w(_f2r, _f2r, _b))
    def _point_estimate(posteriors, constraints, median):
        assert posteriors.shape == constraints.shape
        fixed = constraints[:, 0] == constraints[:, 1]
        point_estimate = np.zeros(posteriors.shape[0])
        for i in np.flatnonzero(~fixed):
            alpha, beta = posteriors[i]
            point_estimate[i] = gammainc_inv(alpha + 1, 0.5) if median else (alpha + 1)
            point_estimate[i] /= beta
        point_estimate[fixed] = constraints[fixed, 0]
        return point_estimate

    def __init__(self, ts, likelihoods, constraints, mutations_edge):
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
        :param ~np.ndarray constraints: a `ts.num_nodes`-by-two array containing
            lower and upper bounds for each node. If lower and upper bounds
            are the same value, the node is considered fixed.
        :param ~np.ndarray likelihoods: a `ts.num_edges`-by-two array containing
            mutation counts and mutational spans (e.g. edge span multiplied by
            mutation rate) per edge.
        :param ~np.ndarray mutations_edge: an array containing edge indices
            (one per mutation) for which to compute posteriors.
        """

        # TODO: pass in edge table rather than tree sequence
        # TODO: check valid mutations_edge
        self._check_valid_inputs(ts, likelihoods, constraints, mutations_edge)

        # const
        self.parents = ts.edges_parent
        self.children = ts.edges_child
        self.likelihoods = likelihoods
        self.constraints = constraints
        self.mutations_edge = mutations_edge

        # mutable
        self.node_factors = np.zeros((ts.num_nodes, 2, 2))
        self.edge_factors = np.zeros((ts.num_edges, 2, 2))
        self.posterior = np.zeros((ts.num_nodes, 2))
        self.log_partition = np.zeros(ts.num_edges)
        self.scale = np.ones(ts.num_nodes)

        # terminal nodes
        has_parent = np.full(ts.num_nodes, False)
        has_child = np.full(ts.num_nodes, False)
        has_parent[self.children] = True
        has_child[self.parents] = True
        self.roots = np.logical_and(has_child, ~has_parent)
        self.leaves = np.logical_and(~has_child, has_parent)
        if np.any(np.logical_and(~has_child, ~has_parent)):
            raise ValueError("Tree sequence contains disconnected nodes")

        # edge traversal order
        edges = np.arange(ts.num_edges, dtype=np.int32)
        self.edge_order = np.concatenate((edges[:-1], np.flip(edges)))
        self.edge_weights = edge_sampling_weight(
            self.leaves,
            ts.edges_parent,
            ts.edges_child,
            ts.edges_left,
            ts.edges_right,
            ts.indexes_edge_insertion_order,
            ts.indexes_edge_removal_order,
        )

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
        min_kl,
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
        :param bool min_kl: minimize KL divergence or match central moments.
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

                # match moments and update factor
                parent_age = constraints[p, LOWER]
                lognorm[i], posterior[c] = approx.leafward_projection(
                    parent_age, child_cavity, edge_likelihood, min_kl
                )
                factors[i, LEAFWARD] *= 1.0 - child_delta
                factors[i, LEAFWARD] += (posterior[c] - child_cavity) / scale[c]

                # upper bound posterior
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

                # match moments and update factor
                child_age = constraints[c, LOWER]
                lognorm[i], posterior[p] = approx.rootward_projection(
                    child_age, parent_cavity, edge_likelihood, min_kl
                )

                factors[i, ROOTWARD] *= 1.0 - parent_delta
                factors[i, ROOTWARD] += (posterior[p] - parent_cavity) / scale[p]

                # upper-bound posterior
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
                lognorm[i], posterior[p], posterior[c] = approx.gamma_projection(
                    parent_cavity, child_cavity, edge_likelihood, min_kl
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
        factors[free, MIXPRIOR] = (posterior[free] - cavity[free]) / scale[
            free, np.newaxis
        ]
        for i in np.flatnonzero(free):
            eta = posterior_damping(posterior[i])
            posterior[i] *= eta
            scale[i] *= eta

        return np.nan

    @staticmethod
    @numba.njit(_f2w(_i1r, _i1r, _i1r, _f2r, _f2r, _f2r, _f3r, _f1r, _b))
    def propagate_mutations(
        mutations_edge,
        edges_parent,
        edges_child,
        likelihoods,
        constraints,
        posterior,
        factors,
        scale,
        min_kl,
    ):
        """
        Calculate posteriors for mutations.

        :param ndarray mutations_edge: integer array giving edge for each
            mutation
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
        :param ndarray scale: array of dimension `[num_nodes]` containing a
            scaling factor for the posteriors, updated in-place.
        :param bool min_kl: minimize KL divergence or match central moments.
        """

        # TODO: scale should be 1.0, can we delete
        # TODO: we don't seem to need to damp?
        # TODO: might as well copy format in other functions and have void return

        assert constraints.shape == posterior.shape
        assert edges_child.size == edges_parent.size
        assert factors.shape == (edges_parent.size, 2, 2)
        assert likelihoods.shape == (edges_parent.size, 2)

        mutations_posterior = np.zeros((mutations_edge.size, 2))
        fixed = constraints[:, LOWER] == constraints[:, UPPER]
        for m, i in enumerate(mutations_edge):
            if i == tskit.NULL:  # skip mutations above root
                mutations_posterior[m] = np.nan
                continue
            p, c = edges_parent[i], edges_child[i]
            if fixed[p] and fixed[c]:
                child_age = constraints[c, 0]
                parent_age = constraints[p, 0]
                mean = 1 / 2 * (child_age + parent_age)
                variance = 1 / 12 * (parent_age - child_age) ** 2
                mutations_posterior[m] = approx.approximate_gamma_mom(mean, variance)
            elif fixed[p] and not fixed[c]:
                child_message = factors[i, LEAFWARD] * scale[c]
                child_delta = 1.0  # hopefully we don't need to damp
                child_cavity = posterior[c] - child_delta * child_message
                edge_likelihood = child_delta * likelihoods[i]
                parent_age = constraints[p, LOWER]
                mutations_posterior[m] = approx.mutation_leafward_projection(
                    parent_age, child_cavity, edge_likelihood, min_kl
                )
            elif fixed[c] and not fixed[p]:
                parent_message = factors[i, ROOTWARD] * scale[p]
                parent_delta = 1.0  # hopefully we don't need to damp
                parent_cavity = posterior[p] - parent_delta * parent_message
                edge_likelihood = parent_delta * likelihoods[i]
                child_age = constraints[c, LOWER]
                mutations_posterior[m] = approx.mutation_rootward_projection(
                    child_age, parent_cavity, edge_likelihood, min_kl
                )
            else:
                parent_message = factors[i, ROOTWARD] * scale[p]
                child_message = factors[i, LEAFWARD] * scale[c]
                parent_delta = 1.0  # hopefully we don't need to damp
                child_delta = 1.0  # hopefully we don't need to damp
                delta = min(parent_delta, child_delta)
                parent_cavity = posterior[p] - delta * parent_message
                child_cavity = posterior[c] - delta * child_message
                edge_likelihood = delta * likelihoods[i]
                mutations_posterior[m] = approx.mutation_gamma_projection(
                    parent_cavity, child_cavity, edge_likelihood, min_kl
                )

        return mutations_posterior

    @staticmethod
    @numba.njit(_void(_i1r, _i1r, _f3w, _f3w, _f1w))
    def rescale_factors(edges_parent, edges_child, node_factors, edge_factors, scale):
        """Incorporate scaling term into factors and reset"""
        p, c = edges_parent, edges_child
        edge_factors[:, ROOTWARD] *= scale[p, np.newaxis]
        edge_factors[:, LEAFWARD] *= scale[c, np.newaxis]
        node_factors[:, MIXPRIOR] *= scale[:, np.newaxis]
        node_factors[:, CONSTRNT] *= scale[:, np.newaxis]
        scale[:] = 1.0

    def iterate(
        self,
        *,
        max_shape=1000,
        min_step=0.1,
        em_maxitt=100,
        em_reltol=1e-8,
        min_kl=False,
        regularise=True,
        check_valid=False,
    ):
        # rootward + leafward pass through edges
        self.propagate_likelihood(
            self.edge_order,
            self.parents,
            self.children,
            self.likelihoods,
            self.constraints,
            self.posterior,
            self.edge_factors,
            self.log_partition,
            self.scale,
            max_shape,
            min_step,
            min_kl,
        )

        # exponential regularization on roots
        if regularise:
            self.propagate_prior(
                self.roots,
                self.posterior,
                self.node_factors,
                self.scale,
                max_shape,
                em_maxitt,
                em_reltol,
            )

        # absorb the scaling term into the factors
        self.rescale_factors(
            self.parents,
            self.children,
            self.node_factors,
            self.edge_factors,
            self.scale,
        )

        if check_valid:  # for debugging
            assert self._check_valid_state(
                self.parents,
                self.children,
                self.posterior,
                self.node_factors,
                self.edge_factors,
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
        edge_weights = (
            np.ones(self.edge_weights.size) if rescale_segsites else self.edge_weights
        )
        nodes_time = self._point_estimate(self.posterior, self.constraints, use_median)
        original_breaks, rescaled_breaks = mutational_timescale(
            nodes_time,
            self.likelihoods,
            self.constraints,
            self.parents,
            self.children,
            edge_weights,
            rescale_intervals,
        )
        self.posterior[:] = piecewise_scale_posterior(
            self.posterior,
            original_breaks,
            rescaled_breaks,
            quantile_width,
            use_median,
        )
        self.mutations_posterior[:] = piecewise_scale_posterior(
            self.mutations_posterior,
            original_breaks,
            rescaled_breaks,
            quantile_width,
            use_median,
        )

    def run(
        self,
        *,
        ep_maxitt=10,
        max_shape=1000,
        min_step=0.1,
        min_kl=False,
        rescale_intervals=1000,
        rescale_segsites=False,
        regularise=True,
        progress=None,
    ):
        nodes_timing = time.time()
        for _ in tqdm(
            np.arange(ep_maxitt),
            desc="Expectation Propagation",
            disable=not progress,
        ):
            self.iterate(
                max_shape=max_shape,
                min_step=min_step,
                min_kl=min_kl,
                regularise=regularise,
            )
        nodes_timing -= time.time()
        skipped_nodes = np.sum(np.isnan(self.log_partition))
        if skipped_nodes:
            logging.info(f"Skipped {skipped_nodes} nodes with invalid posteriors")
        logging.info(f"Calculated node posteriors in {abs(nodes_timing)} seconds")

        muts_timing = time.time()
        self.mutations_posterior = self.propagate_mutations(
            self.mutations_edge,
            self.parents,
            self.children,
            self.likelihoods,
            self.constraints,
            self.posterior,
            self.edge_factors,
            self.scale,
            min_kl,
        )
        muts_timing -= time.time()
        skipped_muts = np.sum(np.isnan(self.mutations_posterior[:, 0]))
        if skipped_muts:
            logging.info(f"Skipped {skipped_muts} mutations with invalid posteriors")
        logging.info(f"Calculated mutation posteriors in {abs(muts_timing)} seconds")

        if rescale_intervals > 0:
            rescale_timing = time.time()
            self.rescale(
                rescale_intervals=rescale_intervals, rescale_segsites=rescale_segsites
            )
            rescale_timing -= time.time()
            logging.info(f"Timescale rescaled in {abs(rescale_timing)} seconds")
