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

import _tsdate
import numpy as np
import tskit
from tqdm.auto import tqdm

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
    def _check_valid_constraints(constraints, edges_parent, edges_child):
        # Check that upper-bound on node age is greater than maximum lower-bound
        # for ages of descendants
        consistent = _tsdate.constraints_are_consistent(
            constraints=constraints, edges_parent=edges_parent, edges_child=edges_child
        )
        if not consistent:
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

    def _check_valid_state(self):
        # Check that the messages sum to the posterior (debugging only)
        posterior_check = self._ep.assemble_factors()
        np.testing.assert_allclose(posterior_check, self._ep.posterior)

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
        self._ep = _tsdate.ExpectationPropagation(
            constraints=self.node_constraints,
            edges_parent=self.edge_parents,
            edges_child=self.edge_children,
            edges_likelihood=self.edge_likelihoods,
            blocks_left=self.block_nodes[ROOTWARD],
            blocks_right=self.block_nodes[LEAFWARD],
            blocks_likelihood=self.block_likelihoods,
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
        self._ep.propagate_likelihood(
            order=self.block_order,
            unphased=USE_BLOCK_LIKELIHOOD,
            max_shape=max_shape,
            min_step=min_step,
        )

        logger.debug("Rootward + leafward pass through edges")
        self._ep.propagate_likelihood(
            order=self.edge_order,
            unphased=USE_EDGE_LIKELIHOOD,
            max_shape=max_shape,
            min_step=min_step,
        )

        if regularise:
            logger.debug("Exponential regularization on roots")
            self._ep.propagate_prior(
                free=self.unconstrained_roots,
                max_shape=max_shape,
                em_maxitt=em_maxitt,
                em_reltol=em_reltol,
            )

        logger.debug("Absorbing scaling term into the factors")
        self._ep.rescale_factors()
        self.node_posterior = self._ep.posterior
        self.edge_logconst = self._ep.edges_lognorm
        self.block_logconst = self._ep.blocks_lognorm

        if check_valid:  # for debugging
            self._check_valid_state()

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
        likelihoods = reallocate_unphased(  # correct counts for unphased singletons
            edges_likelihood=likelihoods,
            mutations_phase=self.mutation_phase,
            mutations_block=self.mutation_blocks,
            blocks_edges=self.block_edges,
        )
        nodes_fixed = self.node_constraints[:, 0] == self.node_constraints[:, 1]
        mutations_fixed = np.isnan(self.mutation_posterior[:, 0])
        nodes_time, _ = self.node_moments()
        rescaled_nodes_time = nodes_time.copy()
        for _ in np.arange(rescale_iterations):  # estimate time rescaling
            original_breaks, rescaled_breaks = mutational_timescale(
                nodes_time=rescaled_nodes_time,
                edges_likelihood=likelihoods,
                nodes_fixed=nodes_fixed,
                edges_parent=self.edge_parents,
                edges_child=self.edge_children,
                max_intervals=rescale_intervals,
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
        unphased = self.mutation_blocks != tskit.NULL
        logger.debug("Passing through unphased singletons")
        posterior, phase = self._ep.propagate_mutations(
            mutations_edge=self.mutation_blocks[unphased],
            unphased=USE_BLOCK_LIKELIHOOD,
        )
        self.mutation_posterior[unphased] = posterior
        self.mutation_phase[unphased] = phase
        logger.debug("Passing through phased mutations")
        posterior, phase = self._ep.propagate_mutations(
            mutations_edge=self.mutation_edges[~unphased],
            unphased=USE_EDGE_LIKELIHOOD,
        )
        self.mutation_posterior[~unphased] = posterior
        self.mutation_phase[~unphased] = phase
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
