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
import numba
import numpy as np
from numba.types import void as _void
from tqdm.auto import tqdm

from . import approx
from . import mixture
from .approx import _b
from .approx import _f
from .approx import _f1r
from .approx import _f1w
from .approx import _f2r
from .approx import _f2w
from .approx import _f3w
from .approx import _i
from .approx import _i1r


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
    def _check_valid_inputs(ts, likelihoods, constraints, prior):
        if likelihoods.shape != (ts.num_edges, 2):
            raise ValueError("Edge likelihoods are the wrong shape")
        if constraints.shape != (ts.num_nodes, 2):
            raise ValueError("Node age constraints are the wrong shape")
        if np.any(likelihoods < 0.0):
            raise ValueError("Edge likelihoods contains negative values")
        if np.any(constraints < 0.0):
            raise ValueError("Node age constraints contain negative values")
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

    def __init__(self, ts, likelihoods, constraints, prior):
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
        :param ~np.ndarray prior: a `K`-by-three array containing parameters of
            an i.i.d. gamma mixture prior with `K` components, used for all
            nonfixed nodes. Each row contains the weight, shape, and rate
            for a mixture component.
        """

        self._check_valid_inputs(ts, likelihoods, constraints, prior)

        # const
        self.parents = ts.edges_parent
        self.children = ts.edges_child
        self.likelihoods = likelihoods
        self.constraints = constraints

        # mutable
        self.prior = prior.copy()
        self.node_factors = np.zeros((ts.num_nodes, 2, 2))
        self.edge_factors = np.zeros((ts.num_edges, 2, 2))
        # self.node_lognorm = np.zeros(ts.num_nodes)
        # self.edge_lognorm = np.zeros(ts.num_nodes)
        self.posterior = np.zeros((ts.num_nodes, 2))
        self.log_partition = np.zeros(ts.num_edges)
        self.scale = np.ones(ts.num_nodes)

        # get edge traversal order
        node_is_fixed = constraints[:, LOWER] == constraints[:, UPPER]
        child_is_contemporary = np.logical_and(
            constraints[ts.edges_child, LOWER] == 0.0,
            node_is_fixed[ts.edges_child],
        )
        edges = np.arange(ts.num_edges, dtype=np.int32)
        contemp = edges[child_is_contemporary]
        noncontemp = edges[~child_is_contemporary]
        self.edge_order = np.concatenate(  # rootward + leafward
            (noncontemp[:-1], np.flip(noncontemp))
        )

        # edges attached to contemporary nodes are visited once
        for i in contemp:
            p, c = ts.edges_parent[i], ts.edges_child[i]
            assert np.all(constraints[c] == 0.0)
            self.edge_factors[i, ROOTWARD] = self.likelihoods[i]
            self.posterior[p] += self.likelihoods[i]
            # self.node_lognorm[i] += ... # TODO

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

        TODO: return max difference in natural parameters for stopping criterion

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
    @numba.njit(_f(_f2w, _f2w, _f2w, _f3w, _f1w, _f, _i, _f))
    def propagate_prior(
        prior, constraints, posterior, factors, scale, max_shape, em_maxitt, em_reltol
    ):
        """
        Update approximating factors for global prior at each node.

        :param ndarray constraints: rows are nodes, columns are upper and
            lower bounds for node age.
        :param ndarray prior: rows are mixture components, columns are
            zeroth, first, and second natural parameters of gamma mixture
            components. Updated in place.
        :param ndarray posterior: rows are nodes, columns are first and
            second natural parameters of gamma posteriors. Updated in
            place.
        :param ndarray factors: rows are nodes, columns index different
            types of updates. Updated in place.
        :param ndarray scale: array of dimension `[num_nodes]` containing a
            scaling factor for the posteriors, updated in-place.
        :param float max_shape: the maximum allowed shape for node posteriors.
        :param int em_maxitt: the maximum number of EM iterations to use when
            fitting the mixture model.
        :param int em_reltol: the termination criterion for relative change in
            log-likelihood.
        """

        assert prior.shape[1] == 3
        assert constraints.shape == posterior.shape
        assert factors.shape == (constraints.shape[0], 2, 2)
        assert scale.size == constraints.shape[0]
        assert max_shape >= 1.0

        if prior.shape[0] == 0:
            return 0.0

        def posterior_damping(x):
            return _rescale(x, max_shape)

        lognorm = np.zeros(constraints.shape[0])  # TODO: move to member

        # fit a mixture-of-gamma model to cavity distributions for unconstrained nodes
        free = np.logical_and(
            constraints[:, LOWER] == 0.0, constraints[:, UPPER] == np.inf
        )
        cavity = posterior - factors[:, MIXPRIOR] * scale[:, np.newaxis]
        prior[:], posterior[free], lognorm[free] = mixture.fit_gamma_mixture(
            prior, cavity[free], em_maxitt, em_reltol, False
        )

        # reset nodes that were skipped (b/c of improper posteriors)
        skipped = np.logical_and(free, ~np.isfinite(lognorm))
        posterior[skipped] = (
            cavity[skipped] + factors[skipped, MIXPRIOR] * scale[skipped, np.newaxis]
        )

        # the remaining nodes may be updated
        updated = np.logical_and(free, np.isfinite(lognorm))
        factors[updated, MIXPRIOR] = (posterior[updated] - cavity[updated]) / scale[
            updated, np.newaxis
        ]

        # rescale posterior to keep shape bounded
        for i in np.flatnonzero(free):
            eta = posterior_damping(posterior[i])
            posterior[i] *= eta
            scale[i] *= eta

        return np.nan

    @staticmethod
    @numba.njit(_f(_f2r, _f2w, _f3w, _f1w, _f, _f, _b))
    def propagate_constraints(
        constraints, posterior, factors, scale, max_shape, min_step, min_kl
    ):
        """
        Update approximating factors for node age constraints (indicator
        functions) at each node.

        :param ndarray constraints: rows are nodes, columns are lower and
            upper bounds for age.
        :param ndarray posterior: rows are nodes, columns are first and
            second natural parameters of gamma posteriors. Updated in
            place.
        :param ndarray factors: rows are edges, columns are different
            types of updates. Updated in place.
        :param ndarray scale: array of dimension `[num_nodes]` containing a
            scaling factor for the posteriors, updated in-place.
        :param float max_shape: the maximum allowed shape for node posteriors.
        :param float min_step: the minimum allowed step size in (0, 1).
        :param bool min_kl: minimize KL divergence or match central moments.
        """

        assert constraints.shape == posterior.shape
        assert factors.shape == (constraints.shape[0], 2, 2)
        assert scale.size == constraints.shape[0]
        assert max_shape >= 1.0
        assert 0.0 < min_step < 1.0

        def cavity_damping(x, y):
            return _damp(x, y, min_step)

        def posterior_damping(x):
            return _rescale(x, max_shape)

        lognorm = np.zeros(constraints.shape[0])  # TODO: move to member

        bounded = np.logical_or(
            constraints[:, LOWER] > 0.0, constraints[:, UPPER] < np.inf
        )

        for i in np.flatnonzero(bounded):
            if constraints[i, LOWER] == constraints[i, UPPER]:
                continue
            message = factors[i, CONSTRNT] * scale[i]
            delta = cavity_damping(posterior[i], message)
            cavity = posterior[i] - delta * message
            lognorm[i], posterior[i] = approx.truncated_projection(
                constraints[i], cavity, min_kl
            )
            factors[i, CONSTRNT] *= 1.0 - delta
            factors[i, CONSTRNT] += (posterior[i] - cavity) / scale[i]
            eta = posterior_damping(posterior[i])
            posterior[i] *= eta
            scale[i] *= eta

        return np.nan

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
        self, em_maxitt=10, em_reltol=1e-6, max_shape=1000, min_step=0.1, min_kl=True
    ):
        """
        Update approximating factors.

        Returns the approximate log marginal likelihood (TODO)
        """

        # mixture prior over unconstrained nodes
        self.propagate_prior(
            self.prior,
            self.constraints,
            self.posterior,
            self.node_factors,
            self.scale,
            max_shape,
            em_maxitt,
            em_reltol,
        )

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

        # upper and lower bounds on node age
        # self.propagate_constraints(
        #     self.constraints,
        #     self.posterior,
        #     self.node_factors,
        #     self.scale,
        #     max_shape,
        #     min_step,
        #     min_kl,
        # )

        # absorb the scaling term into the factors
        self.rescale_factors(
            self.parents,
            self.children,
            self.node_factors,
            self.edge_factors,
            self.scale,
        )

        # for debugging
        # assert self._check_valid_state(
        #    self.parents, self.children, self.posterior,
        #    self.node_factors, self.edge_factors,
        # )

        return np.nan  # TODO: placeholder for marginal likelihood

    def run(
        self,
        *,
        ep_maxitt=20,
        em_maxitt=10,
        max_shape=1000,
        min_step=0.1,
        min_kl=True,
        progress=None
    ):
        for itt in tqdm(
            np.arange(ep_maxitt),
            desc="Expectation Propagation",
            disable=not progress,
        ):
            self.iterate(
                em_maxitt=em_maxitt if itt else 0,
                max_shape=max_shape,
                min_step=min_step,
                min_kl=min_kl,
            )
