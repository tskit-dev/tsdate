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
Infer the age of nodes conditional on a tree sequence topology.
"""
import functools
import itertools
import json
import logging
import multiprocessing
import operator
from collections import defaultdict

import numba
import numpy as np
import scipy.stats
import tskit
from tqdm import tqdm

from . import approx
from . import base
from . import prior
from . import provenance
from . import util

FORMAT_NAME = "tsdate"


class Likelihoods:
    """
    A class to store and process likelihoods. Likelihoods for edges are stored as a
    flattened lower triangular matrix of all the possible delta t's. This class also
    provides methods for accessing this lower triangular matrix, multiplying it, etc.

    If ``standardize`` is true, routines will operate to standardize the likelihoods
    such that their maximum is one (in linear space) or zero (in log space)
    """

    probability_space = base.LIN
    identity_constant = 1.0
    null_constant = 0.0

    def __init__(
        self,
        ts,
        timepoints,
        mutation_rate=None,
        recombination_rate=None,
        *,
        eps=0,
        fixed_node_set=None,
        standardize=True,
        progress=False,
    ):
        self.ts = ts
        self.timepoints = timepoints
        self.fixednodes = (
            set(ts.samples()) if fixed_node_set is None else fixed_node_set
        )
        self.mut_rate = mutation_rate
        self.rec_rate = recombination_rate
        self.standardize = standardize
        self.grid_size = len(timepoints)
        self.tri_size = self.grid_size * (self.grid_size + 1) / 2
        self.ll_mut = {}
        self.mut_edges = self.get_mut_edges(ts)
        self.progress = progress
        # Need to set eps properly in the 2 lines below, to account for values in the
        # same timeslice
        self.timediff_lower_tri = np.concatenate(
            [
                self.timepoints[time_index] - self.timepoints[0 : time_index + 1] + eps
                for time_index in np.arange(len(self.timepoints))
            ]
        )
        self.timediff = self.timepoints - self.timepoints[0] + eps

        # The mut_ll contains unpacked (1D) lower triangular matrices. We need to
        # index this by row and by column index.
        self.row_indices = []
        for time in range(self.grid_size):
            n = np.arange(self.grid_size)
            self.row_indices.append((((n * (n + 1)) // 2) + time)[time:])
        self.col_indices = []
        running_sum = 0  # use this to find the index of the last element of
        # each column in order to appropriately sum the vv by columns.
        for i in np.arange(self.grid_size):
            arr = np.arange(running_sum, running_sum + self.grid_size - i)
            index = arr[-1]
            running_sum = index + 1
            val = arr[0]
            self.col_indices.append(val)

        # These are used for transforming an array of grid_size into one of tri_size
        # By repeating elements over rows to form an upper or a lower triangular matrix
        self.to_lower_tri = np.concatenate(
            [np.arange(time_idx + 1) for time_idx in np.arange(self.grid_size)]
        )
        self.to_upper_tri = np.concatenate(
            [
                np.arange(time_idx, self.grid_size)
                for time_idx in np.arange(self.grid_size + 1)
            ]
        )

    @staticmethod
    def get_mut_edges(ts):
        """
        Get the number of mutations on each edge in the tree sequence.
        """
        mut_edges = np.zeros(ts.num_edges, dtype=np.int64)
        for m in ts.mutations():
            mut_edges[m.edge] += 1
        return mut_edges

    @staticmethod
    def _lik(muts, span, dt, mutation_rate, standardize=True):
        """
        The likelihood of an edge given a number of mutations, as set of time deltas (dt)
        and a span. This is a static function to allow parallelization
        """
        ll = scipy.stats.poisson.pmf(muts, dt * mutation_rate * span)
        if standardize:
            return ll / np.max(ll)
        else:
            return ll

    @staticmethod
    def _lik_wrapper(muts_span, dt, mutation_rate, standardize=True):
        """
        A wrapper to allow this _lik to be called by pool.imap_unordered, returning the
        mutation and span values
        """
        return muts_span, Likelihoods._lik(
            muts_span[0], muts_span[1], dt, mutation_rate, standardize=standardize
        )

    def precalculate_mutation_likelihoods(self, num_threads=None, unique_method=0):
        """
        We precalculate these because the pmf function is slow, but can be trivially
        parallelised. We store the likelihoods in a cache because they only depend on
        the number of mutations and the span, so can potentially be reused.

        However, we don't bother storing the likelihood for edges above a *fixed* node,
        because (a) these are only used once per node and (b) sample edges are often
        long, and hence their span will be unique. This also allows us to deal easily
        with fixed nodes at explicit times (rather than in time slices)
        """

        if self.mut_rate is None:
            raise RuntimeError(
                "Cannot calculate mutation likelihoods with no mutation_rate set"
            )
        if unique_method == 0:
            self.unfixed_likelihood_cache = {
                (muts, e.span): None
                for muts, e in zip(self.mut_edges, self.ts.edges())
                if e.child not in self.fixednodes
            }
        else:
            edges = self.ts.tables.edges
            fixed_nodes = np.array(list(self.fixednodes))
            keys = np.unique(
                np.core.records.fromarrays(
                    (self.mut_edges, edges.right - edges.left), names="muts,span"
                )[np.logical_not(np.isin(edges.child, fixed_nodes))]
            )
            if unique_method == 1:
                self.unfixed_likelihood_cache = dict.fromkeys({tuple(t) for t in keys})
            else:
                self.unfixed_likelihood_cache = {tuple(t): None for t in keys}

        if num_threads:
            f = functools.partial(  # Set constant values for params for static _lik
                self._lik_wrapper,
                dt=self.timediff_lower_tri,
                mutation_rate=self.mut_rate,
                standardize=self.standardize,
            )
            if num_threads == 1:
                # Useful for testing
                for key in tqdm(
                    self.unfixed_likelihood_cache.keys(),
                    disable=not self.progress,
                    desc="Precalculating Likelihoods",
                ):
                    returned_key, likelihoods = f(key)
                    self.unfixed_likelihood_cache[returned_key] = likelihoods
            else:
                with tqdm(
                    total=len(self.unfixed_likelihood_cache.keys()),
                    disable=not self.progress,
                    desc="Precalculating Likelihoods",
                ) as prog_bar:
                    with multiprocessing.Pool(processes=num_threads) as pool:
                        for key, pmf in pool.imap_unordered(
                            f, self.unfixed_likelihood_cache.keys()
                        ):
                            self.unfixed_likelihood_cache[key] = pmf
                            prog_bar.update()
        else:
            for muts, span in tqdm(
                self.unfixed_likelihood_cache.keys(),
                disable=not self.progress,
                desc="Precalculating Likelihoods",
            ):
                self.unfixed_likelihood_cache[muts, span] = self._lik(
                    muts,
                    span,
                    dt=self.timediff_lower_tri,
                    mutation_rate=self.mut_rate,
                    standardize=self.standardize,
                )

    def get_mut_lik_fixed_node(self, edge):
        """
        Get the mutation likelihoods for an edge whose child is at a
        fixed time, but whose parent may take any of the time slices in the timepoints
        that are equal to or older than the child age. This is not cached, as it is
        likely to be unique for each edge
        """
        assert (
            edge.child in self.fixednodes
        ), "Wrongly called fixed node function on non-fixed node"
        assert (
            self.mut_rate is not None
        ), "Cannot calculate mutation likelihoods with no mutation_rate set"

        mutations_on_edge = self.mut_edges[edge.id]
        child_time = self.ts.node(edge.child).time
        assert child_time == 0
        # Temporary hack - we should really take a more precise likelihood
        return self._lik(
            mutations_on_edge,
            edge.span,
            self.timediff,
            self.mut_rate,
            standardize=self.standardize,
        )

    def get_mut_lik_lower_tri(self, edge):
        """
        Get the cached mutation likelihoods for an edge with non-fixed parent and child
        nodes, returning values for all the possible time differences between timepoints
        These values are returned as a flattened lower triangular matrix, the
        form required in the inside algorithm.

        """
        # Debugging asserts - should probably remove eventually
        assert (
            edge.child not in self.fixednodes
        ), "Wrongly called lower_tri function on fixed node"
        assert hasattr(
            self, "unfixed_likelihood_cache"
        ), "Must call `precalculate_mutation_likelihoods()` before getting likelihoods"

        mutations_on_edge = self.mut_edges[edge.id]
        return self.unfixed_likelihood_cache[mutations_on_edge, edge.span]

    def get_mut_lik_upper_tri(self, edge):
        """
        Same as :meth:`get_mut_lik_lower_tri`, but the returned array is ordered as
        flattened upper triangular matrix (suitable for the outside algorithm), rather
        than a lower triangular one
        """
        return self.get_mut_lik_lower_tri(edge)[np.concatenate(self.row_indices)]

    # The following functions don't access the likelihoods directly, but allow
    # other input arrays of length grid_size to be repeated in such a way that they can
    # be directly multiplied by the unpacked lower triangular matrix, or arrays of length
    # of the number of cells in the lower triangular matrix to be summed (e.g. by row)
    # to give a shorter array of length grid_size

    def make_lower_tri(self, input_array):
        """
        Repeat the input array row-wise to make a flattened lower triangular matrix
        """
        assert len(input_array) == self.grid_size
        return input_array[self.to_lower_tri]

    def rowsum_lower_tri(self, input_array):
        """
        Describe the reduceat trickery here. Presumably the opposite of make_lower_tri
        """
        assert len(input_array) == self.tri_size
        return np.add.reduceat(input_array, self.row_indices[0])

    def make_upper_tri(self, input_array):
        """
        Repeat the input array row-wise to make a flattened upper triangular matrix
        """
        assert len(input_array) == self.grid_size
        return input_array[self.to_upper_tri]

    def rowsum_upper_tri(self, input_array):
        """
        Describe the reduceat trickery here. Presumably the opposite of make_upper_tri
        """
        assert len(input_array) == self.tri_size
        return np.add.reduceat(input_array, self.col_indices)

    # Mutation & recombination algorithms on a tree sequence

    def n_breaks(self, edge):
        """
        Number of known breakpoints, only used in recombination likelihood calc
        """
        return (edge.left != 0) + (edge.right != self.ts.get_sequence_length())

    def combine(self, lik_1, lik_2):
        return lik_1 * lik_2

    def ratio(self, lik_1, lik_2, div_0_null=False):
        """
        Return the ratio of lik_1 to lik_2. In linear space, this divides lik_1 by lik_2
        If div_0_null==True, then 0/0 is set to the null_constant
        """
        with np.errstate(divide="ignore", invalid="ignore"):
            ret = lik_1 / lik_2
        if div_0_null:
            ret[np.isnan(ret)] = self.null_constant
        return ret

    def _recombination_lik(self, edge, fixed=True):
        # Needs to return a lower tri *or* flattened array depending on `fixed`
        raise NotImplementedError(
            "Using the recombination clock is not currently supported"
            ". See https://github.com/awohns/tsdate/issues/5 for details"
        )
        # return (
        #     np.power(prev_state, self.n_breaks(edge)) *
        #     np.exp(-(prev_state * self.rec_rate * edge.span * 2)))

    def get_inside(self, arr, edge):
        liks = self.identity_constant
        if self.rec_rate is not None:
            liks = self._recombination_lik(edge)
        if self.mut_rate is not None:
            liks *= self.get_mut_lik_lower_tri(edge)
        return self.rowsum_lower_tri(arr * liks)

    def get_outside(self, arr, edge):
        liks = self.identity_constant
        if self.rec_rate is not None:
            liks = self._recombination_lik(edge)
        if self.mut_rate is not None:
            liks *= self.get_mut_lik_upper_tri(edge)
        return self.rowsum_upper_tri(arr * liks)

    def get_fixed(self, arr, edge):
        liks = self.identity_constant
        if self.rec_rate is not None:
            liks = self._recombination_lik(edge, fixed=True)
        if self.mut_rate is not None:
            liks *= self.get_mut_lik_fixed_node(edge)
        return arr * liks

    def scale_geometric(self, fraction, value):
        return value**fraction


class LogLikelihoods(Likelihoods):
    """
    Identical to the Likelihoods class but stores and returns log likelihoods
    """

    probability_space = base.LOG
    identity_constant = 0.0
    null_constant = -np.inf

    """
    Uses an alternative to logsumexp, useful for large grid sizes, see
    http://www.nowozin.net/sebastian/blog/streaming-log-sum-exp-computation.html
    """

    @staticmethod
    @numba.jit(nopython=True)
    def logsumexp(X):
        alpha = -np.Inf
        r = 0.0
        for x in X:
            if x != -np.Inf:
                if x <= alpha:
                    r += np.exp(x - alpha)
                else:
                    r *= np.exp(alpha - x)
                    r += 1.0
                    alpha = x
        return np.log(r) + alpha

    @staticmethod
    def _lik(muts, span, dt, mutation_rate, standardize=True):
        """
        The likelihood of an edge given a number of mutations, as set of time deltas (dt)
        and a span. This is a static function to allow parallelization
        """
        ll = scipy.stats.poisson.logpmf(muts, dt * mutation_rate * span)
        if standardize:
            return ll - np.max(ll)
        else:
            return ll

    @staticmethod
    def _lik_wrapper(muts_span, dt, mutation_rate, standardize=True):
        """
        Needs redefining to refer to the LogLikelihoods class
        """
        return muts_span, LogLikelihoods._lik(
            muts_span[0], muts_span[1], dt, mutation_rate, standardize=standardize
        )

    def rowsum_lower_tri(self, input_array):
        """
        The function below is equivalent to (but numba makes it faster than)
        np.logaddexp.reduceat(input_array, self.row_indices[0])
        """
        assert len(input_array) == self.tri_size
        res = list()
        i_start = self.row_indices[0][0]
        for i in self.row_indices[0][1:]:
            res.append(self.logsumexp(input_array[i_start:i]))
            i_start = i
        res.append(self.logsumexp(input_array[i:]))
        return np.array(res)

    def rowsum_upper_tri(self, input_array):
        """
        The function below is equivalent to (but numba makes it faster than)
        np.logaddexp.reduceat(input_array, self.col_indices)
        """
        assert len(input_array) == self.tri_size
        res = list()
        i_start = self.col_indices[0]
        for i in self.col_indices[1:]:
            res.append(self.logsumexp(input_array[i_start:i]))
            i_start = i
        res.append(self.logsumexp(input_array[i:]))
        return np.array(res)

    def _recombination_loglik(self, edge, fixed=True):
        # Needs to return a lower tri *or* flattened array depending on `fixed`
        raise NotImplementedError(
            "Using the recombination clock is not currently supported"
            ". See https://github.com/awohns/tsdate/issues/5 for details"
        )
        # return (
        #     np.power(prev_state, self.n_breaks(edge)) *
        #     np.exp(-(prev_state * self.rec_rate * edge.span * 2)))

    def combine(self, loglik_1, loglik_2):
        return loglik_1 + loglik_2

    def ratio(self, loglik_1, loglik_2, div_0_null=False):
        """
        In log space, likelihood ratio is loglik_1 - loglik_2
        If div_0_null==True, then if either is -inf it returns -inf (the null_constant)
        """
        with np.errstate(divide="ignore", invalid="ignore"):
            ret = loglik_1 - loglik_2
        if div_0_null:
            ret[np.isnan(ret)] = self.null_constant
        return ret

    def get_inside(self, arr, edge):
        log_liks = self.identity_constant
        if self.rec_rate is not None:
            log_liks = self._recombination_loglik(edge)
        if self.mut_rate is not None:
            log_liks += self.get_mut_lik_lower_tri(edge)
        return self.rowsum_lower_tri(arr + log_liks)

    def get_outside(self, arr, edge):
        log_liks = self.identity_constant
        if self.rec_rate is not None:
            log_liks = self._recombination_loglik(edge)
        if self.mut_rate is not None:
            log_liks += self.get_mut_lik_upper_tri(edge)
        return self.rowsum_upper_tri(arr + log_liks)

    def get_fixed(self, arr, edge):
        log_liks = self.identity_constant
        if self.rec_rate is not None:
            log_liks = self._recombination_loglik(edge, fixed=True)
        if self.mut_rate is not None:
            log_liks += self.get_mut_lik_fixed_node(edge)
        return arr + log_liks

    def scale_geometric(self, fraction, value):
        return fraction * value


class VariationalLikelihoods:
    """
    A class to store and process likelihoods for use in variational inference.
    """

    probability_space = base.GAMMA_PAR
    identity_constant = np.array([1.0, 0.0], dtype=float)  # "improper" gamma prior
    null_constant = 0.0
    timepoints = np.array([0, np.inf], dtype=float)
    grid_size = 2

    def __init__(
        self,
        ts,
        mutation_rate=None,
        recombination_rate=None,
        *,
        fixed_node_set=None,
    ):
        self.ts = ts
        self.fixednodes = (
            set(ts.samples()) if fixed_node_set is None else fixed_node_set
        )
        self.mut_rate = mutation_rate
        self.rec_rate = recombination_rate
        self.mut_edges = self.get_mut_edges(ts)
        self.identity_constant.flags.writeable = False
        self.timepoints.flags.writeable = False

    def to_gamma(self, edge):
        """
        Return the shape and rate parameters of the (gamma) posterior of edge
        length, given an improper (constant) prior.
        """
        y = self.mut_edges[edge.id]
        mu = edge.span * self.mut_rate
        return np.array([y + 1, mu])

    @staticmethod
    def get_mut_edges(ts):
        """
        Get the number of mutations on each edge in the tree sequence.
        """
        return Likelihoods.get_mut_edges(ts)

    @staticmethod
    def combine(base, message):
        """
        Multiply two gamma PDFs
        """
        return base + message + [-1, 0]

    @staticmethod
    def ratio(base, message):
        """
        Divide two gamma PDFs
        """
        return base - message + [1, 0]

    @staticmethod
    def scale_geometric(fraction, pars):
        """
        Scale the parameters of a gamma distribution by raising the PDF to a
        fractional power.
        """
        assert 1 >= fraction >= 0
        new_pars = pars.copy()
        new_pars[0] = fraction * (new_pars[0] - 1) + 1
        new_pars[1] = fraction * new_pars[1]
        return new_pars


class InOutAlgorithms:
    """
    Contains the inside and outside algorithms
    """

    def __init__(self, priors, lik, *, progress=False):
        if (
            lik.fixednodes.intersection(priors.nonfixed_nodes)
            or len(lik.fixednodes) + len(priors.nonfixed_nodes) != lik.ts.num_nodes
        ):
            raise ValueError(
                "The prior and likelihood objects disagree on which nodes are fixed"
            )
        if not np.allclose(lik.timepoints, priors.timepoints):
            raise ValueError(
                "The prior and likelihood objects disagree on the timepoints used"
            )

        self.priors = priors
        self.nonfixed_nodes = priors.nonfixed_nodes
        self.lik = lik
        self.ts = lik.ts

        self.fixednodes = lik.fixednodes
        self.progress = progress
        # If necessary, convert priors to log space
        self.priors.force_probability_space(lik.probability_space)

        self.spans = np.bincount(
            self.ts.tables.edges.child,
            weights=self.ts.tables.edges.right - self.ts.tables.edges.left,
        )
        self.spans = np.pad(self.spans, (0, self.ts.num_nodes - len(self.spans)))

        self.root_spans = defaultdict(float)
        for tree in self.ts.trees():
            root = util.get_single_root(tree)
            if root is not None:
                self.root_spans[
                    root
                ] += tree.span  # Count span if we have a single tree
        # Add on the spans when this is a root
        for root, span_when_root in self.root_spans.items():
            self.spans[root] += span_when_root

    # === Grouped edge iterators ===

    def edges_by_parent_asc(self):
        """
        Return an itertools.groupby object of edges grouped by parent in ascending order
        of the time of the parent. Since tree sequence properties guarantee that edges
        are listed in nondecreasing order of parent time
        (https://tskit.readthedocs.io/en/latest/data-model.html#edge-requirements)
        we can simply use the standard edge order
        """
        return itertools.groupby(self.ts.edges(), operator.attrgetter("parent"))

    def edges_by_child_desc(self):
        """
        Return an itertools.groupby object of edges grouped by child in descending order
        of the time of the child.
        """
        it = (
            self.ts.edge(u)
            for u in np.lexsort(
                (self.ts.edges_child, -self.ts.nodes_time[self.ts.edges_child])
            )
        )
        return itertools.groupby(it, operator.attrgetter("child"))

    def edges_by_child_then_parent_desc(self):
        """
        Return an itertools.groupby object of edges grouped by child in descending order
        of the time of the child, then by descending order of age of child
        """
        wtype = np.dtype(
            [
                ("child_age", self.ts.tables.nodes.time.dtype),
                ("child_node", self.ts.tables.edges.child.dtype),
                ("parent_age", self.ts.tables.nodes.time.dtype),
            ]
        )
        w = np.empty(self.ts.num_edges, dtype=wtype)
        w["child_age"] = self.ts.tables.nodes.time[self.ts.tables.edges.child]
        w["child_node"] = self.ts.tables.edges.child
        w["parent_age"] = -self.ts.tables.nodes.time[self.ts.tables.edges.parent]
        sorted_child_parent = (
            self.ts.edge(i)
            for i in reversed(
                np.argsort(w, order=("child_age", "child_node", "parent_age"))
            )
        )
        return itertools.groupby(sorted_child_parent, operator.attrgetter("child"))

    # === MAIN ALGORITHMS ===

    def inside_pass(self, *, standardize=True, cache_inside=False, progress=None):
        """
        Use dynamic programming to find approximate posterior to sample from
        """
        if progress is None:
            progress = self.progress
        inside = self.priors.clone_with_new_data(  # store inside matrix values
            grid_data=np.nan, fixed_data=self.lik.identity_constant
        )
        if cache_inside:
            g_i = np.full(
                (self.ts.num_edges, self.lik.grid_size), self.lik.identity_constant
            )
        denominator = np.full(self.ts.num_nodes, np.nan)
        # Iterate through the nodes via groupby on parent node
        for parent, edges in tqdm(
            self.edges_by_parent_asc(),
            desc="Inside",
            total=inside.num_nonfixed,
            disable=not progress,
        ):
            """
            for each node, find the conditional prob of age at every time
            in time grid
            """
            if parent in self.fixednodes:
                continue  # there is no hidden state for this parent - it's fixed
            val = self.priors[parent].copy()
            for edge in edges:
                spanfrac = edge.span / self.spans[edge.child]
                # Calculate vals for each edge
                if edge.child in self.fixednodes:
                    # NB: geometric scaling works exactly when all nodes fixed in graph
                    # but is an approximation when times are unknown.
                    daughter_val = self.lik.scale_geometric(
                        spanfrac, inside[edge.child]
                    )
                    edge_lik = self.lik.get_fixed(daughter_val, edge)
                else:
                    inside_values = inside[edge.child]
                    if np.ndim(inside_values) == 0 or np.all(np.isnan(inside_values)):
                        # Child appears fixed, or we have not visited it. Either our
                        # edge order is wrong (bug) or we have hit a dangling node
                        raise ValueError(
                            "The input tree sequence includes "
                            "dangling nodes: please simplify it"
                        )
                    daughter_val = self.lik.scale_geometric(
                        spanfrac, self.lik.make_lower_tri(inside[edge.child])
                    )
                    edge_lik = self.lik.get_inside(daughter_val, edge)
                val = self.lik.combine(val, edge_lik)
                if cache_inside:
                    g_i[edge.id] = edge_lik
            denominator[parent] = (
                np.max(val) if standardize else self.lik.identity_constant
            )
            inside[parent] = self.lik.ratio(val, denominator[parent])
        if cache_inside:
            self.g_i = self.lik.ratio(
                g_i, denominator[self.ts.tables.edges.child, None]
            )
        # Keep the results in this object
        self.inside = inside
        self.denominator = denominator

    def outside_pass(
        self,
        *,
        standardize=False,
        ignore_oldest_root=False,
        progress=None,
    ):
        """
        Computes the full posterior distribution on nodes, returning the
        posterior values. These are *not* probabilities, as they do not sum to one:
        to convert to probabilities, call posterior.to_probabilities()

        Standardizing *during* the outside process may be necessary if there is
        overflow, but means that we cannot  check the total functional value at each node

        Ignoring the oldest root may also be necessary when the oldest root node
        causes numerical stability issues.

        The rows in the posterior returned correspond to node IDs as given by
        self.nodes
        """
        if progress is None:
            progress = self.progress
        if not hasattr(self, "inside"):
            raise RuntimeError("You have not yet run the inside algorithm")

        outside = self.inside.clone_with_new_data(
            grid_data=0, probability_space=base.LIN
        )
        for root, span_when_root in self.root_spans.items():
            outside[root] = span_when_root / self.spans[root]
        outside.force_probability_space(self.inside.probability_space)

        for child, edges in tqdm(
            self.edges_by_child_desc(),
            desc="Outside",
            total=len(np.unique(self.ts.tables.edges.child)),
            disable=not progress,
        ):
            if child in self.fixednodes:
                continue
            val = np.full(self.lik.grid_size, self.lik.identity_constant)
            for edge in edges:
                if ignore_oldest_root:
                    if edge.parent == self.ts.num_nodes - 1:
                        continue
                if edge.parent in self.fixednodes:
                    raise RuntimeError(
                        "Fixed nodes cannot currently be parents in the TS"
                    )
                # Geometric scaling works exactly for all nodes fixed in graph
                # but is an approximation when times are unknown.
                spanfrac = edge.span / self.spans[child]
                try:
                    inside_div_gi = self.lik.ratio(
                        self.inside[edge.parent], self.g_i[edge.id], div_0_null=True
                    )
                except AttributeError:  # we haven't cached g_i so we recalculate
                    daughter_val = self.lik.scale_geometric(
                        spanfrac, self.lik.make_lower_tri(self.inside[edge.child])
                    )
                    edge_lik = self.lik.get_inside(daughter_val, edge)
                    cur_g_i = self.lik.ratio(edge_lik, self.denominator[child])
                    inside_div_gi = self.lik.ratio(
                        self.inside[edge.parent], cur_g_i, div_0_null=True
                    )
                parent_val = self.lik.scale_geometric(
                    spanfrac,
                    self.lik.make_upper_tri(
                        self.lik.combine(outside[edge.parent], inside_div_gi)
                    ),
                )
                if standardize:
                    parent_val = self.lik.ratio(parent_val, np.max(parent_val))
                edge_lik = self.lik.get_outside(parent_val, edge)
                val = self.lik.combine(val, edge_lik)

            # vv[0] = 0  # Seems a hack: internal nodes should be allowed at time 0
            assert self.denominator[edge.child] > self.lik.null_constant
            outside[child] = self.lik.ratio(val, self.denominator[child])
            if standardize:
                outside[child] = self.lik.ratio(val, np.max(val))
        self.outside = outside
        posterior = outside.clone_with_new_data(
            grid_data=self.lik.combine(self.inside.grid_data, outside.grid_data),
            fixed_data=np.nan,
        )  # We should never use the posterior for a fixed node
        return posterior

    def outside_maximization(self, *, eps, progress=None):
        if progress is None:
            progress = self.progress
        if not hasattr(self, "inside"):
            raise RuntimeError("You have not yet run the inside algorithm")

        maximized_node_times = np.zeros(self.ts.num_nodes, dtype="int")

        if self.lik.probability_space == base.LOG:
            poisson = scipy.stats.poisson.logpmf
        elif self.lik.probability_space == base.LIN:
            poisson = scipy.stats.poisson.pmf

        mut_edges = self.lik.mut_edges
        mrcas = np.where(
            np.isin(
                np.arange(self.ts.num_nodes), self.ts.tables.edges.child, invert=True
            )
        )[0]
        for i in mrcas:
            if i not in self.fixednodes:
                maximized_node_times[i] = np.argmax(self.inside[i])

        for child, edges in tqdm(
            self.edges_by_child_then_parent_desc(),
            desc="Maximization",
            total=len(np.unique(self.ts.tables.edges.child)),
            disable=not progress,
        ):
            if child in self.fixednodes:
                continue
            for edge_index, edge in enumerate(edges):
                if edge_index == 0:
                    youngest_par_index = maximized_node_times[edge.parent]
                    parent_time = self.lik.timepoints[maximized_node_times[edge.parent]]
                    ll_mut = poisson(
                        mut_edges[edge.id],
                        (
                            parent_time
                            - self.lik.timepoints[: youngest_par_index + 1]
                            + eps
                        )
                        * self.lik.mut_rate
                        * edge.span,
                    )
                    result = self.lik.ratio(ll_mut, np.max(ll_mut))
                else:
                    cur_parent_index = maximized_node_times[edge.parent]
                    if cur_parent_index < youngest_par_index:
                        youngest_par_index = cur_parent_index
                    parent_time = self.lik.timepoints[maximized_node_times[edge.parent]]
                    ll_mut = poisson(
                        mut_edges[edge.id],
                        (
                            parent_time
                            - self.lik.timepoints[: youngest_par_index + 1]
                            + eps
                        )
                        * self.lik.mut_rate
                        * edge.span,
                    )
                    result[: youngest_par_index + 1] = self.lik.combine(
                        self.lik.ratio(
                            ll_mut[: youngest_par_index + 1],
                            np.max(ll_mut[: youngest_par_index + 1]),
                        ),
                        result[: youngest_par_index + 1],
                    )
            inside_val = self.inside[child][: (youngest_par_index + 1)]

            maximized_node_times[child] = np.argmax(
                self.lik.combine(result[: youngest_par_index + 1], inside_val)
            )

        return self.lik.timepoints[np.array(maximized_node_times).astype("int")]


class ExpectationPropagation(InOutAlgorithms):
    """
    Expectation propagation, where the edge factors are approximated
    by the product of two gamma distributions
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        assert self.priors.probability_space == base.GAMMA_PAR
        assert self.lik.probability_space == base.GAMMA_PAR
        assert self.lik.grid_size == 2
        assert self.priors.timepoints.size == 2

        # Messages passed from factors in the direction of roots
        self.parent_message = np.tile(
            self.lik.identity_constant,
            (self.ts.num_edges, 1),
        )

        # Messages passed from factors in the direction of leaves
        self.child_message = np.tile(
            self.lik.identity_constant,
            (self.ts.num_edges, 1),
        )

        # Normalizing constants from each factor
        self.factor_norm = np.full(self.ts.num_edges, 0.0)

        # the approximate posterior marginals
        self.posterior = self.priors.clone_with_new_data(
            grid_data=self.priors.grid_data.copy(),
            fixed_data=np.nan,
        )

        # factors for edges leading from fixed nodes are invariant
        # and can be incorporated into the posterior beforehand
        for edge in self.ts.edges():
            if edge.child in self.fixednodes:
                self.parent_message[edge.id] = self.lik.to_gamma(edge)
                self.posterior[edge.parent] = self.lik.combine(
                    self.posterior[edge.parent], self.parent_message[edge.id]
                )
                # self.factor_norm[edge.id] += ... # TODO

    def edges_by_parent_asc(self):
        """
        Edges in order of increasing age of parent
        """
        return self.ts.edges()

    def edges_by_child_desc(self):
        """
        Edges in order of decreasing age of child
        """
        wtype = np.dtype(
            [
                ("child_age", self.ts.tables.nodes.time.dtype),
                ("child_node", self.ts.tables.edges.child.dtype),
            ]
        )
        w = np.empty(self.ts.num_edges, dtype=wtype)
        w["child_age"] = self.ts.tables.nodes.time[self.ts.tables.edges.child]
        w["child_node"] = self.ts.tables.edges.child
        sorted_child_parent = (
            self.ts.edge(i)
            for i in reversed(np.argsort(w, order=("child_age", "child_node")))
        )
        return sorted_child_parent

    def propagate(self, *, edges, progress=None):
        """
        Update approximating factor for each edge
        """
        if progress is None:
            progress = self.progress
        # TODO: this will still converge if parallelized (potentially slower)
        for edge in tqdm(
            edges,
            desc="Expectation Propagation",
            total=self.ts.num_edges,
            disable=not progress,
        ):
            if edge.child in self.fixednodes:
                continue
            if edge.parent in self.fixednodes:
                raise ValueError("Internal nodes can not be fixed in EP algorithm")
            edge_lik = self.lik.to_gamma(edge)
            edge_lik += np.array([-1.0, 0.0])  # to Poisson, TODO cleanup
            # cavity posteriors without approximate edge factor
            parent_cavity = self.lik.ratio(
                self.posterior[edge.parent], self.parent_message[edge.id]
            )
            child_cavity = self.lik.ratio(
                self.posterior[edge.child], self.child_message[edge.id]
            )
            # target posterior matches cavity with exact edge factor
            (
                norm_const,
                self.posterior[edge.parent],
                self.posterior[edge.child],
            ) = approx.gamma_projection(*parent_cavity, *child_cavity, *edge_lik)
            # store approximate edge factors including normalizer
            self.parent_message[edge.id] = self.lik.ratio(
                self.posterior[edge.parent], parent_cavity
            )
            self.child_message[edge.id] = self.lik.ratio(
                self.posterior[edge.child], child_cavity
            )
            self.factor_norm[edge.id] = norm_const

    def iterate(self, *, progress=None, **kwargs):
        """
        Update edge factors from leaves to root then from root to leaves,
        and return approximate log marginal likelihood
        """
        self.propagate(edges=self.edges_by_parent_asc(), progress=progress)
        self.propagate(edges=self.edges_by_child_desc(), progress=progress)
        # TODO
        # marginal_lik = np.sum(self.factor_norm)
        # return marginal_lik


def posterior_mean_var(ts, posterior, *, fixed_node_set=None):
    """
    Mean and variance of node age. Fixed nodes will be given a mean
    of their exact time in the tree sequence, and zero variance (as long as they are
    identified by the fixed_node_set).
    If fixed_node_set is None, we attempt to date all the non-sample nodes
    Also assigns the estimated mean and variance of the age of each node
    as metadata in the tree sequence.
    """
    mn_post = np.full(ts.num_nodes, np.nan)  # Fill with NaNs so we detect when there's
    vr_post = np.full(ts.num_nodes, np.nan)  # been an error
    tables = ts.dump_tables()

    if fixed_node_set is None:
        fixed_node_set = ts.samples()
    fixed_nodes = np.array(list(fixed_node_set))
    mn_post[fixed_nodes] = tables.nodes.time[fixed_nodes]
    vr_post[fixed_nodes] = 0

    metadata_array = tskit.unpack_bytes(
        ts.tables.nodes.metadata, ts.tables.nodes.metadata_offset
    )
    for u in posterior.nonfixed_nodes:
        probs = posterior[u]
        times = posterior.timepoints
        mn_post[u] = np.sum(probs * times) / np.sum(probs)
        vr_post[u] = np.sum(((mn_post[u] - (times)) ** 2) * (probs / np.sum(probs)))
        metadata_array[u] = json.dumps({"mn": mn_post[u], "vr": vr_post[u]}).encode()
    md, md_offset = tskit.pack_bytes(metadata_array)
    tables.nodes.set_columns(
        flags=tables.nodes.flags,
        time=tables.nodes.time,
        population=tables.nodes.population,
        individual=tables.nodes.individual,
        metadata=md,
        metadata_offset=md_offset,
    )
    ts = tables.tree_sequence()
    return ts, mn_post, vr_post


def constrain_ages_topo(ts, post_mn, eps, nodes_to_date=None, progress=False):
    """
    If predicted node times violate topology, restrict node ages so that they
    must be older than all their children.
    """
    new_mn_post = np.copy(post_mn)
    if nodes_to_date is None:
        nodes_to_date = np.arange(ts.num_nodes, dtype=np.uint64)
        nodes_to_date = nodes_to_date[~np.isin(nodes_to_date, ts.samples())]

    tables = ts.tables
    parents = tables.edges.parent
    nd_children = tables.edges.child[np.argsort(parents)]
    parents = sorted(parents)
    parents_unique = np.unique(parents, return_index=True)
    parent_indices = parents_unique[1][np.isin(parents_unique[0], nodes_to_date)]
    for index, nd in tqdm(
        enumerate(sorted(nodes_to_date)), desc="Constrain Ages", disable=not progress
    ):
        if index + 1 != len(nodes_to_date):
            children_index = np.arange(parent_indices[index], parent_indices[index + 1])
        else:
            children_index = np.arange(parent_indices[index], ts.num_edges)
        children = nd_children[children_index]
        time = np.max(new_mn_post[children])
        if new_mn_post[nd] <= time:
            new_mn_post[nd] = time + eps
    return new_mn_post


def date(
    tree_sequence,
    mutation_rate,
    population_size=None,
    recombination_rate=None,
    time_units=None,
    priors=None,
    *,
    Ne=None,
    return_posteriors=None,
    progress=False,
    expectation_propagation=0,  # TODO document
    global_prior=True,  # TODO document
    **kwargs,
):
    """
    Take a tree sequence (which could have
    :data:`uncalibrated <tskit.TIME_UNITS_UNCALIBRATED>` node times) and assign new times
    to non-sample nodes using the `tsdate` algorithm. If a mutation_rate is given,
    the mutation clock is used. The recombination clock is unsupported at this time.
    If neither a mutation_rate nor a
    recombination_rate is given, a topology-only clock is used. Times associated with
    mutations and non-sample nodes in the input tree sequence are not used in inference
    and will be removed.

    .. note::
        If posteriors are returned via the ``return_posteriors`` option, the output will
        be a tuple ``(ts, posteriors)``, where ``posteriors`` is a dictionary suitable
        for reading as a pandas ``DataFrame`` object, using ``pd.DataFrame(posteriors)``.
        Each node whose time was inferred corresponds to an item in this dictionary,
        with the key being the node ID and the value a 1D array of probabilities of the
        node being in a given time slice (or ``None`` if the "inside_outside" method
        was not used). The start and end times of each time slice are given as 1D
        arrays in the dictionary, under keys named ``"start_time"`` and ``end_time"``.
        As timeslices may not be not of uniform width, it is important to divide the
        posterior probabilities by ``end_time - start_time`` when assessing the shape
        of the probability density function over time.

    :param TreeSequence tree_sequence: The input :class:`tskit.TreeSequence`, treated as
        one whose non-sample nodes are undated.
    :param PopulationSizeHistory population_size: The estimated (diploid) effective
        population size used to construct the (default) conditional coalescent
        prior. This may be a single value (for a population with constant size), or
        a :class:`PopulationSizeHistory` object (for a population with time-varying
        size). This is used when ``priors`` is ``None``.  Conversely, if ``priors``
        is not ``None``, no ``population_size`` value should be given.
    :param float mutation_rate: The estimated mutation rate per unit of genome per
        unit time. If provided, the dating algorithm will use a mutation rate clock to
        help estimate node dates. Default: ``None``
    :param float recombination_rate: The estimated recombination rate per unit of genome
        per unit time. If provided, the dating algorithm will use a recombination rate
        clock to help estimate node dates. Default: ``None``
    :param str time_units: The time units used by the ``mutation_rate`` and
        ``recombination_rate`` values, and stored in the ``time_units`` attribute of the
        output tree sequence. If the conditional coalescent prior is used,
        then this is also applies to the value of ``population_size``, which in
        standard coalescent theory is measured in generations. Therefore if you
        wish to use mutation and recombination rates measured in (say) years,
        and are using the conditional coalescent prior, the ``population_size``
        value which you provide must be scaled by multiplying by the number of
        years per generation. If ``None`` (default), assume ``"generations"``.
    :param NodeGridValues priors: NodeGridValue object containing the prior probabilities
        for each node at a set of discrete time points. If ``None`` (default), use the
        conditional coalescent prior with a standard set of time points as given by
        :func:`build_prior_grid`.
    :param bool return_posteriors: If ``True``, instead of returning just a dated tree
        sequence, return a tuple of ``(dated_ts, posteriors)`` (see note above).
    :param float eps: Specify minimum distance separating time points. Also specifies
        the error factor in time difference calculations. Default: 1e-6
    :param int num_threads: The number of threads to use. A simpler unthreaded algorithm
        is used unless this is >= 1. Default: None
    :param string method: What estimation method to use: can be
        "inside_outside" (empirically better, theoretically problematic) or
        "maximization" (worse empirically, especially with gamma approximated priors,
        but theoretically robust). If ``None`` (default) use "inside_outside"
    :param string probability_space: Should the internal algorithm save probabilities in
        "logarithmic" (slower, less liable to to overflow) or "linear" space (fast, may
        overflow). Default: "logarithmic"
    :param bool ignore_oldest_root: Should the oldest root in the tree sequence be
        ignored in the outside algorithm (if "inside_outside" is used as the method).
        Ignoring outside root provides greater stability when dating tree sequences
        inferred from real data. Default: False
    :param bool progress: Whether to display a progress bar. Default: False
    :param float Ne: the estimated (diploid) effective population size used to
        construct the (default) conditional coalescent prior. This is used when
        ``priors`` is ``None``.  Conversely, if ``priors`` is not ``None``, no
        ``population_size`` value should be given.  (Deprecated, use the
        ``population_size`` argument instead).
    :return: A copy of the input tree sequence but with altered node times, or (if
        ``return_posteriors`` is True) a tuple of that tree sequence plus a dictionary
        of posterior probabilities from the "inside_outside" estimation ``method``.
    :rtype: tskit.TreeSequence or (tskit.TreeSequence, dict)
    """
    if time_units is None:
        time_units = "generations"
    if Ne is not None:
        if population_size is not None:
            raise ValueError(
                "Only one of Ne (deprecated) or population_size may be specified"
            )
        else:
            population_size = Ne
    if expectation_propagation > 0:
        tree_sequence, dates, posteriors, timepoints, eps, nds = variational_dates(
            tree_sequence,
            population_size=population_size,
            mutation_rate=mutation_rate,
            recombination_rate=recombination_rate,
            priors=priors,
            progress=progress,
            expectation_propagation=expectation_propagation,
            global_prior=global_prior,
        )
    else:
        tree_sequence, dates, posteriors, timepoints, eps, nds = get_dates(
            tree_sequence,
            population_size=population_size,
            mutation_rate=mutation_rate,
            recombination_rate=recombination_rate,
            priors=priors,
            progress=progress,
            **kwargs,
        )
    constrained = constrain_ages_topo(tree_sequence, dates, eps, nds, progress)
    tables = tree_sequence.dump_tables()
    tables.time_units = time_units
    tables.nodes.time = constrained
    # Remove any times associated with mutations
    tables.mutations.time = np.full(tree_sequence.num_mutations, tskit.UNKNOWN_TIME)
    tables.sort()
    # TODO: record population_size provenance, or record that it is omitted
    provenance.record_provenance(
        tables,
        "date",
        mutation_rate=mutation_rate,
        recombination_rate=recombination_rate,
        progress=progress,
        **kwargs,
    )
    if return_posteriors:
        pst = {"start_time": timepoints, "end_time": np.append(timepoints[1:], np.inf)}
        for n in nds:
            pst[n] = None if posteriors is None else posteriors[n]
        return tables.tree_sequence(), pst
    else:
        return tables.tree_sequence()


def get_dates(
    tree_sequence,
    mutation_rate,
    population_size=None,
    recombination_rate=None,
    priors=None,
    *,
    eps=1e-6,
    num_threads=None,
    method="inside_outside",
    outside_standardize=True,
    ignore_oldest_root=False,
    progress=False,
    cache_inside=False,
    probability_space=base.LOG,
):
    """
    Infer dates for the nodes in a tree sequence, returning an array of inferred dates
    for nodes, plus other variables such as the posteriors object
    etc. Parameters are identical to the date() method, which calls this method, then
    injects the resulting date estimates into the tree sequence

    :return: a tuple of ``(mn_post, posteriors, timepoints, eps, nodes_to_date)``.
        If the "inside_outside" method is used, ``posteriors`` will contain the
        posterior probabilities for each node in each time slice, else the returned
        variable will be ``None``.
    """
    # Stuff yet to be implemented. These can be deleted once fixed
    for sample in tree_sequence.samples():
        if tree_sequence.node(sample).time != 0:
            raise NotImplementedError("Samples must all be at time 0")
    fixed_nodes = set(tree_sequence.samples())

    # Default to not creating approximate priors unless ts has > 1000 samples
    approx_priors = False
    if tree_sequence.num_samples > 1000:
        approx_priors = True

    if priors is None:
        if population_size is None:
            raise ValueError(
                "Must specify population size if priors are not already built \
                        using tsdate.build_prior_grid()"
            )
        priors = prior.build_grid(
            tree_sequence,
            population_size=population_size,
            eps=eps,
            progress=progress,
            approximate_priors=approx_priors,
        )
    else:
        logging.info("Using user-specified priors")
        if population_size is not None:
            raise ValueError(
                "Cannot specify population size in tsdate.date() or tsdate.get_dates() \
                        if specifying priors from tsdate.build_prior_grid()"
            )
        priors = priors

    if probability_space != base.LOG:
        liklhd = Likelihoods(
            tree_sequence,
            priors.timepoints,
            mutation_rate,
            recombination_rate,
            eps=eps,
            fixed_node_set=fixed_nodes,
            progress=progress,
        )
    else:
        liklhd = LogLikelihoods(
            tree_sequence,
            priors.timepoints,
            mutation_rate,
            recombination_rate,
            eps=eps,
            fixed_node_set=fixed_nodes,
            progress=progress,
        )

    if mutation_rate is not None:
        liklhd.precalculate_mutation_likelihoods(num_threads=num_threads)

    dynamic_prog = InOutAlgorithms(priors, liklhd, progress=progress)
    dynamic_prog.inside_pass(cache_inside=False)

    posterior = None
    if method == "inside_outside":
        posterior = dynamic_prog.outside_pass(
            standardize=outside_standardize, ignore_oldest_root=ignore_oldest_root
        )
        # Turn the posterior into probabilities
        posterior.standardize()  # Just to make sure there are no floating point issues
        posterior.force_probability_space(base.LIN)
        posterior.to_probabilities()
        tree_sequence, mn_post, _ = posterior_mean_var(
            tree_sequence, posterior, fixed_node_set=fixed_nodes
        )
    elif method == "maximization":
        if mutation_rate is not None:
            mn_post = dynamic_prog.outside_maximization(eps=eps)
        else:
            raise ValueError("Outside maximization method requires mutation rate")
    else:
        raise ValueError(
            "estimation method must be either 'inside_outside' or 'maximization'"
        )

    return (
        tree_sequence,
        mn_post,
        posterior,
        priors.timepoints,
        eps,
        priors.nonfixed_nodes,
    )


def variational_mean_var(ts, posterior, *, fixed_node_set=None):
    """
    Mean and variance of node age from variational posterior (e.g. gamma
    distributions).  Fixed nodes will be given a mean of their exact time in
    the tree sequence, and zero variance (as long as they are identified by the
    fixed_node_set).  If fixed_node_set is None, we attempt to date all the
    non-sample nodes Also assigns the estimated mean and variance of the age of
    each node as metadata in the tree sequence.
    """
    mn_post = np.full(ts.num_nodes, np.nan)  # Fill with NaNs so we detect when there's
    vr_post = np.full(ts.num_nodes, np.nan)  # been an error
    tables = ts.dump_tables()

    if fixed_node_set is None:
        fixed_node_set = ts.samples()
    fixed_nodes = np.array(list(fixed_node_set))
    mn_post[fixed_nodes] = tables.nodes.time[fixed_nodes]
    vr_post[fixed_nodes] = 0

    assert np.all(posterior.grid_data[:, 0] > 0), "Invalid posterior"

    metadata_array = tskit.unpack_bytes(
        ts.tables.nodes.metadata, ts.tables.nodes.metadata_offset
    )
    for u in posterior.nonfixed_nodes:
        # TODO: with method posterior.mean_and_var(node_id) this could be
        # easily combined with posterior_mean_var
        pars = posterior[u]
        mn_post[u] = pars[0] / pars[1]
        vr_post[u] = pars[0] / pars[1] ** 2
        metadata_array[u] = json.dumps({"mn": mn_post[u], "vr": vr_post[u]}).encode()
    md, md_offset = tskit.pack_bytes(metadata_array)
    tables.nodes.set_columns(
        flags=tables.nodes.flags,
        time=tables.nodes.time,
        population=tables.nodes.population,
        individual=tables.nodes.individual,
        metadata=md,
        metadata_offset=md_offset,
    )
    ts = tables.tree_sequence()
    return ts, mn_post, vr_post


def variational_dates(
    tree_sequence,
    mutation_rate,
    population_size=None,
    recombination_rate=None,
    priors=None,
    *,
    expectation_propagation=1,
    global_prior=True,
    eps=1e-6,
    ignore_oldest_root=False,
    progress=False,
    cache_inside=False,
):
    """
    TODO update docstring

    Infer dates for the nodes in a tree sequence, returning an array of inferred dates
    for nodes, plus other variables such as the posteriors object
    etc. Parameters are identical to the date() method, which calls this method, then
    injects the resulting date estimates into the tree sequence

    :return: a tuple of ``(mn_post, posteriors, timepoints, eps, nodes_to_date)``.
        If the "inside_outside" method is used, ``posteriors`` will contain the
        posterior probabilities for each node in each time slice, else the returned
        variable will be ``None``.
    """
    # TODO: non-contemporary samples must have priors specified: if so, they'll
    # work fine with this algorithm.
    for sample in tree_sequence.samples():
        if tree_sequence.node(sample).time != 0:
            raise NotImplementedError("Samples must all be at time 0")
    fixed_nodes = set(tree_sequence.samples())

    assert expectation_propagation > 0

    # Default to not creating approximate priors unless ts has > 1000 samples
    approx_priors = False
    if tree_sequence.num_samples > 1000:
        approx_priors = True

    if priors is None:
        if population_size is None:
            raise ValueError(
                "Must specify population size if priors are not already "
                "built using tsdate.build_parameter_grid()"
            )
        priors = prior.parameter_grid(
            tree_sequence,
            population_size=population_size,
            progress=progress,
            approximate_priors=approx_priors,
        )
    else:
        logging.info("Using user-specified priors")
        if population_size is not None:
            raise ValueError(
                "Cannot specify population size in tsdate.date() or "
                "tsdate.variational_dates() if specifying priors from "
                "tsdate.build_parameter_grid()"
            )
        priors = priors
    if global_prior:
        logging.info("Pooling node-specific priors into global prior")
        priors.grid_data[:] = approx.average_gammas(
            priors.grid_data[:, 0], priors.grid_data[:, 1]
        )

    liklhd = VariationalLikelihoods(
        tree_sequence,
        mutation_rate,
        recombination_rate,
        fixed_node_set=fixed_nodes,
    )

    dynamic_prog = ExpectationPropagation(priors, liklhd, progress=progress)
    for _ in range(expectation_propagation):
        dynamic_prog.iterate()
    posterior = dynamic_prog.posterior
    tree_sequence, mn_post, _ = variational_mean_var(
        tree_sequence, posterior, fixed_node_set=fixed_nodes
    )

    return (
        tree_sequence,
        mn_post,
        posterior,
        np.array([]),
        eps,
        priors.nonfixed_nodes,
    )
