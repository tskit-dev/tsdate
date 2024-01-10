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
from collections import namedtuple

import numba
import numpy as np
import scipy.stats
import tskit
from tqdm.auto import tqdm

from . import approx
from . import base
from . import demography
from . import mixture
from . import prior
from . import provenance

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
        standardize=False,
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
            if m.edge != tskit.NULL:
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
            fixed_nodes = np.array(list(self.fixednodes))
            keys = np.unique(
                np.core.records.fromarrays(
                    (self.mut_edges, self.ts.edges_right - self.ts.edges_left),
                    names="muts,span",
                )[np.logical_not(np.isin(self.ts.edges_child, fixed_nodes))]
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

    def marginalize(self, lik):
        """
        Return the sum of likelihoods
        """
        return np.sum(lik)

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
        return -np.Inf if r == 0 else np.log(r) + alpha

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

    def marginalize(self, loglik):
        """
        Return the logged sum of likelihoods
        """
        return self.logsumexp(loglik)

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
    Has two main purposes: to store mutation counts / rates for Poisson
    likelihoods per edge, and to perform binary operations gamma distributions
    in terms of their natural parameterization (e.g. adding / subtracting sufficient
    statistics is equivalent to multiplying / dividing gamma PDFs).
    """

    probability_space = base.GAMMA_PAR
    identity_constant = np.array([0.0, 0.0], dtype=float)
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

    def to_natural(self, edge):
        """
        Return the natural parameters of the (gamma) posterior of edge
        length, given a Poisson likelihood for mutation counts
        """
        y = self.mut_edges[edge.id]
        mu = edge.span * self.mut_rate
        return np.array([y, mu])

    @staticmethod
    def get_mut_edges(ts):
        """
        Get the number of mutations on each edge in the tree sequence.
        """
        return Likelihoods.get_mut_edges(ts)


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
            self.ts.edges_child,
            weights=self.ts.edges_right - self.ts.edges_left,
        )
        self.spans = np.pad(self.spans, (0, self.ts.num_nodes - len(self.spans)))

        self.root_spans = defaultdict(float)
        for tree in self.ts.trees(root_threshold=2):
            if tree.has_single_root:
                self.root_spans[tree.root] += tree.span
        # Add on the spans when this is a root
        for root, span_when_root in self.root_spans.items():
            self.spans[root] += span_when_root

    # === Grouped edge iterators ===

    def edges_by_parent_asc(self, grouped=True):
        """
        Return an itertools.groupby object of edges grouped by parent in ascending order
        of the time of the parent. Since tree sequence properties guarantee that edges
        are listed in nondecreasing order of parent time
        (https://tskit.readthedocs.io/en/latest/data-model.html#edge-requirements)
        we can simply use the standard edge order
        """
        if grouped:
            return itertools.groupby(self.ts.edges(), operator.attrgetter("parent"))
        else:
            return self.ts.edges()

    def edges_by_child_desc(self, grouped=True):
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
        if grouped:
            return itertools.groupby(it, operator.attrgetter("child"))
        else:
            return it

    def edges_by_child_then_parent_desc(self, grouped=True):
        """
        Return an itertools.groupby object of edges grouped by child in descending order
        of the time of the child, then by descending order of age of child
        """
        wtype = np.dtype(
            [
                ("child_age", self.ts.nodes_time.dtype),
                ("child_node", self.ts.edges_child.dtype),
                ("parent_age", self.ts.nodes_time.dtype),
            ]
        )
        w = np.empty(self.ts.num_edges, dtype=wtype)
        w["child_age"] = self.ts.nodes_time[self.ts.edges_child]
        w["child_node"] = self.ts.edges_child
        w["parent_age"] = -self.ts.nodes_time[self.ts.edges_parent]
        sorted_child_parent = (
            self.ts.edge(i)
            for i in reversed(
                np.argsort(w, order=("child_age", "child_node", "parent_age"))
            )
        )
        if grouped:
            return itertools.groupby(sorted_child_parent, operator.attrgetter("child"))
        else:
            return sorted_child_parent

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
        assert (
            self.lik.standardize is False
        ), "Marginal likelihood requires unstandardized mutation likelihoods"
        marginal_lik = self.lik.identity_constant
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
            if standardize:
                marginal_lik = self.lik.combine(marginal_lik, denominator[parent])
        if cache_inside:
            self.g_i = self.lik.ratio(g_i, denominator[self.ts.edges_child, None])
        # Keep the results in this object
        self.inside = inside
        self.denominator = denominator
        # Calculate marginal likelihood
        for root, span_when_root in self.root_spans.items():
            spanfrac = span_when_root / self.spans[root]
            root_val = self.lik.scale_geometric(spanfrac, inside[root])
            marginal_lik = self.lik.combine(
                marginal_lik, self.lik.marginalize(root_val)
            )
        return marginal_lik

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
        overflow, but means that we cannot check the total functional value at each node

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
            total=len(np.unique(self.ts.edges_child)),
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
            np.isin(np.arange(self.ts.num_nodes), self.ts.edges_child, invert=True)
        )[0]
        for i in mrcas:
            if i not in self.fixednodes:
                maximized_node_times[i] = np.argmax(self.inside[i])

        for child, edges in tqdm(
            self.edges_by_child_then_parent_desc(),
            desc="Maximization",
            total=len(np.unique(self.ts.edges_child)),
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

    def __init__(self, *args, global_prior, **kwargs):
        """
        `global_prior` contains parameters of a gamma mixture, used as an iid
        prior, wheras `self.priors` (node-specific priors) are not used, except
        to provide a list of nonfixed nodes.
        """
        super().__init__(*args, **kwargs)

        assert self.priors.probability_space == base.GAMMA_PAR
        assert self.lik.probability_space == base.GAMMA_PAR
        assert self.lik.grid_size == 2
        assert self.priors.timepoints.size == 2

        # global distribution of node ages
        self.global_prior = global_prior.copy()

        # messages passed from prior to nodes
        self.prior_messages = np.zeros((self.ts.num_nodes, 2))

        # mutation likelihoods, as gamma natural parameters
        self.likelihoods = np.zeros((self.ts.num_edges, 2))
        for e in self.ts.edges():
            self.likelihoods[e.id] = self.lik.to_natural(e)

        # messages passed from edge likelihoods to nodes
        self.messages = np.zeros((self.ts.num_edges, 2, 2))

        # normalizing constants from each edge likelihood
        self.log_partition = np.zeros(self.ts.num_edges)

        # the approximate posterior marginals
        self.posterior = np.zeros((self.ts.num_nodes, 2))

        # edge, node traversal order
        self.edges, self.leaves = self.factorize(self.ts.edges(), self.fixednodes)
        self.freenodes = self.priors.nonfixed_nodes

        # factors for edges leading from fixed nodes are invariant
        # and can be incorporated into the posterior beforehand
        for i, p, c in self.leaves:
            if self.ts.nodes_time[c] != 0.0:
                raise ValueError("Fixed nodes must be at time zero")
            self.messages[i, 0] = self.likelihoods[i]
            self.posterior[p] += self.likelihoods[i]
            # self.log_partition[i] += ... # TODO

    @staticmethod
    def factorize(edge_list, fixed_nodes):
        """Split edges into internal and external"""
        internal = []
        external = []
        for edge in edge_list:
            i, p, c = edge.id, edge.parent, edge.child
            if p in fixed_nodes:
                raise ValueError("Internal nodes cannot be fixed")
            if c in fixed_nodes:
                external.extend([i, p, c])
            else:
                internal.extend([i, p, c])
        internal = np.array(internal, dtype=np.int32).reshape(-1, 3)
        external = np.array(external, dtype=np.int32).reshape(-1, 3)
        return internal, external

    @staticmethod
    @numba.njit("f8(i4[:, :], f8[:, :], f8[:, :], f8[:, :, :], f8[:], f8, b1)")
    def propagate_likelihood(
        edges,
        likelihoods,
        posterior,
        messages,
        log_partition,
        max_shape,
        min_kl,
    ):
        """
        Update approximating factors for each edge.

        TODO: return max difference in natural parameters for stopping criterion

        :param ndarray edges: integer array of dimension `[num_edges, 3]`
            containing edge id, parent id, and child id.
        :param ndarray likelihoods: array of dimension `[num_edges, 2]`
            containing mutation count and mutational target size per edge.
        :param ndarray posterior: array of dimension `[num_nodes, 2]`
            containing natural parameters for each node, updated in-place.
        :param ndarray messages: array of dimension `[num_edges, 2, 2]`
            containing parent/child messages (natural parameters) for each edge,
            updated in-place.
        :param ndarray log_partition: array of dimension `[num_edges]`
            containing the approximate normalizing constants per edge,
            updated in-place.
        """

        assert max_shape >= 1.0

        # Bound the shape parameter for the posterior and cavity distributions
        # so that lower_cavi < lower_post < upper_post < upper_cavi.
        upper_post = 1.0 * max_shape - 1.0
        lower_post = 1.0 / max_shape - 1.0
        upper_cavi = 2.0 * max_shape - 1.0
        lower_cavi = 0.5 / max_shape - 1.0

        def cavity_damping(x, y):
            assert upper_cavi > x[0] > lower_cavi
            d = 1.0
            if (y[0] > 0.0) and (x[0] - y[0] < lower_cavi):
                d = min(d, (x[0] - lower_cavi) / y[0])
            if (y[0] < 0.0) and (x[0] - y[0] > upper_cavi):
                d = min(d, (x[0] - upper_cavi) / y[0])
            if (y[1] > 0.0) and (x[1] - y[1] < 0.0):
                d = min(d, x[1] / y[1])
            assert 0.0 < d <= 1.0
            return d

        def posterior_damping(x):
            assert x[0] > -1.0 and x[1] > 0.0
            d = 1.0
            if x[0] > upper_post:
                d = upper_post / x[0]
            if x[0] < lower_post:
                d = lower_post / x[0]
            assert 0.0 < d <= 1.0
            return d

        scale = np.ones(posterior.shape[0])
        for i, p, c in edges:
            # Damped downdate to ensure proper cavity distributions
            parent_message = messages[i, 0] * scale[p]
            child_message = messages[i, 1] * scale[c]
            parent_delta = cavity_damping(posterior[p], parent_message)
            child_delta = cavity_damping(posterior[c], child_message)
            delta = min(parent_delta, child_delta)
            # The cavity posteriors: the approximation omitting the variational
            # factors for this edge.
            parent_cavity = posterior[p] - delta * parent_message
            child_cavity = posterior[c] - delta * child_message
            # The edge likelihood, scaled by the damping factor
            edge_likelihood = delta * likelihoods[i]
            # The target posterior: the cavity multiplied by the edge
            # likelihood then projected onto a gamma via moment matching.
            logconst, parent_post, child_post = approx.gamma_projection(
                parent_cavity, child_cavity, edge_likelihood, min_kl
            )
            # The messages: the difference in natural parameters between the
            # target and cavity posteriors.
            messages[i, 0] += (parent_post - posterior[p]) / scale[p]
            messages[i, 1] += (child_post - posterior[c]) / scale[c]
            # Contribution to the marginal likelihood from the edge
            log_partition[i] = logconst  # TODO: incomplete
            # Constrain the messages so that the gamma shape parameter for each
            # posterior is bounded (e.g. set a maximum precision for log(age)).
            parent_eta = posterior_damping(parent_post)
            child_eta = posterior_damping(child_post)
            posterior[p] = parent_eta * parent_post
            posterior[c] = child_eta * child_post
            scale[p] *= parent_eta
            scale[c] *= child_eta

        # move the scaling term into the messages
        for i, p, c in edges:
            messages[i, 0] *= scale[p]
            messages[i, 1] *= scale[c]

        return 0.0  # TODO, placeholder

    @staticmethod
    @numba.njit("f8(i4[:], f8[:, :], f8[:, :], f8[:, :], f8, i4, f8)")
    def propagate_prior(
        nodes, global_prior, posterior, messages, max_shape, em_maxitt, em_reltol
    ):
        """
        Update approximating factors for global prior at each node.

        :param ndarray nodes: ids of nodes that should be updated
        :param ndarray global_prior: rows are mixture components, columns are
            zeroth, first, and second natural parameters of gamma mixture
            components. Updated in place.
        :param ndarray posterior: rows are nodes, columns are first and
            second natural parameters of gamma posteriors. Updated in
            place.
        :param ndarray messages: rows are edges, columns are first and
            second natural parameters of prior messages. Updated in place.
        :param float max_shape: the maximum allowed shape for node posteriors
        :param int em_maxitt: the maximum number of EM iterations to use when
            fitting the mixture model
        :param int em_reltol: the termination criterion for relative change in
            log-likelihood
        """

        if global_prior.shape[0] == 0:
            return 0.0

        assert max_shape >= 1.0

        upper = max_shape - 1.0
        lower = 1.0 / max_shape - 1.0

        def posterior_damping(x):
            assert x[0] > -1.0 and x[1] > 0.0
            d = 1.0
            if x[0] > upper:
                d = upper / x[0]
            if x[0] < lower:
                d = lower / x[0]
            assert 0.0 < d <= 1.0
            return d

        cavity = np.zeros(posterior.shape)
        cavity[nodes] = posterior[nodes] - messages[nodes]
        global_prior, posterior[nodes] = mixture.fit_gamma_mixture(
            global_prior, cavity[nodes], em_maxitt, em_reltol, False
        )
        messages[nodes] = posterior[nodes] - cavity[nodes]

        for n in nodes:
            eta = posterior_damping(posterior[n])
            posterior[n] *= eta
            messages[n] *= eta

        return 0.0

    def iterate(self, em_maxitt=100, em_reltol=1e-6, max_shape=1000, min_kl=True):
        """
        Update edge factors from leaves to root then from root to leaves,
        and return approximate log marginal likelihood (TODO)
        """

        # prior update
        self.propagate_prior(
            self.freenodes,
            self.global_prior,
            self.posterior,
            self.prior_messages,
            max_shape,
            em_maxitt,
            em_reltol,
        )

        # rootward pass
        self.propagate_likelihood(
            self.edges,
            self.likelihoods,
            self.posterior,
            self.messages,
            self.log_partition,
            max_shape,
            min_kl,
        )

        # leafward pass
        self.propagate_likelihood(
            self.edges[::-1],
            self.likelihoods,
            self.posterior,
            self.messages,
            self.log_partition,
            max_shape,
            min_kl,
        )

        return np.nan  # TODO: placeholder for marginal likelihood


def constrain_ages_topo(ts, node_times, epsilon, progress=False):
    # If node_times violate the topology in ts, return increased node_times so that each
    # node is guaranteed to be older than any of its children.
    edges_parent = ts.edges_parent
    edges_child = ts.edges_child

    new_node_times = np.copy(node_times)
    # Traverse through the ARG, ensuring children come before parents.
    # This can be done by iterating over groups of edges with the same parent
    new_parent_edge_idx = np.where(np.diff(edges_parent) != 0)[0] + 1
    for edges_start, edges_end in tqdm(
        zip(
            itertools.chain([0], new_parent_edge_idx),
            itertools.chain(new_parent_edge_idx, [len(edges_parent)]),
        ),
        desc="Constrain Ages",
        total=len(new_parent_edge_idx) + 1,
        disable=not progress,
    ):
        parent = edges_parent[edges_start]
        child_ids = edges_child[edges_start:edges_end]  # May contain dups
        oldest_child_time = np.max(new_node_times[child_ids])
        if oldest_child_time >= new_node_times[parent]:
            new_node_times[parent] = oldest_child_time + epsilon
    return new_node_times


# Classes for each method
Results = namedtuple(
    "Results",
    ["posterior_mean", "posterior_var", "posterior_obj", "mutation_likelihood"],
)


class EstimationMethod:
    """
    Base class to hold the various estimation methods. Override prior_grid_func_name with
    something like "parameter_grid" or "prior_grid".
    """

    prior_grid_func_name = None

    def run():
        # Subclasses should override to return a return a Results object
        raise NotImplementedError(
            "Base class 'EstimationMethod' not intended for direct use"
        )

    def __init__(
        self,
        ts,
        *,
        mutation_rate=None,
        population_size=None,
        recombination_rate=None,
        time_units=None,
        priors=None,
        return_posteriors=None,
        return_likelihood=None,
        record_provenance=None,
        progress=None,
    ):
        # Set up all the generic params describe in the tsdate.date function, and define
        # priors if not passed-in already
        self.ts = ts
        self.mutation_rate = mutation_rate
        self.recombination_rate = recombination_rate
        self.return_posteriors = return_posteriors
        self.return_likelihood = return_likelihood
        self.pbar = progress
        self.time_units = "generations" if time_units is None else time_units
        if record_provenance is None:
            record_provenance = True
        Ne = population_size  # shorthand
        if isinstance(Ne, dict):
            Ne = demography.PopulationSizeHistory(**Ne)

        self.provenance_params = None
        if record_provenance:
            self.provenance_params = dict(
                mutation_rate=mutation_rate,
                recombination_rate=recombination_rate,
                time_units=time_units,
                progress=progress,
                # demography.PopulationSizeHistory provides as_dict() for saving
                population_size=Ne.as_dict() if hasattr(Ne, "as_dict") else Ne,
            )

        if priors is None:
            if Ne is None:
                raise ValueError(
                    "Must specify population size if priors are not already built using"
                    f"tsdate.build_{self.prior_grid_func_name}()"
                )
            mk_prior = getattr(prior, self.prior_grid_func_name)
            # Default to not creating approximate priors unless ts has
            # greater than DEFAULT_APPROX_PRIOR_SIZE samples
            approx = True if ts.num_samples > base.DEFAULT_APPROX_PRIOR_SIZE else False
            self.priors = mk_prior(ts, Ne, approximate_priors=approx, progress=progress)
        else:
            logging.info("Using user-specified priors")
            if Ne is not None:
                raise ValueError(
                    "Cannot specify population size if specifying priors "
                    f"from tsdate.build_{self.prior_grid_func_name}()"
                )
            self.priors = priors

    def get_modified_ts(self, result, eps):
        # Return a new ts based on the existing one, but with the various
        # time-related information correctly set.
        tables = self.ts.dump_tables()
        nodes = tables.nodes
        if self.provenance_params is not None:
            provenance.record_provenance(tables, self.name, **self.provenance_params)
        nodes.time = constrain_ages_topo(self.ts, result.posterior_mean, eps, self.pbar)
        tables.time_units = self.time_units
        tables.mutations.time = np.full(self.ts.num_mutations, tskit.UNKNOWN_TIME)
        # Add posterior mean and variance to node metadata
        if result.posterior_obj is not None:
            metadata_array = tskit.unpack_bytes(nodes.metadata, nodes.metadata_offset)
            for u in result.posterior_obj.nonfixed_nodes:
                metadata_array[u] = json.dumps(
                    {
                        "mn": result.posterior_mean[u],
                        "vr": result.posterior_var[u],
                    }
                ).encode()
            nodes.packset_metadata(metadata_array)
        tables.sort()
        return tables.tree_sequence()

    def parse_result(self, result, epsilon, extra_posterior_cols=None):
        # Construct the tree sequence to return and add other stuff we might want to
        # return. pst_cols is a dict to be appended to the output posterior dict
        ret = [self.get_modified_ts(result, epsilon)]
        if self.return_posteriors:
            pst_dict = None
            if result.posterior_obj is not None:
                pst_dict = result.posterior_obj.nonfixed_dict()
                pst_dict.update(extra_posterior_cols or {})
            ret.append(pst_dict)
        if self.return_likelihood:
            ret.append(result.mutation_likelihood)
        return tuple(ret) if len(ret) > 1 else ret.pop()

    def get_fixed_nodes_set(self):
        # TODO: modify to allow non-contemporary samples. If these have priors specified
        # they should work fine with these algorithms.
        for sample in self.ts.samples():
            if self.ts.node(sample).time != 0:
                raise NotImplementedError("Samples must all be at time 0")
        return set(self.ts.samples())


class DiscreteTimeMethod(EstimationMethod):
    prior_grid_func_name = "prior_grid"

    @staticmethod
    def mean_var(ts, posterior):
        """
        Mean and variance of node age given an atomic time discretization. Fixed
        nodes will be given a mean of their exact time in the tree sequence, and
        zero variance. This is a static method for ease of testing.
        """
        mn_post = np.full(ts.num_nodes, np.nan)  # Fill with NaNs so we detect when
        va_post = np.full(ts.num_nodes, np.nan)  # there's been an error

        is_fixed = np.ones(posterior.num_nodes, dtype=bool)
        is_fixed[posterior.nonfixed_nodes] = False
        mn_post[is_fixed] = ts.nodes_time[is_fixed]
        va_post[is_fixed] = 0

        for u in posterior.nonfixed_nodes:
            probs = posterior[u]
            times = posterior.timepoints
            mn_post[u] = np.sum(probs * times) / np.sum(probs)
            va_post[u] = np.sum(((mn_post[u] - (times)) ** 2) * (probs / np.sum(probs)))

        return mn_post, va_post

    def main_algorithm(self, probability_space, epsilon, num_threads):
        # Algorithm class is shared by inside-outside & outside-maximization methods
        if probability_space != base.LOG:
            liklhd = Likelihoods(
                self.ts,
                self.priors.timepoints,
                self.mutation_rate,
                self.recombination_rate,
                eps=epsilon,
                fixed_node_set=self.get_fixed_nodes_set(),
                progress=self.pbar,
            )
        else:
            liklhd = LogLikelihoods(
                self.ts,
                self.priors.timepoints,
                self.mutation_rate,
                self.recombination_rate,
                eps=epsilon,
                fixed_node_set=self.get_fixed_nodes_set(),
                progress=self.pbar,
            )
        if self.mutation_rate is not None:
            liklhd.precalculate_mutation_likelihoods(num_threads=num_threads)

        return InOutAlgorithms(self.priors, liklhd, progress=self.pbar)


class InsideOutsideMethod(DiscreteTimeMethod):
    name = "inside_outside"

    def run(
        self,
        eps,
        outside_standardize,
        ignore_oldest_root,
        probability_space,
        num_threads=None,
        cache_inside=None,
    ):
        if self.provenance_params is not None:
            self.provenance_params.update(
                {k: v for k, v in locals().items() if k != "self"}
            )
        dynamic_prog = self.main_algorithm(probability_space, eps, num_threads)
        marginal_likl = dynamic_prog.inside_pass(cache_inside=cache_inside)
        posterior_obj = dynamic_prog.outside_pass(
            standardize=outside_standardize, ignore_oldest_root=ignore_oldest_root
        )
        # Turn the posterior into probabilities
        posterior_obj.standardize()  # Just to ensure there are no floating point issues
        posterior_obj.force_probability_space(base.LIN)
        posterior_obj.to_probabilities()

        posterior_mean, posterior_var = self.mean_var(self.ts, posterior_obj)
        return Results(posterior_mean, posterior_var, posterior_obj, marginal_likl)


class MaximizationMethod(DiscreteTimeMethod):
    name = "maximization"

    def __init__(self, ts, **kwargs):
        super().__init__(ts, **kwargs)
        if self.return_posteriors:
            raise ValueError("Cannot return posterior with maximization method")

    def run(
        self,
        eps,
        probability_space=None,
        num_threads=None,
        cache_inside=None,
    ):
        if self.mutation_rate is None:
            raise ValueError("Outside maximization method requires mutation rate")
        if self.provenance_params is not None:
            self.provenance_params.update(
                {k: v for k, v in locals().items() if k != "self"}
            )
        dynamic_prog = self.main_algorithm(probability_space, eps, num_threads)
        marginal_likl = dynamic_prog.inside_pass(cache_inside=cache_inside)
        posterior_mean = dynamic_prog.outside_maximization(eps=eps)
        return Results(posterior_mean, None, None, marginal_likl)


class VariationalGammaMethod(EstimationMethod):
    prior_grid_func_name = "parameter_grid"
    name = "variational_gamma"

    def __init__(self, ts, **kwargs):
        super().__init__(ts, **kwargs)
        # convert priors to natural parameterization
        for n in self.priors.nonfixed_nodes:
            if not np.all(self.priors[n] > 0.0):
                raise ValueError(
                    f"Non-positive shape/rate parameters for node {n}: "
                    f"{self.priors[n]}"
                )
            self.priors[n][0] -= 1.0

    @staticmethod
    def mean_var(ts, posterior):
        """
        Mean and variance of node age from variational posterior (e.g. gamma
        distributions).  Fixed nodes will be given a mean of their exact time in
        the tree sequence, and zero variance (as long as they are identified by the
        fixed_node_set). This is a static method for ease of testing.
        """

        assert posterior.grid_data.shape[1] == 2
        assert np.all(posterior.grid_data > 0)

        mn_post = np.full(ts.num_nodes, np.nan)  # Fill with NaNs so we detect when
        va_post = np.full(ts.num_nodes, np.nan)  # there's been an error

        is_fixed = np.ones(posterior.num_nodes, dtype=bool)
        is_fixed[posterior.nonfixed_nodes] = False
        mn_post[is_fixed] = ts.nodes_time[is_fixed]
        va_post[is_fixed] = 0

        for node in posterior.nonfixed_nodes:
            pars = posterior[node]
            mn_post[node] = pars[0] / pars[1]
            va_post[node] = pars[0] / pars[1] ** 2
        return mn_post, va_post

    def main_algorithm(self):
        lik = VariationalLikelihoods(
            self.ts,
            self.mutation_rate,
            self.recombination_rate,
            fixed_node_set=self.get_fixed_nodes_set(),
        )
        return ExpectationPropagation(
            self.priors, lik, progress=self.pbar, global_prior=self.global_prior
        )

    def run(
        self,
        eps,
        max_iterations,
        max_shape,
        match_central_moments,
        prior_mixture_dim,
        em_iterations,
    ):
        if self.provenance_params is not None:
            self.provenance_params.update(
                {k: v for k, v in locals().items() if k != "self"}
            )
        if not max_iterations >= 1:
            raise ValueError("Maximum number of EP iterations must be greater than 0")
        if not (isinstance(prior_mixture_dim, int) and prior_mixture_dim > 0):
            raise ValueError("Number of mixture components must be a positive integer")
        if self.mutation_rate is None:
            raise ValueError("Variational gamma method requires mutation rate")

        # initialize weights/shapes/rates for i.i.d mixture prior
        # note that self.priors (node-specific priors) are not currently
        # used except for initialization of the mixture
        self.global_prior = mixture.initialize_mixture(
            self.priors.grid_data, prior_mixture_dim
        )
        self.priors.grid_data[:] = [0.0, 0.0]  # TODO: support node-specific priors

        # match sufficient statistics or match central moments
        min_kl = not match_central_moments
        dynamic_prog = self.main_algorithm()
        for itt in tqdm(
            np.arange(max_iterations),
            desc="Expectation Propagation",
            disable=not self.pbar,
        ):
            dynamic_prog.iterate(
                em_maxitt=em_iterations if itt else 0,
                max_shape=max_shape,
                min_kl=min_kl,
            )

        num_skipped = np.sum(np.isnan(dynamic_prog.log_partition))
        if num_skipped > 0:
            logging.info(
                f"Skipped {num_skipped} messages with invalid posterior updates."
            )

        posterior_obj = self.priors.clone_with_new_data(
            grid_data=dynamic_prog.posterior[self.priors.nonfixed_nodes, :]
        )
        posterior_obj.grid_data[:, 0] += 1  # to shape/rate parameterization
        posterior_mean, posterior_var = self.mean_var(self.ts, posterior_obj)
        # TODO: return marginal likelihood
        return Results(posterior_mean, posterior_var, posterior_obj, None)


def maximization(
    tree_sequence,
    *,
    eps=None,
    num_threads=None,
    probability_space=None,
    # below deliberately undocumented
    cache_inside=None,
    # Other params documented in `.date()`
    **kwargs,
):
    """
    Infer dates for nodes in a genealogical graph using the "outside maximization"
    algorithm. This approximates the marginal posterior distribution of a node's
    age using an atomic discretization of time (e.g. point masses at particular
    timepoints).

    This estimation method comprises a single "inside" step followed by an
    "outside maximization" step. The inside step passes backwards in time from the
    samples to the roots of the graph,taking account of the distributions of times of
    each node's child (and if a ``mutation_rate`` is given, the the number of mutations
    on each edge). The outside maximization step passes forwards in time from the roots,
    updating each node's time on the basis of the most likely timepoint for
    each parent of that node. This provides a reasonable point estimate for node times,
    but does not generate a true posterior time distribution.

    For example:

    .. code-block:: python

      new_ts = tsdate.maximization(ts, mutation_rate=1e-8, population_size=1e4)

    .. note::
        The prior parameters for each node-to-be-dated take the form of probabilities
        for each node at a set of discrete timepoints. If the ``priors`` parameter is
        used, it must specify an object constructed using :func:`build_prior_grid`
        (this can be used to define the number and position of the timepoints).
        If ``priors`` is not used, ``population_size`` must be provided,
        which is used to create a default prior derived from the conditional coalescent
        (tilted according to population size and weighted by the genomic
        span over which a node has a given number of descendant samples). This default
        prior assumes the nodes to be dated are all the non-sample nodes in the input
        tree sequence, and that they are contemporaneous.

    :param ~tskit.TreeSequence tree_sequence: The input tree sequence to be dated.
    :param float eps: The error factor in time difference calculations, and the
        minimum distance separating parent and child ages in the returned tree sequence.
        Default: None, treated as 1e-6.
    :param int num_threads: The number of threads to use when precalculating likelihoods.
        A simpler unthreaded algorithm is used unless this is >= 1. Default: None
    :param string probability_space: Should the internal algorithm save
        probabilities in "logarithmic" (slower, less liable to to overflow) or
        "linear" space (fast, may overflow). Default: None treated as"logarithmic"
    :param \\**kwargs: Other keyword arguments as described in the :func:`date` wrapper
        function, notably ``mutation_rate``, and ``population_size`` or ``priors``.
        Further arguments include ``time_units``, ``progress``, and
        ``record_provenance``. The additional ``return_likelihood`` argument can be used
        to return additional information (see below). Posteriors cannot be returned using
        this estimation method.
    :return:
        - **ts** (:class:`~tskit.TreeSequence`) -- a copy of the input tree sequence with
          updated node times based on the posterior mean, corrected where necessary to
          ensure that parents are strictly older than all their children by an amount
          given by the ``eps`` parameter.
        - **marginal_likelihood** (:py:class:`float`) -- (Only returned if
          ``return_likelihood`` is ``True``) The marginal likelihood of
          the mutation data given the inferred node times.
    """
    if eps is None:
        eps = 1e-6
    if probability_space is None:
        probability_space = base.LOG

    dating_method = MaximizationMethod(tree_sequence, **kwargs)
    result = dating_method.run(
        eps=eps,
        num_threads=num_threads,
        cache_inside=cache_inside,
        probability_space=probability_space,
    )
    return dating_method.parse_result(result, eps)


def inside_outside(
    tree_sequence,
    *,
    eps=None,
    num_threads=None,
    outside_standardize=None,
    ignore_oldest_root=None,
    probability_space=None,
    # below deliberately undocumented
    cache_inside=False,
    # Other params documented in `.date()`
    **kwargs,
):
    """
    Infer dates for nodes in a genealogical graph using the "inside outside" algorithm.
    This approximates the marginal posterior distribution of a node's age using an
    atomic discretization of time (e.g. point masses at particular timepoints).

    Currently, this estimation method comprises a single "inside" followed by a similar
    "outside" step. The inside step passes backwards in time from the samples to the
    roots of the graph,taking account of the distributions of times of each node's child
    (and if a ``mutation_rate`` is given, the the number of mutations on each edge).
    The outside step passes forwards in time from the roots, incorporating the time
    distributions for each node's parents. If there are (undirected) cycles in the
    underlying graph, this method does not provide a theoretically exact estimate
    of the marginal posterior distribution of node ages, but in practice it
    results in an accurate approximation.

    For example:

    .. code-block:: python

      new_ts = tsdate.inside_outside(ts, mutation_rate=1e-8, population_size=1e4)

    .. note::
        The prior parameters for each node-to-be-dated take the form of probabilities
        for each node at a set of discrete timepoints. If the ``priors`` parameter is
        used, it must specify an object constructed using :func:`build_prior_grid`
        (this can be used to define the number and position of the timepoints).
        If ``priors`` is not used, ``population_size`` must be provided,
        which is used to create a default prior derived from the conditional coalescent
        (tilted according to population size and weighted by the genomic
        span over which a node has a given number of descendant samples). This default
        prior assumes the nodes to be dated are all the non-sample nodes in the input
        tree sequence, and that they are contemporaneous.

    :param ~tskit.TreeSequence tree_sequence: The input tree sequence to be dated.
    :param float eps: The error factor in time difference calculations, and the
        minimum distance separating parent and child ages in the returned tree sequence.
        Default: None, treated as 1e-6.
    :param int num_threads: The number of threads to use when precalculating likelihoods.
        A simpler unthreaded algorithm is used unless this is >= 1. Default: None
    :param bool outside_standardize: Should the likelihoods be standardized during the
        outside step? This can help to avoid numerical under/overflow. Using
        unstandardized values is mostly useful for testing (e.g. to obtain, in the
        outside step, the total functional value for each node).
        Default: None, treated as True.
    :param bool ignore_oldest_root: Should the oldest root in the tree sequence be
        ignored in the outside algorithm (if ``"inside_outside"`` is used as the method).
        Ignoring outside root can provide greater stability when dating tree sequences
        inferred from real data, in particular if all local trees are assumed to coalesce
        in a single "grand MRCA", as in older versions of ``tsinfer``.
        Default: None, treated as False.
    :param string probability_space: Should the internal algorithm save
        probabilities in "logarithmic" (slower, less liable to to overflow) or
        "linear" space (fast, may overflow). Default: "logarithmic"
    :param \\**kwargs: Other keyword arguments as described in the :func:`date` wrapper
        function, notably ``mutation_rate``, and ``population_size`` or ``priors``.
        Further arguments include ``time_units``, ``progress``, and
        ``record_provenance``. The additional arguments ``return_posteriors`` and
        ``return_likelihood`` can be used to return additional information (see below).
    :return:
        - **ts** (:class:`~tskit.TreeSequence`) -- a copy of the input tree sequence with
          updated node times based on the posterior mean, corrected where necessary to
          ensure that parents are strictly older than all their children by an amount
          given by the ``eps`` parameter.
        - **posteriors** (:py:class:`dict`) -- (Only returned if ``return_posteriors``
          is ``True``) A dictionary of posterior probabilities.
          Each node whose time was inferred corresponds to an item in this dictionary
          whose key is the node ID and value is an array of probabilities of the node
          being at a list of timepoints. Timepoint values are provided in the
          returned dictionary under the key named "time". When read
          as a pandas ``DataFrame`` object using ``pd.DataFrame(posteriors)``,
          the rows correspond to labelled timepoints and columns are
          headed by their respective node ID.
        - **marginal_likelihood** (:py:class:`float`) -- (Only returned if
          ``return_likelihood`` is ``True``) The marginal likelihood of
          the mutation data given the inferred node times.
    """
    if eps is None:
        eps = 1e-6
    if probability_space is None:
        probability_space = base.LOG
    if outside_standardize is None:
        outside_standardize = True
    if ignore_oldest_root is None:
        ignore_oldest_root = False
    dating_method = InsideOutsideMethod(tree_sequence, **kwargs)
    result = dating_method.run(
        eps=eps,
        num_threads=num_threads,
        outside_standardize=outside_standardize,
        ignore_oldest_root=ignore_oldest_root,
        cache_inside=cache_inside,
        probability_space=probability_space,
    )
    return dating_method.parse_result(
        result, eps, {"time": result.posterior_obj.timepoints}
    )


def variational_gamma(
    tree_sequence,
    *,
    eps=None,
    max_iterations=None,
    max_shape=None,
    match_central_moments=None,
    prior_mixture_dim=None,
    em_iterations=None,
    **kwargs,
):
    """
    Infer dates for nodes in a tree sequence using expectation propagation,
    which approximates the marginal posterior distribution of a given node's
    age with a gamma distribution. Convergence to the correct posterior moments
    is obtained by updating the distributions for node dates using several rounds
    of iteration. For example:

    .. code-block:: python

      new_ts = tsdate.variational_gamma(
          ts, mutation_rate=1e-8, population_size=1e4, max_iterations=10)

    An i.i.d. gamma mixture is used as a prior for each node, that is
    initialized from the conditional coalescent and updated via expectation
    maximization in each iteration. If node-specific priors are supplied
    (via a grid of shape/rate parameters) then these are used for
    initialization.

    .. note::
        The prior parameters for each node-to-be-dated take the form of a
        gamma-distributed prior on node age, parameterised by shape and rate.
        If the ``priors`` parameter is used, it must specify an object constructed
        using :func:`build_parameter_grid`. If not used, ``population_size`` must be
        provided, which is used to create an iid prior derived from the conditional
        coalescent prior (tilted according to population size), assuming the nodes
        to be dated are all the non-sample nodes in the input tree sequence.

    :param ~tskit.TreeSequence tree_sequence: The input tree sequence to be dated.
    :param float eps: The minimum distance separating parent and child ages in
        the returned tree sequence. Default: None, treated as 1e-6
    :param int max_iterations: The number of iterations used in the expectation
        propagation algorithm. Default: None, treated as 20.
    :param float max_shape: The maximum value for the shape parameter in the variational
        posteriors. This is equivalent to the maximum precision (inverse variance) on a
        logarithmic scale. Default: None, treated as 1000.
    :param bool match_central_moments: If `True`, each expectation propgation
        update matches mean and variance rather than expected gamma sufficient
        statistics. Faster with a similar accuracy, but does not exactly minimize
        Kullback-Leibler divergence. Default: None, treated as False.
    :param int prior_mixture_dim: The number of components in the i.i.d. mixture prior
        for node ages. Default: None, treated as 1.
    :param int em_iterations: The number of expectation maximization iterations used
        to optimize the i.i.d. mixture prior. Setting to zero disables optimization.
        Default: None, treated as 10.
    :param \\**kwargs: Other keyword arguments as described in the :func:`date` wrapper
        function, notably ``mutation_rate``, and ``population_size`` or ``priors``.
        Further arguments include ``time_units``, ``progress``, and
        ``record_provenance``. The additional arguments ``return_posteriors`` and
        ``return_likelihood`` can be used to return additional information (see below).
    :return:
        - **ts** (:class:`~tskit.TreeSequence`) -- a copy of the input tree sequence with
          updated node times based on the posterior mean, corrected where necessary to
          ensure that parents are strictly older than all their children by an amount
          given by the ``eps`` parameter.
        - **posteriors** (:py:class:`dict`) -- (Only returned if ``return_posteriors``
          is ``True``) A dictionary of posterior probabilities.
          Each node whose time was inferred corresponds to an item in this dictionary
          whose key is the node ID and value is an array of the ``[shape, rate]``
          parameters of the posterior gamma distribution for that node. When read
          as a pandas ``DataFrame`` object using ``pd.DataFrame(posteriors)``,
          the first row of the data frame is the shape and the second the rate
          parameter, each column being headed by the respective node ID.
        - **marginal_likelihood** (:py:class:`float`) -- (Only returned if
          ``return_likelihood`` is ``True``) The marginal likelihood of
          the mutation data given the inferred node times. Not currently
          implemented for this method (set to ``None``)
    """
    if eps is None:
        eps = 1e-6
    if max_iterations is None:
        max_iterations = 20
    if max_shape is None:
        max_shape = 1000
    if match_central_moments is None:
        match_central_moments = False
    if prior_mixture_dim is None:
        prior_mixture_dim = 1
    if em_iterations is None:
        em_iterations = 10

    dating_method = VariationalGammaMethod(tree_sequence, **kwargs)
    result = dating_method.run(
        eps=eps,
        max_iterations=max_iterations,
        max_shape=max_shape,
        match_central_moments=match_central_moments,
        prior_mixture_dim=prior_mixture_dim,
        em_iterations=em_iterations,
    )
    return dating_method.parse_result(result, eps, {"parameter": ["shape", "rate"]})


estimation_methods = {
    "inside_outside": inside_outside,
    "maximization": maximization,
    "variational_gamma": variational_gamma,
}
"""
The names of available estimation methods, each mapped to a function to carry
out the appropriate method. Names can be passed as strings to the
:func:`~tsdate.date` function, or each named function can be called directly:

* :func:`tsdate.inside_outside`: empirically better, theoretically problematic.
* :func:`tsdate.maximization`: worse empirically, especially with gamma approximated
  priors, but theoretically robust
* :func:`tsdate.variational_gamma`: variational approximation, empirically most accurate.
"""


def date(
    tree_sequence,
    mutation_rate,
    population_size=None,
    recombination_rate=None,
    time_units=None,
    priors=None,
    method=None,
    *,
    return_posteriors=None,
    return_likelihood=None,
    progress=None,
    record_provenance=True,
    # Deprecated params
    Ne=None,
    # Other kwargs documented in the functions for each specific estimation-method
    **kwargs,
):
    """
    Infer dates for nodes in a genealogical graph (or :ref:`ARG<tutorials:sec_args>`)
    stored in the :ref:`succinct tree sequence<tskit:sec_introduction>` format.
    New times are assigned to nodes using the estimation algorithm specified by
    ``method`` (see note below). If a ``mutation_rate`` is given,
    the mutation clock is used. The recombination clock is unsupported at this
    time.  If neither a ``mutation_rate`` nor a ``recombination_rate`` is given, a
    topology-only clock is used. Times associated with mutations and times associated
    with non-fixed (non-sample) nodes are overwritten. For example:

    .. code-block:: python

      mu = 1e-8
      Ne = ts.diversity()/4/mu  #  In the absence of external info, use ts for prior Ne
      new_ts = tsdate.date(ts, mutation_rate=mu, population_size=Ne)

    .. note::
        This is a wrapper for the named functions that are listed in
        :data:`~tsdate.core.estimation_methods`. Details and specific parameters for
        each estimation method are given in the documentation for those functions.

    :param ~tskit.TreeSequence tree_sequence: The input tree sequence to be dated (for
        example one with :data:`uncalibrated<tskit.TIME_UNITS_UNCALIBRATED>` node times).
    :param float or ~demography.PopulationSizeHistory population_size: The estimated
        (diploid) effective population size used to construct the (default) conditional
        coalescent prior. For a population with constant size, this can be given as a
        single value (for example, as commonly estimated by the observed genetic
        diversity of the sample divided by four-times the expected mutation rate).
        Alternatively, for a population with time-varying size, this can be given
        directly as a :class:`~demography.PopulationSizeHistory` object or a parameter
        dictionary passed to initialise a :class:`~demography.PopulationSizeHistory`
        object. The ``population_size`` parameter is only used when ``priors`` is
        ``None``. Conversely, if ``priors`` is not ``None``, no ``population_size``
        value should be specified.
    :param float mutation_rate: The estimated mutation rate per unit of genome per
        unit time. If provided, the dating algorithm will use a mutation rate clock to
        help estimate node dates. Default: ``None``
    :param float recombination_rate: The estimated recombination rate per unit of genome
        per unit time. If provided, the dating algorithm will use a recombination rate
        clock to help estimate node dates. Default: ``None`` (not currently implemented)
    :param str time_units: The time units used by the ``mutation_rate`` and
        ``recombination_rate`` values, and stored in the ``time_units`` attribute of the
        output tree sequence. If the conditional coalescent prior is used,
        then this is also applies to the value of ``population_size``, which in
        standard coalescent theory is measured in generations. Therefore if you
        wish to use mutation and recombination rates measured in (say) years,
        and are using the conditional coalescent prior, the ``population_size``
        value which you provide must be scaled by multiplying by the number of
        years per generation. If ``None`` (default), assume ``"generations"``.
    :param tsdate.base.NodeGridValues priors: NodeGridValues object containing the prior
        parameters for each node-to-be-dated. Note that different estimation methods may
        require different types of prior, as described in the documentation for each
        estimation method.
    :param string method: What estimation method to use. See
        :data:`~tsdate.core.estimation_methods` for possible values.
        If ``None`` (default) the "inside_outside" method is currently chosen.
    :param bool return_posteriors: If ``True``, instead of returning just a dated tree
        sequence, return a tuple of ``(dated_ts, posteriors)``.
        Default: None, treated as False.
    :param bool return_likelihood: If ``True``, return the log marginal likelihood
        from the inside algorithm in addition to the dated tree sequence. If
        ``return_posteriors`` is also ``True``, then the marginal likelihood
        will be the last element of the tuple. Default: None, treated as False.
    :param bool progress: Show a progress bar. Default: None, treated as False.
    :param bool record_provenance: Should the tsdate command be appended to the
        provenence information in the returned tree sequence?
        Default: None, treated as True.
    :param float Ne: Deprecated, use the``population_size`` argument instead.
    :param \\**kwargs: Other keyword arguments specific to the
        :data:`estimation method<tsdate.core.estimation_methods>` used. These are
        documented in those specific functions.
    :return:
        A copy of the input tree sequence but with updated node times, or (if
        ``return_posteriors`` or ``return_likelihood`` is True) a tuple of that
        tree sequence plus a dictionary of posterior probabilities and/or the
        marginal likelihood given the mutations on the tree sequence.
    """
    # Only the .date() wrapper needs to consider the deprecated "Ne" param
    if Ne is not None:
        if population_size is not None:
            raise ValueError(
                "Only one of Ne (deprecated) or population_size may be specified"
            )
        else:
            population_size = Ne
    if method is None:
        method = "inside_outside"  # may change later
    if method not in estimation_methods:
        raise ValueError(f"method must be one of {list(estimation_methods.keys())}")

    return estimation_methods[method](
        tree_sequence,
        population_size=population_size,
        mutation_rate=mutation_rate,
        recombination_rate=recombination_rate,
        time_units=time_units,
        priors=priors,
        progress=progress,
        return_posteriors=return_posteriors,
        return_likelihood=return_likelihood,
        record_provenance=record_provenance,
        **kwargs,
    )
