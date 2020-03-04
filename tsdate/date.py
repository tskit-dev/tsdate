# MIT License
#
# Copyright (c) 2020 University of Oxford
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
from collections import defaultdict
import itertools
import multiprocessing
import operator
import functools
import logging

import numba
import numpy as np
import scipy.stats
from tqdm import tqdm

from . import base, util, prior

FORMAT_NAME = "tsdate"


class Likelihoods:
    """
    A class to store and process likelihoods. Likelihoods for edges are stored as a
    flattened lower triangular matrix of all the possible delta t's. This class also
    provides methods for accessing this lower triangular matrix, multiplying it, etc.
    """
    probability_space = base.LIN
    identity_constant = 1.0
    null_constant = 0.0

    def __init__(self, ts, timepoints, theta=None, rho=None, *,
                 eps=0, fixed_node_set=None, normalize=True, progress=False):
        self.ts = ts
        self.timepoints = timepoints
        self.fixednodes = set(ts.samples()) if fixed_node_set is None else fixed_node_set
        self.theta = theta
        self.rho = rho
        self.normalize = normalize
        self.grid_size = len(timepoints)
        self.tri_size = self.grid_size * (self.grid_size + 1) / 2
        self.ll_mut = {}
        self.mut_edges = self.get_mut_edges(ts)
        self.progress = progress
        # Need to set eps properly in the 2 lines below, to account for values in the
        # same timeslice
        self.timediff_lower_tri = np.concatenate(
            [self.timepoints[time_index] - self.timepoints[0:time_index + 1] + eps
                for time_index in np.arange(len(self.timepoints))])
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
            [np.arange(time_idx + 1) for time_idx in np.arange(self.grid_size)])
        self.to_upper_tri = np.concatenate(
            [np.arange(time_idx, self.grid_size)
                for time_idx in np.arange(self.grid_size + 1)])

    @staticmethod
    def get_mut_edges(ts):
        """
        Get the number of mutations on each edge in the tree sequence.
        """
        edge_diff_iter = ts.edge_diffs()
        right = 0
        edges_by_child = {}  # contains {child_node:edge_id}
        mut_edges = np.zeros(ts.num_edges, dtype=np.int64)
        for site in ts.sites():
            while right <= site.position:
                (left, right), edges_out, edges_in = next(edge_diff_iter)
                for e in edges_out:
                    del edges_by_child[e.child]
                for e in edges_in:
                    assert e.child not in edges_by_child
                    edges_by_child[e.child] = e.id
            for m in site.mutations:
                # In some cases, mutations occur above the root
                # These don't provide any information for the inside step
                if m.node in edges_by_child:
                    edge_id = edges_by_child[m.node]
                    mut_edges[edge_id] += 1
        return mut_edges

    @staticmethod
    def _lik(muts, span, dt, theta, normalize=True):
        """
        The likelihood of an edge given a number of mutations, as set of time deltas (dt)
        and a span. This is a static function to allow parallelization
        """
        ll = scipy.stats.poisson.pmf(muts, dt * theta / 2 * span)
        if normalize:
            return ll / np.max(ll)
        else:
            return ll

    @staticmethod
    def _lik_wrapper(muts_span, dt, theta, normalize=True):
        """
        A wrapper to allow this _lik to be called by pool.imap_unordered, returning the
        mutation and span values
        """
        return muts_span, Likelihoods._lik(
            muts_span[0], muts_span[1], dt, theta, normalize=normalize)

    def precalculate_mutation_likelihoods(
            self, num_threads=None, unique_method=0):
        """
        We precalculate these because the pmf function is slow, but can be trivially
        parallelised. We store the likelihoods in a cache because they only depend on
        the number of mutations and the span, so can potentially be reused.

        However, we don't bother storing the likelihood for edges above a *fixed* node,
        because (a) these are only used once per node and (b) sample edges are often
        long, and hence their span will be unique. This also allows us to deal easily
        with fixed nodes at explicit times (rather than in time slices)
        """

        if self.theta is None:
            raise RuntimeError("Cannot calculate mutation likelihoods with no theta set")
        if unique_method == 0:
            self.unfixed_likelihood_cache = {
                (muts, util.edge_span(e)): None for muts, e in
                zip(self.mut_edges, self.ts.edges())
                if e.child not in self.fixednodes}
        else:
            edges = self.ts.tables.edges
            fixed_nodes = np.array(list(self.fixednodes))
            keys = np.unique(
                np.core.records.fromarrays(
                    (self.mut_edges, edges.right - edges.left), names='muts,span')[
                        np.logical_not(np.isin(edges.child, fixed_nodes))])
            if unique_method == 1:
                self.unfixed_likelihood_cache = dict.fromkeys({tuple(t) for t in keys})
            else:
                self.unfixed_likelihood_cache = {tuple(t): None for t in keys}

        if num_threads:
            f = functools.partial(  # Set constant values for params for static _lik
                self._lik_wrapper, dt=self.timediff_lower_tri, theta=self.theta,
                normalize=self.normalize)
            if num_threads == 1:
                # Useful for testing
                for key in tqdm(self.unfixed_likelihood_cache.keys(),
                                disable=not self.progress,
                                desc="Precalculating Likelihoods"):
                    returned_key, likelihoods = f(key)
                    self.unfixed_likelihood_cache[returned_key] = likelihoods
            else:
                with tqdm(total=len(self.unfixed_likelihood_cache.keys()),
                          disable=not self.progress,
                          desc="Precalculating Likelihoods") as prog_bar:
                    with multiprocessing.Pool(processes=num_threads) as pool:
                        for key, pmf in pool.imap_unordered(
                                f, self.unfixed_likelihood_cache.keys()):
                            self.unfixed_likelihood_cache[key] = pmf
                            prog_bar.update()
        else:
            for muts, span in tqdm(self.unfixed_likelihood_cache.keys(),
                                   disable=not self.progress,
                                   desc="Precalculating Likelihoods"):
                self.unfixed_likelihood_cache[muts, span] = self._lik(
                    muts, span, dt=self.timediff_lower_tri, theta=self.theta,
                    normalize=self.normalize)

    def get_mut_lik_fixed_node(self, edge):
        """
        Get the mutation likelihoods for an edge whose child is at a
        fixed time, but whose parent may take any of the time slices in the timepoints
        that are equal to or older than the child age. This is not cached, as it is
        likely to be unique for each edge
        """
        assert edge.child in self.fixednodes, \
            "Wrongly called fixed node function on non-fixed node"
        assert self.theta is not None, \
            "Cannot calculate mutation likelihoods with no theta set"

        mutations_on_edge = self.mut_edges[edge.id]
        child_time = self.ts.node(edge.child).time
        assert child_time == 0
        # Temporary hack - we should really take a more precise likelihood
        return self._lik(mutations_on_edge, util.edge_span(edge), self.timediff,
                         self.theta, normalize=self.normalize)

    def get_mut_lik_lower_tri(self, edge):
        """
        Get the cached mutation likelihoods for an edge with non-fixed parent and child
        nodes, returning values for all the possible time differences between timepoints
        These values are returned as a flattened lower triangular matrix, the
        form required in the inside algorithm.

        """
        # Debugging asserts - should probably remove eventually
        assert edge.child not in self.fixednodes, \
            "Wrongly called lower_tri function on fixed node"
        assert hasattr(self, "unfixed_likelihood_cache"), \
            "Must call `precalculate_mutation_likelihoods()` before getting likelihoods"

        mutations_on_edge = self.mut_edges[edge.id]
        return self.unfixed_likelihood_cache[mutations_on_edge, util.edge_span(edge)]

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

    def reduce(self, lik_1, lik_2, div_0_null=False):
        """
        In linear space, this divides lik_1 by lik_2
        If div_0_null==True, then 0/0 is set to the null_constant

        NB: "reduce" is not a very good name for the function: can we think of
        something better that will also be meaningful in log space?
        """
        with np.errstate(divide='ignore', invalid='ignore'):
            ret = lik_1/lik_2
        if div_0_null:
            ret[np.isnan(ret)] = self.null_constant
        return ret

    def _recombination_lik(self, edge, fixed=True):
        # Needs to return a lower tri *or* flattened array depending on `fixed`
        raise NotImplementedError(
            "Using the recombination clock is not currently supported"
            ". See https://github.com/awohns/tsdate/issues/5 for details")
        # return (
        #     np.power(prev_state, self.n_breaks(edge)) *
        #     np.exp(-(prev_state * self.rho * edge.span * 2)))

    def get_inside(self, arr, edge):
        liks = self.identity_constant
        if self.rho is not None:
            liks = self._recombination_lik(edge)
        if self.theta is not None:
            liks *= self.get_mut_lik_lower_tri(edge)
        return self.rowsum_lower_tri(arr * liks)

    def get_outside(self, arr, edge):
        liks = self.identity_constant
        if self.rho is not None:
            liks = self._recombination_lik(edge)
        if self.theta is not None:
            liks *= self.get_mut_lik_upper_tri(edge)
        return self.rowsum_upper_tri(arr * liks)

    def get_fixed(self, arr, edge):
        liks = self.identity_constant
        if self.rho is not None:
            liks = self._recombination_lik(edge, fixed=True)
        if self.theta is not None:
            liks *= self.get_mut_lik_fixed_node(edge)
        return arr * liks

    def scale_geometric(self, fraction, value):
        return value ** fraction


class LogLikelihoods(Likelihoods):
    """
    Identical to the Likelihoods class but stores and returns log likelihoods
    """
    probability_space = base.LOG
    identity_constant = 0.0
    null_constant = -np.inf

    @staticmethod
    @numba.jit(nopython=True)
    def logsumexp(X):
        r = 0.0
        for x in X:
            r += np.exp(x)
        return np.log(r)

    @staticmethod
    def _lik(muts, span, dt, theta, normalize=True):
        """
        The likelihood of an edge given a number of mutations, as set of time deltas (dt)
        and a span. This is a static function to allow parallelization
        """
        ll = scipy.stats.poisson.logpmf(muts, dt * theta / 2 * span)
        if normalize:
            return ll - np.max(ll)
        else:
            return ll

    @staticmethod
    def _lik_wrapper(muts_span, dt, theta, normalize=True):
        """
        Needs redefining to refer to the LogLikelihoods class
        """
        return muts_span, LogLikelihoods._lik(
            muts_span[0], muts_span[1], dt, theta, normalize=normalize)

    def rowsum_lower_tri(self, input_array):
        """
        The function below is equivalent to (but numba makes it faster than)
        np.logaddexp.reduceat(input_array, self.row_indices[0])
        """
        assert len(input_array) == self.tri_size
        res = list()
        i_start = self.row_indices[0][0]
        for cur_index, i in enumerate(self.row_indices[0][1:]):
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
        for cur_index, i in enumerate(self.col_indices[1:]):
            res.append(self.logsumexp(input_array[i_start:i]))
            i_start = i
        res.append(self.logsumexp(input_array[i:]))
        return np.array(res)

    def _recombination_loglik(self, edge, fixed=True):
        # Needs to return a lower tri *or* flattened array depending on `fixed`
        raise NotImplementedError(
            "Using the recombination clock is not currently supported"
            ". See https://github.com/awohns/tsdate/issues/5 for details")
        # return (
        #     np.power(prev_state, self.n_breaks(edge)) *
        #     np.exp(-(prev_state * self.rho * edge.span * 2)))

    def combine(self, loglik_1, loglik_2):
        return loglik_1 + loglik_2

    def reduce(self, loglik_1, loglik_2, div_0_null=False):
        """
        In log space, loglik_1 - loglik_2
        If div_0_null==True, then if either is -inf it returns -inf (the null_constant)
        """
        with np.errstate(divide='ignore', invalid='ignore'):
            ret = loglik_1 - loglik_2
        if div_0_null:
            ret[np.isnan(ret)] = self.null_constant
        return ret

    def get_inside(self, arr, edge):
        log_liks = self.identity_constant
        if self.rho is not None:
            log_liks = self._recombination_loglik(edge)
        if self.theta is not None:
            log_liks += self.get_mut_lik_lower_tri(edge)
        return self.rowsum_lower_tri(arr + log_liks)

    def get_outside(self, arr, edge):
        log_liks = self.identity_constant
        if self.rho is not None:
            log_liks = self._recombination_loglik(edge)
        if self.theta is not None:
            log_liks += self.get_mut_lik_upper_tri(edge)
        return self.rowsum_upper_tri(arr + log_liks)

    def get_fixed(self, arr, edge):
        log_liks = self.identity_constant
        if self.rho is not None:
            log_liks = self._recombination_loglik(edge, fixed=True)
        if self.theta is not None:
            log_liks += self.get_mut_lik_fixed_node(edge)
        return arr + log_liks

    def scale_geometric(self, fraction, value):
        return fraction * value


class LogLikelihoodsStreaming(LogLikelihoods):
    """
    Identical to the LogLikelihoods class but uses an alternative to logsumexp,
    useful for large grid sizes, see
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


class InOutAlgorithms:
    """
    Contains the inside and outside algorithms
    """
    def __init__(self, priors, lik, *, progress=False):
        if (lik.fixednodes.intersection(priors.nonfixed_nodes) or
                len(lik.fixednodes) + len(priors.nonfixed_nodes) != lik.ts.num_nodes):
            raise ValueError(
                "The prior and likelihood objects disagree on which nodes are fixed")
        if not np.allclose(lik.timepoints, priors.timepoints):
            raise ValueError(
                "The prior and likelihood objects disagree on the timepoints used")

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
            weights=self.ts.tables.edges.right - self.ts.tables.edges.left)
        self.spans = np.pad(self.spans, (0, self.ts.num_nodes-len(self.spans)))

        self.root_spans = defaultdict(float)
        for tree in self.ts.trees():
            root = util.get_single_root(tree)
            if root is not None:
                self.root_spans[root] += tree.span  # Count span if we have a single tree
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
        return itertools.groupby(self.ts.edges(), operator.attrgetter('parent'))

    def edges_by_child_desc(self):
        """
        Return an itertools.groupby object of edges grouped by child in descending order
        of the time of the child.
        """
        wtype = np.dtype([('Childage', 'f4'), ('Childnode', 'f4')])
        w = np.empty(
            len(self.ts.tables.nodes.time[self.ts.tables.edges.child[:]]), dtype=wtype)
        w['Childage'] = self.ts.tables.nodes.time[self.ts.tables.edges.child[:]]
        w['Childnode'] = self.ts.tables.edges.child[:]
        sorted_child_parent = (self.ts.edge(i) for i in reversed(
            np.argsort(w, order=('Childage', 'Childnode'))))
        return itertools.groupby(sorted_child_parent, operator.attrgetter('child'))

    def edges_by_child_then_parent_desc(self):
        """
        Return an itertools.groupby object of edges grouped by child in descending order
        of the time of the child, then by descending order of age of child
        """
        wtype = np.dtype([('Childage', 'f4'), ('Childnode', 'f4'), ('Parentage', 'f4')])
        w = np.empty(
            len(self.ts.tables.nodes.time[self.ts.tables.edges.child[:]]), dtype=wtype)
        w['Childage'] = self.ts.tables.nodes.time[self.ts.tables.edges.child[:]]
        w['Childnode'] = self.ts.tables.edges.child[:]
        w['Parentage'] = -self.ts.tables.nodes.time[self.ts.tables.edges.parent[:]]
        sorted_child_parent = (self.ts.edge(i) for i in reversed(
            np.argsort(w, order=('Childage', 'Childnode', 'Parentage'))))
        return itertools.groupby(sorted_child_parent, operator.attrgetter('child'))

    # === MAIN ALGORITHMS ===

    def inside_pass(self, *, normalize=True, cache_inside=False, progress=None):
        """
        Use dynamic programming to find approximate posterior to sample from
        """
        if progress is None:
            progress = self.progress
        inside = self.priors.clone_with_new_data(  # store inside matrix values
            grid_data=np.nan, fixed_data=self.lik.identity_constant)
        if cache_inside:
            g_i = np.full(
                (self.ts.num_edges, self.lik.grid_size), self.lik.identity_constant)
        norm = np.full(self.ts.num_nodes, np.nan)
        # Iterate through the nodes via groupby on parent node
        for parent, edges in tqdm(
                self.edges_by_parent_asc(), desc="Inside",
                total=inside.num_nonfixed, disable=not progress):
            """
            for each node, find the conditional prob of age at every time
            in time grid
            """
            if parent in self.fixednodes:
                continue  # there is no hidden state for this parent - it's fixed
            val = self.priors[parent].copy()
            for edge in edges:
                spanfrac = util.edge_span(edge) / self.spans[edge.child]
                # Calculate vals for each edge
                if edge.child in self.fixednodes:
                    # NB: geometric scaling works exactly when all nodes fixed in graph
                    # but is an approximation when times are unknown.
                    daughter_val = self.lik.scale_geometric(spanfrac, inside[edge.child])
                    edge_lik = self.lik.get_fixed(daughter_val, edge)
                else:
                    inside_values = inside[edge.child]
                    if np.ndim(inside_values) == 0 or np.all(np.isnan(inside_values)):
                        # Child appears fixed, or we have not visited it. Either our
                        # edge order is wrong (bug) or we have hit a dangling node
                        raise ValueError("The input tree sequence includes "
                                         "dangling nodes: please simplify it")
                    daughter_val = self.lik.scale_geometric(
                        spanfrac, self.lik.make_lower_tri(inside[edge.child]))
                    edge_lik = self.lik.get_inside(daughter_val, edge)
                val = self.lik.combine(val, edge_lik)
                if cache_inside:
                    g_i[edge.id] = edge_lik
            norm[parent] = np.max(val) if normalize else 1
            inside[parent] = self.lik.reduce(val, norm[parent])
        if cache_inside:
            self.g_i = self.lik.reduce(g_i, norm[self.ts.tables.edges.child, None])
        # Keep the results in this object
        self.inside = inside
        self.norm = norm

    def outside_pass(self, *, normalize=False, progress=None,
                     probability_space_returned=base.LIN):
        """
        Computes the full posterior distribution on nodes.
        Input is population scaled mutation and recombination rates.

        Normalising may be necessary if there is overflow, but means that we cannot
        check the total functional value at each node

        The rows in the posterior returned correspond to node IDs as given by
        self.nodes
        """
        if progress is None:
            progress = self.progress
        if not hasattr(self, "inside"):
            raise RuntimeError("You have not yet run the inside algorithm")

        outside = self.inside.clone_with_new_data(
            grid_data=0, probability_space=base.LIN)
        for root, span_when_root in self.root_spans.items():
            outside[root] = span_when_root / self.spans[root]
        outside.force_probability_space(self.inside.probability_space)

        for child, edges in tqdm(
                self.edges_by_child_desc(), desc="Outside",
                total=len(np.unique(self.ts.tables.edges.child)),
                disable=not progress):
            if child in self.fixednodes:
                continue
            val = np.full(self.lik.grid_size, self.lik.identity_constant)
            for edge in edges:
                if edge.parent in self.fixednodes:
                    raise RuntimeError(
                        "Fixed nodes cannot currently be parents in the TS")
                # Geometric scaling works exactly for all nodes fixed in graph
                # but is an approximation when times are unknown.
                spanfrac = util.edge_span(edge) / self.spans[child]
                try:
                    inside_div_gi = self.lik.reduce(
                        self.inside[edge.parent], self.g_i[edge.id], div_0_null=True)
                except AttributeError:  # we haven't cached g_i so we recalculate
                    daughter_val = self.lik.scale_geometric(
                        spanfrac, self.lik.make_lower_tri(self.inside[edge.child]))
                    edge_lik = self.lik.get_inside(daughter_val, edge)
                    cur_g_i = self.lik.reduce(edge_lik, self.norm[child])
                    inside_div_gi = self.lik.reduce(
                        self.inside[edge.parent], cur_g_i, div_0_null=True)
                parent_val = self.lik.scale_geometric(
                    spanfrac,
                    self.lik.make_upper_tri(
                        self.lik.combine(outside[edge.parent], inside_div_gi)))
                edge_lik = self.lik.get_outside(parent_val, edge)
                val = self.lik.combine(val, edge_lik)

            # vv[0] = 0  # Seems a hack: internal nodes should be allowed at time 0
            assert self.norm[edge.child] > self.lik.null_constant
            outside[child] = self.lik.reduce(val, self.norm[child])
            if normalize:
                outside[child] = self.lik.reduce(val, np.max(val))
        posterior = outside.clone_with_new_data(
           grid_data=self.lik.combine(self.inside.grid_data, outside.grid_data),
           fixed_data=np.nan)  # We should never use the posterior for a fixed node
        posterior.normalize()
        posterior.force_probability_space(probability_space_returned)
        self.outside = outside
        return posterior

    def outside_maximization(self, *, eps=1e-6, progress=None):
        if progress is None:
            progress = self.progress
        if not hasattr(self, "inside"):
            raise RuntimeError("You have not yet run the inside algorithm")

        maximized_node_times = np.zeros(self.ts.num_nodes, dtype='int')

        if self.lik.probability_space == base.LOG:
            poisson = scipy.stats.poisson.logpmf
        elif self.lik.probability_space == base.LIN:
            poisson = scipy.stats.poisson.pmf

        mut_edges = self.lik.mut_edges
        mrcas = np.where(np.isin(
            np.arange(self.ts.num_nodes), self.ts.tables.edges.child, invert=True))[0]
        for i in mrcas:
            if i not in self.fixednodes:
                maximized_node_times[i] = np.argmax(self.inside[i])

        for child, edges in tqdm(
                self.edges_by_child_then_parent_desc(), desc="Maximization",
                total=len(np.unique(self.ts.tables.edges.child)),
                disable=not progress):
            if child in self.fixednodes:
                continue
            for edge_index, edge in enumerate(edges):
                if edge_index == 0:
                    youngest_par_index = maximized_node_times[edge.parent]
                    parent_time = self.lik.timepoints[maximized_node_times[edge.parent]]
                    ll_mut = poisson(
                        mut_edges[edge.id],
                        (parent_time - self.lik.timepoints[:youngest_par_index + 1] +
                            eps) * self.lik.theta / 2 * util.edge_span(edge))
                    result = self.lik.reduce(ll_mut, np.max(ll_mut))
                else:
                    cur_parent_index = maximized_node_times[edge.parent]
                    if cur_parent_index < youngest_par_index:
                        youngest_par_index = cur_parent_index
                    parent_time = self.lik.timepoints[maximized_node_times[edge.parent]]
                    ll_mut = poisson(
                        mut_edges[edge.id],
                        (parent_time - self.lik.timepoints[:youngest_par_index + 1] +
                            eps) * self.lik.theta / 2 * util.edge_span(edge))
                    result[:youngest_par_index + 1] = self.lik.combine(
                        self.lik.reduce(ll_mut[:youngest_par_index + 1],
                                        np.max(ll_mut[:youngest_par_index + 1])),
                        result[:youngest_par_index + 1])
            inside_val = self.inside[child][:(youngest_par_index + 1)]

            maximized_node_times[child] = np.argmax(self.lik.combine(
                result[:youngest_par_index + 1], inside_val))

        return self.lik.timepoints[np.array(maximized_node_times).astype('int')]


def posterior_mean_var(ts, timepoints, posterior, *, fixed_node_set=None):
    """
    Mean and variance of node age in scaled time. Fixed nodes will be given a mean
    of their exact time in the tree sequence, and zero variance (as long as they are
    identified by the fixed_node_set
    If fixed_node_set is None, we attempt to date all the non-sample nodes
    """
    mn_post = np.full(ts.num_nodes, np.nan)  # Fill with NaNs so we detect when there's
    vr_post = np.full(ts.num_nodes, np.nan)  # been an error

    fixed_nodes = np.array(list(fixed_node_set))
    mn_post[fixed_nodes] = ts.tables.nodes.time[fixed_nodes]
    vr_post[fixed_nodes] = 0

    for row, node_id in zip(posterior.grid_data, posterior.nonfixed_nodes):
        mn_post[node_id] = np.sum(row * timepoints) / np.sum(row)
        vr_post[node_id] = (np.sum(row * timepoints ** 2) / np.sum(row) -
                            mn_post[node_id] ** 2)
    return mn_post, vr_post


def constrain_ages_topo(ts, post_mn, timepoints, eps, nodes_to_date=None,
                        progress=False):
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
    nd_children = tables.edges.child
    for nd in tqdm(sorted(nodes_to_date), desc="Constrain Ages",
                   disable=not progress):
        children = nd_children[parents == nd]
        time = new_mn_post[children]
        if np.any(new_mn_post[nd] <= time):
            # closest_time = (np.abs(grid - max(time))).argmin()
            # new_mn_post[nd] = grid[closest_time] + eps
            new_mn_post[nd] = np.max(time) + eps
    return new_mn_post


def date(
        tree_sequence, Ne, mutation_rate=None, recombination_rate=None, priors=None, *,
        progress=False, **kwargs):
    """
    Take a tree sequence with arbitrary node times and recalculate node times using
    the `tsdate` algorithm. If a mutation_rate is given, the mutation clock is used. The
    recombination clock is unsupported at this time. If neither a mutation_rate nor a
    recombination_rate is given, a topology-only clock is used.

    :param TreeSequence tree_sequence: The input :class:`tskit.TreeSequence`, treated as
        undated.
    :param float Ne: The estimated (diploid) effective population size: must be
        specified.
    :param float mutation_rate: The estimated mutation rate per unit of genome per
        generation. If provided, the dating algorithm will use a mutation rate clock to
        help estimate node dates.
    :param float recombination_rate: The estimated recombination rate per unit of genome
        per generation. If provided, the dating algorithm will use a recombination rate
        clock to help estimate node dates.
    :param NodeGridValues priors: NodeGridValue object containing the prior probabilities
        for each node at a set of discrete time points
    :param float eps: Specify minimum distance separating time points. Also specifies
        the error factor in time difference calculations.
    :param int num_threads: The number of threads to use. A simpler unthreaded algorithm
        is used unless this is >= 1 (default: None).
    :param string method: What estimation method to use: can be
        "inside_outside" (empirically better, theoretically problematic) or
        "maximization" (worse empirically, especially with gamma approximated priors,
        but theoretically robust). Default: "inside_outside".
    :param bool probability_space: Should the internal algorithm save probabilities in
        "logarithmic" (slower, less liable to to overflow) or "linear" space (fast, may
        overflow). Default: "logarithmic"
    :param bool progress: Whether to display a progress bar.
    :return: A tree sequence with inferred node times in units of generations.
    :rtype: tskit.TreeSequence
    """
    dates, _, timepoints, eps, nds = get_dates(
        tree_sequence, Ne, mutation_rate, recombination_rate, priors, progress=progress,
        **kwargs)
    constrained = constrain_ages_topo(tree_sequence, dates, timepoints, eps, nds,
                                      progress)
    tables = tree_sequence.dump_tables()
    tables.nodes.time = constrained * 2 * Ne
    tables.sort()
    return tables.tree_sequence()


def get_dates(
        tree_sequence, Ne, mutation_rate=None, recombination_rate=None, priors=None, *,
        eps=1e-6, num_threads=None, method='inside_outside', outside_normalize=True,
        progress=False, cache_inside=False,
        probability_space=base.LOG):
    """
    Infer dates for the nodes in a tree sequence, returning an array of inferred dates
    for nodes, plus other variables such as the distribution of posterior probabilities
    etc. Parameters are identical to the date() method, which calls this method, then
    injects the resulting date estimates into the tree sequence

    :return: tuple(mn_post, posterior, timepoints, eps, nodes_to_date)
    """
    # Stuff yet to be implemented. These can be deleted once fixed
    for sample in tree_sequence.samples():
        if tree_sequence.node(sample).time != 0:
            raise NotImplementedError(
                "Samples must all be at time 0")
    fixed_nodes = set(tree_sequence.samples())

    # Default to not creating approximate priors unless ts has > 1000 samples
    approx_priors = False
    if tree_sequence.num_samples > 1000:
        approx_priors = True

    if priors is None:
        priors = prior.build_grid(tree_sequence, eps=eps, progress=progress,
                                  approximate_priors=approx_priors)
    else:
        logging.info("Using user-specified priors")
        priors = priors

    theta = rho = None

    if mutation_rate is not None:
        theta = 4 * Ne * mutation_rate
    if recombination_rate is not None:
        rho = 4 * Ne * recombination_rate

    if probability_space != base.LOG:
        liklhd = Likelihoods(tree_sequence, priors.timepoints, theta, rho,
                             eps=eps, fixed_node_set=fixed_nodes, progress=progress)
    else:
        liklhd = LogLikelihoods(tree_sequence, priors.timepoints, theta, rho,
                                eps=eps, fixed_node_set=fixed_nodes, progress=progress)

    if theta is not None:
        liklhd.precalculate_mutation_likelihoods(num_threads=num_threads)

    dynamic_prog = InOutAlgorithms(priors, liklhd, progress=progress)
    dynamic_prog.inside_pass(cache_inside=False)

    posterior = None
    if method == 'inside_outside':
        posterior = dynamic_prog.outside_pass(normalize=outside_normalize)
        mn_post, _ = posterior_mean_var(tree_sequence, priors.timepoints, posterior,
                                        fixed_node_set=fixed_nodes)
    elif method == 'maximization':
        if theta is not None:
            mn_post = dynamic_prog.outside_maximization(eps=eps)
        else:
            raise ValueError("Outside maximization method requires mutation rate")
    else:
        raise ValueError(
            "estimation method must be either 'inside_outside' or 'maximization'")

    return mn_post, posterior, priors.timepoints, eps, priors.nonfixed_nodes
