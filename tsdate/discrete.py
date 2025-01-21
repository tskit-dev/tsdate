# Classes used for discete time algorithms (inside-outside and outsize_maximization)
# Note that these methods are no longer the default method used by tsdate
import functools
import itertools
import multiprocessing
import operator
from collections import defaultdict

import numpy as np
import scipy.stats
import tskit
from tqdm.auto import tqdm

from .accelerate import numba_jit
from .node_time_class import LIN_GRID, LOG_GRID


class Likelihoods:
    """
    A class to store and process likelihoods. Likelihoods for edges are stored as a
    flattened lower triangular matrix of all the possible delta t's. This class also
    provides methods for accessing this lower triangular matrix, multiplying it, etc.

    If ``standardize`` is true, routines will operate to standardize the likelihoods
    such that their maximum is one (in linear space) or zero (in log space)
    """

    probability_space = LIN_GRID
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
        self.fixednodes = set(ts.samples()) if fixed_node_set is None else fixed_node_set
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
        for t in range(self.grid_size):
            n = np.arange(self.grid_size)
            self.row_indices.append((((n * (n + 1)) // 2) + t)[t:])
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
            liks *= self._recombination_lik(edge)
        if self.mut_rate is not None:
            liks *= self.get_mut_lik_lower_tri(edge)
        return self.rowsum_lower_tri(arr * liks)

    def get_outside(self, arr, edge):
        liks = self.identity_constant
        if self.rec_rate is not None:
            liks *= self._recombination_lik(edge)
        if self.mut_rate is not None:
            liks *= self.get_mut_lik_upper_tri(edge)
        return self.rowsum_upper_tri(arr * liks)

    def get_fixed(self, arr, edge):
        liks = self.identity_constant
        if self.rec_rate is not None:
            liks *= self._recombination_lik(edge, fixed=True)
        if self.mut_rate is not None:
            liks *= self.get_mut_lik_fixed_node(edge)
        return arr * liks

    def scale_geometric(self, fraction, value):
        return value**fraction


class LogLikelihoods(Likelihoods):
    """
    Identical to the Likelihoods class but stores and returns log likelihoods
    """

    probability_space = LOG_GRID
    identity_constant = 0.0
    null_constant = -np.inf

    """
    Uses an alternative to logsumexp, useful for large grid sizes, see
    http://www.nowozin.net/sebastian/blog/streaming-log-sum-exp-computation.html
    """

    @staticmethod
    @numba_jit
    def logsumexp(X):
        alpha = -np.inf
        r = 0.0
        for x in X:
            if x != -np.inf:
                if x <= alpha:
                    r += np.exp(x - alpha)
                else:
                    r *= np.exp(alpha - x)
                    r += 1.0
                    alpha = x
        return -np.inf if r == 0 else np.log(r) + alpha

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
            log_liks += self._recombination_loglik(edge)
        if self.mut_rate is not None:
            log_liks += self.get_mut_lik_lower_tri(edge)
        return self.rowsum_lower_tri(arr + log_liks)

    def get_outside(self, arr, edge):
        log_liks = self.identity_constant
        if self.rec_rate is not None:
            log_liks += self._recombination_loglik(edge)
        if self.mut_rate is not None:
            log_liks += self.get_mut_lik_upper_tri(edge)
        return self.rowsum_upper_tri(arr + log_liks)

    def get_fixed(self, arr, edge):
        log_liks = self.identity_constant
        if self.rec_rate is not None:
            log_liks += self._recombination_loglik(edge, fixed=True)
        if self.mut_rate is not None:
            log_liks += self.get_mut_lik_fixed_node(edge)
        return arr + log_liks

    def scale_geometric(self, fraction, value):
        return fraction * value


class BeliefPropagation:
    """
    The class that encapsulates running exact belief propagation models,
    in particular the discrete-time inside and outside algorithms.
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
        # Return an itertools.groupby object of edges grouped by parent in ascending
        # order of the time of the parent. As tree sequence properties guarantee that
        # edges are listed in nondecreasing order of parent time
        # (https://tskit.readthedocs.io/en/latest/data-model.html#edge-requirements)
        # we can simply use the standard edge order
        if grouped:
            return itertools.groupby(self.ts.edges(), operator.attrgetter("parent"))
        else:
            return self.ts.edges()

    def edges_by_child_desc(self, grouped=True):
        # Return an itertools.groupby object of edges grouped by child in descending
        # order of the time of the child.
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
        # Return an itertools.groupby object of edges grouped by child in descending
        # order of the time of the child, then by descending order of age of child
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
        # Use dynamic programming to find approximate posterior to sample from
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
                    daughter_val = self.lik.scale_geometric(spanfrac, inside[edge.child])
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
            marginal_lik = self.lik.combine(marginal_lik, self.lik.marginalize(root_val))
        return marginal_lik

    def outside_pass(
        self,
        *,
        standardize=False,
        ignore_oldest_root=False,
        progress=None,
    ):
        # Computes the full posterior distribution on nodes, returning the
        # posterior values. These are *not* probabilities, as they do not sum to one:
        # to convert to probabilities, call posterior.to_probabilities()
        #
        # Standardizing *during* the outside process may be necessary if there is
        # overflow, but means that we cannot check the total functional value at each node
        #
        # Ignoring the oldest root may also be necessary when the oldest root node
        # causes numerical stability issues.
        if progress is None:
            progress = self.progress
        if not hasattr(self, "inside"):
            raise RuntimeError("You have not yet run the inside algorithm")

        outside = self.inside.clone_with_new_data(grid_data=0, probability_space=LIN_GRID)
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
        self.outside = outside  # useful to access for testing purposes
        self.posterior_grid = outside.clone_with_new_data(
            grid_data=self.lik.combine(self.inside.grid_data, outside.grid_data),
            fixed_data=np.nan,
        )  # NB: we should never use the posterior for a fixed node

    def node_posteriors(self):
        """
        Return the distribution of posterior node times as a structured array. The
        returned value can be e.g. read into ``pandas.DataFrame`` for further analysis.

        .. note::
            The ``outside_maximization`` method does not provide node time posteriors.

        :return: The distribution of posterior node times as a structured array with
            columns as timepoints. Row ``i`` corresponds to the probabilities of
            node ``i`` lying at each timepoint. Nodes with fixed times are set to
            ``np.nan`` for the entire row.
        :rtype: numpy.ndarray
        """
        try:
            return self.posterior_grid.node_probability_array()
        except AttributeError:
            raise ValueError(
                "Cannot get posteriors without running the outside algorithm"
            ) from None

    def outside_maximization(self, *, eps, progress=None):
        if progress is None:
            progress = self.progress
        if not hasattr(self, "inside"):
            raise RuntimeError("You have not yet run the inside algorithm")

        maximized_node_times = np.zeros(self.ts.num_nodes, dtype="int")

        if self.lik.probability_space == LOG_GRID:
            poisson = scipy.stats.poisson.logpmf
        elif self.lik.probability_space == LIN_GRID:
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
        # The outside_maximization method does not provide a full posterior but
        # simply the means of the posterior distributions
        self.posterior_mean = self.lik.timepoints[
            np.array(maximized_node_times).astype("int")
        ]
