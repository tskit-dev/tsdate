# MIT License
#
# Copyright (c) 2020 University of Oxford
# Copyright (c) 2021-2023 Tskit Developers
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
Routines and classes for creating priors and timeslices for use in tsdate
"""
import logging
import os
from collections import defaultdict
from collections import namedtuple

import numba
import numpy as np
import scipy.cluster
import scipy.special
import scipy.stats
import tskit
from tqdm.auto import tqdm

from . import base
from . import cache
from . import demography
from . import provenance
from . import util


class PriorParams(namedtuple("PriorParamsBase", "alpha, beta, mean, var")):
    @classmethod
    def field_index(cls, fieldname):
        return np.where([f == fieldname for f in cls._fields])[0][0]


def lognorm_approx(mean, var):
    """
    alpha is mean of underlying normal distribution
    beta is variance of underlying normal distribution
    """
    beta = np.log(var / (mean**2) + 1)
    alpha = np.log(mean) - 0.5 * beta
    return alpha, beta


def gamma_approx(mean, variance):
    """
    Returns alpha and beta of a gamma distribution for a given mean and variance
    """

    return (mean**2) / variance, mean / variance


@numba.njit("float64[:, :](float64[:, :])")
def _marginalize_over_ancestors(val):
    """
    Integrate an expectation over counts of extant ancestors. In a tree with
    "n" tips, the probability that there are "a" extent ancestors when a
    subtree of size "k" coalesces is hypergeometric-ish (Wuif & Donnelly 1998),
    and may be calculated recursively over increasing "a" and decreasing "k"
    (e.g. using recursive relationships for binomial coefficients).
    """
    n, N = val.shape  # number of tips, number of moments
    pr_a_ln = [np.nan, np.nan, 0.0]  # log Pr(a | k, n)
    out = np.zeros((n + 1, N))
    for k in range(n - 1, 1, -1):
        const = np.log(n - k) + np.log(k - 2) - np.log(k + 1)
        for a in range(2, n - k + 2):
            out[k] += np.exp(pr_a_ln[a]) * val[a]
            if k > 2:  # Pr(a | k, n) to Pr(a | k - 1, n)
                pr_a_ln[a] += const - np.log(n - a - k + 2)
        if k > 2:  # Pr(n - k + 1 | k - 1, n) to Pr(n - k + 2 | k - 1, n)
            pr_a_ln.append(pr_a_ln[-1] + np.log(n - k + 2) - np.log(k + 1) - const)
    out[n] = val[1]
    return out


@numba.njit("float64[:](uint64)")
def conditional_coalescent_variance(num_tips):
    """
    Variance of node age conditional on the number of descendant leaves, under
    the standard coalescent. Returns array indexed by number of descendant
    leaves.
    """

    coal_rates = np.array(
        [2 / (i * (i - 1)) if i > 1 else 0.0 for i in range(1, num_tips + 1)]
    )

    # hypoexponential mean and variance; e.g. conditional on the number of
    # extant ancestors when the node coalesces, the expected time of
    # coalescence is the sum of exponential RVs (Wuif and Donnelly 1998)
    mean = coal_rates.copy()
    variance = coal_rates.copy() ** 2
    for i in range(coal_rates.size - 2, 0, -1):
        mean[i] += mean[i + 1]
        variance[i] += variance[i + 1]

    # marginalize over number of ancestors using recursive algorithm
    moments = _marginalize_over_ancestors(np.stack((mean, variance + mean**2), 1))

    return moments[:, 1] - moments[:, 0] ** 2


class ConditionalCoalescentTimes:
    """
    Make and store conditional coalescent priors for different numbers of total samples
    """

    def __init__(
        self,
        precalc_approximation_n,
        prior_distr="lognorm",
        progress=False,
    ):
        """
        :param bool precalc_approximation_n: the size of tree used for
            approximate prior (larger numbers give a better approximation).
            If 0 or otherwise falsey, do not precalculate,
            and therefore do not allow approximate priors to be used
        """
        self.n_approx = precalc_approximation_n
        self.prior_store = {}
        self.progress = progress
        self.mean_column = PriorParams.field_index("mean")
        self.var_column = PriorParams.field_index("var")

        if precalc_approximation_n:
            # Create lookup table based on a large n that can be used for n > ~50
            filename = self.get_precalc_cache(precalc_approximation_n)
            if os.path.isfile(filename):
                # Have already calculated and stored this
                self.approx_priors = np.genfromtxt(filename)
            else:
                # Calc and store
                self.approx_priors = self.precalculate_priors_for_approximation(
                    precalc_approximation_n,
                )
        else:
            self.approx_priors = None

        self.prior_distr = prior_distr
        if prior_distr == "lognorm":
            self.func_approx = lognorm_approx
        elif prior_distr == "gamma":
            self.func_approx = gamma_approx
        else:
            raise ValueError("prior distribution must be lognorm or gamma")

    def __getitem__(self, total_tips):
        """
        Return a numpy array of conditional coalescent prior parameters plus mean and
        var (row N indicates the parameters for a N descendant tips) for a given
        number of total tips in the tree
        """
        return self.prior_store[total_tips]

    def __str__(self):
        s = [f"Conditional coalescent params under the {self.prior_distr} distibution:"]
        for t, priors in self.prior_store.items():
            indent = len(str(t))
            s.append(f"{t} total tips  ({list(PriorParams._fields)})")
            for i, prior_params in enumerate(priors):
                s.append(f" {i:>{indent}} descendants {prior_params}")
        return "\n".join(s)

    def prior_with_max_total_tips(self):
        return self.prior_store.get(max(self.prior_store.keys()))

    def add(self, total_tips, approximate=None):
        """
        Create and store a numpy array used to lookup prior params and mean + variance
        of ages for nodes with descendant sample tips range from 2..``total_tips``
        given that the total number of tips in the coalescent tree is
        ``total_tips``. The array is indexed by (num_tips / total_tips).
        """
        if total_tips in self.prior_store:
            return  # Already calculated for this number of total tips
        if approximate is not None:
            self.approximate = approximate
        else:
            if total_tips >= base.DEFAULT_APPROX_PRIOR_SIZE:
                self.approximate = True
            else:
                self.approximate = False

        if self.approximate and self.approx_priors is None:
            raise RuntimeError(
                "You cannot add an approximate prior unless you initialize"
                " the ConditionalCoalescentTimes object with a non-zero number"
            )

        if not self.approximate and total_tips >= base.DEFAULT_APPROX_PRIOR_SIZE:
            logging.warning(
                "Calculating exact priors for more than "
                f"{base.DEFAULT_APPROX_PRIOR_SIZE} tips. Consider "
                "setting `approximate=True` for a faster calculation."
            )

        # alpha/beta and mean/var are simply transformations of one another
        # for the gamma, mean = alpha / beta and var = alpha / (beta **2)
        # for the lognormal, see lognorm_approx for definition
        # mean and var are used in generating the mixture prior
        # alpha and beta are used for generating prior probabilities
        # they are stored separately to obviate need to move between them
        # We should only use prior[2] upwards
        priors = np.full(
            (total_tips + 1, len(PriorParams._fields)), np.nan, dtype=base.FLOAT_DTYPE
        )

        if self.approximate:
            get_tau_var = self.tau_var_lookup
        else:
            get_tau_var = self.tau_var_exact

        all_tips = np.arange(2, total_tips + 1)
        variances = get_tau_var(total_tips, all_tips)
        # priors.loc[1] is distribution of times of a "coalescence node" ending
        # in a single sample - equivalent to the time of the sample itself, so
        # it should have var = 0 and mean = sample.time
        if self.prior_distr == "lognorm":
            # For a lognormal, alpha = -inf and beta = 0 sets mean == var == 0
            priors[1] = PriorParams(alpha=-np.inf, beta=0, mean=0, var=0)
        elif self.prior_distr == "gamma":
            # For a gamma, alpha = 0 and beta = 1 sets mean (a/b) == var (a / b^2) == 0
            priors[1] = PriorParams(alpha=0, beta=1, mean=0, var=0)
        for var, tips in zip(variances, all_tips):
            # NB: it should be possible to vectorize this in numpy
            expectation = self.tau_expect(tips, total_tips)
            alpha, beta = self.func_approx(expectation, var)
            priors[tips] = PriorParams(
                alpha=alpha, beta=beta, mean=expectation, var=var
            )
        self.prior_store[total_tips] = priors

    def precalculate_priors_for_approximation(self, precalc_approximation_n):
        n = precalc_approximation_n
        logging.warning(
            "Initialising your tsdate installation by creating a user cache of "
            "conditional coalescent prior values for {} tips".format(n)
        )
        logging.info(
            "Creating prior lookup table for a total tree of n={} tips"
            " in `{}`, this may take some time for large n".format(
                n, self.get_precalc_cache(n)
            )
        )
        # The first value should be zero tips, we don't want the 1 tip value
        prior_lookup_table = np.zeros((n, 2))
        all_tips = np.arange(2, n + 1)
        prior_lookup_table[1:, 0] = all_tips / n
        prior_lookup_table[1:, 1] = conditional_coalescent_variance(n + 1)[all_tips]
        np.savetxt(self.get_precalc_cache(n), prior_lookup_table)
        return prior_lookup_table

    def clear_precalculated_priors(self):
        if os.path.isfile(self.get_precalc_cache(self.n_approx)):
            os.remove(self.get_precalc_cache(self.n_approx))
        else:
            logging.debug(
                "Precalculated priors in `{}` not yet created, so cannot be"
                " cleared".format(self.get_precalc_cache(self.n_approx))
            )

    @staticmethod
    def get_precalc_cache(precalc_approximation_n):
        cache_dir = cache.get_cache_dir()
        return os.path.join(
            cache_dir,
            f"prior_{precalc_approximation_n}df_{provenance.__version__}.txt",
        )

    @staticmethod
    def tau_expect(i, n):
        if i == n:
            return 2 * (1 - (1 / n))
        else:
            return (i - 1) / n

    @staticmethod
    def tau_var_mrca(n):
        value = np.arange(2, n + 1)
        var = np.sum(1 / ((value**2) * ((value - 1) ** 2)))
        return np.abs(4 * var)

    # The following are not static as they may need to access self.approx_priors for this
    # instance
    def tau_var_lookup(self, total_tips, all_tips):
        """
        Lookup tau_var if approximate is True
        """
        interpolated_priors = np.interp(
            all_tips / total_tips, self.approx_priors[:, 0], self.approx_priors[:, 1]
        )

        # insertion_point = np.searchsorted(all_tips / self.total_tips,
        #    self.approx_priors[:, 0])
        # interpolated_priors = self.approx_priors[insertion_point, 1]

        # The final MRCA we calculate exactly
        interpolated_priors[all_tips == total_tips] = self.tau_var_mrca(total_tips)
        return interpolated_priors

    def tau_var_exact(self, total_tips, all_tips):
        return conditional_coalescent_variance(total_tips)[all_tips]

    def mixture_expect_and_var(self, mixture, weight_by_log_span=False):
        """
        Return the expectation and variance of a coalescent mixture
        mixture is a dict of numpy recarrays of the form
        {N: {descendant_tips: [N_tips], span: [N_spans]}}

        weight_by_log_span is a boolean that determines whether the
        weights are taken as the log of the span of the node
        (plus one to avoid log(0). Testing indicates that
        this gives a slightly better fit to the observed values
        under the coalescent with recombination.

        Note, however, that both the expected mean and (especially)
        the expected variance are substantially affected by the
        *total length* of the span rather than just the relative
        weights, which is not taken into account here

        """
        expectation = 0
        first = secnd = 0
        weight_sum = 0
        for N, tip_dict in mixture.items():
            # assert 1 not in tip_dict.descendant_tips
            mean_time = self[N][tip_dict["descendant_tips"], self.mean_column]
            var_time = self[N][tip_dict["descendant_tips"], self.var_column]
            # Add one to avoid log(0)
            w = np.log(tip_dict["span"] + 1) if weight_by_log_span else tip_dict["span"]
            # Mixture expectation
            expectation += np.sum(mean_time * w)
            # Mixture variance
            first += np.sum(var_time * w)
            secnd += np.sum(mean_time**2 * w)
            weight_sum += np.sum(w)
        mean = expectation / weight_sum
        var = (first + secnd) / weight_sum - (mean**2)
        return mean, var

    def get_mixture_prior_params(self, spans_by_samples):
        """
        Given an object that can be queried for spans by num descendant tips
        for a node, and a set of conditional coalescent priors for different
        numbers of sample tips under a node, return distribution parameters
        (shape and scale) that best fit the distribution for that node.

        :param .SpansBySamples spans_by_samples: An instance of the
            :class:`SpansBySamples` class that can be used to obtain
            spans for each node to use as weights.
        :return: A numpy array whose rows corresponds to the node id in
            ``spans_by_samples.nodes_to_date`` and whose columns are the parameter
            columns in PriorParams (i.e. not including the mean and variance)
            This can be used to approximate the probabilities of times for that
            node by matching against an appropriate distribution (e.g. gamma or lognorm)
        :rtype:  numpy.ndarray
        """

        param_cols = np.array(
            [i for i, f in enumerate(PriorParams._fields) if f not in ("mean", "var")]
        )

        seen_mixtures = {}
        # allocate space for params for all nodes, even though we only use nodes_to_date
        num_nodes, num_params = spans_by_samples.ts.num_nodes, len(param_cols)
        priors = np.full((num_nodes + 1, num_params), np.nan, dtype=base.FLOAT_DTYPE)
        for node in tqdm(
            spans_by_samples.nodes_to_date,
            total=len(spans_by_samples.nodes_to_date),
            disable=not self.progress,
            desc="Find Mixture Priors",
        ):
            mixture = spans_by_samples.get_spans(node)
            if len(mixture) == 1:
                # The norm: this node spans trees that all have the same set of samples
                total_tips, span_arr = next(iter(mixture.items()))
                if span_arr.shape[0] == 1:
                    d_tips = span_arr["descendant_tips"][0]
                    # This node is not a mixture - can use the standard coalescent prior
                    priors[node] = self[total_tips][d_tips, param_cols]
                elif span_arr.shape[0] <= 5:
                    # Making mixture priors is a little expensive. We can help by caching
                    # in those cases where we have only a few mixtures
                    # (arbitrarily set here as <= 5 mixtures)
                    mixture_hash = (total_tips, span_arr.tobytes())
                    if mixture_hash not in seen_mixtures:
                        priors[node] = seen_mixtures[mixture_hash] = self.func_approx(
                            *self.mixture_expect_and_var(mixture)
                        )
                    else:
                        priors[node] = seen_mixtures[mixture_hash]
                else:
                    # a large number of mixtures in this node - don't bother caching
                    priors[node] = self.func_approx(
                        *self.mixture_expect_and_var(mixture)
                    )
            else:
                # The node spans trees with multiple total tip numbers,
                # don't use the cache
                priors[node] = self.func_approx(*self.mixture_expect_and_var(mixture))
        # Check that references to the tskit.NULL'th node return NaNs, as we will later
        # be indexing into the prior array using a node mapping which could have NULLs
        assert np.all(np.isnan(priors[tskit.NULL, :]))
        return priors


class SpansBySamples:
    """
    A class to efficiently calculate the genomic spans covered by each
    non-sample node, broken down by the number of samples that descend
    directly from that node. This is used to calculate the conditional
    coalescent prior. The main method is :meth:`get_spans`, which
    returns the spans for each node, broken down by the number of
    samples under different regions.

    .. note:: This assumes that all edges connect to the same tree - i.e.
        there is only a single topology present at each point in the
        genome. Equivalently, it assumes that only one of the roots in
        a tree has descending edges (all other roots represent isolated
        "missing data" nodes.

    :ivar tree_sequence: A reference to the tree sequence that was used to
        generate the spans
    :vartype tree_sequence: tskit.TreeSequence
    :ivar total_fixed_at_0_counts: A numpy array of unique numbers which list,
        in no particular order, the various sample counts among the trees
        in this tree sequence. In the simplest case of a tree sequence with
        no missing data, all trees have the same count of numbers of samples,
        and there will be only a single number in this array, equal to
        :attr:`.tree_sequence.num_samples`. However, where samples contain
        :ref:`missing data <sec_data_model_missing_data>`,
        some trees will contain fewer sample nodes, so this array will also
        contain additional numbers, all of which will be less than
        :attr:`.tree_sequence.num_samples`.
    :vartype total_fixed_at_0_counts: numpy.ndarray (dtype=np.uint64)
    :ivar node_spans: A numpy array of size :attr:`.tree_sequence.num_nodes`
        containing the genomic span covered by each node (including sample nodes)
    :vartype node_spans: numpy.ndarray (dtype=np.uint64)
    :ivar nodes_to_date: An numpy array containing all the node ids in the tree
        sequence that we wish to date. These are usually all the non-sample nodes,
        and also provide the node numbers that are valid parameters for the
        :meth:`get_spans` method.
    :vartype nodes_to_date: numpy.ndarray (dtype=np.uint32)
    """

    def __init__(self, tree_sequence, *, progress=False, allow_unary=False):
        """
        :param TreeSequence tree_sequence: The input :class:`tskit.TreeSequence`.
        """

        self.ts = tree_sequence
        self.sample_node_set = set(self.ts.samples())
        if np.any(self.ts.tables.nodes.time[self.ts.samples()] != 0):
            raise ValueError(
                "The SpansBySamples class needs a tree seq with all samples at time 0"
            )
        self.progress = progress

        # We will store the spans in here, and normalize them at the end
        self._spans = defaultdict(
            lambda: defaultdict(lambda: defaultdict(base.FLOAT_DTYPE))
        )

        if not allow_unary:
            if has_locally_unary_nodes(self.ts):
                raise ValueError(
                    "The input tree sequence has unary nodes: tsdate currently requires "
                    "that these are removed using `simplify(keep_unary=False)`"
                )

        with tqdm(total=3, desc="TipCount", disable=not self.progress) as progressbar:
            (
                node_spans,
                trees_with_undated,
                total_fixed_at_0_per_tree,
            ) = self.first_pass(allow_unary=allow_unary)
            progressbar.update()

            # A set of the total_num_tips in different trees (used for missing data)
            self.total_fixed_at_0_counts = set(np.unique(total_fixed_at_0_per_tree))
            # The complete spans for each node, used e.g. for normalizing
            self.node_spans = node_spans

            # Check for remaining undated nodes (all unary ones)
            if self.nodes_remain_to_date():
                self.second_pass(trees_with_undated, total_fixed_at_0_per_tree)
            progressbar.update()
            if self.nodes_remain_to_date():
                self.third_pass(trees_with_undated, total_fixed_at_0_per_tree)
            progressbar.update()
        self.finalize()
        progressbar.close()

    def __repr__(self):
        ret = []
        for n in range(self.ts.num_nodes):
            items = []
            for tot_tips, spans in self.get_spans(n).items():
                items.append(
                    "[{}] / {} ".format(
                        ", ".join([f"{a}: {b}" for a, b in spans]), tot_tips
                    )
                )
            ret.append(f"Node {n: >3}: " + "{" + ", ".join(items) + "}")
        return "\n".join(ret)

    def nodes_remaining_to_date(self):
        """
        Return a set of the node IDs that we want to date, but which haven't had a
        set of spans allocated which could be used to date the node.
        """
        return {
            n
            for n in range(self.ts.num_nodes)
            if not (n in self._spans or n in self.sample_node_set)
        }

    def nodes_remain_to_date(self):
        """
        A more efficient version of nodes_remaining_to_date() that simply tells us if
        there are any more nodes that remain to date, but does not identify which ones
        """
        if self.ts.num_nodes - len(self.sample_node_set) - len(self._spans) != 0:
            # we should always have equal or fewer results than nodes to date
            assert len(self._spans) < self.ts.num_nodes - len(self.sample_node_set)
            return True
        return False

    def first_pass(self, allow_unary=False):
        """
        Returns a tuple of the span that each node covers, a list of the tree indices of
        trees that have undated nodes (used to quickly revist these trees later), and the
        number of valid samples (tips) in each tree.
        """
        logging.debug("Assigning priors to most non-fixed nodes")
        # The following 3 variables will be returned by this function
        node_spans = np.zeros(self.ts.num_nodes)
        trees_with_undated = []  # Used to revisit trees with nodes that need dating
        n_tips_per_tree = np.full(self.ts.num_trees, tskit.NULL, dtype=np.int64)

        # Some useful local tracking variables
        num_children = np.full(self.ts.num_nodes, 0, dtype=np.int32)
        # Store the last recorded genome position for this node.
        # If set to np.nan, this indicates that we are not currently tracking this node
        stored_pos = np.full(self.ts.num_nodes, np.nan)

        # used to emit a warning if necessary
        self.has_unary = False

        def save_to_spans(prev_tree, node, num_fixed_at_0_treenodes):
            """
            A convenience function to save accumulated tracked node data at the current
            breakpoint. If this is a non-fixed node which needs dating, we save the
            span by # descendant tips into self._spans. If it is a sample node, we
            return True and do not save into self._spans. If the node was skipped because
            it is a unary node at the top of the tree, return None.
            """
            if np.isnan(stored_pos[node]):
                # Don't save ones that we aren't tracking
                return False
            n_fixed_at_0 = prev_tree.num_tracked_samples(node)
            if n_fixed_at_0 > 0:
                coverage = prev_tree.interval[1] - stored_pos[node]
                node_spans[node] += coverage
            else:
                coverage = 0
                raise ValueError(
                    "Node {} is dangling (no descendant samples) at pos {}: "
                    "this node will have no weight in this region. Run "
                    "`simplify(keep_unary=False)` before dating this tree "
                    "sequence".format(node, stored_pos[node])
                )
            if node in self.sample_node_set:
                return True
            if prev_tree.num_children(node) > 1:
                # This is a coalescent node
                self._spans[node][num_fixed_at_0_treenodes][n_fixed_at_0] += coverage
            else:
                if not self.has_unary:
                    self.has_unary = True
                # Treat unary nodes differently: mixture of coalescent nodes above+below
                unary_nodes_above = 0
                top_node = prev_tree.parent(node)
                try:  # Find coalescent node above
                    while prev_tree.num_children(top_node) == 1:
                        unary_nodes_above += 1
                        top_node = prev_tree.parent(top_node)
                except ValueError:  # Happens if we have hit the root
                    assert top_node == tskit.NULL
                    logging.debug(
                        "Unary node `{}` exists above highest coalescence in tree {}."
                        " Skipping for now".format(node, prev_tree.index)
                    )
                    return None
                # for unary nodes, a proportion of the span is allocated
                # according to the coalescent node above and the coalescent
                # node below. If there are extra unary nodes above or below
                # weight = 1/2 from parent. If one unary node above, 1/4 from parent, etc
                wt = 2 ** (unary_nodes_above + 1)  # 1/wt from abpve
                iwt = wt / (wt - 1.0)  # 1/iwt from below
                top_node_tips = prev_tree.num_tracked_samples(top_node)
                self._spans[node][num_fixed_at_0_treenodes][top_node_tips] += (
                    coverage / wt
                )
                # The rest from the node below
                #  NB: coalescent node below should have same num_tracked_samples as this
                # TODO - assumes no internal unary sample nodes at 0 (impossible)
                self._spans[node][num_fixed_at_0_treenodes][n_fixed_at_0] += (
                    coverage / iwt
                )
            return True

        # We iterate over edge_diffs to calculate, as nodes change their descendant tips,
        # the genomic coverage for each node partitioned into descendant tip numbers
        edge_diff_iter = self.ts.edge_diffs()
        # There are 2 possible algorithms - the simplest is to find all the affected
        # nodes in the new tree, and use the constant-time "Tree.num_tracked_samples()"
        # to recalculate the number of samples under each affected tip.
        # The more complex one keeps track of the samples added and subtracted
        # each time, basically implementing the num_samples() method itself

        # Initialise with first tree
        for node in self.ts.first().nodes():
            stored_pos[node] = 0
        # Consume the first edge diff: normally this is when most edges come in
        num_fixed_at_0_treenodes = 0
        _, _, e_in = next(edge_diff_iter)
        for e in e_in:
            if e.child in self.sample_node_set:
                num_fixed_at_0_treenodes += 1
            if e.parent != tskit.NULL:
                num_children[e.parent] += 1
        n_tips_per_tree[0] = num_fixed_at_0_treenodes

        # Iterate over trees and remaining edge diffs
        focal_tips = list(self.sample_node_set)
        for prev_tree in tqdm(
            self.ts.trees(tracked_samples=focal_tips, root_threshold=2),
            desc="Find Node Spans",
            total=self.ts.num_trees,
            disable=not self.progress,
        ):
            if prev_tree.has_multiple_roots:
                raise ValueError(f"Tree {prev_tree.index} has multiple roots")
            try:
                # Get the edge diffs from the prev tree to the new tree
                _, e_out, e_in = next(edge_diff_iter)
            except StopIteration:
                # Last tree, save all the remaining nodes
                for node in prev_tree.nodes():
                    if save_to_spans(prev_tree, node, num_fixed_at_0_treenodes) is None:
                        trees_with_undated.append(prev_tree.index)
                assert prev_tree.index == self.ts.num_trees - 1
                continue

            fixed_at_0_nodes_out = set()
            fixed_at_0_nodes_in = set()
            disappearing_nodes = set()
            changed_nodes = set()
            for e in e_out:
                # No need to add the parents, as we'll traverse up the previous tree
                # from these points and be guaranteed to hit them too.
                changed_nodes.add(e.child)
                if e.parent != tskit.NULL:
                    num_children[e.parent] -= 1
                    if num_children[e.parent] == 0:
                        disappearing_nodes.add(e.parent)
            for e in e_out:
                # Since a node only has one parent edge, any edge children going out
                # will be lost, unless they are reintroduced in the edges_in, or are
                # the root node
                if num_children[e.child] == 0:
                    disappearing_nodes.add(e.child)
                    if e.child in self.sample_node_set:  # all samples are at 0
                        fixed_at_0_nodes_out.add(e.child)

            for e in e_in:
                # Edge children are always new
                if e.child in self.sample_node_set:
                    fixed_at_0_nodes_in.add(e.child)
                # This may change in the upcoming tree
                changed_nodes.add(e.child)
                disappearing_nodes.discard(e.child)
                if e.parent != tskit.NULL:
                    # parent nodes might be added in the next tree, and we won't
                    # necessarily traverse up to them in the prev tree, so they need
                    # to be added to the possibly changing nodes
                    changed_nodes.add(e.parent)
                    # If a parent or child come in, they definitely won't be disappearing
                    num_children[e.parent] += 1
                    disappearing_nodes.discard(e.parent)
            # Add unary nodes below the altered ones, as their result is calculated
            # from the coalescent node above
            unary_descendants = set()
            for node in changed_nodes:
                children = prev_tree.children(node)
                for child in children:
                    while True:
                        children = prev_tree.children(child)
                        if len(children) != 1:
                            break
                        unary_descendants.add(child)
                        child = children[0]

            # find all the nodes in the tree that might have changed their number
            # of descendants, and reset. This might include nodes that are not in
            # the prev tree, but will be in the next one (so we still need to
            # set the stored position). Also track visited_nodes so we don't repeat
            visited_nodes = set()
            for node in changed_nodes | unary_descendants:
                while node != tskit.NULL:  # if root or node not in tree
                    if node in visited_nodes:
                        break
                    visited_nodes.add(node)
                    # Node is in tree
                    if save_to_spans(prev_tree, node, num_fixed_at_0_treenodes) is None:
                        trees_with_undated.append(prev_tree.index)
                    if node in disappearing_nodes:
                        # Not tracking this in the future
                        stored_pos[node] = np.nan
                    else:
                        stored_pos[node] = prev_tree.interval[1]
                    node = prev_tree.parent(node)

            # If total number of samples has changed: we need to save
            #  everything so far & reset all starting positions
            if len(fixed_at_0_nodes_in) != len(fixed_at_0_nodes_out):
                for node in prev_tree.nodes():
                    if node in visited_nodes:
                        # We have already saved these - no need to again
                        continue
                    if save_to_spans(prev_tree, node, num_fixed_at_0_treenodes) is None:
                        trees_with_undated.append(prev_tree.index)
                    if node in disappearing_nodes:
                        # Not tracking this in the future
                        stored_pos[node] = np.nan
                    else:
                        stored_pos[node] = prev_tree.interval[1]
                num_fixed_at_0_treenodes += len(fixed_at_0_nodes_in) - len(
                    fixed_at_0_nodes_out
                )
            n_tips_per_tree[prev_tree.index + 1] = num_fixed_at_0_treenodes

        if self.has_unary:
            if allow_unary:
                logging.warning(
                    "The input tree sequence has unary nodes: tsdate may give "
                    "poor results. Remove them using `simplify(keep_unary=False)`"
                )
            else:
                raise ValueError(
                    "The input tree sequence has unary nodes: tsdate currently "
                    "requires these to be removed using `simplify(keep_unary=False)`"
                )
        return node_spans, trees_with_undated, n_tips_per_tree

    def second_pass(self, trees_with_undated, n_tips_per_tree):
        """
        Check for nodes which have unassigned prior params after the first
        pass. We should see if can we assign params for these node priors using
        now-parameterized nodes. This requires another pass through the
        tree sequence. If there is no non-parameterized node above, then
        we can simply assign this the coalescent maximum
        """
        logging.debug(
            "Assigning priors to skipped unary nodes, via linked nodes with new priors"
        )
        unassigned_nodes = self.nodes_remaining_to_date()
        # Simple algorithm does this treewise
        tree_iter = self.ts.trees()
        tree = next(tree_iter)
        for tree_id in tqdm(
            trees_with_undated, desc="2nd pass", disable=not self.progress
        ):
            while tree.index != tree_id:
                tree = next(tree_iter)
            for node in unassigned_nodes:
                if tree.parent(node) == tskit.NULL:
                    continue
                    # node is either the root or (more likely) not in
                    # this tree
                assert tree.num_samples(node) > 0  # No dangling nodes allowed
                assert tree.num_children(node) == 1
                n = node
                done = False
                while not done:
                    n = tree.parent(n)
                    if n == tskit.NULL or n in self._spans:
                        done = True
                if n == tskit.NULL:
                    continue
                else:
                    logging.debug(
                        "Assigning prior to unary node {}: connected to node {} which"
                        " has a prior in tree {}".format(node, n, tree_id)
                    )
                    for n_tips, spans in self._spans[n].items():
                        for k, v in spans.items():
                            if k <= 0:
                                raise ValueError(f"Node {n} has no fixed descendants")
                            local_weight = v / self.node_spans[n]
                            self._spans[node][n_tips][k] += tree.span * local_weight / 2
                    assert tree.num_children(node) == 1
                    total_tips = n_tips_per_tree[tree_id]
                    desc_tips = tree.num_samples(node)
                    self._spans[node][total_tips][desc_tips] += tree.span / 2

    def third_pass(self, trees_with_undated, n_tips_per_tree):
        """
        We STILL have some missing priors.
        These must be unconnected to higher
        nodes in the tree, so we can simply give them the max depth
        """
        logging.debug(
            "Assigning priors to remaining (unconnected) unary nodes using max depth"
        )
        max_samples = self.ts.num_samples
        unassigned_nodes = self.nodes_remaining_to_date()
        tree_iter = self.ts.trees()
        tree = next(tree_iter)
        for tree_id in tqdm(
            trees_with_undated, desc="3rd pass", disable=not self.progress
        ):
            while tree.index != tree_id:
                tree = next(tree_iter)
            for node in unassigned_nodes:
                if tree.is_internal(node):
                    assert tree.num_children(node) == 1
                    total_tips = n_tips_per_tree[tree_id]
                    # above, we set the maximum
                    self._spans[node][max_samples][max_samples] += tree.span / 2
                    # below, we do as before
                    desc_tips = tree.num_samples(node)
                    self._spans[node][total_tips][desc_tips] += tree.span / 2

    def finalize(self):
        """
        normalize the spans in self._spans by the values in self.node_spans,
        and overwrite the results (as we don't need them any more), providing a
        shortcut to by setting node_span_data. Also provide the
        nodes_to_date value.
        """
        assert not hasattr(self, "node_span_data"), "Already finalized"
        spans_dtype = np.dtype(
            {
                "names": ("descendant_tips", "span"),
                "formats": (np.uint64, base.FLOAT_DTYPE),
            }
        )

        if self.nodes_remain_to_date():
            raise ValueError(
                "When finalising node spans, found the following nodes not in any tree;"
                " you must simplify your tree sequence first: {}".format(
                    self.nodes_remaining_to_date()
                )
            )

        for node, spans_by_total_tips in self._spans.items():
            self._spans[node] = {}  # Overwrite, so we don't leave the old data around
            for num_samples, spans in sorted(spans_by_total_tips.items()):
                wt = np.array([(k, v) for k, v in spans.items()], dtype=spans_dtype)
                self._spans[node][num_samples] = wt
        # Assign into the instance, for further reference
        self.node_span_data = self._spans
        self.nodes_to_date = np.array(list(self._spans.keys()), dtype=np.uint64)

    def get_spans(self, node):
        """
        Access the main calculated results from this class, returning spans
        for a node contained within a dict of dicts. Spans for each node
        are divided into regions with different numbers of sample descendants,
        and sum to the total span over which that node is present
        in trees along the tree sequence. They are used to construct
        the mixed conditional coalescent prior. For each coalescent node, the
        returned spans are categorised firstly by the total number of sample
        nodes (or "tips") ( :math:`T` ) in the tree(s) covered by this node,
        then by the number of descendant samples, :math:`k`. In other words,
        ``spans(u)[T][k]`` gives the fraction of the genome over which node
        ``u`` is present in a tree of ``T`` total samples with exactly ``k``
        samples descending from the node. Although ``k`` may take any value
        from 2 up to ``T``, the values are likely to be very sparse, and many
        values of both ``T`` and ``k`` are likely to be missing from the
        returned spans. For example, if there are no trees in which the node
        ``u`` has exactly 2 descendant samples, then none of the inner
        dictionaries returned by this method will have a key of 2.

        Non-coalescent (unary) regions of nodes are treated differently. A unary
        region of a node returns a 50:50  mix of the coalescent node above and
        the coalescent node below it.

        :param int node: The node for which we want spans.
        :return: A dictionary, whose keys ( :math:`n_t` ) are the total number of
            samples in the trees in a tree sequence, and whose values are
            themselves a dictionary where key :math:`k` gives the genomic
            span for :math:`k` descendant samples, as a floating point number.
            For any node ``u``, the normalization means that all the spans should
            sum to ``self.node_spans[u]``.
        :rtype: dict(int, numpy.ndarray)'
        """
        return self.node_span_data[node]

    def lookup_span(self, node, total_tips, descendant_tips):
        # Only used for testing
        which = self.get_spans(node)[total_tips]["descendant_tips"] == descendant_tips
        return self.get_spans(node)[total_tips]["span"][which]


def create_timepoints(base_priors, n_points=21):
    """
    Create the time points by finding union of the quantiles of the distributions.
    For a node with k descendants we have approximate distributions (either lognorm or
    gamma): a reasonable way to create timepoints is to take all the distributions,
    quantile them up, and then take the union of the quantiles, thinning it to make
    each timepoint no more than 0.05 of a quantile apart. This function does this in an
    iterative way.
    """
    # Assume that the best set of priors to use are those for n descendant tips out of
    # max total tips, where n = 2 .. max_total_tips. This is only relevant when we have
    # missing samples, otherwise we only have one set of priors anyway
    prior_params = base_priors.prior_with_max_total_tips()
    # Percentages - current day samples should be at time 0, so we omit this
    # We can't include the top end point, as this leads to NaNs
    percentiles = np.linspace(0, 1, n_points + 1)[1:-1]
    # percentiles = np.append(percentiles, 0.999999)
    param_cols = np.where([f not in ("mean", "var") for f in PriorParams._fields])[0]
    """
    get the set of times from gamma percent point function at the given
    percentiles specifies the value of the RV such that the prob of the var
    being less than or equal to that value equals the given probability
    """
    if base_priors.prior_distr == "lognorm":

        def lognorm_ppf(percentiles, alpha, beta):
            return scipy.stats.lognorm.ppf(
                percentiles, s=np.sqrt(beta), scale=np.exp(alpha)
            )

        ppf = lognorm_ppf

        def lognorm_cdf(t_set, alpha, beta):
            return scipy.stats.lognorm.cdf(t_set, s=np.sqrt(beta), scale=np.exp(alpha))

        cdf = lognorm_cdf

    elif base_priors.prior_distr == "gamma":

        def gamma_ppf(percentiles, alpha, beta):
            return scipy.stats.gamma.ppf(percentiles, alpha, scale=1 / beta)

        ppf = gamma_ppf

        def gamma_cdf(t_set, alpha, beta):
            return scipy.stats.gamma.cdf(t_set, alpha, scale=1 / beta)

        cdf = gamma_cdf
    else:
        raise ValueError("prior distribution must be lognorm or gamma")

    t_set = ppf(percentiles, *prior_params[2, param_cols])
    max_tips = len(prior_params)  # Num rows in prior_params == prior_params.shape[0]
    # progressively add timepoints
    max_sep = 1.0 / (n_points - 1)
    if max_tips > 2:
        for i in np.arange(3, max_tips):
            # cdf percentiles of existing timepoints
            proj = cdf(t_set, *prior_params[i, param_cols])
            """
            thin the timepoints, only add additional quantiles if they're more than
            a certain max_sep fraction (e.g. 0.05) from another quantile
            """
            tmp = np.asarray([min(abs(val - proj)) for val in percentiles])
            wd = np.where(tmp > max_sep)

            if len(wd[0]) > 0:
                t_set = np.concatenate(
                    [t_set, ppf(percentiles[wd], *prior_params[i, param_cols])]
                )

    t_set = sorted(t_set)
    return np.insert(t_set, 0, 0)


def fill_priors(
    node_parameters, timepoints, ts, population_size, *, prior_distr, progress=False
):
    """
    Take the alpha and beta values from the node_parameters array, which contains
    one row for each node in the TS (including fixed nodes)
    and fill out a NodeGridValues object with the prior values from the
    gamma or lognormal distribution with those parameters.

    The `population_size` can be a scalar, or an object with a `.to_natural_timescale`
    method used to map from coalescent to generational timescale.

    TODO - what if there is an internal fixed node? Should we truncate

    TODO - support times scaled by generation length?
    """
    if prior_distr == "lognorm":
        cdf_func = scipy.stats.lognorm.cdf
        main_param = np.sqrt(node_parameters[:, PriorParams.field_index("beta")])
        scale_param = np.exp(node_parameters[:, PriorParams.field_index("alpha")])
    elif prior_distr == "gamma":
        cdf_func = scipy.stats.gamma.cdf
        main_param = node_parameters[:, PriorParams.field_index("alpha")]
        scale_param = 1 / node_parameters[:, PriorParams.field_index("beta")]
    else:
        raise ValueError("prior distribution must be lognorm or gamma")

    datable_nodes = np.ones(ts.num_nodes, dtype=bool)
    datable_nodes[ts.samples()] = False
    datable_nodes = np.where(datable_nodes)[0]

    # convert timepoints to generational timescale
    prior_times = base.NodeGridValues(
        ts.num_nodes,
        datable_nodes[np.argsort(ts.tables.nodes.time[datable_nodes])].astype(np.int32),
        population_size.to_natural_timescale(timepoints),
    )

    # TO DO - this can probably be done in an single numpy step rather than a for loop
    for node in tqdm(
        datable_nodes, desc="Assign Prior to Each Node", disable=not progress
    ):
        # NB: prior CDF is evaluated on coalescent timescale
        with np.errstate(divide="ignore", invalid="ignore"):
            prior_node = cdf_func(timepoints, main_param[node], scale=scale_param[node])
        # force age to be less than max value
        prior_node = np.divide(prior_node, np.max(prior_node))
        # prior in each epoch
        prior_times[node] = np.concatenate([np.array([0]), np.diff(prior_node)])
    # standardize so max value is 1
    prior_times.standardize()
    return prior_times


class MixturePrior:
    """
    Maps ConditionalCoalescentPrior onto nodes in a tree sequence and creates
    time-discretised priors
    """

    def __init__(
        self,
        tree_sequence,
        approximate_priors=False,
        approx_prior_size=None,
        prior_distribution="lognorm",
        allow_unary=False,
        progress=False,
    ):
        if approximate_priors:
            if not approx_prior_size:
                approx_prior_size = base.DEFAULT_APPROX_PRIOR_SIZE
        else:
            if approx_prior_size is not None:
                raise ValueError(
                    "Can't set approx_prior_size if approximate_prior is False"
                )

        contmpr_ts, node_map = util.reduce_to_contemporaneous(tree_sequence)
        if contmpr_ts.num_nodes != tree_sequence.num_nodes:
            raise ValueError(
                "Passed tree sequence is not simplified and/or contains "
                "noncontemporaneous samples"
            )
        span_data = SpansBySamples(
            contmpr_ts, progress=progress, allow_unary=allow_unary
        )

        base_priors = ConditionalCoalescentTimes(
            approx_prior_size, prior_distribution, progress=progress
        )

        base_priors.add(contmpr_ts.num_samples, approximate_priors)
        for total_fixed in span_data.total_fixed_at_0_counts:
            # For missing data: trees vary in total fixed node count =>
            # have different priors
            if total_fixed > 0:
                base_priors.add(total_fixed, approximate_priors)
        prior_params_contmpr = base_priors.get_mixture_prior_params(span_data)

        # Map the nodes in the prior params back to the node ids in the original ts
        self.prior_params = prior_params_contmpr[node_map, :]
        self.base_priors = base_priors
        self.tree_sequence = tree_sequence
        self.prior_distribution = prior_distribution

    def make_discretised_prior(self, population_size, timepoints=20, progress=False):
        """
        Calculate prior grid for a set of timepoints and a population size history
        """

        if isinstance(population_size, (int, float, np.ndarray)):
            population_size = demography.PopulationSizeHistory(population_size)

        if isinstance(timepoints, int):
            if timepoints < 2:
                raise ValueError("You must have at least 2 time points")
            timepoints = create_timepoints(self.base_priors, timepoints + 1)
        elif isinstance(timepoints, np.ndarray):
            try:
                timepoints = np.sort(
                    timepoints.astype(base.FLOAT_DTYPE, casting="safe")
                )
            except TypeError:
                raise TypeError("Timepoints array cannot be converted to float dtype")
            if len(timepoints) < 2:
                raise ValueError("You must have at least 2 time points")
            elif np.any(timepoints < 0):
                raise ValueError("Timepoints cannot be negative")
            elif np.any(np.unique(timepoints, return_counts=True)[1] > 1):
                raise ValueError("Timepoints cannot have duplicate values")
            # timepoints are assumed to be on generational scale, so convert to
            # coalescent timescale to evaluate prior
            timepoints = population_size.to_coalescent_timescale(timepoints)
        else:
            raise ValueError(
                "time_slices must be an integer or a numpy array of floats"
            )

        # Set all fixed nodes (i.e. samples) to have 0 variance
        priors = fill_priors(
            self.prior_params,
            timepoints,
            self.tree_sequence,
            population_size,
            prior_distr=self.prior_distribution,
            progress=progress,
        )
        return priors

    def make_parameter_grid(self, population_size, progress=False):
        """
        Adjust prior parameters given a population size history
        """

        if self.prior_distribution != "gamma":
            raise ValueError("Parameter grid may only be calculated with gamma priors")

        if isinstance(population_size, (int, float, np.ndarray)):
            population_size = demography.PopulationSizeHistory(population_size)

        ts = self.tree_sequence

        datable_nodes = np.ones(ts.num_nodes, dtype=bool)
        datable_nodes[ts.samples()] = False
        datable_nodes = np.where(datable_nodes)[0]

        prior_pars = base.NodeGridValues(
            self.tree_sequence.num_nodes,
            datable_nodes[np.argsort(ts.tables.nodes.time[datable_nodes])].astype(
                np.int32
            ),
            np.array([0, np.inf]),
        )
        prior_pars.probability_space = base.GAMMA_PAR

        shape = self.prior_params[:, PriorParams.field_index("alpha")]
        rate = self.prior_params[:, PriorParams.field_index("beta")]
        for node in tqdm(
            datable_nodes, desc="Assign Prior to Each Node", disable=not progress
        ):
            prior_pars[node] = population_size.gamma_to_natural(shape[node], rate[node])

        return prior_pars


def prior_grid(
    tree_sequence,
    population_size,
    timepoints=20,
    *,
    approximate_priors=False,
    approx_prior_size=None,
    prior_distribution="lognorm",
    # Parameters below undocumented
    progress=False,
    allow_unary=False,
):
    """
    Using the conditional coalescent, calculate the prior distribution for the age of
    each node, given the number of contemporaneous samples below it, and
    the discretised time slices at which to evaluate node age.

    :param tskit.TreeSequence tree_sequence: The input :class:`tskit.TreeSequence`,
        treated as undated.
    :param float or demography.PopulationSizeHistory population_size: The estimated
        (diploid) effective population size used to construct the prior.
        For a population with constant size, this can be given as a single
        value. For a population with time-varying size, this can be given directly as
        a :class:`~demography.PopulationSizeHistory` object or a parameter dictionary
        passed to initialise a :class:`~demography.PopulationSizeHistory` object.
        Using standard (unscaled) values for ``population_size`` results in a prior where
        times are measured in generations.
    :param int or array_like timepoints: The number of quantiles used to create the
        time slices, or manually-specified time slices as a numpy array. Default: 20
    :param bool approximate_priors: Whether to use a precalculated approximation to the
        treewise conditional coalescent prior if there are large numbers of sample tips.
        If an approximate prior has not been precalculated, tsdate will do so and cache
        the result. Default: False
    :param int approx_prior_size: Number of samples above which a precalculated prior is
        used. Only valid if ``approximate_priors`` is True. Default: ``None``, treated as
        :data:`~tsdate.base.DEFAULT_APPROX_PRIOR_SIZE` if ``approximate_priors`` is True.
    :param string prior_distr: What distribution to use to approximate the conditional
        coalescent prior. Can be "lognorm" for the lognormal distribution (generally a
        better fit, but slightly slower to calculate) or "gamma" for the gamma
        distribution (slightly faster, but a poorer fit for recent nodes). Default:
        "lognorm"
    :return: A prior object to pass to :func:`date` and similar functions containing
        prior values for inference and a discretised time grid
    :rtype:  base.NodeGridValues
    """

    mixture_prior = MixturePrior(
        tree_sequence,
        approximate_priors,
        approx_prior_size,
        prior_distribution,
        allow_unary,
        progress,
    )
    return mixture_prior.make_discretised_prior(population_size, timepoints)


def parameter_grid(
    tree_sequence,
    population_size,
    *,
    approximate_priors=False,
    approx_prior_size=None,
    # Parameters below undocumented
    progress=False,
    allow_unary=False,
):
    """
    Using the conditional coalescent, calculate the prior distribution for the age of
    each node, given the number of contemporaneous samples below it, and
    return parameters (shape and rate of gamma) in a grid

    :param tskit.TreeSequence tree_sequence: The input tree sequence, treated as
        undated.
    :param float population_size: The estimated (diploid) effective population
        size: must be specified. May be a single value, or a two-column array with
        epoch breakpoints and effective population sizes. Using standard (unscaled)
        values for ``population_size`` results in a prior where times are measured
        in generations.
    :param bool approximate_priors: Whether to use a precalculated approximation to the
        treewise conditional coalescent prior if there are large numbers of sample tips.
        If an approximate prior has not been precalculated, tsdate will do so and cache
        the result. Default: False
    :param int approx_prior_size: Number of samples above which a precalculated prior is
        used. Only valid if ``approximate_priors`` is True. Default: ``None``, treated as
        :data:`~tsdate.base.DEFAULT_APPROX_PRIOR_SIZE` if ``approximate_priors`` is True.
    :rtype:  base.NodeGridValues
    """

    mixture_prior = MixturePrior(
        tree_sequence,
        approximate_priors,
        approx_prior_size,
        "gamma",
        allow_unary,
        progress,
    )
    return mixture_prior.make_parameter_grid(population_size)


def has_locally_unary_nodes(ts):
    for tree, ediff in zip(ts.trees(), ts.edge_diffs()):
        changed = {
            e.parent for edges in (ediff.edges_out, ediff.edges_in) for e in edges
        }
        if (tree.num_children_array[list(changed)] == 1).any():
            return True
    return False
