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
Routines and classes for creating priors and timeslices for use in tsdate
"""
import logging
import os
from collections import defaultdict
from collections import namedtuple

import numpy as np
import scipy.stats
import tskit
from scipy.special import comb
from tqdm import tqdm

from . import base
from . import cache
from . import provenance
from . import util


PriorParams_base = namedtuple("PriorParams", "alpha, beta, mean, var")


class PriorParams(PriorParams_base):
    @classmethod
    def field_index(cls, fieldname):
        return np.where([f == fieldname for f in cls._fields])[0][0]


def lognorm_approx(mean, var):
    """
    alpha is mean of underlying normal distribution
    beta is variance of underlying normal distribution
    """
    beta = np.log(var / (mean ** 2) + 1)
    alpha = np.log(mean) - 0.5 * beta
    return alpha, beta


def gamma_approx(mean, variance):
    """
    Returns alpha and beta of a gamma distribution for a given mean and variance
    """

    return (mean ** 2) / variance, mean / variance


class ConditionalCoalescentTimes:
    """
    Make and store conditional coalescent priors for different numbers of total samples
    """

    def __init__(self, precalc_approximation_n, prior_distr="lognorm", progress=False):
        """
        :param bool precalc_approximation_n: the size of tree used for
            approximate prior (larger numbers give a better approximation).
            If 0 or otherwise falsey, do not precalculate,
            and therefore do no allow approximate priors to be used
        """
        self.n_approx = precalc_approximation_n
        self.prior_store = {}
        self.progress = progress

        if precalc_approximation_n:
            # Create lookup table based on a large n that can be used for n > ~50
            filename = self.get_precalc_cache(precalc_approximation_n)
            if os.path.isfile(filename):
                # Have already calculated and stored this
                self.approx_priors = np.genfromtxt(filename)
            else:
                # Calc and store
                self.approx_priors = self.precalculate_priors_for_approximation(
                    precalc_approximation_n
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

    def prior_with_max_total_tips(self):
        return self.prior_store.get(max(self.prior_store.keys()))

    def add(self, total_tips, approximate=None):
        """
        Create and store a numpy array used to lookup prior paramsn and mean + variance
        of ages for nodes with descendant sample tips range from 2..``total_tips``
        given that the total number of tips in the coalescent tree is
        ``total_tips``. The array is indexed by (num_tips / total_tips).

        Note: estimated times are scaled by inputted Ne and are haploid
        """
        if total_tips in self.prior_store:
            return  # Already calculated for this number of total tips
        if approximate is not None:
            self.approximate = approximate
        else:
            if total_tips >= 100:
                self.approximate = True
            else:
                self.approximate = False

        if self.approximate and self.approx_priors is None:
            raise RuntimeError(
                "You cannot add an approximate prior unless you initialize"
                " the ConditionalCoalescentTimes object with a non-zero number"
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
        prior_lookup_table[1:, 1] = [self.tau_var(val, n + 1) for val in all_tips]
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
    def m_prob(m, i, n):
        """
        Corollary 2 in Wiuf and Donnelly (1999). Probability of one
        ancestor to entire sample at time tau
        """
        return (comb(n - m - 1, i - 2, exact=True) * comb(m, 2, exact=True)) / comb(
            n, i + 1, exact=True
        )

    @staticmethod
    def tau_expect(i, n):
        if i == n:
            return 2 * (1 - (1 / n))
        else:
            return (i - 1) / n

    @staticmethod
    def tau_squared_conditional(m, n):
        """
        Gives expectation of tau squared conditional on m
        Equation (10) from Wiuf and Donnelly (1999).
        """
        t_sum = np.sum(1 / np.arange(m, n + 1) ** 2)
        return 8 * t_sum + (8 / n) - (8 / m) - (8 / (n * m))

    @staticmethod
    def tau_var(i, n):
        """
        For the last coalesence (n=2), calculate the Tmrca of the whole sample
        """
        if i == n:
            value = np.arange(2, n + 1)
            var = np.sum(1 / ((value ** 2) * ((value - 1) ** 2)))
            return np.abs(4 * var)
        else:
            tau_square_sum = 0
            for m in range(2, n - i + 2):
                tau_square_sum += ConditionalCoalescentTimes.m_prob(
                    m, i, n
                ) * ConditionalCoalescentTimes.tau_squared_conditional(m, n)
            return np.abs(
                (ConditionalCoalescentTimes.tau_expect(i, n) ** 2) - (tau_square_sum)
            )

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
        interpolated_priors[all_tips == total_tips] = self.tau_var(
            total_tips, total_tips
        )
        return interpolated_priors

    def tau_var_exact(self, total_tips, all_tips):
        # TODO, vectorize this properly
        return [
            self.tau_var(tips, total_tips)
            for tips in tqdm(
                all_tips,
                desc="Calculating Node Age Variances",
                disable=not self.progress,
            )
        ]

    def get_mixture_prior_params(self, spans_by_samples):
        """
        Given an object that can be queried for tip weights for a node,
        and a set of conditional coalescent priors for different
        numbers of sample tips under a node, return distribution parameters
        (shape and scale) that best fit the distribution for that node.

        :param .SpansBySamples spans_by_samples: An instance of the
            :class:`SpansBySamples` class that can be used to obtain
            weights for each.
        :return: A numpy array whose rows corresponds to the node id in
            ``spans_by_samples.nodes_to_date`` and whose columns are the parameter
            columns in PriorParams (i.e. not including the mean and variance)
            This can be used to approximate the probabilities of times for that
            node by matching against an appropriate distribution (e.g. gamma or lognorm)
        :rtype:  numpy.ndarray
        """

        mean_column = PriorParams.field_index("mean")
        var_column = PriorParams.field_index("var")
        param_cols = np.array(
            [i for i, f in enumerate(PriorParams._fields) if f not in ("mean", "var")]
        )

        def mixture_expect_and_var(mixture):
            expectation = 0
            first = secnd = 0
            for N, tip_dict in mixture.items():
                # assert 1 not in tip_dict.descendant_tips
                mean = self[N][tip_dict["descendant_tips"], mean_column]
                var = self[N][tip_dict["descendant_tips"], var_column]
                # Mixture expectation
                expectation += np.sum(mean * tip_dict["weight"])
                # Mixture variance
                first += np.sum(var * tip_dict["weight"])
                secnd += np.sum(mean ** 2 * tip_dict["weight"])
            mean = expectation
            var = first + secnd - (expectation ** 2)
            return mean, var

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
            mixture = spans_by_samples.get_weights(node)
            if len(mixture) == 1:
                # The norm: this node spans trees that all have the same set of samples
                total_tips, weight_arr = next(iter(mixture.items()))
                if weight_arr.shape[0] == 1:
                    d_tips = weight_arr["descendant_tips"][0]
                    # This node is not a mixture - can use the standard coalescent prior
                    priors[node] = self[total_tips][d_tips, param_cols]
                elif weight_arr.shape[0] <= 5:
                    # Making mixture priors is a little expensive. We can help by caching
                    # in those cases where we have only a few mixtures
                    # (arbitrarily set here as <= 5 mixtures)
                    mixture_hash = (total_tips, weight_arr.tobytes())
                    if mixture_hash not in seen_mixtures:
                        priors[node] = seen_mixtures[mixture_hash] = self.func_approx(
                            *mixture_expect_and_var(mixture)
                        )
                    else:
                        priors[node] = seen_mixtures[mixture_hash]
                else:
                    # a large number of mixtures in this node - don't bother caching
                    priors[node] = self.func_approx(*mixture_expect_and_var(mixture))
            else:
                # The node spans trees with multiple total tip numbers,
                # don't use the cache
                priors[node] = self.func_approx(*mixture_expect_and_var(mixture))
        # Check that references to the tskit.NULL'th node return NaNs, as we will later
        # be indexing into the prior array using a node mapping which could have NULLs
        assert np.all(np.isnan(priors[tskit.NULL, :]))
        return priors


class SpansBySamples:
    """
    A class to efficiently calculate the genomic spans covered by each
    non-sample node, broken down by the number of samples that descend
    directly from that node. This is used to calculate the conditional
    coalescent prior. The main method is :meth:`get_weights`, which
    returns the spans for a node, normalized by the total span that that
    node covers in the tree sequence.

    .. note:: This assumes that all edges connect to the same tree - i.e.
        there is only a single topology present at each point in the
        genome. Equivalently, it assumes that only one of the roots in
        a tree has descending edges (all other roots represent isolated
        "missing data" nodes.

    :ivar tree_sequence: A reference to the tree sequence that was used to
        generate the spans and weights
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
        :meth:`weights` method.
    :vartype nodes_to_date: numpy.ndarray (dtype=np.uint32)
    """

    def __init__(self, tree_sequence, progress=False):
        """
        :param TreeSequence ts: The input :class:`tskit.TreeSequence`.
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

        with tqdm(total=3, desc="TipCount", disable=not self.progress) as progressbar:
            (
                node_spans,
                trees_with_undated,
                total_fixed_at_0_per_tree,
            ) = self.first_pass()
            progressbar.update()

            # A set of the total_num_tips in different trees (used for missing data)
            self.total_fixed_at_0_counts = set(np.unique(total_fixed_at_0_per_tree))
            # The complete spans for each node, used e.g. for normalising
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
            for tot_tips, weights in self.get_weights(n).items():
                items.append(
                    "[{}] / {} ".format(
                        ", ".join([f"{a}: {b}" for a, b in weights]), tot_tips
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

    def first_pass(self):
        """
        Returns a tuple of the span that each node covers, a list of the tree indices of
        trees that have undated nodes (used to quickly revist these trees later), and the
        number of valid samples (tips) in each tree.
        """
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
                logging.warning(
                    "Node {} is dangling (no descendant samples) at pos {}: "
                    "this node will have no weight in this region".format(
                        node, stored_pos[node]
                    )
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
                # Weights are exponential fractions: if no unary nodes above, we have
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
            self.ts.trees(tracked_samples=focal_tips),
            desc="Find Node Spans",
            total=self.ts.num_trees,
            disable=not self.progress,
        ):
            util.get_single_root(prev_tree)  # Check only one root
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
                if children is not None:
                    if len(children) == 1:
                        # Keep descending
                        node == children[0]
                        while True:
                            children = prev_tree.children(node)
                            if len(children) != 1:
                                break
                            unary_descendants.add(node)
                            node = children[0]
                    else:
                        # Descend all branches, looking for unary nodes
                        for node in prev_tree.children(node):
                            while True:
                                children = prev_tree.children(node)
                                if len(children) != 1:
                                    break
                                unary_descendants.add(node)
                                node = children[0]

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
            logging.warning(
                "The input tree sequence has unary nodes: tsdate currently works "
                "better if these are removed using `simplify(keep_unary=False)`"
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
                    for n_tips, weights in self._spans[n].items():
                        for k, v in weights.items():
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
        shortcut to by setting normalized_node_span_data. Also provide the
        nodes_to_date value.
        """
        assert not hasattr(self, "normalized_node_span_data"), "Already finalized"
        weight_dtype = np.dtype(
            {
                "names": ("descendant_tips", "weight"),
                "formats": (np.uint64, base.FLOAT_DTYPE),
            }
        )

        if self.nodes_remain_to_date():
            logging.warning(
                "When finalising node spans, found the following nodes not in any tree;"
                " you should probably simplify your tree sequence: {}".format(
                    self.nodes_remaining_to_date()
                )
            )

        for node, weights_by_total_tips in self._spans.items():
            self._spans[node] = {}  # Overwrite, so we don't leave the old data around
            for num_samples, weights in sorted(weights_by_total_tips.items()):
                wt = np.array([(k, v) for k, v in weights.items()], dtype=weight_dtype)
                with np.errstate(invalid="ignore"):
                    # Allow self.node_spans[node]=0 -> nan
                    wt["weight"] /= self.node_spans[node]
                self._spans[node][num_samples] = wt
        # Assign into the instance, for further reference
        self.normalized_node_span_data = self._spans
        self.nodes_to_date = np.array(list(self._spans.keys()), dtype=np.uint64)

    def get_weights(self, node):
        """
        Access the main calculated results from this class, returning weights
        for a node contained within a dict of dicts. Weights for each node
        (i.e. normalized genomic spans) sum to one, and are used to construct
        the mixed conditional coalescent prior. For each coalescent node, the
        returned weights are categorised firstly by the total number of sample
        nodes (or "tips") ( :math:`T` ) in the tree(s) covered by this node,
        then by the number of descendant samples, :math:`k`. In other words,
        ``weights(u)[T][k]`` gives the fraction of the genome over which node
        ``u`` is present in a tree of ``T`` total samples with exactly ``k``
        samples descending from the node. Although ``k`` may take any value
        from 2 up to ``T``, the values are likely to be very sparse, and many
        values of both ``T`` and ``k`` are likely to be missing from the
        returned weights. For example, if there are no trees in which the node
        ``u`` has exactly 2 descendant samples, then none of the inner
        dictionaries returned by this method will have a key of 2.

        Non-coalescent (unary) nodes are treated differently. A unary node
        returns a 50:50  mix of the coalescent node above and the coalescent
        node below it.

        :param int node: The node for which we want weights.
        :return: A dictionary, whose keys ( :math:`n_t` ) are the total number of
            samples in the trees in a tree sequence, and whose values are
            themselves a dictionary where key :math:`k` gives the weight (genomic
            span, normalized by the total span over which the node exists) for
            :math:`k` descendant samples, as a floating point number. For any node,
            the normalisation means that all the weights should sum to one.
        :rtype: dict(int, numpy.ndarray)'
        """
        return self.normalized_node_span_data[node]

    def lookup_weight(self, node, total_tips, descendant_tips):
        # Only used for testing
        which = self.get_weights(node)[total_tips]["descendant_tips"] == descendant_tips
        return self.get_weights(node)[total_tips]["weight"][which]


def create_timepoints(base_priors, prior_distr, n_points=21):
    """
    Create the time points by finding union of the quantiles of the gammas
    For a node with k descendants we have gamma approxs.
    Reasonable way to create timepoints is to take all the distributions
    quantile them up, and then take the union of the quantiles.
    Then thin this, making it no more than 0.05 of a quantile apart.
    Takes all the gamma distributions, finds quantiles, takes union,
    and thins them. Does this in an iterative way.
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
    if prior_distr == "lognorm":

        def lognorm_ppf(percentiles, alpha, beta):
            return scipy.stats.lognorm.ppf(
                percentiles, s=np.sqrt(beta), scale=np.exp(alpha)
            )

        ppf = lognorm_ppf

        def lognorm_cdf(t_set, alpha, beta):
            return scipy.stats.lognorm.cdf(t_set, s=np.sqrt(beta), scale=np.exp(alpha))

        cdf = lognorm_cdf

    elif prior_distr == "gamma":

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


def fill_priors(node_parameters, timepoints, ts, *, prior_distr, progress=False):
    """
    Take the alpha and beta values from the node_parameters array, which contains
    one row for each node in the TS (including fixed nodes)
    and fill out a NodeGridValues object with the prior values from the
    gamma or lognormal distribution with those parameters.

    TODO - what if there is an internal fixed node? Should we truncate
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
    prior_times = base.NodeGridValues(
        ts.num_nodes,
        datable_nodes[np.argsort(ts.tables.nodes.time[datable_nodes])].astype(np.int32),
        timepoints,
    )

    # TO DO - this can probably be done in an single numpy step rather than a for loop
    for node in tqdm(
        datable_nodes, desc="Assign Prior to Each Node", disable=not progress
    ):
        with np.errstate(divide="ignore", invalid="ignore"):
            prior_node = cdf_func(timepoints, main_param[node], scale=scale_param[node])
        # force age to be less than max value
        prior_node = np.divide(prior_node, np.max(prior_node))
        # prior in each epoch
        prior_times[node] = np.concatenate([np.array([0]), np.diff(prior_node)])
    # normalize so max value is 1
    prior_times.normalize()
    return prior_times


def build_grid(
    tree_sequence,
    timepoints=20,
    *,
    approximate_priors=False,
    approx_prior_size=None,
    prior_distribution="lognorm",
    eps=1e-6,
    progress=False,
):
    """
    Using the conditional coalescent, calculate the prior distribution for the age of
    each node given the number of contemporaneous samples below it, and the discretised
    time slices at which to evaluate node age.

    :param TreeSequence tree_sequence: The input :class:`tskit.TreeSequence`, treated as
        undated
    :param int_or_array_like timepoints: The number of quantiles used to create the
        time slices, or manually-specified time slices as a numpy array. Default: 20
    :param bool approximate_priors: Whether to use a precalculated approximate prior or
        exactly calculate prior. If approximate prior has not been precalculated, tsdate
        will do so and cache the result. Default: False
    :param int approx_prior_size: Number of samples from which to precalculate prior.
        Should only enter value if approximate_priors=True. If approximate_priors=True
        and no value specified, defaults to 1000. Default: None
    :param string prior_distr: What distribution to use to approximate the conditional
        coalescent prior. Can be "lognorm" for the lognormal distribution (generally a
        better fit, but slightly slower to calculate) or "gamma" for the gamma
        distribution (slightly faster, but a poorer fit for recent nodes). Default:
        "lognorm"
    :param float eps: Specify minimum distance separating points in the time grid. Also
        specifies the error factor in time difference calculations. Default: 1e-6
    :return: A prior object to pass to tsdate.date() containing prior values for
        inference and a discretised time grid
    :rtype:  base.NodeGridValues Object
    """
    if approximate_priors:
        if not approx_prior_size:
            approx_prior_size = 1000
    else:
        if approx_prior_size is not None:
            raise ValueError(
                "Can't set approx_prior_size if approximate_prior is False"
            )

    contmpr_ts, node_map = util.reduce_to_contemporaneous(tree_sequence)
    span_data = SpansBySamples(contmpr_ts, progress=progress)

    base_priors = ConditionalCoalescentTimes(
        approx_prior_size, prior_distribution, progress=progress
    )

    base_priors.add(contmpr_ts.num_samples, approximate_priors)
    for total_fixed in span_data.total_fixed_at_0_counts:
        # For missing data: trees vary in total fixed node count => have different priors
        if total_fixed > 0:
            base_priors.add(total_fixed, approximate_priors)

    if isinstance(timepoints, int):
        if timepoints < 2:
            raise ValueError("You must have at least 2 time points")
        timepoints = create_timepoints(base_priors, prior_distribution, timepoints + 1)
    elif isinstance(timepoints, np.ndarray):
        try:
            timepoints = np.sort(timepoints.astype(base.FLOAT_DTYPE, casting="safe"))
        except TypeError:
            raise TypeError("Timepoints array cannot be converted to float dtype")
        if len(timepoints) < 2:
            raise ValueError("You must have at least 2 time points")
        elif np.any(timepoints < 0):
            raise ValueError("Timepoints cannot be negative")
        elif np.any(np.unique(timepoints, return_counts=True)[1] > 1):
            raise ValueError("Timepoints cannot have duplicate values")
    else:
        raise ValueError("time_slices must be an integer or a numpy array of floats")

    prior_params_contmpr = base_priors.get_mixture_prior_params(span_data)
    # Map the nodes in the prior params back to the node ids in the original ts
    prior_params = prior_params_contmpr[node_map, :]
    # Set all fixed nodes (i.e. samples) to have 0 variance
    priors = fill_priors(
        prior_params,
        timepoints,
        tree_sequence,
        prior_distr=prior_distribution,
        progress=progress,
    )
    return priors
