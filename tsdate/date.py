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
from collections import defaultdict, namedtuple
import logging
import os
import itertools
import multiprocessing
import operator
import functools

import tskit

import numba
import numpy as np
import scipy.stats
from scipy.special import comb
from tqdm import tqdm

FORMAT_NAME = "tsdate"
FORMAT_VERSION = [1, 0]
FLOAT_DTYPE = np.float64
LIN = "linear"
LOG = "logarithmic"


PriorParams_base = namedtuple("PriorParams", 'alpha, beta, mean, var')


class PriorParams(PriorParams_base):
    @classmethod
    def field_index(cls, fieldname):
        return np.where([f == fieldname for f in cls._fields])[0][0]


# Local functions to allow tsdate to work with non-dev versions of tskit
def edge_span(edge):
    return edge.right - edge.left


def tree_num_children(tree, node):
    return len(tree.children(node))


def tree_is_isolated(tree, node):
    return tree_num_children(tree, node) == 0 and tree.parent(node) == tskit.NULL


def tree_iterator_len(it):
    return it.tree_sequence.num_trees


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


class ConditionalCoalescentTimes():
    """
    Make and store conditional coalescent priors
    """

    def __init__(self, precalc_approximation_n, prior_distr='lognorm'):
        """
        :param bool precalc_approximation_n: the size of tree used for
            approximate prior (larger numbers give a better approximation).
            If 0 or otherwise falsey, do not precalculate,
            and therefore do no allow approximate priors to be used
        """
        self.n_approx = precalc_approximation_n
        self.prior_store = {}

        if precalc_approximation_n:
            # Create lookup table based on a large n that can be used for n > ~50
            filename = self.precalc_approx_fn(precalc_approximation_n)
            if os.path.isfile(filename):
                # Have already calculated and stored this
                self.approx_prior = np.genfromtxt(filename)
            else:
                # Calc and store
                self.approx_prior = self.precalculate_prior_for_approximation(
                    precalc_approximation_n)
        else:
            self.approx_prior = None

        self.prior_distr = prior_distr
        if prior_distr == 'lognorm':
            self.func_approx = lognorm_approx
        elif prior_distr == 'gamma':
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

        if self.approximate and self.approx_prior is None:
            raise RuntimeError(
                "You cannot add an approximate prior unless you initialize"
                " the ConditionalCoalescentTimes object with a non-zero number")

        # alpha/beta and mean/var are simply transformations of one another
        # for the gamma, mean = alpha / beta and var = alpha / (beta **2)
        # for the lognormal, see lognorm_approx for definition
        # mean and var are used in generating the mixture prior
        # alpha and beta are used for generating prior probabilities
        # they are stored separately to obviate need to move between them
        # We should only use prior[2] upwards
        prior = np.full(
            (total_tips + 1, len(PriorParams._fields)), np.nan, dtype=FLOAT_DTYPE)

        if self.approximate:
            get_tau_var = self.tau_var_lookup
        else:
            get_tau_var = self.tau_var_exact

        all_tips = np.arange(2, total_tips + 1)
        variances = get_tau_var(total_tips, all_tips)
        # prior.loc[1] is distribution of times of a "coalescence node" ending
        # in a single sample - equivalent to the time of the sample itself, so
        # it should have var = 0 and mean = sample.time
        if self.prior_distr == 'lognorm':
            # For a lognormal, alpha = -inf and beta = 1 sets mean == var == 0
            prior[1] = PriorParams(alpha=-np.inf, beta=0, mean=0, var=0)
        elif self.prior_distr == 'gamma':
            # For a gamma, alpha = 0 and beta = 1 sets mean (a/b) == var (a / b^2) == 0
            prior[1] = PriorParams(alpha=0, beta=1, mean=0, var=0)
        for var, tips in zip(variances, all_tips):
            # NB: it should be possible to vectorize this in numpy
            expectation = self.tau_expect(tips, total_tips)
            alpha, beta = self.func_approx(expectation, var)
            prior[tips] = PriorParams(alpha=alpha, beta=beta, mean=expectation, var=var)
        self.prior_store[total_tips] = prior

    def precalculate_prior_for_approximation(self, precalc_approximation_n):
        n = precalc_approximation_n
        logging.debug(
            "Creating prior lookup table for a total tree of n={} tips"
            " in `{}`, this may take some time for large n"
            .format(n, self.precalc_approx_fn(n)))
        # The first value should be zero tips, we don't want the 1 tip value
        prior_lookup_table = np.zeros((n, 2))
        all_tips = np.arange(2, n + 1)
        prior_lookup_table[1:, 0] = all_tips / n
        prior_lookup_table[1:, 1] = [self.tau_var(val, n + 1) for val in all_tips]
        np.savetxt(self.precalc_approx_fn(n), prior_lookup_table)
        return prior_lookup_table

    def clear_precalculated_prior(self):
        if os.path.isfile(self.precalc_approx_fn(self.n_approx)):
            os.remove(self.precalc_approx_fn(self.n_approx))
        else:
            logging.debug(
                "Precalculated prior in `{}` has not been created, so cannot be cleared"
                .format(self.precalc_approx_fn(self.n_approx)))

    @staticmethod
    def precalc_approx_fn(precalc_approximation_n):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(script_dir)
        return os.path.join(
            parent_dir, "data", "prior_{}df.txt".format(precalc_approximation_n))

    @staticmethod
    def m_prob(m, i, n):
        """
        Corollary 2 in Wiuf and Donnelly (1999). Probability of one
        ancestor to entire sample at time tau
        """
        return (comb(n - m - 1, i - 2, exact=True) *
                comb(m, 2, exact=True)) / comb(n, i + 1, exact=True)

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
                tau_square_sum += (
                    ConditionalCoalescentTimes.m_prob(m, i, n) *
                    ConditionalCoalescentTimes.tau_squared_conditional(m, n))
            return np.abs(
                (ConditionalCoalescentTimes.tau_expect(i, n) ** 2) -
                (tau_square_sum))

    # The following are not static as they may need to access self.approx_prior for this
    # instance
    def tau_var_lookup(self, total_tips, all_tips):
        """
        Lookup tau_var if approximate is True
        """
        interpolated_prior = np.interp(all_tips / total_tips,
                                       self.approx_prior[:, 0], self.approx_prior[:, 1])

        # insertion_point = np.searchsorted(all_tips / self.total_tips,
        #    self.approx_prior[:, 0])
        # interpolated_prior = self.approx_prior[insertion_point, 1]

        # The final MRCA we calculate exactly
        interpolated_prior[all_tips == total_tips] = \
            self.tau_var(total_tips, total_tips)
        return interpolated_prior

    def tau_var_exact(self, total_tips, all_tips):
        # TODO, vectorize this properly
        return [self.tau_var(tips, total_tips) for tips in all_tips]

    def get_mixture_prior_params(self, spans_by_samples):
        """
        Given an object that can be queried for tip weights for a node,
        and a set of conditional coalescent priors for different
        numbers of sample tips under a node, return the alpha and beta
        parameters of the gamma distribution that approximates the
        distribution of times for each node by mixing gamma distributions
        fitted to the basic_priors.

        :param .SpansBySamples spans_by_samples: An instance of the
            :class:`SpansBySamples` class that can be used to obtain
            weights for each.
        :return: A numpy array whose rows correspons to the node id in
            ``spans_by_samples.nodes_to_date`` and whose columns are PriorParams.
            This can be used to approximate the probabilities of times for that
            node by matching agaist an appropriate distribution (e.g. gamma or lognorm)
        :rtype:  numpy.ndarray
        """

        mean_column = PriorParams.field_index('mean')
        var_column = PriorParams.field_index('var')
        param_cols = np.array(
            [i for i, f in enumerate(PriorParams._fields) if f not in ('mean', 'var')])

        def mixture_expect_and_var(mixture):
            expectation = 0
            first = secnd = 0
            for N, tip_dict in mixture.items():
                # assert 1 not in tip_dict.descendant_tips
                mean = self[N][tip_dict['descendant_tips'], mean_column]
                var = self[N][tip_dict['descendant_tips'], var_column]
                # Mixture expectation
                expectation += np.sum(mean * tip_dict['weight'])
                # Mixture variance
                first += np.sum(var * tip_dict['weight'])
                secnd += np.sum(mean ** 2 * tip_dict['weight'])
            mean = expectation
            var = first + secnd - (expectation ** 2)
            return mean, var

        seen_mixtures = {}
        # allocate space for params for all nodes, even though we only use nodes_to_date
        num_nodes, num_params = spans_by_samples.ts.num_nodes, len(param_cols)
        prior = np.full((num_nodes + 1, num_params), np.nan, dtype=FLOAT_DTYPE)
        for node in spans_by_samples.nodes_to_date:
            mixture = spans_by_samples.get_weights(node)
            if len(mixture) == 1:
                # The norm: this node spans trees that all have the same set of samples
                total_tips, weight_arr = next(iter(mixture.items()))
                if weight_arr.shape[0] == 1:
                    d_tips = weight_arr['descendant_tips'][0]
                    # This node is not a mixture - can use the standard coalescent prior
                    prior[node] = self[total_tips][d_tips, param_cols]
                elif weight_arr.shape[0] <= 5:
                    # Making mixture priors is a little expensive. We can help by caching
                    # in those cases where we have only a few mixtures
                    # (arbitrarily set here as <= 5 mixtures)
                    mixture_hash = (
                        total_tips,
                        weight_arr.tostring())
                    if mixture_hash not in seen_mixtures:
                        prior[node] = seen_mixtures[mixture_hash] = \
                            self.func_approx(*mixture_expect_and_var(mixture))
                    else:
                        prior[node] = seen_mixtures[mixture_hash]
                else:
                    # a large number of mixtures in this node - don't bother caching
                    prior[node] = self.func_approx(*mixture_expect_and_var(mixture))
            else:
                # The node spans trees with multiple total tip numbers,
                # don't use the cache
                prior[node] = self.func_approx(*mixture_expect_and_var(mixture))
        return prior


class SpansBySamples:
    """
    A class to calculate the genomic spans covered by each
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

    def __init__(self, tree_sequence, fixed_nodes=None, progress=False):
        """
        :param TreeSequence ts: The input :class:`tskit.TreeSequence`.
        :param iterable fixed_nodes: A list of all the nodes in the tree sequence
            whose time is treated as fixed. These nodes will be used to calculate
            prior values for any ancestral nodes. Normally the fixed nodes are
            equivalent to ``ts.samples()``, but this parameter is available so
            that a pre-calculated set can be passed in, to save the expense of
            re-calculating it when setting up the class. If ``None`` (the default)
            a set of fixed_nodes will be constructed during initialization.

            Currently, only the nodes in this set that are at time 0 will be used
            to calculate the prior.
        """

        self.ts = tree_sequence
        self.fixed_nodes = set(self.ts.samples()) if fixed_nodes is None else \
            set(fixed_nodes)
        self.progress = progress
        # TODO check that all fixed nodes are marked as samples
        self.fixed_at_0_nodes = {n for n in self.fixed_nodes
                                 if self.ts.node(n).time == 0}

        # We will store the spans in here, and normalize them at the end
        self._spans = defaultdict(lambda: defaultdict(lambda: defaultdict(FLOAT_DTYPE)))

        with tqdm(total=3, desc="TipCount", disable=not self.progress) as progressbar:
            node_spans, trees_with_undated, total_fixed_at_0_per_tree = self.first_pass()
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
        repr = []
        for n in range(self.ts.num_nodes):
            items = []
            for tot_tips, weights in self.get_weights(n).items():
                items.append("[{}] / {} ".format(
                    ", ".join(["{}: {}".format(a, b) for a, b in weights]), tot_tips))
            repr.append("Node {: >3}: ".format(n) + '{' + ", ".join(items) + '}')
        return "\n".join(repr)

    def nodes_remaining_to_date(self):
        """
        Return a set of the node IDs that we want to date, but which haven't had a
        set of spans allocated which could be used to date the node.
        """
        return {n for n in range(self.ts.num_nodes) if not(
            n in self._spans or n in self.fixed_nodes)}

    def nodes_remain_to_date(self):
        """
        A more efficient version of nodes_remaining_to_date() that simply tells us if
        there are any more nodes that remain to date, but does not identify which ones
        """
        if self.ts.num_nodes - len(self.fixed_nodes) - len(self._spans) != 0:
            # we should always have equal or fewer results than nodes to date
            assert len(self._spans) < self.ts.num_nodes - len(self.fixed_nodes)
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

        def save_to_spans(prev_tree, node, num_fixed_at_0_treenodes):
            """
            A convenience function to save accumulated tracked node data at the current
            breakpoint. If this is a non-fixed node which needs dating, we save the
            span by # descendant tips into self._spans. If the node was skipped because
            it is a unary node at the top of the tree, return None.
            """
            if np.isnan(stored_pos[node]):
                # Don't save ones that we aren't tracking
                return False
            coverage = prev_tree.interval[1] - stored_pos[node]
            node_spans[node] += coverage
            if node in self.fixed_nodes:
                return True
            n_fixed_at_0 = prev_tree.num_tracked_samples(node)
            if n_fixed_at_0 == 0:
                raise ValueError(
                    "Invalid tree sequence: node {} has no descendant samples".format(
                        node))
            if tree_num_children(prev_tree, node) > 1:
                # This is a coalescent node
                self._spans[node][num_fixed_at_0_treenodes][n_fixed_at_0] += coverage
            else:
                # Treat unary nodes differently: mixture of coalescent nodes above+below
                unary_nodes_above = 0
                top_node = prev_tree.parent(node)
                try:  # Find coalescent node above
                    while tree_num_children(prev_tree, top_node) == 1:
                        unary_nodes_above += 1
                        top_node = prev_tree.parent(top_node)
                except ValueError:  # Happens if we have hit the root
                    assert top_node == tskit.NULL
                    logging.debug(
                        "Unary node `{}` exists above highest coalescence in tree {}."
                        " Skipping for now".format(node, prev_tree.index))
                    return None
                # Weights are exponential fractions: if no unary nodes above, we have
                # weight = 1/2 from parent. If one unary node above, 1/4 from parent, etc
                wt = 2**(unary_nodes_above+1)  # 1/wt from abpve
                iwt = wt/(wt - 1.0)            # 1/iwt from below
                top_node_tips = prev_tree.num_tracked_samples(top_node)
                self._spans[node][num_fixed_at_0_treenodes][top_node_tips] += coverage/wt
                # The rest from the node below
                #  NB: coalescent node below should have same num_tracked_samples as this
                # TODO - assumes no internal unary sample nodes at 0 (impossible)
                self._spans[node][num_fixed_at_0_treenodes][n_fixed_at_0] += coverage/iwt
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
            if e.child in self.fixed_at_0_nodes:
                num_fixed_at_0_treenodes += 1
            if e.parent != tskit.NULL:
                num_children[e.parent] += 1
        n_tips_per_tree[0] = num_fixed_at_0_treenodes

        # Iterate over trees and remaining edge diffs
        focal_tips = list(self.fixed_at_0_nodes)
        for prev_tree in tqdm(
                self.ts.trees(tracked_samples=focal_tips),
                desc="Find Node Spans", total=self.ts.num_trees,
                disable=not self.progress):

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
                    if e.child in self.fixed_at_0_nodes:
                        fixed_at_0_nodes_out.add(e.child)

            for e in e_in:
                # Edge children are always new
                if e.child in self.fixed_at_0_nodes:
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
                num_fixed_at_0_treenodes += (len(fixed_at_0_nodes_in) -
                                             len(fixed_at_0_nodes_out))
            n_tips_per_tree[prev_tree.index+1] = num_fixed_at_0_treenodes

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
            "Assigning priors to skipped unary nodes, via linked nodes with new priors")
        unassigned_nodes = self.nodes_remaining_to_date()
        # Simple algorithm does this treewise
        tree_iter = self.ts.trees()
        tree = next(tree_iter)
        for tree_id in tqdm(trees_with_undated, desc="2nd pass",
                            disable=not self.progress):
            while tree.index != tree_id:
                tree = next(tree_iter)
            for node in unassigned_nodes:
                if tree.parent(node) == tskit.NULL:
                    continue
                    # node is either the root or (more likely) not in
                    # this tree
                assert tree.num_samples(node) > 0
                assert tree_num_children(tree, node) == 1
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
                        " has a prior in tree {}".format(node, n, tree_id))
                    for n_tips, weights in self._spans[n].items():
                        for k, v in weights.items():
                            if k <= 0:
                                raise ValueError(
                                    "Node {} has no fixed descendants".format(n))
                            local_weight = v / self.node_spans[n]
                            self._spans[node][n_tips][k] += tree.span * local_weight / 2
                    assert tree_num_children(tree, node) == 1
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
            "Assigning priors to remaining (unconnected) unary nodes using max depth")
        max_samples = self.ts.num_samples
        unassigned_nodes = self.nodes_remaining_to_date()
        tree_iter = self.ts.trees()
        tree = next(tree_iter)
        for tree_id in tqdm(trees_with_undated, desc="3rd pass",
                            disable=not self.progress):
            while tree.index != tree_id:
                tree = next(tree_iter)
            for node in unassigned_nodes:
                if tree.is_internal(node):
                    assert tree_num_children(tree, node) == 1
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
        assert not hasattr(self, 'normalized_node_span_data'), "Already finalized"
        weight_dtype = np.dtype({
            'names': ('descendant_tips', 'weight'),
            'formats': (np.uint64, FLOAT_DTYPE)})

        if self.nodes_remain_to_date():
            raise ValueError(
                "When finalising node spans, found the following nodes not in any tree;"
                " try simplifing your tree sequence: {}"
                .format(self.nodes_remaining_to_date()))

        for node, weights_by_total_tips in self._spans.items():
            self._spans[node] = {}  # Overwrite, so we don't leave the old data around
            for num_samples, weights in sorted(weights_by_total_tips.items()):
                wt = np.array([(k, v) for k, v in weights.items()], dtype=weight_dtype)
                wt['weight'] /= self.node_spans[node]
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
        which = self.get_weights(node)[total_tips]['descendant_tips'] == descendant_tips
        return self.get_weights(node)[total_tips]['weight'][which]


def create_timepoints(age_prior, prior_distr, n_points=21):
    """
    Create the time points by finding union of the quantiles of the gammas
    For a node with k descendants we have gamma approxs.
    Reasonable way to create timepoints is to take all the distributions
    quantile them up, and then take the union of the quantiles.
    Then thin this, making it no more than 0.05 of a quantile apart.
    Takes all the gamma distributions, finds quantiles, takes union,
    and thins them. Does this in an iterative way.
    """
    # Percentages - current day samples should be at time 0, so we omit this
    # We can't include the top end point, as this leads to NaNs
    percentiles = np.linspace(0, 1, n_points + 1)[1:-1]
    # percentiles = np.append(percentiles, 0.999999)
    param_cols = np.where([f not in ('mean', 'var') for f in PriorParams._fields])[0]
    """
    get the set of times from gamma percent point function at the given
    percentiles specifies the value of the RV such that the prob of the var
    being less than or equal to that value equals the given probability
    """
    if prior_distr == 'lognorm':
        def lognorm_ppf(percentiles, alpha, beta):
            return scipy.stats.lognorm.ppf(percentiles, s=np.sqrt(beta),
                                           scale=np.exp(alpha))
        ppf = lognorm_ppf

        def lognorm_cdf(t_set, alpha, beta):
            return scipy.stats.lognorm.cdf(t_set, s=np.sqrt(beta), scale=np.exp(alpha))
        cdf = lognorm_cdf

    elif prior_distr == 'gamma':
        def gamma_ppf(percentiles, alpha, beta):
            return scipy.stats.gamma.ppf(percentiles, alpha, scale=1 / beta)
        ppf = gamma_ppf

        def gamma_cdf(t_set, alpha, beta):
            return scipy.stats.gamma.cdf(t_set, alpha, scale=1 / beta)
        cdf = gamma_cdf
    else:
        raise ValueError("prior distribution must be lognorm or gamma")

    t_set = ppf(percentiles, *age_prior[2, param_cols])

    # progressively add timepoints
    max_sep = 1.0 / (n_points - 1)
    if age_prior.shape[0] > 2:
        for i in np.arange(3, age_prior.shape[0]):
            # gamma percentiles of existing timepoints
            proj = cdf(t_set, *age_prior[i, param_cols])
            """
            thin the timepoints, only add additional quantiles if they're more than
            a certain max_sep fraction (e.g. 0.05) from another quantile
            """
            tmp = np.asarray([min(abs(val - proj)) for val in percentiles])
            wd = np.where(tmp > max_sep)

            if len(wd[0]) > 0:
                t_set = np.concatenate([
                    t_set,
                    ppf(percentiles[wd], *age_prior[i, param_cols])])

    t_set = sorted(t_set)
    return np.insert(t_set, 0, 0)


class NodeGridValues:
    """
    A class to store grid values for node ids. For some nodes (fixed ones), only a single
    value needs to be stored. For non-fixed nodes, an array of grid_size variables
    is required, e.g. in order to store all the possible values for each of the hidden
    states in the grid

    :ivar num_nodes: The number of nodes that will be stored in this object
    :vartype num_nodes: int
    :ivar nonfixed_nodes: a (possibly empty) numpy array of unique positive node ids each
        of which must be less than num_nodes. Each will have an array of grid_size
        associated with it. All others (up to num_nodes) will be associated with a single
        scalar value instead.
    :vartype nonfixed_nodes: numpy.ndarray
    :ivar timepoints: Array of time points
    :vartype timepoints: numpy.ndarray
    :ivar fill_value: What should we fill the data arrays with to start with
    :vartype fill_value: numpy.scalar
    """

    def __init__(self, num_nodes, nonfixed_nodes, timepoints,
                 fill_value=np.nan, dtype=FLOAT_DTYPE):
        """
        :param numpy.ndarray grid: The input numpy.ndarray.
        """
        if nonfixed_nodes.ndim != 1:
            raise ValueError("nonfixed_nodes must be a 1D numpy array")
        if np.any((nonfixed_nodes < 0) | (nonfixed_nodes >= num_nodes)):
            raise ValueError(
                "All non fixed node ids must be between zero and the total node number")
        grid_size = len(timepoints) if type(timepoints) is np.ndarray else timepoints
        self.timepoints = timepoints
        # Make timepoints immutable so no risk of overwritting them with copy
        self.timepoints.setflags(write=False)
        self.num_nodes = num_nodes
        self.nonfixed_nodes = nonfixed_nodes
        self.num_nonfixed = len(nonfixed_nodes)
        self.grid_data = np.full((self.num_nonfixed, grid_size), fill_value, dtype=dtype)
        self.fixed_data = np.full(num_nodes - self.num_nonfixed, fill_value, dtype=dtype)
        self.row_lookup = np.empty(num_nodes, dtype=np.int64)
        # non-fixed nodes get a positive value, indicating lookup in the grid_data array
        self.row_lookup[nonfixed_nodes] = np.arange(self.num_nonfixed)
        # fixed nodes get a negative value from -1, indicating lookup in the scalar array
        self.row_lookup[np.logical_not(np.isin(np.arange(num_nodes), nonfixed_nodes))] =\
            -np.arange(num_nodes - self.num_nonfixed) - 1
        self.probability_space = LIN

    def force_probability_space(self, probability_space):
        """
        probability_space can be "logarithmic" or "linear": this function will force
        the current probability space to the desired type
        """
        descr = self.probability_space, " probabilities into", probability_space, "space"
        if probability_space == LIN:
            if self.probability_space == LIN:
                pass
            elif self.probability_space == LOG:
                self.grid_data = np.exp(self.grid_data)
                self.fixed_data = np.exp(self.fixed_data)
                self.probability_space = LIN
            else:
                logging.warning("Cannot force", *descr)
        elif probability_space == LOG:
            if self.probability_space == LOG:
                pass
            elif self.probability_space == LIN:
                with np.errstate(divide='ignore'):
                    self.grid_data = np.log(self.grid_data)
                    self.fixed_data = np.log(self.fixed_data)
                self.probability_space = LOG
            else:
                logging.warning("Cannot force", *descr)
        else:
            logging.warning("Cannot force", *descr)

    def normalize(self):
        """
        normalize grid and fixed data so the max is one
        """
        rowmax = self.grid_data[:, 1:].max(axis=1)
        if self.probability_space == LIN:
            self.grid_data = self.grid_data / rowmax[:, np.newaxis]
        elif self.probability_space == LOG:
            self.grid_data = self.grid_data - rowmax[:, np.newaxis]
        else:
            raise RuntimeError("Probability space is not", LIN, "or", LOG)

    def __getitem__(self, node_id):
        index = self.row_lookup[node_id]
        if index < 0:
            return self.fixed_data[1 + index]
        else:
            return self.grid_data[index, :]

    def __setitem__(self, node_id, value):
        index = self.row_lookup[node_id]
        if index < 0:
            self.fixed_data[1 + index] = value
        else:
            self.grid_data[index, :] = value

    def clone_with_new_data(
            self, grid_data=np.nan, fixed_data=None, probability_space=None):
        """
        Take the row indices etc from an existing NodeGridValues object and make a new
        similar one but with different data. If grid_data is a single number, fill the
        entire data array with that, otherwise assume the data is a numpy array of the
        correct size to fill the gridded data. If grid_data is None, fill with NaN

        If fixed_data is None and grid_data is a single number, use the same value as
        grid_data for the fixed data values. If fixed_data is None and grid_data is an
        array, set the fixed data to np.nan
        """
        def fill_fixed(orig, fixed_data):
            if type(fixed_data) is np.ndarray:
                if orig.fixed_data.shape != fixed_data.shape:
                    raise ValueError(
                        "The fixed data array must be the same shape as the original")
                return fixed_data
            else:
                return np.full(
                    orig.fixed_data.shape, fixed_data, dtype=orig.fixed_data.dtype)
        new_obj = NodeGridValues.__new__(NodeGridValues)
        new_obj.num_nodes = self.num_nodes
        new_obj.nonfixed_nodes = self.nonfixed_nodes
        new_obj.num_nonfixed = self.num_nonfixed
        new_obj.row_lookup = self.row_lookup
        new_obj.timepoints = self.timepoints
        if type(grid_data) is np.ndarray:
            if self.grid_data.shape != grid_data.shape:
                raise ValueError(
                    "The grid data array must be the same shape as the original")
            new_obj.grid_data = grid_data
            new_obj.fixed_data = fill_fixed(
                self, np.nan if fixed_data is None else fixed_data)
        else:
            if grid_data == 0:  # Fast allocation
                new_obj.grid_data = np.zeros(
                    self.grid_data.shape, dtype=self.grid_data.dtype)
            else:
                new_obj.grid_data = np.full(
                    self.grid_data.shape, grid_data, dtype=self.grid_data.dtype)
            new_obj.fixed_data = fill_fixed(
                self, grid_data if fixed_data is None else fixed_data)
        if probability_space is None:
            new_obj.probability_space = self.probability_space
        else:
            new_obj.probability_space = probability_space
        return new_obj


def fill_prior(distr_parameters, timepoints, ts, nodes_to_date, prior_distr,
               progress=False):
    """
    Take the alpha and beta values from the distr_parameters data frame
    and fill out a NodeGridValues object with the prior values from the
    gamma or lognormal distribution with those parameters.

    TODO - what if there is an internal fixed node? Should we truncate
    """
    # Sort nodes-to-date by time, as that's the order given when iterating over edges
    prior_times = NodeGridValues(
        ts.num_nodes,
        nodes_to_date[np.argsort(ts.tables.nodes.time[nodes_to_date])].astype(np.int32),
        timepoints)
    if prior_distr == 'lognorm':
        cdf_func = scipy.stats.lognorm.cdf
        main_param = np.sqrt(distr_parameters[:, PriorParams.field_index('beta')])
        scale_param = np.exp(distr_parameters[:, PriorParams.field_index('alpha')])
    elif prior_distr == 'gamma':
        cdf_func = scipy.stats.gamma.cdf
        main_param = distr_parameters[:, PriorParams.field_index('alpha')]
        scale_param = 1 / distr_parameters[:, PriorParams.field_index('beta')]
    else:
        raise ValueError("prior distribution must be lognorm or gamma")

    for node in tqdm(nodes_to_date, desc="Assign prior to each node",
                     disable=not progress):
        prior_node = cdf_func(timepoints, main_param[node], scale=scale_param[node])
        # force age to be less than max value
        prior_node = np.divide(prior_node, np.max(prior_node))
        # prior in each epoch
        prior_times[node] = np.concatenate([np.array([0]), np.diff(prior_node)])
    # normalize so max value is 1
    prior_times.normalize()
    return prior_times


class Likelihoods:
    """
    A class to store and process likelihoods. Likelihoods for edges are stored as a
    flattened lower triangular matrix of all the possible delta t's. This class also
    provides methods for accessing this lower triangular matrix, multiplying it, etc.
    """
    probability_space = LIN
    identity_constant = 1.0
    null_constant = 0.0

    def __init__(self, ts, timepoints, theta=None, eps=0, fixed_node_set=None,
                 normalize=True, progress=False):
        self.ts = ts
        self.timepoints = timepoints
        self.fixednodes = set(ts.samples()) if fixed_node_set is None else fixed_node_set
        self.theta = theta
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
        return muts_span, Likelihoods._lik(muts_span[0], muts_span[1], dt, theta,
                                           normalize=normalize)

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
                (muts, edge_span(e)): None for muts, e in
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
                self._lik_wrapper, dt=self.timediff_lower_tri, theta=self.theta)
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
        return self._lik(mutations_on_edge, edge_span(edge), self.timediff, self.theta,
                         normalize=self.normalize)

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
        return self.unfixed_likelihood_cache[mutations_on_edge, edge_span(edge)]

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

    def _recombination_lik(self, rho, edge, fixed=True):
        # Needs to return a lower tri *or* flattened array depending on `fixed`
        raise NotImplementedError
        # return (
        #     np.power(prev_state, self.n_breaks(edge)) *
        #     np.exp(-(prev_state * rho * edge.span * 2)))

    def get_inside(self, arr, edge, theta=None, rho=None):
        liks = self.identity_constant
        if rho is not None:
            liks = self._recombination_lik(rho, edge)
        if theta is not None:
            liks *= self.get_mut_lik_lower_tri(edge)
        return self.rowsum_lower_tri(arr * liks)

    def get_outside(self, arr, edge, theta=None, rho=None):
        liks = self.identity_constant
        if rho is not None:
            liks = self._recombination_lik(rho, edge)
        if theta is not None:
            liks *= self.get_mut_lik_upper_tri(edge)
        return self.rowsum_upper_tri(arr * liks)

    def get_fixed(self, arr, edge, theta=None, rho=None):
        liks = self.identity_constant
        if rho is not None:
            liks = self._recombination_lik(rho, edge, fixed=True)
        if theta is not None:
            liks *= self.get_mut_lik_fixed_node(edge)
        return arr * liks

    def scale_geometric(self, fraction, value):
        return value ** fraction


class LogLikelihoods(Likelihoods):
    """
    Identical to the Likelihoods class but stores and returns log likelihoods
    """
    probability_space = LOG
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

    def _recombination_loglik(self, rho, edge, fixed=True):
        # Needs to return a lower tri *or* flattened array depending on `fixed`
        raise NotImplementedError
        # return (
        #     np.power(prev_state, self.n_breaks(edge)) *
        #     np.exp(-(prev_state * rho * edge.span * 2)))

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

    def get_inside(self, arr, edge, theta=None, rho=None):
        log_liks = self.identity_constant
        if rho is not None:
            log_liks = self._recombination_loglik(rho, edge)
        if theta is not None:
            log_liks += self.get_mut_lik_lower_tri(edge)
        return self.rowsum_lower_tri(arr + log_liks)

    def get_outside(self, arr, edge, theta=None, rho=None):
        log_liks = self.identity_constant
        if rho is not None:
            log_liks = self._recombination_loglik(rho, edge)
        if theta is not None:
            log_liks += self.get_mut_lik_upper_tri(edge)
        return self.rowsum_upper_tri(arr + log_liks)

    def get_fixed(self, arr, edge, theta=None, rho=None):
        log_liks = self.identity_constant
        if rho is not None:
            log_liks = self._recombination_loglik(rho, edge, fixed=True)
        if theta is not None:
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
    def __init__(self, ts, prior, lik, progress=False):
        self.ts = ts
        self.prior = prior
        self.nonfixed_nodes = prior.nonfixed_nodes
        self.lik = lik
        self.fixednodes = lik.fixednodes
        self.progress = progress
        # If necessary, convert prior to log space
        self.prior.force_probability_space(lik.probability_space)

        self.spans = np.bincount(
            self.ts.tables.edges.child,
            weights=self.ts.tables.edges.right - self.ts.tables.edges.left)
        self.spans = np.pad(self.spans, (0, self.ts.num_nodes-len(self.spans)))

        self.root_spans = defaultdict(float)
        for tree in self.ts.trees():
            # TODO - use new 'root_threshold=2' to avoid having to check isolated nodes
            n_roots_in_tree = 0
            for root in tree.roots:
                if tree_num_children(tree, root) == 0:
                    # Isolated node
                    continue
                n_roots_in_tree += 1
                if n_roots_in_tree > 1:
                    raise ValueError("Invalid tree sequence: tree {} has >1 root".format(
                        tree.index))
                self.root_spans[root] += tree.span
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
        child_edges = (self.ts.edge(i) for i in reversed(
            np.argsort(self.ts.tables.nodes.time[self.ts.tables.edges.child[:]])))
        return itertools.groupby(child_edges, operator.attrgetter('child'))

    def edges_by_child_then_parent_desc(self):
        """
        Return an itertools.groupby object of edges grouped by child in descending order
        of the time of the child, then by descending order of age of child
        """
        wtype = np.dtype([('Childage', 'f4'), ('Parentage', 'f4')])
        w = np.empty(
            len(self.ts.tables.nodes.time[self.ts.tables.edges.child[:]]), dtype=wtype)
        w['Childage'] = self.ts.tables.nodes.time[self.ts.tables.edges.child[:]]
        w['Parentage'] = self.ts.tables.nodes.time[self.ts.tables.edges.parent[:]]
        sorted_child_parent = (self.ts.edge(i) for i in reversed(
            np.argsort(w, order=('Childage', 'Parentage'))))
        return itertools.groupby(sorted_child_parent, operator.attrgetter('child'))

    # === MAIN ALGORITHMS ===

    def inside_pass(self, theta, rho, *, normalize=True, cache_inside=False,
                    progress=None):
        """
        Use dynamic programming to find approximate posterior to sample from
        """
        if progress is None:
            progress = self.progress
        inside = self.prior.clone_with_new_data(  # store inside matrix values
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
            val = self.prior[parent].copy()
            for edge in edges:
                spanfrac = edge_span(edge) / self.spans[edge.child]
                # Calculate vals for each edge
                if edge.child in self.fixednodes:
                    # NB: geometric scaling works exactly when all nodes fixed in graph
                    # but is an approximation when times are unknown.
                    daughter_val = self.lik.scale_geometric(spanfrac, inside[edge.child])
                    edge_lik = self.lik.get_fixed(daughter_val, edge, theta, rho)
                else:
                    daughter_val = self.lik.scale_geometric(
                        spanfrac, self.lik.make_lower_tri(inside[edge.child]))
                    edge_lik = self.lik.get_inside(daughter_val, edge, theta, rho)
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

    def outside_pass(
            self, theta, rho, *,
            normalize=False, progress=None, probability_space_returned=LIN):
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
            grid_data=0, probability_space=LIN)
        for root, span_when_root in self.root_spans.items():
            outside[root] = span_when_root / self.spans[root]
        outside.force_probability_space(self.inside.probability_space)

        for child, edges in tqdm(
                self.edges_by_child_desc(), desc="Outside",
                total=self.ts.num_edges, disable=not progress):
            if child in self.fixednodes:
                continue
            val = np.full(self.lik.grid_size, self.lik.identity_constant)
            for edge in edges:
                if edge.parent in self.fixednodes:
                    raise RuntimeError(
                        "Fixed nodes cannot currently be parents in the TS")
                # Geometric scaling works exactly for all nodes fixed in graph
                # but is an approximation when times are unknown.
                spanfrac = edge_span(edge) / self.spans[child]
                try:
                    inside_div_gi = self.lik.reduce(
                        self.inside[edge.parent], self.g_i[edge.id], div_0_null=True)
                except AttributeError:  # we haven't cached g_i so we recalculate
                    daughter_val = self.lik.scale_geometric(
                        spanfrac, self.lik.make_lower_tri(self.inside[edge.child]))
                    edge_lik = self.lik.get_inside(daughter_val, edge, theta, rho)
                    cur_g_i = self.lik.reduce(edge_lik, self.norm[child])
                    inside_div_gi = self.lik.reduce(
                        self.inside[edge.parent], cur_g_i, div_0_null=True)
                parent_val = self.lik.scale_geometric(
                    spanfrac,
                    self.lik.make_upper_tri(
                        self.lik.combine(outside[edge.parent], inside_div_gi)))
                edge_lik = self.lik.get_outside(parent_val, edge, theta, rho)
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

    def outside_maximization(self, theta, eps=1e-6, progress=None):
        if progress is None:
            progress = self.progress
        if not hasattr(self, "inside"):
            raise RuntimeError("You have not yet run the inside algorithm")
        maximized_node_times = np.zeros(self.ts.num_nodes, dtype='int')

        if self.lik.probability_space == LOG:
            poisson = scipy.stats.poisson.logpmf
        elif self.lik.probability_space == LIN:
            poisson = scipy.stats.poisson.pmf

        mut_edges = self.lik.mut_edges
        mrcas = np.where(np.isin(
            np.arange(self.ts.num_nodes), self.ts.tables.edges.child, invert=True))[0]
        for i in mrcas:
            if i not in self.fixednodes:
                maximized_node_times[i] = np.argmax(self.inside[i])

        for child, edges in tqdm(
                self.edges_by_child_then_parent_desc(), desc="Maximization",
                total=self.ts.num_edges, disable=not progress):
            if child in self.fixednodes:
                continue
            for edge_index, edge in enumerate(edges):
                if edge_index == 0:
                    youngest_par_index = maximized_node_times[edge.parent]
                    parent_time = self.lik.timepoints[maximized_node_times[edge.parent]]
                    ll_mut = poisson(
                        mut_edges[edge.id],
                        (parent_time - self.lik.timepoints[:youngest_par_index + 1] +
                            eps) * theta / 2 * edge_span(edge))
                    result = self.lik.reduce(ll_mut, np.max(ll_mut))
                else:
                    cur_parent_index = maximized_node_times[edge.parent]
                    if cur_parent_index < youngest_par_index:
                        youngest_par_index = cur_parent_index
                    parent_time = self.lik.timepoints[maximized_node_times[edge.parent]]
                    ll_mut = poisson(
                        mut_edges[edge.id],
                        (parent_time - self.lik.timepoints[:youngest_par_index + 1] +
                            eps) * theta / 2 * edge_span(edge))
                    result[:youngest_par_index + 1] = self.lik.combine(
                        self.lik.reduce(ll_mut[:youngest_par_index + 1],
                                        np.max(ll_mut[:youngest_par_index + 1])),
                        result[:youngest_par_index + 1])
            inside_val = self.inside[child][:(youngest_par_index + 1)]

            maximized_node_times[child] = np.argmax(self.lik.combine(
                result[:youngest_par_index + 1], inside_val))

        return self.lik.timepoints[np.array(maximized_node_times).astype('int')]


def posterior_mean_var(ts, timepoints, posterior, fixed_node_set=None):
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


def build_prior_grid(tree_sequence, timepoints=20, approximate_prior=None,
                     prior_distribution="lognorm", eps=1e-6, progress=False):
    """
    Create prior distribution for the age of each node and the discretised time slices at
    which to evaluate node age.

    :param TreeSequence tree_sequence: The input :class:`tskit.TreeSequence`, treated as
        undated
    :param int_or_array_like timepoints: The number of quantiles used to create the
        time slices, or manually-specified time slices as a numpy array
    :param bool approximate_prior: Whether to use a precalculated approximate prior or
        exactly calculate prior
    :param string prior_distr: What distribution to use to approximate the conditional
        coalescent prior. Can be "lognorm" for the lognormal distribution (generally a
        better fit, but slightly slower to calculate) or "gamma" for the gamma
        distribution (slightly faster, but a poorer fit for recent nodes). Default:
        "lognorm"
    :param float eps: Specify minimum distance separating points in the time grid. Also
        specifies the error factor in time difference calculations.
    :return: A prior object to pass to tsdate.date() containing prior values for
        inference and a discretised time grid
    :rtype:  tsdate.NodeGridValues Object
    """

    fixed_node_set = set(tree_sequence.samples())
    span_data = SpansBySamples(tree_sequence, fixed_node_set, progress=progress)
    nodes_to_date = span_data.nodes_to_date
    max_sample_size_before_approximation = None if approximate_prior is False else 1000

    if prior_distribution not in ('lognorm', 'gamma'):
        raise ValueError("prior distribution must be lognorm or gamma")

    base_priors = ConditionalCoalescentTimes(max_sample_size_before_approximation,
                                             prior_distribution)
    base_priors.add(len(fixed_node_set), approximate_prior)
    for total_fixed in span_data.total_fixed_at_0_counts:
        # For missing data: trees vary in total fixed node count => have different priors
        base_priors.add(total_fixed, approximate_prior)

    if isinstance(timepoints, int):
        if timepoints < 2:
            raise ValueError("You must have at least 2 time points")
        timepoints = create_timepoints(
            base_priors[tree_sequence.num_samples], prior_distribution, timepoints + 1)
    elif isinstance(timepoints, np.ndarray):
        try:
            timepoints = np.sort(timepoints.astype(FLOAT_DTYPE, casting='safe'))
        except TypeError:
            logging.debug("Timepoints array cannot be converted to float dtype")
        if len(timepoints) < 2:
            raise ValueError("You must have at least 2 time points")
        elif np.any(timepoints < 0):
            raise ValueError("Timepoints cannot be negative")
        elif np.any(np.unique(timepoints, return_counts=True)[1] > 1):
            raise ValueError("Timepoints cannot have duplicate values")
    else:
        raise ValueError("time_slices must be an integer or a numpy array of floats")

    prior_params = base_priors.get_mixture_prior_params(span_data)
    prior = fill_prior(prior_params, timepoints, tree_sequence, nodes_to_date,
                       prior_distribution, progress)
    return prior


def date(
        tree_sequence, Ne, mutation_rate=None, recombination_rate=None, prior=None, *,
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
    :param NodeGridValues prior: NodeGridValue object containing the prior and
        time points
    :param float eps: Specify minimum distance separating time points. Also specifies
        the error factor in time difference calculations.
    :param int num_threads: The number of threads to use. A simpler unthreaded algorithm
        is used unless this is >= 1 (default: None).
    :param string method: What estimation method to use: can be
        "inside_outside" (empirically better, theoretically problematic) or
        "maximization" (worse empirically, especially with a gamma approximated prior,
        but theoretically robust). Default: "inside_outside".
    :param bool probability_space: Should the internal algorithm save probabilities in
        "logarithmic" (slower, less liable to to overflow) or "linear" space (fast, may
        overflow). Default: "logarithmic"
    :param bool progress: Whether to display a progress bar.
    :return: A tree sequence with inferred node times in units of generations.
    :rtype: tskit.TreeSequence
    """
    dates, _, timepoints, eps, nds = get_dates(
        tree_sequence, Ne, mutation_rate, recombination_rate, prior, progress=progress,
        **kwargs)
    constrained = constrain_ages_topo(tree_sequence, dates, timepoints, eps, nds,
                                      progress)
    tables = tree_sequence.dump_tables()
    tables.nodes.time = constrained * 2 * Ne
    tables.sort()
    return tables.tree_sequence()


def get_dates(
        tree_sequence, Ne, mutation_rate=None, recombination_rate=None, prior=None, *,
        eps=1e-6, num_threads=None, method='inside_outside', outside_normalize=True,
        progress=False, cache_inside=False,
        probability_space=LOG):
    """
    Infer dates for the nodes in a tree sequence, returning an array of inferred dates
    for nodes, plus other variables such as the distribution of posterior probabilities
    etc. Parameters are identical to the date() method, which calls this method, then
    injects the resulting date estimates into the tree sequence

    :return: tuple(mn_post, posterior, timepoints, eps, nodes_to_date)
    """
    # Stuff yet to be implemented. These can be deleted once fixed
    if recombination_rate is not None:
        raise NotImplementedError(
            "Using the recombination clock is not currently supported"
            ". See https://github.com/awohns/tsdate/issues/5 for details")

    for sample in tree_sequence.samples():
        if tree_sequence.node(sample).time != 0:
            raise NotImplementedError(
                "Samples must all be at time 0")
    fixed_node_set = set(tree_sequence.samples())

    if prior is None:
        prior = build_prior_grid(tree_sequence, eps=eps, progress=progress)
    else:
        logging.info("Using user-specified prior")
        prior = prior

    theta = rho = None

    if mutation_rate is not None:
        theta = 4 * Ne * mutation_rate
    if recombination_rate is not None:
        rho = 4 * Ne * recombination_rate

    if probability_space != LOG:
        liklhd = Likelihoods(tree_sequence, prior.timepoints, theta, eps,
                             fixed_node_set, progress=progress)
    else:
        liklhd = LogLikelihoods(tree_sequence, prior.timepoints, theta, eps,
                                fixed_node_set, progress=progress)

    if theta is not None:
        liklhd.precalculate_mutation_likelihoods(num_threads=num_threads)

    dynamic_prog = InOutAlgorithms(tree_sequence, prior, liklhd, progress=progress)
    dynamic_prog.inside_pass(theta, rho, cache_inside=False)

    posterior = None
    if method == 'inside_outside':
        posterior = dynamic_prog.outside_pass(theta, rho, normalize=outside_normalize)
        mn_post, _ = posterior_mean_var(tree_sequence, prior.timepoints, posterior,
                                        fixed_node_set)
    elif method == 'maximization':
        if theta is not None:
            mn_post = dynamic_prog.outside_maximization(theta, eps)
        else:
            raise ValueError("Outside maximization method requires mutation rate")
    else:
        raise ValueError(
            "estimation method must be either 'inside_outside' or 'maximization'")

    return mn_post, posterior, prior.timepoints, eps, prior.nonfixed_nodes
