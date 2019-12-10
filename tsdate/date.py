# MIT License
#
# Copyright (c) 2019 Anthony Wilder Wohns
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
import functools

import tskit

import pandas as pd
import numpy as np
import scipy.stats
from scipy.special import comb
from tqdm import tqdm

FORMAT_NAME = "tsdate"
FORMAT_VERSION = [1, 0]

# Hack: monkey patches to allow tsdate to work with non-dev versions of tskit
# TODO - remove when tskit 0.2.4 is released
tskit.Edge.span = property(lambda edge: (edge.right - edge.left))  # NOQA


def gamma_approx(mean, variance):
    """
    Returns alpha and beta of a gamma distribution for a given mean and variance
    """

    return (mean ** 2) / variance, mean / variance


class ConditionalCoalescentTimes():
    """
    Make and store conditional coalescent priors
    """
    def __init__(self, precalc_approximation_n):
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

    def __getitem__(self, total_tips):
        """
        Return a pandas dataframe for the conditional number of total tips in the tree.
        Return a pandas dataframe of conditional prior on age of node
        """
        return self.prior_store[total_tips]

    def add(self, total_tips, approximate=None):
        """
        Create a pandas dataframe to lookup prior mean and variance of
        ages for nodes with descendant sample tips range from 2..``total_tips``
        given that the total number of tips in the coalescent tree is
        ``total_tips``. The array is indexed by (num_tips / total_tips).

        Note: estimated times are scaled by inputted Ne and are haploid
        """
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

        prior = pd.DataFrame(
            index=np.arange(1, total_tips + 1),
            columns=["Alpha", "Beta"], dtype=float)
        # prior.loc[1] is distribution of times of a "coalescence node" ending
        # in a single sample - equivalent to the time of the sample itself, so
        # it should have var = 0 and mean = sample.time
        # Setting alpha = 0 and beta = 1 sets mean (a/b) == var (a / b^2) == 0
        prior.loc[1] = [0, 1]

        if self.approximate:
            get_tau_var = self.tau_var_lookup
        else:
            get_tau_var = self.tau_var_exact

        all_tips = np.arange(2, total_tips + 1)
        variances = get_tau_var(total_tips, all_tips)
        for var, tips in zip(variances, all_tips):
            # NB: it should be possible to vectorize this in numpy
            expectation = self.tau_expect(tips, total_tips)
            alpha, beta = gamma_approx(expectation, var)
            prior.loc[tips] = [alpha, beta]
        prior.index.name = 'Num_Tips'
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


def get_mixture_prior(spans_by_samples, basic_priors):
    """
    Given an object that can be queried for tip weights for a node,
    and a set of conditional coalescent priors for different
    numbers of sample tips under a node, return the alpha and beta
    parameters of the gamma distribution that approximates the
    distribution of times for each node by mixing gamma distributions
    fitted to the basic_priors.

    :param .SpansBySamples spans_by_samples: An instance of the
        :class:`SpansBySamples` class that can be used to obtain
        weights for each .
    :param .ConditionalCoalescentTimes basic_priors: An instance of
        the :class:`ConditionalCoalescentTimes` class, which provides
        a set of dataframes containing the theoretical distribution of
        coalescent times conditioned on the numbers of tips under a node.
        This is used to obtain the node date priors for mixing.
    :return: A data frame giving the alpha and beta parameters for each
        node id in ``spans_by_samples.nodes_to_date``, which can be used
        to approximate the probabilities of times for that node used a
        gamma distribution.
    :rtype:  pandas.DataFrame
    """

    def mixture_expect_and_var(mixture):
        expectation = 0
        first = secnd = third = 0
        for N, tip_dict in mixture.items():
            cur_age_prior = basic_priors[N].loc[tip_dict.descendant_tips]
            alpha = cur_age_prior['Alpha'].values
            beta = cur_age_prior['Beta'].values

            # Expectation
            expectation += np.sum((alpha / beta) * tip_dict.weight)

            # Variance
            first += np.sum(alpha / (beta ** 2) * tip_dict.weight)
            secnd += np.sum((alpha / beta) ** 2 * tip_dict.weight)
            third += np.sum((alpha / beta) * tip_dict.weight) ** 2

        return expectation, first + secnd - third

    seen_mixtures = {}
    prior = pd.DataFrame(
        index=list(spans_by_samples.nodes_to_date),
        columns=["Alpha", "Beta"], dtype=float)
    for node in spans_by_samples.nodes_to_date:
        mixture = spans_by_samples.get_weights(node)
        if len(mixture) == 1:
            # The norm: this node spans trees that all have the same set of samples
            total_tips, weight_tuple = next(iter(mixture.items()))
            if len(weight_tuple.weight) == 1:
                d_tips = weight_tuple.descendant_tips[0]
                # This node is not a mixture - can use the standard coalescent prior
                prior.loc[node] = basic_priors[total_tips].loc[d_tips]
            elif len(weight_tuple.weight) <= 5:
                # Making mixture priors is a little expensive. We can help by caching
                # in those cases where we have only a few mixtures (arbitrarily set here
                # as <= 5 mixtures
                mixture_hash = (
                    total_tips,
                    weight_tuple.descendant_tips.tostring(),
                    weight_tuple.weight.tostring())
                if mixture_hash not in seen_mixtures:
                    prior.loc[node] = seen_mixtures[mixture_hash] = \
                        gamma_approx(*mixture_expect_and_var(mixture))
                else:
                    prior.loc[node] = seen_mixtures[mixture_hash]
            else:
                # a large number of mixtures in this node - don't bother caching
                prior.loc[node] = gamma_approx(*mixture_expect_and_var(mixture))
        else:
            # The node spans trees with multiple total tip numbers, don't use the cache
            prior.loc[node] = gamma_approx(*mixture_expect_and_var(mixture))

    prior.index.name = "Node"
    return prior


Weights = namedtuple('Weights', 'descendant_tips weight')


class SpansBySamples:
    """
    A class to calculate and return the genomic spans covered by each
    non-sample node, broken down by the number of samples that descend
    directly from that node. This is used to calculate the conditional
    coalescent prior. The main method is :meth:`normalised_spans`, which
    returns the spans for a node, normalised by the total span that that
    node covers in the tree sequence.

    :ivar tree_sequence: A reference to the tree sequence that was used to
        generate the spans and weights
    :vartype tree_sequence: tskit.TreeSequence
    :ivar num_samples_set: A numpy array of unique numbers which list,
        in no particular order, the various sample counts among the trees
        in this tree sequence. In the simplest case of a tree sequence with
        no missing data, all trees have the same count of numbers of samples,
        and there will be only a single number in this array, equal to
        :attr:`.tree_sequence.num_samples`. However, where samples contain
        :ref:`missing data <sec_data_model_missing_data>`,
        some trees will contain fewer sample nodes, so this array will also
        contain additional numbers, all of which will be less than
        :attr:`.tree_sequence.num_samples`.
    :vartype num_samples_set: numpy.ndarray (dtype=np.uint64)
    :ivar node_total_span: A numpy array of size :attr:`.tree_sequence.num_nodes`
        containing the genomic span covered by each node (including sample nodes)
    :vartype node_total_span: numpy.ndarray (dtype=np.uint64)
    :ivar nodes_to_date: An numpy array containing all the node ids in the tree
        sequence that we wish to date. These are usually all the non-sample nodes,
        and also provide the node numbers that are valid parameters for the
        :meth:`weights` method.
    :vartype nodes_to_date: numpy.ndarray (dtype=np.uint32)
    """
    def __init__(self, tree_sequence, fixed_node_set=None):
        """
        :param TreeSequence ts: The input :class:`tskit.TreeSequence`.
        :param set fixed_node_set: A set of all the samples in the tree sequence.
            This should be equivalent to ``set(ts.samples()``, but is provided
            as an optional parameter so that a pre-calculated set can be passed
            in, to save the expense of re-calculating it when setting up the
            class. If ``None`` (the default) a fixed_node_set will be constructed
            during initialization.
        """
        self.ts = tree_sequence
        self.fixed_node_set = set(self.ts.samples()) if fixed_node_set is None else \
            fixed_node_set
        # We will store the spans in here, and normalise them at the end
        self._spans = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))

        node_total_span, trees_with_undated, n_tips_per_tree = self.set_spans_initial()
        # Provide a set of the total_num_tips in different trees (used for missing data)
        self.num_samples_set = set(np.unique(n_tips_per_tree))
        # Provide the complete spans for each node, used e.g. for normalising
        self.node_total_span = node_total_span

        # Check for remaining undated nodes (all unary ones)
        if self.nodes_remain_to_date():
            self.set_spans_connected_unary(trees_with_undated, n_tips_per_tree)
        if self.nodes_remain_to_date():
            self.set_spans_unconnected_unary(trees_with_undated, n_tips_per_tree)

        self.finalise()

    def nodes_remaining_to_date(self):
        """
        Return a set of the node IDs that we want to date, but which haven't had a
        set of spans allocated which could be used to date the node.
        """
        return {n for n in range(self.ts.num_nodes) if not(
            n in self._spans or n in self.fixed_node_set)}

    def nodes_remain_to_date(self):
        """
        A more efficient version of nodes_remaining_to_date() that simply tells us if
        there are any more nodes that remain to date, but does not identify which ones
        """
        if self.ts.num_nodes - self.ts.num_samples - len(self._spans) != 0:
            # we should always have equal or fewer results than nodes to date
            assert len(self._spans) < self.ts.num_nodes - len(self.fixed_node_set)
            return True
        return False

    def set_spans_initial(self):
        """
        Returns a tuple of the span that each node covers, a list of the tree indices of
        trees that have undated nodes (used to quickly revist these trees later), and the
        number of valid samples (tips) in each tree.
        """
        # The following 3 variables will be returned by this function
        node_total_span = np.zeros(self.ts.num_nodes)
        trees_with_undated = []  # Used to revisit trees with nodes that need dating
        n_tips_per_tree = np.full(self.ts.num_trees, tskit.NULL, dtype=np.int64)

        # Some useful local tracking variables
        num_children = np.full(self.ts.num_nodes, 0, dtype=np.int32)
        # Store the last recorded genome position for this node.
        # If set to np.nan, this indicates that we are not currently tracking this node
        stored_pos = np.full(self.ts.num_nodes, np.nan)

        def save_to_span_data(prev_tree, node, num_fixed_treenodes):
            """
            A convenience function to save accumulated tracked node data at the current
            breakpoint. If this is a non-fixed node which needs dating, we save the
            span by # descendant tips into self._spans. If the node was skipped because
            it is a unary node at the top of the tree, return None
            """
            if np.isnan(stored_pos[node]):
                # Don't save ones that we aren't tracking
                return False
            coverage = prev_tree.interval[1] - stored_pos[node]
            node_total_span[node] += coverage
            if node in self.fixed_node_set:
                return True
            n_tips = prev_tree.num_samples(node)
            assert n_tips > 0
            if len(prev_tree.children(node)) > 1:
                # This is a coalescent node
                self._spans[node][num_fixed_treenodes][n_tips] += coverage
            else:
                # Treat unary nodes differently: mixture of coalescent nodes above+below
                top_node = prev_tree.parent(node)
                try:  # Find coalescent node above
                    while len(prev_tree.children(top_node)) == 1:
                        top_node = prev_tree.parent(top_node)
                except ValueError:  # Happens if we have hit the root
                    assert top_node == tskit.NULL
                    logging.debug(
                        "Unary node `{}` exists above highest coalescence in tree {}."
                        " Skipping for now".format(node, prev_tree.index))
                    return None
                # Half from the node above
                top_node_tips = prev_tree.num_samples(top_node)
                self._spans[node][num_fixed_treenodes][top_node_tips] += coverage/2
                # Half from the node below
                #  NB: coalescent node below should have same num_samples as this one
                self._spans[node][num_fixed_treenodes][n_tips] += coverage/2
            return True

        # We iterate over edge_diffs to calculate, as nodes change their descendant tips,
        # the genomic coverage for each node partitioned into descendant tip numbers
        edge_diff_iter = self.ts.edge_diffs()
        # There are 2 possible algorithms - the simplest is to find all the affected
        # nodes in the new tree, and use the constant-time "Tree.num_samples()"
        # to recalculate the number of samples under each affected tip.
        # The more complex one keeps track of the samples added and subtracted
        # each time, basically implementing the num_samples() method itself

        # Initialise with first tree
        for node in self.ts.first().nodes():
            stored_pos[node] = 0
        # Consume the first edge diff: normally this is when most edges come in
        num_fixed_treenodes = 0
        _, _, e_in = next(edge_diff_iter)
        for e in e_in:
            if e.child in self.fixed_node_set:
                num_fixed_treenodes += 1
            if e.parent != tskit.NULL:
                num_children[e.parent] += 1
        n_tips_per_tree[0] = num_fixed_treenodes

        # Iterate over trees and remaining edge diffs
        for prev_tree in self.ts.trees(sample_counts=True):
            try:
                # Get the edge diffs from the prev tree to the new tree
                _, e_out, e_in = next(edge_diff_iter)
            except StopIteration:
                # Last tree, save all the remaining nodes
                for node in prev_tree.nodes():
                    if save_to_span_data(prev_tree, node, num_fixed_treenodes) is None:
                        trees_with_undated.append(prev_tree.index)
                assert prev_tree.index == self.ts.num_trees - 1
                break

            fixed_nodes_out = set()
            fixed_nodes_in = set()
            disappearing_nodes = set()
            changed_nodes = set()
            for e in e_out:
                # Since a node only has one parent edge, any edge children going out
                # will be lost, unless they are reintroduced in the edges_in
                if e.child in self.fixed_node_set:
                    fixed_nodes_out.add(e.child)
                # No need to add the parents, as we'll traverse up the previous tree
                # from these points and be guaranteed to hit them too.
                changed_nodes.add(e.child)
                disappearing_nodes.add(e.child)
                if e.parent != tskit.NULL:
                    num_children[e.parent] -= 1
                    if num_children[e.parent] == 0:
                        disappearing_nodes.add(e.parent)

            for e in e_in:
                # Edge children are always new
                if e.child in self.fixed_node_set:
                    fixed_nodes_in.add(e.child)
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
                    if save_to_span_data(prev_tree, node, num_fixed_treenodes) is None:
                        trees_with_undated.append(prev_tree.index)
                    if node in disappearing_nodes:
                        # Not tracking this in the future
                        stored_pos[node] = np.nan
                    else:
                        stored_pos[node] = prev_tree.interval[1]
                    node = prev_tree.parent(node)

            # If total number of samples has changed: we need to save
            #  everything so far & reset all starting positions
            if len(fixed_nodes_in) != len(fixed_nodes_out):
                for node in prev_tree.nodes():
                    if node in visited_nodes:
                        # We have already saved these - no need to again
                        continue
                    if save_to_span_data(prev_tree, node, num_fixed_treenodes) is None:
                        trees_with_undated.append(prev_tree.index)
                    if node in disappearing_nodes:
                        # Not tracking this in the future
                        stored_pos[node] = np.nan
                    else:
                        stored_pos[node] = prev_tree.interval[1]
                num_fixed_treenodes += len(fixed_nodes_in) - len(fixed_nodes_out)
            n_tips_per_tree[prev_tree.index+1] = num_fixed_treenodes

        return node_total_span, trees_with_undated, n_tips_per_tree

    def set_spans_connected_unary(self, trees_with_undated, n_tips_per_tree):
        """
        Check for nodes with unassigned prior params. We should see
        if can we assign params for these node priors using
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
        for tree_id in trees_with_undated:
            while tree.index != tree_id:
                tree = next(tree_iter)
            for node in unassigned_nodes:
                if tree.parent(node) == tskit.NULL:
                    continue
                    # node is either the root or (more likely) not in
                    # this tree
                assert tree.num_samples(node) > 0
                assert len(tree.children(node)) == 1
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
                            if k == 0:
                                raise ValueError("Oh dear 2")
                            local_weight = v / self.node_total_span[n]
                            self._spans[node][n_tips][k] += tree.span * local_weight / 2
                    assert len(tree.children(node)) == 1
                    total_tips = n_tips_per_tree[tree_id]
                    desc_tips = tree.num_samples(node)
                    self._spans[node][total_tips][desc_tips] += tree.span / 2
                    self.node_total_span[node] += tree.span

    def set_spans_unconnected_unary(self, trees_with_undated, n_tips_per_tree):
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
        for tree_id in trees_with_undated:
            while tree.index != tree_id:
                tree = next(tree_iter)
            for node in unassigned_nodes:
                if tree.is_internal(node):
                    assert len(tree.children(node)) == 1
                    total_tips = n_tips_per_tree[tree_id]
                    # above, we set the maximum
                    self._spans[node][max_samples][max_samples] += tree.span / 2
                    # below, we do as before
                    desc_tips = tree.num_samples(node)
                    self._spans[node][total_tips][desc_tips] += tree.span / 2
                    self.node_total_span[node] += tree.span

    def finalise(self):
        """
        Normalise the spans in self._spans by the values in self.node_total_span,
        and overwrite the results (as we don't need them any more), providing a
        shortcut to by setting normalised_node_spans. Also provide the nodes_to_date
        value.
        """
        assert not hasattr(self, 'normalised_node_spans')  # Check not already finalised
        if self.nodes_remain_to_date():
            raise ValueError(
                "When finalising node spans, found the following nodes not in any tree;"
                " try simplifing your tree sequence: {}"
                .format(self.nodes_remaining_to_date()))

        for node, weights_by_total_tips in self._spans.items():
            self._spans[node] = {}  # Overwrite, so we don't leave the old data around
            for num_samples, weights in sorted(weights_by_total_tips.items()):
                self._spans[node][num_samples] = Weights(
                    # we use np.int64 as it's faster to look up in pandas dataframes
                    descendant_tips=np.array(list(weights.keys()), dtype=np.int64),
                    weight=np.array(list(weights.values()))/self.node_total_span[node])
        # Assign into the instance, for further reference
        self.normalised_node_spans = self._spans
        self.nodes_to_date = np.array(list(self._spans.keys()), dtype=np.uint64)

    def get_weights(self, node):
        """
        Access the main calculated results from this class, returning weights
        for a node contained within a dict of dicts. Weights for each node
        (i.e. normalised genomic spans) sum to one, and are used to construct
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
            span, normalised by the total span over which the node exists) for
            :math:`k` descendant samples, as a floating point number. For any node,
            the normalisation means that all the weights should sum to one.
        :rtype: dict(int, dict(int, float))'
        """
        return self.normalised_node_spans[node]

    def lookup_weight(self, node, total_tips, descendant_tips):
        # Only used for testing
        which = self.get_weights(node)[total_tips].descendant_tips == descendant_tips
        return self.get_weights(node)[total_tips].weight[which]


def create_time_grid(age_prior, n_points=21):
    """
    Create the time grid by finding union of the quantiles of the gammas
    For a node with k descendants we have gamma approxs.
    Natural grid would be to take all the distributions
    quantile them up, and then take the union of the quantiles.
    Then thin this, making it no more than 0.05 of a quantile apart.
    Takes all the gamma distributions, finds quantiles, takes union,
    and thins them. Does this in an iterative way.
    """
    # Percentages - current day samples should be at time 0, so we omit this
    # We can't include the top end point, as this leads to NaNs
    percentiles = np.linspace(0, 1, n_points + 1)[1:-1]
    # percentiles = np.append(percentiles, 0.999999)
    """
    get the set of times from gamma percent point function at the given
    percentiles specifies the value of the RV such that the prob of the var
    being less than or equal to that value equals the given probability
    """
    t_set = scipy.stats.gamma.ppf(percentiles, age_prior.loc[2, "Alpha"],
                                  scale=1 / age_prior.loc[2, "Beta"])

    # progressively add values to the grid
    max_sep = 1.0 / (n_points - 1)
    if age_prior.shape[0] > 2:
        for i in np.arange(3, age_prior.shape[0] + 1):
            # gamma percentiles of existing times in grid
            proj = scipy.stats.gamma.cdf(t_set, age_prior.loc[i, "Alpha"],
                                         scale=1 / age_prior.loc[i, "Beta"])
            """
            thin the grid, only add additional quantiles if they're more than
            a certain max_sep fraction (e.g. 0.05) from another quantile
            """
            tmp = np.asarray([min(abs(val - proj)) for val in percentiles])
            wd = np.where(tmp > max_sep)

            if len(wd[0]) > 0:
                t_set = np.concatenate(
                    [t_set, np.array(
                        scipy.stats.gamma.ppf(
                            percentiles[wd], age_prior.loc[i, "Alpha"],
                            scale=1 / age_prior.loc[i, "Beta"]))])

    t_set = sorted(t_set)
    return np.insert(t_set, 0, 0)


def iterate_parent_edges(ts):
    if ts.num_edges > 0:
        # Fix this when reversed iterator is merged to main tskit
        # tskit github issue #304
        # but instead should iterate over edge ids in reverse
        # i.e. edge_ids = list(reversed(range(ts.num_edges)))
        all_edges = list(ts.edges())
        parent_edges = all_edges[:1]
        parent_edges[0] = (0, parent_edges[0])
        cur_parent = all_edges[:1][0].parent
        last_parent = -1
        for index, edge in enumerate(all_edges[1:]):
            if edge.parent != last_parent and edge.parent != cur_parent:
                yield parent_edges
                cur_parent = edge.parent
                parent_edges = []
            parent_edges.append((index + 1, edge))
        yield parent_edges


def get_prior_values(mixture_prior, grid, ts, nodes_to_date):
    prior_times = np.zeros((ts.num_nodes, len(grid)))
    prior_times[:, 0] = 1
    for node in nodes_to_date:
        prior_node = scipy.stats.gamma.cdf(
            grid, mixture_prior.loc[node, "Alpha"],
            scale=1 / mixture_prior.loc[node, "Beta"])
        # force age to be less than max value
        prior_node = np.divide(prior_node, max(prior_node))
        # prior in each epoch
        prior_intervals = np.concatenate(
            [np.array([0]), np.diff(prior_node)])

        # normalize so max value is 1
        prior_intervals = np.divide(prior_intervals, max(prior_intervals[1:]))
        prior_times[node, :] = prior_intervals
    return prior_times


class Likelihoods:
    """
    A class to store the likelihoods for edges. These are stored as a flattened
    lower triangular matrix of all the possible delta t's. This class also provides
    methods for accessing this lower triangular matrix, multiplying it, etc.
    """
    def __init__(self, ts, grid, eps):
        self.ts = ts
        self.grid = grid
        self.grid_size = len(grid)
        self.tri_size = self.grid_size * (self.grid_size + 1) / 2
        self.ll_mut = {}
        self.mut_edges = self.get_mut_edges(ts)
        self.timediff_lower_tri = np.concatenate(
            [self.grid[time] - self.grid[0:time + 1] + eps
                for time in np.arange(len(self.grid))])

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
                # These don't provide any information for the upward step
                if m.node in edges_by_child:
                    edge_id = edges_by_child[m.node]
                    mut_edges[edge_id] += 1
        return mut_edges

    @staticmethod
    def _lik(muts_span, dt, theta):
        # Has to be a static function to allow multiprocessing
        return (
            muts_span,
            scipy.stats.poisson.pmf(muts_span[0], dt * (theta / 2 * (muts_span[1]))))

    def precalculate_mutation_likelihoods(
            self, theta, num_threads=1, unique_method=0):
        """
        We precalculate these because the pmf function is slow, but can be trivially
        parallelised
        """
        if unique_method == 0:
            self.likelihood_for_n_muts_and_span = {
                (muts, e.right-e.left): None for muts, e in
                zip(self.mut_edges, self.ts.edges())}
        elif unique_method == 1:
            edges = self.ts.tables.edges
            keys = np.unique(
                np.core.records.fromarrays(
                    (self.mut_edges, edges.left-edges.right), names='muts,span'))
            self.likelihood_for_n_muts_and_span = dict.fromkeys({tuple(t) for t in keys})
            # self.likelihood_for_n_muts_and_span = {tuple(t): None for t in keys}

        if num_threads > 1:
            with multiprocessing.Pool(processes=num_threads) as pool:
                # Use functools.partial to set constant values for params to get_ll
                f = functools.partial(self._lik, dt=self.timediff_lower_tri, theta=theta)
                for key, pmf in pool.imap_unordered(
                        f, self.likelihood_for_n_muts_and_span.keys()):
                    self.likelihood_for_n_muts_and_span[key] = pmf
        else:
            for key in self.likelihood_for_n_muts_and_span.keys():
                self.likelihood_for_n_muts_and_span[key] = self._lik(
                    key, dt=self.timediff_lower_tri, theta=theta)[1]

    def get_mut_lik_lower_tri(self, edge):
        """
        Get the mutation likelihoods for an edge, for all the possible time diffences
        between times on the grid, and cache them by using the number of mutations and
        span as the key. If we ask again for set of likelihoods with the same # mutations
        and span, simply return the cached value.

        The time differences used in the likelihoods are ordered as a flattened lower
        triangular matrix, the form required in the upward algorithm.

        TO DO - return different likelihoods for sample edges. These can be an array of
        a much shorter length
        """
        mutations_on_edge = self.mut_edges[edge.id]
        try:
            return self.likelihood_for_n_muts_and_span[mutations_on_edge, edge.span]
        except AttributeError:
            raise RuntimeError("You must call `precalculate_mutation_likelihoods` first")

    def get_mut_lik_upper_tri(self, edge):
        """
        Same as :meth:`get_mut_lik_lower_tri`, but the returned array is ordered as
        flattened upper triangular matrix (suitable for the downward algorithm), rather
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


def upward_algorithm(ts, prior_values, theta, rho, lls, return_log=True, progress=False):
    """
    Use dynamic programming to find approximate posterior to sample from
    """

    upward = np.zeros((ts.num_nodes, lls.grid_size))  # store upward matrix
    g_i = np.zeros((ts.num_nodes, lls.grid_size))  # store g of i

    # initialize tips at time 0 to prob=1
    # TODO - account for ancient samples at different ages
    upward[ts.samples(), 0] = 1
    g_i[ts.samples(), 0] = 1

    # Iterate through the nodes via groupby on parent node
    for parent_group in tqdm(
        iterate_parent_edges(ts), total=ts.num_nodes - ts.num_samples,
            disable=not progress):
        """
        for each node, find the conditional prob of age at every time
        in time grid
        """
        parent = parent_group[0][1].parent
        val = prior_values[parent].copy()
        g_val = np.ones(lls.grid_size)
        g_val[0] = 0
        for edge_index, edge in parent_group:
            # Calculate vals for each edge
            dt = lls.make_lower_tri(upward[edge.child])
            if theta is not None and rho is not None:
                b_l = (edge.left != 0)
                b_r = (edge.right != ts.get_sequence_length())
                ll_rec = np.power(
                    dt, b_l + b_r) * np.exp(-(dt * rho * edge.span * 2))
                ll_mut = lls.get_mut_lik_lower_tri(edge)
                vv = lls.rowsum_lower_tri(ll_mut * ll_rec * dt)
                val *= vv
                g_val *= vv
            elif theta is not None:
                ll_mut = lls.get_mut_lik_lower_tri(edge)
                vv = lls.rowsum_lower_tri(ll_mut * dt)
                val *= vv
                g_val *= vv
            elif rho is not None:
                b_l = (edge.left != 0)
                b_r = (edge.right != ts.get_sequence_length())
                ll_rec = np.power(
                    dt, b_l + b_r) * np.exp(-(dt * rho * edge.span * 2))
                vv = lls.rowsum_lower_tri(ll_rec * dt)
                val *= vv
                g_val *= vv
            else:
                # Topology-only clock
                vv = lls.rowsum_lower_tri(dt)
                val *= vv
                g_val *= vv

        upward[parent] = val / np.max(val)
        g_i[parent] = g_val / np.max(g_val)
    if return_log:
        return np.log(upward + 1e-10), np.log(g_i + 1e-10)
    else:
        return upward, g_i


# TODO: Account for multiple parents, fix the log of zero thing
def downward_algorithm(
        ts, log_upward, log_g_i, theta, rho, lls, spans,
        dated_node_set=None):
    """
    Computes the full posterior distribution on nodes.
    Input is log of upward matrix, log of g_i matrix, time grid, population scaled
    mutation and recombination rate. Spans of each edge, epsilon, likelihoods of each
    edge, and the columns of the lower triangular matrix of likelihoods.
    """
    if dated_node_set is None:
        dated_node_set = set(ts.samples())
    node_has_date = np.zeros(ts.num_nodes, dtype=bool)
    node_has_date[list(dated_node_set)] = True
    downward = np.zeros((ts.num_nodes, lls.grid_size))  # store downward matrix
    norm = np.zeros((ts.num_nodes))  # normalizing constants
    norm[node_has_date] = 1  # normalizing constants of all nodes with known dates == 1

    for tree in ts.trees():
        for root in tree.roots:
            if len(tree.get_children(root)) == 0:
                print("Node not in tree")
                continue
            downward[root, 1:] += (1 * tree.span) / spans[root]
    downward[node_has_date, 0] = 1

    child_edges = (ts.edge(i) for i in
                   reversed(np.argsort(ts.tables.nodes.time[ts.tables.edges.child[:]])))

    for child, edges in tqdm(
            itertools.groupby(child_edges, key=lambda x: x.child),
            total=ts.num_nodes - np.sum(node_has_date)):
        if child not in dated_node_set:

            edges = list(edges)
            for edge in edges:
                dt = lls.make_upper_tri(
                    downward[edge.parent] *
                    np.exp(np.subtract(log_upward[edge.parent], log_g_i[edge.child])))
                if theta is not None and rho is not None:
                    b_l = (edge.left != 0)
                    b_r = (edge.right != ts.get_sequence_length())
                    ll_rec = np.power(
                        dt, b_l + b_r) * np.exp(-(dt * rho * edge.span * 2))
                    ll_mut = lls.get_mut_lik_upper_tri(edge)
                    vv = lls.rowsum_upper_tri(dt * ll_mut * ll_rec)
                elif theta is not None:
                    ll_mut = lls.get_mut_lik_upper_tri(edge)
                    vv = lls.rowsum_upper_tri(dt * ll_mut)
                elif rho is not None:
                    b_l = (edge.left != 0)
                    b_r = (edge.right != ts.get_sequence_length())
                    ll_rec = np.power(
                        dt, b_l + b_r) * np.exp(-(dt * rho * edge.span * 2))
                    vv = lls.rowsum_upper_tri(dt * ll_rec)
                else:
                    # Topology-only clock
                    vv = lls.rowsum_upper_tri(dt)
            downward[edge.child, 1:] = vv[1:] / max(vv[1:])
            downward[edge.child, np.isnan(downward[edge.child, :])] = 0
    log_downward = np.log(downward)
    posterior = np.exp(log_upward + log_downward)
    posterior = posterior / np.sum(posterior, axis=1)[:, None]
    return posterior, downward


def posterior_mean_var(ts, grid, upward, fixed_nodes=None, nodes_to_date=None):
    """
    Mean and variance of node age in scaled time
    If nodes_to_date is None, we attempt to date all the non-sample nodes
    """
    mn_post = np.zeros(ts.num_nodes)
    vr_post = np.zeros(ts.num_nodes)
    if nodes_to_date is None:
        nodes_to_date = np.arange(ts.num_nodes, dtype=np.uint64)
        nodes_to_date = nodes_to_date[~np.isin(nodes_to_date, ts.samples())]

    for nd in nodes_to_date:
        mn_post[nd] = sum(upward[nd, ] * grid) / sum(upward[nd, ])
        vr_post[nd] = (
            sum(upward[nd, ] * grid ** 2) /
            sum(upward[nd, ]) - mn_post[nd] ** 2)
    return mn_post, vr_post


def restrict_ages_topo(ts, upward_mn, grid, eps, nodes_to_date=None):
    """
    If predicted node times violate topology, restrict node ages so that they
    must be older than all their children.
    """
    new_mn_post = np.copy(upward_mn)
    if nodes_to_date is None:
        nodes_to_date = np.arange(ts.num_nodes, dtype=np.uint64)
        nodes_to_date = nodes_to_date[~np.isin(nodes_to_date, ts.samples())]

    tables = ts.tables
    for nd in sorted(nodes_to_date):
        children = tables.edges.child[tables.edges.parent == nd]
        time = new_mn_post[children]
        if np.any(new_mn_post[nd] <= time):
            # closest_time = (np.abs(grid - max(time))).argmin()
            # new_mn_post[nd] = grid[closest_time] + eps
            new_mn_post[nd] = max(time) + eps
    return new_mn_post


def return_ts(ts, vals, Ne):
    """
    Output new inferred tree sequence with node ages assigned.
    """
    tables = ts.dump_tables()
    tables.nodes.time = vals * 2 * Ne
    tables.sort()
    return tables.tree_sequence()


def date(
        tree_sequence, Ne, mutation_rate=None, recombination_rate=None,
        time_grid='adaptive', grid_slices=50, eps=1e-6, num_threads=0,
        approximate_prior=None, progress=False):
    """
    Take a tree sequence with arbitrary node times and recalculate node times using
    the `tsdate` algorithm. If both a mutation_rate and recombination_rate are given, a
    joint mutation and recombination clock is used to date the tree sequence. If only
    mutation_rate is given, only the mutation clock is used; similarly if only
    recombination_rate is given, only the recombination clock is used. If neither are
    given, a topology-only clock is used (**details***).

    :param TreeSequence tree_sequence: The input :class:`tskit.TreeSequence`, treated as
        undated.
    :param float Ne: The estimated effective population size
    :param float mutation_rate: The estimated mutation rate per unit of genome. If
        provided, the dating algorithm will use a mutation rate clock to help estimate
        node dates
    :param float recombination_rate: The estimated recombination rate per unit of genome.
        If provided, the dating algorithm will use a recombination rate clock to help
        estimate node dates
    :param string time_grid: How to space out the time grid. Currently can be either
        "adaptive" (the default) or "uniform". The adaptive time grid spaces out time
        points in even quantiles over the expected prior.
    :param int grid_slices: The number of slices used in the time grid
    :param float eps: The precision required (** deeper explanation required **)
    :param int num_threads: The number of threads to use. If
        this is <= 0 then a simpler sequential algorithm is used (default).
    :param bool approximate_prior: Whether to use a precalculated approximate prior or
        exactly calculate prior
    :param bool progress: Whether to display a progress bar.
    :return: A tree sequence with inferred node times.
    :rtype: tskit.TreeSequence
    """
    if grid_slices < 2:
        raise ValueError("You must have at least 2 slices in the time grid")

    # Stuff yet to be implemented. These can be deleted once fixed
    if recombination_rate is not None:
        raise NotImplementedError(
            "Using the recombination clock is not currently supported"
            ". See https://github.com/awohns/tsdate/issues/5 for details")
    if num_threads > 0:
        raise NotImplementedError(
            "Using multiple threads is not yet implemented")

    for sample in tree_sequence.samples():
        if tree_sequence.node(sample).time != 0:
            raise NotImplementedError(
                "Samples must all be at time 0")

    fixed_node_set = set(tree_sequence.samples())

    span_data = SpansBySamples(tree_sequence, fixed_node_set)
    spans = span_data.node_total_span
    nodes_to_date = span_data.nodes_to_date
    max_sample_size_before_approximation = None if approximate_prior is False else 1000
    priors = ConditionalCoalescentTimes(max_sample_size_before_approximation)
    priors.add(tree_sequence.num_samples, approximate_prior)
    # Add in priors for trees with different sample numbers (missing data only)
    for num_samples in span_data.num_samples_set:
        if num_samples != tree_sequence.num_samples:
            priors.add(num_samples, approximate_prior)

    if time_grid == 'uniform':
        grid = np.linspace(0, 8, grid_slices + 1)
    elif time_grid == 'adaptive':
        # Use the prior for the complete TS
        grid = create_time_grid(
            priors[tree_sequence.num_samples], grid_slices + 1)
    else:
        raise ValueError("time_grid must be either 'adaptive' or 'uniform'")

    mixture_prior = get_mixture_prior(span_data, priors)
    prior_vals = get_prior_values(mixture_prior, grid, tree_sequence, nodes_to_date)

    theta = rho = None
    liklhd = Likelihoods(tree_sequence, grid, eps)

    if mutation_rate is not None:
        theta = 4 * Ne * mutation_rate
        liklhd.precalculate_mutation_likelihoods(theta, num_threads=num_threads)
    if recombination_rate is not None:
        rho = 4 * Ne * recombination_rate

    logged_upward, logged_g_i = upward_algorithm(
        tree_sequence, prior_vals, theta, rho, liklhd,
        progress=progress)
    posterior, downward = downward_algorithm(
        tree_sequence, logged_upward, logged_g_i,
        theta, rho, liklhd, spans, fixed_node_set)
    mn_post, _ = posterior_mean_var(tree_sequence, grid, posterior,
                                    nodes_to_date=nodes_to_date)
    new_mn_post = restrict_ages_topo(tree_sequence, mn_post, grid, eps,
                                     nodes_to_date=nodes_to_date)
    dated_ts = return_ts(tree_sequence, new_mn_post, Ne)
    return dated_ts
