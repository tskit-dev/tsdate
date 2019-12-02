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

import pandas as pd
import numpy as np
import scipy.stats
from scipy.special import comb
from tqdm import tqdm

import tskit

FORMAT_NAME = "tsdate"
FORMAT_VERSION = [1, 0]


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
        mixture = spans_by_samples.weights(node)
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

    def __init__(self, ts, fixed_node_set=None):
        """
        :param TreeSequence ts: The input :class:`tskit.TreeSequence`.
        :param set fixed_node_set: A set of all the samples in the tree sequence.
            This should be equivalent to ``set(ts.samples()``, but is provided
            as an optional parameter so that a pre-calculated set can be passed
            in, to save the expense of re-calculating it when setting up the
            class. If ``None`` (the default) a fixed_node_set will be constructed
            during initialization.
        """
        self.tree_sequence = ts
        if fixed_node_set is None:
            fixed_node_set = set(ts.samples())
        valid_samples_in_tree = np.full(ts.num_trees, tskit.NULL, dtype=np.int64)
        num_children = np.full(ts.num_nodes, 0, dtype=np.int32)
        self.num_samples_set = set()  # The num_samples in different trees (missing data)
        self.node_total_span = np.zeros(ts.num_nodes)
        result = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))

        # Store the last recorded genome position for this node. If set to np.nan, this
        # indicates that we are not currently tracking this node
        stored_pos = np.full(ts.num_nodes, np.nan)
        trees_with_unassigned_nodes = set()  # Used to quickly skip trees later

        # Use edge_diffs to calculate genomic coverage of each node, partitioned
        # into different numbers of descendant tips.

        for node in ts.first().nodes():
            stored_pos[node] = 0

        num_fixed_treenodes = 0
        edge_diff_iter = ts.edge_diffs()
        _, _, e_in = next(edge_diff_iter)  # all edges come in to make a tree
        for e in e_in:
            if e.child in fixed_node_set:
                num_fixed_treenodes += 1
        self.num_samples_set.add(num_fixed_treenodes)
        valid_samples_in_tree[0] = num_fixed_treenodes

        # There are 2 possible algorithms - the simplest is to find all the affected
        # nodes in the new tree, and use the constant-time "Tree.num_samples()"
        # to recalculate the number of samples under each affected tip.
        # The more complex one keeps track of the samples added and subtracted
        # each time, basically implementing the num_samples() method itself
        def save_to_results(tree, node, num_fixed_treenodes):
            if np.isnan(stored_pos[node]):
                # Don't save ones that we aren't tracking
                return
            coverage = tree.interval[1] - stored_pos[node]
            self.node_total_span[node] += coverage
            if node in fixed_node_set:
                return
            # print("Saving node: ", node)
            n_tips = tree.num_samples(node)
            if len(tree.children(node)) == 1:
                # print("UNARY NODE DETECTED")
                # Unary nodes are treated differently: we need to
                # take a mixture of the coalescent nodes above and below
                #  ABOVE:
                n = node
                done = False
                while not done:
                    n = tree.parent(n)
                    if n == tskit.NULL or len(tree.children(n)) > 1:
                        done = True  # Found a coalescent node
                if n == tskit.NULL:
                    logging.debug(
                        "Unary node {} exists above highest coalescence in tree {}."
                        " Skipping for now".format(node, i))
                    trees_with_unassigned_nodes.add(i)
                    return
                # Half from the node above
                result[node][num_fixed_treenodes][tree.num_samples(n)] += coverage / 2
                #  BELOW: coalescent node below should have same num_samples as this one
                result[node][num_fixed_treenodes][n_tips] += coverage / 2
            else:
                result[node][num_fixed_treenodes][n_tips] += coverage

        # START ALGORITHM HERE
        for i, tree in enumerate(ts.trees(sample_counts=True)):
            # print(">>> TREE", i, tree.draw_text(), "\nTracking {}".format(
            #    np.where(stored_pos != np.nan)[0]))
            try:
                # Get the edge diffs from the current tree to the new tree
                _, e_out, e_in = next(edge_diff_iter)
            except StopIteration:
                # Last tree, save all the remaining nodes
                for node in tree.nodes():
                    save_to_results(tree, node, num_fixed_treenodes)
                break

            fixed_nodes_out = set()
            fixed_nodes_in = set()
            disappearing_nodes = set()
            changed_nodes = set()
            for e in e_out:
                # Since a node only has one parent edge, any edge children going out
                # will be lost, unless they are reintroduced
                if e.child in fixed_node_set:
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
                if e.child in fixed_node_set:
                    fixed_nodes_in.add(e.child)
                # This may have changed in the new tree
                changed_nodes.add(e.child)
                disappearing_nodes.discard(e.child)
                if e.parent != tskit.NULL:
                    # parent nodes might be added in the next tree, and we won't
                    # necessarily traverse up to them in the current tree, so they need
                    # to be added to the possibly changing nodes
                    changed_nodes.add(e.parent)
                    # If a parent or child come in, they definitely won't be disappearing
                    num_children[e.parent] += 1
                    disappearing_nodes.discard(e.parent)

            # Add unary nodes below the altered ones, as their result is calculated
            # from the coalescent node above
            unary_descendants = set()
            for node in changed_nodes:
                children = tree.children(node)
                if children is not None:
                    if len(children) == 1:
                        # Keep descending
                        node == children[0]
                        while True:
                            children = tree.children(node)
                            if len(children) != 1:
                                break
                            unary_descendants.add(node)
                            node = children[0]
                    else:
                        # Descend all branches, looking for unary nodes
                        for node in tree.children(node):
                            while True:
                                children = tree.children(node)
                                if len(children) != 1:
                                    break
                                unary_descendants.add(node)
                                node = children[0]

            # find all the nodes in the tree that might have changed their number
            # of descendants, and reset. This might include nodes that are not in
            # the current tree, but will be in the next one (so we still need to
            # set the stored position). Also track visited_nodes so we don't repeat
            visited_nodes = set()
            for node in changed_nodes | unary_descendants:
                while node != tskit.NULL:  # if root or node not in tree
                    if node in visited_nodes:
                        break
                    visited_nodes.add(node)
                    # Node is in current tree
                    save_to_results(tree, node, num_fixed_treenodes)
                    if node in disappearing_nodes:
                        # Not tracking this in the future
                        stored_pos[node] = np.nan
                    else:
                        stored_pos[node] = tree.interval[1]
                    node = tree.parent(node)

            # If total number of samples has changed: we need to save
            #  everything so far & reset all starting positions
            if len(fixed_nodes_in) != len(fixed_nodes_out):
                for node in tree.nodes():
                    if node in visited_nodes:
                        # We have already saved these - no need to again
                        continue
                    save_to_results(tree, node, num_fixed_treenodes)
                    if node in disappearing_nodes:
                        # Not tracking this in the future
                        stored_pos[node] = np.nan
                    else:
                        stored_pos[node] = tree.interval[1]
                num_fixed_treenodes += len(fixed_nodes_in) - len(fixed_nodes_out)
                self.num_samples_set.add(num_fixed_treenodes)
            valid_samples_in_tree[i+1] = num_fixed_treenodes

        if ts.num_nodes - ts.num_samples - len(result) != 0:
            logging.debug(
                "Assigning priors to skipped unary nodes, via linked nodes\
                with new priors")
            # We have some nodes with unassigned prior params. We should see
            # if can we assign params for these node priors using
            # now-parameterized nodes. This requires another pass through the
            # tree sequence. If there is no non-parameterized node above, then
            # we can simply assign this the coalescent maximum
            unassigned_nodes = set(
                [n.id for n in ts.nodes()
                    if not n.is_sample() and n.id not in result])
            for i, tree in enumerate(ts.trees()):
                if i not in trees_with_unassigned_nodes:
                    continue
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
                        if n == tskit.NULL or n in result:
                            done = True
                    if n == tskit.NULL:
                        continue
                    else:
                        logging.debug(
                            "Assigning prior to unary node {}: connected to\
                            node {} which"
                            "has a prior in tree {}".format(node, n, i))
                        for local_valid, weights in result[n].items():
                            for k, v in weights.items():
                                local_weight = v / self.node_total_span[n]
                                result[node][local_valid][k] += tree.span * \
                                    local_weight / 2
                        assert len(tree.children(node)) == 1
                        num_valid = valid_samples_in_tree[i]
                        result[node][num_valid][tree.num_samples(node)] += \
                            tree.span / 2
                        self.node_total_span[node] += tree.span

        if ts.num_nodes - ts.num_samples - len(result) != 0:
            logging.debug(
                "Assigning priors to remaining (unconnected) unary nodes\
                using max depth")
            # We STILL have some missing priors.
            # These must be unconnected to higher
            # nodes in the tree, so we can simply give them the max depth
            max_samples = ts.num_samples
            unassigned_nodes = set(
                [n.id for n in ts.nodes()
                    if not n.is_sample() and n.id not in result])
            for i, tree in enumerate(ts.trees()):
                if i not in trees_with_unassigned_nodes:
                    continue
                for node in unassigned_nodes:
                    if tree.is_internal(node):
                        assert len(tree.children(node)) == 1
                        # above, we set the maximum
                        result[node][max_samples][max_samples] += tree.span / 2
                        # below, we do as before
                        assert len(tree.children(node)) == 1
                        num_valid = valid_samples_in_tree[i]
                        result[node][num_valid][tree.num_samples(node)] += \
                            tree.span / 2
                        self.node_total_span[node] += tree.span

        if ts.num_nodes - ts.num_samples != len(result):
            assert len(result) < ts.num_nodes - ts.num_samples, "There's a bug!"
            covered_nodes = set(result.keys()) | fixed_node_set
            missing_nodes = [n for n in range(ts.num_nodes) if n not in covered_nodes]
            raise ValueError(
                "The following nodes are not in any tree;"
                " please simplify your tree sequence: {}".format(missing_nodes))

        for node, weights_by_total_tips in result.items():
            result[node] = {}
            for num_samples, weights in sorted(weights_by_total_tips.items()):
                result[node][num_samples] = Weights(
                    # we use np.int64 as it's faster to look up in pandas dataframes
                    descendant_tips=np.array(list(weights.keys()), dtype=np.int64),
                    weight=np.array(list(weights.values()))/self.node_total_span[node])
        # Assign into the instance, for further reference
        self.normalised_node_spans = result
        self.nodes_to_date = np.array(list(result.keys()), dtype=np.uint64)

    def weights(self, node):
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
        which = self.weights(node)[total_tips].descendant_tips == descendant_tips
        return self.weights(node)[total_tips].weight[which]


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


def get_mut_edges(ts):
    """
    Assign mutations to edges in the tree sequence.
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
            # These don't provide any information for the forwards step
            if m.node in edges_by_child:
                edge_id = edges_by_child[m.node]
                mut_edges[edge_id] += 1
    return(mut_edges)


def get_ll(muts_span, dt, theta):
    # Has to be a top-level function to allow multiprocessing
    return (
        muts_span[0],
        muts_span[1],
        scipy.stats.poisson.pmf(muts_span[0], dt * (theta / 2 * (muts_span[1]))))


def get_mut_ll(ts, grid, theta, eps, num_threads=1):
    """
    Precalculate the likelihood for each unique edge.
    An edge is considered unique if it has a unique number of mutations and
    span.
    Constructs a lower triangular matrix of all possible delta t's, but
    stores this in 1d to save space.
    get_rows_cols() returns rows and columns to index into likelihoods.
    """
    dt = np.concatenate([grid[time] - grid[0:time + 1] +
                        eps for time in np.arange(len(grid))])
    result = {}

    def uniq(iterable):
        for element in iterable:
            if element not in result:
                result[element] = None  # Mark this result as visited processed
                yield element

    mut_edges = get_mut_edges(ts)
    spans = ts.tables.edges.right[:] - ts.tables.edges.left[:]

    if num_threads > 1:
        with multiprocessing.Pool(processes=num_threads) as pool:
            # Use functools.partial to set constant values for params to get_ll
            f = functools.partial(get_ll, dt=dt, theta=theta)
            for muts, span, pmf in pool.imap_unordered(f, uniq(zip(mut_edges, spans))):
                result[(muts, span)] = pmf
    else:
        for muts_span in uniq(zip(mut_edges, spans)):
            result[muts_span] = get_ll(muts_span, dt=dt, theta=theta)[2]

    return result


def get_rows_cols(grid):
    """
    Constructs a lower triangular matrix of all possible delta t's, but
    stores this in 1d to save space.
    Computes numpy array of rows and columns to index into the likelihoods
    for the forward and backward algorithms, respectively.
    """
    def row(time, grid):
        start = (time * (time + 1)) // 2
        end = start + time + 1
        return np.arange(start, end)

    def column(time, grid):
        n = np.arange(len(grid))
        return ((((n * (n + 1)) // 2) + time)[time:])

    rows = [row(time, grid) for time in range(len(grid))]
    cols = [column(time, grid) for time in range(len(grid))]

    return rows, cols


def forward_algorithm(ts, prior_values, grid, theta, rho, eps, rows, cols,
                      lls=None, progress=False):
    """
    Use dynamic programming to find approximate posterior to sample from
    """

    forwards = np.zeros((ts.num_nodes, len(grid)))  # store forward matrix
    g_i = np.zeros((ts.num_nodes, len(grid)))  # store g of i

    # initialize tips at time 0 to prob=1
    # TODO - account for ancient samples at different ages
    forwards[ts.samples(), 0] = 1
    g_i[ts.samples(), 0] = 1

    norm = np.zeros((ts.num_nodes))  # normalizing constants
    norm[ts.samples()] = 1  # set all tips normalizing constants to 1

    mut_edges = get_mut_edges(ts)
    matrix_indices = np.concatenate(
        [np.arange(time + 1) for time in np.arange(len(grid))])

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
        g_val = np.ones(len(grid))
        g_val[0] = 0
        for edge_index, edge in parent_group:
            # Calculate vals for each edge
            span = edge.right - edge.left
            dt = forwards[edge.child][matrix_indices]
            if theta is not None and rho is not None:
                b_l = (edge.left != 0)
                b_r = (edge.right != ts.get_sequence_length())
                ll_rec = np.power(
                    dt, b_l + b_r) * np.exp(-(dt * rho * span * 2))
                ll_mut = lls[mut_edges[edge_index], span]
                vv = ll_mut * ll_rec * dt
                vv = np.add.reduceat(vv, cols[0])
                val *= vv
                g_val *= vv
            elif theta is not None:
                ll_mut = lls[mut_edges[edge_index], span]
                vv = ll_mut * dt
                vv = np.add.reduceat(vv, cols[0])
                val *= vv
                g_val *= vv
            elif rho is not None:
                b_l = (edge.left != 0)
                b_r = (edge.right != ts.get_sequence_length())
                ll_rec = np.power(
                    dt, b_l + b_r) * np.exp(-(dt * rho * span * 2))
                vv = ll_rec * dt
                vv = np.add.reduceat(vv, cols[0])
                val *= vv
                g_val *= vv
            else:
                # Topology-only clock
                vv = np.add.reduceat(dt, cols[0])

        forwards[parent] = val
        g_i[parent] = g_val
        norm[parent] = np.max(forwards[parent, :])
        forwards[parent, :] = np.divide(forwards[parent, :], norm[parent])
        g_i_norm = np.max(g_i[parent, :])
        g_i[parent, :] = np.divide(g_i[parent, :], g_i_norm)
    logged_forwards = np.log(forwards + 1e-10)
    logged_g_i = np.log(g_i + 1e-10)
    return forwards, g_i, logged_forwards, logged_g_i


# TODO: Account for multiple parents, fix the log of zero thing
def backward_algorithm(
        ts, forwards, g_i, grid, theta, rho, spans, eps, lls, rows, cols,
        dated_node_set=None):
    """
    Computes the full posterior distribution on nodes.
    Input is log of forwards matrix, log of g_i matrix, time grid, population scaled
    mutation and recombination rate. Spans of each edge, epsilon, likelihoods of each
    edge, and the columns of the lower triangular matrix of likelihoods.
    """
    if dated_node_set is None:
        dated_node_set = set(ts.samples())
    node_has_date = np.zeros(ts.num_nodes, dtype=bool)
    node_has_date[list(dated_node_set)] = True
    backwards = np.zeros((ts.num_nodes, len(grid)))  # store backwards matrix
    mut_edges = get_mut_edges(ts)
    norm = np.zeros((ts.num_nodes))  # normalizing constants
    norm[node_has_date] = 1  # normalizing constants of all nodes with known dates == 1
    matrix_indices = np.concatenate([np.arange(time, len(grid))
                                     for time in np.arange(len(grid) + 1)])

    for tree in ts.trees():
        for root in tree.roots:
            if len(tree.get_children(root)) == 0:
                print("Node not in tree")
                continue
            backwards[root, 1:] += (1 * tree.span) / spans[root]
    backwards[node_has_date, 0] = 1

    # The mut_ll is indexing the lower triangular matrix by rows, we now need a column-
    # based index. end_col find the index of the last element of each column in order
    # to appropriately sum the vv by columns.
    running_sum = 0
    end_col = list()
    for i in np.arange(len(grid)):
        arr = np.arange(running_sum, running_sum + len(grid) - i)
        index = arr[-1]
        running_sum = index + 1
        val = arr[0]
        end_col.append(val)

    child_edges = (ts.edge(i) for i in
                   reversed(np.argsort(ts.tables.nodes.time[ts.tables.edges.child[:]])))

    for child, edges in tqdm(
            itertools.groupby(child_edges, key=lambda x: x.child),
            total=ts.num_nodes - np.sum(node_has_date)):
        if child not in dated_node_set:

            edges = list(edges)
            for edge in edges:
                dt = (backwards[edge.parent] *
                      np.exp(np.subtract(forwards[edge.parent],
                                         g_i[edge.child])))[matrix_indices]
                span = edge.right - edge.left
                if theta is not None and rho is not None:
                    b_l = (edge.left != 0)
                    b_r = (edge.right != ts.get_sequence_length())
                    ll_rec = np.power(
                        dt, b_l + b_r) * np.exp(-(dt * rho * span * 2))
                    ll_mut = lls[mut_edges[edge.id], span][np.concatenate(cols)]
                    vv = dt * ll_mut * ll_rec
                    vv = np.add.reduceat(vv, end_col)
                elif theta is not None:
                    ll_mut = lls[mut_edges[edge.id], span][np.concatenate(cols)]
                    vv = dt * ll_mut
                    vv = np.add.reduceat(vv, end_col)
                elif rho is not None:
                    b_l = (edge.left != 0)
                    b_r = (edge.right != ts.get_sequence_length())
                    ll_rec = np.power(
                        dt, b_l + b_r) * np.exp(-(dt * rho * span * 2))
                    vv = dt * ll_rec
                    vv = np.add.reduceat(vv, end_col)
                else:
                    # Topology-only clock
                    vv = np.add.reduceat(dt, end_col)
            backwards[edge.child, 1:] = vv[1:]
            norm[edge.child] = max(backwards[edge.child, :])
            backwards[edge.child, :] = \
                np.divide(backwards[edge.child, :], norm[edge.child])
            backwards[edge.child, :][np.isnan(backwards[edge.child, :])] = 0
    backwards_log = np.log(backwards)
    posterior = np.exp(forwards + backwards_log)
    posterior = posterior / np.sum(posterior, axis=1)[:, None]
    return posterior, backwards


def forwards_mean_var(ts, grid, forwards, fixed_nodes=None, nodes_to_date=None):
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
        mn_post[nd] = sum(forwards[nd, ] * grid) / sum(forwards[nd, ])
        vr_post[nd] = (
            sum(forwards[nd, ] * grid ** 2) /
            sum(forwards[nd, ]) - mn_post[nd] ** 2)
    return mn_post, vr_post


def restrict_ages_topo(ts, forwards_mn, grid, eps, nodes_to_date=None):
    """
    If predicted node times violate topology, restrict node ages so that they
    must be older than all their children.
    """
    new_mn_post = np.copy(forwards_mn)
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
    tables.nodes.time = vals * Ne
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
    :param int num_threads: The number of threads to use (currently only used for
        calculating the poisson likelihood function). If
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

    theta = rho = mut_lls = None
    rows, cols = get_rows_cols(grid)

    if mutation_rate is not None:
        theta = 4 * Ne * mutation_rate
        mut_lls = get_mut_ll(tree_sequence, grid, theta, eps, num_threads=num_threads)
    if recombination_rate is not None:
        rho = 4 * Ne * recombination_rate

    forwards, g_i, logged_forwards, logged_g_i = forward_algorithm(
        tree_sequence, prior_vals, grid, theta, rho, eps, rows, cols,
        lls=mut_lls, progress=progress)
    posterior, backward = backward_algorithm(
        tree_sequence, logged_forwards, logged_g_i,
        grid, theta, rho, spans, eps, mut_lls, rows, cols, fixed_node_set)
    mn_post, _ = forwards_mean_var(tree_sequence, grid, posterior,
                                   nodes_to_date=nodes_to_date)
    new_mn_post = restrict_ages_topo(tree_sequence, mn_post, grid, eps,
                                     nodes_to_date=nodes_to_date)
    dated_ts = return_ts(tree_sequence, new_mn_post, Ne)
    return dated_ts
