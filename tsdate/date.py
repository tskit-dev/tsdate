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
from collections import defaultdict
import logging
import os

import tskit
import pandas as pd
import numpy as np
import scipy.stats
from scipy.special import comb
from tqdm import tqdm

FORMAT_NAME = "tsdate"
FORMAT_VERSION = [1, 0]


def gamma_approx(self, mean, variance):
    """
    Returns alpha and beta of a gamma distribution for a given mean and variance
    """
    return (mean ** 2) / variance, mean / variance


class prior_maker():
    def __init__(self, total_tips, approximate=None):
        self.total_tips = total_tips
        if approximate is not None:
            self.approximate = approximate
        else:
            if total_tips >= 100:
                self.approximate = True
            else:
                self.approximate = False

        if self.approximate:
            if os.path.exists("data/prior_df.csv"):
                self.prior_df = pd.read_csv('data/prior_df.csv', index_col=0)
            else:
                # Create lookup table that is close for n > ~50
                logging.debug("Creating prior lookup table, this will take\
                    ~5 minutes")
                n = 1000
                prior_lookup_table = {val / n: self.tau_var(val, n + 1)
                                      for val in np.arange(1, n + 1)}
                self.prior_df = pd.DataFrame.from_dict(
                    prior_lookup_table, orient='index')
                self.prior_df.to_csv("../data/prior_df.csv")

    def m_prob(self, m, i, n):
        """
        Corollary 2 in Wiuf and Donnelly (1999). Probability of one
        ancestor to entire sample at time tau
        """
        return (comb(n - m - 1, i - 2, exact=True)
                * comb(m, 2, exact=True)) / comb(n, i + 1, exact=True)

    def tau_expect(self, i, n):
        if i == n:
            return 2 * (1 - (1 / n))
        else:
            return (i - 1) / n

    def tau_squared_conditional(self, m, n):
        """
        Gives expectation of tau squared conditional on m
        Equation (10) from Wiuf and Donnelly (1999).
        """
        t_sum = 0
        for t in range(m, n + 1):
            t_sum += ((1 / (t ** 2)))
        return 8 * t_sum + (8 / n) - (8 / m) - (8 / (n * m))

    def tau_var(self, i, n):
        """
        For the last coalesence (n=2), calculate the Tmrca of the whole sample
        """
        if i == n:
            var = 0
            for value in range(2, n + 1):
                var += 1 / ((value ** 2) * ((value - 1) ** 2))
            return abs(4 * var)
        else:
            tau_square_sum = 0
            for m in range(2, n - i + 2):
                tau_square_sum += (self.m_prob(m, i, n) *
                                   self.tau_squared_conditional(m, n))
            return np.abs((self.tau_expect(i, n) ** 2) - (tau_square_sum))

    def tau_var_lookup(self, i, n):
        """
        Lookup tau_var if n greater than 50
        """
        if i == n:
            return self.tau_var(i, n)
        else:
            return self.prior_df.iloc[self.prior_df.index.searchsorted(i / n)].values[0]

    def make_prior(self):
        """
        Return a pandas dataframe of conditional prior on age of node
        Note: estimated times are scaled by inputted Ne and are haploid
        """
        prior = pd.DataFrame(index=np.arange(1, self.total_tips + 1),
                             columns=["Alpha", "Beta"], dtype=float)
        # prior.loc[1] is distribution of times of a "coalescence node" ending
        # in a single sample - equivalent to the time of the sample itself, so
        # it should have var = 0 and mean = sample.time
        # Setting alpha = 0 and beta = 1 sets mean (a/b) == var (a / b^2) == 0
        prior.loc[1] = [0, 1]

        if self.approximate:
            get_tau_var = self.tau_var_lookup
        else:
            get_tau_var = self.tau_var

        for tips in np.arange(2, self.total_tips + 1):
            # NB: it should be possible to vectorize this in numpy
            expectation = self.tau_expect(tips, self.total_tips)
            var = get_tau_var(tips, self.total_tips)
            alpha, beta = gamma_approx(expectation, var)
            prior.loc[tips] = [alpha, beta]
        prior.index.name = 'Num_Tips'

        return prior

def get_mixture_prior(self, node_mixtures, age_prior):
    """
    Given a dictionary of nodes with their tip weights,
    return alpha and beta of mixture distributions
    mixture input is a list of numpy arrays
    """

    def mix_expect(node):
        expectation = 0
        for N, tip_dict in node_mixtures[node].items():
            alpha = age_prior[N].loc[np.array(list(tip_dict.keys())), "Alpha"]
            beta = age_prior[N].loc[np.array(list(tip_dict.keys())), "Beta"]
            expectation += sum(
                (alpha / beta) * np.array(list(tip_dict.values())))
        return expectation

    def mix_var(node):
        first = second = third = 0
        for N, tip_dict in node_mixtures[node].items():
            alpha = age_prior[N].loc[np.array(list(tip_dict.keys())), "Alpha"]
            beta = age_prior[N].loc[np.array(list(tip_dict.keys())), "Beta"]
            first += sum(alpha / (beta ** 2) * np.array(list(tip_dict.values())))
            second += sum((alpha / beta) ** 2 * np.array(list(tip_dict.values())))
            third += sum((alpha / beta) * np.array(list(tip_dict.values()))) ** 2
        return first + second - third

    prior = pd.DataFrame(
        index=node_mixtures.keys(), columns=["Alpha", "Beta"], dtype=float)
    for node in node_mixtures:
        prior.loc[node] = gamma_approx(mix_expect(node), mix_var(node))
    prior.index.name = "Node"
    return prior


def find_node_tip_weights(tree_sequence):
    """
    Given a tree sequence, for each non-sample node (i.e. those
    for which we want to infer a date) calculate the fraction of
    the sequence with 1 descendant sample, 2 descendant samples,
    3 descendant samples etc. Non-coalescent (unary) nodes should
    take a 50:50 mix of the coalescent nodes above and below them.

    :param TreeSequence tree_sequence: The input :class:`tskit.TreeSequence`.
    :returns: a tuple of a set and a defaultdict. The set gives
    the total number of samples at different points in the tree
    sequence (for tree sequences without missing data, this
    should always be a single value, equal to
    `tree_sequence.num_samples`). The defaultdict, is a
    collection of dictionaries keyed by node id. The values for
    each of these dictionaries sum to one, with the keys
    specifying the number of samples under the relevant node.
    :rtype: tuple(set, defaultdict)
    """
    result = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
    spans = defaultdict(float)
    samples = set(tree_sequence.samples())
    curr_samples = set()
    trees_with_unassigned_nodes = set()  # Used to quickly skip trees later
    valid_samples_in_tree = np.full(tree_sequence.num_trees, tskit.NULL)

    for i, ((_, e_out, e_in), tree) in enumerate(
            zip(tree_sequence.edge_diffs(), tree_sequence.trees())):
        # In cases with missing data, the total number of relevant
        # samples will not be tree_sequence.num_samples
        curr_samples.difference_update(
            [e.parent for e in e_out if e.parent in samples])
        curr_samples.difference_update(
            [e.child for e in e_out if e.child in samples])
        curr_samples.update(
            [e.parent for e in e_in if e.parent in samples])
        curr_samples.update(
            [e.child for e in e_in if e.child in samples])

        num_valid = len(curr_samples)  # Number of non-missing samples in this tree
        valid_samples_in_tree[i] = num_valid
        span = tree.span

        # Identify numbers of parents for each node. We could probably
        # implement a more efficient algorithm by using e_out and e_in,
        # and traversing up the tree from the edge parents, revising the
        # number of tips under each parent node
        for node in tree.nodes():
            if tree.is_sample(node):
                continue  # Don't calculate for sample nodes as they have a date
            n_samples = tree.num_samples(node)
            if n_samples == 0:
                raise ValueError(
                    "Tree " + str(i) +
                    " contains a node with no descendant samples." +
                    " Please simplify your tree sequence before dating.")
                continue  # Don't count any nodes
            if len(tree.children(node)) > 1:
                result[node][num_valid][n_samples] += span
                spans[node] += span
            else:
                # UNARY NODES: take a mixture of the coalescent nodes above and below
                #  above:
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
                    continue
                # Half from the node above
                result[node][num_valid][tree.num_samples(n)] += span/2

                #  coalescent node below should have same num_samples as this one
                assert len(tree.children(node)) == 1
                result[node][num_valid][tree.num_samples(node)] += span/2

                spans[node] += span

    if tree_sequence.num_nodes - tree_sequence.num_samples - len(result) != 0:
        logging.debug(
            "Assigning priors to skipped unary nodes, via linked nodes\
            with new priors")
        # We have some nodes with unassigned prior params. We should see
        # if can we assign params for these node priors using
        # now-parameterized nodes. This requires another pass through the
        # tree sequence. If there is no non-parameterized node above, then
        # we can simply assign this the coalescent maximum
        curr_samples = set()
        unassigned_nodes = set(
            [n.id for n in tree_sequence.nodes()
                if not n.is_sample() and n.id not in result])
        for i, tree in enumerate(tree_sequence.trees()):
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
                            local_weight = v / spans[n]
                            result[node][local_valid][k] += tree.span *\
                                local_weight / 2
                    assert len(tree.children(node)) == 1
                    num_valid = valid_samples_in_tree[i]
                    result[node][num_valid][tree.num_samples(node)] += tree.span / 2
                    spans[node] += tree.span

    if tree_sequence.num_nodes - tree_sequence.num_samples - len(result) != 0:
        logging.debug(
            "Assigning priors to remaining (unconnected) unary nodes\
            using max depth")
        # We STILL have some missing priors. These must be unconnected to higher
        # nodes in the tree, so we can simply give them the max depth
        max_samples = tree_sequence.num_samples
        curr_samples = set()
        unassigned_nodes = set(
            [n.id for n in tree_sequence.nodes()
                if not n.is_sample() and n.id not in result])
        for i, tree in enumerate(tree_sequence.trees()):
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
                    result[node][num_valid][tree.num_samples(node)] += tree.span / 2
                    spans[node] += tree.span

    if tree_sequence.num_nodes - tree_sequence.num_samples != len(result):
        raise ValueError(
            "There are some nodes which are not in any tree."
            " Please simplify your tree sequence.")

    for node, weights in result.items():
        result[node] = {}
        for num_samples, w in weights.items():
            result[node][num_samples] = {k: v / spans[node] for k, v in w.items()}

    return np.unique(valid_samples_in_tree), result

    
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
    percentiles = np.linspace(0, 1, n_points+1)[1:-1]
    # percentiles = np.append(percentiles, 0.999999)
    """
    get the set of times from gamma percent point function at the given
    percentiles specifies the value of the RV such that the prob of the var
    being less than or equal to that value equals the given probability
    """
    t_set = scipy.stats.gamma.ppf(percentiles, age_prior.loc[2, "Alpha"],
                                  scale=1 / age_prior.loc[2, "Beta"])

    # progressively add values to the grid
    max_sep = 1.0/(n_points-1)
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


def iterate_child_edges(ts):
    if ts.num_edges > 0:
        all_edges = list(ts.edges())
        child_edges = all_edges[-1:]
        child_edges[0] = (0, child_edges[0])
        cur_child = all_edges[-1:][0].child
        last_children = np.arange(ts.num_samples)
        for index, edge in enumerate(np.flip(all_edges[0:-1])):
            if edge.child != cur_child and edge.child not in last_children:
                yield child_edges
                cur_child = edge.child
                child_edges = []
            child_edges.append((index + 1, edge))
        yield child_edges


def get_prior_values(mixture_prior, grid, ts):
    prior_times = np.zeros((ts.num_nodes, len(grid)))
    for node in ts.nodes():
        if not node.is_sample():
            prior_node = scipy.stats.gamma.cdf(
                grid, mixture_prior.loc[node.id, "Alpha"],
                scale=1 / mixture_prior.loc[node.id, "Beta"])
            prior_node = prior_node / max(prior_node)
            # density of proposal in each epoch
            prior_intervals = np.concatenate(
                [np.array([0]), np.diff(prior_node)])

            # normalize so max value is 1
            prior_intervals = prior_intervals / max(prior_intervals[1:])
            prior_times[node.id, :] = prior_intervals
        else:
            prior_times[node.id, 0] = 1
    return prior_times


def get_approx_post(ts, prior_values, grid, theta, rho,
                    eps, progress):
    """
    Use dynamic programming to find approximate posterior to sample from
    """

    approx_post = np.zeros((ts.num_nodes, len(grid)))  # store forward matrix
    # initialize tips at time 0 to prob=1
    # TODO - account for ancient samples at different ages
    approx_post[ts.samples(), 0] = 1

    norm = np.zeros((ts.num_nodes))  # normalizing constants
    norm[ts.samples()] = 1  # set all tips normalizing constants to 1


    # mut_edges = np.empty(ts.num_edges)
    # for index, edge in enumerate(ts.tables.edges):
    #     # Not all sites necessarily have a mutation
    #     mut_positions = ts.tables.sites.position[
    #         ts.tables.mutations.site[ts.tables.mutations.node == edge.child]]
    #     mut_edges[index] = np.sum(np.logical_and(edge.left < mut_positions,
    #                               edge.right > mut_positions))


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
            edge_id = edges_by_child[m.node]
            mut_edges[edge_id] += 1


    # Iterate through the nodes via groupby on parent node
    for parent_group in tqdm(iterate_parent_edges(ts), disable=not progress):
        """
        for each node, find the conditional prob of age at every time
        in time grid
        """
        parent = parent_group[0][1].parent

        # For each node iterate through the time grid...
        # Internal nodes can ONLY start at 2nd t_grid point
        for time in np.arange(1, len(grid)):
            # get list of time differences for possible t'primes
            dt = grid[time] - grid[0:time + 1] + eps
            # how much does prior change over that interval in grid
            val = prior_values[parent, time]

            for edge_index, edge in parent_group:
                # Calculate vals for each edge
                span = edge.right - edge.left

                if theta is not None and rho is not None:
                    lk_mut = scipy.stats.poisson.pmf(
                        mut_edges[edge_index], dt * (theta / 2 * span))
                    b_l = (edge.left != 0)
                    b_r = (edge.right != ts.get_sequence_length())
                    lk_rec = np.power(
                        dt, b_l + b_r) * np.exp(-(dt * rho * span * 2))
                    vv = sum(approx_post[edge.child, 0:time + 1] * (
                        lk_mut * lk_rec))
                elif theta is not None:
                    lk_mut = scipy.stats.poisson.pmf(
                        mut_edges[edge_index], dt * (theta / 2 * span))
                    vv = sum(approx_post[edge.child, 0:time + 1] * lk_mut)
                elif rho is not None:
                    b_l = (edge.left != 0)
                    b_r = (edge.right != ts.get_sequence_length())
                    lk_rec = np.power(
                        dt, b_l + b_r) * np.exp(-(dt * rho * span * 2))
                    vv = sum(approx_post[edge.child, 0:time + 1] * lk_rec)

                else:
                    # Topology-only clock
                    vv = sum(
                        approx_post[edge.child, 0:time + 1] * 1 / len(
                            grid))

                val = val * vv
            approx_post[parent, time] = val

        norm[parent] = max(approx_post[parent, :])
        approx_post[parent, :] = approx_post[parent, :] / norm[parent]
    approx_post = np.insert(approx_post, 0, grid, 0)
    return approx_post


def approx_post_mean_var(ts, grid, approx_post):
    """
    Mean and variance of node age in scaled time
    """
    mn_post = np.zeros(ts.num_nodes)
    vr_post = np.zeros(ts.num_nodes)

    nonsample_nodes = np.arange(ts.num_nodes)
    nonsample_nodes = nonsample_nodes[~np.isin(nonsample_nodes, ts.samples())]
    for nd in nonsample_nodes:
        mn_post[nd] = sum(approx_post[nd, ] * grid) / sum(approx_post[nd, ])
        vr_post[nd] = (
            sum(approx_post[nd, ] * grid ** 2) /
            sum(approx_post[nd, ]) - mn_post[nd] ** 2)
    return mn_post, vr_post


def restrict_ages_topo(ts, approx_post_mn, grid, eps):
    """
    If predicted node times violate topology, restrict node ages so that they
    must be older than all their children.
    """
    new_mn_post = np.copy(approx_post_mn)
    tables = ts.tables
    nonsample_nodes = np.arange(ts.num_nodes)
    nonsample_nodes = nonsample_nodes[~np.isin(nonsample_nodes, ts.samples())]
    for nd in nonsample_nodes:
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
    :param int num_threads: The number of threads to use. If
        this is <= 0 then a simpler sequential algorithm is used (default).
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

    num_samples, tip_weights = find_node_tip_weights(tree_sequence)
    prior_df = {tree_sequence.num_samples:
        prior_maker(tree_sequence.num_samples, approximate_prior).make_prior()}
    # Add in priors for trees with different sample numbers (missing data only)
    for s in num_samples:
        if s != tree_sequence.num_samples:
            prior_df[s] = prior_maker(s, approximate_prior).make_prior()

    if time_grid == 'uniform':
        grid = np.linspace(0, 8, grid_slices+1)
    elif time_grid == 'adaptive':
        # Use the prior for the complete TS
        grid = create_time_grid(prior_df[tree_sequence.num_samples], grid_slices+1)
    else:
        raise ValueError("time_grid must be either 'adaptive' or 'uniform'")

    mixture_prior = get_mixture_prior(tip_weights, prior_df)
    prior_vals = get_prior_values(mixture_prior, grid, tree_sequence)

    theta = None
    rho = None
    if mutation_rate is not None:
        theta = 4 * Ne * mutation_rate
    if recombination_rate is not None:
        rho = 4 * Ne * recombination_rate
    approx_post = get_approx_post(tree_sequence, prior_vals, grid,
                                  theta, rho, eps, progress)
    mn_post, _ = approx_post_mean_var(tree_sequence, grid, approx_post)
    new_mn_post = restrict_ages_topo(tree_sequence, mn_post, grid, eps)
    dated_ts = return_ts(tree_sequence, new_mn_post, Ne)
    return dated_ts
