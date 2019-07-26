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

import pandas as pd
import numpy as np
import scipy.stats
from scipy.special import comb


def alpha_prob(m, i, n):
    """
    Corollary 2 in Wiuf and Donnelly. j equals 1. j
    is ancestors of D (which has i samples). m is the number
    of ancestors of the entire sample. let alpha*(1) be the
    number of ancestors to the whole sample at time tau
    """
    return((comb(n - m - 1, i - 2) * comb(m, 2)) / comb(n, i + 1))


def tau_expect(i, n):
    if i == n:
        return(2 * (1 - (1 / n)))
    else:
        return((i - 1) / n)


def expect_tau_cond_alpha(alpha, n):
    return(2 * ((1 / alpha) - (1 / n)))


def tau_squared_conditional(alpha, n):
    """
    Gives expectation of tau squared conditional on alpha
    """
    j_sum = 0
    for j in range(alpha, n + 1):
        j_sum += ((1 / (j ** 2)))
    return(8 * j_sum + (8 / n) - (8 / alpha) - (8 / (n * alpha)))


def tau_var(i, n):
    """
    For the last coalesence (n=2), calculate the Tmrca of the whole sample
    """
    if i == n:
        var = 0
        for value in range(2, n + 1):
            var += 1 / ((value ** 2) * ((value - 1) ** 2))
        return(abs(4 * var))
    else:
        tau_square_sum = 0
        for alpha in range(2, n - i + 2):
            tau_square_sum += (alpha_prob(alpha, i, n) *
                               tau_squared_conditional(alpha, n))
        return(np.abs((tau_expect(i, n) ** 2) - (tau_square_sum)))


def gamma_approx(mean, variance, ne=1):
    """
    Returns alpha and beta for the mean and variance
    """
    if ne != 1:
        mean = mean * 2 * ne
        variance = variance * 4 * ne * ne
    return(((mean ** 2) / variance), mean / variance)


def make_prior(n=10, ne=1):
    """
    Return a pandas dataframe of conditional prior on age of node
    """
    prior = pd.DataFrame(index=np.arange(0, n - 1),
                         columns=["Num_Tips", "Expected_Age",
                         "Var_Age", "Alpha", "Beta"], dtype=float)
    prior.loc[0] = [1, 0, 0, 0, 0]
    for tips in np.arange(2, n + 1):
        expectation = tau_expect(tips, n)
        var = tau_var(tips, n)
        alpha, beta = gamma_approx(expectation, var, ne)
        prior.loc[tips - 1] = [tips, expectation, var, alpha, beta]
    prior = prior.set_index('Num_Tips')
    return(prior)


def create_time_grid(age_prior, del_p=0.05):
    """
    Create the time grid by finding union of the quantiles of the gammas
    For a node with k descendenants we have gamma approxs.
    Natural grid would be to take all the distributions
    quantile them up, and then take the union of the quantiles.
    Then thin this, making it no more than 0.05 of a quantile apart.
    Takes all the gamma distributions, finds quantiles, takes union,
    and thins them. Does this in an iterative way.
    """
    # Percentages
    percentiles = np.arange(del_p, 1, del_p)

    """
    get the set of times from gamma percent point function at the given
    percentiles specifies the value of the RV such that the prob of the var
    being less than or equal to that value equals the given probability
    """
    t_set = scipy.stats.gamma.ppf(percentiles, age_prior.loc[2]["Alpha"],
                                  scale=1 / age_prior.loc[2]["Beta"])

    # progressively add values to the grid
    if age_prior.shape[0] > 2:
        for i in np.arange(3, age_prior.shape[0] + 1):
            # gamma percentiles of existing times in grid
            proj = scipy.stats.gamma.cdf(t_set, age_prior.loc[i]["Alpha"],
                                         scale=1 / age_prior.loc[i]["Beta"])
            """
            thin the grid, only add additional quantiles if they're more than
            0.05 from another quantile
            """
            tmp = np.asarray([min(abs(val - proj)) for val in percentiles])
            wd = np.where(tmp > del_p)

            if len(wd[0]) > 0:
                t_set = np.concatenate(
                    [t_set, np.array(scipy.stats.gamma.ppf(percentiles[wd], age_prior.loc[i]["Alpha"], scale=1 / age_prior.loc[i]["Beta"]))])

    t_set = sorted(t_set)
    return(np.insert(t_set, 0, 0))


def find_node_tip_weights_ts(ts):
    """
    Given a tree sequence, calculate the weights for the number of tips below
    each node
    """
    all_tip_weights = dict()

    for node in np.arange(ts.num_samples, ts.num_nodes):
        tip_length_dict = defaultdict(list)
        total_length = 0
        for tree in ts.trees():
            if tree.is_internal(node):
                length = tree.get_interval()[1] - tree.get_interval()[0]
                tip_length_dict[len(list(tree.get_leaves(node)))].append(
                    length)
                total_length += length
        tip_length_dict = {tip: sum(lengths) / total_length
                           for tip, lengths in tip_length_dict.items()}
        all_tip_weights[node] = tip_length_dict

    return(all_tip_weights)


def get_mixture_prior_ts_new(node_mixtures, age_prior):
    """
    Given a dictionary of nodes with their tip weights,
    return alpha and beta of mixture distributions
    mixture input is a list of numpy arrays
    """

    def mix_expect(node):
        alpha = age_prior.loc[np.array(list(
                              node_mixtures[node].keys()))]["Alpha"]
        beta = age_prior.loc[np.array(list(
                             node_mixtures[node].keys()))]["Beta"]
        return(sum((alpha / beta) *
               np.array(list(node_mixtures[node].values()))))

    def mix_var(node):
        alpha = age_prior.loc[np.array(list(
                              node_mixtures[node].keys()))]["Alpha"]
        beta = age_prior.loc[np.array(list(
                             node_mixtures[node].keys()))]["Beta"]
        first = sum(alpha / (beta ** 2) *
                    np.array(list(node_mixtures[node].values())))
        second = sum((alpha / beta) **
                     2 * np.array(list(node_mixtures[node].values())))
        third = sum((alpha / beta) *
                    np.array(list(node_mixtures[node].values()))) ** 2
        return(first + second - third)

    expect_var = {node: (mix_expect(node), mix_var(node))
                  for node in node_mixtures}
    prior = pd.DataFrame(columns=["Node", "Expected_Age", "Var_Age",
                                  "Alpha", "Beta"], dtype=float)
    alpha_beta = {node: (expect, var, (expect ** 2) /
                         var, expect / var)
                  for node, (expect, var) in expect_var.items()}
    for index, (node, (expect, var, alpha, beta)) in enumerate(alpha_beta.items()):
        prior.loc[index] = [node, expect, var, alpha, beta]
    prior = prior.set_index("Node")
    return(prior)


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


def get_prior_values(mixture_prior, time_grid, ts):
    prior_times = np.zeros((ts.num_nodes, len(time_grid)))
    for node in ts.nodes():
        if node.flags != 1:
            prior_node = scipy.stats.gamma.cdf(
                time_grid, mixture_prior.loc[node.id]["Alpha"],
                scale=1 / mixture_prior.loc[node.id]["Beta"])
            prior_node = prior_node / max(prior_node)
            # density of proposal in each epoch
            prior_intervals = np.concatenate(
                [np.array([0]), np.diff(prior_node)])

            # normalize so max value is 1
            prior_intervals = prior_intervals / max(prior_intervals[1:])
            prior_times[node.id, :] = prior_intervals
        else:
            prior_times[node.id, 0] = 1
    return(prior_times)


def get_approx_post(ts, prior_values, time_grid, eps=1e-6, clock="combined",
                theta=1, rho=1, del_p=0.02):
    """
    Use dynamic programming to find approximate posterior to sample from
    """

    approx_post = np.zeros((ts.num_nodes,
                            len(time_grid)))  # store forward matrix
    approx_post[0:ts.num_samples, 0] = 1  # initalize tips to time 0

    norm = np.zeros((ts.num_nodes))  # normalizing constants
    norm[0:ts.num_samples] = 1  # set all tips normalizing constants to 1

    mut_edges = np.empty(ts.num_edges)
    for index, edge in enumerate(ts.tables.edges):
        mut_positions = ts.tables.sites.position[
            ts.tables.mutations.node == edge.child]
        mut_edges[index] = np.sum(np.logical_and(edge.left < mut_positions,
                                  edge.right > mut_positions))

    parent_group = iterate_parent_edges(ts)
    # Iterate through the nodes via groupby on parent node
    for parent_group in iterate_parent_edges(ts):
        """
        for each node, find the conditional prob of age at every time
        in time grid
        """
        parent = parent_group[0][1].parent

        # For each node iterate through the time grid...
        # Internal nodes can ONLY start at 2nd t_grid point
        for time in np.arange(1, len(time_grid)):
            # get list of time differences for possible t'primes
            dt = time_grid[time] - time_grid[0:time + 1] + eps
            # how much does prior change over that interval in time_grid
            val = prior_values[parent, time]

            for edge_index, edge in parent_group:
                # Calculate vals for each edge
                span = edge.right - edge.left

                if clock == "combined":
                    lk_mut = scipy.stats.poisson.pmf(
                        mut_edges[edge_index], dt * (theta / 2 * span))
                    b_l = (edge.left != 0)
                    b_r = (edge.right != ts.get_sequence_length())
                    lk_rec = np.power(dt, b_l + b_r) * np.exp(-(dt * rho * span * 2))
                    vv = sum(approx_post[edge.child, 0:time + 1] * (
                        lk_mut * lk_rec))
                elif clock == "mutation":
                    lk_mut = scipy.stats.poisson.pmf(mut_edges[edge_index], dt * (theta / 2 * span))
                    vv = sum(approx_post[edge.child, 0:time + 1] * lk_mut)
                elif clock == "recombination":
                    b_l = (edge.left != 0)
                    b_r = (edge.right != ts.get_sequence_length())
                    lk_rec = np.power(dt, b_l + b_r) * np.exp(-(dt *
                        rho * span * 2))
                    vv = sum(approx_post[edge.child, 0:time + 1] * lk_rec)

                elif clock == "topo":
                    vv = sum(approx_post[edge.child, 0:time + 1] * 1 / len(time_grid))

                val = val * vv
            approx_post[parent, time] = val

        norm[parent] = max(approx_post[parent, :])
        approx_post[parent, :] = approx_post[parent, :] / norm[parent]
    approx_post = np.insert(approx_post, 0, time_grid, 0)
    return(approx_post)

def approx_post_mean_var(ts, time_grid, approx_post):
    # Mean and variance of node age
    mn_post = np.zeros(ts.num_nodes)
    vr_post = np.zeros(ts.num_nodes)

    for i in np.arange(ts.num_samples, ts.num_nodes):
        mn_post[i] = sum(approx_post[i, ] * time_grid) / sum(approx_post[i, ])
        vr_post[i] = sum(approx_post[i, ] * time_grid ** 2) / sum(approx_post[i, ]) - mn_post[i] ** 2
    return(mn_post, vr_post)

def age_inference(ts, grid, clock, theta, rho, del_p, output):
    # Run inference
    tip_weights = find_node_tip_weights_ts(ts)
    prior = make_prior(n=ts.num_samples)

    if grid == 'uniform':
        time_grid = np.arange(0, 8, del_p)
    elif grid == 'union':
        time_grid = create_time_grid(prior, del_p=del_p)
    else:
        print("Must enter union or uniform for time grid")

    mixture_prior = get_mixture_prior_ts_new(tip_weights, prior)
    prior_vals = get_prior_values(mixture_prior, time_grid, ts)
    approx_post = get_approx_post(ts, prior_vals, time_grid,
                                  clock=clock, theta=theta,
                                  rho=rho, del_p=del_p)
    
    return(approx_post_mean_var(ts, time_grid, approx_post))
    #np.savetxt(output, approx_post, delimiter=",")
