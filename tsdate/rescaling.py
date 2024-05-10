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
Utilities for rescaling time according to a mutational clock
"""
from math import inf
from math import log

import numba
import numpy as np
import tskit
from numba.types import UniTuple as _unituple

from .approx import _b
from .approx import _b1r
from .approx import _f
from .approx import _f1r
from .approx import _f1w
from .approx import _f2r
from .approx import _f2w
from .approx import _i
from .approx import _i1r
from .approx import _i1w
from .approx import approximate_gamma_iqr
from .hypergeo import _gammainc_inv as gammainc_inv
from .util import mutation_span_array  # NOQA: F401


@numba.njit(_i1w(_f1r, _i))
def _fixed_changepoints(counts, epochs):
    """
    Find breakpoints such that `counts` is divided roughly equally across `epochs`
    """
    assert epochs > 0
    Y = np.append(0.0, np.cumsum(counts))
    Z = Y / Y[-1]
    z = np.linspace(0, 1, epochs + 1)
    e = np.searchsorted(Z, z, "right") - 1
    if e[0] > 0:
        e[0] = 0
    if e[-1] < counts.size:
        e[-1] = counts.size
    return e.astype(np.int32)


@numba.njit(_i1w(_f1r, _f1r, _f, _f, _f))
def _poisson_changepoints(counts, offset, penalty, min_counts, min_offset):
    """
    Given Poisson counts and offsets for a sequence of observations, find the set
    of changepoints for the Poisson rate that maximizes the profile likelihood
    under a linear penalty on complexity (e.g. penalty == 2 is AIC).

    See: "Optimal detection of changepoints with a linear computation cost"
    (https://doi.org/10.1080/01621459.2012.737745)
    """

    assert counts.size == offset.size
    assert min_counts >= 0
    assert min_offset >= 0
    assert penalty >= 0

    N = np.append(0, np.cumsum(offset))
    Y = np.append(0, np.cumsum(counts))

    def f(i, j):  # loss
        n = N[j] - N[i]
        y = Y[j] - Y[i]
        s = n < min_offset or y < min_counts
        return inf if s else -2 * y * (log(y) - log(n) - 1)

    dim = counts.size
    cost = np.empty(dim)
    F = np.empty(dim + 1)
    C = {0: np.empty(0, dtype=np.int64)}

    F[0] = -penalty
    for j in np.arange(1, dim + 1):
        argmin, minval = 0, np.inf
        for i in C:  # minimize
            cost[i] = F[i] + f(i, j) + penalty
            if cost[i] < minval:
                minval = cost[i]
                argmin = i
        F[j] = minval
        for i in set(C):  # prune
            if cost[i] > F[j] + penalty:
                C.pop(i)
        C[j] = np.append(C[argmin], argmin)

    breaks = np.append(C[dim], dim).astype(np.int32)
    return breaks


@numba.njit(_unituple(_f1w, 2)(_f1r, _f2r, _f2r, _i1r, _i1r, _f1r, _i))
def mutational_timescale(
    nodes_time,
    likelihoods,
    constraints,
    edges_parent,
    edges_child,
    edges_weight,
    max_intervals,
):
    """
    Rescale node ages so that the instantaneous mutation rate is constant.
    Edges with a negative duration are ignored when calculating the total
    rate. Returns a rescaled point estimate and the posterior.

    :param np.ndarray nodes_time: point estimates for node ages
    :param np.ndarray likelihoods: edges are rows; mutation
        counts and mutational span are columns
    :param np.ndarray constraints: lower and upper bounds on node age
    :param np.ndarray edges_parent: node index for the parent of each edge
    :param np.ndarray edges_child: node index for the child of each edge
    :param np.ndarray edges_weight: a weight for each edge
    :param int max_intervals: maximum number of intervals within which to
        estimate the time scaling
    """

    assert edges_parent.size == edges_child.size == edges_weight.size
    assert likelihoods.shape[0] == edges_parent.size and likelihoods.shape[1] == 2
    assert constraints.shape[0] == nodes_time.size and constraints.shape[1] == 2
    assert max_intervals > 0

    nodes_fixed = constraints[:, 0] == constraints[:, 1]
    assert np.all(nodes_time[nodes_fixed] == constraints[nodes_fixed, 0])

    # index node by unique time breaks
    nodes_order = np.argsort(nodes_time)
    nodes_index = np.zeros(nodes_time.size, dtype=np.int32)
    epoch_breaks = [0.0]
    k = 0
    for i, j in zip(nodes_order[1:], nodes_order[:-1]):
        if nodes_time[i] > nodes_time[j]:
            epoch_breaks.append(nodes_time[i])
            k += 1
        nodes_index[i] = k
    epoch_breaks = np.array(epoch_breaks)
    epoch_length = np.diff(epoch_breaks)
    num_epochs = epoch_length.size

    # instantaneous mutation rate per edge
    edges_length = nodes_time[edges_parent] - nodes_time[edges_child]
    edges_subset = edges_length > 0
    edges_counts = likelihoods.copy()
    edges_counts[edges_subset, 0] /= edges_length[edges_subset]

    # pass over edges, measuring overlap with each time interval
    epoch_counts = np.zeros((num_epochs, 2))
    for e in np.flatnonzero(edges_subset):
        p, c = edges_parent[e], edges_child[e]
        a, b = nodes_index[c], nodes_index[p]
        if a < num_epochs:
            epoch_counts[a] += edges_counts[e] * edges_weight[e]
        if b < num_epochs:
            epoch_counts[b] -= edges_counts[e] * edges_weight[e]
    counts = np.cumsum(epoch_counts[:, 0])
    offset = np.cumsum(epoch_counts[:, 1])

    # rescale time such that mutation density is constant between changepoints
    # TODO: use poisson changepoints to further refine
    changepoints = _fixed_changepoints(offset * epoch_length, max_intervals)
    changepoints = np.union1d(changepoints, nodes_index[nodes_fixed])
    adjust = np.zeros(changepoints.size)
    k = 0
    for i, j in zip(changepoints[:-1], changepoints[1:]):
        assert j > i
        # TODO: when changepoint intersects a fixed node?
        n = np.sum(offset[i:j])
        y = np.sum(counts[i:j])
        z = np.sum(epoch_length[i:j])
        assert n > 0, "Zero edge span in interval"
        adjust[k + 1] = z * y / n
        k += 1
    adjust = np.cumsum(adjust)
    origin = epoch_breaks[changepoints]

    return origin, adjust


@numba.njit(_f2w(_f2r, _f1r, _f1r, _f, _b))
def piecewise_scale_posterior(
    posteriors,
    original_breaks,
    rescaled_breaks,
    quantile_width,
    use_median,
):
    """
    :param float quantile_width: width of interquantile range to use for estimating
        rescaled shape parameter, e.g. 0.5 uses interquartile range
    """

    assert original_breaks.size == rescaled_breaks.size
    assert 1 > quantile_width > 0

    dim = posteriors.shape[0]
    quant_lower = quantile_width / 2
    quant_upper = 1 - quantile_width / 2

    # use posterior mean or median as a point estimate
    freed = np.logical_and(posteriors[:, 0] > -1, posteriors[:, 1] > 0)
    lower = np.zeros(dim)
    upper = np.zeros(dim)
    midpt = np.zeros(dim)
    for i in np.flatnonzero(freed):
        alpha, beta = posteriors[i]
        lower[i] = gammainc_inv(alpha + 1, quant_lower) / beta
        upper[i] = gammainc_inv(alpha + 1, quant_upper) / beta
        midpt[i] = gammainc_inv(alpha + 1, 0.5) if use_median else (alpha + 1)
        midpt[i] /= beta

    # rescale quantiles
    assert np.all(np.diff(rescaled_breaks) > 0)
    assert np.all(np.diff(original_breaks) > 0)
    scalings = np.append(np.diff(rescaled_breaks) / np.diff(original_breaks), 0)

    def rescale(x):
        i = np.searchsorted(original_breaks, x, "right") - 1
        assert i.min() >= 0 and i.max() < scalings.size  # DEBUG
        return rescaled_breaks[i] + scalings[i] * (x - original_breaks[i])

    midpt = rescale(midpt)
    lower = rescale(lower)
    upper = rescale(upper)

    # reproject posteriors using inter-quantile range
    # TODO: catch rare cases where lower/upper quantiles are nearly identical
    new_posteriors = np.zeros(posteriors.shape)
    for i in np.flatnonzero(freed):
        alpha, beta = approximate_gamma_iqr(
            quant_lower, quant_upper, lower[i], upper[i]
        )
        beta = gammainc_inv(alpha + 1, 0.5) if use_median else (alpha + 1)
        beta /= midpt[i]  # choose rate so as to keep mean or median
        new_posteriors[i] = alpha, beta

    return new_posteriors


@numba.njit(_f1w(_b1r, _i1r, _i1r, _f1r, _f1r, _i1r, _i1r))
def edge_sampling_weight(
    is_leaf,
    edges_parent,
    edges_child,
    edges_left,
    edges_right,
    insert_index,
    remove_index,
):
    """
    Calculate the probability that a randomly selected root-to-leaf path from a
    random point on the sequence contains a given edge, for all edges.

    :param np.ndarray is_leaf: boolean array indicating whether a node is a leaf
    """
    num_nodes = is_leaf.size
    num_edges = edges_child.size

    insert_position = edges_left[insert_index]
    remove_position = edges_right[remove_index]
    sequence_length = remove_position[-1]

    nodes_parent = np.full(num_nodes, tskit.NULL)
    nodes_edge = np.full(num_nodes, tskit.NULL)
    nodes_leaves = np.zeros(num_nodes)
    edges_leaves = np.zeros(num_edges)

    nodes_leaves[is_leaf] = 1.0
    total_leaves = 0.0
    position = 0.0
    a, b = 0, 0
    while position < sequence_length:
        edges_out = []
        while b < num_edges and remove_position[b] == position:
            edges_out.append(remove_index[b])
            b += 1

        edges_in = []
        while a < num_edges and insert_position[a] == position:  # edges in
            edges_in.append(insert_index[a])
            a += 1

        remainder = sequence_length - position

        for e in edges_out:
            p, c = edges_parent[e], edges_child[e]
            update = nodes_leaves[c]
            while p != tskit.NULL:
                u = nodes_edge[c]
                edges_leaves[u] -= update * remainder
                c, p = p, nodes_parent[p]
            p, c = edges_parent[e], edges_child[e]
            while p != tskit.NULL:
                nodes_leaves[p] -= update
                p = nodes_parent[p]
            nodes_parent[c] = tskit.NULL
            nodes_edge[c] = tskit.NULL
            if is_leaf[c]:
                total_leaves -= remainder

        for e in edges_in:
            p, c = edges_parent[e], edges_child[e]
            nodes_parent[c] = p
            nodes_edge[c] = e
            if is_leaf[c]:
                total_leaves += remainder
            update = nodes_leaves[c]
            while p != tskit.NULL:
                nodes_leaves[p] += update
                p = nodes_parent[p]
            p, c = edges_parent[e], edges_child[e]
            while p != tskit.NULL:
                u = nodes_edge[c]
                edges_leaves[u] += update * remainder
                c, p = p, nodes_parent[p]

        position = sequence_length
        if b < num_edges:
            position = min(position, remove_position[b])
        if a < num_edges:
            position = min(position, insert_position[a])

    edges_leaves /= total_leaves
    return edges_leaves


# TODO: standalone API for rescaling
# def rescale_tree_sequence(
#     ts, mutation_rate, *, rescaling_intervals=1000, match_segregating_sites=False
# ):
#     """
#     Adjust the time scaling of a tree sequence so that expected mutational area
#     matches the expected number of mutations on a path from leaf to root, where
#     the expectation is taken over all paths and bases in the sequence.
#
#     :param tskit.TreeSequence ts: the tree sequence to rescale
#     :param float mutation_rate: the per-base mutation rate
#     :param int rescaling_intervals: the number of time intervals for which
#         to estimate a separate time rescaling parameter
#     :param bool match_segregating_sites: if True, match the total number of
#         mutations rather than the average number of differences from the ancestral
#         state
#     """
#     if match_segregating_sites:
#         edge_weights = np.ones(ts.num_edges)
#     else:
#         has_parent = np.full(ts.num_nodes, False)
#         has_child = np.full(ts.num_nodes, False)
#         has_parent[ts.edges_child] = True
#         has_child[ts.edges_parent] = True
#         is_leaf = np.logical_and(~has_child, has_parent)
#         edge_weights = edge_sampling_weight(
#             is_leaf,
#             ts.edges_parent,
#             ts.edges_child,
#             ts.edges_left,
#             ts.edges_right,
#             ts.indexes_edge_insertion_order,
#             ts.indexes_edge_removal_order,
#         )
#     # estimate time rescaling parameter within intervals
#     samples = list(ts.samples())
#     if not np.all(ts.nodes_time[samples] == 0.0):
#         raise ValueError("Normalisation not implemented for ancient samples")
#     constraints = np.zeros((ts.num_nodes, 2))
#     constraints[:, 1] = np.inf
#     constraints[samples, :] = ts.nodes_time[samples, np.newaxis]
#     mutations_span, mutations_edge = mutation_span_array(ts)
#     mutations_span[:, 1] *= mutation_rate
#     original_breaks, rescaled_breaks = mutational_timescale(
#         ts.nodes_time,
#         mutations_span,
#         constraints,
#         ts.edges_parent,
#         ts.edges_child,
#         edge_weights,
#         rescaling_intervals,
#     )
#     # rescale node time
#     assert np.all(np.diff(rescaled_breaks) > 0)
#     assert np.all(np.diff(original_breaks) > 0)
#     scalings = np.append(np.diff(rescaled_breaks) / np.diff(original_breaks), 0)
#     idx = np.searchsorted(original_breaks, ts.nodes_time, "right") - 1
#     nodes_time = rescaled_breaks[idx] + scalings[idx] * (
#         ts.nodes_time - original_breaks[idx]
#     )
#     # calculate mutation time
#     mutations_parent = ts.edges_parent[mutations_edge]
#     mutations_child = ts.edges_child[mutations_edge]
#     mutations_time = (nodes_time[mutations_parent] + nodes_time[mutations_child]) / 2
#     above_root = mutations_edge == tskit.NULL
#     mutations_time[above_root] = nodes_time[mutations_child[above_root]]
#     tables = ts.dump_tables()
#     tables.nodes.time = nodes_time
#     tables.mutations.time = mutations_time
#     return tables.tree_sequence()
