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

from math import inf, log

import numba
import numpy as np
import tskit

from .accelerate import numba_jit
from .approx import (
    _b,
    _b1r,
    _f,
    _f1r,
    _f1w,
    _f2r,
    _f2w,
    _i,
    _i1r,
    _i1w,
    _tuple,
    _unituple,
    approximate_gamma_iqr,
)
from .hypergeo import _gammainc_inv as gammainc_inv
from .util import mutation_span_array  # NOQA: F401


@numba_jit(_i1w(_f1r, _i))
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


@numba_jit(_i1w(_f1r, _f1r, _f, _f, _f))
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


@numba_jit(
    _tuple((_f2w, _i1w))(_b1r, _i1r, _f1r, _i1r, _i1r, _f1r, _f1r, _i1r, _i1r, _f, _b)
)
def _count_mutations(
    node_is_sample,
    mutations_node,
    mutations_position,
    edges_parent,
    edges_child,
    edges_left,
    edges_right,
    indexes_insert,
    indexes_remove,
    sequence_length,
    size_biased,
):
    assert edges_parent.size == edges_child.size == edges_left.size == edges_right.size
    assert indexes_insert.size == indexes_remove.size == edges_parent.size
    assert mutations_node.size == mutations_position.size

    num_mutations = mutations_node.size
    num_edges = edges_parent.size
    num_nodes = node_is_sample.size

    indexes_mutation = np.argsort(mutations_position)
    position_insert = edges_left[indexes_insert]
    position_remove = edges_right[indexes_remove]
    position_mutation = mutations_position[indexes_mutation]

    nodes_samples = np.zeros(num_nodes)
    nodes_edge = np.full(num_nodes, tskit.NULL)
    nodes_parent = np.full(num_nodes, tskit.NULL)
    mutations_edge = np.full(num_mutations, tskit.NULL)
    edges_mutations = np.zeros(num_edges)
    edges_span = np.zeros(num_edges)

    nodes_samples[node_is_sample] = 1.0
    left = 0.0
    a, b, d = 0, 0, 0
    while a < num_edges or b < num_edges:
        remainder = sequence_length - left

        while b < num_edges and position_remove[b] == left:  # edges out
            e = indexes_remove[b]
            p, c = edges_parent[e], edges_child[e]
            nodes_edge[c] = tskit.NULL
            nodes_parent[c] = tskit.NULL
            if size_biased:
                while p != tskit.NULL:  # downdate sample counts
                    edges_span[e] -= nodes_samples[c] * remainder
                    nodes_samples[p] -= nodes_samples[c]
                    e, p = nodes_edge[p], nodes_parent[p]
            else:
                edges_span[e] -= remainder
            b += 1

        while a < num_edges and position_insert[a] == left:  # edges in
            e = indexes_insert[a]
            p, c = edges_parent[e], edges_child[e]
            nodes_edge[c] = e
            nodes_parent[c] = p
            if size_biased:
                while p != tskit.NULL:  # update sample counts
                    edges_span[e] += nodes_samples[c] * remainder
                    nodes_samples[p] += nodes_samples[c]
                    e, p = nodes_edge[p], nodes_parent[p]
            else:
                edges_span[e] += remainder
            a += 1

        right = sequence_length
        if b < num_edges:
            right = min(right, position_remove[b])
        if a < num_edges:
            right = min(right, position_insert[a])
        left = right

        while d < num_mutations and position_mutation[d] < right:
            m = indexes_mutation[d]
            c = mutations_node[m]
            e = nodes_edge[c]
            if e != tskit.NULL:
                mutations_edge[m] = e
                edges_mutations[e] += nodes_samples[c] if size_biased else 1.0
            d += 1

    mutations_edge = mutations_edge.astype(np.int32)
    edges_stats = np.column_stack((edges_mutations, edges_span))

    return edges_stats, mutations_edge


def count_mutations(ts, node_is_sample=None, size_biased=False):
    """
    Return an array with `num_edges` rows, and columns that are the number of
    mutations per edge and the total span per edge. If `size_biased` is `True`,
    then mutations and edges are weighted by frequency.

    Note that weighting edges by frequency is done tree-by-tree.
    """
    # TODO: adjust spans by an accessibility mask:
    # need to supply cumulative accessible sequence at each
    # breakpoint

    if node_is_sample is None:
        node_is_sample = np.full(ts.num_nodes, False)
        node_is_sample[list(ts.samples())] = True
    else:
        assert node_is_sample.size != ts.num_nodes

    return _count_mutations(
        node_is_sample,
        ts.mutations_node,
        ts.sites_position[ts.mutations_site],
        ts.edges_parent,
        ts.edges_child,
        ts.edges_left,
        ts.edges_right,
        ts.indexes_edge_insertion_order,
        ts.indexes_edge_removal_order,
        ts.sequence_length,
        size_biased,
    )


@numba_jit(_tuple((_f1w, _f1w, _f1w, _i1w))(_f1r, _f2r, _i1r, _i1r))
def mutational_area(
    nodes_time,
    likelihoods,
    edges_parent,
    edges_child,
):
    """
    Calculate the total number of mutations and mutational area per inter-node
    interval. These are infinitesimal; e.g. the actual count in an interval is
    `returned_count * duration`.

    :param np.ndarray nodes_time: point estimates for node ages
    :param np.ndarray likelihoods: edges are rows; mutation
        counts and mutational span are columns
    :param np.ndarray edges_parent: node index for the parent of each edge
    :param np.ndarray edges_child: node index for the child of each edge
    :param np.ndarray edges_weight: a weight for each edge
    """

    assert edges_parent.size == edges_child.size
    assert likelihoods.shape == (edges_parent.size, 2)

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
    num_epochs = epoch_breaks.size - 1

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
            epoch_counts[a] += edges_counts[e]
        if b < num_epochs:
            epoch_counts[b] -= edges_counts[e]
    counts = np.cumsum(epoch_counts[:, 0])
    offset = np.cumsum(epoch_counts[:, 1])
    duration = np.diff(epoch_breaks)

    return counts, offset, duration, nodes_index


# --- version before refactor, keeping around for reference ---
# @numba_jit(_unituple(_f1w, 2)(_f1r, _f2r, _f2r, _i1r, _i1r, _f1r, _i))
# def mutational_timescale(
#    nodes_time,
#    likelihoods,
#    constraints,
#    edges_parent,
#    edges_child,
#    edges_weight,
#    max_intervals,
# ):
#    """
#    Rescale node ages so that the instantaneous mutation rate is constant.
#    Edges with a negative duration are ignored when calculating the total
#    rate. Returns a rescaled point estimate and the posterior.
#
#    :param np.ndarray nodes_time: point estimates for node ages
#    :param np.ndarray likelihoods: edges are rows; mutation
#        counts and mutational span are columns
#    :param np.ndarray constraints: lower and upper bounds on node age
#    :param np.ndarray edges_parent: node index for the parent of each edge
#    :param np.ndarray edges_child: node index for the child of each edge
#    :param np.ndarray edges_weight: a weight for each edge
#    :param int max_intervals: maximum number of intervals within which to
#        estimate the time scaling
#    """
#
#    assert edges_parent.size == edges_child.size == edges_weight.size
#    assert likelihoods.shape[0] == edges_parent.size and likelihoods.shape[1] == 2
#    assert constraints.shape[0] == nodes_time.size and constraints.shape[1] == 2
#    assert max_intervals > 0
#
#    nodes_fixed = constraints[:, 0] == constraints[:, 1]
#    assert np.all(nodes_time[nodes_fixed] == constraints[nodes_fixed, 0])
#
#    # index node by unique time breaks
#    nodes_order = np.argsort(nodes_time)
#    nodes_index = np.zeros(nodes_time.size, dtype=np.int32)
#    epoch_breaks = [0.0]
#    k = 0
#    for i, j in zip(nodes_order[1:], nodes_order[:-1]):
#        if nodes_time[i] > nodes_time[j]:
#            epoch_breaks.append(nodes_time[i])
#            k += 1
#        nodes_index[i] = k
#    epoch_breaks = np.array(epoch_breaks)
#    epoch_length = np.diff(epoch_breaks)
#    num_epochs = epoch_length.size
#
#    # instantaneous mutation rate per edge
#    edges_length = nodes_time[edges_parent] - nodes_time[edges_child]
#    edges_subset = edges_length > 0
#    edges_counts = likelihoods.copy()
#    edges_counts[edges_subset, 0] /= edges_length[edges_subset]
#
#    # pass over edges, measuring overlap with each time interval
#    epoch_counts = np.zeros((num_epochs, 2))
#    for e in np.flatnonzero(edges_subset):
#        p, c = edges_parent[e], edges_child[e]
#        a, b = nodes_index[c], nodes_index[p]
#        if a < num_epochs:
#            epoch_counts[a] += edges_counts[e] * edges_weight[e]
#        if b < num_epochs:
#            epoch_counts[b] -= edges_counts[e] * edges_weight[e]
#    counts = np.cumsum(epoch_counts[:, 0])
#    offset = np.cumsum(epoch_counts[:, 1])
#
#    # rescale time such that mutation density is constant between changepoints
#    # TODO: use poisson changepoints to further refine
#    changepoints = _fixed_changepoints(offset * epoch_length, max_intervals)
#    changepoints = np.union1d(changepoints, nodes_index[nodes_fixed])
#    adjust = np.zeros(changepoints.size)
#    k = 0
#    for i, j in zip(changepoints[:-1], changepoints[1:]):
#        assert j > i
#        # TODO: when changepoint intersects a fixed node?
#        n = np.sum(offset[i:j])
#        y = np.sum(counts[i:j])
#        z = np.sum(epoch_length[i:j])
#        assert n > 0, "Zero edge span in interval"
#        adjust[k + 1] = z * y / n
#        k += 1
#    adjust = np.cumsum(adjust)
#    origin = epoch_breaks[changepoints]
#
#    return origin, adjust


@numba_jit(_unituple(_f1w, 2)(_f1r, _f2r, _b1r, _i1r, _i1r, _i))
def mutational_timescale(
    nodes_time,
    likelihoods,
    nodes_fixed,
    edges_parent,
    edges_child,
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
    :param int max_intervals: maximum number of intervals within which to
        estimate the time scaling
    """

    assert edges_parent.size == edges_child.size
    assert likelihoods.shape[0] == edges_parent.size
    assert likelihoods.shape[1] == 2
    assert nodes_fixed.size == nodes_time.size
    assert max_intervals > 0

    counts, offset, duration, indexes = mutational_area(
        nodes_time,
        likelihoods,
        edges_parent,
        edges_child,
    )

    # rescale time such that mutation density is constant between changepoints
    # TODO: use poisson changepoints to further refine
    epoch_breaks = np.append(0.0, np.cumsum(duration))
    changepoints = _fixed_changepoints(offset * duration, max_intervals)
    changepoints = np.unique(changepoints)
    # changepoints = np.union1d(changepoints, indexes[nodes_fixed])
    adjust = np.zeros(changepoints.size)

    # --- without any internal constraints ---
    k = 0
    for i, j in zip(changepoints[:-1], changepoints[1:]):
        assert j > i
        # TODO: when changepoint intersects a fixed node?
        n = np.sum(offset[i:j])
        y = np.sum(counts[i:j])
        z = np.sum(duration[i:j])
        assert n > 0, "Zero edge span in interval"
        adjust[k + 1] = z * y / n
        k += 1

    adjust = np.cumsum(adjust)
    origin = epoch_breaks[changepoints]

    return origin, adjust


@numba.njit(_f2w(_f2r, _b1r, _f1r, _f1r, _f, _f))
def piecewise_scale_posterior(
    posteriors,
    posteriors_fixed,
    original_breaks,
    rescaled_breaks,
    quantile_width,
    max_shape,
):
    """
    :param np.ndarray posteriors_fixed: if True do not rescale corresponding posterior
    :param float quantile_width: width of interquantile range to use for estimating
        rescaled shape parameter, e.g. 0.5 uses interquartile range
    """

    assert posteriors_fixed.size == posteriors.shape[0]
    assert original_breaks.size == rescaled_breaks.size
    assert 1 > quantile_width > 0

    dim = posteriors.shape[0]
    quant_lower = quantile_width / 2
    quant_upper = 1 - quantile_width / 2

    freed = ~posteriors_fixed
    assert np.all(np.logical_and(posteriors[freed, 0] > -1, posteriors[freed, 1] > 0))

    # use posterior mean as a point estimate
    lower = np.zeros(dim)
    upper = np.zeros(dim)
    midpt = np.zeros(dim)
    for i in np.flatnonzero(freed):
        alpha, beta = posteriors[i]
        lower[i] = gammainc_inv(alpha + 1, quant_lower) / beta
        upper[i] = gammainc_inv(alpha + 1, quant_upper) / beta
        midpt[i] = (alpha + 1) / beta

    # rescale quantiles
    assert np.all(np.diff(rescaled_breaks) > 0), "Use fewer rescaling intervals"
    assert np.all(np.diff(original_breaks) > 0), "Use fewer rescaling intervals"
    scalings = np.append(np.diff(rescaled_breaks) / np.diff(original_breaks), 0)

    def rescale(x):
        i = np.searchsorted(original_breaks, x, "right") - 1
        assert i.min() >= 0  # DEBUG
        assert i.max() < scalings.size  # DEBUG
        return rescaled_breaks[i] + scalings[i] * (x - original_breaks[i])

    midpt = rescale(midpt)
    lower = rescale(lower)
    upper = rescale(upper)

    # reproject posteriors using inter-quantile range
    new_posteriors = np.full(posteriors.shape, np.nan)
    for i in np.flatnonzero(freed):
        alpha, beta = approximate_gamma_iqr(
            quant_lower, quant_upper, lower[i], upper[i], max_shape
        )
        beta = (alpha + 1) / midpt[i]  # choose rate so as to keep mean
        new_posteriors[i] = alpha, beta

    return new_posteriors


@numba_jit(_f1w(_f1r, _b1r, _f1r, _f1r))
def piecewise_scale_point_estimate(
    point_estimate,
    point_fixed,
    original_breaks,
    rescaled_breaks,
):
    assert np.all(np.diff(rescaled_breaks) > 0), "Use fewer rescaling intervals"
    assert np.all(np.diff(original_breaks) > 0), "Use fewer rescaling intervals"
    scalings = np.append(np.diff(rescaled_breaks) / np.diff(original_breaks), 0)
    idx = np.searchsorted(original_breaks, point_estimate, "right") - 1
    rescaled_estimate = rescaled_breaks[idx] + \
        scalings[idx] * (point_estimate - original_breaks[idx])  # fmt: skip
    rescaled_estimate[point_fixed] = point_estimate[point_fixed]
    return rescaled_estimate


# standalone API for rescaling (TODO: needs testing)
def rescale_tree_sequence(
    ts,
    mutation_rate,
    *,
    num_intervals=100,
    num_iterations=10,
    match_segregating_sites=False,
):
    """
    Adjust the time scaling of a tree sequence so that expected mutational area
    matches the expected number of mutations on a path from leaf to root, where
    the expectation is taken over all paths and bases in the sequence.

    :param tskit.TreeSequence ts: the tree sequence to rescale
    :param float mutation_rate: the per-base mutation rate
    :param int num_intervals: the number of time intervals for which
        to estimate a separate time rescaling parameter
    :param int num_iterations: the number of iterations to repeat rescaling
    :param bool match_segregating_sites: if True, match the total number of
        mutations rather than the average number of differences from the ancestral
        state
    :param bool progress: if True, show a progress bar
    """
    samples = list(ts.samples())
    if not np.all(ts.nodes_time[samples] == 0.0):
        raise ValueError("Normalisation not implemented for ancient samples")
    constraints = np.zeros((ts.num_nodes, 2))
    constraints[:, 1] = np.inf
    constraints[samples, :] = ts.nodes_time[samples, np.newaxis]
    if match_segregating_sites:
        mutations_span, mutations_edge = count_mutations(ts)
    else:
        mutations_span, mutations_edge = count_mutations(ts, size_biased=True)
    mutations_span[:, 1] *= mutation_rate
    # rescale node ages
    fixed_nodes = constraints[:, 0] == constraints[:, 1]
    nodes_time = ts.nodes_time.copy()
    for _ in np.arange(num_iterations):
        original_breaks, rescaled_breaks = mutational_timescale(
            nodes_time,
            mutations_span,
            constraints,
            ts.edges_parent,
            ts.edges_child,
            num_intervals,
        )
        nodes_time = piecewise_scale_point_estimate(
            nodes_time, fixed_nodes, original_breaks, rescaled_breaks
        )
        assert np.allclose(nodes_time[fixed_nodes], ts.nodes_time[fixed_nodes])
    # calculate mutation ages
    mutations_parent = ts.edges_parent[mutations_edge]
    mutations_child = ts.edges_child[mutations_edge]
    mutations_time = (nodes_time[mutations_parent] + nodes_time[mutations_child]) / 2
    above_root = mutations_edge == tskit.NULL
    assert np.allclose(mutations_child[~above_root], ts.mutations_node[~above_root])
    mutations_time[above_root] = nodes_time[ts.mutations_node[above_root]]
    tables = ts.dump_tables()
    tables.nodes.time = nodes_time
    tables.mutations.time = mutations_time
    tables.sort()
    tables.build_index()
    tables.compute_mutation_parents()
    ts = tables.tree_sequence()
    return ts
