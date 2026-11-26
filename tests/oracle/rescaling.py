"""
Oracles for the rescaling functions (``lib/rescaling.c``).
"""

from math import inf

import numpy as np
import tskit

from . import errors

# Plain `math.log` raises on 0, where C gives -inf; C sums sequentially,
# whereas numpy sums pairwise (`seq_sum`).
from ._libm import log, seq_sum
from .approx import approximate_gamma_iqr
from .errors import library_error
from .hypergeo import gammainc_inv


def fixed_changepoints(counts, epochs):
    """
    Find breakpoints such that `counts` is divided roughly equally across `epochs`
    """
    if not (epochs > 0):
        raise library_error(errors.NONPOSITIVE_INTERVALS)
    Y = np.append(0.0, np.cumsum(counts))
    Z = Y / Y[-1]
    z = np.linspace(0, 1, epochs + 1)
    e = np.searchsorted(Z, z, "right") - 1
    if e[0] > 0:
        e[0] = 0
    if e[-1] < counts.size:
        e[-1] = counts.size
    return e.astype(np.int32)


def poisson_changepoints(counts, offset, penalty, min_counts, min_offset):
    """
    Given Poisson counts and offsets for a sequence of observations, find the set
    of changepoints for the Poisson rate that maximizes the profile likelihood
    under a linear penalty on complexity (e.g. penalty == 2 is AIC).

    See: "Optimal detection of changepoints with a linear computation cost"
    (https://doi.org/10.1080/01621459.2012.737745)
    """

    if counts.size != offset.size:
        raise ValueError("counts and offset must have the same size")
    if not (min_counts >= 0 and min_offset >= 0 and penalty >= 0):
        raise library_error(errors.NEGATIVE_CHANGEPOINT_PARAMETER)

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


def count_mutations(
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
    """
    The mutations on each edge and the edge's span, weighted by the number of samples
    below the edge if ``size_biased``, as ``(edges_stats, mutations_edge)``.
    """
    if not (edges_parent.size == edges_child.size == edges_left.size == edges_right.size):
        raise ValueError("Edge arrays must have the same size")
    if not (indexes_insert.size == indexes_remove.size == edges_parent.size):
        raise ValueError("Edge index arrays must have one entry per edge")
    if mutations_node.size != mutations_position.size:
        raise ValueError("Mutation arrays must have the same size")

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

    if edges_parent.size != edges_child.size:
        raise ValueError("Edge arrays must have the same size")
    if likelihoods.shape != (edges_parent.size, 2):
        raise ValueError("Likelihoods must have one row of two columns per edge")

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

    if edges_parent.size != edges_child.size:
        raise ValueError("Edge arrays must have the same size")
    if likelihoods.shape != (edges_parent.size, 2):
        raise ValueError("Likelihoods must have one row of two columns per edge")
    if nodes_fixed.size != nodes_time.size:
        raise ValueError("nodes_fixed and nodes_time must have the same size")
    if not (max_intervals > 0):
        raise library_error(errors.NONPOSITIVE_INTERVALS)

    counts, offset, duration, indexes = mutational_area(
        nodes_time,
        likelihoods,
        edges_parent,
        edges_child,
    )

    # rescale time such that mutation density is constant between changepoints
    # TODO: use poisson changepoints to further refine
    epoch_breaks = np.append(0.0, np.cumsum(duration))
    changepoints = fixed_changepoints(offset * duration, max_intervals)
    changepoints = np.unique(changepoints)
    # changepoints = np.union1d(changepoints, indexes[nodes_fixed])
    adjust = np.zeros(changepoints.size)

    # --- without any internal constraints ---
    k = 0
    for i, j in zip(changepoints[:-1], changepoints[1:]):
        assert j > i
        # TODO: when changepoint intersects a fixed node?
        n = seq_sum(offset[i:j])
        y = seq_sum(counts[i:j])
        z = seq_sum(duration[i:j])
        if not (n > 0):
            raise library_error(errors.ZERO_EDGE_SPAN)
        adjust[k + 1] = z * y / n
        k += 1

    adjust = np.cumsum(adjust)
    origin = epoch_breaks[changepoints]

    return origin, adjust


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

    if not (posteriors.shape[1] == 2 and posteriors_fixed.size == posteriors.shape[0]):
        raise ValueError("Posteriors must have two columns and one row per fixed flag")
    if not (original_breaks.size == rescaled_breaks.size and original_breaks.size > 0):
        raise ValueError("Breakpoints must be non-empty and of the same size")
    if not (1 > quantile_width > 0):
        raise library_error(errors.INVALID_QUANTILE_WIDTH)

    if not (
        np.all(np.diff(rescaled_breaks) > 0) and np.all(np.diff(original_breaks) > 0)
    ):
        raise library_error(errors.NON_INCREASING_BREAKPOINTS)

    dim = posteriors.shape[0]
    quant_lower = quantile_width / 2
    quant_upper = 1 - quantile_width / 2

    freed = ~posteriors_fixed
    if not np.all(np.logical_and(posteriors[freed, 0] > -1, posteriors[freed, 1] > 0)):
        raise library_error(errors.INVALID_POSTERIOR)
    if dim == 0:
        raise library_error(errors.NO_POSTERIORS)

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
    scalings = np.append(np.diff(rescaled_breaks) / np.diff(original_breaks), 0)

    def rescale(x):
        i = np.searchsorted(original_breaks, x, "right") - 1
        if not (i.min() >= 0):
            raise library_error(errors.TIME_BELOW_FIRST_BREAKPOINT)
        assert i.max() < scalings.size
        return rescaled_breaks[i] + scalings[i] * (x - original_breaks[i])

    midpt = rescale(midpt)
    lower = rescale(lower)
    upper = rescale(upper)

    # reproject posteriors using inter-quantile range
    new_posteriors = np.full(posteriors.shape, np.nan)
    for i in np.flatnonzero(freed):
        # The library reports a float division by zero here (as
        # `ZeroDivisionError` in the oracle), where numpy scalars give inf.
        # `approximate_gamma_iqr` divides by its `x1` argument (`x2 / x1`, or `... / x`
        # in `upper_bound`) in every branch that doesn't raise earlier, so a zero lower
        # quantile raises there.
        if lower[i] == 0.0:
            raise ZeroDivisionError("division by zero")
        alpha, beta = approximate_gamma_iqr(
            quant_lower, quant_upper, lower[i], upper[i], max_shape
        )
        if midpt[i] == 0.0:
            raise ZeroDivisionError("division by zero")
        beta = (alpha + 1) / midpt[i]  # choose rate so as to keep mean
        new_posteriors[i] = alpha, beta

    return new_posteriors


def piecewise_scale_point_estimate(
    point_estimate,
    point_fixed,
    original_breaks,
    rescaled_breaks,
):
    """
    Node ages mapped piecewise linearly from ``original_breaks`` to
    ``rescaled_breaks``; ages where ``point_fixed`` are unchanged.
    """
    if not (
        np.all(np.diff(rescaled_breaks) > 0) and np.all(np.diff(original_breaks) > 0)
    ):
        raise library_error(errors.NON_INCREASING_BREAKPOINTS)
    scalings = np.append(np.diff(rescaled_breaks) / np.diff(original_breaks), 0)
    idx = np.searchsorted(original_breaks, point_estimate, "right") - 1
    rescaled_estimate = rescaled_breaks[idx] + \
        scalings[idx] * (point_estimate - original_breaks[idx])  # fmt: skip
    rescaled_estimate[point_fixed] = point_estimate[point_fixed]
    return rescaled_estimate
