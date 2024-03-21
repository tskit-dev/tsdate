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
from numba.types import Tuple as _tuple
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
from .hypergeo import _gammainc as gammainc
from .hypergeo import _gammainc_inv as gammainc_inv


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


# @numba.njit(_f1w(_f1r, _f2r, _i1r, _i1r))
# def scale_time_by_mutations(
#     nodes_time, likelihoods, edges_parent, edges_child,
# ):
#     """
#     `edges_span` is pre-multiplied by mutation rate
#     """
#
#     edges_muts = likelihoods[:, 0].copy()
#     edges_span = likelihoods[:, 1].copy()
#
#     # index node by unique time breaks
#     nodes_order = np.argsort(nodes_time)
#     nodes_index = np.zeros(nodes_time.size, dtype=np.int32)
#     time_breaks = [0.0]
#     k = 0
#     for i, j in zip(nodes_order[1:], nodes_order[:-1]):
#         if nodes_time[i] > nodes_time[j]:
#             time_breaks.append(nodes_time[i])
#             k += 1
#         nodes_index[i] = k
#     time_breaks = np.array(time_breaks)
#     time_interval = np.diff(time_breaks)
#
#     # pass over edges, measuring overlap with each time interval
#     area = np.zeros(time_interval.size)
#     muts = np.zeros(time_interval.size)
#     for e in range(edges_parent.size):
#         p, c = edges_parent[e], edges_child[e]
#         length = nodes_time[p] - nodes_time[c]
#         if length > 0:
#             for j in range(nodes_index[c], nodes_index[p]):
#                 area[j] += time_interval[j] * edges_span[e]
#                 muts[j] += time_interval[j] * edges_muts[e] / length
#
#     # rescale time such that mutation density is constant
#     for i, t in enumerate(time_interval):
#         time_breaks[i + 1] = time_breaks[i] + t * muts[i] / area[i]
#
#     return time_breaks[nodes_index]


# @numba.njit(_f1w(_f1r, _f2r, _i1r, _i1r, _f, _f))
# def scale_time_by_mutations(nodes_time, likelihoods, edges_parent, edges_child, min_counts, min_offset):
#    """
#    Rescale node ages so that the instantaneous mutation rate is constant.
#    Edges with a negative duration are ignored when calculating the total
#    rate.
#
#    :param np.ndarray nodes_time: array of node ages
#    :param np.ndarray likelihoods: edges are rows; mutation
#        counts and mutational span are columns
#    :param np.ndarray edges_parent: node index for the parent of each edge
#    :param np.ndarray edges_child: node index for the child of each edge
#    :param float min_counts: minimum number of mutations per bin for time
#        rescaling
#    :param float min_offset: minimum mutational target per bin for time
#        rescaling
#    """
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
#            epoch_counts[a] += edges_counts[e]
#        if b < num_epochs:
#            epoch_counts[b] -= edges_counts[e]
#    counts = np.cumsum(epoch_counts[:, 0])
#    offset = np.cumsum(epoch_counts[:, 1])
#
#    # fit a Poisson changepoint model to time intervals
#    changepoints = _poisson_changepoints(
#        counts,
#        offset,
#        2.0, #np.log(num_epochs), # BIC-ish penalty
#        min_counts,
#        min_offset,
#    )
#    epoch_scales = np.empty(num_epochs)
#    for i, j in zip(changepoints[:-1], changepoints[1:]):
#        n, y = np.sum(offset[i:j]), np.sum(counts[i:j])
#        epoch_scales[i:j] = y / n
#    epoch_adjust = np.append(0, np.cumsum(epoch_length * epoch_scales))
#
#    #epoch_scales = counts / offset
#    #epoch_adjust = np.append(0, np.cumsum(epoch_length * epoch_scales))
#
#    return epoch_adjust[nodes_index]


@numba.njit(_tuple((_f2w, _f2w))(_f2r, _f2r, _f2r, _i1r, _i1r, _f, _i, _b))
def scale_time_by_mutations(
    posteriors,
    likelihoods,
    constraints,
    edges_parent,
    edges_child,
    quantile_width,
    max_intervals,
    use_median,
):
    """
    Rescale node ages so that the instantaneous mutation rate is constant.
    Edges with a negative duration are ignored when calculating the total
    rate. Returns a rescaled point estimate and the posterior.

    :param np.ndarray posteriors: natural parameters of variational posteriors
    :param np.ndarray likelihoods: edges are rows; mutation
        counts and mutational span are columns
    :param np.ndarray constraints: lower and upper bounds on node age
    :param np.ndarray edges_parent: node index for the parent of each edge
    :param np.ndarray edges_child: node index for the child of each edge
    :param float quantile_width: width of interquantile range to use for estimating
        rescaled shape parameter, e.g. 0.5 uses interquartile range
    :param int max_intervals: maximum number of intervals within which to
        estimate the time scaling
    :param bool use_median: if True, use the posterior median rather than the
        mean for the rescaling
    """

    # use posterior mean or median as a point estimate
    num_nodes = posteriors.shape[0]
    nodes_fixed = constraints[:, 0] == constraints[:, 1]
    nodes_lower = np.zeros(num_nodes)
    nodes_upper = np.zeros(num_nodes)
    nodes_midpt = np.zeros(num_nodes)
    quant_lower = quantile_width / 2
    quant_upper = 1 - quant_lower
    for i in range(num_nodes):
        if nodes_fixed[i]:
            nodes_lower[i] = constraints[i, 0]
            nodes_midpt[i] = constraints[i, 0]
            nodes_upper[i] = constraints[i, 0]
        else:
            alpha, beta = posteriors[i]
            nodes_midpt[i] = (
                gammainc_inv(alpha + 1, 0.5) / beta
                if use_median
                else (alpha + 1) / beta
            )
            nodes_lower[i] = gammainc_inv(alpha, quant_lower) / beta
            nodes_upper[i] = gammainc_inv(alpha, quant_upper) / beta
    nodes_time = nodes_midpt

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
            epoch_counts[a] += edges_counts[e]
        if b < num_epochs:
            epoch_counts[b] -= edges_counts[e]
    counts = np.cumsum(epoch_counts[:, 0])
    offset = np.cumsum(epoch_counts[:, 1])

    # rescale time such that mutation density is constant between changepoints
    length = epoch_length
    changepoints = _fixed_changepoints(counts * length, max_intervals)
    changepoints = np.union1d(changepoints, nodes_index[nodes_fixed])
    scales = np.zeros(changepoints.size)
    adjust = np.zeros(changepoints.size)
    origin = epoch_breaks[changepoints]
    k = 0
    for i, j in zip(changepoints[:-1], changepoints[1:]):
        assert j > i
        # TODO: do the right thing when a changepoint intersects a fixed node
        n = np.sum(offset[i:j])
        y = np.sum(counts[i:j])
        z = np.sum(length[i:j])
        assert n > 0, "Zero edge span in interval"
        scales[k] = y / n
        adjust[k + 1] = z * y / n
        k += 1
    adjust = np.cumsum(adjust)
    index = np.searchsorted(origin, nodes_time, "right") - 1
    assert index[0] == 0 and index[-1] < scales.size
    nodes_adjust = adjust[index] + scales[index] * (nodes_time - origin[index])

    # rescale posteriors using inter-quantile range
    index_lower = np.searchsorted(origin, nodes_lower, "right") - 1
    index_upper = np.searchsorted(origin, nodes_upper, "right") - 1
    nodes_lower = adjust[index_lower] + scales[index_lower] * (
        nodes_lower - origin[index_lower]
    )
    nodes_upper = adjust[index_upper] + scales[index_upper] * (
        nodes_upper - origin[index_upper]
    )
    posteriors_adjust = np.zeros(posteriors.shape)
    for i in np.flatnonzero(~nodes_fixed):
        alpha, _ = approximate_gamma_iqr(
            quant_lower, quant_upper, nodes_lower[i], nodes_upper[i]
        )
        beta = gammainc_inv(alpha + 1, 0.5) if use_median else (alpha + 1)
        beta /= nodes_adjust[i]  # choose rate so as to keep mean or median
        posteriors_adjust[i] = alpha, beta

    timescale = np.zeros((3, adjust.size))
    timescale[0] = origin
    timescale[1] = scales
    timescale[2] = adjust

    return posteriors_adjust, timescale


# @numba.njit(_f1w(_f1r, _f2r, _i1r, _i1r))
# def scale_time_by_mutations_constr(
#   nodes_time, likelihoods, constraints, edges_parent, edges_child
# ):
#    """
#    Rescale node ages so that the instantaneous mutation rate is constant.
#    Edges with a negative duration are ignored when calculating the total
#    rate.
#
#    ..note::
#        A node is considered fixed if its lower and upper bounds (in
#        `constraints`) are equal.  The ages of fixed nodes are conditioned
#        upon; other finite bounds are currently ignored.
#
#    :param np.ndarray nodes_time: array of node ages
#    :param np.ndarray likelihoods: edges are rows; mutation counts and
#        mutational span are columns
#    :param np.ndarray constraints: nodes are rows; lower and upper bounds
#        on age are columns. If bounds are equal, a node is considered
#        fixed.
#    :param np.ndarray edges_parent: node index for the parent of each edge
#    :param np.ndarray edges_child: node index for the child of each edge
#    """
#
#    assert edges_parent.size == edges_child.size == likelihoods.shape[0]
#    assert nodes_time.size == constraints.shape[0]
#    assert likelihoods.shape[1] == constraints.shape[1] == 2
#
#    nodes_time = nodes_time.copy()
#    nodes_fixed = constraints[:, 0] == constraints[:, 1]
#    nodes_time[nodes_fixed] = constraints[nodes_fixed, 0]
#    nodes_order = np.argsort(nodes_time)
#    assert nodes_fixed[nodes_order[0]], "Youngest node must be fixed"
#
#    # index node by unique time breaks
#    nodes_index = np.zeros(nodes_time.size, dtype=np.int32)
#    epoch_breaks = [0.0]
#    epoch_clamps = [0]
#    k = 0
#    for i, j in zip(nodes_order[1:], nodes_order[:-1]):
#        if nodes_time[i] > nodes_time[j]:
#            epoch_breaks.append(nodes_time[i])
#            k += 1
#        nodes_index[i] = k
#        if nodes_fixed[i]:
#            epoch_clamps.append(k)
#    epoch_breaks = np.array(epoch_breaks)
#    epoch_length = np.diff(epoch_breaks)
#    num_epochs = epoch_length.size
#    epoch_clamps = np.append(epoch_clamps, num_epochs)
#
#    # instantaneous mutation rate per edge
#    edges_length = nodes_time[edges_parent] - nodes_time[edges_child]
#    edges_subset = edges_length > 0
#    edges_counts = likelihoods.copy()
#    edges_counts[edges_subset, 0] /= edges_length[edges_subset]
#
#    # accumulate edge overlap with each time interval
#    leafw_counts = np.zeros((num_epochs, 2))
#    rootw_counts = np.zeros((num_epochs, 2))
#    for e in np.flatnonzero(edges_subset):
#        p, c = edges_parent[e], edges_child[e]
#        a, b = nodes_index[c] - 1, nodes_index[p]
#        if a >= 0:
#            leafw_counts[a] += edges_counts[e]
#        if b < num_epochs:
#            rootw_counts[b] += edges_counts[e]
#    total_counts = np.sum(edges_counts[edges_subset], axis=0)
#    rootw_counts[:, 0] = rootw_counts[:, 0].cumsum()
#    rootw_counts[:, 1] = rootw_counts[:, 1].cumsum()
#    leafw_counts[::-1, 0] = leafw_counts[::-1, 0].cumsum()
#    leafw_counts[::-1, 1] = leafw_counts[::-1, 1].cumsum()
#    epoch_counts = total_counts[np.newaxis, :] - rootw_counts - leafw_counts
#    assert np.all(epoch_counts[:, 1] > 0), "No overlap between epoch and any edge"
#
#    # rescale time such that mutation density is constant
#    epoch_adjust = np.full(num_epochs + 1, np.nan)
#    for i, j in zip(epoch_clamps[:-1], epoch_clamps[1:]):
#        # I think the indexing may be off by 1 here
#        if j == num_epochs: # poisson rescaling
#            epoch_length[i:] = np.cumsum(
#                epoch_length[i:] * epoch_counts[i:, 0] / epoch_counts[i:, 1]
#            )
#        else: # multinomial rescaling
#            # not sure if this is right, think it through
#            epoch_length[i:j] = epoch_counts[i:j, 0] / epoch_counts[i:j, 1]
#            epoch_length[i:j] /= np.sum(epoch_length[i:j])
#        # ??? = epoch_breaks[i] probably? the lhs indexing is wrong for the last clamp?
#        epoch_adjust[i:j] = np.append(???, ??? + epoch_length[i:j])
#
#    return epoch_adjust[nodes_index]


# TODO: test rescaling with fixed nodes
# TODO: test constrained least squares with fixed nodes

# ---- delete

# def mutation_scaling(nodes_time, rescaled_nodes_time):
#     # collapse intervals that have zero length
#     node_order = np.argsort(nodes_time)
#     time_intervals = np.diff(nodes_time[node_order])
#     rescaled_time_intervals = np.diff(rescaled_nodes_time[node_order])
#
#     nodes_index = np.zeros(nodes_time.size, dtype=np.int32)
#     time_breaks = [0.0]
#     rescaled_time_breaks = [0.0]
#     rate = []
#
#     time = 0
#     rescaled_time = 0
#     interval = 0
#     rescaled_interval = 0
#     k = 0
#     for i, (x, y) in enumerate(zip(time_intervals, rescaled_time_intervals)):
#         time += x
#         interval += x
#         rescaled_time += y
#         rescaled_interval += y
#         nodes_index[i + 1] = k
#         if interval > 0 and rescaled_interval > 0:
#             rate.append(interval / rescaled_interval)
#             time_breaks.append(time)
#             rescaled_time_breaks.append(rescaled_time)
#             interval = 0
#             rescaled_interval = 0
#             k += 1
#
#     rate = np.array(rate)
#     time_breaks = np.array(time_breaks)
#     rescaled_time_breaks = np.array(rescaled_time_breaks)
#     assert np.all(rate > 0)
#
#     return time_breaks, rate, rescaled_time_breaks
#
#
# @numba.njit(_f2w(_f2r, _f2r, _f2r, _i1r, _i1r))
# def normalise_posteriors(
#     posteriors, likelihoods, constraints, edges_child, edges_parent
# ):
#     """
#     Estimate a piecewise-constant time rescaling using the mutational clock,
#     then match moments to update the posteriors given the time rescaling
#     """
#
#     # tol = 1e-10
#
#     free = constraints[:, 0] != constraints[:, 1]
#     nodes_time = np.zeros(free.size)
#     nodes_time[free] = (posteriors[free, 0] + 1) / posteriors[free, 1]
#     nodes_time[~free] = constraints[~free, 0]
#     rescaled_time = scale_time_by_mutations(
#         nodes_time, likelihoods, edges_parent, edges_child
#     )
#
#     # collapse intervals that have zero length
#     nodes_order = np.argsort(nodes_time)
#     nodes_index = np.zeros(nodes_time.size, dtype=np.int32)
#     breaks = [0.0]
#     skaerb = [0.0]
#     scale = []
#     dx, dy = 0.0, 0.0
#     num_breaks = 0
#     for i, j in zip(nodes_order[1:], nodes_order[:-1]):
#         dx += nodes_time[i] - nodes_time[j]
#         dy += rescaled_time[i] - rescaled_time[j]
#         if dx > 0 and dy > 0:
#             breaks.append(dx + breaks[num_breaks])
#             skaerb.append(dy + skaerb[num_breaks])
#             scale.append(dx / dy)
#             dx, dy = 0.0, 0.0
#             num_breaks += 1
#         nodes_index[i] = num_breaks
#     breaks[num_breaks] = np.inf
#     skaerb[num_breaks] = np.inf
#     num_breaks += 1
#
#     # integrate posterior moments over piecewise-constant time-rescaling
#     lo = np.zeros(3)  # move into parallel loop
#     up = np.zeros(3)
#     new_posteriors = np.zeros(posteriors.shape)
#     for i in np.flatnonzero(free):
#         shape, rate = posteriors[i, 0] + 1, posteriors[i, 1]
#         sc = np.array([1.0, shape / rate, shape * (shape + 1) / rate**2])
#
#         mn = 0.0
#         sq = 0.0
#
#         lo[:] = 0.0
#         for j in range(num_breaks - 1):
#             u = 1.0 / scale[j]
#             for s in range(3):
#                 up[s] = gammainc(shape + s, rate * breaks[j + 1])
#             di = sc * (up - lo)
#             dt = skaerb[j] - u * breaks[j]
#             mn += di[0] * dt + di[1] * u
#             sq += di[0] * dt**2 + 2 * di[1] * u * dt + di[2] * u**2
#             lo[:] = up[:]
#
#         # k = nodes_index[i]
#         # for s in range(3):
#         #    lo[s] = gammainc(shape + s, rate * breaks[k])
#         # for j in range(k, num_breaks - 1):
#         #    u = 1.0 / scale[j]
#         #    for s in range(3):
#         #        up[s] = gammainc(shape + s, rate * breaks[j + 1])
#         #    di = sc * (up - lo)
#         #    dt = skaerb[j] - u * breaks[j]
#         #    mn += di[0] * dt + di[1] * u
#         #    sq += di[0] * dt ** 2 + 2 * di[1] * u * dt + di[2] * u ** 2
#         #    if 1 - up[0] < tol: # check upper tail
#         #        break
#         #    lo[:] = up[:]
#
#         # for s in range(3):
#         #    up[s] = gammainc(shape + s, rate * breaks[k])
#         # for j in range(k, 0, -1):
#         #    u = 1.0 / scale[j - 1]
#         #    for s in range(3):
#         #        lo[s] = gammainc(shape + s, rate * breaks[j - 1])
#         #    di = sc * (up - lo)
#         #    dt = skaerb[j - 1] - u * breaks[j - 1]
#         #    mn += di[0] * dt + di[1] * u
#         #    sq += di[0] * dt ** 2 + 2 * di[1] * u * dt + di[2] * u ** 2
#         #    if lo[0] < tol: # check lower tail
#         #        break
#         #    up[:] = lo[:]
#
#         va = sq - mn**2
#         new_posteriors[i] = [mn**2 / va, mn / va]
#
#     return new_posteriors
