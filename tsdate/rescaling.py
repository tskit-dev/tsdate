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

import _tsdate
import numpy as np
import tskit

from .util import mutation_span_array  # NOQA: F401


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
        assert node_is_sample.size == ts.num_nodes

    return _tsdate.count_mutations(
        nodes_sample=node_is_sample,
        mutations_node=ts.mutations_node,
        mutations_position=ts.sites_position[ts.mutations_site],
        edges_parent=ts.edges_parent,
        edges_child=ts.edges_child,
        edges_left=ts.edges_left,
        edges_right=ts.edges_right,
        indexes_insert=ts.indexes_edge_insertion_order,
        indexes_remove=ts.indexes_edge_removal_order,
        sequence_length=ts.sequence_length,
        size_biased=size_biased,
    )


mutational_area = _tsdate.mutational_area


mutational_timescale = _tsdate.mutational_timescale


def _check_increasing_breaks(original_breaks, rescaled_breaks):
    if not np.all(np.diff(rescaled_breaks) > 0) or not np.all(
        np.diff(original_breaks) > 0
    ):
        raise ValueError(
            "Rescaling produced non-increasing time breakpoints. "
            "Try setting `rescaling_intervals` to a smaller value "
            "(e.g. half its current value)."
        )


def piecewise_scale_posterior(
    posteriors,
    posteriors_fixed,
    original_breaks,
    rescaled_breaks,
    quantile_width,
    max_shape,
):
    _check_increasing_breaks(original_breaks, rescaled_breaks)
    return _tsdate.piecewise_scale_posterior(
        posteriors=posteriors,
        posteriors_fixed=posteriors_fixed,
        original_breaks=original_breaks,
        rescaled_breaks=rescaled_breaks,
        quantile_width=quantile_width,
        max_shape=max_shape,
    )


def piecewise_scale_point_estimate(
    point_estimate, point_fixed, original_breaks, rescaled_breaks
):
    _check_increasing_breaks(original_breaks, rescaled_breaks)
    return _tsdate.piecewise_scale_point_estimate(
        point_estimate=point_estimate,
        point_fixed=point_fixed,
        original_breaks=original_breaks,
        rescaled_breaks=rescaled_breaks,
    )


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
            nodes_time=nodes_time,
            edges_likelihood=mutations_span,
            nodes_fixed=constraints,
            edges_parent=ts.edges_parent,
            edges_child=ts.edges_child,
            max_intervals=num_intervals,
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
