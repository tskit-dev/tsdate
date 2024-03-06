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
Utility functions for tsdate. Many of these can be removed when tskit is updated to
a more recent version which has the functionality built-in
"""
import json
import logging

import numba
import numpy as np
import tskit
from numba.types import UniTuple as _unituple

from . import provenance
from .approx import _b1r
from .approx import _f
from .approx import _f1r
from .approx import _f1w
from .approx import _f2r
from .approx import _f2w
from .approx import _i
from .approx import _i1r
from .approx import _i1w
from .hypergeo import _gammainc as gammainc


logger = logging.getLogger(__name__)


def reduce_to_contemporaneous(ts):
    """
    Simplify the ts to only the contemporaneous samples, and return the new ts + node map
    """
    samples = ts.samples()
    contmpr_samples = samples[ts.nodes_time[samples] == 0]
    return ts.simplify(
        contmpr_samples,
        map_nodes=True,
        keep_unary=True,
        filter_populations=False,
        filter_sites=False,
        record_provenance=False,
        filter_individuals=False,
    )


def preprocess_ts(
    tree_sequence,
    *,
    minimum_gap=None,
    remove_telomeres=None,
    filter_populations=False,
    filter_individuals=False,
    filter_sites=False,
    delete_intervals=None,
    **kwargs,
):
    """
    Function to prepare tree sequences for dating by removing gaps without sites and
    simplifying the tree sequence. Large regions without data can cause
    overflow/underflow errors in the inside-outside algorithm and poor performance more
    generally. Removed regions are recorded in the provenance of the resulting tree
    sequence.

    :param tskit.TreeSequence tree_sequence: The input tree sequence
        to be preprocessed.
    :param float minimum_gap: The minimum gap between sites to remove from the tree
        sequence. Default: ``None`` treated as ``1000000``
    :param bool remove_telomeres: Should all material before the first site and after the
        last site be removed, regardless of the length. Default: ``None`` treated as
        ``True``
    :param bool filter_populations: parameter passed to the ``tskit.simplify``
        command. Unlike calling that command directly, this defaults to ``False``, such
        that all populations in the tree sequence are kept.
    :param bool filter_individuals: parameter passed to the ``tskit.simplify``
        command. Unlike calling that command directly, this defaults to ``False``, such
        that all individuals in the tree sequence are kept
    :param bool filter_sites: parameter passed to the ``tskit.simplify``
        command. Unlike calling that command directly, this defaults to ``False``, such
        that all sites in the tree sequence are kept
    :param array_like delete_intervals: A list (start, end) pairs describing the
        genomic intervals (gaps) to delete. This is usually left as ``None``
        (the default) in which case ``minimum_gap`` and ``remove_telomeres`` are used
        to determine the gaps to remove, and the calculated intervals are recorded in
        the provenance of the resulting tree sequence.
    :param \\**kwargs: All further keyword arguments are passed to the ``tskit.simplify``
        command.

    :return: A tree sequence with gaps removed.
    :rtype: tskit.TreeSequence
    """

    logger.info("Beginning preprocessing")
    logger.info(f"Minimum_gap: {minimum_gap} and remove_telomeres: {remove_telomeres}")
    if delete_intervals is not None and (
        minimum_gap is not None or remove_telomeres is not None
    ):
        raise ValueError(
            "Cannot specify both delete_intervals and minimum_gap/remove_telomeres"
        )

    tables = tree_sequence.dump_tables()
    sites = tables.sites.position[:]
    if delete_intervals is None:
        if minimum_gap is None:
            minimum_gap = 1000000
        if remove_telomeres is None:
            remove_telomeres = True

        if tree_sequence.num_sites < 1:
            raise ValueError("Invalid tree sequence: no sites present")
        delete_intervals = []
        if remove_telomeres:
            first_site = sites[0] - 1
            if first_site > 0:
                delete_intervals.append([0, first_site])
                logger.info(
                    "REMOVING TELOMERE: Snip topology "
                    "from 0 to first site at {}.".format(first_site)
                )
            last_site = sites[-1] + 1
            sequence_length = tables.sequence_length
            if last_site < sequence_length:
                delete_intervals.append([last_site, sequence_length])
                logger.info(
                    "REMOVING TELOMERE: Snip topology "
                    "from {} to end of sequence at {}.".format(
                        last_site, sequence_length
                    )
                )
        gaps = sites[1:] - sites[:-1]
        threshold_gaps = np.where(gaps >= minimum_gap)[0]
        for gap in threshold_gaps:
            gap_start = sites[gap] + 1
            gap_end = sites[gap + 1] - 1
            if gap_end > gap_start:
                logger.info(
                    "Gap Size is {}. Snip topology "
                    "from {} to {}.".format(gap_end - gap_start, gap_start, gap_end)
                )
                delete_intervals.append([gap_start, gap_end])
        delete_intervals = sorted(delete_intervals, key=lambda x: x[0])
    if len(delete_intervals) > 0:
        tables.delete_intervals(
            delete_intervals, simplify=False, record_provenance=False
        )
        tables.simplify(
            filter_populations=filter_populations,
            filter_individuals=filter_individuals,
            filter_sites=filter_sites,
            record_provenance=False,
            **kwargs,
        )
    else:
        logger.info("No gaps to remove")
        tables.simplify(
            filter_populations=filter_populations,
            filter_individuals=filter_individuals,
            filter_sites=filter_sites,
            record_provenance=False,
            **kwargs,
        )
    provenance.record_provenance(
        tables,
        "preprocess_ts",
        minimum_gap=minimum_gap,
        remove_telomeres=remove_telomeres,
        filter_populations=filter_populations,
        filter_individuals=filter_individuals,
        filter_sites=filter_sites,
        delete_intervals=delete_intervals,
    )
    return tables.tree_sequence()


def nodes_time_unconstrained(tree_sequence):
    """
    Return the unconstrained node times for every node in a tree sequence that has
    been dated using ``tsdate`` with the inside-outside algorithm (these times are
    stored in the node metadata). Will produce an error if the tree sequence does
    not contain this information.
    """
    nodes_time = tree_sequence.nodes_time.copy()
    metadata = tree_sequence.tables.nodes.metadata
    metadata_offset = tree_sequence.tables.nodes.metadata_offset
    for index, met in enumerate(tskit.unpack_bytes(metadata, metadata_offset)):
        if index not in tree_sequence.samples():
            try:
                nodes_time[index] = json.loads(met.decode())["mn"]
            except (KeyError, json.decoder.JSONDecodeError):
                raise ValueError(
                    "Tree Sequence must be tsdated with the Inside-Outside Method."
                )
    return nodes_time


def sites_time_from_ts(
    tree_sequence, *, unconstrained=True, node_selection="child", min_time=1
):
    """
    Returns an estimated "time" for each site. This is the estimated age of the oldest
    MRCA which possesses a derived variant at that site, and is useful for performing
    (re)inference of a tree sequence. It is calculated from the ages of nodes, with the
    appropriate nodes identified by the position of mutations in the trees.

    If node times in the tree sequence have been estimated by ``tsdate`` using the
    inside-outside algorithm, then as well as a time in the tree sequence, nodes will
    store additional time estimates that have not been explictly constrained by the
    tree topology. By default, this function tries to use these "unconstrained" times,
    although this is likely to fail (with a warning) on tree sequences that have not
    been processed by ``tsdate``: in this case the standard node times can be used by
    setting ``unconstrained=False``.

    The concept of a site time is meaningless for non-variable sites, and
    so the returned time for these sites is ``np.nan`` (note that this is not exactly
    the same as tskit.UNKNOWN_TIME, which marks sites that could have a meaningful time
    but whose time estimate is unknown).

    :param tskit.TreeSequence tree_sequence: The input tree sequence.
    :param bool unconstrained: Use estimated node times which have not been constrained
        by tree topology. If ``True`` (default), this requires a tree sequence which has
        been dated using the ``tsdate`` inside-outside algorithm. If this is not the
        case, specify ``False`` to use the standard tree sequence node times.
    :param str node_selection: Defines how site times are calculated from the age of
        the upper and lower nodes that bound each mutation at the site. Options are
        "child", "parent", "arithmetic" or "geometric", with the following meanings

        * ``'child'`` (default): the site time is the age of the oldest node
          *below* each mutation at the site
        * ``'parent'``: the site time is the age of the oldest node *above* each
          mutation at the site
        * ``'arithmetic'``: the arithmetic mean of the ages of the node above and the
          node below each mutation is calculated; the site time is the oldest
          of these means.
        * ``'geometric'``: the geometric mean of the ages of the node above and the
          node below each mutation is calculated; the site time is the oldest
          of these means

    :param float min_time: A site time of zero implies that no MRCA in the past
        possessed the derived variant, so the variant cannot be used for inferring
        relationships between the samples. To allow all variants to be potentially
        available for inference, if a site time would otherwise be calculated as zero
        (for example, where the ``mutation_age`` parameter is "child" or "geometric"
        and all mutations at a site are associated with leaf nodes), a minimum site
        greater than 0 is recommended. By default this is set to 1, which is generally
        reasonable for times measured in generations or years, although it is also
        fine to set this to a small epsilon value.

    :return: Array of length tree_sequence.num_sites with estimated time of each site
    :rtype: numpy.ndarray(dtype=np.float64)
    """
    if tree_sequence.num_sites < 1:
        raise ValueError("Invalid tree sequence: no sites present")
    if node_selection not in ["arithmetic", "geometric", "child", "parent"]:
        raise ValueError(
            "The node_selection parameter must be "
            "'child', 'parent', 'arithmetic', or 'geometric'"
        )
    if unconstrained:
        try:
            nodes_time = nodes_time_unconstrained(tree_sequence)
        except ValueError as e:
            e.args += "Try calling sites_time_from_ts() with unconstrained=False."
            raise
    else:
        nodes_time = tree_sequence.nodes_time
    sites_time = np.full(tree_sequence.num_sites, np.nan)

    for tree in tree_sequence.trees():
        for site in tree.sites():
            for mutation in site.mutations:
                parent_node = tree.parent(mutation.node)
                if node_selection == "child" or parent_node == tskit.NULL:
                    age = nodes_time[mutation.node]
                else:
                    parent_age = nodes_time[parent_node]
                    if node_selection == "parent":
                        age = parent_age
                    elif node_selection == "arithmetic":
                        age = (nodes_time[mutation.node] + parent_age) / 2
                    elif node_selection == "geometric":
                        age = np.sqrt(nodes_time[mutation.node] * parent_age)
                if np.isnan(sites_time[site.id]) or sites_time[site.id] < age:
                    sites_time[site.id] = age
            if sites_time[site.id] < min_time:
                sites_time[site.id] = min_time
    return sites_time


def add_sampledata_times(samples, sites_time):
    """
    Return a tsinfer.SampleData file with estimated times associated with sites.
    Ensures that each site's time is at least as old as the oldest historic sample
    carrying a derived allele at that site.

    :param tsinfer.formats.SampleData samples: A tsinfer SampleData object to
        add site times to. Any historic individuals in this SampleData file are used to
        constrain site times.

    :return: A tsinfer.SampleData file
    :rtype: tsinfer.SampleData
    """
    if samples.num_sites != len(sites_time):
        raise ValueError(
            "sites_time should contain the same number of sites as the SampleData file"
        )
    # Get constraints from ancients
    sites_bound = samples.min_site_times(individuals_only=True)
    # Use maximum of constraints and estimated site times
    sites_time = np.maximum(sites_time, sites_bound)
    copy = samples.copy()
    copy.sites_time[:] = sites_time
    copy.finalise()
    return copy


def mutation_span_array(tree_sequence):
    """Extract mutation counts and spans per edge into a two-column array"""
    mutation_spans = np.zeros((tree_sequence.num_edges, 2))
    for mut in tree_sequence.mutations():
        if mut.edge != tskit.NULL:
            mutation_spans[mut.edge, 0] += 1
    for edge in tree_sequence.edges():
        mutation_spans[edge.id, 1] = edge.span
    return mutation_spans


@numba.njit(_unituple(_i1w, 4)(_i1r, _i1r, _f1r, _f1r, _i1r, _b1r))
def _split_disjoint_nodes(
    edges_parent, edges_child, edges_left, edges_right, mutations_edge, nodes_exclude
):
    """
    Split disconnected regions of nodes into separate nodes.

    Returns updated edges_parent, edges_child, mutations_node, and indices indicating
    from which original node the new nodes are derived.
    """
    assert edges_parent.size == edges_child.size == edges_left.size == edges_right.size
    num_edges = edges_parent.size
    num_nodes = nodes_exclude.size
    num_mutations = mutations_edge.size

    # For each edge, check whether parent/child is separated by a gap from the
    # previous edge involving either parent/child. Label disconnected segments
    # per node by integers starting at zero.
    edges_order = np.argsort(edges_left)
    # TODO: is a sort really needed here?
    edges_segments = np.full((2, num_edges), -1, dtype=np.int32)
    nodes_segments = np.full(num_nodes, -1, dtype=np.int32)
    nodes_right = np.full(nodes_exclude.size, -np.inf, dtype=np.float64)
    for e in edges_order:
        nodes = edges_parent[e], edges_child[e]
        for i, n in enumerate(nodes):
            if nodes_exclude[n]:
                continue
            nodes_segments[n] += edges_left[e] > nodes_right[n]
            edges_segments[i, e] = nodes_segments[n]
            nodes_right[n] = max(nodes_right[n], edges_right[e])

    # Create "nodes_segments[i]" supplementary nodes by copying node "i".
    # Store the id of the first supplement for each node in "nodes_map".
    nodes_order = [i for i in range(num_nodes)]
    nodes_map = np.full(num_nodes, -1, dtype=np.int32)
    for i, s in enumerate(nodes_segments):
        for j in range(s):
            if j == 0:
                nodes_map[i] = num_nodes
            nodes_order.append(i)
            num_nodes += 1
    nodes_order = np.array(nodes_order, dtype=np.int32)

    # Relabel the nodes on each edge given "nodes_map"
    for e in edges_order:
        nodes = edges_parent[e], edges_child[e]
        for i, n in enumerate(nodes):
            if edges_segments[i, e] > 0:
                edges_segments[i, e] += nodes_map[n] - 1
            else:
                edges_segments[i, e] = n
    edges_parent, edges_child = edges_segments[0, ...], edges_segments[1, ...]

    # Relabel node under each mutation
    mutations_node = np.full(num_mutations, tskit.NULL, dtype=np.int32)
    for i, e in enumerate(mutations_edge):
        if e != tskit.NULL:
            mutations_node[i] = edges_child[e]

    return edges_parent, edges_child, mutations_node, nodes_order


def split_disjoint_nodes(ts):
    """
    For each non-sample node, split regions separated by gaps into distinct
    nodes.

    Where there are multiple disconnected regions, the leftmost one is assigned
    the ID of the original node, and the remainder are assigned new node IDs.
    Population, flags, individual, time, and metadata are all copied into the
    new nodes.
    """

    mutations_edge = np.full(ts.num_mutations, tskit.NULL, dtype=np.int32)
    for m in ts.mutations():
        mutations_edge[m.id] = m.edge

    node_is_sample = np.bitwise_and(ts.nodes_flags, tskit.NODE_IS_SAMPLE).astype(bool)
    edges_parent, edges_child, mutations_node, nodes_order = _split_disjoint_nodes(
        ts.edges_parent,
        ts.edges_child,
        ts.edges_left,
        ts.edges_right,
        mutations_edge,
        node_is_sample,
    )

    # TODO: correctly handle mutations above root (m.edge == tskit.NULL)
    nonsegregating = np.flatnonzero(mutations_node == tskit.NULL)
    mutations_node[nonsegregating] = ts.mutations_node[nonsegregating]

    tables = ts.dump_tables()
    tables.nodes.set_columns(
        flags=tables.nodes.flags[nodes_order],
        time=tables.nodes.time[nodes_order],
        population=tables.nodes.population[nodes_order],
        individual=tables.nodes.individual[nodes_order],
    )
    # TODO: copy existing metadata for original nodes
    # TODO: add new metadata indicating origin for split nodes
    # TODO: add flag for split nodes
    tables.edges.parent = edges_parent
    tables.edges.child = edges_child
    tables.mutations.node = mutations_node
    tables.sort()

    return tables.tree_sequence()


@numba.njit(_f1w(_f1r, _b1r, _i1r, _i1r, _f, _i))
def _constrain_ages(
    nodes_time, nodes_fixed, edges_parent, edges_child, epsilon, max_iterations
):
    """
    Approximate least squares solution to the positive branch length
    constraint, using the method of alternating projections. Loosely based on
    Dykstra's algorithm, see:

    Dykstra RL, "An algorithm for restricted least squares regression", JASA
    1983
    """
    assert nodes_time.size == nodes_fixed.size
    assert edges_parent.size == edges_child.size

    num_edges = edges_parent.size
    nodes_time = nodes_time.copy()
    edges_cavity = np.zeros((num_edges, 2))
    for _ in range(max_iterations):  # method of alternating projections
        if np.all(nodes_time[edges_parent] - nodes_time[edges_child] > 0):
            return nodes_time
        for e in range(num_edges):
            p, c = edges_parent[e], edges_child[e]
            nodes_time[c] -= edges_cavity[e, 0]
            nodes_time[p] -= edges_cavity[e, 1]
            adjustment = nodes_time[c] - nodes_time[p]  # + epsilon
            edges_cavity[e, :] = 0.0
            if adjustment > 0:
                assert not nodes_fixed[p]  # TODO: no reason not to support this
                edges_cavity[e, 0] = 0 if nodes_fixed[c] else -adjustment / 2
                edges_cavity[e, 1] = adjustment if nodes_fixed[c] else adjustment / 2
            nodes_time[c] += edges_cavity[e, 0]
            nodes_time[p] += edges_cavity[e, 1]
    # print(
    #   "min length:", np.min(nodes_time[edges_parent] - nodes_time[edges_child])
    # )
    for e in range(num_edges):  # force constraint
        p, c = edges_parent[e], edges_child[e]
        if nodes_time[c] >= nodes_time[p]:
            nodes_time[p] = nodes_time[c] + epsilon

    return nodes_time


def constrain_ages(ts, nodes_time, epsilon=1e-6, max_iterations=0):
    """
    Use a hybrid approach to adjust node times such that branch lengths are
    positive. The first pass iteratively solves a constrained least squares
    problem that seeks to find constrained ages as close as possible to
    unconstrained ages. Progress is initially fast but typically becomes quite
    slow, so after a fixed number of iterations the iterative algorithm
    terminates and the constraint is forced.

    :param tskit.TreeSequence ts: The input tree sequence, with arbitrary node
        times.
    :param np.ndarray nodes_times: Unconstrained node ages to inject into the
        tree sequence.
    :param float epsilon: The minimum allowed branch length when forcing
        positive branch lengths.
    :param int max_iterations: The number of iterations of alternating
        projections before forcing positive branch lengths.

    :return np.ndarray: Constrained node ages
    """

    assert nodes_time.size == ts.num_nodes
    assert epsilon >= 0
    assert max_iterations >= 0

    node_is_sample = np.bitwise_and(ts.nodes_flags, tskit.NODE_IS_SAMPLE).astype(bool)
    constrained_nodes_time = _constrain_ages(
        nodes_time,
        node_is_sample,
        ts.edges_parent,
        ts.edges_child,
        epsilon,
        max_iterations,
    )

    return constrained_nodes_time


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


@numba.njit(_f1w(_f1r, _f2r, _i1r, _i1r))
def scale_time_by_mutations(nodes_time, likelihoods, edges_parent, edges_child):
    """
    Rescale node ages so that the instantaneous mutation rate is constant.
    Edges with a negative duration are ignored when calculating the total
    rate.

    :param np.ndarray nodes_time: array of node ages
    :param np.ndarray likelihoods: edges are rows; mutation
        counts and mutational span are columns
    :param np.ndarray edges_parent: node index for the parent of each edge
    :param np.ndarray edges_child: node index for the child of each edge
    """

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
    epoch_counts[:, 0] = np.cumsum(epoch_counts[:, 0])
    epoch_counts[:, 1] = np.cumsum(epoch_counts[:, 1])
    assert np.all(epoch_counts[:, 1] > 0)

    # rescale time such that mutation density is constant
    epoch_scales = epoch_counts[:, 0] / epoch_counts[:, 1]
    epoch_adjust = np.append(0, np.cumsum(epoch_length * epoch_scales))

    return epoch_adjust[nodes_index]


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


def mutation_scaling(nodes_time, rescaled_nodes_time):
    # collapse intervals that have zero length
    node_order = np.argsort(nodes_time)
    time_intervals = np.diff(nodes_time[node_order])
    rescaled_time_intervals = np.diff(rescaled_nodes_time[node_order])

    nodes_index = np.zeros(nodes_time.size, dtype=np.int32)
    time_breaks = [0.0]
    rescaled_time_breaks = [0.0]
    rate = []

    time = 0
    rescaled_time = 0
    interval = 0
    rescaled_interval = 0
    k = 0
    for i, (x, y) in enumerate(zip(time_intervals, rescaled_time_intervals)):
        time += x
        interval += x
        rescaled_time += y
        rescaled_interval += y
        nodes_index[i + 1] = k
        if interval > 0 and rescaled_interval > 0:
            rate.append(interval / rescaled_interval)
            time_breaks.append(time)
            rescaled_time_breaks.append(rescaled_time)
            interval = 0
            rescaled_interval = 0
            k += 1

    rate = np.array(rate)
    time_breaks = np.array(time_breaks)
    rescaled_time_breaks = np.array(rescaled_time_breaks)
    assert np.all(rate > 0)

    return time_breaks, rate, rescaled_time_breaks


@numba.njit(_f2w(_f2r, _f2r, _f2r, _i1r, _i1r))
def normalise_posteriors(
    posteriors, likelihoods, constraints, edges_child, edges_parent
):
    """
    Estimate a piecewise-constant time rescaling using the mutational clock,
    then match moments to update the posteriors given the time rescaling
    """

    # tol = 1e-10

    free = constraints[:, 0] != constraints[:, 1]
    nodes_time = np.zeros(free.size)
    nodes_time[free] = (posteriors[free, 0] + 1) / posteriors[free, 1]
    nodes_time[~free] = constraints[~free, 0]
    rescaled_time = scale_time_by_mutations(
        nodes_time, likelihoods, edges_parent, edges_child
    )

    # collapse intervals that have zero length
    nodes_order = np.argsort(nodes_time)
    nodes_index = np.zeros(nodes_time.size, dtype=np.int32)
    breaks = [0.0]
    skaerb = [0.0]
    scale = []
    dx, dy = 0.0, 0.0
    num_breaks = 0
    for i, j in zip(nodes_order[1:], nodes_order[:-1]):
        dx += nodes_time[i] - nodes_time[j]
        dy += rescaled_time[i] - rescaled_time[j]
        if dx > 0 and dy > 0:
            breaks.append(dx + breaks[num_breaks])
            skaerb.append(dy + skaerb[num_breaks])
            scale.append(dx / dy)
            dx, dy = 0.0, 0.0
            num_breaks += 1
        nodes_index[i] = num_breaks
    breaks[num_breaks] = np.inf
    skaerb[num_breaks] = np.inf
    num_breaks += 1

    # integrate posterior moments over piecewise-constant time-rescaling
    lo = np.zeros(3)  # move into parallel loop
    up = np.zeros(3)
    new_posteriors = np.zeros(posteriors.shape)
    for i in np.flatnonzero(free):
        shape, rate = posteriors[i, 0] + 1, posteriors[i, 1]
        sc = np.array([1.0, shape / rate, shape * (shape + 1) / rate**2])

        mn = 0.0
        sq = 0.0

        lo[:] = 0.0
        for j in range(num_breaks - 1):
            u = 1.0 / scale[j]
            for s in range(3):
                up[s] = gammainc(shape + s, rate * breaks[j + 1])
            di = sc * (up - lo)
            dt = skaerb[j] - u * breaks[j]
            mn += di[0] * dt + di[1] * u
            sq += di[0] * dt**2 + 2 * di[1] * u * dt + di[2] * u**2
            lo[:] = up[:]

        # k = nodes_index[i]
        # for s in range(3):
        #    lo[s] = gammainc(shape + s, rate * breaks[k])
        # for j in range(k, num_breaks - 1):
        #    u = 1.0 / scale[j]
        #    for s in range(3):
        #        up[s] = gammainc(shape + s, rate * breaks[j + 1])
        #    di = sc * (up - lo)
        #    dt = skaerb[j] - u * breaks[j]
        #    mn += di[0] * dt + di[1] * u
        #    sq += di[0] * dt ** 2 + 2 * di[1] * u * dt + di[2] * u ** 2
        #    if 1 - up[0] < tol: # check upper tail
        #        break
        #    lo[:] = up[:]

        # for s in range(3):
        #    up[s] = gammainc(shape + s, rate * breaks[k])
        # for j in range(k, 0, -1):
        #    u = 1.0 / scale[j - 1]
        #    for s in range(3):
        #        lo[s] = gammainc(shape + s, rate * breaks[j - 1])
        #    di = sc * (up - lo)
        #    dt = skaerb[j - 1] - u * breaks[j - 1]
        #    mn += di[0] * dt + di[1] * u
        #    sq += di[0] * dt ** 2 + 2 * di[1] * u * dt + di[2] * u ** 2
        #    if lo[0] < tol: # check lower tail
        #        break
        #    up[:] = lo[:]

        va = sq - mn**2
        new_posteriors[i] = [mn**2 / va, mn / va]

    return new_posteriors
