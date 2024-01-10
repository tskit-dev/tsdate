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
from .approx import _f1r
from .approx import _i1r
from .approx import _i1w


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


# TODO: numba.njit
def _split_root_nodes(ts):
    """
    Split roots whenever the set of children changes. Nodes will only be split
    on the interior of the intervals where they are roots.

    Returns new edges (parent, child, left, right) and the original ids for
    each node.
    """

    num_nodes = ts.num_nodes
    num_edges = ts.num_edges

    # Find locations where root node changes
    roots_node = []
    roots_breaks = []
    last_root = None
    for t in ts.trees():
        root = tskit.NULL if t.num_edges == 0 else t.root
        if root != last_root:
            roots_node.append(root)
            roots_breaks.append(t.interval.left)
        last_root = root
    roots_breaks.append(ts.sequence_length)
    roots_node = np.array(roots_node, dtype=np.int32)
    roots_breaks = np.array(roots_breaks, dtype=np.float64)

    # Segment roots at edge additions/removals
    add_breaks = {n: list() for n in roots_node if n != tskit.NULL}
    for e in range(num_edges):
        p = ts.edges_parent[e]
        if p in add_breaks:
            for x in (ts.edges_left[e], ts.edges_right[e]):
                i = np.searchsorted(roots_breaks, x, side="right") - 1
                if x == ts.sequence_length:
                    continue
                if (
                    p == roots_node[i] and x > roots_breaks[i]
                ):  # store *internal* breaks for root segments
                    add_breaks[p].append(x)

    # Create a new node for each segment except the leftmost
    add_nodes = {}
    add_split = {}
    nodes_order = [i for i in range(num_nodes)]
    for p in add_breaks:
        breaks = np.unique(np.asarray(add_breaks[p]))
        if breaks.size > 0:
            add_split[p] = breaks
            add_nodes[p] = [p]  # segment left of first break retains original node ID
            for _ in range(breaks.size):
                add_nodes[p].append(num_nodes)
                nodes_order.append(p)
                num_nodes += 1

    # Split each edge along the union of parent/child segments
    new_parent = list(ts.edges_parent)
    new_child = list(ts.edges_child)
    new_left = list(ts.edges_left)
    new_right = list(ts.edges_right)
    for e in range(num_edges):
        p, c = ts.edges_parent[e], ts.edges_child[e]

        if not (p in add_nodes or c in add_nodes):  # no breaks in parent/child
            continue

        # find parent/child breaks on edge
        left, right = ts.edges_left[e], ts.edges_right[e]
        p_nodes = add_nodes.get(p, [p])
        c_nodes = add_nodes.get(c, [c])
        p_split = add_split.get(p, np.empty(0))
        c_split = add_split.get(c, np.empty(0))
        e_split = np.unique(np.append(p_split, c_split))
        e_split = e_split[np.logical_and(e_split > left, e_split < right)]

        e_split = np.append(e_split, right)
        p_index = np.searchsorted(p_split, e_split, side="left")
        c_index = np.searchsorted(c_split, e_split, side="left")
        for x, i, j in zip(e_split, p_index, c_index):
            new_p, new_c = p_nodes[i], c_nodes[j]
            if (
                left == new_left[e]
            ):  # segment left of first break retains original edge ID
                new_parent[e] = new_p
                new_child[e] = new_c
                new_left[e] = left
                new_right[e] = x
            else:
                new_parent.append(new_p)
                new_child.append(new_c)
                new_left.append(left)
                new_right.append(x)
            left = x
        assert left == right

    nodes_order = np.array(nodes_order, dtype=np.int32)
    new_parent = np.array(new_parent, dtype=np.int32)
    new_child = np.array(new_child, dtype=np.int32)
    new_left = np.array(new_left, dtype=np.float64)
    new_right = np.array(new_right, dtype=np.float64)

    return new_parent, new_child, new_left, new_right, nodes_order


def split_root_nodes(ts):
    """
    Split roots whenever the set of children changes. Nodes are only split in the
    interior of intervals where they are roots.
    """

    edges_parent, edges_child, edges_left, edges_right, nodes_order = _split_root_nodes(
        ts
    )

    # TODO: correctly handle mutations above root (m.edge == tskit.NULL)
    mutations_node = ts.mutations_node.copy()
    for m in ts.mutations():
        if m.edge != tskit.NULL:
            mutations_node[m.id] = edges_child[m.edge]

    tables = ts.dump_tables()
    tables.nodes.set_columns(
        flags=tables.nodes.flags[nodes_order],
        time=tables.nodes.time[nodes_order],
        individual=tables.nodes.individual[nodes_order],
        population=tables.nodes.population[nodes_order],
    )
    # TODO: copy existing metadata for original nodes
    # TODO: add new metadata indicating origin for split nodes
    # TODO: add flag for split nodes
    tables.edges.set_columns(
        parent=edges_parent,
        child=edges_child,
        left=edges_left,
        right=edges_right,
    )
    tables.mutations.node = mutations_node

    tables.sort()
    tables.edges.squash()
    tables.sort()

    return tables.tree_sequence()
