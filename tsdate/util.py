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

import tsdate
from . import provenance
from .approx import _b
from .approx import _b1r
from .approx import _f
from .approx import _f1r
from .approx import _f1w
from .approx import _i
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
    delete_intervals=None,
    split_disjoint=None,
    filter_populations=False,
    filter_individuals=False,
    filter_sites=False,
    record_provenance=None,
    **kwargs,
):
    """
    Function to prepare tree sequences for dating by modifying the tree sequence
    to increase the accuracy of dating. This can involve removing data-poor regions,
    removing locally-unary segments of nodes via simplification, and splitting
    discontinuous nodes.

    :param tskit.TreeSequence tree_sequence: The input tree sequence
        to be preprocessed.
    :param float minimum_gap: The minimum gap between sites to remove from the tree
        sequence. Default: ``None`` treated as ``1000000``. Removed regions are recorded
        in the provenance of the resulting tree sequence.
    :param bool remove_telomeres: Should all material before the first site and after the
        last site be removed, regardless of the length. Default: ``None`` treated as
        ``True``
    :param array_like delete_intervals: A list (start, end) pairs describing the
        genomic intervals (gaps) to delete. This is usually left as ``None``
        (the default) in which case ``minimum_gap`` and ``remove_telomeres`` are used
        to determine the gaps to remove, and the calculated intervals are recorded in
        the provenance of the resulting tree sequence.
    :param bool split_disjoint: Run the {func}`split_disjoint_nodes` function
        on the returned tree sequence, breaking any disjoint node into nodes that can
        be dated separately (Default: ``None`` treated as ``True``).
    :param bool filter_populations: parameter passed to the
        {meth}`tskit.TreeSequence.simplify` command. Unlike calling that command
        directly, this defaults to ``False``, such that all populations in the tree
        sequence are kept.
    :param bool filter_individuals: parameter passed to the
        {meth}`tskit.TreeSequence.simplify` command. Unlike calling that command
        directly, this defaults to ``False``, such
        that all individuals in the tree sequence are kept.
    :param bool filter_sites: parameter passed to the
        {meth}`tskit.TreeSequence.simplify` command. Unlike calling that command
        directly, this defaults to ``False``, such
        that all sites in the tree sequence are kept.
    :param bool record_provenance: If ``True``, record details of this call to
        simplify in the returned tree sequence's provenance information
        (Default: ``None`` treated as ``True``).
    :param \\**kwargs: All further keyword arguments are passed to the
        {meth}`tskit.TreeSequence.simplify` command.

    :return: A tree sequence with gaps removed.
    :rtype: tskit.TreeSequence
    """

    logger.info("Beginning preprocessing")
    logger.info(f"Minimum_gap: {minimum_gap} and remove_telomeres: {remove_telomeres}")
    if split_disjoint is None:
        split_disjoint = True
    if record_provenance is None:
        record_provenance = True
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
    if record_provenance:
        provenance.record_provenance(
            tables,
            "preprocess_ts",
            minimum_gap=minimum_gap,
            remove_telomeres=remove_telomeres,
            split_disjoint=split_disjoint,
            filter_populations=filter_populations,
            filter_individuals=filter_individuals,
            filter_sites=filter_sites,
            delete_intervals=delete_intervals,
        )
    ts = tables.tree_sequence()
    if split_disjoint:
        ts = split_disjoint_nodes(ts, record_provenance=False)
    return ts


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
    mutation_edges = np.zeros(tree_sequence.num_mutations, dtype=np.int32)
    for mut in tree_sequence.mutations():
        mutation_edges[mut.id] = mut.edge
        if mut.edge != tskit.NULL:
            mutation_spans[mut.edge, 0] += 1
    for edge in tree_sequence.edges():
        mutation_spans[edge.id, 1] = edge.span
    return mutation_spans, mutation_edges


# Some functions for changing tskit metadata
# See https://github.com/tskit-dev/tskit/discussions/2954
# TODO - potentially possible to speed up using numba?
def _reorder_nodes(node_table, order, extra_md_dict):
    # extra_md_dict ({rowid: new_byte_metadata}) can be used to pass metadata to replace
    # the existing metadata in a row. This works by creating new rows for the metadata,
    # based on the algorithm in https://github.com/tskit-dev/tskit/discussions/2954
    data = [node_table.metadata]
    # add a list of new byte arrays, then concat
    md_dtype, md_off_dtype = node_table.metadata.dtype, node_table.metadata_offset.dtype
    data += [np.array(bytearray(v), dtype=md_dtype) for v in extra_md_dict.values()]
    md = np.concatenate(data)
    if len(md) == 0:  # Common edge case: no metadata
        md_off = np.zeros(len(order) + 1, dtype=md_off_dtype)
    else:
        extra_offsets = np.cumsum([len(d) for d in data], dtype=md_off_dtype)[1:]
        md_off = np.concatenate((node_table.metadata_offset, extra_offsets))
        arr = tskit.unpack_arrays(md, md_off)
        if len(extra_md_dict) > 0:
            # map the keys in extra_md_dict to the new row ids
            d = {k: i + node_table.num_rows for i, k in enumerate(extra_md_dict.keys())}
            md, md_off = tskit.pack_arrays([arr[d.get(i, i)] for i in order], md_dtype)
        else:
            md, md_off = tskit.pack_arrays([arr[i] for i in order], md_dtype)
    node_table.set_columns(
        flags=node_table.flags[order],
        time=node_table.time[order],
        population=node_table.population[order],
        individual=node_table.individual[order],
        metadata=md,
        metadata_offset=md_off,
    )


@numba.njit(_unituple(_i1w, 4)(_i1r, _i1r, _f1r, _f1r, _b1r))
def _split_disjoint_nodes(
    edges_parent, edges_child, edges_left, edges_right, node_excluded
):
    """
    Split disconnected regions of nodes into separate nodes.

    Returns updated edges_parent, edges_child, mutations_node, and indices indicating
    from which original node the new nodes are derived.
    """
    assert edges_parent.size == edges_child.size == edges_left.size == edges_right.size
    num_edges = edges_parent.size
    num_nodes = node_excluded.size

    # For each edge, check whether parent/child is separated by a gap from the
    # previous edge involving either parent/child. Label disconnected segments
    # per node by integers starting at zero.
    edges_order = np.argsort(edges_left)
    # TODO: is a sort really needed here?
    edges_segments = np.full((2, num_edges), -1, dtype=np.int32)
    nodes_segments = np.full(num_nodes, -1, dtype=np.int32)
    nodes_right = np.full(node_excluded.size, -np.inf, dtype=np.float64)
    for e in edges_order:
        nodes = edges_parent[e], edges_child[e]
        for i, n in enumerate(nodes):
            if node_excluded[n]:
                continue
            nodes_segments[n] += edges_left[e] > nodes_right[n]
            edges_segments[i, e] = nodes_segments[n]
            nodes_right[n] = max(nodes_right[n], edges_right[e])

    # Create "nodes_segments[i]" supplementary nodes by copying node "i".
    # Store the id of the first supplement for each node in "nodes_map".
    split_nodes = []  # the nodes in the original that were split
    nodes_map = np.full(num_nodes, -1, dtype=np.int32)
    for i, s in enumerate(nodes_segments):
        for j in range(s):
            if j == 0:
                nodes_map[i] = num_nodes
            split_nodes.append(i)
            num_nodes += 1
    split_nodes = np.array(split_nodes, dtype=np.int32)
    nodes_order = np.arange(num_nodes, dtype=np.int32)
    if len(split_nodes) > 0:
        nodes_order[-len(split_nodes) :] = split_nodes

    # Relabel the nodes on each edge given "nodes_map"
    for e in edges_order:
        nodes = edges_parent[e], edges_child[e]
        for i, n in enumerate(nodes):
            if edges_segments[i, e] > 0:
                edges_segments[i, e] += nodes_map[n] - 1
            else:
                edges_segments[i, e] = n
    edges_parent, edges_child = edges_segments[0, ...], edges_segments[1, ...]

    return edges_parent, edges_child, nodes_order, split_nodes


@numba.njit(_i1w(_i1r, _f1r, _i1r, _i1r, _i1r, _f1r, _f1r, _i1r, _i1r))
def _relabel_mutations_node(
    mutations_node,
    mutations_position,
    nodes_order,
    edges_parent,
    edges_child,
    edges_left,
    edges_right,
    insert_index,
    remove_index,
):
    """
    Traverse trees, maintaining a mapping between old and new node IDs in the
    current tree.  Update `mutations_node` to reflect new IDs.
    """
    assert edges_parent.size == edges_child.size == edges_left.size == edges_right.size
    assert edges_parent.size == insert_index.size == remove_index.size
    assert mutations_position.size == mutations_node.size

    num_nodes = nodes_order.size
    num_edges = edges_parent.size
    num_mutations = mutations_position.size

    insert_position = edges_left[insert_index]
    remove_position = edges_right[remove_index]
    sequence_length = remove_position[-1]

    output = np.full(num_mutations, tskit.NULL, dtype=np.int32)
    nodes_map = np.full(num_nodes, tskit.NULL, dtype=np.int32)
    a, b, m = 0, 0, 0
    left = 0.0
    while left < sequence_length:
        while b < num_edges and remove_position[b] == left:  # edges out
            b += 1

        while a < num_edges and insert_position[a] == left:  # edges in
            e = insert_index[a]
            c, p = edges_child[e], edges_parent[e]
            nodes_map[nodes_order[c]] = c
            nodes_map[nodes_order[p]] = p
            a += 1

        right = sequence_length
        if b < num_edges:
            right = min(right, remove_position[b])
        if a < num_edges:
            right = min(right, insert_position[a])
        left = right

        while m < num_mutations and mutations_position[m] < right:
            assert nodes_map[mutations_node[m]] != tskit.NULL
            output[m] = nodes_map[mutations_node[m]]
            m += 1

    return output


def split_disjoint_nodes(ts, *, record_provenance=None):
    """
    For each non-sample node, split regions separated by gaps into distinct
    nodes, returning a tree sequence with potentially duplicated nodes.

    Where there are multiple disconnected regions, the leftmost one is assigned
    the ID of the original node, and the remainder are assigned new node IDs.
    Population, flags, individual, time, and metadata are all copied into the
    new nodes. Nodes that have been split will be flagged with
    ``tsdate.NODE_SPLIT_BY_PREPROCESS``. The metadata of these nodes will also be
    updated with an `unsplit_node_id` field giving the node ID in the input tree
    sequence to which they correspond. If this metadata cannot be set, a warning
    is emitted.

    :param bool record_provenance: If ``True``, record details of this call in the
        returned tree sequence's provenance information (Default: ``None`` treated
        as ``True``).
    """
    metadata_key = "unsplit_node_id"
    if record_provenance is None:
        record_provenance = True
    node_is_sample = np.bitwise_and(ts.nodes_flags, tskit.NODE_IS_SAMPLE).astype(bool)
    edges_parent, edges_child, nodes_order, split_nodes = _split_disjoint_nodes(
        ts.edges_parent,
        ts.edges_child,
        ts.edges_left,
        ts.edges_right,
        node_is_sample,
    )

    mutations_node = _relabel_mutations_node(
        ts.mutations_node,
        ts.sites_position[ts.mutations_site],
        nodes_order,
        edges_parent,
        edges_child,
        ts.edges_left,
        ts.edges_right,
        ts.indexes_edge_insertion_order,
        ts.indexes_edge_removal_order,
    )
    tables = ts.dump_tables()

    # Update the nodes table (complex because we have made new nodes)
    flags = tables.nodes.flags
    flags[split_nodes] |= tsdate.NODE_SPLIT_BY_PREPROCESS
    tables.nodes.flags = flags
    extra_md = {}
    try:
        for u in split_nodes:
            md = ts.node(u).metadata
            md[metadata_key] = int(u)
            extra_md[u] = tables.nodes.metadata_schema.validate_and_encode_row(md)
    except (TypeError, tskit.MetadataValidationError):
        logger.warning(f"Could not set '{metadata_key}' on node metadata")
    _reorder_nodes(tables.nodes, nodes_order, extra_md)
    # Update the edges table
    tables.edges.parent = edges_parent
    tables.edges.child = edges_child
    # Update the mutations table
    tables.mutations.node = mutations_node
    tables.sort()

    assert np.array_equal(
        tables.nodes.time[tables.mutations.node], ts.nodes_time[ts.mutations_node]
    )

    if record_provenance:
        provenance.record_provenance(
            tables,
            "split_disjoint_nodes",
        )
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
    :param np.ndarray nodes_time: Unconstrained node ages to inject into the
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
    modified = np.sum(~np.isclose(nodes_time, constrained_nodes_time))
    if modified:
        logging.info(f"Modified ages of {modified} nodes to satisfy constraints")

    return constrained_nodes_time


def constrain_mutations(ts, nodes_time, mutations_edge):
    """
    If the mutation is above a root, its age set to the age of the root. If
    the mutation is between two internal nodes, the edge midpoint is used.

    :param tskit.TreeSequence ts: The input tree sequence, with arbitrary node
        times.
    :param np.ndarray nodes_time: Constrained node ages.
    :param np.ndarray mutations_edge: The edge that each mutation falls on.

    :return np.ndarray: Constrained mutation ages
    """

    parent = ts.edges_parent[mutations_edge]
    child = ts.edges_child[mutations_edge]
    parent_time = nodes_time[parent]
    child_time = nodes_time[child]
    assert np.all(parent_time > child_time), "Negative branch lengths"

    mutations_time = (child_time + parent_time) / 2
    internal = mutations_edge != tskit.NULL
    constrained_time = np.full(mutations_time.size, tskit.UNKNOWN_TIME)
    constrained_time[internal] = mutations_time[internal]
    constrained_time[~internal] = nodes_time[ts.mutations_node[~internal]]

    external = np.sum(~internal)
    if external:
        logging.info(f"Set ages of {external} nonsegregating mutations to root times.")

    return constrained_time


@numba.njit(_b(_i1r, _f1r, _f1r, _i1r, _i1r, _f, _i))
def _contains_unary_nodes(
    edges_parent,
    edges_left,
    edges_right,
    indexes_insert,
    indexes_remove,
    sequence_length,
    num_nodes,
):
    assert edges_parent.size == edges_left.size == edges_right.size
    assert indexes_insert.size == indexes_remove.size == edges_parent.size

    num_edges = edges_parent.size
    nodes_children = np.zeros(num_nodes, dtype=np.int32)
    position_insert = edges_left[indexes_insert]
    position_remove = edges_right[indexes_remove]

    left = 0.0
    a, b = 0, 0
    while a < num_edges or b < num_edges:
        check = set()

        while b < num_edges and position_remove[b] == left:  # edges out
            e = indexes_remove[b]
            p = edges_parent[e]
            nodes_children[p] -= 1
            check.add(p)
            b += 1

        while a < num_edges and position_insert[a] == left:  # edges in
            e = indexes_insert[a]
            p = edges_parent[e]
            nodes_children[p] += 1
            check.add(p)
            a += 1

        for p in check:
            if nodes_children[p] == 1:
                return True

        right = sequence_length
        if b < num_edges:
            right = min(right, position_remove[b])
        if a < num_edges:
            right = min(right, position_insert[a])
        left = right

    return False


def contains_unary_nodes(ts):
    """
    Check if any node in the tree sequence is unary over some portion of its span
    """

    return _contains_unary_nodes(
        ts.edges_parent,
        ts.edges_left,
        ts.edges_right,
        ts.indexes_edge_insertion_order,
        ts.indexes_edge_removal_order,
        ts.sequence_length,
        ts.num_nodes,
    )
