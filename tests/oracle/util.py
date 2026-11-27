"""
Oracles for the tree sequence utilities (``lib/util.c``).
"""

import numpy as np
import tskit

from . import errors
from .errors import library_error


def split_disjoint_nodes(
    edges_parent, edges_child, edges_left, edges_right, nodes_excluded
):
    """
    Split disconnected regions of nodes into separate nodes.

    Returns updated edges_parent, edges_child, mutations_node, and indices indicating
    from which original node the new nodes are derived.
    """
    if not (edges_parent.size == edges_child.size == edges_left.size == edges_right.size):
        raise ValueError("Edge arrays must have the same size")
    num_edges = edges_parent.size
    num_nodes = nodes_excluded.size

    # For each edge, check whether parent/child is separated by a gap from the
    # previous edge involving either parent/child. Label disconnected segments
    # per node by integers starting at zero.
    edges_order = np.argsort(edges_left)
    # TODO: is a sort really needed here?
    edges_segments = np.full((2, num_edges), -1, dtype=np.int32)
    nodes_segments = np.full(num_nodes, -1, dtype=np.int32)
    nodes_right = np.full(nodes_excluded.size, -np.inf, dtype=np.float64)
    for e in edges_order:
        nodes = edges_parent[e], edges_child[e]
        for i, n in enumerate(nodes):
            if nodes_excluded[n]:
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


def relabel_mutations_node(
    mutations_node,
    mutations_position,
    nodes_order,
    edges_parent,
    edges_child,
    edges_left,
    edges_right,
    indexes_insert,
    indexes_remove,
):
    """
    Traverse trees, maintaining a mapping between old and new node IDs in the
    current tree.  Update `mutations_node` to reflect new IDs.
    """
    if not (
        edges_parent.size
        == edges_child.size
        == edges_left.size
        == edges_right.size
        == indexes_insert.size
        == indexes_remove.size
    ):
        raise ValueError("Edge arrays must have the same size")
    if mutations_position.size != mutations_node.size:
        raise ValueError("Mutation arrays must have the same size")

    num_nodes = nodes_order.size
    num_edges = edges_parent.size
    num_mutations = mutations_position.size

    insert_position = edges_left[indexes_insert]
    remove_position = edges_right[indexes_remove]
    sequence_length = remove_position[-1]

    output = np.full(num_mutations, tskit.NULL, dtype=np.int32)
    nodes_map = np.full(num_nodes, tskit.NULL, dtype=np.int32)
    a, b, m = 0, 0, 0
    left = 0.0
    while left < sequence_length:
        while b < num_edges and remove_position[b] == left:  # edges out
            b += 1

        while a < num_edges and insert_position[a] == left:  # edges in
            e = indexes_insert[a]
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
            if nodes_map[mutations_node[m]] == tskit.NULL:
                raise library_error(errors.MUTATION_NODE_NOT_IN_TREE)
            output[m] = nodes_map[mutations_node[m]]
            m += 1

    return output


def constrain_ages(
    nodes_time, nodes_fixed, edges_parent, edges_child, epsilon, max_iterations
):
    """
    Approximate least squares solution to the positive branch length
    constraint, using the method of alternating projections. Loosely based on
    Dykstra's algorithm, see:

    Dykstra RL, "An algorithm for restricted least squares regression", JASA
    1983
    """
    if nodes_time.size != nodes_fixed.size:
        raise ValueError("Node arrays must have the same size")
    if edges_parent.size != edges_child.size:
        raise ValueError("Edge arrays must have the same size")

    num_edges = edges_parent.size
    nodes_time = nodes_time.copy()
    edges_cavity = np.zeros((num_edges, 2))
    for _ in range(max_iterations):  # method of alternating projections
        if np.all(nodes_time[edges_parent] - nodes_time[edges_child] > epsilon):
            return nodes_time
        for e in range(num_edges):
            p, c = edges_parent[e], edges_child[e]
            nodes_time[c] -= edges_cavity[e, 0]
            nodes_time[p] -= edges_cavity[e, 1]
            adjustment = nodes_time[c] - nodes_time[p]  # + epsilon
            edges_cavity[e, :] = 0.0
            if adjustment > 0:
                if nodes_fixed[c] and nodes_fixed[p]:
                    raise library_error(errors.FIXED_NODES_VIOLATE_CONSTRAINT)
                if not nodes_fixed[c] and not nodes_fixed[p]:
                    edges_cavity[e, 0] = -adjustment / 2
                    edges_cavity[e, 1] = adjustment / 2
                elif nodes_fixed[c] and not nodes_fixed[p]:
                    edges_cavity[e, 0] = 0
                    edges_cavity[e, 1] = adjustment
                elif not nodes_fixed[c] and nodes_fixed[p]:
                    edges_cavity[e, 0] = -adjustment
                    edges_cavity[e, 1] = 0
            nodes_time[c] += edges_cavity[e, 0]
            nodes_time[p] += edges_cavity[e, 1]
    # print(
    #   "min length:", np.min(nodes_time[edges_parent] - nodes_time[edges_child])
    # )

    # force remaining constraint, which can change the ages of fixed nodes
    for e in range(num_edges):
        p, c = edges_parent[e], edges_child[e]
        # TODO: even if nodes_fixed[p], this will still change the age
        if nodes_time[c] + epsilon >= nodes_time[p]:
            nodes_time[p] = nodes_time[c] + epsilon

    return nodes_time


def contains_unary_nodes(
    nodes_mask,
    edges_parent,
    edges_left,
    edges_right,
    indexes_insert,
    indexes_remove,
    sequence_length,
):
    """
    Whether a node outside ``nodes_mask`` has exactly one child in some tree.
    """
    if not (
        edges_parent.size
        == edges_left.size
        == edges_right.size
        == indexes_insert.size
        == indexes_remove.size
    ):
        raise ValueError("Edge arrays must have the same size")

    num_edges = edges_parent.size
    nodes_children = np.zeros(nodes_mask.size, dtype=np.int32)
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
            if not nodes_mask[p] and nodes_children[p] == 1:
                return True

        right = sequence_length
        if b < num_edges:
            right = min(right, position_remove[b])
        if a < num_edges:
            right = min(right, position_insert[a])
        left = right

    return False
