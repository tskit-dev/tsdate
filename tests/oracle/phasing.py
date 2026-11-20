"""Oracles for the jitted functions in ``tsdate.phasing``."""

import numpy as np
import tskit


def reallocate_unphased(edges_likelihood, mutations_phase, mutations_block, blocks_edges):
    """
    Add a proportion of each unphased singleton mutation to one of the two
    edges to which it maps
    """
    assert mutations_phase.size == mutations_block.size
    assert blocks_edges.shape[1] == 2

    num_edges = edges_likelihood.shape[0]
    edges_unphased = np.full(num_edges, False)
    edges_unphased[blocks_edges[:, 0]] = True
    edges_unphased[blocks_edges[:, 1]] = True

    num_unphased = np.sum(edges_likelihood[edges_unphased, 0])
    edges_likelihood[edges_unphased, 0] = 0.0
    for m, b in enumerate(mutations_block):
        if b == tskit.NULL:
            continue
        i, j = blocks_edges[b]
        assert tskit.NULL < i < num_edges
        assert edges_unphased[i]
        assert tskit.NULL < j < num_edges
        assert edges_unphased[j]
        if np.isnan(mutations_phase[m]):  # TODO: fix rare numerical issue
            continue
        assert 0.0 <= mutations_phase[m] <= 1.0
        edges_likelihood[i, 0] += mutations_phase[m]
        edges_likelihood[j, 0] += 1 - mutations_phase[m]
    assert np.isclose(num_unphased, np.sum(edges_likelihood[edges_unphased, 0]))


def _block_singletons(
    individuals_unphased,
    nodes_individual,
    mutations_node,
    mutations_position,
    edges_parent,
    edges_child,
    edges_left,
    edges_right,
    indexes_insert,
    indexes_remove,
    sequence_length,
):
    """
    TODO
    """
    assert edges_parent.size == edges_child.size == edges_left.size == edges_right.size
    assert indexes_insert.size == indexes_remove.size == edges_parent.size
    assert mutations_node.size == mutations_position.size

    num_mutations = mutations_node.size
    num_edges = edges_parent.size
    num_individuals = individuals_unphased.size

    indexes_mutation = np.argsort(mutations_position)
    position_insert = edges_left[indexes_insert]
    position_remove = edges_right[indexes_remove]
    position_mutation = mutations_position[indexes_mutation]

    individuals_edges = np.full((num_individuals, 2), tskit.NULL)
    individuals_position = np.full(num_individuals, np.nan)
    individuals_singletons = np.zeros(num_individuals)
    individuals_block = np.full(num_edges, tskit.NULL)
    mutations_block = np.full(num_mutations, tskit.NULL)

    blocks_span = []
    blocks_singletons = []
    blocks_edges = []
    blocks_order = []

    num_blocks = 0
    left = 0.0
    a, b, d = 0, 0, 0
    while a < num_edges or b < num_edges:
        while b < num_edges and position_remove[b] == left:  # edges out
            e = indexes_remove[b]
            c = edges_child[e]
            i = nodes_individual[c]
            if i != tskit.NULL and individuals_unphased[i]:
                u, v = individuals_edges[i]
                assert u == e or v == e
                s = u if v == e else v
                individuals_edges[i] = s, tskit.NULL
                if s != tskit.NULL:  # flush block
                    blocks_order.append(individuals_block[i])
                    blocks_edges.extend([e, s])
                    blocks_singletons.append(individuals_singletons[i])
                    blocks_span.append(left - individuals_position[i])
                    individuals_position[i] = np.nan
                    individuals_block[i] = tskit.NULL
                    individuals_singletons[i] = 0.0
            b += 1

        while a < num_edges and position_insert[a] == left:  # edges in
            e = indexes_insert[a]
            c = edges_child[e]
            i = nodes_individual[c]
            if i != tskit.NULL and individuals_unphased[i]:
                u, v = individuals_edges[i]
                assert u == tskit.NULL or v == tskit.NULL
                individuals_edges[i] = [e, max(u, v)]
                individuals_position[i] = left
                if individuals_block[i] == tskit.NULL:
                    individuals_block[i] = num_blocks
                    num_blocks += 1
            a += 1

        right = sequence_length
        if b < num_edges:
            right = min(right, position_remove[b])
        if a < num_edges:
            right = min(right, position_insert[a])
        left = right

        while d < num_mutations and position_mutation[d] < right:  # mutations
            m = indexes_mutation[d]
            c = mutations_node[m]
            i = nodes_individual[c]
            if i != tskit.NULL and individuals_unphased[i]:
                mutations_block[m] = individuals_block[i]
                individuals_singletons[i] += 1.0
            d += 1

    mutations_block = mutations_block.astype(np.int32)
    blocks_edges = np.array(blocks_edges, dtype=np.int32).reshape(-1, 2)
    blocks_singletons = np.array(blocks_singletons)
    blocks_span = np.array(blocks_span)
    blocks_order = np.array(blocks_order)
    blocks_stats = np.column_stack((blocks_singletons, blocks_span))
    assert num_blocks == blocks_edges.shape[0] == blocks_stats.shape[0]

    # sort block arrays so that mutations_block points to correct row
    blocks_order = np.argsort(blocks_order)
    blocks_edges = blocks_edges[blocks_order]
    blocks_stats = blocks_stats[blocks_order]

    return blocks_stats, blocks_edges, mutations_block


def _mutation_frequency(
    nodes_sample,
    mutations_node,
    mutations_position,
    edges_parent,
    edges_child,
    edges_left,
    edges_right,
    indexes_insert,
    indexes_remove,
    sequence_length,
):
    """
    TODO
    """
    assert edges_parent.size == edges_child.size == edges_left.size == edges_right.size
    assert indexes_insert.size == indexes_remove.size == edges_parent.size
    assert mutations_node.size == mutations_position.size

    num_nodes, num_sample_sets = nodes_sample.shape
    num_mutations = mutations_node.size
    num_edges = edges_parent.size

    indexes_mutation = np.argsort(mutations_position)
    position_insert = edges_left[indexes_insert]
    position_remove = edges_right[indexes_remove]
    position_mutation = mutations_position[indexes_mutation]

    nodes_parent = np.full(num_nodes, tskit.NULL)
    nodes_samples = np.zeros((num_nodes, num_sample_sets), dtype=np.int32)
    mutations_freq = np.zeros((num_mutations, num_sample_sets), dtype=np.int32)

    # TODO: there's a better way than passing a big bool array
    for i in range(num_sample_sets):
        nodes_samples[nodes_sample[:, i], i] = 1.0

    left = 0.0
    a, b, d = 0, 0, 0
    while a < num_edges or b < num_edges:
        while b < num_edges and position_remove[b] == left:  # edges out
            e = indexes_remove[b]
            p, c = edges_parent[e], edges_child[e]
            nodes_parent[c] = tskit.NULL
            while p != tskit.NULL:
                nodes_samples[p] -= nodes_samples[c]
                p = nodes_parent[p]
            b += 1

        while a < num_edges and position_insert[a] == left:  # edges in
            e = indexes_insert[a]
            p, c = edges_parent[e], edges_child[e]
            nodes_parent[c] = p
            while p != tskit.NULL:
                nodes_samples[p] += nodes_samples[c]
                p = nodes_parent[p]
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
            mutations_freq[m] = nodes_samples[c]
            d += 1

    return mutations_freq
