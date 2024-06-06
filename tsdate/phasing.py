# MIT License
#
# Copyright (c) 2021-23 Tskit Developers
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
Tools for phasing singleton mutations
"""

import numba
import numpy as np
import tskit

from .approx import _b1r, _b2r, _f, _f1r, _f2w, _i1r, _i1w, _i2r, _i2w, _tuple, _void

# --- machinery used by ExpectationPropagation class --- #


@numba.njit(_void(_f2w, _f1r, _i1r, _i2r))
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


@numba.njit(
    _tuple((_f2w, _i2w, _i1w))(
        _b1r, _i1r, _i1r, _f1r, _i1r, _i1r, _f1r, _f1r, _i1r, _i1r, _f
    )
)
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


def block_singletons(ts, individuals_unphased):
    """
    TODO
    """
    for i in ts.individuals():
        if individuals_unphased[i.id]:
            if i.nodes.size != 2:
                raise ValueError("Singleton blocking assumes diploid individuals")
            if not np.all(ts.nodes_time[i.nodes] == 0.0):
                raise ValueError("Singleton blocking assumes contemporary individuals")

    # TODO: adjust spans by an accessibility mask
    return _block_singletons(
        individuals_unphased,
        ts.nodes_individual,
        ts.mutations_node,
        ts.sites_position[ts.mutations_site],
        ts.edges_parent,
        ts.edges_child,
        ts.edges_left,
        ts.edges_right,
        ts.indexes_edge_insertion_order,
        ts.indexes_edge_removal_order,
        ts.sequence_length,
    )


@numba.njit(_i2w(_b2r, _i1r, _f1r, _i1r, _i1r, _f1r, _f1r, _i1r, _i1r, _f))
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


def mutation_frequency(ts, sample_sets=None):
    """
    TODO
    """
    if sample_sets is None:
        sample_sets = [list(ts.samples())]

    nodes_sample = np.full((ts.num_nodes, len(sample_sets)), False)
    for i, s in enumerate(sample_sets):
        assert min(s) >= 0 and max(s) < ts.num_samples, "Sample out of range"  # NOQA: PT018
        nodes_sample[s, i] = True

    return _mutation_frequency(
        nodes_sample,
        ts.mutations_node,
        ts.sites_position[ts.mutations_site],
        ts.edges_parent,
        ts.edges_child,
        ts.edges_left,
        ts.edges_right,
        ts.indexes_edge_insertion_order,
        ts.indexes_edge_removal_order,
        ts.sequence_length,
    ).squeeze()


# --- helper functions --- #


def remove_singletons(ts):
    """
    Remove all singleton mutations from the tree sequence.

    Return the new ts, along with the id of the removed mutations in the
    original tree sequence.
    """

    nodes_sample = np.bitwise_and(ts.nodes_flags, tskit.NODE_IS_SAMPLE).astype(bool)
    assert np.sum(nodes_sample) == ts.num_samples
    assert np.all(~nodes_sample[ts.edges_parent]), "Sample node has a child"
    singletons = nodes_sample[ts.mutations_node]

    metadata = np.array(
        tskit.unpack_strings(
            ts.tables.mutations.metadata,
            ts.tables.mutations.metadata_offset,
        )
    )

    state = np.array(
        tskit.unpack_strings(
            ts.tables.mutations.derived_state,
            ts.tables.mutations.derived_state_offset,
        )
    )

    singleton_derived = state[singletons]
    singleton_ancestral = np.array(
        tskit.unpack_strings(
            ts.tables.sites.ancestral_state,
            ts.tables.sites.ancestral_state_offset,
        )
    )
    singleton_ancestral = singleton_ancestral[ts.mutations_site]
    singleton_ancestral = singleton_ancestral[singletons]

    metadata, metadata_offset = tskit.pack_strings(metadata[~singletons])
    state, state_offset = tskit.pack_strings(state[~singletons])

    tables = ts.dump_tables()
    tables.mutations.set_columns(
        node=ts.mutations_node[~singletons],
        time=ts.mutations_time[~singletons],
        site=ts.mutations_site[~singletons],
        derived_state=state,
        derived_state_offset=state_offset,
        metadata=metadata,
        metadata_offset=metadata_offset,
    )
    tables.sort()
    tables.build_index()
    tables.compute_mutation_parents()

    singleton_individual = ts.nodes_individual[ts.mutations_node[singletons]]
    singleton_position = ts.sites_position[ts.mutations_site[singletons]]
    removed_singletons = (
        singleton_position,
        singleton_individual,
        singleton_ancestral,
        singleton_derived,
    )

    return tables.tree_sequence(), removed_singletons


def insert_unphased_singletons(
    ts,
    position,
    individual,
    ancestral_state,
    derived_state,
):
    """
    Insert unphased singletons into the tree sequence. The phase is arbitrarily chosen
    so that the mutation subtends the node with the highest id, at a given position for a
    a given individual.

    :param tskit.TreeSequence ts: the tree sequence to add singletons to
    :param np.ndarray position: the position of the variants
    :param np.ndarray individual: the individual id in which the variant occurs
    :param np.ndarray ancestral_state: the ancestral state of the variant
    :param np.ndarray derived_state: the derived state of the variant

    :returns: A copy of the tree sequence with singletons inserted
    """
    # TODO: provenance / metdata
    tables = ts.dump_tables()
    individuals_node = {i.id: max(i.nodes) for i in ts.individuals()}
    sites_id = {p: i for i, p in enumerate(ts.sites_position)}
    for pos, ind, ref, alt in zip(position, individual, ancestral_state, derived_state):
        if ind not in individuals_node:
            raise LookupError(f"Individual {ind} is not in the tree sequence")
        if pos in sites_id:
            if ref != ts.site(sites_id[pos]).ancestral_state:
                raise ValueError(
                    f"Existing site at position {pos} has a different ancestral state"
                )
            muts = ts.site(sites_id[pos]).mutations
            set_time = len(muts) and np.isfinite(muts[0].time)
        else:
            sites_id[pos] = tables.sites.add_row(position=pos, ancestral_state=ref)
            set_time = False
        # TODO: more efficient to do in bulk?
        site = sites_id[pos]
        node = individuals_node[ind]
        time = ts.nodes_time[node] if set_time else tskit.UNKNOWN_TIME
        tables.mutations.add_row(
            site=site,
            node=node,
            time=time,
            derived_state=alt,
        )
    tables.sort()
    tables.build_index()
    tables.compute_mutation_parents()
    return tables.tree_sequence()


def rephase_singletons(ts, use_node_times=True, random_seed=None):
    """
    Rephase singleton mutations in the tree sequence. If `use_node_times`
    is True, singletons are added to permissable branches with probability
    proportional to the branch length (and with equal probability otherwise).

    This is not efficient, and is intended for benchmarking/testing.
    """
    rng = np.random.default_rng(random_seed)

    mutations_node = ts.mutations_node.copy()
    mutations_time = ts.mutations_time.copy()

    singletons = np.bitwise_and(ts.nodes_flags[mutations_node], tskit.NODE_IS_SAMPLE)
    singletons = np.flatnonzero(singletons)
    tree = ts.first()
    for i in singletons:
        position = ts.sites_position[ts.mutations_site[i]]
        individual = ts.nodes_individual[ts.mutations_node[i]]
        time = ts.nodes_time[ts.mutations_node[i]]
        assert individual != tskit.NULL
        assert time == 0.0
        tree.seek(position)
        nodes_id = ts.individual(individual).nodes
        nodes_length = np.array([tree.time(tree.parent(n)) - time for n in nodes_id])
        nodes_prob = nodes_length if use_node_times else np.ones(nodes_id.size)
        nodes_prob /= nodes_prob.sum()
        mutations_node[i] = rng.choice(nodes_id, p=nodes_prob, size=1)[0]
        if not np.isnan(mutations_time[i]):
            parent_time = tree.time(tree.parent(mutations_node[i]))
            mutations_time[i] = (time + parent_time) / 2

    tables = ts.dump_tables()
    tables.mutations.node = mutations_node
    tables.mutations.time = mutations_time
    tables.sort()
    return tables.tree_sequence()
