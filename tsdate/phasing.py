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

import _tsdate
import numpy as np
import tskit

# --- machinery used by ExpectationPropagation class --- #


reallocate_unphased = _tsdate.reallocate_unphased
_block_singletons = _tsdate.block_singletons


def block_singletons(ts, individuals_unphased):
    """
    Group the singleton mutations of unphased individuals into blocks: the regions
    over which both edges above an individual's two nodes are unchanged. The
    unphased individuals must be diploid and contemporary.

    :param tskit.TreeSequence ts: The tree sequence.
    :param numpy.ndarray individuals_unphased: Boolean array, one entry per
        individual, true for the individuals whose singletons are blocked.
    :return: A tuple ``(blocks_stats, blocks_edges, mutations_block)``:
        ``blocks_stats`` has one row per block, the number of singletons in the
        block and its span; ``blocks_edges`` has one row per block, the two edges of
        the block; ``mutations_block`` gives the block of each mutation, or
        ``tskit.NULL`` for mutations outside the unphased individuals.
    :raises ValueError: If an unphased individual is not diploid or not
        contemporary.
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


_mutation_frequency = _tsdate.mutation_frequency


def mutation_frequency(ts, sample_sets=None):
    """
    The number of samples of each sample set that carry each mutation, that is the
    samples below the mutation's node in the tree at its position.

    :param tskit.TreeSequence ts: The tree sequence.
    :param list sample_sets: A list of lists of sample node ids (default: one set of
        all samples).
    :return: An integer array of shape ``(num_mutations, num_sample_sets)``, with
        dimensions of size one removed (so 1-d for a single sample set).
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
    tables.build_index()
    tables.compute_mutation_parents()
    return tables.tree_sequence()
