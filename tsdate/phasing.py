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
import numpy as np
import tskit

#def _mutations_frequency(ts):
#
#def mutations_frequency(ts):

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

    old_metadata = np.array(tskit.unpack_strings(
        ts.tables.mutations.metadata, 
        ts.tables.mutations.metadata_offset,
    ))
    old_state = np.array(tskit.unpack_strings(
        ts.tables.mutations.derived_state, 
        ts.tables.mutations.derived_state_offset,
    ))
    new_metadata, new_metadata_offset = tskit.pack_strings(old_metadata[~singletons])
    new_state, new_state_offset = tskit.pack_strings(old_state[~singletons])

    tables = ts.dump_tables()
    tables.mutations.set_columns(
        node=ts.mutations_node[~singletons],
        time=ts.mutations_time[~singletons],
        site=ts.mutations_site[~singletons],
        derived_state=new_state,
        derived_state_offset=new_state_offset,
        metadata=new_metadata,
        metadata_offset=new_metadata_offset,
    )
    tables.sort()
    tables.build_index()
    tables.compute_mutation_parents()

    return tables.tree_sequence(), np.flatnonzero(singletons)


def rephase_singletons(ts, use_node_times=True, random_seed=None):
    """
    Rephase singleton mutations in the tree sequence. How If `use_node_times`
    is True, singletons are added to permissable branches with probability
    proportional to the branch length (and with equal probability otherwise).
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
        mutations_node[i] = rng.choice(nodes_id, p=nodes_prob / nodes_prob.sum(), size=1)
        if not np.isnan(mutations_time[i]):
            mutations_time[i] = (time + tree.time(tree.parent(mutations_node[i]))) / 2

    # TODO: add metadata with phase probability
    tables = ts.dump_tables()
    tables.mutations.node = mutations_node
    tables.mutations.time = mutations_time
    tables.sort()
    return tables.tree_sequence(), singletons


def insert_unphased_singletons(ts, position, individual, reference_state, alternate_state, allow_overlapping_sites=False):
    """
    Insert unphased singletons into the tree sequence. The phase is arbitrarily chosen 
    so that the mutation subtends the node with the lowest id, at a given position for a
    a given individual.

    :param tskit.TreeSequence ts: the tree sequence to add singletons to
    :param np.ndarray position: the position of the variants
    :param np.ndarray individual: the individual id in which the variant occurs
    :param np.ndarray reference_state: the reference state of the variant
    :param np.ndarray alternate_state: the alternate state of the variant
    :param bool allow_overlapping_sites: whether to permit insertion of
        singletons at existing sites (in which case the reference states must be
        consistent)

    :returns: A copy of the tree sequence with singletons inserted
    """
    # TODO: provenance / metdata
    tables = ts.dump_tables()
    individuals_node = {i.id: min(i.nodes) for i in ts.individuals()}
    sites_id = {p: i for i, p in enumerate(ts.sites_position)}
    overlap = False
    for pos, ind, ref, alt in zip(position, individual, reference_state, alternate_state):
        if ind not in individuals_nodes:
            raise LookupError(f"Individual {ind} is not in the tree sequence")
        if pos in sites_id:
            if not allow_overlapping_sites:
                raise ValueError(f"A site already exists at position {pos}")
            if ref != ts.site(sites_id[pos]).ancestral_state:
                raise ValueError(
                    f"Existing site at position {pos} has a different ancestral state"
                )
            overlap = True
        else:
            sites_id[pos] = tables.sites.add_row(position=pos, ancestral_state=ref)
        tables.mutations.add_row(
            site=sites_id[pos],
            node=individuals_node[ind],
            time=tskit.UNKNOWN_TIME,
            derived_state=alt,
        )
    tables.sort()
    if allow_overlapping_sites and overlap:
        tables.build_index()
        tables.compute_mutation_parents()
    return tables.tree_sequence()


def accumulate_unphased(edges_mutations, mutations_phase, mutations_block, block_edges):
    """
    Add a proportion of each unphased singleton mutation to one of the two
    edges to which it maps. 
    """
    unphased = mutations_block != tskit.NULL
    assert np.all(mutations_phase[~unphased] == 1.0)
    assert np.all(
        np.logical_and(
            mutations_phase[unphased] <= 1.0, 
            mutations_phase[unphased] >= 0.0,
        )
    )
    for b in mutations_block[unphased]:
        if b == tskit.NULL:
            continue
        i, j = block_edges[b]
        edges_mutations[i] += mutations_phase
        edges_mutations[j] += 1 - mutations_phase
    assert np.sum(edges_mutations) == mutations_block.size
    return edges_mutations
        


