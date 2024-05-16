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
