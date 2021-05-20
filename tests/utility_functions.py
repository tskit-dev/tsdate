# MIT License
#
# Copyright (C) 2020 University of Oxford
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""
A collection of utilities to edit and construct tree sequences for testing purposes
"""
import io

import msprime
import numpy as np
import tskit


def add_grand_mrca(ts):
    """
    Function to add a grand mrca node to a tree sequence
    """
    grand_mrca = ts.max_root_time + 1
    tables = ts.dump_tables()
    new_node_number = tables.nodes.add_row(time=grand_mrca)
    for tree in ts.trees():
        tables.edges.add_row(
            tree.interval[0], tree.interval[1], new_node_number, tree.root
        )
    tables.sort()
    return tables.tree_sequence()


def single_tree_ts_n2():
    r"""
    Simple case where we have n = 2 and one tree. [] marks a sample
         2
        / \
      [0] [1]
    """
    nodes = io.StringIO(
        """\
    id      is_sample   time
    0       1           0
    1       1           0
    2       0           1
    """
    )
    edges = io.StringIO(
        """\
    left    right   parent  child
    0       1       2       0,1
    """
    )
    return tskit.load_text(nodes=nodes, edges=edges, strict=False)


def single_tree_ts_n3():
    r"""
    Simple case where we have n = 3 and one tree.
            4
           / \
          3   \
         / \   \
       [0] [1] [2]
    """
    nodes = io.StringIO(
        """\
    id      is_sample   time
    0       1           0
    1       1           0
    2       1           0
    3       0           1
    4       0           2
    """
    )
    edges = io.StringIO(
        """\
    left    right   parent  child
    0       1       3       0,1
    0       1       4       2,3
    """
    )
    return tskit.load_text(nodes=nodes, edges=edges, strict=False)


def single_tree_ts_n4():
    r"""
    Simple case where we have n = 4 and one tree.
              6
             / \
            5   \
           / \   \
          4   \   \
         / \   \   \
       [0] [1] [2] [3]
    """
    nodes = io.StringIO(
        """\
    id      is_sample   time
    0       1           0
    1       1           0
    2       1           0
    3       1           0
    4       0           1
    5       0           2
    6       0           3
    """
    )
    edges = io.StringIO(
        """\
    left    right   parent  child
    0       1       4       0,1
    0       1       5       2,4
    0       1       6       3,5
    """
    )
    return tskit.load_text(nodes=nodes, edges=edges, strict=False)


def single_tree_ts_mutation_n3():
    r"""
    Simple case where we have n = 3 and one tree.
            4
           / \
          3   x
         / \   \
       [0] [1] [2]
    """
    nodes = io.StringIO(
        """\
    id      is_sample   time
    0       1           0
    1       1           0
    2       1           0
    3       0           1
    4       0           2
    """
    )
    edges = io.StringIO(
        """\
    left    right   parent  child
    0       1       3       0,1
    0       1       4       2,3
    """
    )
    sites = io.StringIO(
        """\
    position    ancestral_state
    0.5         0
    """
    )
    mutations = io.StringIO(
        """\
    site    node    derived_state
    0       2       1
    """
    )
    return tskit.load_text(
        nodes=nodes, edges=edges, sites=sites, mutations=mutations, strict=False
    )


def site_no_mutations():
    r"""
    Simple case where we have n = 3 and one tree.
    The single site has no derived alleles.
            4
           / \
          3   x
         / \   \
       [0] [1] [2]
    """
    nodes = io.StringIO(
        """\
    id      is_sample   time
    0       1           0
    1       1           0
    2       1           0
    3       0           1
    4       0           2
    """
    )
    edges = io.StringIO(
        """\
    left    right   parent  child
    0       1       3       0,1
    0       1       4       2,3
    """
    )
    sites = io.StringIO(
        """\
    position    ancestral_state
    0.5         0
    """
    )
    return tskit.load_text(nodes=nodes, edges=edges, sites=sites, strict=False)


def single_tree_all_samples_one_mutation_n3():
    r"""
    Simple case where we have n = 3 and one tree.
            4
           / \
          3   x
         / \   \
       [0] [1] [2]
    """
    nodes = io.StringIO(
        """\
    id      is_sample   time
    0       1           0
    1       1           0
    2       1           0
    3       1           1
    4       1           2
    """
    )
    edges = io.StringIO(
        """\
    left    right   parent  child
    0       1       3       0,1
    0       1       4       2,3
    """
    )
    sites = io.StringIO(
        """\
    position    ancestral_state
    0.5         0
    """
    )
    mutations = io.StringIO(
        """\
    site    node    derived_state
    0       2       1
    """
    )
    return tskit.load_text(
        nodes=nodes, edges=edges, sites=sites, mutations=mutations, strict=False
    )


def gils_example_tree():
    r"""
    Simple case where we have n = 3 and one tree.
    Mutations marked on each branch by *.
             4
            / \
           /   \
          /     *
         3       *
        / \       *
       *   *       *
      *     \       \
    [0]     [1]     [2]
    """
    nodes = io.StringIO(
        """\
    id      is_sample   time
    0       1           0
    1       1           0
    2       1           0
    3       0           1
    4       0           2
    """
    )
    edges = io.StringIO(
        """\
    left    right   parent  child
    0       1       3       0,1
    0       1       4       2,3
    """
    )
    sites = io.StringIO(
        """\
    position    ancestral_state
    0.1         0
    0.2         0
    0.3         0
    0.4         0
    0.5         0
    0.6         0
    0.7         0
    """
    )
    mutations = io.StringIO(
        """\
    site    node    derived_state
    0       0       1
    1       0       1
    2       1       1
    3       2       1
    4       2       1
    5       2       1
    6       2       1
    """
    )
    return tskit.load_text(
        nodes=nodes, edges=edges, sites=sites, mutations=mutations, strict=False
    )


def polytomy_tree_ts():
    r"""
    Simple case where we have n = 3 and a polytomy.
          3
         /|\
        / | \
      [0][1][2]
    """
    nodes = io.StringIO(
        """\
    id      is_sample   time
    0       1           0
    1       1           0
    2       1           0
    3       0           1
    """
    )
    edges = io.StringIO(
        """\
    left    right   parent  child
    0       1       3       0,1,2
    """
    )
    return tskit.load_text(nodes=nodes, edges=edges, strict=False)


def single_tree_ts_internal_n3():
    r"""
    Simple case where we have n = 3 and one tree.
    Node 3 is an internal sample.
            4
           / \
          3   \
         / \   \
       [0] [1] [2]
    """
    nodes = io.StringIO(
        """\
    id      is_sample   time
    0       1           0
    1       1           0
    2       1           0
    3       1           1
    4       0           2
    """
    )
    edges = io.StringIO(
        """\
    left    right   parent  child
    0       1       3       0,1
    0       1       4       2,3
    """
    )
    return tskit.load_text(nodes=nodes, edges=edges, strict=False)


def two_tree_ts():
    r"""
    Simple case where we have n = 3 and 2 trees.
                   .    5
                   .   / \
            4      .  |   4
           / \     .  |   |\
          3   \    .  |   | \
         / \   \   .  |   |  \
       [0] [1] [2] . [0] [1] [2]
    """
    nodes = io.StringIO(
        """\
    id      is_sample   time
    0       1           0
    1       1           0
    2       1           0
    3       0           1
    4       0           2
    5       0           3
    """
    )
    edges = io.StringIO(
        """\
    left    right   parent  child
    0       0.2     3       0,1
    0       1       4       2
    0       0.2     4       3
    0.2     1       4       1
    0.2     1       5       0,4
    """
    )
    return tskit.load_text(nodes=nodes, edges=edges, strict=False)


def two_tree_ts_extra_length():
    r"""
    Simple case where we have n = 3 and 2 trees, but with extra length
    for testing keep_intervals() and delete_intervals().
                   .    5
                   .   / \
            4      .  |   4
           / \     .  |   |\
          3   \    .  |   | \
         / \   \   .  |   |  \
       [0] [1] [2] . [0] [1] [2]
    """
    nodes = io.StringIO(
        """\
    id      is_sample   time
    0       1           0
    1       1           0
    2       1           0
    3       0           1
    4       0           2
    5       0           3
    """
    )
    edges = io.StringIO(
        """\
    left    right   parent  child
    0       0.2     3       0,1
    0       1.5     4       2
    0       0.2     4       3
    0.2     1.5     4       1
    0.2     1.5     5       0,4
    """
    )
    return tskit.load_text(nodes=nodes, edges=edges, strict=False)


def two_tree_ts_n3_non_contemporaneous():
    r"""
    Simple case where we have n = 3 and two trees with node 2 ancient.
                   .    5
                   .   / \
            4      .  |   4
           / \     .  |   |\
          3  [2]   .  |   |[2]
         / \       .  |   |
       [0] [1]     . [0] [1]
    """
    ts = two_tree_ts()
    tables = ts.dump_tables()
    time = tables.nodes.time
    time[2] = time[3]
    tables.nodes.time = time
    return tables.tree_sequence()


def single_tree_ts_with_unary():
    r"""
    Simple case where we have n = 3 and some unary nodes.
            7
           / \
          5   \
          |    \
          4     6
          |     |
          3     |
         / \    |
       [0] [1] [2]
    """
    nodes = io.StringIO(
        """\
    id      is_sample   time
    0       1           0
    1       1           0
    2       1           0
    3       0           1
    4       0           2
    5       0           3
    6       0           2
    7       0           4
    """
    )
    edges = io.StringIO(
        """\
    left    right   parent  child
    0       1       3       0,1
    0       1       6       2
    0       1       4       3
    0       1       5       4
    0       1       7       5,6
    """
    )
    return tskit.load_text(nodes=nodes, edges=edges, strict=False)


def two_tree_ts_with_unary_n3():
    r"""
    Simple case where we have n = 3 and node 5 is an internal, unary node in the first
    tree. In the second tree, node t is the root, but still unary.
             6        .      5
           /   \      .      |
          4     5     .      4
          |     |     .     /  \
          3     |     .    3    \
         / \    |     .   / \    \
       [0] [1] [2]    . [0]  [1] [2]
    """
    nodes = io.StringIO(
        """\
    id      is_sample   time
    0       1           0
    1       1           0
    2       1           0
    3       0           1
    4       0           2
    5       0           3
    6       0           4
    """
    )
    edges = io.StringIO(
        """\
    left    right   parent  child
    0       2       3       0,1
    0       1       5       2
    0       2       4       3
    0       1       6       4,5
    1       2       4       2
    1       2       5       4
    """
    )
    return tskit.load_text(nodes=nodes, edges=edges, strict=False)


def two_tree_mutation_ts():
    r"""
    Simple case where we have n = 3, 2 trees, three mutations.
                   .     5
                   .    / \
            4      .   |   4
           / \     .   |   |\
          x   \    .   |   | \
         x     \   .   x   |  \
        /      |   .   |   |   |
       3       |   .   |   |   |
      / \      |   .   |   |   |
    [0] [1]   [2]  .  [0] [1] [2]
    """
    nodes = io.StringIO(
        """\
    id      is_sample   time
    0       1           0
    1       1           0
    2       1           0
    3       0           1
    4       0           2
    5       0           3
    """
    )
    edges = io.StringIO(
        """\
    left    right   parent  child
    0       0.2     3       0,1
    0       1       4       2
    0       0.2     4       3
    0.2     1       4       1
    0.2     1       5       0,4
    """
    )
    sites = io.StringIO(
        """\
    position    ancestral_state
    0.1         0
    0.15         0
    0.8         0
    """
    )
    mutations = io.StringIO(
        """\
    site    node    derived_state
    0       3       1
    1       3       1
    2       0       1
    """
    )
    return tskit.load_text(
        nodes=nodes, edges=edges, sites=sites, mutations=mutations, strict=False
    )


def two_tree_two_mrcas():
    r"""
    Simple case where we have n = 4, 2 trees, one mutation.
             6             |
            / \            |            7
           /   \           |           / \
          /     \          |          /   x
         /       \         |         /     \
        /         \        |        /       \
       4           5       |       4         5
      / \         / \      |      / \       / \
     /   \       /   \     |     /   \     /   \
   [0]   [1]   [2]   [3]   |   [0]   [1] [2]   [3]
    """
    nodes = io.StringIO(
        """\
    id      is_sample   time
    0       1           0
    1       1           0
    2       1           0
    3       1           0
    4       0           1
    5       0           1
    6       0           3
    7       0           2
    """
    )
    edges = io.StringIO(
        """\
    left    right   parent  child
    0       1       4       0,1
    0       1       5       2,3
    0       0.3     6       4
    0       0.3     6       5
    0.3     1       7       4
    0.3     1       7       5
    """
    )
    sites = io.StringIO(
        """\
    position    ancestral_state
    0.5         0
    """
    )
    mutations = io.StringIO(
        """\
    site    node    derived_state
    0       5       1
    """
    )

    return tskit.load_text(
        nodes=nodes, edges=edges, sites=sites, mutations=mutations, strict=False
    )


def loopy_tree():
    r"""
    Testing a tree with a loop.
                   .          7
                   .         / \
                   .        /   |
                   .       /    |
         6         .      /     6
        / \        .     /     / \
       /   5       .    /     /   |
      /   / \      .   /     /    |
     /   |   \     .  |     |     |
    /    |    \    .  |     |     |
   |     4     |   .  |     4     |
   |    / \    |   .  |    / \    |
  [0] [1] [2] [3]  . [0] [1] [2] [3]
    """
    nodes = io.StringIO(
        """\
    id      is_sample   time
    0       1           0
    1       1           0
    2       1           0
    3       1           0
    4       0           1
    5       0           2
    6       0           3
    7       0           4
    """
    )
    edges = io.StringIO(
        """\
    left    right   parent  child
    0       1       4       0,1
    0       0.2     5       2,4
    0       0.2     6       5
    0       1       6       3
    0.2     1       6       4
    0.2     1       7       2
    0.2     1       7       6
    """
    )
    return tskit.load_text(nodes=nodes, edges=edges, strict=False)


def single_tree_ts_n3_sample_as_parent():
    r"""
    Simple case where we have n = 3 and one tree. Node 3 is a sample.
            4
           / \
          3   \
         / \   \
       [0] [1] [2]
    """
    nodes = io.StringIO(
        """\
    id      is_sample   time
    0       1           0
    1       1           0
    2       1           0
    3       1           1
    4       0           2
    """
    )
    edges = io.StringIO(
        """\
    left    right   parent  child
    0       1       3       0,1
    0       1       4       2,3
    """
    )
    return tskit.load_text(nodes=nodes, edges=edges, strict=False)


def single_tree_ts_n2_dangling():
    r"""
    Simple case where we have n = 2 and one tree. Node 0 is dangling.
            4
           / \
          3   \
         / \   \
        0  [1] [2]
    """
    nodes = io.StringIO(
        """\
    id      is_sample   time
    0       0           0
    1       1           0
    2       1           0
    3       0           1
    4       0           2
    """
    )
    edges = io.StringIO(
        """\
    left    right   parent  child
    0       1       3       0,1
    0       1       4       2,3
    """
    )
    return tskit.load_text(nodes=nodes, edges=edges, strict=False)


def two_tree_ts_n2_part_dangling():
    r"""
    Simple case where we have n = 2 and two trees. Node 0 is dangling in the first tree.
            4                 4
           / \               / \
          3   \             3   \
         / \   \             \   \
        0   \   \             0   \
             \   \             \   \
             [1] [2]           [1] [2]
    """
    nodes = io.StringIO(
        """\
    id      is_sample   time
    0       0           0.5
    1       1           0
    2       1           0
    3       0           1
    4       0           2
    """
    )
    edges = io.StringIO(
        """\
    left    right   parent  child
    0       1       3       0
    0       0.5     3       1
    0.5     1       0       1
    0       1       4       2,3
    """
    )
    return tskit.load_text(nodes=nodes, edges=edges, strict=False)


def single_tree_ts_2mutations_multiallelic_n3():
    r"""
    Simple case where we have n = 3 and one tree.
    Site is multiallelic.
            4
           x \
          3   x
         / \   \
       [0] [1] [2]
    """
    nodes = io.StringIO(
        """\
    id      is_sample   time
    0       1           0
    1       1           0
    2       1           0
    3       0           1
    4       0           2
    """
    )
    edges = io.StringIO(
        """\
    left    right   parent  child
    0       1       3       0,1
    0       1       4       2,3
    """
    )
    sites = io.StringIO(
        """\
    position    ancestral_state
    0.5         0
    """
    )
    mutations = io.StringIO(
        """\
    site    node    derived_state
    0       2       1
    0       3       2
    """
    )
    return tskit.load_text(
        nodes=nodes, edges=edges, sites=sites, mutations=mutations, strict=False
    )


def single_tree_ts_2mutations_singletons_n3():
    r"""
    Simple case where we have n = 3 and one tree.
    Site has two singleton mutations.
            4
           / \
          3   x
         / x   \
       [0] [1] [2]
    """
    nodes = io.StringIO(
        """\
    id      is_sample   time
    0       1           0
    1       1           0
    2       1           0
    3       0           1
    4       0           2
    """
    )
    edges = io.StringIO(
        """\
    left    right   parent  child
    0       1       3       0,1
    0       1       4       2,3
    """
    )
    sites = io.StringIO(
        """\
    position    ancestral_state
    0.5         0
    """
    )
    mutations = io.StringIO(
        """\
    site    node    derived_state
    0       1       1
    0       2       1
    """
    )
    return tskit.load_text(
        nodes=nodes, edges=edges, sites=sites, mutations=mutations, strict=False
    )


def single_tree_ts_2mutations_n3():
    r"""
    Simple case where we have n = 3 and one tree.
    Site has two mutations with different times.
            4
           x \
          3   \
         / x   \
       [0] [1] [2]
    """
    nodes = io.StringIO(
        """\
    id      is_sample   time
    0       1           0
    1       1           0
    2       1           0
    3       0           10
    4       0           20
    """
    )
    edges = io.StringIO(
        """\
    left    right   parent  child
    0       1       3       0,1
    0       1       4       2,3
    """
    )
    sites = io.StringIO(
        """\
    position    ancestral_state
    0.5         0
    """
    )
    mutations = io.StringIO(
        """\
    site    node    derived_state
    0       3       1
    0       1       0
    """
    )
    return tskit.load_text(
        nodes=nodes, edges=edges, sites=sites, mutations=mutations, strict=False
    )


def ts_w_data_desert(gap_start, gap_end, length):
    """
    Inside/Outside algorithm has been observed to give overflow/underflow when
    attempting to date tree sequences with large regions without data. Test
    that preprocess_ts removes regions of a specified size that have no data.
    """
    ts = msprime.simulate(
        100, mutation_rate=10, recombination_rate=1, length=length, random_seed=10
    )
    tables = ts.dump_tables()
    sites = tables.sites.position[:]
    tables.delete_sites(np.where(np.logical_and(sites > gap_start, sites < gap_end))[0])
    deleted_ts = tables.tree_sequence()
    return deleted_ts


def truncate_ts_samples(ts, average_span, random_seed, min_span=5):
    """
    Create a tree sequence that has sample nodes which have been truncated
    so that they span only a small region of the genome. The length of the
    truncated spans is given by a poisson distribution whose mean is average_span
    but which cannot go below a fixed min_span, or above the sequence_length

    Samples are truncated by removing the edges that connect them to the rest
    of the tree.
    """

    np.random.seed(random_seed)
    # Make a list of (left,right) tuples giving the new limits of each sample
    # Keyed by sample ID.
    # for simplicity, we pick lengths from a poisson distribution of av 300 bp
    span = np.random.poisson(average_span, ts.num_samples)
    span = np.maximum(span, min_span)
    span = np.minimum(span, ts.sequence_length)
    start = np.random.uniform(0, ts.sequence_length - span)
    to_slice = {id_: (a, b) for id_, a, b in zip(ts.samples(), start, start + span)}

    tables = ts.dump_tables()
    tables.edges.clear()
    for e in ts.tables.edges:
        if e.child not in to_slice:
            left, right = e.left, e.right
        else:
            if e.right <= to_slice[e.child][0] or e.left >= to_slice[e.child][1]:
                continue  # this edge is outside the focal region
            else:
                left = max(e.left, to_slice[e.child][0])
                right = min(e.right, to_slice[e.child][1])
        tables.edges.add_row(left, right, e.parent, e.child)
    # Remove mutations above isolated nodes
    mutations = tables.mutations
    keep_mutations = np.ones((mutations.num_rows,), dtype=bool)
    positions = tables.sites.position[:]
    for i, m in enumerate(mutations):
        if m.node in to_slice:
            if not (to_slice[m.node][0] <= positions[m.site] < to_slice[m.node][1]):
                keep_mutations[i] = False
    new_ds, new_ds_offset = tskit.tables.keep_with_offset(
        keep_mutations, mutations.derived_state, mutations.derived_state_offset
    )
    new_md, new_md_offset = tskit.tables.keep_with_offset(
        keep_mutations, mutations.metadata, mutations.metadata_offset
    )
    mutations_map = np.append(np.cumsum(keep_mutations) - 1, [-1])
    mutations_map = mutations_map.astype(mutations.parent.dtype)
    # parent -1 always maps to parent -1
    tables.mutations.set_columns(
        site=mutations.site[keep_mutations],
        node=mutations.node[keep_mutations],
        derived_state=new_ds,
        derived_state_offset=new_ds_offset,
        parent=mutations_map[mutations.parent[keep_mutations]],
        metadata=new_md,
        metadata_offset=new_md_offset,
    )
    return tables.tree_sequence().simplify(
        filter_populations=False,
        filter_individuals=False,
        filter_sites=False,
        keep_unary=True,
    )
