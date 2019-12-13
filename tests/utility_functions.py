# MIT License
#
# Copyright (c) 2018-2019 Tskit Developers
# Copyright (C) 2017 University of Oxford
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

import numpy as np
import tskit
import io


def single_tree_ts_n2():
    r"""
    Simple case where we have n = 2 and one tree.
         2
        / \
       0   1
    """
    nodes = io.StringIO("""\
    id      is_sample   time
    0       1           0
    1       1           0
    2       0           1
    """)
    edges = io.StringIO("""\
    left    right   parent  child
    0       1       2       0,1
    """)
    return(tskit.load_text(nodes=nodes, edges=edges, strict=False))


def single_tree_ts_n3():
    r"""
    Simple case where we have n = 3 and one tree.
            4
           / \
          3   \
         / \   \
        0   1   2
    """
    nodes = io.StringIO("""\
    id      is_sample   time
    0       1           0
    1       1           0
    2       1           0
    3       0           1
    4       0           2
    """)
    edges = io.StringIO("""\
    left    right   parent  child
    0       1       3       0,1
    0       1       4       2,3
    """)
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
        0   1   2   3
    """
    nodes = io.StringIO("""\
    id      is_sample   time
    0       1           0
    1       1           0
    2       1           0
    3       1           0
    4       0           1
    5       0           2
    6       0           3
    """)
    edges = io.StringIO("""\
    left    right   parent  child
    0       1       4       0,1
    0       1       5       2,4
    0       1       6       3,5
    """)
    return tskit.load_text(nodes=nodes, edges=edges, strict=False)


def single_tree_ts_mutation_n3():
    r"""
    Simple case where we have n = 3 and one tree.
            4
           / \
          3   x
         / \   \
        0   1   2
    """
    nodes = io.StringIO("""\
    id      is_sample   time
    0       1           0
    1       1           0
    2       1           0
    3       0           1
    4       0           2
    """)
    edges = io.StringIO("""\
    left    right   parent  child
    0       1       3       0,1
    0       1       4       2,3
    """)
    sites = io.StringIO("""\
    position    ancestral_state
    0.5         0
    """)
    mutations = io.StringIO("""\
    site    node    derived_state
    0       2       1
    """)
    return tskit.load_text(nodes=nodes, edges=edges, sites=sites,
                           mutations=mutations, strict=False)


def single_tree_all_samples_one_mutation_n3():
    r"""
    Simple case where we have n = 3 and one tree.
            4
           / \
          3   x
         / \   \
        0   1   2
    """
    nodes = io.StringIO("""\
    id      is_sample   time
    0       1           0
    1       1           0
    2       1           0
    3       1           1
    4       1           2
    """)
    edges = io.StringIO("""\
    left    right   parent  child
    0       1       3       0,1
    0       1       4       2,3
    """)
    sites = io.StringIO("""\
    position    ancestral_state
    0.5         0
    """)
    mutations = io.StringIO("""\
    site    node    derived_state
    0       2       1
    """)
    return tskit.load_text(nodes=nodes, edges=edges, sites=sites,
                           mutations=mutations, strict=False)


def polytomy_tree_ts():
    r"""
    Simple case where we have n = 3 and a polytomy.
          3
         /|\
        0 1 2
    """
    nodes = io.StringIO("""\
    id      is_sample   time
    0       1           0
    1       1           0
    2       1           0
    3       0           1
    """)
    edges = io.StringIO("""\
    left    right   parent  child
    0       1       3       0,1,2
    """)
    return tskit.load_text(nodes=nodes, edges=edges, strict=False)


def single_tree_ts_internal_n3():
    r"""
    Simple case where we have n = 3 and one tree.
            4
           / \
          3   \
         / \   \
        0   1   2
    """
    nodes = io.StringIO("""\
    id      is_sample   time
    0       1           0
    1       1           0
    2       1           0
    3       1           1
    4       0           2
    """)
    edges = io.StringIO("""\
    left    right   parent  child
    0       1       3       0,1
    0       1       4       2,3
    """)
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
        0   1   2  .  0   1   2
    """
    nodes = io.StringIO("""\
    id      is_sample   time
    0       1           0
    1       1           0
    2       1           0
    3       0           1
    4       0           2
    5       0           3
    """)
    edges = io.StringIO("""\
    left    right   parent  child
    0       0.2     3       0,1
    0       1       4       2
    0       0.2     4       3
    0.2     1       4       1
    0.2     1       5       0,4
    """)
    return tskit.load_text(nodes=nodes, edges=edges, strict=False)


def single_tree_ts_with_unary():
    r"""
    Simple case where we have n = 3 and some unary nodes.
            6
           / 5
          4   \
          3    \
         / \    \
        0   1    2
    """
    nodes = io.StringIO("""\
    id      is_sample   time
    0       1           0
    1       1           0
    2       1           0
    3       0           1
    4       0           2
    5       0           3
    6       0           4
    """)
    edges = io.StringIO("""\
    left    right   parent  child
    0       1       3       0,1
    0       1       5       2
    0       1       4       3
    0       1       6       4,5
    """)
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
     0   1     2   .   0   1   2
    """
    nodes = io.StringIO("""\
    id      is_sample   time
    0       1           0
    1       1           0
    2       1           0
    3       0           1
    4       0           2
    5       0           3
    """)
    edges = io.StringIO("""\
    left    right   parent  child
    0       0.2     3       0,1
    0       1       4       2
    0       0.2     4       3
    0.2     1       4       1
    0.2     1       5       0,4
    """)
    sites = io.StringIO("""\
    position    ancestral_state
    0.1         0
    0.15         0
    0.8         0
    """)
    mutations = io.StringIO("""\
    site    node    derived_state
    0       3       1
    1       3       1
    2       0       1
    """)
    return tskit.load_text(nodes=nodes, edges=edges, sites=sites,
                           mutations=mutations, strict=False)


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
    start = np.random.uniform(0, ts.sequence_length-span)
    to_slice = {id: (a, b) for id, a, b in zip(ts.samples(), start, start + span)}

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
    keep_mutations = np.ones((mutations.num_rows, ), dtype=bool)
    positions = tables.sites.position[:]
    for i, m in enumerate(mutations):
        if m.node in to_slice:
            if not(to_slice[m.node][0] <= positions[m.site] < to_slice[m.node][1]):
                keep_mutations[i] = False
    new_ds, new_ds_offset = tskit.tables.keep_with_offset(
        keep_mutations, mutations.derived_state, mutations.derived_state_offset)
    new_md, new_md_offset = tskit.tables.keep_with_offset(
        keep_mutations, mutations.metadata, mutations.metadata_offset)
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
        metadata_offset=new_md_offset)
    return tables.tree_sequence().simplify(
        filter_populations=False,
        filter_individuals=False,
        filter_sites=False,
        keep_unary=True)
