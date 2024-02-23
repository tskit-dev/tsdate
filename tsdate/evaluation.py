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
Tools for comparing node times between tree sequences with different node sets
"""
import copy
from collections import defaultdict
from itertools import product

import numpy as np
import scipy.sparse
import tskit


class CladeMap:
    """
    An iterator across trees that maintains a mapping from a clade (a `frozenset` of
    sample IDs) to a `set` of nodes. When there are unary nodes, there may be multiple
    nodes associated with each clade.
    """

    def __init__(self, ts):
        self._nil = frozenset()
        self._nodes = defaultdict(set)  # nodes[clade] = {node ids}
        self._clades = defaultdict(frozenset)  # clades[node] = {sample ids}
        self.tree_sequence = ts
        self.tree = ts.first(sample_lists=True)
        for node in self.tree.nodes():
            clade = frozenset(self.tree.samples(node))
            self._nodes[clade].add(node)
            self._clades[node] = clade
        self._prev = copy.deepcopy(self._clades)
        self._diff = ts.edge_diffs()
        next(self._diff)

    def _propagate(self, edge, downdate=False):
        """
        Traverse path from `edge.parent` to root, either adding or removing the
        state (clade) associated with `edge.child` from the state of each
        visited node. Return a set with the node ids encountered during
        traversal.
        """
        nodes = set()
        node = edge.parent
        clade = self._clades[edge.child]
        while node != tskit.NULL:
            last = self._clades[node]
            self._clades[node] = last - clade if downdate else last | clade
            if len(last):
                self._nodes[last].remove(node)
                if len(self._nodes[last]) == 0:
                    del self._nodes[last]
            self._nodes[self._clades[node]].add(node)
            nodes.add(node)
            node = self.tree.parent(node)
        return nodes

    def next(self):  # noqa: A003
        """
        Advance to the next tree, returning the difference between trees as a
        dictionary of the form `node : (last_clade, next_clade)`
        """
        nodes = set()  # nodes with potentially altered clades
        diff = {}  # diff[node] = (prev_clade, curr_clade)

        if self.tree.index + 1 == self.tree_sequence.num_trees:
            return None

        # Subtract clades subtended by outgoing edges
        edge_diff = next(self._diff)
        for eo in edge_diff.edges_out:
            nodes |= self._propagate(eo, downdate=True)

        # Prune nodes that are no longer in tree
        for node in self._nodes[self._nil]:
            diff[node] = (self._prev[node], self._nil)
            del self._clades[node]
        nodes -= self._nodes[self._nil]
        self._nodes[self._nil].clear()

        # Add clades subtended by incoming edges
        self.tree.next()
        for ei in edge_diff.edges_in:
            nodes |= self._propagate(ei, downdate=False)

        # Find difference in clades between adjacent trees
        for node in nodes:
            diff[node] = (self._prev[node], self._clades[node])
            if self._prev[node] == self._clades[node]:
                del diff[node]

        # Sync previous and current states
        for node, (_, curr) in diff.items():
            if curr == self._nil:
                del self._prev[node]
            else:
                self._prev[node] = curr

        return diff

    @property
    def interval(self):
        """
        Return interval spanned by tree
        """
        return self.tree.interval

    def clades(self):
        """
        Return set of clades in tree
        """
        return self._nodes.keys() - self._nil

    def __getitem__(self, clade):
        """
        Return set of nodes associated with a given clade.
        """
        return frozenset(self._nodes[clade]) if frozenset(clade) in self else self._nil

    def __contains__(self, clade):
        """
        Check if a clade is present in the tree
        """
        return clade in self._nodes


def shared_node_spans(ts, other):
    """
    Calculate the spans over which pairs of nodes in two tree sequences are
    ancestral to indentical sets of samples.

    Returns a sparse matrix where rows correspond to nodes in `ts` and columns
    correspond to nodes in `other`.
    """

    if ts.sequence_length != other.sequence_length:
        raise ValueError("Tree sequences must be of equal sequence length.")

    if ts.num_samples != other.num_samples:
        raise ValueError("Tree sequences must have the same numbers of samples.")

    nil = frozenset()

    # Initialize clade iterators
    query = CladeMap(ts)
    target = CladeMap(other)

    # Initialize buffer[clade] = (query_nodes, target_nodes, left_coord)
    modified = query.clades() | target.clades()
    buffer = {}

    # Build sparse matrix of matches in triplet format
    query_node = []
    target_node = []
    shared_span = []
    right = 0
    while True:
        left = right
        right = min(query.interval[1], target.interval[1])

        # Flush pairs of nodes that no longer have matching clades
        for clade in modified:  # flush:
            if clade in buffer:
                n_i, n_j, start = buffer.pop(clade)
                span = left - start
                for i, j in product(n_i, n_j):
                    query_node.append(i)
                    target_node.append(j)
                    shared_span.append(span)

        # Add new pairs of nodes with matching clades
        for clade in modified:
            assert clade not in buffer
            if clade in query and clade in target:
                n_i, n_j = query[clade], target[clade]
                buffer[clade] = (n_i, n_j, left)

        if right == ts.sequence_length:
            break

        # Find difference in clades with advance to next tree
        modified.clear()
        for clade_map in (query, target):
            if clade_map.interval[1] == right:
                clade_diff = clade_map.next()
                for (prev, curr) in clade_diff.values():
                    if prev != nil:
                        modified.add(prev)
                    if curr != nil:
                        modified.add(curr)

    # Flush final tree
    for clade in buffer:
        n_i, n_j, start = buffer[clade]
        span = right - start
        for i, j in product(n_i, n_j):
            query_node.append(i)
            target_node.append(j)
            shared_span.append(span)

    numer = scipy.sparse.coo_matrix(
        (shared_span, (query_node, target_node)),
        shape=(ts.num_nodes, other.num_nodes),
    ).tocsr()

    return numer


def match_node_ages(ts, other):
    """
    For each node in `ts`, return the age of a matched node from `other`.  Node
    matching is accomplished by calculating the intervals over which pairs of
    nodes (one from `ts`, one from `other`) subtend the same set of samples.

    Returns three vectors of length `ts.num_nodes`: the age of the best
    matching node in `other` (e.g.  with the longest shared span); the
    proportion of the node span in `ts` that is covered by the best match; and
    the id of the best match in `other`.

    If either tree sequence contains unary nodes, then there may be multiple
    matches with the same span for a single node. In this case, the returned
    match is the node with the smallest integer id.
    """

    shared_spans = shared_node_spans(ts, other)
    matched_span = shared_spans.max(axis=1).todense().A1
    best_match = shared_spans.argmax(axis=1).A1
    # NB: if there are multiple nodes with the largest span in a row,
    # argmax returns the node with the smallest integer id
    matched_time = other.nodes_time[best_match]

    best_match[matched_span == 0] = tskit.NULL
    matched_time[matched_span == 0] = np.nan

    return matched_time, matched_span, best_match


def node_spans(ts):
    """
    Returns the array of "node spans", i.e., the `j`th entry gives
    the total span over which node `j` is in the tree (i.e., does
    not have 'missing data' there).
    """
    child_spans = np.bincount(
        ts.edges_child,
        weights=ts.edges_right - ts.edges_left,
        minlength=ts.num_nodes,
    )
    for t in ts.trees():
        span = t.span
        for r in t.roots:
            # do this check to exempt 'missing data'
            if t.num_children(r) > 0:
                child_spans[r] += span
    return child_spans


def total_span(ts):
    """
    Returns the total length of all "node spans", computed from
    `node_spans(ts)`.
    """
    ts_node_spans = node_spans(ts)
    ts_total_span = np.sum(ts_node_spans)
    return ts_total_span


def tree_discrepancy(ts, other):
    """
    For two tree sequences `ts` and `other`,
    this method returns three values, as a tuple:
    1. The fraction of the total span of `ts` over which each nodes' descendant
    sample set does not match its' best match's descendant sample set.
    2. The root mean squared difference
    between the times of the nodes in `ts`
    and times of their best matching nodes in `other`,
    with the average weighted by the nodes' spans in `ts`.
    3. The proportion of the span in `other` that is correctly
    represented in `ts` (i.e., the total matching span divided
    by the total span of `other`).

    This is done as follows:

    For each node in `ts` the best matching node(s) from `other`
    has the longest matching span using `shared_node_spans`.
    If there are multiple matches with the same longest shared span
    for a single node, the best match is the match that is closest in time.
    The discrepancy is:
    ..math::

    d(ts, other) = 1 -
    \\left(sum_{x\\in \\operatorname{ts}}
    \\min_{y\\in \\operatorname{other}}
    |t_x - t_y| \\max{y \\in \\operatorname{other}}
    \frac{1}{T}* \\operatorname{shared_span}(x,y)\right),

    where :math: `T` is the sum of spans of all nodes in `ts`.

    Returns three values:
    `discrepancy` (float) the value computed above
    `root-mean-squared discrepancy` (float)
    `proportion of span of `other` correctly matching in `ts` (float)

    """

    shared_spans = shared_node_spans(ts, other)
    # Find all potential matches for a node based on max shared span length
    max_span = shared_spans.max(axis=1).toarray().flatten()
    col_ind = shared_spans.indices
    row_ind = np.repeat(
        np.arange(shared_spans.shape[0]), repeats=np.diff(shared_spans.indptr)
    )
    # mask to find all potential node matches
    match = shared_spans.data == max_span[row_ind]
    # scale with difference in node times
    # determine best matches with the best_match_matrix
    ts_times = ts.nodes_time[row_ind[match]]
    other_times = other.nodes_time[col_ind[match]]
    time_difference = np.absolute(np.asarray(ts_times - other_times))
    # If a node x in `ts` has no match then we set time_difference to zero
    # This node then does not effect the rmse
    for j in range(len(shared_spans.data[match])):
        if shared_spans.data[match][j] == 0:
            time_difference[j] = 0.0
    # If two nodes have the same time, then
    # time_difference is zero, which causes problems with argmin
    # Instead we store data as 1/(1+x) and find argmax
    best_match_matrix = scipy.sparse.coo_matrix(
        (
            1 / (1 + time_difference),
            (row_ind[match], col_ind[match]),
        ),
        shape=(ts.num_nodes, other.num_nodes),
    )
    # Between each pair of nodes, find the maximum shared span
    best_match = best_match_matrix.argmax(axis=1).A1
    best_match_spans = shared_spans[np.arange(len(best_match)), best_match].reshape(-1)
    # Return the discrepancy between ts and other
    ts_node_spans = node_spans(ts)
    total_node_spans_ts = total_span(ts)
    total_node_spans_other = total_span(other)
    discrepancy = 1 - np.sum(best_match_spans) / total_node_spans_ts
    true_proportion = (1 - discrepancy) * total_node_spans_ts / total_node_spans_other
    # Compute the root-mean-square discrepancy in time
    # with averaged weighted by span in ts
    time_matrix = scipy.sparse.csr_matrix(
        (time_difference, (row_ind[match], col_ind[match])),
        shape=(ts.num_nodes, other.num_nodes),
    )
    time_discrepancies = np.asarray(
        time_matrix[np.arange(len(best_match)), best_match].reshape(-1)
    )
    product = np.multiply((time_discrepancies**2), ts_node_spans)
    rmse = np.sqrt(np.sum(product) / total_node_spans_ts)
    return discrepancy, rmse, true_proportion
