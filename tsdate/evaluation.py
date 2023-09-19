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

def tree_discrepancy(ts, other):
    """
    For two tree sequences `ts` and `other`, the method `tree_discrepancy` returns a value which is the sum of differences in spans between two best matching nodes. 
    
    Using `shared_node_spans`, for each node in `ts` we find a best match from the tree `other`. If either tree sequence contains unary nodes there may be multiple matches with the same span for a single node. In this case, the return match is the node with the closest time. 
    \$ d(ts, other) = 
    sum_{x\in ts} \min_{y\in other} |t_x - t_y| \max{y \in other}
    shared_span(x,y) \$
    
    Returns two values:
    `discrepancy` (float) The total shared span divided by the total node span of ts; this is the proportion of span of ts that is represented in other.
    `root-mean-squared discrespancy` (float) with the average weighted by the span in ts.
    """
    
    shared_spans = shared_node_spans(ts, other)
    # Find all potential matches for a node based on max shared span length
    max_span = shared_spans.max(axis=1).toarray().flatten()
    col_ind = shared_spans.indices
    row_ind = np.repeat(np.arange(shared_spans.shape[0]),
                        repeats = np.diff(shared_spans.indptr))
    
    match = shared_spans.data == max_span[row_ind]
    # Construct a matrix of potiential matches and
    # scale with difference in node times
    match_matrix = scipy.sparse.coo_matrix(
    (shared_spans.data[match], (row_ind[match], col_ind[match])),
    shape = (ts.num_nodes, other.num_nodes),
    )
    
    ts_times = ts.nodes_time[row_ind[match]]
    other_times = other.nodes_time[col_ind[match]]
    time_matrix = scipy.sparse.coo_matrix(
    (np.absolute(np.asarray(ts_times-other_times)), (row_ind[match], col_ind[match])),
    shape = (ts.num_nodes, other.num_nodes),
    )
    discrepancy_matrix = match_matrix.multiply(time_matrix).tocsr()
    # Between each pair of nodes, find the minimum
    ''' WARNING.
    argmin will just output arg of the first zero in each row of the discrepancy matrix which may not be correct. 
    if time_matrix has zero as an explicit entry at (i,j) we auto declare the (i,j) pair to be the best match so that best_match[i]=j.
    '''
    best_match = discrepancy_matrix.argmin(axis=1).A1
    for i,j,data in zip(time_matrix.row, time_matrix.col, time_matrix.data):
        if data == 0:
            best_match[i] = j
    # Find the shared_spans of all of the best matches
    best_match_spans = np.asarray([shared_spans[i,j] for i,j in enumerate(best_match)]).reshape(-1)
    print(best_match_spans)
    # Return the discrepancy between ts and other
    total_node_spans = shared_node_spans(ts,ts).trace()
    discrepancy = np.sum(best_match_spans)/total_node_spans
    
    # Compute the root-mean-square discrepancy in time
    # with averaged weighted by span in ts
    ' I think this might be correct but im not 100% sure '
    time_discrepancies = np.asarray([discrepancy_matrix[i,j] for i,j in enumerate(best_match)]).reshape(-1)
    rmse = np.sqrt(np.sum(time_discrepancies)/ts.num_nodes)
    
    return discrepancy, rmse
    