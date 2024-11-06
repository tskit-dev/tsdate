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

import numpy as np
import scipy.sparse

from .evaluation import shared_node_spans


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
