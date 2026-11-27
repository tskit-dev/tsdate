/*
 * MIT License
 *
 * Copyright (c) 2026 Tskit Developers
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

/*
 * Utilities for the tree sequence tables: splitting
 * nodes that are disconnected along the sequence, relabelling mutations accordingly,
 * constraining node ages, and a test for unary nodes.
 */

#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "mathutil.h"
#include "tsdate.h"

/* A key with the index it came from. */
typedef struct {
    double key;
    tsd_id_t index;
} tsd_sort_item_t;

/*
 * Stable bottom-up merge sort of `items` by key, using `buffer` (the same size) as
 * scratch. Written out, rather than using qsort, because the comparison dominates the
 * cost of sorting the edges and cannot be inlined through qsort.
 */
static void
sort_items(tsd_size_t num_items, tsd_sort_item_t *items, tsd_sort_item_t *buffer)
{
    tsd_sort_item_t *src = items;
    tsd_sort_item_t *dst = buffer;
    tsd_sort_item_t *swap;
    tsd_size_t width, start, mid, stop, i, j, k;

    for (width = 1; width < num_items; width *= 2) {
        for (start = 0; start < num_items; start += 2 * width) {
            mid = start + width < num_items ? start + width : num_items;
            stop = start + 2 * width < num_items ? start + 2 * width : num_items;
            i = start;
            j = mid;
            for (k = start; k < stop; k++) {
                if (i < mid && (j >= stop || !(src[j].key < src[i].key))) {
                    dst[k] = src[i];
                    i++;
                } else {
                    dst[k] = src[j];
                    j++;
                }
            }
        }
        swap = src;
        src = dst;
        dst = swap;
    }
    if (src != items) {
        memcpy(items, src, num_items * sizeof(*items));
    }
}

/*
 * The indexes that sort `keys`, ties broken by index (np.argsort, made deterministic:
 * the callers do not depend on the order of ties).
 */
int
tsd_argsort(tsd_size_t num_items, const double *keys, tsd_id_t *order)
{
    int ret = 0;
    tsd_sort_item_t *items = NULL;
    tsd_sort_item_t *buffer = NULL;
    const tsd_size_t alloc = num_items > 0 ? num_items : 1;
    tsd_size_t j;

    items = malloc(alloc * sizeof(*items));
    buffer = malloc(alloc * sizeof(*buffer));
    if (items == NULL || buffer == NULL) {
        ret = tsd_trace_error(TSD_ERR_NO_MEMORY);
        goto out;
    }
    for (j = 0; j < num_items; j++) {
        items[j].key = keys[j];
        items[j].index = (tsd_id_t) j;
    }
    sort_items(num_items, items, buffer);
    for (j = 0; j < num_items; j++) {
        order[j] = items[j].index;
    }
out:
    tsd_safe_free(items);
    tsd_safe_free(buffer);
    return ret;
}

int
tsd_check_ids(
    tsd_size_t num_ids, const tsd_id_t *ids, tsd_size_t bound, int allow_null, int err)
{
    tsd_size_t j;
    tsd_id_t id;

    for (j = 0; j < num_ids; j++) {
        id = ids[j];
        if (allow_null && id == TSD_NULL) {
            continue;
        }
        if (id < 0 || (tsd_size_t) id >= bound) {
            return tsd_trace_error(err);
        }
    }
    return 0;
}

int
tsd_check_edges(tsd_size_t num_nodes, tsd_size_t num_edges, const tsd_id_t *edges_parent,
    const tsd_id_t *edges_child)
{
    int ret;

    ret = tsd_check_ids(
        num_edges, edges_parent, num_nodes, 0, TSD_ERR_NODE_OUT_OF_BOUNDS);
    if (ret != 0) {
        return ret;
    }
    return tsd_check_ids(
        num_edges, edges_child, num_nodes, 0, TSD_ERR_NODE_OUT_OF_BOUNDS);
}

/* Whether each id in [0, num_edges) appears once in `index`. */
static int
is_permutation(tsd_size_t num_edges, const tsd_id_t *index, uint8_t *seen)
{
    tsd_size_t j;

    memset(seen, 0, num_edges * sizeof(*seen));
    for (j = 0; j < num_edges; j++) {
        if (seen[index[j]]) {
            return 0;
        }
        seen[index[j]] = 1;
    }
    return 1;
}

/* Whether `position[index[j]]` does not decrease with j. */
static int
is_sorted_by(tsd_size_t num_edges, const tsd_id_t *index, const double *position)
{
    tsd_size_t j;

    for (j = 1; j < num_edges; j++) {
        if (position[index[j]] < position[index[j - 1]]) {
            return 0;
        }
    }
    return 1;
}

int
tsd_check_edge_sweep(tsd_size_t num_edges, const double *edges_left,
    const double *edges_right, const tsd_id_t *indexes_insert,
    const tsd_id_t *indexes_remove, double sequence_length)
{
    int ret = 0;
    uint8_t *seen = NULL;
    tsd_size_t e;

    for (e = 0; e < num_edges; e++) {
        if (!(edges_left[e] >= 0 && edges_left[e] < edges_right[e]
                && edges_right[e] <= sequence_length)) {
            ret = tsd_trace_error(TSD_ERR_BAD_EDGE_INTERVAL);
            goto out;
        }
    }
    ret = tsd_check_ids(
        num_edges, indexes_insert, num_edges, 0, TSD_ERR_EDGE_OUT_OF_BOUNDS);
    if (ret != 0) {
        goto out;
    }
    ret = tsd_check_ids(
        num_edges, indexes_remove, num_edges, 0, TSD_ERR_EDGE_OUT_OF_BOUNDS);
    if (ret != 0) {
        goto out;
    }
    seen = malloc(num_edges > 0 ? num_edges : 1);
    if (seen == NULL) {
        ret = tsd_trace_error(TSD_ERR_NO_MEMORY);
        goto out;
    }
    if (!is_permutation(num_edges, indexes_insert, seen)
        || !is_permutation(num_edges, indexes_remove, seen)
        || !is_sorted_by(num_edges, indexes_insert, edges_left)
        || !is_sorted_by(num_edges, indexes_remove, edges_right)) {
        ret = tsd_trace_error(TSD_ERR_BAD_EDGE_INDEXES);
        goto out;
    }
out:
    tsd_safe_free(seen);
    return ret;
}

/*
 * Label the disconnected segments of each non-excluded node along its edges: the
 * segment of an edge increases when the edge starts after the previous edges of the
 * node have ended. `edges_segments` holds the labels of the parents (first
 * `num_edges` entries) and children.
 */
static int
label_node_segments(tsd_size_t num_nodes, tsd_size_t num_edges,
    const tsd_id_t *edges_parent, const tsd_id_t *edges_child, const double *edges_left,
    const double *edges_right, const uint8_t *node_excluded, tsd_id_t *edges_segments,
    tsd_id_t *nodes_segments)
{
    int ret = 0;
    tsd_id_t *edges_order = NULL;
    double *nodes_right = NULL;
    tsd_size_t j, k, side;
    tsd_id_t e, n;
    int new_segment;

    edges_order = malloc((num_edges > 0 ? num_edges : 1) * sizeof(*edges_order));
    nodes_right = malloc((num_nodes > 0 ? num_nodes : 1) * sizeof(*nodes_right));
    if (edges_order == NULL || nodes_right == NULL) {
        ret = tsd_trace_error(TSD_ERR_NO_MEMORY);
        goto out;
    }
    ret = tsd_argsort(num_edges, edges_left, edges_order);
    if (ret != 0) {
        goto out;
    }
    for (j = 0; j < num_nodes; j++) {
        nodes_segments[j] = -1;
        nodes_right[j] = -INFINITY;
    }
    for (j = 0; j < 2 * num_edges; j++) {
        edges_segments[j] = -1;
    }
    for (j = 0; j < num_edges; j++) {
        e = edges_order[j];
        for (side = 0; side < 2; side++) {
            n = side == 0 ? edges_parent[e] : edges_child[e];
            if (node_excluded[n]) {
                continue;
            }
            new_segment = edges_left[e] > nodes_right[n];
            nodes_segments[n] = (tsd_id_t) (nodes_segments[n] + new_segment);
            k = side * num_edges + (tsd_size_t) e;
            edges_segments[k] = nodes_segments[n];
            nodes_right[n] = tsd_max(nodes_right[n], edges_right[e]);
        }
    }
out:
    tsd_safe_free(edges_order);
    tsd_safe_free(nodes_right);
    return ret;
}

/*
 * Split disconnected regions of nodes into separate nodes. A node with s + 1
 * disconnected segments gets s copies, numbered from `num_nodes`; the segments after
 * the first are relabelled onto the copies.
 *
 * Writes the relabelled edges to `new_parent` and `new_child`. Allocates
 * `*nodes_order`, of length `num_nodes + *num_split`: the identity on the original
 * nodes, followed by the original node of each copy (the "split nodes"). The caller
 * frees it.
 */
int
tsd_split_disjoint_nodes(tsd_size_t num_nodes, tsd_size_t num_edges,
    const tsd_id_t *edges_parent, const tsd_id_t *edges_child, const double *edges_left,
    const double *edges_right, const uint8_t *node_excluded, tsd_id_t *new_parent,
    tsd_id_t *new_child, tsd_size_t *num_split, tsd_id_t **nodes_order)
{
    int ret = 0;
    tsd_id_t *edges_segments = NULL;
    tsd_id_t *nodes_segments = NULL;
    tsd_id_t *nodes_map = NULL;
    tsd_id_t *order = NULL;
    tsd_size_t j, side, k, total, next;
    tsd_id_t s, n, segment, e;

    ret = tsd_check_edges(num_nodes, num_edges, edges_parent, edges_child);
    if (ret != 0) {
        goto out;
    }
    edges_segments = malloc((2 * num_edges + 1) * sizeof(*edges_segments));
    nodes_segments = malloc((num_nodes > 0 ? num_nodes : 1) * sizeof(*nodes_segments));
    nodes_map = malloc((num_nodes > 0 ? num_nodes : 1) * sizeof(*nodes_map));
    if (edges_segments == NULL || nodes_segments == NULL || nodes_map == NULL) {
        ret = tsd_trace_error(TSD_ERR_NO_MEMORY);
        goto out;
    }
    ret = label_node_segments(num_nodes, num_edges, edges_parent, edges_child,
        edges_left, edges_right, node_excluded, edges_segments, nodes_segments);
    if (ret != 0) {
        goto out;
    }

    total = 0;
    for (j = 0; j < num_nodes; j++) {
        if (nodes_segments[j] > 0) {
            total += (tsd_size_t) nodes_segments[j];
        }
    }
    order = malloc((num_nodes + total + 1) * sizeof(*order));
    if (order == NULL) {
        ret = tsd_trace_error(TSD_ERR_NO_MEMORY);
        goto out;
    }
    /* Copies of node j are numbered consecutively, the first being nodes_map[j] */
    next = num_nodes;
    for (j = 0; j < num_nodes; j++) {
        order[j] = (tsd_id_t) j;
        nodes_map[j] = TSD_NULL;
        s = nodes_segments[j];
        if (s > 0) {
            nodes_map[j] = (tsd_id_t) next;
        }
        while (s > 0) {
            order[next] = (tsd_id_t) j;
            next++;
            s--;
        }
    }

    for (e = 0; (tsd_size_t) e < num_edges; e++) {
        for (side = 0; side < 2; side++) {
            n = side == 0 ? edges_parent[e] : edges_child[e];
            k = side * num_edges + (tsd_size_t) e;
            segment = edges_segments[k];
            if (segment > 0) {
                segment = (tsd_id_t) (nodes_map[n] + segment - 1);
            } else {
                segment = n;
            }
            if (side == 0) {
                new_parent[e] = segment;
            } else {
                new_child[e] = segment;
            }
        }
    }
    *num_split = total;
    *nodes_order = order;
    order = NULL;
out:
    tsd_safe_free(edges_segments);
    tsd_safe_free(nodes_segments);
    tsd_safe_free(nodes_map);
    tsd_safe_free(order);
    return ret;
}

/*
 * Traverse the trees, maintaining the map from the old to the new id of the nodes in
 * the current tree, and write the new node of each mutation. Mutations beyond the
 * last edge, which the sweep does not reach, are NULL.
 */
int
tsd_relabel_mutations_node(tsd_size_t num_nodes, tsd_size_t num_edges,
    tsd_size_t num_mutations, const tsd_id_t *mutations_node,
    const double *mutations_position, const tsd_id_t *nodes_order,
    const tsd_id_t *edges_parent, const tsd_id_t *edges_child, const double *edges_left,
    const double *edges_right, const tsd_id_t *insert_index,
    const tsd_id_t *remove_index, tsd_id_t *output)
{
    int ret = 0;
    tsd_id_t *nodes_map = NULL;
    tsd_size_t a, b, m, j;
    tsd_id_t e, c, p;
    double left, right, sequence_length;

    ret = tsd_check_edges(num_nodes, num_edges, edges_parent, edges_child);
    if (ret != 0) {
        goto out;
    }
    ret = tsd_check_ids(
        num_nodes, nodes_order, num_nodes, 0, TSD_ERR_NODE_OUT_OF_BOUNDS);
    if (ret != 0) {
        goto out;
    }
    ret = tsd_check_ids(
        num_mutations, mutations_node, num_nodes, 0, TSD_ERR_NODE_OUT_OF_BOUNDS);
    if (ret != 0) {
        goto out;
    }
    ret = tsd_check_edge_sweep(
        num_edges, edges_left, edges_right, insert_index, remove_index, INFINITY);
    if (ret != 0) {
        goto out;
    }
    nodes_map = malloc((num_nodes > 0 ? num_nodes : 1) * sizeof(*nodes_map));
    if (nodes_map == NULL) {
        ret = tsd_trace_error(TSD_ERR_NO_MEMORY);
        goto out;
    }
    for (j = 0; j < num_nodes; j++) {
        nodes_map[j] = TSD_NULL;
    }
    for (m = 0; m < num_mutations; m++) {
        output[m] = TSD_NULL;
    }

    sequence_length = 0.0;
    if (num_edges > 0) {
        sequence_length = edges_right[remove_index[num_edges - 1]];
    }
    a = 0;
    b = 0;
    m = 0;
    left = 0.0;
    while (left < sequence_length) {
        while (b < num_edges && edges_right[remove_index[b]] == left) { /* edges out */
            b++;
        }
        while (a < num_edges && edges_left[insert_index[a]] == left) { /* edges in */
            e = insert_index[a];
            c = edges_child[e];
            p = edges_parent[e];
            nodes_map[nodes_order[c]] = c;
            nodes_map[nodes_order[p]] = p;
            a++;
        }
        right = sequence_length;
        if (b < num_edges) {
            right = tsd_min(right, edges_right[remove_index[b]]);
        }
        if (a < num_edges) {
            right = tsd_min(right, edges_left[insert_index[a]]);
        }
        left = right;
        while (m < num_mutations && mutations_position[m] < right) {
            if (nodes_map[mutations_node[m]] == TSD_NULL) {
                ret = tsd_trace_error(TSD_ERR_MUTATION_NODE_NOT_IN_TREE);
                goto out;
            }
            output[m] = nodes_map[mutations_node[m]];
            m++;
        }
    }
out:
    tsd_safe_free(nodes_map);
    return ret;
}

/* Whether every edge is longer than epsilon, as a vector check of the ages. */
static int
edges_satisfy_constraint(tsd_size_t num_edges, const double *nodes_time,
    const tsd_id_t *edges_parent, const tsd_id_t *edges_child, double epsilon)
{
    tsd_size_t e;
    double length;

    for (e = 0; e < num_edges; e++) {
        length = nodes_time[edges_parent[e]] - nodes_time[edges_child[e]];
        if (!(length > epsilon)) {
            return 0;
        }
    }
    return 1;
}

/*
 * Approximate least squares solution to the positive branch length constraint, using
 * the method of alternating projections. Loosely based on Dykstra's algorithm: Dykstra
 * RL, "An algorithm for restricted least squares regression", JASA 1983.
 *
 * Writes the constrained ages to `out_time`, which does not alias `nodes_time`.
 */
int
tsd_constrain_ages(tsd_size_t num_nodes, tsd_size_t num_edges, const double *nodes_time,
    const uint8_t *nodes_fixed, const tsd_id_t *edges_parent,
    const tsd_id_t *edges_child, double epsilon, int64_t max_iterations,
    double *out_time)
{
    int ret = 0;
    double *edges_cavity = NULL;
    int64_t iteration;
    tsd_size_t e;
    tsd_id_t p, c;
    double adjustment;
    double *cavity;

    ret = tsd_check_edges(num_nodes, num_edges, edges_parent, edges_child);
    if (ret != 0) {
        goto out;
    }
    edges_cavity = calloc(2 * num_edges + 1, sizeof(*edges_cavity));
    if (edges_cavity == NULL) {
        ret = tsd_trace_error(TSD_ERR_NO_MEMORY);
        goto out;
    }
    memcpy(out_time, nodes_time, num_nodes * sizeof(*out_time));

    for (iteration = 0; iteration < max_iterations; iteration++) {
        if (edges_satisfy_constraint(
                num_edges, out_time, edges_parent, edges_child, epsilon)) {
            goto out;
        }
        for (e = 0; e < num_edges; e++) {
            p = edges_parent[e];
            c = edges_child[e];
            cavity = edges_cavity + 2 * e;
            out_time[c] -= cavity[0];
            out_time[p] -= cavity[1];
            adjustment = out_time[c] - out_time[p];
            cavity[0] = 0.0;
            cavity[1] = 0.0;
            if (adjustment > 0) {
                if (nodes_fixed[c] && nodes_fixed[p]) {
                    ret = tsd_trace_error(TSD_ERR_FIXED_NODES_VIOLATE_CONSTRAINT);
                    goto out;
                }
                if (!nodes_fixed[c] && !nodes_fixed[p]) {
                    cavity[0] = -adjustment / 2;
                    cavity[1] = adjustment / 2;
                } else if (nodes_fixed[c]) {
                    cavity[1] = adjustment;
                } else {
                    cavity[0] = -adjustment;
                }
            }
            out_time[c] += cavity[0];
            out_time[p] += cavity[1];
        }
    }

    /* Force the remaining constraint, which can change the ages of fixed nodes */
    for (e = 0; e < num_edges; e++) {
        p = edges_parent[e];
        c = edges_child[e];
        if (out_time[c] + epsilon >= out_time[p]) {
            out_time[p] = out_time[c] + epsilon;
        }
    }
out:
    tsd_safe_free(edges_cavity);
    return ret;
}

/* Whether any unmasked node among the parents of the given edges has one child. */
static int
parents_are_unary(tsd_size_t num_edges, const tsd_id_t *edges_parent,
    const tsd_id_t *indexes, tsd_size_t start, tsd_size_t stop,
    const uint8_t *nodes_mask, const tsd_id_t *nodes_children)
{
    tsd_size_t j;
    tsd_id_t p;

    tsd_bug_assert(stop <= num_edges);
    for (j = start; j < stop; j++) {
        p = edges_parent[indexes[j]];
        if (!nodes_mask[p] && nodes_children[p] == 1) {
            return 1;
        }
    }
    return 0;
}

/*
 * Whether any unmasked node has exactly one child edge in any tree. Sets
 * `*contains_unary` to 1 or 0.
 */
int
tsd_contains_unary_nodes(tsd_size_t num_nodes, tsd_size_t num_edges,
    const uint8_t *nodes_mask, const tsd_id_t *edges_parent, const double *edges_left,
    const double *edges_right, const tsd_id_t *indexes_insert,
    const tsd_id_t *indexes_remove, double sequence_length, int *contains_unary)
{
    int ret = 0;
    tsd_id_t *nodes_children = NULL;
    tsd_size_t a, b, a_start, b_start;
    tsd_id_t p;
    double left, right;

    ret = tsd_check_ids(
        num_edges, edges_parent, num_nodes, 0, TSD_ERR_NODE_OUT_OF_BOUNDS);
    if (ret != 0) {
        goto out;
    }
    ret = tsd_check_edge_sweep(num_edges, edges_left, edges_right, indexes_insert,
        indexes_remove, sequence_length);
    if (ret != 0) {
        goto out;
    }
    nodes_children = calloc(num_nodes > 0 ? num_nodes : 1, sizeof(*nodes_children));
    if (nodes_children == NULL) {
        ret = tsd_trace_error(TSD_ERR_NO_MEMORY);
        goto out;
    }
    *contains_unary = 0;
    a = 0;
    b = 0;
    left = 0.0;
    while (a < num_edges || b < num_edges) {
        a_start = a;
        b_start = b;
        while (b < num_edges && edges_right[indexes_remove[b]] == left) { /* out */
            p = edges_parent[indexes_remove[b]];
            nodes_children[p]--;
            b++;
        }
        while (a < num_edges && edges_left[indexes_insert[a]] == left) { /* in */
            p = edges_parent[indexes_insert[a]];
            nodes_children[p]++;
            a++;
        }
        /* The nodes to check are the parents of the edges that changed */
        if (parents_are_unary(num_edges, edges_parent, indexes_remove, b_start, b,
                nodes_mask, nodes_children)
            || parents_are_unary(num_edges, edges_parent, indexes_insert, a_start, a,
                nodes_mask, nodes_children)) {
            *contains_unary = 1;
            goto out;
        }
        right = sequence_length;
        if (b < num_edges) {
            right = tsd_min(right, edges_right[indexes_remove[b]]);
        }
        if (a < num_edges) {
            right = tsd_min(right, edges_left[indexes_insert[a]]);
        }
        left = right;
    }
out:
    tsd_safe_free(nodes_children);
    return ret;
}
