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
 * Rescaling of time so that the mutation rate is constant. Operation order and libm
 * calls match the oracles in tests/oracle, which the tests compare to the last bit or
 * two.
 *
 * Arrays whose size is not known beforehand are allocated with room for one more
 * element than needed, so that empty inputs still give valid pointers.
 */

#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "mathutil.h"
#include "tsdate.h"

/* The null id, as tskit.NULL */
#define NO_ID ((tsd_id_t) - 1)

/* A value with its position, for sorting (the equivalent of np.argsort). */
typedef struct {
    double value;
    tsd_id_t index;
} sort_item_t;

/* Order by value, NaN last, ties by position. */
static int
compare_sort_items(const void *a, const void *b)
{
    const sort_item_t *x = (const sort_item_t *) a;
    const sort_item_t *y = (const sort_item_t *) b;
    int x_nan = isnan(x->value);
    int y_nan = isnan(y->value);
    int ret;

    if (x_nan != y_nan) {
        ret = x_nan ? 1 : -1;
    } else if (!x_nan && x->value < y->value) {
        ret = -1;
    } else if (!x_nan && x->value > y->value) {
        ret = 1;
    } else {
        ret = (x->index > y->index) - (x->index < y->index);
    }
    return ret;
}

static int
compare_ids(const void *a, const void *b)
{
    tsd_id_t x = *(const tsd_id_t *) a;
    tsd_id_t y = *(const tsd_id_t *) b;

    return (x > y) - (x < y);
}

/* Whether `a` is less than or equal to `key`, where NaN `key` is above everything. */
static int
less_equal(double a, double key)
{
    return isnan(key) || a <= key;
}

/*
 * The first index in [min_idx, max_idx] at which the sorted array `a` exceeds `key`
 * (the search of np.searchsorted(a, key, "right")): the number of entries less than or
 * equal to `key` if the range contains it. NaN is greater than every number.
 */
static tsd_size_t
bisect_right(const double *a, double key, tsd_size_t min_idx, tsd_size_t max_idx)
{
    tsd_size_t mid_idx;

    while (min_idx < max_idx) {
        mid_idx = min_idx + ((max_idx - min_idx) >> 1);
        if (less_equal(a[mid_idx], key)) {
            min_idx = mid_idx + 1;
        } else {
            max_idx = mid_idx;
        }
    }
    return min_idx;
}

/* np.searchsorted(a, key, "right") for one key */
static tsd_size_t
searchsorted_right(tsd_size_t n, const double *a, double key)
{
    return bisect_right(a, key, 0, n);
}

/* Whether np.all(np.diff(a) > 0) */
static int
is_increasing(tsd_size_t n, const double *a)
{
    tsd_size_t j;
    double diff;

    for (j = 1; j < n; j++) {
        diff = a[j] - a[j - 1];
        if (!(diff > 0)) {
            return 0;
        }
    }
    return 1;
}

/*
 * The time `x` mapped from the original to the rescaled breakpoints by the piece
 * `i`, which has the slope (the last piece is flat, as it has no upper breakpoint).
 */
static double
rescale_piece(tsd_size_t n, const double *original_breaks, const double *rescaled_breaks,
    tsd_size_t i, double x)
{
    double scaling = 0.0;
    double offset;

    if (i + 1 < n) {
        scaling = (rescaled_breaks[i + 1] - rescaled_breaks[i])
                  / (original_breaks[i + 1] - original_breaks[i]);
    }
    offset = x - original_breaks[i];
    return rescaled_breaks[i] + scaling * offset;
}

void
tsd_epoch_stats_free(tsd_epoch_stats_t *stats)
{
    tsd_safe_free(stats->counts);
    tsd_safe_free(stats->offset);
    tsd_safe_free(stats->duration);
    stats->num_epochs = 0;
}

/* ------------------------------------------------------------------------- *
 * Changepoints
 * ------------------------------------------------------------------------- */

int
tsd_fixed_changepoints(tsd_size_t num_counts, const double *counts, int64_t epochs,
    tsd_id_t **breaks, tsd_size_t *num_breaks)
{
    int ret = 0;
    double *cumulative = NULL;
    tsd_id_t *e = NULL;
    tsd_size_t j, num_e, min_idx, max_idx;
    double total, step, z, last_z;

    if (epochs <= 0) {
        ret = tsd_trace_error(TSD_ERR_NONPOSITIVE_INTERVALS);
        goto out;
    }
    num_e = (tsd_size_t) epochs + 1;
    cumulative = malloc((num_counts + 1) * sizeof(*cumulative));
    e = malloc(num_e * sizeof(*e));
    if (cumulative == NULL || e == NULL) {
        ret = tsd_trace_error(TSD_ERR_NO_MEMORY);
        goto out;
    }
    /* The cumulative counts, normalised by the total. Division by zero is not an
     * error: the result is NaN, which compares as greater than every number. */
    cumulative[0] = 0.0;
    for (j = 0; j < num_counts; j++) {
        cumulative[j + 1] = cumulative[j] + counts[j];
    }
    total = cumulative[num_counts];
    for (j = 0; j <= num_counts; j++) {
        cumulative[j] = cumulative[j] / total;
    }
    /*
     * Equally spaced quantiles, as np.linspace(0, 1, epochs + 1). The quantiles
     * increase, so each search starts from the last result.
     */
    step = 1.0 / (double) epochs;
    last_z = 0.0;
    min_idx = 0;
    max_idx = num_counts + 1;
    for (j = 0; j < num_e; j++) {
        z = 0.0 + (double) j * step;
        if (j + 1 == num_e) {
            z = 1.0;
        }
        if (j == 0 || less_equal(last_z, z)) {
            max_idx = num_counts + 1;
        } else {
            min_idx = 0;
            max_idx = max_idx < num_counts + 1 ? max_idx + 1 : num_counts + 1;
        }
        last_z = z;
        min_idx = bisect_right(cumulative, z, min_idx, max_idx);
        e[j] = (tsd_id_t) ((int64_t) min_idx - 1);
    }
    if (e[0] > 0) {
        e[0] = 0;
    }
    if (e[num_e - 1] < (int64_t) num_counts) {
        e[num_e - 1] = (tsd_id_t) num_counts;
    }
    *breaks = e;
    *num_breaks = num_e;
    e = NULL;
out:
    tsd_safe_free(cumulative);
    tsd_safe_free(e);
    return ret;
}

/* The loss of a constant rate for the observations from `i` to `j` */
static double
poisson_loss(const double *cum_offset, const double *cum_counts, tsd_size_t i,
    tsd_size_t j, double min_counts, double min_offset)
{
    double n = cum_offset[j] - cum_offset[i];
    double y = cum_counts[j] - cum_counts[i];
    double loss;

    if (n < min_offset || y < min_counts) {
        return INFINITY;
    }
    loss = -2.0 * y * (log(y) - log(n) - 1.0);
    return loss;
}

int
tsd_poisson_changepoints(tsd_size_t num_counts, const double *counts,
    const double *offset, double penalty, double min_counts, double min_offset,
    tsd_id_t **breaks, tsd_size_t *num_breaks)
{
    int ret = 0;
    double *cum_offset = NULL;
    double *cum_counts = NULL;
    double *cost = NULL;
    double *best = NULL;
    tsd_id_t *previous = NULL;
    tsd_id_t *candidates = NULL;
    tsd_id_t *path = NULL;
    tsd_size_t j, k, c, i, num_candidates, num_kept, argmin, num_path;
    tsd_id_t current;
    double min_cost;
    const tsd_size_t dim = num_counts;

    if (!(min_counts >= 0 && min_offset >= 0 && penalty >= 0)) {
        ret = tsd_trace_error(TSD_ERR_NEGATIVE_CHANGEPOINT_PARAMETER);
        goto out;
    }
    cum_offset = malloc((dim + 1) * sizeof(*cum_offset));
    cum_counts = malloc((dim + 1) * sizeof(*cum_counts));
    cost = malloc((dim + 1) * sizeof(*cost));
    best = malloc((dim + 1) * sizeof(*best));
    previous = malloc((dim + 1) * sizeof(*previous));
    candidates = malloc((dim + 1) * sizeof(*candidates));
    if (cum_offset == NULL || cum_counts == NULL || cost == NULL || best == NULL
        || previous == NULL || candidates == NULL) {
        ret = tsd_trace_error(TSD_ERR_NO_MEMORY);
        goto out;
    }
    cum_offset[0] = 0.0;
    cum_counts[0] = 0.0;
    for (j = 0; j < dim; j++) {
        cum_offset[j + 1] = cum_offset[j] + offset[j];
        cum_counts[j + 1] = cum_counts[j] + counts[j];
    }

    /*
     * Optimal partitioning with pruning: `candidates` are the start positions of the
     * last segment that can still be optimal, in increasing order; `previous[j]` is
     * the start of the last segment of the best partition of the first `j`
     * observations.
     */
    best[0] = -penalty;
    candidates[0] = 0;
    num_candidates = 1;
    for (j = 1; j <= dim; j++) {
        argmin = 0;
        min_cost = INFINITY;
        for (c = 0; c < num_candidates; c++) {
            i = (tsd_size_t) candidates[c];
            cost[i]
                = best[i]
                  + poisson_loss(cum_offset, cum_counts, i, j, min_counts, min_offset)
                  + penalty;
            if (cost[i] < min_cost) {
                min_cost = cost[i];
                argmin = i;
            }
        }
        best[j] = min_cost;
        num_kept = 0;
        for (c = 0; c < num_candidates; c++) {
            i = (tsd_size_t) candidates[c];
            if (!(cost[i] > best[j] + penalty)) {
                candidates[num_kept] = candidates[c];
                num_kept++;
            }
        }
        candidates[num_kept] = (tsd_id_t) j;
        num_candidates = num_kept + 1;
        previous[j] = (tsd_id_t) argmin;
    }

    /* The changepoints are those on the path from `dim` back to zero */
    num_path = 0;
    current = (tsd_id_t) dim;
    while (current != 0) {
        current = previous[current];
        num_path++;
    }
    path = malloc((num_path + 1) * sizeof(*path));
    if (path == NULL) {
        ret = tsd_trace_error(TSD_ERR_NO_MEMORY);
        goto out;
    }
    path[num_path] = (tsd_id_t) dim;
    current = (tsd_id_t) dim;
    for (k = num_path; k > 0; k--) {
        current = previous[current];
        path[k - 1] = current;
    }
    *breaks = path;
    *num_breaks = num_path + 1;
    path = NULL;
out:
    tsd_safe_free(cum_offset);
    tsd_safe_free(cum_counts);
    tsd_safe_free(cost);
    tsd_safe_free(best);
    tsd_safe_free(previous);
    tsd_safe_free(candidates);
    tsd_safe_free(path);
    return ret;
}

/* ------------------------------------------------------------------------- *
 * Counting mutations
 * ------------------------------------------------------------------------- */

/*
 * Add (sign = 1) or remove (sign = -1) the span of the edge `e`, to the parent `p` of
 * its child `c`, and to every ancestor of `p`, in proportion to the number of
 * samples below `c`; and update the sample counts of the ancestors.
 */
static void
count_update_ancestors(tsd_id_t e, tsd_id_t p, tsd_id_t c, double sign, double remainder,
    const tsd_id_t *nodes_edge, const tsd_id_t *nodes_parent, double *nodes_samples,
    double *edges_stats)
{
    tsd_id_t next_e, next_p;
    double span;

    while (p != NO_ID) {
        span = nodes_samples[c] * remainder;
        edges_stats[2 * e + 1] += sign * span;
        nodes_samples[p] += sign * nodes_samples[c];
        next_e = nodes_edge[p];
        next_p = nodes_parent[p];
        e = next_e;
        p = next_p;
    }
}

int
tsd_count_mutations(tsd_size_t num_nodes, const uint8_t *node_is_sample,
    tsd_size_t num_mutations, const tsd_id_t *mutations_node,
    const double *mutations_position, tsd_size_t num_edges, const tsd_id_t *edges_parent,
    const tsd_id_t *edges_child, const double *edges_left, const double *edges_right,
    const tsd_id_t *indexes_insert, const tsd_id_t *indexes_remove,
    double sequence_length, int size_biased, double *edges_stats,
    tsd_id_t *mutations_edge)
{
    int ret = 0;
    sort_item_t *mutation_order = NULL;
    double *nodes_samples = NULL;
    tsd_id_t *nodes_edge = NULL;
    tsd_id_t *nodes_parent = NULL;
    tsd_size_t j, a, b, d;
    tsd_id_t e, p, c, m;
    double left, right, remainder, edge_position;

    mutation_order = malloc((num_mutations + 1) * sizeof(*mutation_order));
    nodes_samples = malloc((num_nodes + 1) * sizeof(*nodes_samples));
    nodes_edge = malloc((num_nodes + 1) * sizeof(*nodes_edge));
    nodes_parent = malloc((num_nodes + 1) * sizeof(*nodes_parent));
    if (mutation_order == NULL || nodes_samples == NULL || nodes_edge == NULL
        || nodes_parent == NULL) {
        ret = tsd_trace_error(TSD_ERR_NO_MEMORY);
        goto out;
    }
    for (j = 0; j < num_mutations; j++) {
        mutation_order[j].value = mutations_position[j];
        mutation_order[j].index = (tsd_id_t) j;
        mutations_edge[j] = NO_ID;
    }
    qsort(mutation_order, num_mutations, sizeof(*mutation_order), compare_sort_items);
    for (j = 0; j < num_nodes; j++) {
        nodes_samples[j] = node_is_sample[j] ? 1.0 : 0.0;
        nodes_edge[j] = NO_ID;
        nodes_parent[j] = NO_ID;
    }
    memset(edges_stats, 0, 2 * num_edges * sizeof(*edges_stats));

    left = 0.0;
    a = 0;
    b = 0;
    d = 0;
    while (a < num_edges || b < num_edges) {
        remainder = sequence_length - left;

        /* edges out */
        while (b < num_edges && edges_right[indexes_remove[b]] == left) {
            e = indexes_remove[b];
            p = edges_parent[e];
            c = edges_child[e];
            nodes_edge[c] = NO_ID;
            nodes_parent[c] = NO_ID;
            if (size_biased) {
                count_update_ancestors(e, p, c, -1.0, remainder, nodes_edge,
                    nodes_parent, nodes_samples, edges_stats);
            } else {
                edges_stats[2 * e + 1] -= remainder;
            }
            b++;
        }

        /* edges in */
        while (a < num_edges && edges_left[indexes_insert[a]] == left) {
            e = indexes_insert[a];
            p = edges_parent[e];
            c = edges_child[e];
            nodes_edge[c] = e;
            nodes_parent[c] = p;
            if (size_biased) {
                count_update_ancestors(e, p, c, 1.0, remainder, nodes_edge, nodes_parent,
                    nodes_samples, edges_stats);
            } else {
                edges_stats[2 * e + 1] += remainder;
            }
            a++;
        }

        right = sequence_length;
        if (b < num_edges) {
            edge_position = edges_right[indexes_remove[b]];
            right = tsd_min(right, edge_position);
        }
        if (a < num_edges) {
            edge_position = edges_left[indexes_insert[a]];
            right = tsd_min(right, edge_position);
        }
        left = right;

        while (d < num_mutations && mutation_order[d].value < right) {
            m = mutation_order[d].index;
            c = mutations_node[m];
            e = nodes_edge[c];
            if (e != NO_ID) {
                mutations_edge[m] = e;
                edges_stats[2 * e] += size_biased ? nodes_samples[c] : 1.0;
            }
            d++;
        }
    }
out:
    tsd_safe_free(mutation_order);
    tsd_safe_free(nodes_samples);
    tsd_safe_free(nodes_edge);
    tsd_safe_free(nodes_parent);
    return ret;
}

/* ------------------------------------------------------------------------- *
 * Mutational area and the time scale
 * ------------------------------------------------------------------------- */

int
tsd_mutational_area(tsd_size_t num_nodes, const double *nodes_time, tsd_size_t num_edges,
    const double *likelihoods, const tsd_id_t *edges_parent, const tsd_id_t *edges_child,
    tsd_epoch_stats_t *stats, tsd_id_t *nodes_index)
{
    int ret = 0;
    sort_item_t *order = NULL;
    double *epoch_breaks = NULL;
    double *epoch_counts = NULL;
    double *counts = NULL;
    double *offset = NULL;
    double *duration = NULL;
    tsd_size_t j, k, num_breaks, num_epochs, a, b;
    tsd_id_t p, c, node, previous;
    double length, edge_count, edge_span;

    order = malloc((num_nodes + 1) * sizeof(*order));
    epoch_breaks = malloc((num_nodes + 1) * sizeof(*epoch_breaks));
    if (order == NULL || epoch_breaks == NULL) {
        ret = tsd_trace_error(TSD_ERR_NO_MEMORY);
        goto out;
    }

    /* Index the nodes by unique time breaks; the first epoch starts at zero */
    for (j = 0; j < num_nodes; j++) {
        order[j].value = nodes_time[j];
        order[j].index = (tsd_id_t) j;
        nodes_index[j] = 0;
    }
    qsort(order, num_nodes, sizeof(*order), compare_sort_items);
    epoch_breaks[0] = 0.0;
    num_breaks = 1;
    k = 0;
    for (j = 1; j < num_nodes; j++) {
        node = order[j].index;
        previous = order[j - 1].index;
        if (nodes_time[node] > nodes_time[previous]) {
            epoch_breaks[num_breaks] = nodes_time[node];
            num_breaks++;
            k++;
        }
        nodes_index[node] = (tsd_id_t) k;
    }
    num_epochs = num_breaks - 1;

    epoch_counts = calloc(2 * num_epochs + 1, sizeof(*epoch_counts));
    counts = malloc((num_epochs + 1) * sizeof(*counts));
    offset = malloc((num_epochs + 1) * sizeof(*offset));
    duration = malloc((num_epochs + 1) * sizeof(*duration));
    if (epoch_counts == NULL || counts == NULL || offset == NULL || duration == NULL) {
        ret = tsd_trace_error(TSD_ERR_NO_MEMORY);
        goto out;
    }

    /* Pass over edges, measuring overlap with each time interval, using the
     * instantaneous mutation rate of each edge. */
    for (j = 0; j < num_edges; j++) {
        p = edges_parent[j];
        c = edges_child[j];
        length = nodes_time[p] - nodes_time[c];
        if (!(length > 0)) {
            continue;
        }
        edge_count = likelihoods[2 * j] / length;
        edge_span = likelihoods[2 * j + 1];
        a = (tsd_size_t) nodes_index[c];
        b = (tsd_size_t) nodes_index[p];
        if (a < num_epochs) {
            epoch_counts[2 * a] += edge_count;
            epoch_counts[2 * a + 1] += edge_span;
        }
        if (b < num_epochs) {
            epoch_counts[2 * b] -= edge_count;
            epoch_counts[2 * b + 1] -= edge_span;
        }
    }
    for (k = 0; k < num_epochs; k++) {
        counts[k] = epoch_counts[2 * k];
        offset[k] = epoch_counts[2 * k + 1];
        if (k > 0) {
            counts[k] += counts[k - 1];
            offset[k] += offset[k - 1];
        }
        duration[k] = epoch_breaks[k + 1] - epoch_breaks[k];
    }
    stats->num_epochs = num_epochs;
    stats->counts = counts;
    stats->offset = offset;
    stats->duration = duration;
    counts = NULL;
    offset = NULL;
    duration = NULL;
out:
    tsd_safe_free(order);
    tsd_safe_free(epoch_breaks);
    tsd_safe_free(epoch_counts);
    tsd_safe_free(counts);
    tsd_safe_free(offset);
    tsd_safe_free(duration);
    return ret;
}

/*
 * The sum of x[i:j] in the sense of Python slices of an array of the given size (a
 * negative `i` counts from the end), accumulated from the first element.
 */
static double
slice_sum(const double *x, tsd_size_t size, tsd_id_t i, tsd_id_t j)
{
    int64_t start = i;
    int64_t stop = j;
    int64_t isize = (int64_t) size;
    int64_t k;
    double sum;

    if (start < 0) {
        start = start + isize < 0 ? 0 : start + isize;
    }
    if (stop < 0) {
        stop = stop + isize < 0 ? 0 : stop + isize;
    }
    start = start > isize ? isize : start;
    stop = stop > isize ? isize : stop;
    if (start >= stop) {
        return 0.0;
    }
    sum = x[start];
    for (k = start + 1; k < stop; k++) {
        sum += x[k];
    }
    return sum;
}

/* Sort `ids` and remove duplicates (np.unique); returns the number that remain. */
static tsd_size_t
unique_ids(tsd_size_t n, tsd_id_t *ids)
{
    tsd_size_t j;
    tsd_size_t num_unique = 0;

    qsort(ids, n, sizeof(*ids), compare_ids);
    for (j = 0; j < n; j++) {
        if (j == 0 || ids[j] != ids[j - 1]) {
            ids[num_unique] = ids[j];
            num_unique++;
        }
    }
    return num_unique;
}

int
tsd_mutational_timescale(tsd_size_t num_nodes, const double *nodes_time,
    tsd_size_t num_edges, const double *likelihoods, const tsd_id_t *edges_parent,
    const tsd_id_t *edges_child, int64_t max_intervals, tsd_size_t *num_breaks,
    double **origin_out, double **adjust_out)
{
    int ret = 0;
    tsd_epoch_stats_t stats = { 0, NULL, NULL, NULL };
    tsd_id_t *nodes_index = NULL;
    tsd_id_t *changepoints = NULL;
    double *epoch_breaks = NULL;
    double *weight = NULL;
    double *adjust = NULL;
    double *origin = NULL;
    tsd_size_t k, num_epochs, num_changepoints, num_unique;
    tsd_id_t i, j, index;
    double n, y, z;

    if (max_intervals <= 0) {
        ret = tsd_trace_error(TSD_ERR_NONPOSITIVE_INTERVALS);
        goto out;
    }
    nodes_index = malloc((num_nodes + 1) * sizeof(*nodes_index));
    if (nodes_index == NULL) {
        ret = tsd_trace_error(TSD_ERR_NO_MEMORY);
        goto out;
    }
    ret = tsd_mutational_area(num_nodes, nodes_time, num_edges, likelihoods,
        edges_parent, edges_child, &stats, nodes_index);
    if (ret != 0) {
        goto out;
    }
    num_epochs = stats.num_epochs;
    epoch_breaks = malloc((num_epochs + 1) * sizeof(*epoch_breaks));
    weight = malloc((num_epochs + 1) * sizeof(*weight));
    if (epoch_breaks == NULL || weight == NULL) {
        ret = tsd_trace_error(TSD_ERR_NO_MEMORY);
        goto out;
    }
    epoch_breaks[0] = 0.0;
    for (k = 0; k < num_epochs; k++) {
        epoch_breaks[k + 1] = epoch_breaks[k] + stats.duration[k];
        weight[k] = stats.offset[k] * stats.duration[k];
    }

    /* Rescale time such that mutation density is constant between changepoints */
    ret = tsd_fixed_changepoints(
        num_epochs, weight, max_intervals, &changepoints, &num_changepoints);
    if (ret != 0) {
        goto out;
    }
    num_unique = unique_ids(num_changepoints, changepoints);
    adjust = malloc((num_unique + 1) * sizeof(*adjust));
    origin = malloc((num_unique + 1) * sizeof(*origin));
    if (adjust == NULL || origin == NULL) {
        ret = tsd_trace_error(TSD_ERR_NO_MEMORY);
        goto out;
    }
    adjust[0] = 0.0;
    for (k = 0; k + 1 < num_unique; k++) {
        i = changepoints[k];
        j = changepoints[k + 1];
        tsd_bug_assert(j > i);
        n = slice_sum(stats.offset, num_epochs, i, j);
        y = slice_sum(stats.counts, num_epochs, i, j);
        z = slice_sum(stats.duration, num_epochs, i, j);
        if (!(n > 0)) {
            ret = tsd_trace_error(TSD_ERR_ZERO_EDGE_SPAN);
            goto out;
        }
        adjust[k + 1] = z * y / n;
    }
    for (k = 0; k < num_unique; k++) {
        if (k > 0) {
            adjust[k] += adjust[k - 1];
        }
        /* a negative changepoint indexes from the end, as in numpy */
        index = changepoints[k];
        if (index < 0) {
            index = (tsd_id_t) ((tsd_size_t) index + num_epochs + 1);
        }
        origin[k] = epoch_breaks[index];
    }
    *num_breaks = num_unique;
    *origin_out = origin;
    *adjust_out = adjust;
    origin = NULL;
    adjust = NULL;
out:
    tsd_epoch_stats_free(&stats);
    tsd_safe_free(nodes_index);
    tsd_safe_free(changepoints);
    tsd_safe_free(epoch_breaks);
    tsd_safe_free(weight);
    tsd_safe_free(adjust);
    tsd_safe_free(origin);
    return ret;
}

/* ------------------------------------------------------------------------- *
 * Rescaling point estimates and posteriors
 * ------------------------------------------------------------------------- */

int
tsd_piecewise_scale_point_estimate(tsd_size_t num_points, const double *point_estimate,
    const uint8_t *point_fixed, tsd_size_t num_breaks, const double *original_breaks,
    const double *rescaled_breaks, double *rescaled)
{
    int ret = 0;
    tsd_size_t j;
    int64_t idx;
    double x;

    tsd_bug_assert(num_breaks > 0);
    if (!is_increasing(num_breaks, rescaled_breaks)
        || !is_increasing(num_breaks, original_breaks)) {
        ret = tsd_trace_error(TSD_ERR_NON_INCREASING_BREAKPOINTS);
        goto out;
    }
    for (j = 0; j < num_points; j++) {
        x = point_estimate[j];
        idx = (int64_t) searchsorted_right(num_breaks, original_breaks, x) - 1;
        if (idx < 0) {
            /* a time before the first breakpoint wraps to the last */
            idx += (int64_t) num_breaks;
        }
        rescaled[j] = rescale_piece(
            num_breaks, original_breaks, rescaled_breaks, (tsd_size_t) idx, x);
        if (point_fixed[j]) {
            rescaled[j] = x;
        }
    }
out:
    return ret;
}

/* Rescale `x` in place, or fail if it is below the first breakpoint. */
static int
rescale_quantile(tsd_size_t num_breaks, const double *original_breaks,
    const double *rescaled_breaks, double *x)
{
    int ret = 0;
    int64_t idx;

    idx = (int64_t) searchsorted_right(num_breaks, original_breaks, *x) - 1;
    if (idx < 0) {
        ret = tsd_trace_error(TSD_ERR_TIME_BELOW_FIRST_BREAKPOINT);
        goto out;
    }
    *x = rescale_piece(
        num_breaks, original_breaks, rescaled_breaks, (tsd_size_t) idx, *x);
out:
    return ret;
}

int
tsd_piecewise_scale_posterior(tsd_size_t num_posteriors, const double *posteriors,
    const uint8_t *posteriors_fixed, tsd_size_t num_breaks,
    const double *original_breaks, const double *rescaled_breaks, double quantile_width,
    double max_shape, tsd_gammaincinv_fn gammaincinv, double *new_posteriors)
{
    int ret = 0;
    double *lower = NULL;
    double *upper = NULL;
    double *midpt = NULL;
    tsd_size_t j;
    double alpha, beta, quant_lower, quant_upper, new_beta;
    double gamma_pars[2];

    tsd_bug_assert(num_breaks > 0);
    if (!(quantile_width < 1 && quantile_width > 0)) {
        ret = tsd_trace_error(TSD_ERR_INVALID_QUANTILE_WIDTH);
        goto out;
    }
    if (!is_increasing(num_breaks, rescaled_breaks)
        || !is_increasing(num_breaks, original_breaks)) {
        ret = tsd_trace_error(TSD_ERR_NON_INCREASING_BREAKPOINTS);
        goto out;
    }
    quant_lower = quantile_width / 2;
    quant_upper = 1 - quantile_width / 2;
    for (j = 0; j < num_posteriors; j++) {
        if (!posteriors_fixed[j]
            && !(posteriors[2 * j] > -1 && posteriors[2 * j + 1] > 0)) {
            ret = tsd_trace_error(TSD_ERR_INVALID_POSTERIOR);
            goto out;
        }
    }
    if (num_posteriors == 0) {
        /* no posterior to reduce */
        ret = tsd_trace_error(TSD_ERR_NO_POSTERIORS);
        goto out;
    }
    lower = calloc(num_posteriors, sizeof(*lower));
    upper = calloc(num_posteriors, sizeof(*upper));
    midpt = calloc(num_posteriors, sizeof(*midpt));
    if (lower == NULL || upper == NULL || midpt == NULL) {
        ret = tsd_trace_error(TSD_ERR_NO_MEMORY);
        goto out;
    }

    /* use posterior mean as a point estimate */
    for (j = 0; j < num_posteriors; j++) {
        if (!posteriors_fixed[j]) {
            alpha = posteriors[2 * j];
            beta = posteriors[2 * j + 1];
            lower[j] = gammaincinv(alpha + 1, quant_lower) / beta;
            upper[j] = gammaincinv(alpha + 1, quant_upper) / beta;
            midpt[j] = (alpha + 1) / beta;
        }
    }
    /* rescale the quantiles; those of fixed posteriors are zero, and are checked too */
    for (j = 0; j < num_posteriors; j++) {
        ret = rescale_quantile(num_breaks, original_breaks, rescaled_breaks, &midpt[j]);
        if (ret != 0) {
            goto out;
        }
    }
    for (j = 0; j < num_posteriors; j++) {
        ret = rescale_quantile(num_breaks, original_breaks, rescaled_breaks, &lower[j]);
        if (ret != 0) {
            goto out;
        }
    }
    for (j = 0; j < num_posteriors; j++) {
        ret = rescale_quantile(num_breaks, original_breaks, rescaled_breaks, &upper[j]);
        if (ret != 0) {
            goto out;
        }
    }

    /* reproject posteriors using inter-quantile range */
    for (j = 0; j < num_posteriors; j++) {
        new_posteriors[2 * j] = NAN;
        new_posteriors[2 * j + 1] = NAN;
        if (posteriors_fixed[j]) {
            continue;
        }
        if (lower[j] == 0.0) {
            ret = tsd_trace_error(TSD_ERR_ZERO_DIVISION);
            goto out;
        }
        ret = tsd_approximate_gamma_iqr(quant_lower, quant_upper, lower[j], upper[j],
            max_shape, gammaincinv, gamma_pars);
        if (ret != 0) {
            goto out;
        }
        if (midpt[j] == 0.0) {
            ret = tsd_trace_error(TSD_ERR_ZERO_DIVISION);
            goto out;
        }
        /* choose rate so as to keep mean */
        new_beta = (gamma_pars[0] + 1) / midpt[j];
        new_posteriors[2 * j] = gamma_pars[0];
        new_posteriors[2 * j + 1] = new_beta;
    }
out:
    tsd_safe_free(lower);
    tsd_safe_free(upper);
    tsd_safe_free(midpt);
    return ret;
}
