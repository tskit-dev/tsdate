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
 * Phasing of singleton mutations in unphased individuals.
 */

#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "mathutil.h"
#include "tsdate.h"

/* The state of the sweep over trees that assigns singletons to blocks. */
typedef struct {
    const uint8_t *individuals_unphased;
    const tsd_id_t *nodes_individual;
    const tsd_id_t *edges_child;
    tsd_id_t *individuals_edges; /* two edges per individual */
    double *individuals_position;
    double *individuals_singletons;
    tsd_id_t *individuals_block;
    double *blocks_stats;
    tsd_id_t *blocks_edges;
    tsd_size_t num_blocks;
    tsd_size_t num_flushed;
} block_state_t;

/* An edge ends: if its individual then has a single edge, the block is complete. */
static void
block_edge_out(block_state_t *state, tsd_id_t e, double left)
{
    const tsd_id_t i = state->nodes_individual[state->edges_child[e]];
    tsd_id_t *edges;
    tsd_id_t s, block;

    if (i == TSD_NULL || !state->individuals_unphased[i]) {
        return;
    }
    edges = state->individuals_edges + 2 * i;
    tsd_bug_assert(edges[0] == e || edges[1] == e);
    s = edges[1] == e ? edges[0] : edges[1];
    edges[0] = s;
    edges[1] = TSD_NULL;
    if (s != TSD_NULL) {
        block = state->individuals_block[i];
        tsd_bug_assert(block != TSD_NULL);
        state->blocks_edges[2 * block] = e;
        state->blocks_edges[2 * block + 1] = s;
        state->blocks_stats[2 * block] = state->individuals_singletons[i];
        state->blocks_stats[2 * block + 1] = left - state->individuals_position[i];
        state->num_flushed++;
        state->individuals_position[i] = NAN;
        state->individuals_block[i] = TSD_NULL;
        state->individuals_singletons[i] = 0.0;
    }
}

/* An edge starts: it becomes an edge of its individual, and may start a block. */
static int
block_edge_in(block_state_t *state, tsd_id_t e, double left)
{
    int ret = 0;
    const tsd_id_t i = state->nodes_individual[state->edges_child[e]];
    tsd_id_t *edges;
    tsd_id_t u, v;

    if (i == TSD_NULL || !state->individuals_unphased[i]) {
        goto out;
    }
    edges = state->individuals_edges + 2 * i;
    u = edges[0];
    v = edges[1];
    if (u != TSD_NULL && v != TSD_NULL) {
        ret = tsd_trace_error(TSD_ERR_UNPHASED_INDIVIDUAL_EDGES);
        goto out;
    }
    edges[0] = e;
    edges[1] = v > u ? v : u;
    state->individuals_position[i] = left;
    if (state->individuals_block[i] == TSD_NULL) {
        state->individuals_block[i] = (tsd_id_t) state->num_blocks;
        state->num_blocks++;
    }
out:
    return ret;
}

/*
 * Group the singleton mutations of unphased individuals into blocks: the regions over
 * which both haplotype edges of an individual are unchanged. A mutation's block is its
 * individual's block at its position, or TSD_NULL.
 *
 * Writes `mutations_block` (one id per mutation) and allocates `blocks_stats` (rows of
 * the number of singletons and the span of each block) and `blocks_edges` (rows of
 * the two edges of each block), both `*num_blocks` rows, indexed by block id. The
 * caller frees them.
 */
int
tsd_block_singletons(tsd_size_t num_individuals, tsd_size_t num_edges,
    tsd_size_t num_mutations, const uint8_t *individuals_unphased,
    const tsd_id_t *nodes_individual, const tsd_id_t *mutations_node,
    const double *mutations_position, const tsd_id_t *edges_child,
    const double *edges_left, const double *edges_right, const tsd_id_t *indexes_insert,
    const tsd_id_t *indexes_remove, double sequence_length, tsd_size_t *num_blocks,
    double **blocks_stats, tsd_id_t **blocks_edges, tsd_id_t *mutations_block)
{
    int ret = 0;
    block_state_t state;
    tsd_id_t *indexes_mutation = NULL;
    tsd_id_t *individuals_edges = NULL;
    tsd_id_t *individuals_block = NULL;
    double *individuals_position = NULL;
    double *individuals_singletons = NULL;
    double *stats = NULL;
    tsd_id_t *block_edges = NULL;
    const tsd_size_t num_rows = num_edges > 0 ? num_edges : 1;
    const tsd_size_t num_ind = num_individuals > 0 ? num_individuals : 1;
    tsd_size_t a, b, d, j;
    tsd_id_t m, i;
    double left, right;

    indexes_mutation
        = malloc((num_mutations > 0 ? num_mutations : 1) * sizeof(tsd_id_t));
    individuals_edges = malloc(2 * num_ind * sizeof(*individuals_edges));
    individuals_block = malloc(num_ind * sizeof(*individuals_block));
    individuals_position = malloc(num_ind * sizeof(*individuals_position));
    individuals_singletons = malloc(num_ind * sizeof(*individuals_singletons));
    stats = malloc(2 * num_rows * sizeof(*stats));
    block_edges = malloc(2 * num_rows * sizeof(*block_edges));
    if (indexes_mutation == NULL || individuals_edges == NULL
        || individuals_block == NULL || individuals_position == NULL
        || individuals_singletons == NULL || stats == NULL || block_edges == NULL) {
        ret = tsd_trace_error(TSD_ERR_NO_MEMORY);
        goto out;
    }
    ret = tsd_argsort(num_mutations, mutations_position, indexes_mutation);
    if (ret != 0) {
        goto out;
    }
    for (j = 0; j < num_individuals; j++) {
        individuals_edges[2 * j] = TSD_NULL;
        individuals_edges[2 * j + 1] = TSD_NULL;
        individuals_block[j] = TSD_NULL;
        individuals_position[j] = NAN;
        individuals_singletons[j] = 0.0;
    }
    for (j = 0; j < num_mutations; j++) {
        mutations_block[j] = TSD_NULL;
    }

    state.individuals_unphased = individuals_unphased;
    state.nodes_individual = nodes_individual;
    state.edges_child = edges_child;
    state.individuals_edges = individuals_edges;
    state.individuals_position = individuals_position;
    state.individuals_singletons = individuals_singletons;
    state.individuals_block = individuals_block;
    state.blocks_stats = stats;
    state.blocks_edges = block_edges;
    state.num_blocks = 0;
    state.num_flushed = 0;

    a = 0;
    b = 0;
    d = 0;
    left = 0.0;
    while (a < num_edges || b < num_edges) {
        while (b < num_edges && edges_right[indexes_remove[b]] == left) {
            block_edge_out(&state, indexes_remove[b], left);
            b++;
        }
        while (a < num_edges && edges_left[indexes_insert[a]] == left) {
            ret = block_edge_in(&state, indexes_insert[a], left);
            if (ret != 0) {
                goto out;
            }
            a++;
        }
        right = sequence_length;
        if (b < num_edges) {
            right = tsd_min(right, edges_right[indexes_remove[b]]);
        }
        if (a < num_edges) {
            right = tsd_min(right, edges_left[indexes_insert[a]]);
        }
        left = right;
        while (d < num_mutations && mutations_position[indexes_mutation[d]] < right) {
            m = indexes_mutation[d];
            i = nodes_individual[mutations_node[m]];
            if (i != TSD_NULL && individuals_unphased[i]) {
                mutations_block[m] = individuals_block[i];
                individuals_singletons[i] += 1.0;
            }
            d++;
        }
    }
    /* Every block must have been completed */
    if (state.num_blocks != state.num_flushed) {
        ret = tsd_trace_error(TSD_ERR_UNPHASED_INDIVIDUAL_EDGES);
        goto out;
    }
    *num_blocks = state.num_blocks;
    *blocks_stats = stats;
    *blocks_edges = block_edges;
    stats = NULL;
    block_edges = NULL;
out:
    tsd_safe_free(indexes_mutation);
    tsd_safe_free(individuals_edges);
    tsd_safe_free(individuals_block);
    tsd_safe_free(individuals_position);
    tsd_safe_free(individuals_singletons);
    tsd_safe_free(stats);
    tsd_safe_free(block_edges);
    return ret;
}

/* The sum of the first column of the likelihoods of the flagged edges. */
static double
sum_flagged_likelihoods(
    tsd_size_t num_edges, const double *edges_likelihood, const uint8_t *edges_unphased)
{
    double total = 0.0;
    tsd_size_t e;

    for (e = 0; e < num_edges; e++) {
        if (edges_unphased[e]) {
            total += edges_likelihood[2 * e];
        }
    }
    return total;
}

/*
 * Reallocate the likelihood of the edges of unphased blocks: the mutations in a block
 * give each edge of the block its phase probability of a mutation, and the other
 * edge the remainder. `edges_likelihood` has two columns and is updated in place.
 */
int
tsd_reallocate_unphased(tsd_size_t num_edges, tsd_size_t num_mutations,
    tsd_size_t num_blocks, const double *mutations_phase,
    const tsd_id_t *mutations_block, const tsd_id_t *blocks_edges,
    double *edges_likelihood)
{
    int ret = 0;
    uint8_t *edges_unphased = NULL;
    tsd_size_t m, e, j;
    tsd_id_t b, i, k;
    double num_unphased, phase;

    edges_unphased = calloc(num_edges > 0 ? num_edges : 1, sizeof(*edges_unphased));
    if (edges_unphased == NULL) {
        ret = tsd_trace_error(TSD_ERR_NO_MEMORY);
        goto out;
    }
    /* Edges that are not in range are caught below, if a mutation refers to them */
    for (j = 0; j < 2 * num_blocks; j++) {
        e = (tsd_size_t) blocks_edges[j];
        if (blocks_edges[j] > TSD_NULL && e < num_edges) {
            edges_unphased[e] = 1;
        }
    }
    num_unphased = sum_flagged_likelihoods(num_edges, edges_likelihood, edges_unphased);
    for (e = 0; e < num_edges; e++) {
        if (edges_unphased[e]) {
            edges_likelihood[2 * e] = 0.0;
        }
    }

    for (m = 0; m < num_mutations; m++) {
        b = mutations_block[m];
        if (b == TSD_NULL) {
            continue;
        }
        if (b < 0 || (tsd_size_t) b >= num_blocks) {
            ret = tsd_trace_error(TSD_ERR_BLOCK_OUT_OF_RANGE);
            goto out;
        }
        i = blocks_edges[2 * b];
        k = blocks_edges[2 * b + 1];
        if (i <= TSD_NULL || (tsd_size_t) i >= num_edges || k <= TSD_NULL
            || (tsd_size_t) k >= num_edges) {
            ret = tsd_trace_error(TSD_ERR_BLOCK_OUT_OF_RANGE);
            goto out;
        }
        phase = mutations_phase[m];
        if (isnan(phase)) { /* TODO: fix rare numerical issue */
            continue;
        }
        if (!(phase >= 0.0 && phase <= 1.0)) {
            ret = tsd_trace_error(TSD_ERR_PHASE_OUT_OF_RANGE);
            goto out;
        }
        edges_likelihood[2 * i] += phase;
        edges_likelihood[2 * k] += 1 - phase;
    }
    if (!tsd_isclose(num_unphased,
            sum_flagged_likelihoods(num_edges, edges_likelihood, edges_unphased))) {
        ret = tsd_trace_error(TSD_ERR_UNPHASED_LIKELIHOOD_NOT_CONSERVED);
        goto out;
    }
out:
    tsd_safe_free(edges_unphased);
    return ret;
}

/* The number of members of each sample set below each mutation, for each tree. */
int
tsd_mutation_frequency(tsd_size_t num_nodes, tsd_size_t num_sample_sets,
    tsd_size_t num_edges, tsd_size_t num_mutations, const uint8_t *nodes_sample,
    const tsd_id_t *mutations_node, const double *mutations_position,
    const tsd_id_t *edges_parent, const tsd_id_t *edges_child, const double *edges_left,
    const double *edges_right, const tsd_id_t *indexes_insert,
    const tsd_id_t *indexes_remove, double sequence_length, int32_t *mutations_freq)
{
    int ret = 0;
    tsd_id_t *indexes_mutation = NULL;
    tsd_id_t *nodes_parent = NULL;
    int32_t *nodes_samples = NULL;
    const tsd_size_t S = num_sample_sets;
    tsd_size_t a, b, d, j, k;
    tsd_id_t e, p, c, m;
    double left, right;

    indexes_mutation
        = malloc((num_mutations > 0 ? num_mutations : 1) * sizeof(tsd_id_t));
    nodes_parent = malloc((num_nodes > 0 ? num_nodes : 1) * sizeof(*nodes_parent));
    nodes_samples = malloc((num_nodes * S > 0 ? num_nodes * S : 1) * sizeof(int32_t));
    if (indexes_mutation == NULL || nodes_parent == NULL || nodes_samples == NULL) {
        ret = tsd_trace_error(TSD_ERR_NO_MEMORY);
        goto out;
    }
    ret = tsd_argsort(num_mutations, mutations_position, indexes_mutation);
    if (ret != 0) {
        goto out;
    }
    for (j = 0; j < num_nodes; j++) {
        nodes_parent[j] = TSD_NULL;
        for (k = 0; k < S; k++) {
            nodes_samples[j * S + k] = nodes_sample[j * S + k] ? 1 : 0;
        }
    }
    for (j = 0; j < num_mutations * S; j++) {
        mutations_freq[j] = 0;
    }

    a = 0;
    b = 0;
    d = 0;
    left = 0.0;
    while (a < num_edges || b < num_edges) {
        while (b < num_edges && edges_right[indexes_remove[b]] == left) { /* out */
            e = indexes_remove[b];
            p = edges_parent[e];
            c = edges_child[e];
            nodes_parent[c] = TSD_NULL;
            while (p != TSD_NULL) {
                for (k = 0; k < S; k++) {
                    nodes_samples[(tsd_size_t) p * S + k]
                        -= nodes_samples[(tsd_size_t) c * S + k];
                }
                p = nodes_parent[p];
            }
            b++;
        }
        while (a < num_edges && edges_left[indexes_insert[a]] == left) { /* in */
            e = indexes_insert[a];
            p = edges_parent[e];
            c = edges_child[e];
            nodes_parent[c] = p;
            while (p != TSD_NULL) {
                for (k = 0; k < S; k++) {
                    nodes_samples[(tsd_size_t) p * S + k]
                        += nodes_samples[(tsd_size_t) c * S + k];
                }
                p = nodes_parent[p];
            }
            a++;
        }
        right = sequence_length;
        if (b < num_edges) {
            right = tsd_min(right, edges_right[indexes_remove[b]]);
        }
        if (a < num_edges) {
            right = tsd_min(right, edges_left[indexes_insert[a]]);
        }
        left = right;
        while (d < num_mutations && mutations_position[indexes_mutation[d]] < right) {
            m = indexes_mutation[d];
            c = mutations_node[m];
            for (k = 0; k < S; k++) {
                mutations_freq[(tsd_size_t) m * S + k]
                    = nodes_samples[(tsd_size_t) c * S + k];
            }
            d++;
        }
    }
out:
    tsd_safe_free(indexes_mutation);
    tsd_safe_free(nodes_parent);
    tsd_safe_free(nodes_samples);
    return ret;
}
