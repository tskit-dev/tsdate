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
 * The expectation propagation passes, ported from tsdate/variational.py (numba): the
 * damping helpers, the passes over edges (blocks of unphased singletons), the prior
 * and the mutations. Operation order follows the numba code so that results agree
 * bit for bit.
 */

#include <float.h>
#include <math.h>
#include <stdlib.h>

#include "mathutil.h"
#include "tsdate.h"

/* Indexes into the factors, as in tsdate.variational */
#define ROOTWARD 0
#define LEAFWARD 1
#define MIXPRIOR 0
#define CONSTRNT 1
#define LOWER    0
#define UPPER    1

/* Entry `i` of an array of pairs, `j` of an array of 2x2 factors. */
#define PAIR(array, i)          ((array) + 2 * (tsd_size_t) (i))
#define FACTOR(array, i, which) ((array) + 4 * (tsd_size_t) (i) + 2 * (which))

void
tsd_rescale_factors(const tsd_ep_factors_t *factors)
{
    const double *scale = factors->scale;
    tsd_size_t j, k;
    double *f;

    for (j = 0; j < factors->num_edges; j++) {
        f = FACTOR(factors->edge, j, ROOTWARD);
        for (k = 0; k < 2; k++) {
            f[k] *= scale[factors->edges_parent[j]];
        }
        f = FACTOR(factors->edge, j, LEAFWARD);
        for (k = 0; k < 2; k++) {
            f[k] *= scale[factors->edges_child[j]];
        }
    }
    for (j = 0; j < factors->num_blocks; j++) {
        f = FACTOR(factors->block, j, ROOTWARD);
        for (k = 0; k < 2; k++) {
            f[k] *= scale[factors->blocks_left[j]];
        }
        f = FACTOR(factors->block, j, LEAFWARD);
        for (k = 0; k < 2; k++) {
            f[k] *= scale[factors->blocks_right[j]];
        }
    }
    for (j = 0; j < factors->num_nodes; j++) {
        f = FACTOR(factors->node, j, MIXPRIOR);
        for (k = 0; k < 2; k++) {
            f[k] *= scale[j];
        }
        f = FACTOR(factors->node, j, CONSTRNT);
        for (k = 0; k < 2; k++) {
            f[k] *= scale[j];
        }
        factors->scale[j] = 1.0;
    }
}

void
tsd_assemble_factors(const tsd_ep_factors_t *factors, double *posterior)
{
    tsd_size_t j, k;
    double *post;
    const double *f;

    for (j = 0; j < 2 * factors->num_nodes; j++) {
        posterior[j] = 0.0;
    }
    for (j = 0; j < factors->num_edges; j++) {
        post = PAIR(posterior, factors->edges_parent[j]);
        f = FACTOR(factors->edge, j, ROOTWARD);
        for (k = 0; k < 2; k++) {
            post[k] += f[k];
        }
        post = PAIR(posterior, factors->edges_child[j]);
        f = FACTOR(factors->edge, j, LEAFWARD);
        for (k = 0; k < 2; k++) {
            post[k] += f[k];
        }
    }
    for (j = 0; j < factors->num_blocks; j++) {
        post = PAIR(posterior, factors->blocks_left[j]);
        f = FACTOR(factors->block, j, ROOTWARD);
        for (k = 0; k < 2; k++) {
            post[k] += f[k];
        }
        post = PAIR(posterior, factors->blocks_right[j]);
        f = FACTOR(factors->block, j, LEAFWARD);
        for (k = 0; k < 2; k++) {
            post[k] += f[k];
        }
    }
    for (j = 0; j < factors->num_nodes; j++) {
        post = PAIR(posterior, j);
        f = FACTOR(factors->node, j, MIXPRIOR);
        for (k = 0; k < 2; k++) {
            post[k] += f[k];
        }
    }
    for (j = 0; j < factors->num_nodes; j++) {
        post = PAIR(posterior, j);
        f = FACTOR(factors->node, j, CONSTRNT);
        for (k = 0; k < 2; k++) {
            post[k] += f[k];
        }
    }
}

/*
 * If `x - y` is too small, find `d` so that `x - d * y` is large enough:
 *
 *     x[0] - d * y[0] + 1 >= (x[0] + 1) * s
 *     x[1] - d * y[1] >= y[1] * s
 *
 * for `0 < s < 1`.
 */
int
tsd_damp(const double x[2], const double y[2], double s, double *out)
{
    int ret = 0;
    double a, b, d;

    if (y[0] == 0.0 && y[1] == 0.0 && x[0] == 0.0 && x[1] == 0.0) {
        *out = 1.0;
        goto out;
    }
    if (!(0 < s && s < 1)) {
        ret = tsd_trace_error(TSD_ERR_INVALID_MIN_STEP);
        goto out;
    }
    if (!(0.0 < x[0] + 1 && 0.0 < x[1])) {
        ret = tsd_trace_error(TSD_ERR_INVALID_NODE_POSTERIOR);
        goto out;
    }
    a = 1.0;
    if (!(1 + x[0] - y[0] > (1 + x[0]) * s)) {
        if (y[0] == 0.0) {
            ret = tsd_trace_error(TSD_ERR_ZERO_DIVISION);
            goto out;
        }
        a = (1 - s) * (1 + x[0]) / y[0];
    }
    b = 1.0;
    if (!(x[1] - y[1] > x[1] * s)) {
        if (y[1] == 0.0) {
            ret = tsd_trace_error(TSD_ERR_ZERO_DIVISION);
            goto out;
        }
        b = (1 - s) * x[1] / y[1];
    }
    d = tsd_min(a, b);
    if (!(0.0 < d && d <= 1.0)) {
        ret = tsd_trace_error(TSD_ERR_INVALID_DAMPING);
        goto out;
    }
    *out = d;
out:
    return ret;
}

/* Find `d` so that `d * x[0] + 1 <= s` or `d * x[0] + 1 >= 1 / s`. */
int
tsd_rescale(const double x[2], double s, double *out)
{
    int ret = 0;

    if (x[0] == 0.0 && x[1] == 0.0) {
        *out = 1.0;
        goto out;
    }
    if (!(0 < x[0] + 1 && 0 < x[1])) {
        ret = tsd_trace_error(TSD_ERR_INVALID_NODE_POSTERIOR);
        goto out;
    }
    if (1 + x[0] > s) {
        if (x[0] == 0.0) {
            ret = tsd_trace_error(TSD_ERR_ZERO_DIVISION);
            goto out;
        }
        *out = (s - 1) / x[0];
    } else if (1 + x[0] < 1 / s) {
        /* x[0] is not zero: that would have taken the first branch (s < 1). */
        *out = (1 / s - 1) / x[0];
    } else {
        *out = 1.0;
    }
out:
    return ret;
}

/*
 * Check that the upper bound on node age is greater than the maximum lower bound for
 * ages of descendants.
 */
int
tsd_check_valid_constraints(tsd_size_t num_nodes, tsd_size_t num_edges,
    const double *constraints, const tsd_id_t *edges_parent, const tsd_id_t *edges_child)
{
    int ret = 0;
    double *lower_max = NULL;
    tsd_size_t j;
    tsd_id_t p, c;

    lower_max = malloc((num_nodes > 0 ? num_nodes : 1) * sizeof(*lower_max));
    if (lower_max == NULL) {
        ret = tsd_trace_error(TSD_ERR_NO_MEMORY);
        goto out;
    }
    for (j = 0; j < num_nodes; j++) {
        lower_max[j] = constraints[2 * j + LOWER];
    }
    for (j = 0; j < num_edges; j++) {
        p = edges_parent[j];
        c = edges_child[j];
        lower_max[p] = tsd_max(lower_max[c], lower_max[p]);
    }
    for (j = 0; j < num_nodes; j++) {
        if (lower_max[j] > constraints[2 * j + UPPER]) {
            ret = tsd_trace_error(TSD_ERR_INCONSISTENT_CONSTRAINTS);
            goto out;
        }
    }
out:
    tsd_safe_free(lower_max);
    return ret;
}

/* cavity = posterior - delta * message */
static void
get_cavity(
    const double posterior[2], double delta, const double message[2], double cavity[2])
{
    cavity[0] = posterior[0] - delta * message[0];
    cavity[1] = posterior[1] - delta * message[1];
}

/* The message from a factor, which is stored divided by the scale of the node. */
static void
get_message(const double factor[2], double scale, double message[2])
{
    message[0] = factor[0] * scale;
    message[1] = factor[1] * scale;
}

/*
 * Update a factor given the cavity and the new posterior:
 *     factor = factor * (1 - delta) + (posterior - cavity) / scale
 */
static void
update_factor(double factor[2], double delta, const double posterior[2],
    const double cavity[2], double scale)
{
    const double keep = 1.0 - delta;

    factor[0] *= keep;
    factor[1] *= keep;
    factor[0] += (posterior[0] - cavity[0]) / scale;
    factor[1] += (posterior[1] - cavity[1]) / scale;
}

/* Rescale the posterior of a node to keep its shape bounded, and its scale to match. */
static int
rescale_posterior(double posterior[2], double *scale, double max_shape)
{
    int ret;
    double eta;

    ret = tsd_rescale(posterior, max_shape, &eta);
    if (ret != 0) {
        goto out;
    }
    posterior[0] *= eta;
    posterior[1] *= eta;
    *scale *= eta;
out:
    return ret;
}

/* The arguments of the update of one edge (block), shared by all the edges of a pass. */
typedef struct {
    double max_shape;
    double min_step;
    int unphased;
    const double *constraints;
    const double *likelihoods;
    double *posterior;
    double *scale;
    double *factor; /* the edge or block factors */
    double *lognorm;
} likelihood_pass_t;

/*
 * Update an edge with a fixed parent (`fixed_parent`) or a fixed child: only the
 * free node, `free_node`, has a posterior to update.
 */
static int
update_fixed_edge(const likelihood_pass_t *pass, tsd_id_t i, tsd_id_t fixed_node,
    tsd_id_t free_node, int fixed_parent)
{
    int ret = 0;
    const int which = fixed_parent ? LEAFWARD : ROOTWARD;
    double *post = PAIR(pass->posterior, free_node);
    double *factor = FACTOR(pass->factor, i, which);
    double *scale = pass->scale + free_node;
    const double fixed_age = pass->constraints[2 * (tsd_size_t) fixed_node + LOWER];
    double message[2], cavity[2], edge_likelihood[2], proj[2];
    double delta, logl;

    get_message(factor, *scale, message);
    ret = tsd_damp(post, message, pass->min_step, &delta);
    if (ret != 0) {
        goto out;
    }
    get_cavity(post, delta, message, cavity);
    edge_likelihood[0] = delta * PAIR(pass->likelihoods, i)[0];
    edge_likelihood[1] = delta * PAIR(pass->likelihoods, i)[1];
    if (pass->unphased) {
        ret = tsd_sideways_projection(fixed_age, cavity, edge_likelihood, &logl, proj);
    } else if (fixed_parent) {
        ret = tsd_leafward_projection(fixed_age, cavity, edge_likelihood, &logl, proj);
    } else {
        ret = tsd_rootward_projection(fixed_age, cavity, edge_likelihood, &logl, proj);
    }
    if (ret != 0) {
        goto out;
    }
    pass->lognorm[i] = logl;
    post[0] = proj[0];
    post[1] = proj[1];
    update_factor(factor, delta, post, cavity, *scale);
    ret = rescale_posterior(post, scale, pass->max_shape);
out:
    return ret;
}

/* A block with a single node: the parent of the edge is its child. */
static int
update_twin_edge(const likelihood_pass_t *pass, tsd_id_t i, tsd_id_t node)
{
    int ret = 0;
    double *post = PAIR(pass->posterior, node);
    double *factor = FACTOR(pass->factor, i, ROOTWARD);
    double *scale = pass->scale + node;
    double message[2], cavity[2], edge_likelihood[2], proj[2];
    double delta, logl;

    get_message(factor, *scale, message);
    ret = tsd_damp(post, message, pass->min_step, &delta);
    if (ret != 0) {
        goto out;
    }
    get_cavity(post, delta, message, cavity);
    edge_likelihood[0] = delta * PAIR(pass->likelihoods, i)[0];
    edge_likelihood[1] = delta * PAIR(pass->likelihoods, i)[1];
    if (!pass->unphased) {
        ret = tsd_trace_error(TSD_ERR_TWIN_EDGE_IN_PHASED_PASS);
        goto out;
    }
    ret = tsd_twin_projection(cavity, edge_likelihood, &logl, proj);
    if (ret != 0) {
        goto out;
    }
    pass->lognorm[i] = logl;
    post[0] = proj[0];
    post[1] = proj[1];
    update_factor(factor, delta, post, cavity, *scale);
    ret = rescale_posterior(post, scale, pass->max_shape);
out:
    return ret;
}

/* An edge with both nodes free, and a different parent and child. */
static int
update_free_edge(const likelihood_pass_t *pass, tsd_id_t i, tsd_id_t p, tsd_id_t c)
{
    int ret = 0;
    double *post_p = PAIR(pass->posterior, p);
    double *post_c = PAIR(pass->posterior, c);
    double *factor_p = FACTOR(pass->factor, i, ROOTWARD);
    double *factor_c = FACTOR(pass->factor, i, LEAFWARD);
    double *scale_p = pass->scale + p;
    double *scale_c = pass->scale + c;
    double message_p[2], message_c[2], cavity_p[2], cavity_c[2], edge_likelihood[2];
    double proj_p[2], proj_c[2];
    double delta_p, delta_c, delta, logl;

    get_message(factor_p, *scale_p, message_p);
    get_message(factor_c, *scale_c, message_c);
    ret = tsd_damp(post_p, message_p, pass->min_step, &delta_p);
    if (ret != 0) {
        goto out;
    }
    ret = tsd_damp(post_c, message_c, pass->min_step, &delta_c);
    if (ret != 0) {
        goto out;
    }
    delta = tsd_min(delta_p, delta_c);
    get_cavity(post_p, delta, message_p, cavity_p);
    get_cavity(post_c, delta, message_c, cavity_c);
    edge_likelihood[0] = delta * PAIR(pass->likelihoods, i)[0];
    edge_likelihood[1] = delta * PAIR(pass->likelihoods, i)[1];
    if (pass->unphased) {
        ret = tsd_unphased_projection(
            cavity_p, cavity_c, edge_likelihood, &logl, proj_p, proj_c);
    } else {
        ret = tsd_gamma_projection(
            cavity_p, cavity_c, edge_likelihood, &logl, proj_p, proj_c);
    }
    if (ret != 0) {
        goto out;
    }
    pass->lognorm[i] = logl;
    post_p[0] = proj_p[0];
    post_p[1] = proj_p[1];
    post_c[0] = proj_c[0];
    post_c[1] = proj_c[1];
    update_factor(factor_p, delta, post_p, cavity_p, *scale_p);
    update_factor(factor_c, delta, post_c, cavity_c, *scale_c);
    ret = rescale_posterior(post_p, scale_p, pass->max_shape);
    if (ret != 0) {
        goto out;
    }
    ret = rescale_posterior(post_c, scale_c, pass->max_shape);
out:
    return ret;
}

int
tsd_propagate_likelihood(tsd_size_t num_order, const tsd_id_t *edge_order,
    tsd_size_t num_edges, const tsd_id_t *edges_parent, const tsd_id_t *edges_child,
    const double *likelihoods, const double *constraints, double max_shape,
    double min_step, int unphased, double *posterior, const tsd_ep_factors_t *factors,
    double *lognorm)
{
    int ret = 0;
    const double tiny = sqrt(DBL_MIN);
    likelihood_pass_t pass;
    tsd_size_t j;
    tsd_id_t i, p, c;
    int fixed_p, fixed_c;

    if (!(max_shape >= 1.0)) {
        ret = tsd_trace_error(TSD_ERR_INVALID_MAX_SHAPE);
        goto out;
    }
    if (!(0.0 < min_step && min_step < 1.0)) {
        ret = tsd_trace_error(TSD_ERR_INVALID_MIN_STEP);
        goto out;
    }
    pass.max_shape = max_shape;
    pass.min_step = min_step;
    pass.unphased = unphased;
    pass.constraints = constraints;
    pass.likelihoods = likelihoods;
    pass.posterior = posterior;
    pass.scale = factors->scale;
    pass.factor = unphased ? factors->block : factors->edge;
    pass.lognorm = lognorm;
    tsd_bug_assert(num_edges == (unphased ? factors->num_blocks : factors->num_edges));

    for (j = 0; j < num_order; j++) {
        i = edge_order[j];
        p = edges_parent[i];
        c = edges_child[i];
        if (pass.scale[p] < tiny || pass.scale[c] < tiny) {
            tsd_rescale_factors(factors);
        }
        fixed_p = constraints[2 * (tsd_size_t) p + LOWER]
                  == constraints[2 * (tsd_size_t) p + UPPER];
        fixed_c = constraints[2 * (tsd_size_t) c + LOWER]
                  == constraints[2 * (tsd_size_t) c + UPPER];
        if (fixed_p && fixed_c) {
            continue;
        } else if (fixed_p) {
            /* a sample is the ancestor of another sample */
            ret = update_fixed_edge(&pass, i, p, c, 1);
        } else if (fixed_c) {
            /* a sample has age greater than zero */
            ret = update_fixed_edge(&pass, i, c, p, 0);
        } else if (p == c) {
            /* singleton block with single parent */
            ret = update_twin_edge(&pass, i, p);
        } else {
            ret = update_free_edge(&pass, i, p, c);
        }
        if (ret != 0) {
            goto out;
        }
    }
out:
    return ret;
}

/* The prior cavity of a free node: the posterior without the mixture prior factor. */
static void
get_prior_cavity(const tsd_ep_factors_t *factors, const double *posterior, tsd_size_t j,
    double cavity[2])
{
    double message[2];

    get_message(FACTOR(factors->node, j, MIXPRIOR), factors->scale[j], message);
    get_cavity(PAIR(posterior, j), 1.0, message, cavity);
}

/*
 * The mean over free nodes of shape / rate, with `penalty` added to the rate if
 * `with_penalty`, summing sequentially.
 */
static double
mean_shape_over_rate(const tsd_ep_factors_t *factors, const double *posterior,
    const uint8_t *free, int with_penalty, double penalty)
{
    double cavity[2];
    double shape, rate, term;
    double total = 0.0;
    tsd_size_t j, count = 0;

    for (j = 0; j < factors->num_nodes; j++) {
        if (free[j]) {
            get_prior_cavity(factors, posterior, j, cavity);
            shape = cavity[0] + 1;
            rate = cavity[1];
            if (with_penalty) {
                rate = rate + penalty;
            }
            term = shape / rate;
            total += term;
            count++;
        }
    }
    return total / (double) count;
}

int
tsd_propagate_prior(const uint8_t *free, double max_shape, int64_t em_maxitt,
    double em_reltol, double *posterior, const tsd_ep_factors_t *factors)
{
    int ret = 0;
    const tsd_size_t num_nodes = factors->num_nodes;
    double cavity[2];
    double penalty, delta;
    int64_t itt;
    tsd_size_t j;
    int any_free = 0;

    if (!(max_shape >= 1.0)) {
        ret = tsd_trace_error(TSD_ERR_INVALID_MAX_SHAPE);
        goto out;
    }
    for (j = 0; j < num_nodes; j++) {
        any_free = any_free || free[j];
    }
    if (!any_free) {
        goto out;
    }

    /* fit an exponential to cavity distributions for unconstrained nodes */
    penalty = 1 / mean_shape_over_rate(factors, posterior, free, 0, 0.0);
    itt = 0;
    delta = INFINITY;
    while (fabs(delta) > fabs(penalty) * em_reltol) {
        if (itt > em_maxitt) {
            break;
        }
        delta = 1 / mean_shape_over_rate(factors, posterior, free, 1, penalty) - penalty;
        penalty += delta;
        itt++;
    }
    if (!(penalty > 0)) {
        ret = tsd_trace_error(TSD_ERR_NONPOSITIVE_PENALTY);
        goto out;
    }

    /* update posteriors and rescale to keep shape bounded */
    for (j = 0; j < num_nodes; j++) {
        if (free[j]) {
            get_prior_cavity(factors, posterior, j, cavity);
            PAIR(posterior, j)[1] = cavity[1] + penalty;
            FACTOR(factors->node, j, MIXPRIOR)
            [0] = (PAIR(posterior, j)[0] - cavity[0]) / factors->scale[j];
            FACTOR(factors->node, j, MIXPRIOR)
            [1] = (PAIR(posterior, j)[1] - cavity[1]) / factors->scale[j];
        }
    }
    for (j = 0; j < num_nodes; j++) {
        if (free[j]) {
            ret = rescale_posterior(PAIR(posterior, j), factors->scale + j, max_shape);
            if (ret != 0) {
                goto out;
            }
        }
    }
out:
    return ret;
}

/* The arguments of the projection of mutations; nothing is updated. */
typedef struct {
    int unphased;
    const double *constraints;
    const double *likelihoods;
    const double *posterior;
    const double *scale;
    const double *factor; /* the edge or block factors */
} mutation_pass_t;

/*
 * The posterior of one mutation, from the edge (block) above it: with both nodes
 * fixed, the age of each; with one fixed, the fixed age and the cavity of the other;
 * otherwise the cavities of both. The messages are not damped (the step is one).
 */
static int
project_mutation(const mutation_pass_t *pass, tsd_id_t i, tsd_id_t p, tsd_id_t c,
    double *phase, double proj[2])
{
    int ret = 0;
    const double *constraints = pass->constraints;
    const int fixed_p = constraints[2 * (tsd_size_t) p + LOWER]
                        == constraints[2 * (tsd_size_t) p + UPPER];
    const int fixed_c = constraints[2 * (tsd_size_t) c + LOWER]
                        == constraints[2 * (tsd_size_t) c + UPPER];
    const double age_p = constraints[2 * (tsd_size_t) p + LOWER];
    const double age_c = constraints[2 * (tsd_size_t) c + LOWER];
    const double *post_p = PAIR(pass->posterior, p);
    const double *post_c = PAIR(pass->posterior, c);
    const double *likelihood = PAIR(pass->likelihoods, i);
    double message_p[2], message_c[2], cavity_p[2], cavity_c[2];

    get_message(FACTOR(pass->factor, i, ROOTWARD), pass->scale[p], message_p);
    get_message(FACTOR(pass->factor, i, LEAFWARD), pass->scale[c], message_c);
    get_cavity(post_p, 1.0, message_p, cavity_p);
    get_cavity(post_c, 1.0, message_c, cavity_c);
    if (fixed_p && fixed_c) {
        if (pass->unphased) {
            ret = tsd_mutation_block_projection(age_p, age_c, phase, proj);
        } else {
            ret = tsd_mutation_edge_projection(age_p, age_c, phase, proj);
        }
    } else if (fixed_p) {
        if (pass->unphased) {
            ret = tsd_mutation_sideways_projection(
                age_p, cavity_c, likelihood, phase, proj);
        } else {
            ret = tsd_mutation_leafward_projection(
                age_p, cavity_c, likelihood, phase, proj);
        }
    } else if (fixed_c) {
        if (pass->unphased) {
            ret = tsd_mutation_sideways_projection(
                age_c, cavity_p, likelihood, phase, proj);
        } else {
            ret = tsd_mutation_rootward_projection(
                age_c, cavity_p, likelihood, phase, proj);
        }
    } else if (p == c) {
        /* singleton block with single parent */
        ret = tsd_mutation_twin_projection(cavity_p, likelihood, phase, proj);
    } else if (pass->unphased) {
        ret = tsd_mutation_unphased_projection(
            cavity_p, cavity_c, likelihood, phase, proj);
    } else {
        ret = tsd_mutation_gamma_projection(cavity_p, cavity_c, likelihood, phase, proj);
    }
    return ret;
}

int
tsd_propagate_mutations(tsd_size_t num_order, const tsd_id_t *mutations_order,
    const tsd_id_t *mutations_edge, const tsd_id_t *edges_parent,
    const tsd_id_t *edges_child, const double *likelihoods, const double *constraints,
    const double *posterior, const tsd_ep_factors_t *factors, int unphased,
    double *mutations_posterior, double *mutations_phase)
{
    int ret = 0;
    mutation_pass_t pass;
    tsd_size_t j;
    tsd_id_t m, i;
    double phase, proj[2];

    pass.unphased = unphased;
    pass.constraints = constraints;
    pass.likelihoods = likelihoods;
    pass.posterior = posterior;
    pass.scale = factors->scale;
    pass.factor = unphased ? factors->block : factors->edge;
    for (j = 0; j < num_order; j++) {
        m = mutations_order[j];
        i = mutations_edge[m];
        if (i == TSD_NULL) { /* skip mutations above root */
            continue;
        }
        ret = project_mutation(&pass, i, edges_parent[i], edges_child[i], &phase, proj);
        if (ret != 0) {
            goto out;
        }
        mutations_phase[m] = phase;
        mutations_posterior[2 * (tsd_size_t) m] = proj[0];
        mutations_posterior[2 * (tsd_size_t) m + 1] = proj[1];
    }
out:
    return ret;
}
