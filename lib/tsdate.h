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

#ifndef TSDATE_H
#define TSDATE_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>

#include "err.h"

/* Node, edge, mutation and block ids; matches the tskit id arrays. */
typedef int32_t tsd_id_t;
/* Counts and sizes; matches tsk_size_t. */
typedef uint64_t tsd_size_t;

/*
 * Scalar special functions (hypergeo.c). Functions that can fail return 0 or a
 * negative TSD_ERR_* code and write their results to the trailing output pointers;
 * those that cannot fail return the result. As in the numba code they replace, NaN
 * and infinite results are in-band (not errors): only violated domain assertions
 * (TSD_ERR_INVALID_HYPERGEO) and divisions by zero (TSD_ERR_ZERO_DIVISION) fail.
 */
int tsd_digamma(double x, double *out);
int tsd_trigamma(double x, double *out);
double tsd_betaln(double p, double q);
int tsd_gammainc_der(double p, double x, double *out);
int tsd_hyperu_laplace(double a, double b, double x, double *value, double *dvalue);
int tsd_hyp1f1_laplace(double a, double b, double x, double *out);
int tsd_hyp2f1_laplace(double a, double b, double c, double x, double *out);

/*
 * The inverse of the regularised lower incomplete gamma function, x such that
 * gammainc(a, x) = p, is not implemented in the library: SciPy's gammaincinv is
 * called, as the numba code did. The caller (the Python module) supplies it, and
 * library functions that need it (the approx and rescaling functions) take a
 * parameter of this type, so that the library has no global state and no dependency
 * on SciPy. The function returns NaN, 0 or infinity (not an error) for arguments
 * outside the domain.
 */
typedef double (*tsd_gammaincinv_fn)(double a, double p);

/*
 * Approximations of gamma distributions and of the moments of the EP updates
 * (approx.c).
 *
 * Gamma distributions are written as shape and rate (a, b), with density
 * proportional to t^(a - 1) exp(-b t), or as "natural parameters" (a - 1, b); the
 * latter are what the gamma_* approximation functions and the projections return and
 * take. Errors are as for the special functions above, plus the TSD_ERR_* codes of
 * the approximations whose messages are those of numba's KLMinimizationFailedError.
 *
 * Convention for the moment and projection functions (sections of approx.c):
 *
 *  - A function that can fail returns int: 0 or a negative TSD_ERR_* code. Results
 *    are written through trailing output pointers, which are written only on
 *    success (return 0) and must not alias the inputs.
 *  - Moment functions take, in the order of the numba arguments, ages (`t_i`, `t_j`:
 *    a fixed parent or child) and gamma SHAPE and rate (`a_i`, `b_i`, `a_j`, `b_j`:
 *    not natural parameters), the mutation count `y_ij` and the rate `mu_ij`. They
 *    write the values that numba returns, in the same order: the log normalising
 *    constant (`logl`) or the phase probability (`pr_m`), then means and variances.
 *    A result that numba returns as NaN (an invalid hypergeometric domain) is
 *    returned as NaN in every output, with return value 0.
 *  - Projection functions take the natural parameters of the cavity gammas as
 *    `const double pars_i[2]`, `pars_j[2]` and the edge data as `pars_ij[2]`
 *    (`y_ij`, `mu_ij`), or ages as in the moment functions. They write the log
 *    normalising constant (`*logl`, or `*pr_m` for mutation projections) and the
 *    projected natural parameters through `double proj_i[2]`, `proj_j[2]` (or
 *    `proj_m[2]`). Where numba returns the input parameters because the moments are
 *    unusable, `*logl` is NaN and `proj_*` is a copy of `pars_*`; mutation
 *    projections write NaN to `*pr_m` and to both entries of `proj_m`.
 *  - Functions of the gamma approximations that numba returns as a pair of natural
 *    parameters write them through `double out[2]`.
 *  - The `tsd_valid_*` predicates return 1 if the arguments are usable and 0 if not;
 *    a failure is in-band (numba returned False), not an error.
 */
int tsd_approximate_log_moments(
    double mean, double variance, double *logx, double *xlogx, double *logx2);
int tsd_approximate_gamma_kl(double x, double logx, double out[2]);
int tsd_approximate_gamma_mom(double mean, double variance, double out[2]);
int tsd_approximate_gamma_iqr(double q1, double q2, double x1, double x2,
    double max_shape, tsd_gammaincinv_fn gammaincinv, double out[2]);
/* The natural parameters of the gamma whose mean and mean log are the averages of
 * those of `num_gammas` gammas given by natural parameters `alpha`, `beta`. */
int tsd_average_gammas(
    tsd_size_t num_gammas, const double *alpha, const double *beta, double out[2]);

int tsd_valid_moments(double mn, double va);
int tsd_valid_gamma(double s, double r);
int tsd_valid_hyp1f1(double a, double b, double z);
int tsd_valid_hyperu(double a, double b, double z);
int tsd_valid_hyp2f1(double a, double b, double c, double z);

/* Node moments (approx.c) */
int tsd_moments(double a_i, double b_i, double a_j, double b_j, double y_ij,
    double mu_ij, double *logl, double *mn_i, double *va_i, double *mn_j, double *va_j);
int tsd_rootward_moments(double t_j, double a_i, double b_i, double y_ij, double mu_ij,
    double *logl, double *mn_i, double *va_i);
int tsd_leafward_moments(double t_i, double a_j, double b_j, double y_ij, double mu_ij,
    double *logl, double *mn_j, double *va_j);
int tsd_unphased_moments(double a_i, double b_i, double a_j, double b_j, double y_ij,
    double mu_ij, double *logl, double *mn_i, double *va_i, double *mn_j, double *va_j);
int tsd_twin_moments(double a_i, double b_i, double y_ij, double mu_ij, double *logl,
    double *mn_i, double *va_i);
int tsd_sideways_moments(double t_i, double a_j, double b_j, double y_ij, double mu_ij,
    double *logl, double *mn_j, double *va_j);

/* Node projections */
int tsd_gamma_projection(const double pars_i[2], const double pars_j[2],
    const double pars_ij[2], double *logl, double proj_i[2], double proj_j[2]);
int tsd_leafward_projection(double t_i, const double pars_j[2], const double pars_ij[2],
    double *logl, double proj_j[2]);
int tsd_rootward_projection(double t_j, const double pars_i[2], const double pars_ij[2],
    double *logl, double proj_i[2]);
int tsd_unphased_projection(const double pars_i[2], const double pars_j[2],
    const double pars_ij[2], double *logl, double proj_i[2], double proj_j[2]);
int tsd_twin_projection(
    const double pars_i[2], const double pars_ij[2], double *logl, double proj_i[2]);
int tsd_sideways_projection(double t_i, const double pars_j[2], const double pars_ij[2],
    double *logl, double proj_j[2]);

/* Mutation moments */
int tsd_mutation_moments(double a_i, double b_i, double a_j, double b_j, double y_ij,
    double mu_ij, double *mn_m, double *va_m);
int tsd_mutation_rootward_moments(double t_j, double a_i, double b_i, double y_ij,
    double mu_ij, double *mn_m, double *va_m);
int tsd_mutation_leafward_moments(double t_i, double a_j, double b_j, double y_ij,
    double mu_ij, double *mn_m, double *va_m);
int tsd_mutation_unphased_moments(double a_i, double b_i, double a_j, double b_j,
    double y_ij, double mu_ij, double *pr_m, double *mn_m, double *va_m);
int tsd_mutation_twin_moments(double a_i, double b_i, double y_ij, double mu_ij,
    double *pr_m, double *mn_m, double *va_m);
int tsd_mutation_sideways_moments(double t_i, double a_j, double b_j, double y_ij,
    double mu_ij, double *pr_m, double *mn_m, double *va_m);
int tsd_mutation_edge_moments(double t_i, double t_j, double *mn_m, double *va_m);
int tsd_mutation_block_moments(
    double t_i, double t_j, double *pr_m, double *mn_m, double *va_m);

/* Mutation projections */
int tsd_mutation_gamma_projection(const double pars_i[2], const double pars_j[2],
    const double pars_ij[2], double *pr_m, double proj_m[2]);
int tsd_mutation_leafward_projection(double t_i, const double pars_j[2],
    const double pars_ij[2], double *pr_m, double proj_m[2]);
int tsd_mutation_rootward_projection(double t_j, const double pars_i[2],
    const double pars_ij[2], double *pr_m, double proj_m[2]);
int tsd_mutation_edge_projection(double t_i, double t_j, double *pr_m, double proj_m[2]);
int tsd_mutation_unphased_projection(const double pars_i[2], const double pars_j[2],
    const double pars_ij[2], double *pr_m, double proj_m[2]);
int tsd_mutation_twin_projection(
    const double pars_i[2], const double pars_ij[2], double *pr_m, double proj_m[2]);
int tsd_mutation_sideways_projection(double t_i, const double pars_j[2],
    const double pars_ij[2], double *pr_m, double proj_m[2]);
int tsd_mutation_block_projection(
    double t_i, double t_j, double *pr_m, double proj_m[2]);

/* The id of no node, edge, mutation, individual or block. */
#define TSD_NULL ((tsd_id_t) - 1)

/*
 * Phasing of singletons (phasing.c), tree sequence utilities (util.c) and the
 * coalescent prior (prior.c). Flags are arrays of uint8_t (numpy bool); the arrays
 * of the edges are those of a tree sequence, with the indexes of the insertion and
 * removal orders. Results of unknown size are allocated by the function and freed by
 * the caller.
 */
int tsd_argsort(tsd_size_t num_items, const double *keys, tsd_id_t *order);

int tsd_block_singletons(tsd_size_t num_individuals, tsd_size_t num_edges,
    tsd_size_t num_mutations, const uint8_t *individuals_unphased,
    const tsd_id_t *nodes_individual, const tsd_id_t *mutations_node,
    const double *mutations_position, const tsd_id_t *edges_child,
    const double *edges_left, const double *edges_right, const tsd_id_t *indexes_insert,
    const tsd_id_t *indexes_remove, double sequence_length, tsd_size_t *num_blocks,
    double **blocks_stats, tsd_id_t **blocks_edges, tsd_id_t *mutations_block);
int tsd_reallocate_unphased(tsd_size_t num_edges, tsd_size_t num_mutations,
    tsd_size_t num_blocks, const double *mutations_phase,
    const tsd_id_t *mutations_block, const tsd_id_t *blocks_edges,
    double *edges_likelihood);
int tsd_mutation_frequency(tsd_size_t num_nodes, tsd_size_t num_sample_sets,
    tsd_size_t num_edges, tsd_size_t num_mutations, const uint8_t *nodes_sample,
    const tsd_id_t *mutations_node, const double *mutations_position,
    const tsd_id_t *edges_parent, const tsd_id_t *edges_child, const double *edges_left,
    const double *edges_right, const tsd_id_t *indexes_insert,
    const tsd_id_t *indexes_remove, double sequence_length, int32_t *mutations_freq);

int tsd_split_disjoint_nodes(tsd_size_t num_nodes, tsd_size_t num_edges,
    const tsd_id_t *edges_parent, const tsd_id_t *edges_child, const double *edges_left,
    const double *edges_right, const uint8_t *node_excluded, tsd_id_t *new_parent,
    tsd_id_t *new_child, tsd_size_t *num_split, tsd_id_t **nodes_order);
int tsd_relabel_mutations_node(tsd_size_t num_nodes, tsd_size_t num_edges,
    tsd_size_t num_mutations, const tsd_id_t *mutations_node,
    const double *mutations_position, const tsd_id_t *nodes_order,
    const tsd_id_t *edges_parent, const tsd_id_t *edges_child, const double *edges_left,
    const double *edges_right, const tsd_id_t *insert_index,
    const tsd_id_t *remove_index, tsd_id_t *output);
int tsd_constrain_ages(tsd_size_t num_nodes, tsd_size_t num_edges,
    const double *nodes_time, const uint8_t *nodes_fixed, const tsd_id_t *edges_parent,
    const tsd_id_t *edges_child, double epsilon, int64_t max_iterations,
    double *out_time);
int tsd_contains_unary_nodes(tsd_size_t num_nodes, tsd_size_t num_edges,
    const uint8_t *nodes_mask, const tsd_id_t *edges_parent, const double *edges_left,
    const double *edges_right, const tsd_id_t *indexes_insert,
    const tsd_id_t *indexes_remove, double sequence_length, int *contains_unary);

/* `val` has `num_tips` >= 2 rows; `out` has `num_tips + 1` rows. */
int tsd_marginalize_over_ancestors(
    tsd_size_t num_tips, tsd_size_t num_moments, const double *val, double *out);
/* `variance` has `num_tips + 1` entries; `num_tips` >= 1. */
int tsd_conditional_coalescent_variance(tsd_size_t num_tips, double *variance);

#ifdef __cplusplus
}
#endif

#endif
