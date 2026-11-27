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

#ifndef TSD_ERR_H
#define TSD_ERR_H

#include <stdio.h>
#include <stdlib.h>

/* clang-format off */
#define TSD_ERR_GENERIC                                             -1
#define TSD_ERR_NO_MEMORY                                           -2
#define TSD_ERR_BAD_PARAM                                           -3

/* Floating-point division by zero */
#define TSD_ERR_ZERO_DIVISION                                       -4

/* hypergeo */
#define TSD_ERR_INVALID_HYPERGEO                                    -5

/* approx: failures of the gamma approximations */
#define TSD_ERR_NONPOSITIVE_OR_NONFINITE_MOMENTS                    -6
#define TSD_ERR_JENSEN_VIOLATION                                    -7
#define TSD_ERR_KL_MAX_ITERATIONS                                   -8
#define TSD_ERR_KL_INVALID_SHAPE                                    -9
#define TSD_ERR_NONPOSITIVE_CENTRAL_MOMENTS                         -10
#define TSD_ERR_UNSORTED_QUANTILES                                  -11
#define TSD_ERR_QUANTILE_MAX_ITERATIONS                             -12
#define TSD_ERR_NEGATIVE_SHAPE                                      -13
/* approx: violated preconditions */
#define TSD_ERR_NONPOSITIVE_MOMENTS                                 -14
#define TSD_ERR_INVALID_FIXED_AGE                                   -15

/* phasing */
#define TSD_ERR_UNPHASED_INDIVIDUAL_EDGES                           -16
#define TSD_ERR_BLOCK_OUT_OF_RANGE                                  -17
#define TSD_ERR_PHASE_OUT_OF_RANGE                                  -18
#define TSD_ERR_UNPHASED_LIKELIHOOD_NOT_CONSERVED                   -19

/* util */
#define TSD_ERR_MUTATION_NODE_NOT_IN_TREE                           -20
#define TSD_ERR_FIXED_NODES_VIOLATE_CONSTRAINT                      -21

/* rescaling */
#define TSD_ERR_NONPOSITIVE_INTERVALS                               -22
#define TSD_ERR_NEGATIVE_CHANGEPOINT_PARAMETER                      -23
#define TSD_ERR_ZERO_EDGE_SPAN                                      -24
#define TSD_ERR_NON_INCREASING_BREAKPOINTS                          -25
#define TSD_ERR_INVALID_QUANTILE_WIDTH                              -26
#define TSD_ERR_INVALID_POSTERIOR                                   -27
#define TSD_ERR_TIME_BELOW_FIRST_BREAKPOINT                         -28
#define TSD_ERR_NO_POSTERIORS                                       -29

/* variational */
#define TSD_ERR_INVALID_MAX_SHAPE                                   -30
#define TSD_ERR_INVALID_MIN_STEP                                    -31
#define TSD_ERR_INVALID_NODE_POSTERIOR                              -32
#define TSD_ERR_INVALID_DAMPING                                     -33
#define TSD_ERR_TWIN_EDGE_IN_PHASED_PASS                            -34
#define TSD_ERR_NONPOSITIVE_PENALTY                                 -35
/* clang-format on */

const char *tsd_strerror(int err);

#ifdef TSD_TRACE_ERRORS

static inline int
_tsd_trace_error(int err, int line, const char *file)
{
    fprintf(stderr, "tsdate-trace-error: %d='%s' at line %d in %s\n", err,
        tsd_strerror(err), line, file);
    return err;
}

#define tsd_trace_error(err) (_tsd_trace_error(err, __LINE__, __FILE__))
#else
#define tsd_trace_error(err) (err)
#endif

#define TSD_BUG_ASSERT_MESSAGE                                                          \
    "Please report this bug at https://github.com/tskit-dev/tsdate/issues,"             \
    " ideally with a reproducible example."

/* Cheap internal-invariant check that is always compiled in. */
#define tsd_bug_assert(condition)                                                       \
    do {                                                                                \
        if (!(condition)) {                                                             \
            fprintf(stderr, "Bug detected in %s at line %d. %s\n", __FILE__, __LINE__,  \
                TSD_BUG_ASSERT_MESSAGE);                                                \
            abort();                                                                    \
        }                                                                               \
    } while (0)

void _tsd_safe_free(void **ptr);
#define tsd_safe_free(pointer) _tsd_safe_free((void **) &(pointer))

#endif
