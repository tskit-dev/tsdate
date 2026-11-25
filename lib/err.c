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

#include <stdlib.h>

#include "err.h"

const char *
tsd_strerror(int err)
{
    const char *ret = "Unknown error";

    switch (err) {
        case 0:
            ret = "Normal exit condition. This is not an error!";
            break;
        case TSD_ERR_GENERIC:
            ret = "Generic tsdate error - please file a bug report.";
            break;
        case TSD_ERR_NO_MEMORY:
            ret = "Out of memory";
            break;
        case TSD_ERR_BAD_PARAM:
            ret = "Bad parameter value provided";
            break;
        case TSD_ERR_ZERO_DIVISION:
            ret = "Floating-point division by zero";
            break;
        case TSD_ERR_INVALID_HYPERGEO:
            ret = "Parameters outside the domain of the hypergeometric function";
            break;
        case TSD_ERR_NONPOSITIVE_OR_NONFINITE_MOMENTS:
            ret = "Nonpositive or nonfinite moments";
            break;
        case TSD_ERR_JENSEN_VIOLATION:
            ret = "log E[t] <= E[log t] violates Jensen's inequality";
            break;
        case TSD_ERR_KL_MAX_ITERATIONS:
            ret = "Maximum iterations reached in KL minimization";
            break;
        case TSD_ERR_KL_INVALID_SHAPE:
            ret = "Invalid shape parameter in KL minimization";
            break;
        case TSD_ERR_NONPOSITIVE_CENTRAL_MOMENTS:
            ret = "Nonpositive central moments";
            break;
        case TSD_ERR_UNSORTED_QUANTILES:
            ret = "Quantiles must be sorted";
            break;
        case TSD_ERR_QUANTILE_MAX_ITERATIONS:
            ret = "Maximum iterations reached in quantile matching";
            break;
        case TSD_ERR_NEGATIVE_SHAPE:
            ret = "Negative shape parameter";
            break;
        case TSD_ERR_NONPOSITIVE_MOMENTS:
            ret = "The mean and variance must be positive";
            break;
        case TSD_ERR_INVALID_FIXED_AGE:
            ret = "The age of a fixed node is outside the domain of the function";
            break;
        case TSD_ERR_UNPHASED_INDIVIDUAL_EDGES:
            ret = "An unphased individual does not have exactly two edges over each "
                  "of its blocks";
            break;
        case TSD_ERR_BLOCK_OUT_OF_RANGE:
            ret = "A block or an edge of a block is out of range";
            break;
        case TSD_ERR_PHASE_OUT_OF_RANGE:
            ret = "The phase of a mutation is not between zero and one";
            break;
        case TSD_ERR_UNPHASED_LIKELIHOOD_NOT_CONSERVED:
            ret = "The total likelihood of the unphased edges is not conserved";
            break;
        case TSD_ERR_MUTATION_NODE_NOT_IN_TREE:
            ret = "A mutation is above a node that is not in any tree before it";
            break;
        case TSD_ERR_FIXED_NODES_VIOLATE_CONSTRAINT:
            ret = "An edge between two fixed nodes violates the age constraint";
            break;
        case TSD_ERR_NONPOSITIVE_INTERVALS:
            ret = "The number of intervals must be positive";
            break;
        case TSD_ERR_NEGATIVE_CHANGEPOINT_PARAMETER:
            ret = "The penalty, minimum counts and minimum offset must not be negative";
            break;
        case TSD_ERR_ZERO_EDGE_SPAN:
            ret = "Zero edge span in interval";
            break;
        case TSD_ERR_NON_INCREASING_BREAKPOINTS:
            ret = "Rescaling produced non-increasing time breakpoints. "
                  "Try setting `rescaling_intervals` to a smaller value "
                  "(e.g. half its current value).";
            break;
        case TSD_ERR_INVALID_QUANTILE_WIDTH:
            ret = "The quantile width must be greater than zero and less than one";
            break;
        case TSD_ERR_INVALID_POSTERIOR:
            ret = "A posterior to be rescaled has shape not greater than -1 or rate "
                  "not greater than zero";
            break;
        case TSD_ERR_TIME_BELOW_FIRST_BREAKPOINT:
            ret = "A time to be rescaled is below the first time breakpoint";
            break;
        case TSD_ERR_NO_POSTERIORS:
            ret = "zero-size array to reduction operation minimum which has no "
                  "identity";
            break;
        case TSD_ERR_INVALID_MAX_SHAPE:
            ret = "The maximum shape must be at least one";
            break;
        case TSD_ERR_INVALID_MIN_STEP:
            ret = "The minimum step size must be greater than zero and less than one";
            break;
        case TSD_ERR_INVALID_NODE_POSTERIOR:
            ret = "A node posterior has shape not greater than -1 or rate not greater "
                  "than zero";
            break;
        case TSD_ERR_INVALID_DAMPING:
            ret = "The damping factor is not greater than zero and at most one";
            break;
        case TSD_ERR_INCONSISTENT_CONSTRAINTS:
            ret = "Node age constraints are inconsistent, some descendants have"
                  " lower bounds that exceed upper bounds of their ancestors.";
            break;
        case TSD_ERR_TWIN_EDGE_IN_PHASED_PASS:
            ret = "An edge with the same parent and child can only be updated in an "
                  "unphased pass";
            break;
        case TSD_ERR_NONPOSITIVE_PENALTY:
            ret = "The fitted regularisation penalty is not positive";
            break;
        default:
            break;
    }
    return ret;
}

void
_tsd_safe_free(void **ptr)
{
    if (ptr != NULL) {
        if (*ptr != NULL) {
            free(*ptr);
            *ptr = NULL;
        }
    }
}
