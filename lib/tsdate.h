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

#ifdef __cplusplus
}
#endif

#endif
