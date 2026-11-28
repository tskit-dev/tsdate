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

#ifndef TSD_MATHUTIL_H
#define TSD_MATHUTIL_H

#include <math.h>

/* Constants that <math.h> only defines as extensions (and MSVC not by default). */
#define TSD_PI          3.141592653589793
#define TSD_EULER_GAMMA 0.5772156649015329

/*
 * x ** n for a small positive integer constant n, by repeated squaring (the
 * multiplication order of LLVM's llvm.powi, and of the oracles' `ipow`); ipow(x, 2) is
 * exactly x * x. It can differ in the last bit from pow(x, n).
 */
static inline double
tsd_ipow(double x, int n)
{
    double result = 1.0;

    while (1) {
        if (n & 1) {
            result *= x;
        }
        n >>= 1;
        if (n == 0) {
            return result;
        }
        x *= x;
    }
}

/* Python's min(a, b): b if b < a else a (so NaN handling differs from fmin). */
static inline double
tsd_min(double a, double b)
{
    return b < a ? b : a;
}

/* Python's max(a, b): b if b > a else a (so NaN handling differs from fmax). */
static inline double
tsd_max(double a, double b)
{
    return b > a ? b : a;
}

/* np.isclose(a, b) for finite or NaN arguments, with default rtol and atol. */
static inline int
tsd_isclose(double a, double b)
{
    return fabs(a - b) <= 1e-8 + 1e-5 * fabs(b);
}

#endif
