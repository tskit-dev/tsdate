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

#ifdef __cplusplus
extern "C" {
#endif

#include <stdio.h>
#include <stdlib.h>

/* clang-format off */
#define TSD_ERR_GENERIC                                             -1
#define TSD_ERR_NO_MEMORY                                           -2
#define TSD_ERR_BAD_PARAM                                           -3

/* Floating-point division by zero, where numba raises ZeroDivisionError */
#define TSD_ERR_ZERO_DIVISION                                       -4

/* hypergeo */
#define TSD_ERR_INVALID_HYPERGEO                                    -5
/* clang-format on */

#ifdef __GNUC__
#define WARN_UNUSED    __attribute__((warn_unused_result))
#define unlikely(expr) __builtin_expect(!!(expr), 0)
#define likely(expr)   __builtin_expect(!!(expr), 1)
#else
#define WARN_UNUSED
#define unlikely(expr) (expr)
#define likely(expr)   (expr)
#endif

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

#ifdef __cplusplus
}
#endif

#endif
