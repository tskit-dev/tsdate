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
