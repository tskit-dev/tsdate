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

#define PY_SSIZE_T_CLEAN
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <Python.h>
#include <string.h>
#include <numpy/arrayobject.h>

#include "lib/tsdate.h"

#define MODULE_DOC "Low-level tsdate interface."

static PyObject *TsdLibraryError;

/*
 * SciPy's gammaincinv, taken from the C API of scipy.special.cython_special at module
 * initialisation (as numba did). The third argument is Cython's __pyx_skip_dispatch.
 */
typedef double (*cython_gammaincinv_fn)(double a, double p, int skip_dispatch);
static cython_gammaincinv_fn scipy_gammaincinv;

static void
handle_library_error(int err)
{
    if (err == TSD_ERR_NO_MEMORY) {
        PyErr_NoMemory();
    } else {
        PyErr_Format(TsdLibraryError, "%s", tsd_strerror(err));
    }
}

/* Return the library's message for the given error code. */
static PyObject *
tsdate_strerror(PyObject *Py_UNUSED(self), PyObject *args)
{
    PyObject *ret = NULL;
    int err;

    if (!PyArg_ParseTuple(args, "i", &err)) {
        goto out;
    }
    ret = Py_BuildValue("s", tsd_strerror(err));
out:
    return ret;
}

/* Raise the exception that the given library error code maps to. */
static PyObject *
tsdate_raise_library_error(PyObject *Py_UNUSED(self), PyObject *args)
{
    PyObject *ret = NULL;
    int err;

    if (!PyArg_ParseTuple(args, "i", &err)) {
        goto out;
    }
    handle_library_error(err);
out:
    return ret;
}

/*
 * Scalar functions: "N doubles in, M doubles out" functions of the C library are
 * exposed to Python through the table below, with one generic wrapper. To expose a
 * new function, write an adapter that unpacks `in` into the C call and stores the
 * results in `out`, returning 0 or a library error code, and add a row.
 * The Python function takes `num_in` numbers and returns a float (one output) or a
 * tuple of floats.
 */
#define MAX_SCALAR_IO 16

typedef int (*scalar_adapter_t)(const double *in, double *out);

typedef struct {
    const char *name;
    int num_in;
    int num_out;
    scalar_adapter_t adapter;
} scalar_function_t;

static int
adapt_digamma(const double *in, double *out)
{
    return tsd_digamma(in[0], out);
}

static int
adapt_trigamma(const double *in, double *out)
{
    return tsd_trigamma(in[0], out);
}

static int
adapt_betaln(const double *in, double *out)
{
    out[0] = tsd_betaln(in[0], in[1]);
    return 0;
}

static int
adapt_gammainc_der(const double *in, double *out)
{
    return tsd_gammainc_der(in[0], in[1], out);
}

/* A tsd_gammaincinv_fn: this is what is passed to library functions that need it. */
static double
gammaincinv(double a, double p)
{
    return scipy_gammaincinv(a, p, 0);
}

static int
adapt_gammainc_inv(const double *in, double *out)
{
    out[0] = gammaincinv(in[0], in[1]);
    return 0;
}

static int
adapt_hyperu_laplace(const double *in, double *out)
{
    return tsd_hyperu_laplace(in[0], in[1], in[2], out, out + 1);
}

static int
adapt_hyp1f1_laplace(const double *in, double *out)
{
    return tsd_hyp1f1_laplace(in[0], in[1], in[2], out);
}

static int
adapt_hyp2f1_laplace(const double *in, double *out)
{
    return tsd_hyp2f1_laplace(in[0], in[1], in[2], in[3], out);
}

/* approx: helpers */

static int
adapt_approximate_log_moments(const double *in, double *out)
{
    return tsd_approximate_log_moments(in[0], in[1], out, out + 1, out + 2);
}

static int
adapt_approximate_gamma_kl(const double *in, double *out)
{
    return tsd_approximate_gamma_kl(in[0], in[1], out);
}

static int
adapt_approximate_gamma_mom(const double *in, double *out)
{
    return tsd_approximate_gamma_mom(in[0], in[1], out);
}

static int
adapt_approximate_gamma_iqr(const double *in, double *out)
{
    return tsd_approximate_gamma_iqr(
        in[0], in[1], in[2], in[3], in[4], gammaincinv, out);
}

static int
adapt_valid_moments(const double *in, double *out)
{
    out[0] = tsd_valid_moments(in[0], in[1]);
    return 0;
}

static int
adapt_valid_gamma(const double *in, double *out)
{
    out[0] = tsd_valid_gamma(in[0], in[1]);
    return 0;
}

static int
adapt_valid_hyp1f1(const double *in, double *out)
{
    out[0] = tsd_valid_hyp1f1(in[0], in[1], in[2]);
    return 0;
}

static int
adapt_valid_hyperu(const double *in, double *out)
{
    out[0] = tsd_valid_hyperu(in[0], in[1], in[2]);
    return 0;
}

static int
adapt_valid_hyp2f1(const double *in, double *out)
{
    out[0] = tsd_valid_hyp2f1(in[0], in[1], in[2], in[3]);
    return 0;
}

/*
 * Adapters of the moment and projection functions. Inputs are flat: ages and shape/rate
 * scalars, then the natural parameters `pars_i`, `pars_j`, `pars_ij` (two each) as
 * consecutive doubles. Outputs are flat: `logl` or `pr_m`, then the pairs `proj_i`,
 * `proj_j` or `proj_m`.
 */

/* approx: node moments */
static int
adapt_moments(const double *in, double *out)
{
    return tsd_moments(in[0], in[1], in[2], in[3], in[4], in[5], out, out + 1, out + 2,
        out + 3, out + 4);
}

static int
adapt_rootward_moments(const double *in, double *out)
{
    return tsd_rootward_moments(
        in[0], in[1], in[2], in[3], in[4], out, out + 1, out + 2);
}

static int
adapt_leafward_moments(const double *in, double *out)
{
    return tsd_leafward_moments(
        in[0], in[1], in[2], in[3], in[4], out, out + 1, out + 2);
}

static int
adapt_unphased_moments(const double *in, double *out)
{
    return tsd_unphased_moments(in[0], in[1], in[2], in[3], in[4], in[5], out, out + 1,
        out + 2, out + 3, out + 4);
}

static int
adapt_twin_moments(const double *in, double *out)
{
    return tsd_twin_moments(in[0], in[1], in[2], in[3], out, out + 1, out + 2);
}

static int
adapt_sideways_moments(const double *in, double *out)
{
    return tsd_sideways_moments(
        in[0], in[1], in[2], in[3], in[4], out, out + 1, out + 2);
}

/* approx: node projections */
static int
adapt_gamma_projection(const double *in, double *out)
{
    return tsd_gamma_projection(in, in + 2, in + 4, out, out + 1, out + 3);
}

static int
adapt_leafward_projection(const double *in, double *out)
{
    return tsd_leafward_projection(in[0], in + 1, in + 3, out, out + 1);
}

static int
adapt_rootward_projection(const double *in, double *out)
{
    return tsd_rootward_projection(in[0], in + 1, in + 3, out, out + 1);
}

static int
adapt_unphased_projection(const double *in, double *out)
{
    return tsd_unphased_projection(in, in + 2, in + 4, out, out + 1, out + 3);
}

static int
adapt_twin_projection(const double *in, double *out)
{
    return tsd_twin_projection(in, in + 2, out, out + 1);
}

static int
adapt_sideways_projection(const double *in, double *out)
{
    return tsd_sideways_projection(in[0], in + 1, in + 3, out, out + 1);
}

/* approx: mutation moments */
static int
adapt_mutation_moments(const double *in, double *out)
{
    return tsd_mutation_moments(in[0], in[1], in[2], in[3], in[4], in[5], out, out + 1);
}

static int
adapt_mutation_rootward_moments(const double *in, double *out)
{
    return tsd_mutation_rootward_moments(
        in[0], in[1], in[2], in[3], in[4], out, out + 1);
}

static int
adapt_mutation_leafward_moments(const double *in, double *out)
{
    return tsd_mutation_leafward_moments(
        in[0], in[1], in[2], in[3], in[4], out, out + 1);
}

static int
adapt_mutation_unphased_moments(const double *in, double *out)
{
    return tsd_mutation_unphased_moments(
        in[0], in[1], in[2], in[3], in[4], in[5], out, out + 1, out + 2);
}

static int
adapt_mutation_twin_moments(const double *in, double *out)
{
    return tsd_mutation_twin_moments(in[0], in[1], in[2], in[3], out, out + 1, out + 2);
}

static int
adapt_mutation_sideways_moments(const double *in, double *out)
{
    return tsd_mutation_sideways_moments(
        in[0], in[1], in[2], in[3], in[4], out, out + 1, out + 2);
}

static int
adapt_mutation_edge_moments(const double *in, double *out)
{
    return tsd_mutation_edge_moments(in[0], in[1], out, out + 1);
}

static int
adapt_mutation_block_moments(const double *in, double *out)
{
    return tsd_mutation_block_moments(in[0], in[1], out, out + 1, out + 2);
}

/* approx: mutation projections */
static int
adapt_mutation_gamma_projection(const double *in, double *out)
{
    return tsd_mutation_gamma_projection(in, in + 2, in + 4, out, out + 1);
}

static int
adapt_mutation_leafward_projection(const double *in, double *out)
{
    return tsd_mutation_leafward_projection(in[0], in + 1, in + 3, out, out + 1);
}

static int
adapt_mutation_rootward_projection(const double *in, double *out)
{
    return tsd_mutation_rootward_projection(in[0], in + 1, in + 3, out, out + 1);
}

static int
adapt_mutation_edge_projection(const double *in, double *out)
{
    return tsd_mutation_edge_projection(in[0], in[1], out, out + 1);
}

static int
adapt_mutation_unphased_projection(const double *in, double *out)
{
    return tsd_mutation_unphased_projection(in, in + 2, in + 4, out, out + 1);
}

static int
adapt_mutation_twin_projection(const double *in, double *out)
{
    return tsd_mutation_twin_projection(in, in + 2, out, out + 1);
}

static int
adapt_mutation_sideways_projection(const double *in, double *out)
{
    return tsd_mutation_sideways_projection(in[0], in + 1, in + 3, out, out + 1);
}

static int
adapt_mutation_block_projection(const double *in, double *out)
{
    return tsd_mutation_block_projection(in[0], in[1], out, out + 1);
}

/* clang-format off */
static const scalar_function_t scalar_functions[] = {
    /* hypergeo */
    { "digamma", 1, 1, adapt_digamma },
    { "trigamma", 1, 1, adapt_trigamma },
    { "betaln", 2, 1, adapt_betaln },
    { "gammainc_der", 2, 1, adapt_gammainc_der },
    { "gammainc_inv", 2, 1, adapt_gammainc_inv },
    { "hyperu_laplace", 3, 2, adapt_hyperu_laplace },
    { "hyp1f1_laplace", 3, 1, adapt_hyp1f1_laplace },
    { "hyp2f1_laplace", 4, 1, adapt_hyp2f1_laplace },
    /* approx: helpers */
    { "approximate_log_moments", 2, 3, adapt_approximate_log_moments },
    { "approximate_gamma_kl", 2, 2, adapt_approximate_gamma_kl },
    { "approximate_gamma_mom", 2, 2, adapt_approximate_gamma_mom },
    { "approximate_gamma_iqr", 5, 2, adapt_approximate_gamma_iqr },
    { "valid_moments", 2, 1, adapt_valid_moments },
    { "valid_gamma", 2, 1, adapt_valid_gamma },
    { "valid_hyp1f1", 3, 1, adapt_valid_hyp1f1 },
    { "valid_hyperu", 3, 1, adapt_valid_hyperu },
    { "valid_hyp2f1", 4, 1, adapt_valid_hyp2f1 },
    /* approx: node moments */
    { "moments", 6, 5, adapt_moments },
    { "rootward_moments", 5, 3, adapt_rootward_moments },
    { "leafward_moments", 5, 3, adapt_leafward_moments },
    { "unphased_moments", 6, 5, adapt_unphased_moments },
    { "twin_moments", 4, 3, adapt_twin_moments },
    { "sideways_moments", 5, 3, adapt_sideways_moments },
    /* approx: node projections */
    { "gamma_projection", 6, 5, adapt_gamma_projection },
    { "leafward_projection", 5, 3, adapt_leafward_projection },
    { "rootward_projection", 5, 3, adapt_rootward_projection },
    { "unphased_projection", 6, 5, adapt_unphased_projection },
    { "twin_projection", 4, 3, adapt_twin_projection },
    { "sideways_projection", 5, 3, adapt_sideways_projection },
    /* approx: mutation moments */
    { "mutation_moments", 6, 2, adapt_mutation_moments },
    { "mutation_rootward_moments", 5, 2, adapt_mutation_rootward_moments },
    { "mutation_leafward_moments", 5, 2, adapt_mutation_leafward_moments },
    { "mutation_unphased_moments", 6, 3, adapt_mutation_unphased_moments },
    { "mutation_twin_moments", 4, 3, adapt_mutation_twin_moments },
    { "mutation_sideways_moments", 5, 3, adapt_mutation_sideways_moments },
    { "mutation_edge_moments", 2, 2, adapt_mutation_edge_moments },
    { "mutation_block_moments", 2, 3, adapt_mutation_block_moments },
    /* approx: mutation projections */
    { "mutation_gamma_projection", 6, 3, adapt_mutation_gamma_projection },
    { "mutation_leafward_projection", 5, 3, adapt_mutation_leafward_projection },
    { "mutation_rootward_projection", 5, 3, adapt_mutation_rootward_projection },
    { "mutation_edge_projection", 2, 3, adapt_mutation_edge_projection },
    { "mutation_unphased_projection", 6, 3, adapt_mutation_unphased_projection },
    { "mutation_twin_projection", 4, 3, adapt_mutation_twin_projection },
    { "mutation_sideways_projection", 5, 3, adapt_mutation_sideways_projection },
    { "mutation_block_projection", 2, 3, adapt_mutation_block_projection },
};
/* clang-format on */

#define NUM_SCALAR_FUNCTIONS (sizeof(scalar_functions) / sizeof(scalar_functions[0]))

/* Method definitions of the scalar functions, filled in at module initialisation. */
static PyMethodDef scalar_method_defs[NUM_SCALAR_FUNCTIONS];

/* Parse `num_in` numbers from the argument tuple. Returns 0, or -1 with an exception. */
static int
scalar_parse_args(const scalar_function_t *row, PyObject *args, double *in)
{
    int ret = -1;
    Py_ssize_t j;
    Py_ssize_t num_args = PyTuple_GET_SIZE(args);

    if (num_args != row->num_in) {
        PyErr_Format(PyExc_TypeError, "%s() takes exactly %d arguments (%zd given)",
            row->name, row->num_in, num_args);
        goto out;
    }
    for (j = 0; j < num_args; j++) {
        in[j] = PyFloat_AsDouble(PyTuple_GET_ITEM(args, j));
        if (in[j] == -1.0 && PyErr_Occurred()) {
            goto out;
        }
    }
    ret = 0;
out:
    return ret;
}

/* Build the return value: a float for one output, otherwise a tuple of floats. */
static PyObject *
scalar_build_result(const scalar_function_t *row, const double *out)
{
    PyObject *ret = NULL;
    PyObject *item = NULL;
    PyObject *tuple = NULL;
    int j;

    if (row->num_out == 1) {
        ret = PyFloat_FromDouble(out[0]);
        goto out;
    }
    tuple = PyTuple_New(row->num_out);
    if (tuple == NULL) {
        goto out;
    }
    for (j = 0; j < row->num_out; j++) {
        item = PyFloat_FromDouble(out[j]);
        if (item == NULL) {
            goto out;
        }
        PyTuple_SET_ITEM(tuple, j, item); /* steals the reference */
    }
    ret = tuple;
    tuple = NULL;
out:
    Py_XDECREF(tuple);
    return ret;
}

/* The generic wrapper; `self` is the index of the row in the table. */
static PyObject *
scalar_call(PyObject *self, PyObject *args)
{
    PyObject *ret = NULL;
    const scalar_function_t *row;
    double in[MAX_SCALAR_IO];
    double out[MAX_SCALAR_IO];
    Py_ssize_t index = PyLong_AsSsize_t(self);
    int err;

    row = &scalar_functions[index];
    if (scalar_parse_args(row, args, in) != 0) {
        goto out;
    }
    err = row->adapter(in, out);
    if (err != 0) {
        handle_library_error(err);
        goto out;
    }
    ret = scalar_build_result(row, out);
out:
    return ret;
}

/* Add one Python function per row of the table to the module. */
static int
add_scalar_functions(PyObject *module)
{
    int ret = -1;
    PyObject *index = NULL;
    PyObject *func = NULL;
    size_t j;

    for (j = 0; j < NUM_SCALAR_FUNCTIONS; j++) {
        scalar_method_defs[j].ml_name = scalar_functions[j].name;
        scalar_method_defs[j].ml_meth = (PyCFunction) scalar_call;
        scalar_method_defs[j].ml_flags = METH_VARARGS;
        scalar_method_defs[j].ml_doc = "Scalar function of the C library.";
        index = PyLong_FromSize_t(j);
        if (index == NULL) {
            goto out;
        }
        func = PyCFunction_New(&scalar_method_defs[j], index);
        Py_DECREF(index);
        index = NULL;
        if (func == NULL) {
            goto out;
        }
        if (PyModule_AddObject(module, scalar_functions[j].name, func) != 0) {
            goto out;
        }
        func = NULL; /* the module owns the reference now */
    }
    ret = 0;
out:
    Py_XDECREF(func);
    return ret;
}

/* average_gammas(alpha, beta): natural parameters of the averaged gamma. */
static PyObject *
tsdate_average_gammas(PyObject *Py_UNUSED(self), PyObject *args)
{
    PyObject *ret = NULL;
    PyObject *alpha_obj = NULL;
    PyObject *beta_obj = NULL;
    PyArrayObject *alpha_array = NULL;
    PyArrayObject *beta_array = NULL;
    double out[2];
    npy_intp num_alpha, num_beta;
    int err;

    if (!PyArg_ParseTuple(args, "OO", &alpha_obj, &beta_obj)) {
        goto out;
    }
    alpha_array = (PyArrayObject *) PyArray_FROMANY(
        alpha_obj, NPY_FLOAT64, 1, 1, NPY_ARRAY_IN_ARRAY);
    if (alpha_array == NULL) {
        goto out;
    }
    beta_array = (PyArrayObject *) PyArray_FROMANY(
        beta_obj, NPY_FLOAT64, 1, 1, NPY_ARRAY_IN_ARRAY);
    if (beta_array == NULL) {
        goto out;
    }
    num_alpha = PyArray_DIM(alpha_array, 0);
    num_beta = PyArray_DIM(beta_array, 0);
    if (num_alpha != num_beta) {
        PyErr_SetString(PyExc_ValueError, "Array sizes are not equal");
        goto out;
    }
    Py_BEGIN_ALLOW_THREADS
    err = tsd_average_gammas((tsd_size_t) num_alpha,
        (const double *) PyArray_DATA(alpha_array),
        (const double *) PyArray_DATA(beta_array), out);
    Py_END_ALLOW_THREADS
    if (err != 0) {
        handle_library_error(err);
        goto out;
    }
    ret = Py_BuildValue("dd", out[0], out[1]);
out:
    Py_XDECREF(alpha_array);
    Py_XDECREF(beta_array);
    return ret;
}

/*
 * Phasing, utilities and prior: array-level functions. Each takes the arguments of the
 * numba function it replaces. Size checks that numba asserted raise ValueError.
 */

#define MAX_ARRAYS 10

/* Convert `num_arrays` objects to C-contiguous arrays of the given types and ndim. */
static int
parse_arrays(PyObject **objs, const int *typenums, int ndim, int num_arrays,
    PyArrayObject **arrays)
{
    int j;

    for (j = 0; j < num_arrays; j++) {
        arrays[j] = (PyArrayObject *) PyArray_FROMANY(
            objs[j], typenums[j], ndim, ndim, NPY_ARRAY_IN_ARRAY);
        if (arrays[j] == NULL) {
            return -1;
        }
    }
    return 0;
}

static void
release_arrays(PyArrayObject **arrays, int num_arrays)
{
    int j;

    for (j = 0; j < num_arrays; j++) {
        Py_XDECREF(arrays[j]);
    }
}

#define ARRAY_SIZE(array)      ((tsd_size_t) PyArray_SIZE(array))
#define ARRAY_DIM(array, axis) ((tsd_size_t) PyArray_DIM(array, axis))

/* Whether all the arrays have the same size; raises ValueError if not. */
static int
check_sizes_equal(PyArrayObject **arrays, int num_arrays)
{
    int j;

    for (j = 1; j < num_arrays; j++) {
        if (ARRAY_SIZE(arrays[j]) != ARRAY_SIZE(arrays[0])) {
            PyErr_SetString(PyExc_ValueError, "Array sizes are not equal");
            return -1;
        }
    }
    return 0;
}

/* A new array of the given type and shape, copying `size` bytes from `data`. */
static PyObject *
array_from_data(int typenum, int ndim, npy_intp *dims, const void *data, size_t size)
{
    PyObject *ret = PyArray_SimpleNew(ndim, dims, typenum);

    if (ret != NULL && size > 0) {
        memcpy(PyArray_DATA((PyArrayObject *) ret), data, size);
    }
    return ret;
}

/*
 * block_singletons(individuals_unphased, nodes_individual, mutations_node,
 *     mutations_position, edges_parent, edges_child, edges_left, edges_right,
 *     indexes_insert, indexes_remove, sequence_length)
 * returns (blocks_stats, blocks_edges, mutations_block).
 */
static PyObject *
tsdate_block_singletons(PyObject *Py_UNUSED(self), PyObject *args)
{
    PyObject *ret = NULL;
    PyObject *objs[10];
    const int typenums[10] = { NPY_BOOL, NPY_INT32, NPY_INT32, NPY_FLOAT64, NPY_INT32,
        NPY_INT32, NPY_FLOAT64, NPY_FLOAT64, NPY_INT32, NPY_INT32 };
    PyArrayObject *arrays[10] = { NULL };
    PyObject *stats_array = NULL;
    PyObject *edges_array = NULL;
    PyObject *mutations_block = NULL;
    double *blocks_stats = NULL;
    tsd_id_t *blocks_edges = NULL;
    tsd_size_t num_blocks = 0;
    npy_intp dims[2];
    double sequence_length;
    int err;

    if (!PyArg_ParseTuple(args, "OOOOOOOOOOd", &objs[0], &objs[1], &objs[2], &objs[3],
            &objs[4], &objs[5], &objs[6], &objs[7], &objs[8], &objs[9],
            &sequence_length)) {
        goto out;
    }
    if (parse_arrays(objs, typenums, 1, 10, arrays) != 0) {
        goto out;
    }
    if (check_sizes_equal(arrays + 4, 4) != 0 || check_sizes_equal(arrays + 8, 2) != 0
        || check_sizes_equal(arrays + 2, 2) != 0
        || ARRAY_SIZE(arrays[8]) != ARRAY_SIZE(arrays[4])) {
        PyErr_SetString(PyExc_ValueError, "Array sizes are not equal");
        goto out;
    }
    dims[0] = PyArray_DIM(arrays[2], 0);
    mutations_block = PyArray_SimpleNew(1, dims, NPY_INT32);
    if (mutations_block == NULL) {
        goto out;
    }
    Py_BEGIN_ALLOW_THREADS
    err = tsd_block_singletons(ARRAY_SIZE(arrays[0]), ARRAY_SIZE(arrays[4]),
        ARRAY_SIZE(arrays[2]), (const uint8_t *) PyArray_DATA(arrays[0]),
        (const tsd_id_t *) PyArray_DATA(arrays[1]),
        (const tsd_id_t *) PyArray_DATA(arrays[2]),
        (const double *) PyArray_DATA(arrays[3]),
        (const tsd_id_t *) PyArray_DATA(arrays[5]),
        (const double *) PyArray_DATA(arrays[6]),
        (const double *) PyArray_DATA(arrays[7]),
        (const tsd_id_t *) PyArray_DATA(arrays[8]),
        (const tsd_id_t *) PyArray_DATA(arrays[9]), sequence_length, &num_blocks,
        &blocks_stats, &blocks_edges,
        (tsd_id_t *) PyArray_DATA((PyArrayObject *) mutations_block));
    Py_END_ALLOW_THREADS
    if (err != 0) {
        handle_library_error(err);
        goto out;
    }
    dims[0] = (npy_intp) num_blocks;
    dims[1] = 2;
    stats_array = array_from_data(
        NPY_FLOAT64, 2, dims, blocks_stats, 2 * num_blocks * sizeof(*blocks_stats));
    edges_array = array_from_data(
        NPY_INT32, 2, dims, blocks_edges, 2 * num_blocks * sizeof(*blocks_edges));
    if (stats_array == NULL || edges_array == NULL) {
        goto out;
    }
    ret = Py_BuildValue("OOO", stats_array, edges_array, mutations_block);
out:
    release_arrays(arrays, 10);
    Py_XDECREF(stats_array);
    Py_XDECREF(edges_array);
    Py_XDECREF(mutations_block);
    tsd_safe_free(blocks_stats);
    tsd_safe_free(blocks_edges);
    return ret;
}

/*
 * reallocate_unphased(edges_likelihood, mutations_phase, mutations_block,
 *     blocks_edges): updates edges_likelihood in place.
 */
static PyObject *
tsdate_reallocate_unphased(PyObject *Py_UNUSED(self), PyObject *args)
{
    PyObject *ret = NULL;
    PyObject *likelihood_obj = NULL;
    PyObject *phase_obj = NULL;
    PyObject *block_obj = NULL;
    PyObject *edges_obj = NULL;
    PyArrayObject *likelihood_array = NULL;
    PyArrayObject *phase_array = NULL;
    PyArrayObject *block_array = NULL;
    PyArrayObject *edges_array = NULL;
    int err;

    if (!PyArg_ParseTuple(
            args, "OOOO", &likelihood_obj, &phase_obj, &block_obj, &edges_obj)) {
        goto out;
    }
    likelihood_array = (PyArrayObject *) PyArray_FROMANY(
        likelihood_obj, NPY_FLOAT64, 2, 2, NPY_ARRAY_INOUT_ARRAY);
    phase_array = (PyArrayObject *) PyArray_FROMANY(
        phase_obj, NPY_FLOAT64, 1, 1, NPY_ARRAY_IN_ARRAY);
    block_array = (PyArrayObject *) PyArray_FROMANY(
        block_obj, NPY_INT32, 1, 1, NPY_ARRAY_IN_ARRAY);
    edges_array = (PyArrayObject *) PyArray_FROMANY(
        edges_obj, NPY_INT32, 2, 2, NPY_ARRAY_IN_ARRAY);
    if (likelihood_array == NULL || phase_array == NULL || block_array == NULL
        || edges_array == NULL) {
        goto out;
    }
    if (ARRAY_SIZE(phase_array) != ARRAY_SIZE(block_array)) {
        PyErr_SetString(PyExc_ValueError, "Array sizes are not equal");
        goto out;
    }
    if (ARRAY_DIM(edges_array, 1) != 2 || ARRAY_DIM(likelihood_array, 1) != 2) {
        PyErr_SetString(PyExc_ValueError, "Arrays must have two columns");
        goto out;
    }
    Py_BEGIN_ALLOW_THREADS
    err = tsd_reallocate_unphased(ARRAY_DIM(likelihood_array, 0),
        ARRAY_SIZE(phase_array), ARRAY_DIM(edges_array, 0),
        (const double *) PyArray_DATA(phase_array),
        (const tsd_id_t *) PyArray_DATA(block_array),
        (const tsd_id_t *) PyArray_DATA(edges_array),
        (double *) PyArray_DATA(likelihood_array));
    Py_END_ALLOW_THREADS
    if (err != 0) {
        handle_library_error(err);
        goto out;
    }
    ret = Py_None;
    Py_INCREF(ret);
out:
    Py_XDECREF(likelihood_array);
    Py_XDECREF(phase_array);
    Py_XDECREF(block_array);
    Py_XDECREF(edges_array);
    return ret;
}

/*
 * mutation_frequency(nodes_sample, mutations_node, mutations_position, edges_parent,
 *     edges_child, edges_left, edges_right, indexes_insert, indexes_remove,
 *     sequence_length) returns an int32 array of shape (mutations, sample sets).
 */
static PyObject *
tsdate_mutation_frequency(PyObject *Py_UNUSED(self), PyObject *args)
{
    PyObject *ret = NULL;
    PyObject *sample_obj = NULL;
    PyObject *objs[MAX_ARRAYS];
    const int typenums[9] = { NPY_INT32, NPY_FLOAT64, NPY_INT32, NPY_INT32, NPY_FLOAT64,
        NPY_FLOAT64, NPY_INT32, NPY_INT32 };
    PyArrayObject *sample_array = NULL;
    PyArrayObject *arrays[8] = { NULL };
    PyObject *freq_array = NULL;
    npy_intp dims[2];
    double sequence_length;
    int err;

    if (!PyArg_ParseTuple(args, "OOOOOOOOOd", &sample_obj, &objs[0], &objs[1], &objs[2],
            &objs[3], &objs[4], &objs[5], &objs[6], &objs[7], &sequence_length)) {
        goto out;
    }
    sample_array = (PyArrayObject *) PyArray_FROMANY(
        sample_obj, NPY_BOOL, 2, 2, NPY_ARRAY_IN_ARRAY);
    if (sample_array == NULL) {
        goto out;
    }
    if (parse_arrays(objs, typenums, 1, 8, arrays) != 0) {
        goto out;
    }
    if (check_sizes_equal(arrays + 2, 4) != 0 || check_sizes_equal(arrays + 6, 2) != 0
        || check_sizes_equal(arrays, 2) != 0
        || ARRAY_SIZE(arrays[6]) != ARRAY_SIZE(arrays[2])) {
        PyErr_SetString(PyExc_ValueError, "Array sizes are not equal");
        goto out;
    }
    dims[0] = PyArray_DIM(arrays[0], 0);
    dims[1] = PyArray_DIM(sample_array, 1);
    freq_array = PyArray_SimpleNew(2, dims, NPY_INT32);
    if (freq_array == NULL) {
        goto out;
    }
    Py_BEGIN_ALLOW_THREADS
    err = tsd_mutation_frequency(ARRAY_DIM(sample_array, 0), ARRAY_DIM(sample_array, 1),
        ARRAY_SIZE(arrays[2]), ARRAY_SIZE(arrays[0]),
        (const uint8_t *) PyArray_DATA(sample_array),
        (const tsd_id_t *) PyArray_DATA(arrays[0]),
        (const double *) PyArray_DATA(arrays[1]),
        (const tsd_id_t *) PyArray_DATA(arrays[2]),
        (const tsd_id_t *) PyArray_DATA(arrays[3]),
        (const double *) PyArray_DATA(arrays[4]),
        (const double *) PyArray_DATA(arrays[5]),
        (const tsd_id_t *) PyArray_DATA(arrays[6]),
        (const tsd_id_t *) PyArray_DATA(arrays[7]), sequence_length,
        (int32_t *) PyArray_DATA((PyArrayObject *) freq_array));
    Py_END_ALLOW_THREADS
    if (err != 0) {
        handle_library_error(err);
        goto out;
    }
    ret = freq_array;
    freq_array = NULL;
out:
    Py_XDECREF(sample_array);
    release_arrays(arrays, 8);
    Py_XDECREF(freq_array);
    return ret;
}

/*
 * split_disjoint_nodes(edges_parent, edges_child, edges_left, edges_right,
 *     node_excluded) returns (edges_parent, edges_child, nodes_order, split_nodes).
 */
static PyObject *
tsdate_split_disjoint_nodes(PyObject *Py_UNUSED(self), PyObject *args)
{
    PyObject *ret = NULL;
    PyObject *objs[5];
    const int typenums[5] = { NPY_INT32, NPY_INT32, NPY_FLOAT64, NPY_FLOAT64, NPY_BOOL };
    PyArrayObject *arrays[5] = { NULL };
    PyObject *parent_array = NULL;
    PyObject *child_array = NULL;
    PyObject *order_array = NULL;
    PyObject *split_array = NULL;
    tsd_id_t *nodes_order = NULL;
    tsd_size_t num_nodes, num_split = 0;
    npy_intp dims[1];
    int err;

    if (!PyArg_ParseTuple(
            args, "OOOOO", &objs[0], &objs[1], &objs[2], &objs[3], &objs[4])) {
        goto out;
    }
    if (parse_arrays(objs, typenums, 1, 5, arrays) != 0) {
        goto out;
    }
    if (check_sizes_equal(arrays, 4) != 0) {
        goto out;
    }
    num_nodes = ARRAY_SIZE(arrays[4]);
    dims[0] = PyArray_DIM(arrays[0], 0);
    parent_array = PyArray_SimpleNew(1, dims, NPY_INT32);
    child_array = PyArray_SimpleNew(1, dims, NPY_INT32);
    if (parent_array == NULL || child_array == NULL) {
        goto out;
    }
    Py_BEGIN_ALLOW_THREADS
    err = tsd_split_disjoint_nodes(num_nodes, ARRAY_SIZE(arrays[0]),
        (const tsd_id_t *) PyArray_DATA(arrays[0]),
        (const tsd_id_t *) PyArray_DATA(arrays[1]),
        (const double *) PyArray_DATA(arrays[2]),
        (const double *) PyArray_DATA(arrays[3]),
        (const uint8_t *) PyArray_DATA(arrays[4]),
        (tsd_id_t *) PyArray_DATA((PyArrayObject *) parent_array),
        (tsd_id_t *) PyArray_DATA((PyArrayObject *) child_array), &num_split,
        &nodes_order);
    Py_END_ALLOW_THREADS
    if (err != 0) {
        handle_library_error(err);
        goto out;
    }
    dims[0] = (npy_intp) (num_nodes + num_split);
    order_array = array_from_data(
        NPY_INT32, 1, dims, nodes_order, (num_nodes + num_split) * sizeof(*nodes_order));
    dims[0] = (npy_intp) num_split;
    split_array = array_from_data(
        NPY_INT32, 1, dims, nodes_order + num_nodes, num_split * sizeof(*nodes_order));
    if (order_array == NULL || split_array == NULL) {
        goto out;
    }
    ret = Py_BuildValue("OOOO", parent_array, child_array, order_array, split_array);
out:
    release_arrays(arrays, 5);
    Py_XDECREF(parent_array);
    Py_XDECREF(child_array);
    Py_XDECREF(order_array);
    Py_XDECREF(split_array);
    tsd_safe_free(nodes_order);
    return ret;
}

/*
 * relabel_mutations_node(mutations_node, mutations_position, nodes_order,
 *     edges_parent, edges_child, edges_left, edges_right, insert_index,
 *     remove_index) returns the new node of each mutation.
 */
static PyObject *
tsdate_relabel_mutations_node(PyObject *Py_UNUSED(self), PyObject *args)
{
    PyObject *ret = NULL;
    PyObject *objs[9];
    const int typenums[9] = { NPY_INT32, NPY_FLOAT64, NPY_INT32, NPY_INT32, NPY_INT32,
        NPY_FLOAT64, NPY_FLOAT64, NPY_INT32, NPY_INT32 };
    PyArrayObject *arrays[9] = { NULL };
    PyObject *output = NULL;
    npy_intp dims[1];
    int err;

    if (!PyArg_ParseTuple(args, "OOOOOOOOO", &objs[0], &objs[1], &objs[2], &objs[3],
            &objs[4], &objs[5], &objs[6], &objs[7], &objs[8])) {
        goto out;
    }
    if (parse_arrays(objs, typenums, 1, 9, arrays) != 0) {
        goto out;
    }
    if (check_sizes_equal(arrays + 3, 6) != 0 || check_sizes_equal(arrays, 2) != 0) {
        goto out;
    }
    dims[0] = PyArray_DIM(arrays[0], 0);
    output = PyArray_SimpleNew(1, dims, NPY_INT32);
    if (output == NULL) {
        goto out;
    }
    Py_BEGIN_ALLOW_THREADS
    err = tsd_relabel_mutations_node(ARRAY_SIZE(arrays[2]), ARRAY_SIZE(arrays[3]),
        ARRAY_SIZE(arrays[0]), (const tsd_id_t *) PyArray_DATA(arrays[0]),
        (const double *) PyArray_DATA(arrays[1]),
        (const tsd_id_t *) PyArray_DATA(arrays[2]),
        (const tsd_id_t *) PyArray_DATA(arrays[3]),
        (const tsd_id_t *) PyArray_DATA(arrays[4]),
        (const double *) PyArray_DATA(arrays[5]),
        (const double *) PyArray_DATA(arrays[6]),
        (const tsd_id_t *) PyArray_DATA(arrays[7]),
        (const tsd_id_t *) PyArray_DATA(arrays[8]),
        (tsd_id_t *) PyArray_DATA((PyArrayObject *) output));
    Py_END_ALLOW_THREADS
    if (err != 0) {
        handle_library_error(err);
        goto out;
    }
    ret = output;
    output = NULL;
out:
    release_arrays(arrays, 9);
    Py_XDECREF(output);
    return ret;
}

/*
 * constrain_ages(nodes_time, nodes_fixed, edges_parent, edges_child, epsilon,
 *     max_iterations) returns a new array of node times.
 */
static PyObject *
tsdate_constrain_ages(PyObject *Py_UNUSED(self), PyObject *args)
{
    PyObject *ret = NULL;
    PyObject *objs[4];
    const int typenums[4] = { NPY_FLOAT64, NPY_BOOL, NPY_INT32, NPY_INT32 };
    PyArrayObject *arrays[4] = { NULL };
    PyObject *output = NULL;
    npy_intp dims[1];
    double epsilon;
    long long max_iterations;
    int err;

    if (!PyArg_ParseTuple(args, "OOOOdL", &objs[0], &objs[1], &objs[2], &objs[3],
            &epsilon, &max_iterations)) {
        goto out;
    }
    if (parse_arrays(objs, typenums, 1, 4, arrays) != 0) {
        goto out;
    }
    if (check_sizes_equal(arrays, 2) != 0 || check_sizes_equal(arrays + 2, 2) != 0) {
        goto out;
    }
    dims[0] = PyArray_DIM(arrays[0], 0);
    output = PyArray_SimpleNew(1, dims, NPY_FLOAT64);
    if (output == NULL) {
        goto out;
    }
    Py_BEGIN_ALLOW_THREADS
    err = tsd_constrain_ages(ARRAY_SIZE(arrays[0]), ARRAY_SIZE(arrays[2]),
        (const double *) PyArray_DATA(arrays[0]),
        (const uint8_t *) PyArray_DATA(arrays[1]),
        (const tsd_id_t *) PyArray_DATA(arrays[2]),
        (const tsd_id_t *) PyArray_DATA(arrays[3]), epsilon, (int64_t) max_iterations,
        (double *) PyArray_DATA((PyArrayObject *) output));
    Py_END_ALLOW_THREADS
    if (err != 0) {
        handle_library_error(err);
        goto out;
    }
    ret = output;
    output = NULL;
out:
    release_arrays(arrays, 4);
    Py_XDECREF(output);
    return ret;
}

/*
 * contains_unary_nodes(nodes_mask, edges_parent, edges_left, edges_right,
 *     indexes_insert, indexes_remove, sequence_length, num_nodes) returns a bool.
 */
static PyObject *
tsdate_contains_unary_nodes(PyObject *Py_UNUSED(self), PyObject *args)
{
    PyObject *ret = NULL;
    PyObject *objs[6];
    const int typenums[6]
        = { NPY_BOOL, NPY_INT32, NPY_FLOAT64, NPY_FLOAT64, NPY_INT32, NPY_INT32 };
    PyArrayObject *arrays[6] = { NULL };
    double sequence_length;
    Py_ssize_t num_nodes;
    int contains_unary = 0;
    int err;

    if (!PyArg_ParseTuple(args, "OOOOOOdn", &objs[0], &objs[1], &objs[2], &objs[3],
            &objs[4], &objs[5], &sequence_length, &num_nodes)) {
        goto out;
    }
    if (parse_arrays(objs, typenums, 1, 6, arrays) != 0) {
        goto out;
    }
    if (check_sizes_equal(arrays + 1, 5) != 0) {
        goto out;
    }
    if (num_nodes < 0 || ARRAY_SIZE(arrays[0]) != (tsd_size_t) num_nodes) {
        PyErr_SetString(
            PyExc_ValueError, "Mask size does not match the number of nodes");
        goto out;
    }
    Py_BEGIN_ALLOW_THREADS
    err = tsd_contains_unary_nodes((tsd_size_t) num_nodes, ARRAY_SIZE(arrays[1]),
        (const uint8_t *) PyArray_DATA(arrays[0]),
        (const tsd_id_t *) PyArray_DATA(arrays[1]),
        (const double *) PyArray_DATA(arrays[2]),
        (const double *) PyArray_DATA(arrays[3]),
        (const tsd_id_t *) PyArray_DATA(arrays[4]),
        (const tsd_id_t *) PyArray_DATA(arrays[5]), sequence_length, &contains_unary);
    Py_END_ALLOW_THREADS
    if (err != 0) {
        handle_library_error(err);
        goto out;
    }
    ret = PyBool_FromLong(contains_unary);
out:
    release_arrays(arrays, 6);
    return ret;
}

/* marginalize_over_ancestors(val): val has at least two rows. */
static PyObject *
tsdate_marginalize_over_ancestors(PyObject *Py_UNUSED(self), PyObject *args)
{
    PyObject *ret = NULL;
    PyObject *val_obj = NULL;
    PyArrayObject *val_array = NULL;
    PyObject *output = NULL;
    npy_intp dims[2];
    int err;

    if (!PyArg_ParseTuple(args, "O", &val_obj)) {
        goto out;
    }
    val_array = (PyArrayObject *) PyArray_FROMANY(
        val_obj, NPY_FLOAT64, 2, 2, NPY_ARRAY_IN_ARRAY);
    if (val_array == NULL) {
        goto out;
    }
    if (ARRAY_DIM(val_array, 0) < 2) {
        PyErr_SetString(PyExc_ValueError, "At least two tips are required");
        goto out;
    }
    dims[0] = PyArray_DIM(val_array, 0) + 1;
    dims[1] = PyArray_DIM(val_array, 1);
    output = PyArray_SimpleNew(2, dims, NPY_FLOAT64);
    if (output == NULL) {
        goto out;
    }
    Py_BEGIN_ALLOW_THREADS
    err = tsd_marginalize_over_ancestors(ARRAY_DIM(val_array, 0),
        ARRAY_DIM(val_array, 1), (const double *) PyArray_DATA(val_array),
        (double *) PyArray_DATA((PyArrayObject *) output));
    Py_END_ALLOW_THREADS
    if (err != 0) {
        handle_library_error(err);
        goto out;
    }
    ret = output;
    output = NULL;
out:
    Py_XDECREF(val_array);
    Py_XDECREF(output);
    return ret;
}

/* conditional_coalescent_variance(num_tips): an array of num_tips + 1 variances. */
static PyObject *
tsdate_conditional_coalescent_variance(PyObject *Py_UNUSED(self), PyObject *args)
{
    PyObject *ret = NULL;
    PyObject *output = NULL;
    Py_ssize_t num_tips;
    npy_intp dims[1];
    int err;

    if (!PyArg_ParseTuple(args, "n", &num_tips)) {
        goto out;
    }
    if (num_tips < 1) {
        PyErr_SetString(PyExc_ValueError, "The number of tips must be positive");
        goto out;
    }
    dims[0] = (npy_intp) num_tips + 1;
    output = PyArray_SimpleNew(1, dims, NPY_FLOAT64);
    if (output == NULL) {
        goto out;
    }
    Py_BEGIN_ALLOW_THREADS
    err = tsd_conditional_coalescent_variance(
        (tsd_size_t) num_tips, (double *) PyArray_DATA((PyArrayObject *) output));
    Py_END_ALLOW_THREADS
    if (err != 0) {
        handle_library_error(err);
        goto out;
    }
    ret = output;
    output = NULL;
out:
    Py_XDECREF(output);
    return ret;
}

static PyMethodDef tsdate_methods[] = {
    { "strerror", (PyCFunction) tsdate_strerror, METH_VARARGS,
        "Return the message for a library error code." },
    { "raise_library_error", (PyCFunction) tsdate_raise_library_error, METH_VARARGS,
        "Raise the Python exception corresponding to a library error code." },
    { "average_gammas", (PyCFunction) tsdate_average_gammas, METH_VARARGS,
        "Natural parameters of the gamma averaging the sufficient statistics of an "
        "array of gammas, given as natural parameters." },
    { "block_singletons", (PyCFunction) tsdate_block_singletons, METH_VARARGS,
        "Group the singletons of unphased individuals into blocks." },
    { "reallocate_unphased", (PyCFunction) tsdate_reallocate_unphased, METH_VARARGS,
        "Reallocate the likelihood of unphased blocks, in place." },
    { "mutation_frequency", (PyCFunction) tsdate_mutation_frequency, METH_VARARGS,
        "The number of members of each sample set below each mutation." },
    { "split_disjoint_nodes", (PyCFunction) tsdate_split_disjoint_nodes, METH_VARARGS,
        "Split disconnected regions of nodes into separate nodes." },
    { "relabel_mutations_node", (PyCFunction) tsdate_relabel_mutations_node,
        METH_VARARGS, "The new node of each mutation, after splitting nodes." },
    { "constrain_ages", (PyCFunction) tsdate_constrain_ages, METH_VARARGS,
        "Constrain node ages to be consistent with the edges." },
    { "contains_unary_nodes", (PyCFunction) tsdate_contains_unary_nodes, METH_VARARGS,
        "Whether any unmasked node has a single child in some tree." },
    { "marginalize_over_ancestors", (PyCFunction) tsdate_marginalize_over_ancestors,
        METH_VARARGS, "Integrate an expectation over counts of extant ancestors." },
    { "conditional_coalescent_variance",
        (PyCFunction) tsdate_conditional_coalescent_variance, METH_VARARGS,
        "Variance of node age given the number of descendant leaves." },
    { NULL } /* Sentinel */
};

static struct PyModuleDef tsdatemodule = { PyModuleDef_HEAD_INIT, "_tsdate", MODULE_DOC,
    -1, tsdate_methods, NULL, NULL, NULL, NULL };

/*
 * Look up scipy.special.cython_special.gammaincinv through its Cython C API capsule.
 * The capsule's name is its C signature, which is used as given to get the pointer.
 */
static int
load_scipy_gammaincinv(void)
{
    int ret = -1;
    PyObject *cython_special = NULL;
    PyObject *capi = NULL;
    PyObject *capsule = NULL;
    const char *name;
    void *ptr;

    cython_special = PyImport_ImportModule("scipy.special.cython_special");
    if (cython_special == NULL) {
        PyErr_SetString(
            PyExc_ImportError, "_tsdate requires scipy.special.cython_special");
        goto out;
    }
    capi = PyObject_GetAttrString(cython_special, "__pyx_capi__");
    if (capi == NULL) {
        goto out;
    }
    capsule = PyDict_GetItemString(capi, "gammaincinv"); /* borrowed */
    if (capsule == NULL || !PyCapsule_CheckExact(capsule)) {
        PyErr_SetString(PyExc_ImportError,
            "scipy.special.cython_special does not provide gammaincinv");
        goto out;
    }
    name = PyCapsule_GetName(capsule);
    if (name == NULL) {
        goto out;
    }
    ptr = PyCapsule_GetPointer(capsule, name);
    if (ptr == NULL) {
        goto out;
    }
    scipy_gammaincinv = (cython_gammaincinv_fn) ptr;
    ret = 0;
out:
    Py_XDECREF(cython_special);
    Py_XDECREF(capi);
    return ret;
}

PyObject *
PyInit__tsdate(void)
{
    PyObject *module = PyModule_Create(&tsdatemodule);

    if (module == NULL) {
        return NULL;
    }
    /* Initialise numpy */
    import_array();

    TsdLibraryError = PyErr_NewException("_tsdate.LibraryError", NULL, NULL);
    Py_INCREF(TsdLibraryError);
    PyModule_AddObject(module, "LibraryError", TsdLibraryError);

    if (load_scipy_gammaincinv() != 0) {
        Py_DECREF(module);
        return NULL;
    }
    if (add_scalar_functions(module) != 0) {
        Py_DECREF(module);
        return NULL;
    }

    return module;
}
