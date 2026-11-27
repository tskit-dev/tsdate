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
scalar_digamma(const double *in, double *out)
{
    return tsd_digamma(in[0], out);
}

static int
scalar_trigamma(const double *in, double *out)
{
    return tsd_trigamma(in[0], out);
}

static int
scalar_betaln(const double *in, double *out)
{
    out[0] = tsd_betaln(in[0], in[1]);
    return 0;
}

static int
scalar_gammainc_der(const double *in, double *out)
{
    return tsd_gammainc_der(in[0], in[1], out);
}

static int
scalar_gammainc_inv(const double *in, double *out)
{
    out[0] = tsd_gammaincinv(in[0], in[1]);
    return 0;
}

static int
scalar_hyperu_laplace(const double *in, double *out)
{
    return tsd_hyperu_laplace(in[0], in[1], in[2], out, out + 1);
}

static int
scalar_hyp1f1_laplace(const double *in, double *out)
{
    return tsd_hyp1f1_laplace(in[0], in[1], in[2], out);
}

static int
scalar_hyp2f1_laplace(const double *in, double *out)
{
    return tsd_hyp2f1_laplace(in[0], in[1], in[2], in[3], out);
}

/* approx: helpers */

static int
scalar_approximate_log_moments(const double *in, double *out)
{
    return tsd_approximate_log_moments(in[0], in[1], out, out + 1, out + 2);
}

static int
scalar_approximate_gamma_kl(const double *in, double *out)
{
    return tsd_approximate_gamma_kl(in[0], in[1], out);
}

static int
scalar_approximate_gamma_mom(const double *in, double *out)
{
    return tsd_approximate_gamma_mom(in[0], in[1], out);
}

static int
scalar_approximate_gamma_iqr(const double *in, double *out)
{
    return tsd_approximate_gamma_iqr(in[0], in[1], in[2], in[3], in[4], out);
}

static int
scalar_valid_moments(const double *in, double *out)
{
    out[0] = tsd_valid_moments(in[0], in[1]);
    return 0;
}

static int
scalar_valid_gamma(const double *in, double *out)
{
    out[0] = tsd_valid_gamma(in[0], in[1]);
    return 0;
}

static int
scalar_valid_hyp1f1(const double *in, double *out)
{
    out[0] = tsd_valid_hyp1f1(in[0], in[1], in[2]);
    return 0;
}

static int
scalar_valid_hyperu(const double *in, double *out)
{
    out[0] = tsd_valid_hyperu(in[0], in[1], in[2]);
    return 0;
}

static int
scalar_valid_hyp2f1(const double *in, double *out)
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
scalar_moments(const double *in, double *out)
{
    return tsd_moments(in[0], in[1], in[2], in[3], in[4], in[5], out, out + 1, out + 2,
        out + 3, out + 4);
}

static int
scalar_rootward_moments(const double *in, double *out)
{
    return tsd_rootward_moments(
        in[0], in[1], in[2], in[3], in[4], out, out + 1, out + 2);
}

static int
scalar_leafward_moments(const double *in, double *out)
{
    return tsd_leafward_moments(
        in[0], in[1], in[2], in[3], in[4], out, out + 1, out + 2);
}

static int
scalar_unphased_moments(const double *in, double *out)
{
    return tsd_unphased_moments(in[0], in[1], in[2], in[3], in[4], in[5], out, out + 1,
        out + 2, out + 3, out + 4);
}

static int
scalar_twin_moments(const double *in, double *out)
{
    return tsd_twin_moments(in[0], in[1], in[2], in[3], out, out + 1, out + 2);
}

static int
scalar_sideways_moments(const double *in, double *out)
{
    return tsd_sideways_moments(
        in[0], in[1], in[2], in[3], in[4], out, out + 1, out + 2);
}

/* approx: node projections */
static int
scalar_gamma_projection(const double *in, double *out)
{
    return tsd_gamma_projection(in, in + 2, in + 4, out, out + 1, out + 3);
}

static int
scalar_leafward_projection(const double *in, double *out)
{
    return tsd_leafward_projection(in[0], in + 1, in + 3, out, out + 1);
}

static int
scalar_rootward_projection(const double *in, double *out)
{
    return tsd_rootward_projection(in[0], in + 1, in + 3, out, out + 1);
}

static int
scalar_unphased_projection(const double *in, double *out)
{
    return tsd_unphased_projection(in, in + 2, in + 4, out, out + 1, out + 3);
}

static int
scalar_twin_projection(const double *in, double *out)
{
    return tsd_twin_projection(in, in + 2, out, out + 1);
}

static int
scalar_sideways_projection(const double *in, double *out)
{
    return tsd_sideways_projection(in[0], in + 1, in + 3, out, out + 1);
}

/* approx: mutation moments */
static int
scalar_mutation_moments(const double *in, double *out)
{
    return tsd_mutation_moments(in[0], in[1], in[2], in[3], in[4], in[5], out, out + 1);
}

static int
scalar_mutation_rootward_moments(const double *in, double *out)
{
    return tsd_mutation_rootward_moments(
        in[0], in[1], in[2], in[3], in[4], out, out + 1);
}

static int
scalar_mutation_leafward_moments(const double *in, double *out)
{
    return tsd_mutation_leafward_moments(
        in[0], in[1], in[2], in[3], in[4], out, out + 1);
}

static int
scalar_mutation_unphased_moments(const double *in, double *out)
{
    return tsd_mutation_unphased_moments(
        in[0], in[1], in[2], in[3], in[4], in[5], out, out + 1, out + 2);
}

static int
scalar_mutation_twin_moments(const double *in, double *out)
{
    return tsd_mutation_twin_moments(in[0], in[1], in[2], in[3], out, out + 1, out + 2);
}

static int
scalar_mutation_sideways_moments(const double *in, double *out)
{
    return tsd_mutation_sideways_moments(
        in[0], in[1], in[2], in[3], in[4], out, out + 1, out + 2);
}

static int
scalar_mutation_edge_moments(const double *in, double *out)
{
    return tsd_mutation_edge_moments(in[0], in[1], out, out + 1);
}

static int
scalar_mutation_block_moments(const double *in, double *out)
{
    return tsd_mutation_block_moments(in[0], in[1], out, out + 1, out + 2);
}

/* approx: mutation projections */
static int
scalar_mutation_gamma_projection(const double *in, double *out)
{
    return tsd_mutation_gamma_projection(in, in + 2, in + 4, out, out + 1);
}

static int
scalar_mutation_leafward_projection(const double *in, double *out)
{
    return tsd_mutation_leafward_projection(in[0], in + 1, in + 3, out, out + 1);
}

static int
scalar_mutation_rootward_projection(const double *in, double *out)
{
    return tsd_mutation_rootward_projection(in[0], in + 1, in + 3, out, out + 1);
}

static int
scalar_mutation_edge_projection(const double *in, double *out)
{
    return tsd_mutation_edge_projection(in[0], in[1], out, out + 1);
}

static int
scalar_mutation_unphased_projection(const double *in, double *out)
{
    return tsd_mutation_unphased_projection(in, in + 2, in + 4, out, out + 1);
}

static int
scalar_mutation_twin_projection(const double *in, double *out)
{
    return tsd_mutation_twin_projection(in, in + 2, out, out + 1);
}

static int
scalar_mutation_sideways_projection(const double *in, double *out)
{
    return tsd_mutation_sideways_projection(in[0], in + 1, in + 3, out, out + 1);
}

static int
scalar_mutation_block_projection(const double *in, double *out)
{
    return tsd_mutation_block_projection(in[0], in[1], out, out + 1);
}

/* variational: x and y are the natural parameters of two gammas, in `in` */
static int
scalar_damp(const double *in, double *out)
{
    return tsd_damp(in, in + 2, in[4], out);
}

static int
scalar_rescale(const double *in, double *out)
{
    return tsd_rescale(in, in[2], out);
}

/* clang-format off */
static const scalar_function_t scalar_functions[] = {
    /* hypergeo */
    { "digamma", 1, 1, scalar_digamma },
    { "trigamma", 1, 1, scalar_trigamma },
    { "betaln", 2, 1, scalar_betaln },
    { "gammainc_der", 2, 1, scalar_gammainc_der },
    { "gammainc_inv", 2, 1, scalar_gammainc_inv },
    { "hyperu_laplace", 3, 2, scalar_hyperu_laplace },
    { "hyp1f1_laplace", 3, 1, scalar_hyp1f1_laplace },
    { "hyp2f1_laplace", 4, 1, scalar_hyp2f1_laplace },
    /* approx: helpers */
    { "approximate_log_moments", 2, 3, scalar_approximate_log_moments },
    { "approximate_gamma_kl", 2, 2, scalar_approximate_gamma_kl },
    { "approximate_gamma_mom", 2, 2, scalar_approximate_gamma_mom },
    { "approximate_gamma_iqr", 5, 2, scalar_approximate_gamma_iqr },
    { "valid_moments", 2, 1, scalar_valid_moments },
    { "valid_gamma", 2, 1, scalar_valid_gamma },
    { "valid_hyp1f1", 3, 1, scalar_valid_hyp1f1 },
    { "valid_hyperu", 3, 1, scalar_valid_hyperu },
    { "valid_hyp2f1", 4, 1, scalar_valid_hyp2f1 },
    /* approx: node moments */
    { "moments", 6, 5, scalar_moments },
    { "rootward_moments", 5, 3, scalar_rootward_moments },
    { "leafward_moments", 5, 3, scalar_leafward_moments },
    { "unphased_moments", 6, 5, scalar_unphased_moments },
    { "twin_moments", 4, 3, scalar_twin_moments },
    { "sideways_moments", 5, 3, scalar_sideways_moments },
    /* approx: node projections */
    { "gamma_projection", 6, 5, scalar_gamma_projection },
    { "leafward_projection", 5, 3, scalar_leafward_projection },
    { "rootward_projection", 5, 3, scalar_rootward_projection },
    { "unphased_projection", 6, 5, scalar_unphased_projection },
    { "twin_projection", 4, 3, scalar_twin_projection },
    { "sideways_projection", 5, 3, scalar_sideways_projection },
    /* approx: mutation moments */
    { "mutation_moments", 6, 2, scalar_mutation_moments },
    { "mutation_rootward_moments", 5, 2, scalar_mutation_rootward_moments },
    { "mutation_leafward_moments", 5, 2, scalar_mutation_leafward_moments },
    { "mutation_unphased_moments", 6, 3, scalar_mutation_unphased_moments },
    { "mutation_twin_moments", 4, 3, scalar_mutation_twin_moments },
    { "mutation_sideways_moments", 5, 3, scalar_mutation_sideways_moments },
    { "mutation_edge_moments", 2, 2, scalar_mutation_edge_moments },
    { "mutation_block_moments", 2, 3, scalar_mutation_block_moments },
    /* approx: mutation projections */
    { "mutation_gamma_projection", 6, 3, scalar_mutation_gamma_projection },
    { "mutation_leafward_projection", 5, 3, scalar_mutation_leafward_projection },
    { "mutation_rootward_projection", 5, 3, scalar_mutation_rootward_projection },
    { "mutation_edge_projection", 2, 3, scalar_mutation_edge_projection },
    { "mutation_unphased_projection", 6, 3, scalar_mutation_unphased_projection },
    { "mutation_twin_projection", 4, 3, scalar_mutation_twin_projection },
    { "mutation_sideways_projection", 5, 3, scalar_mutation_sideways_projection },
    { "mutation_block_projection", 2, 3, scalar_mutation_block_projection },
    /* variational: x0, x1, [y0, y1,] s */
    { "damp", 5, 1, scalar_damp },
    { "rescale", 3, 1, scalar_rescale },
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
 * Functions of the tree sequence (phasing, utilities, prior and rescaling). Arrays are
 * converted to C-contiguous arrays of the right type and dimension; sizes of arrays
 * that share a dimension are checked here (ValueError), and everything else by the
 * library.
 */

#define ARRAY_LEN(array) ((tsd_size_t) PyArray_DIM((array), 0))
#define DOUBLES(array)   ((double *) PyArray_DATA(array))
#define IDS(array)       ((tsd_id_t *) PyArray_DATA(array))
#define FLAGS(array)     ((uint8_t *) PyArray_DATA(array))

/* A new array of the given type and shape holding a copy of `data`. */
static PyArrayObject *
new_array_from_data(int typenum, int ndim, npy_intp *dims, const void *data)
{
    PyArrayObject *ret = (PyArrayObject *) PyArray_SimpleNew(ndim, dims, typenum);

    if (ret != NULL) {
        memcpy(PyArray_DATA(ret), data, (size_t) PyArray_NBYTES(ret));
    }
    return ret;
}

static PyObject *
tsdate_block_singletons(PyObject *Py_UNUSED(self), PyObject *args, PyObject *kwds)
{
    PyObject *ret = NULL;
    static char *kwlist[] = { "individuals_unphased", "nodes_individual",
        "mutations_node", "mutations_position", "edges_child", "edges_left",
        "edges_right", "indexes_insert", "indexes_remove", "sequence_length", NULL };
    PyObject *individuals_unphased_obj = NULL;
    PyObject *nodes_individual_obj = NULL;
    PyObject *mutations_node_obj = NULL;
    PyObject *mutations_position_obj = NULL;
    PyObject *edges_child_obj = NULL;
    PyObject *edges_left_obj = NULL;
    PyObject *edges_right_obj = NULL;
    PyObject *indexes_insert_obj = NULL;
    PyObject *indexes_remove_obj = NULL;
    PyArrayObject *individuals_unphased = NULL;
    PyArrayObject *nodes_individual = NULL;
    PyArrayObject *mutations_node = NULL;
    PyArrayObject *mutations_position = NULL;
    PyArrayObject *edges_child = NULL;
    PyArrayObject *edges_left = NULL;
    PyArrayObject *edges_right = NULL;
    PyArrayObject *indexes_insert = NULL;
    PyArrayObject *indexes_remove = NULL;
    PyArrayObject *blocks_stats = NULL;
    PyArrayObject *blocks_edges = NULL;
    PyArrayObject *mutations_block = NULL;
    double *stats = NULL;
    tsd_id_t *edges = NULL;
    double sequence_length;
    tsd_size_t num_edges, num_mutations, num_blocks = 0;
    npy_intp dims[2];
    int err;

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "OOOOOOOOOd", kwlist,
            &individuals_unphased_obj, &nodes_individual_obj, &mutations_node_obj,
            &mutations_position_obj, &edges_child_obj, &edges_left_obj, &edges_right_obj,
            &indexes_insert_obj, &indexes_remove_obj, &sequence_length)) {
        goto out;
    }
    individuals_unphased = (PyArrayObject *) PyArray_FROMANY(
        individuals_unphased_obj, NPY_BOOL, 1, 1, NPY_ARRAY_IN_ARRAY);
    if (individuals_unphased == NULL) {
        goto out;
    }
    nodes_individual = (PyArrayObject *) PyArray_FROMANY(
        nodes_individual_obj, NPY_INT32, 1, 1, NPY_ARRAY_IN_ARRAY);
    if (nodes_individual == NULL) {
        goto out;
    }
    mutations_node = (PyArrayObject *) PyArray_FROMANY(
        mutations_node_obj, NPY_INT32, 1, 1, NPY_ARRAY_IN_ARRAY);
    if (mutations_node == NULL) {
        goto out;
    }
    mutations_position = (PyArrayObject *) PyArray_FROMANY(
        mutations_position_obj, NPY_FLOAT64, 1, 1, NPY_ARRAY_IN_ARRAY);
    if (mutations_position == NULL) {
        goto out;
    }
    edges_child = (PyArrayObject *) PyArray_FROMANY(
        edges_child_obj, NPY_INT32, 1, 1, NPY_ARRAY_IN_ARRAY);
    if (edges_child == NULL) {
        goto out;
    }
    edges_left = (PyArrayObject *) PyArray_FROMANY(
        edges_left_obj, NPY_FLOAT64, 1, 1, NPY_ARRAY_IN_ARRAY);
    if (edges_left == NULL) {
        goto out;
    }
    edges_right = (PyArrayObject *) PyArray_FROMANY(
        edges_right_obj, NPY_FLOAT64, 1, 1, NPY_ARRAY_IN_ARRAY);
    if (edges_right == NULL) {
        goto out;
    }
    indexes_insert = (PyArrayObject *) PyArray_FROMANY(
        indexes_insert_obj, NPY_INT32, 1, 1, NPY_ARRAY_IN_ARRAY);
    if (indexes_insert == NULL) {
        goto out;
    }
    indexes_remove = (PyArrayObject *) PyArray_FROMANY(
        indexes_remove_obj, NPY_INT32, 1, 1, NPY_ARRAY_IN_ARRAY);
    if (indexes_remove == NULL) {
        goto out;
    }
    num_edges = ARRAY_LEN(edges_child);
    num_mutations = ARRAY_LEN(mutations_node);
    if (ARRAY_LEN(edges_left) != num_edges || ARRAY_LEN(edges_right) != num_edges
        || ARRAY_LEN(indexes_insert) != num_edges
        || ARRAY_LEN(indexes_remove) != num_edges) {
        PyErr_SetString(PyExc_ValueError, "Edge arrays must have the same size");
        goto out;
    }
    if (ARRAY_LEN(mutations_position) != num_mutations) {
        PyErr_SetString(PyExc_ValueError, "Mutation arrays must have the same size");
        goto out;
    }
    dims[0] = (npy_intp) num_mutations;
    mutations_block = (PyArrayObject *) PyArray_SimpleNew(1, dims, NPY_INT32);
    if (mutations_block == NULL) {
        goto out;
    }
    Py_BEGIN_ALLOW_THREADS
    err = tsd_block_singletons(ARRAY_LEN(nodes_individual),
        ARRAY_LEN(individuals_unphased), num_edges, num_mutations,
        FLAGS(individuals_unphased), IDS(nodes_individual), IDS(mutations_node),
        DOUBLES(mutations_position), IDS(edges_child), DOUBLES(edges_left),
        DOUBLES(edges_right), IDS(indexes_insert), IDS(indexes_remove), sequence_length,
        &num_blocks, &stats, &edges, IDS(mutations_block));
    Py_END_ALLOW_THREADS
    if (err != 0) {
        handle_library_error(err);
        goto out;
    }
    dims[0] = (npy_intp) num_blocks;
    dims[1] = 2;
    blocks_stats = new_array_from_data(NPY_FLOAT64, 2, dims, stats);
    if (blocks_stats == NULL) {
        goto out;
    }
    blocks_edges = new_array_from_data(NPY_INT32, 2, dims, edges);
    if (blocks_edges == NULL) {
        goto out;
    }
    ret = Py_BuildValue("OOO", blocks_stats, blocks_edges, mutations_block);
out:
    tsd_safe_free(stats);
    tsd_safe_free(edges);
    Py_XDECREF(individuals_unphased);
    Py_XDECREF(nodes_individual);
    Py_XDECREF(mutations_node);
    Py_XDECREF(mutations_position);
    Py_XDECREF(edges_child);
    Py_XDECREF(edges_left);
    Py_XDECREF(edges_right);
    Py_XDECREF(indexes_insert);
    Py_XDECREF(indexes_remove);
    Py_XDECREF(blocks_stats);
    Py_XDECREF(blocks_edges);
    Py_XDECREF(mutations_block);
    return ret;
}

static PyObject *
tsdate_reallocate_unphased(PyObject *Py_UNUSED(self), PyObject *args, PyObject *kwds)
{
    PyObject *ret = NULL;
    static char *kwlist[] = { "edges_likelihood", "mutations_phase", "mutations_block",
        "blocks_edges", NULL };
    PyObject *edges_likelihood_obj = NULL;
    PyObject *mutations_phase_obj = NULL;
    PyObject *mutations_block_obj = NULL;
    PyObject *blocks_edges_obj = NULL;
    PyArrayObject *edges_likelihood = NULL;
    PyArrayObject *mutations_phase = NULL;
    PyArrayObject *mutations_block = NULL;
    PyArrayObject *blocks_edges = NULL;
    PyArrayObject *new_likelihood = NULL;
    int err;

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "OOOO", kwlist, &edges_likelihood_obj,
            &mutations_phase_obj, &mutations_block_obj, &blocks_edges_obj)) {
        goto out;
    }
    edges_likelihood = (PyArrayObject *) PyArray_FROMANY(
        edges_likelihood_obj, NPY_FLOAT64, 2, 2, NPY_ARRAY_IN_ARRAY);
    if (edges_likelihood == NULL) {
        goto out;
    }
    mutations_phase = (PyArrayObject *) PyArray_FROMANY(
        mutations_phase_obj, NPY_FLOAT64, 1, 1, NPY_ARRAY_IN_ARRAY);
    if (mutations_phase == NULL) {
        goto out;
    }
    mutations_block = (PyArrayObject *) PyArray_FROMANY(
        mutations_block_obj, NPY_INT32, 1, 1, NPY_ARRAY_IN_ARRAY);
    if (mutations_block == NULL) {
        goto out;
    }
    blocks_edges = (PyArrayObject *) PyArray_FROMANY(
        blocks_edges_obj, NPY_INT32, 2, 2, NPY_ARRAY_IN_ARRAY);
    if (blocks_edges == NULL) {
        goto out;
    }
    if (PyArray_DIM(edges_likelihood, 1) != 2 || PyArray_DIM(blocks_edges, 1) != 2) {
        PyErr_SetString(
            PyExc_ValueError, "edges_likelihood and blocks_edges must have two columns");
        goto out;
    }
    if (ARRAY_LEN(mutations_block) != ARRAY_LEN(mutations_phase)) {
        PyErr_SetString(PyExc_ValueError, "Mutation arrays must have the same size");
        goto out;
    }
    new_likelihood = (PyArrayObject *) PyArray_NewCopy(edges_likelihood, NPY_CORDER);
    if (new_likelihood == NULL) {
        goto out;
    }
    Py_BEGIN_ALLOW_THREADS
    err = tsd_reallocate_unphased(ARRAY_LEN(new_likelihood), ARRAY_LEN(mutations_phase),
        ARRAY_LEN(blocks_edges), DOUBLES(mutations_phase), IDS(mutations_block),
        IDS(blocks_edges), DOUBLES(new_likelihood));
    Py_END_ALLOW_THREADS
    if (err != 0) {
        handle_library_error(err);
        goto out;
    }
    ret = (PyObject *) new_likelihood;
    new_likelihood = NULL;
out:
    Py_XDECREF(edges_likelihood);
    Py_XDECREF(mutations_phase);
    Py_XDECREF(mutations_block);
    Py_XDECREF(blocks_edges);
    Py_XDECREF(new_likelihood);
    return ret;
}

static PyObject *
tsdate_mutation_frequency(PyObject *Py_UNUSED(self), PyObject *args, PyObject *kwds)
{
    PyObject *ret = NULL;
    static char *kwlist[] = { "nodes_sample", "mutations_node", "mutations_position",
        "edges_parent", "edges_child", "edges_left", "edges_right", "indexes_insert",
        "indexes_remove", "sequence_length", NULL };
    PyObject *nodes_sample_obj = NULL;
    PyObject *mutations_node_obj = NULL;
    PyObject *mutations_position_obj = NULL;
    PyObject *edges_parent_obj = NULL;
    PyObject *edges_child_obj = NULL;
    PyObject *edges_left_obj = NULL;
    PyObject *edges_right_obj = NULL;
    PyObject *indexes_insert_obj = NULL;
    PyObject *indexes_remove_obj = NULL;
    PyArrayObject *nodes_sample = NULL;
    PyArrayObject *mutations_node = NULL;
    PyArrayObject *mutations_position = NULL;
    PyArrayObject *edges_parent = NULL;
    PyArrayObject *edges_child = NULL;
    PyArrayObject *edges_left = NULL;
    PyArrayObject *edges_right = NULL;
    PyArrayObject *indexes_insert = NULL;
    PyArrayObject *indexes_remove = NULL;
    PyArrayObject *mutations_freq = NULL;
    double sequence_length;
    tsd_size_t num_edges, num_mutations;
    npy_intp dims[2];
    int err;

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "OOOOOOOOOd", kwlist, &nodes_sample_obj,
            &mutations_node_obj, &mutations_position_obj, &edges_parent_obj,
            &edges_child_obj, &edges_left_obj, &edges_right_obj, &indexes_insert_obj,
            &indexes_remove_obj, &sequence_length)) {
        goto out;
    }
    nodes_sample = (PyArrayObject *) PyArray_FROMANY(
        nodes_sample_obj, NPY_BOOL, 2, 2, NPY_ARRAY_IN_ARRAY);
    if (nodes_sample == NULL) {
        goto out;
    }
    mutations_node = (PyArrayObject *) PyArray_FROMANY(
        mutations_node_obj, NPY_INT32, 1, 1, NPY_ARRAY_IN_ARRAY);
    if (mutations_node == NULL) {
        goto out;
    }
    mutations_position = (PyArrayObject *) PyArray_FROMANY(
        mutations_position_obj, NPY_FLOAT64, 1, 1, NPY_ARRAY_IN_ARRAY);
    if (mutations_position == NULL) {
        goto out;
    }
    edges_parent = (PyArrayObject *) PyArray_FROMANY(
        edges_parent_obj, NPY_INT32, 1, 1, NPY_ARRAY_IN_ARRAY);
    if (edges_parent == NULL) {
        goto out;
    }
    edges_child = (PyArrayObject *) PyArray_FROMANY(
        edges_child_obj, NPY_INT32, 1, 1, NPY_ARRAY_IN_ARRAY);
    if (edges_child == NULL) {
        goto out;
    }
    edges_left = (PyArrayObject *) PyArray_FROMANY(
        edges_left_obj, NPY_FLOAT64, 1, 1, NPY_ARRAY_IN_ARRAY);
    if (edges_left == NULL) {
        goto out;
    }
    edges_right = (PyArrayObject *) PyArray_FROMANY(
        edges_right_obj, NPY_FLOAT64, 1, 1, NPY_ARRAY_IN_ARRAY);
    if (edges_right == NULL) {
        goto out;
    }
    indexes_insert = (PyArrayObject *) PyArray_FROMANY(
        indexes_insert_obj, NPY_INT32, 1, 1, NPY_ARRAY_IN_ARRAY);
    if (indexes_insert == NULL) {
        goto out;
    }
    indexes_remove = (PyArrayObject *) PyArray_FROMANY(
        indexes_remove_obj, NPY_INT32, 1, 1, NPY_ARRAY_IN_ARRAY);
    if (indexes_remove == NULL) {
        goto out;
    }
    num_edges = ARRAY_LEN(edges_parent);
    num_mutations = ARRAY_LEN(mutations_node);
    if (ARRAY_LEN(edges_child) != num_edges || ARRAY_LEN(edges_left) != num_edges
        || ARRAY_LEN(edges_right) != num_edges || ARRAY_LEN(indexes_insert) != num_edges
        || ARRAY_LEN(indexes_remove) != num_edges) {
        PyErr_SetString(PyExc_ValueError, "Edge arrays must have the same size");
        goto out;
    }
    if (ARRAY_LEN(mutations_position) != num_mutations) {
        PyErr_SetString(PyExc_ValueError, "Mutation arrays must have the same size");
        goto out;
    }
    dims[0] = (npy_intp) num_mutations;
    dims[1] = PyArray_DIM(nodes_sample, 1);
    mutations_freq = (PyArrayObject *) PyArray_SimpleNew(2, dims, NPY_INT32);
    if (mutations_freq == NULL) {
        goto out;
    }
    Py_BEGIN_ALLOW_THREADS
    err = tsd_mutation_frequency(ARRAY_LEN(nodes_sample),
        (tsd_size_t) PyArray_DIM(nodes_sample, 1), num_edges, num_mutations,
        FLAGS(nodes_sample), IDS(mutations_node), DOUBLES(mutations_position),
        IDS(edges_parent), IDS(edges_child), DOUBLES(edges_left), DOUBLES(edges_right),
        IDS(indexes_insert), IDS(indexes_remove), sequence_length,
        (int32_t *) PyArray_DATA(mutations_freq));
    Py_END_ALLOW_THREADS
    if (err != 0) {
        handle_library_error(err);
        goto out;
    }
    ret = (PyObject *) mutations_freq;
    mutations_freq = NULL;
out:
    Py_XDECREF(nodes_sample);
    Py_XDECREF(mutations_node);
    Py_XDECREF(mutations_position);
    Py_XDECREF(edges_parent);
    Py_XDECREF(edges_child);
    Py_XDECREF(edges_left);
    Py_XDECREF(edges_right);
    Py_XDECREF(indexes_insert);
    Py_XDECREF(indexes_remove);
    Py_XDECREF(mutations_freq);
    return ret;
}

static PyObject *
tsdate_split_disjoint_nodes(PyObject *Py_UNUSED(self), PyObject *args, PyObject *kwds)
{
    PyObject *ret = NULL;
    static char *kwlist[] = { "edges_parent", "edges_child", "edges_left", "edges_right",
        "nodes_excluded", NULL };
    PyObject *edges_parent_obj = NULL;
    PyObject *edges_child_obj = NULL;
    PyObject *edges_left_obj = NULL;
    PyObject *edges_right_obj = NULL;
    PyObject *nodes_excluded_obj = NULL;
    PyArrayObject *edges_parent = NULL;
    PyArrayObject *edges_child = NULL;
    PyArrayObject *edges_left = NULL;
    PyArrayObject *edges_right = NULL;
    PyArrayObject *nodes_excluded = NULL;
    PyArrayObject *new_parent = NULL;
    PyArrayObject *new_child = NULL;
    PyArrayObject *nodes_order = NULL;
    PyArrayObject *split_nodes = NULL;
    tsd_id_t *order = NULL;
    tsd_size_t num_nodes, num_edges, num_split = 0;
    npy_intp dims[1];
    int err;

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "OOOOO", kwlist, &edges_parent_obj,
            &edges_child_obj, &edges_left_obj, &edges_right_obj, &nodes_excluded_obj)) {
        goto out;
    }
    edges_parent = (PyArrayObject *) PyArray_FROMANY(
        edges_parent_obj, NPY_INT32, 1, 1, NPY_ARRAY_IN_ARRAY);
    if (edges_parent == NULL) {
        goto out;
    }
    edges_child = (PyArrayObject *) PyArray_FROMANY(
        edges_child_obj, NPY_INT32, 1, 1, NPY_ARRAY_IN_ARRAY);
    if (edges_child == NULL) {
        goto out;
    }
    edges_left = (PyArrayObject *) PyArray_FROMANY(
        edges_left_obj, NPY_FLOAT64, 1, 1, NPY_ARRAY_IN_ARRAY);
    if (edges_left == NULL) {
        goto out;
    }
    edges_right = (PyArrayObject *) PyArray_FROMANY(
        edges_right_obj, NPY_FLOAT64, 1, 1, NPY_ARRAY_IN_ARRAY);
    if (edges_right == NULL) {
        goto out;
    }
    nodes_excluded = (PyArrayObject *) PyArray_FROMANY(
        nodes_excluded_obj, NPY_BOOL, 1, 1, NPY_ARRAY_IN_ARRAY);
    if (nodes_excluded == NULL) {
        goto out;
    }
    num_nodes = ARRAY_LEN(nodes_excluded);
    num_edges = ARRAY_LEN(edges_parent);
    if (ARRAY_LEN(edges_child) != num_edges || ARRAY_LEN(edges_left) != num_edges
        || ARRAY_LEN(edges_right) != num_edges) {
        PyErr_SetString(PyExc_ValueError, "Edge arrays must have the same size");
        goto out;
    }
    dims[0] = (npy_intp) num_edges;
    new_parent = (PyArrayObject *) PyArray_SimpleNew(1, dims, NPY_INT32);
    if (new_parent == NULL) {
        goto out;
    }
    new_child = (PyArrayObject *) PyArray_SimpleNew(1, dims, NPY_INT32);
    if (new_child == NULL) {
        goto out;
    }
    Py_BEGIN_ALLOW_THREADS
    err = tsd_split_disjoint_nodes(num_nodes, num_edges, IDS(edges_parent),
        IDS(edges_child), DOUBLES(edges_left), DOUBLES(edges_right),
        FLAGS(nodes_excluded), IDS(new_parent), IDS(new_child), &num_split, &order);
    Py_END_ALLOW_THREADS
    if (err != 0) {
        handle_library_error(err);
        goto out;
    }
    dims[0] = (npy_intp) (num_nodes + num_split);
    nodes_order = new_array_from_data(NPY_INT32, 1, dims, order);
    if (nodes_order == NULL) {
        goto out;
    }
    dims[0] = (npy_intp) num_split;
    split_nodes = new_array_from_data(NPY_INT32, 1, dims, order + num_nodes);
    if (split_nodes == NULL) {
        goto out;
    }
    ret = Py_BuildValue("OOOO", new_parent, new_child, nodes_order, split_nodes);
out:
    tsd_safe_free(order);
    Py_XDECREF(edges_parent);
    Py_XDECREF(edges_child);
    Py_XDECREF(edges_left);
    Py_XDECREF(edges_right);
    Py_XDECREF(nodes_excluded);
    Py_XDECREF(new_parent);
    Py_XDECREF(new_child);
    Py_XDECREF(nodes_order);
    Py_XDECREF(split_nodes);
    return ret;
}

static PyObject *
tsdate_relabel_mutations_node(PyObject *Py_UNUSED(self), PyObject *args, PyObject *kwds)
{
    PyObject *ret = NULL;
    static char *kwlist[] = { "mutations_node", "mutations_position", "nodes_order",
        "edges_parent", "edges_child", "edges_left", "edges_right", "indexes_insert",
        "indexes_remove", NULL };
    PyObject *mutations_node_obj = NULL;
    PyObject *mutations_position_obj = NULL;
    PyObject *nodes_order_obj = NULL;
    PyObject *edges_parent_obj = NULL;
    PyObject *edges_child_obj = NULL;
    PyObject *edges_left_obj = NULL;
    PyObject *edges_right_obj = NULL;
    PyObject *indexes_insert_obj = NULL;
    PyObject *indexes_remove_obj = NULL;
    PyArrayObject *mutations_node = NULL;
    PyArrayObject *mutations_position = NULL;
    PyArrayObject *nodes_order = NULL;
    PyArrayObject *edges_parent = NULL;
    PyArrayObject *edges_child = NULL;
    PyArrayObject *edges_left = NULL;
    PyArrayObject *edges_right = NULL;
    PyArrayObject *indexes_insert = NULL;
    PyArrayObject *indexes_remove = NULL;
    PyArrayObject *new_node = NULL;
    tsd_size_t num_edges, num_mutations;
    npy_intp dims[1];
    int err;

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "OOOOOOOOO", kwlist,
            &mutations_node_obj, &mutations_position_obj, &nodes_order_obj,
            &edges_parent_obj, &edges_child_obj, &edges_left_obj, &edges_right_obj,
            &indexes_insert_obj, &indexes_remove_obj)) {
        goto out;
    }
    mutations_node = (PyArrayObject *) PyArray_FROMANY(
        mutations_node_obj, NPY_INT32, 1, 1, NPY_ARRAY_IN_ARRAY);
    if (mutations_node == NULL) {
        goto out;
    }
    mutations_position = (PyArrayObject *) PyArray_FROMANY(
        mutations_position_obj, NPY_FLOAT64, 1, 1, NPY_ARRAY_IN_ARRAY);
    if (mutations_position == NULL) {
        goto out;
    }
    nodes_order = (PyArrayObject *) PyArray_FROMANY(
        nodes_order_obj, NPY_INT32, 1, 1, NPY_ARRAY_IN_ARRAY);
    if (nodes_order == NULL) {
        goto out;
    }
    edges_parent = (PyArrayObject *) PyArray_FROMANY(
        edges_parent_obj, NPY_INT32, 1, 1, NPY_ARRAY_IN_ARRAY);
    if (edges_parent == NULL) {
        goto out;
    }
    edges_child = (PyArrayObject *) PyArray_FROMANY(
        edges_child_obj, NPY_INT32, 1, 1, NPY_ARRAY_IN_ARRAY);
    if (edges_child == NULL) {
        goto out;
    }
    edges_left = (PyArrayObject *) PyArray_FROMANY(
        edges_left_obj, NPY_FLOAT64, 1, 1, NPY_ARRAY_IN_ARRAY);
    if (edges_left == NULL) {
        goto out;
    }
    edges_right = (PyArrayObject *) PyArray_FROMANY(
        edges_right_obj, NPY_FLOAT64, 1, 1, NPY_ARRAY_IN_ARRAY);
    if (edges_right == NULL) {
        goto out;
    }
    indexes_insert = (PyArrayObject *) PyArray_FROMANY(
        indexes_insert_obj, NPY_INT32, 1, 1, NPY_ARRAY_IN_ARRAY);
    if (indexes_insert == NULL) {
        goto out;
    }
    indexes_remove = (PyArrayObject *) PyArray_FROMANY(
        indexes_remove_obj, NPY_INT32, 1, 1, NPY_ARRAY_IN_ARRAY);
    if (indexes_remove == NULL) {
        goto out;
    }
    num_edges = ARRAY_LEN(edges_parent);
    num_mutations = ARRAY_LEN(mutations_node);
    if (ARRAY_LEN(edges_child) != num_edges || ARRAY_LEN(edges_left) != num_edges
        || ARRAY_LEN(edges_right) != num_edges || ARRAY_LEN(indexes_insert) != num_edges
        || ARRAY_LEN(indexes_remove) != num_edges) {
        PyErr_SetString(PyExc_ValueError, "Edge arrays must have the same size");
        goto out;
    }
    if (ARRAY_LEN(mutations_position) != num_mutations) {
        PyErr_SetString(PyExc_ValueError, "Mutation arrays must have the same size");
        goto out;
    }
    dims[0] = (npy_intp) num_mutations;
    new_node = (PyArrayObject *) PyArray_SimpleNew(1, dims, NPY_INT32);
    if (new_node == NULL) {
        goto out;
    }
    Py_BEGIN_ALLOW_THREADS
    err = tsd_relabel_mutations_node(ARRAY_LEN(nodes_order), num_edges, num_mutations,
        IDS(mutations_node), DOUBLES(mutations_position), IDS(nodes_order),
        IDS(edges_parent), IDS(edges_child), DOUBLES(edges_left), DOUBLES(edges_right),
        IDS(indexes_insert), IDS(indexes_remove), IDS(new_node));
    Py_END_ALLOW_THREADS
    if (err != 0) {
        handle_library_error(err);
        goto out;
    }
    ret = (PyObject *) new_node;
    new_node = NULL;
out:
    Py_XDECREF(mutations_node);
    Py_XDECREF(mutations_position);
    Py_XDECREF(nodes_order);
    Py_XDECREF(edges_parent);
    Py_XDECREF(edges_child);
    Py_XDECREF(edges_left);
    Py_XDECREF(edges_right);
    Py_XDECREF(indexes_insert);
    Py_XDECREF(indexes_remove);
    Py_XDECREF(new_node);
    return ret;
}

static PyObject *
tsdate_constrain_ages(PyObject *Py_UNUSED(self), PyObject *args, PyObject *kwds)
{
    PyObject *ret = NULL;
    static char *kwlist[] = { "nodes_time", "nodes_fixed", "edges_parent", "edges_child",
        "epsilon", "max_iterations", NULL };
    PyObject *nodes_time_obj = NULL;
    PyObject *nodes_fixed_obj = NULL;
    PyObject *edges_parent_obj = NULL;
    PyObject *edges_child_obj = NULL;
    PyArrayObject *nodes_time = NULL;
    PyArrayObject *nodes_fixed = NULL;
    PyArrayObject *edges_parent = NULL;
    PyArrayObject *edges_child = NULL;
    PyArrayObject *new_time = NULL;
    double epsilon;
    long long max_iterations;
    tsd_size_t num_nodes, num_edges;
    npy_intp dims[1];
    int err;

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "OOOOdL", kwlist, &nodes_time_obj,
            &nodes_fixed_obj, &edges_parent_obj, &edges_child_obj, &epsilon,
            &max_iterations)) {
        goto out;
    }
    nodes_time = (PyArrayObject *) PyArray_FROMANY(
        nodes_time_obj, NPY_FLOAT64, 1, 1, NPY_ARRAY_IN_ARRAY);
    if (nodes_time == NULL) {
        goto out;
    }
    nodes_fixed = (PyArrayObject *) PyArray_FROMANY(
        nodes_fixed_obj, NPY_BOOL, 1, 1, NPY_ARRAY_IN_ARRAY);
    if (nodes_fixed == NULL) {
        goto out;
    }
    edges_parent = (PyArrayObject *) PyArray_FROMANY(
        edges_parent_obj, NPY_INT32, 1, 1, NPY_ARRAY_IN_ARRAY);
    if (edges_parent == NULL) {
        goto out;
    }
    edges_child = (PyArrayObject *) PyArray_FROMANY(
        edges_child_obj, NPY_INT32, 1, 1, NPY_ARRAY_IN_ARRAY);
    if (edges_child == NULL) {
        goto out;
    }
    num_nodes = ARRAY_LEN(nodes_time);
    num_edges = ARRAY_LEN(edges_parent);
    if (ARRAY_LEN(nodes_fixed) != num_nodes) {
        PyErr_SetString(PyExc_ValueError, "Node arrays must have the same size");
        goto out;
    }
    if (ARRAY_LEN(edges_child) != num_edges) {
        PyErr_SetString(PyExc_ValueError, "Edge arrays must have the same size");
        goto out;
    }
    dims[0] = (npy_intp) num_nodes;
    new_time = (PyArrayObject *) PyArray_SimpleNew(1, dims, NPY_FLOAT64);
    if (new_time == NULL) {
        goto out;
    }
    Py_BEGIN_ALLOW_THREADS
    err = tsd_constrain_ages(num_nodes, num_edges, DOUBLES(nodes_time),
        FLAGS(nodes_fixed), IDS(edges_parent), IDS(edges_child), epsilon,
        (int64_t) max_iterations, DOUBLES(new_time));
    Py_END_ALLOW_THREADS
    if (err != 0) {
        handle_library_error(err);
        goto out;
    }
    ret = (PyObject *) new_time;
    new_time = NULL;
out:
    Py_XDECREF(nodes_time);
    Py_XDECREF(nodes_fixed);
    Py_XDECREF(edges_parent);
    Py_XDECREF(edges_child);
    Py_XDECREF(new_time);
    return ret;
}

static PyObject *
tsdate_contains_unary_nodes(PyObject *Py_UNUSED(self), PyObject *args, PyObject *kwds)
{
    PyObject *ret = NULL;
    static char *kwlist[] = { "nodes_mask", "edges_parent", "edges_left", "edges_right",
        "indexes_insert", "indexes_remove", "sequence_length", NULL };
    PyObject *nodes_mask_obj = NULL;
    PyObject *edges_parent_obj = NULL;
    PyObject *edges_left_obj = NULL;
    PyObject *edges_right_obj = NULL;
    PyObject *indexes_insert_obj = NULL;
    PyObject *indexes_remove_obj = NULL;
    PyArrayObject *nodes_mask = NULL;
    PyArrayObject *edges_parent = NULL;
    PyArrayObject *edges_left = NULL;
    PyArrayObject *edges_right = NULL;
    PyArrayObject *indexes_insert = NULL;
    PyArrayObject *indexes_remove = NULL;
    double sequence_length;
    tsd_size_t num_edges;
    int contains_unary = 0;
    int err;

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "OOOOOOd", kwlist, &nodes_mask_obj,
            &edges_parent_obj, &edges_left_obj, &edges_right_obj, &indexes_insert_obj,
            &indexes_remove_obj, &sequence_length)) {
        goto out;
    }
    nodes_mask = (PyArrayObject *) PyArray_FROMANY(
        nodes_mask_obj, NPY_BOOL, 1, 1, NPY_ARRAY_IN_ARRAY);
    if (nodes_mask == NULL) {
        goto out;
    }
    edges_parent = (PyArrayObject *) PyArray_FROMANY(
        edges_parent_obj, NPY_INT32, 1, 1, NPY_ARRAY_IN_ARRAY);
    if (edges_parent == NULL) {
        goto out;
    }
    edges_left = (PyArrayObject *) PyArray_FROMANY(
        edges_left_obj, NPY_FLOAT64, 1, 1, NPY_ARRAY_IN_ARRAY);
    if (edges_left == NULL) {
        goto out;
    }
    edges_right = (PyArrayObject *) PyArray_FROMANY(
        edges_right_obj, NPY_FLOAT64, 1, 1, NPY_ARRAY_IN_ARRAY);
    if (edges_right == NULL) {
        goto out;
    }
    indexes_insert = (PyArrayObject *) PyArray_FROMANY(
        indexes_insert_obj, NPY_INT32, 1, 1, NPY_ARRAY_IN_ARRAY);
    if (indexes_insert == NULL) {
        goto out;
    }
    indexes_remove = (PyArrayObject *) PyArray_FROMANY(
        indexes_remove_obj, NPY_INT32, 1, 1, NPY_ARRAY_IN_ARRAY);
    if (indexes_remove == NULL) {
        goto out;
    }
    num_edges = ARRAY_LEN(edges_parent);
    if (ARRAY_LEN(edges_left) != num_edges || ARRAY_LEN(edges_right) != num_edges
        || ARRAY_LEN(indexes_insert) != num_edges
        || ARRAY_LEN(indexes_remove) != num_edges) {
        PyErr_SetString(PyExc_ValueError, "Edge arrays must have the same size");
        goto out;
    }
    Py_BEGIN_ALLOW_THREADS
    err = tsd_contains_unary_nodes(ARRAY_LEN(nodes_mask), num_edges, FLAGS(nodes_mask),
        IDS(edges_parent), DOUBLES(edges_left), DOUBLES(edges_right),
        IDS(indexes_insert), IDS(indexes_remove), sequence_length, &contains_unary);
    Py_END_ALLOW_THREADS
    if (err != 0) {
        handle_library_error(err);
        goto out;
    }
    ret = PyBool_FromLong(contains_unary);
out:
    Py_XDECREF(nodes_mask);
    Py_XDECREF(edges_parent);
    Py_XDECREF(edges_left);
    Py_XDECREF(edges_right);
    Py_XDECREF(indexes_insert);
    Py_XDECREF(indexes_remove);
    return ret;
}

static PyObject *
tsdate_conditional_coalescent_variance(
    PyObject *Py_UNUSED(self), PyObject *args, PyObject *kwds)
{
    PyObject *ret = NULL;
    static char *kwlist[] = { "num_tips", NULL };
    PyArrayObject *variance = NULL;
    Py_ssize_t num_tips;
    npy_intp dims[1];
    int err;

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "n", kwlist, &num_tips)) {
        goto out;
    }
    if (num_tips < 0) {
        PyErr_SetString(PyExc_ValueError, "The number of tips must be non-negative");
        goto out;
    }
    dims[0] = (npy_intp) num_tips + 1;
    variance = (PyArrayObject *) PyArray_SimpleNew(1, dims, NPY_FLOAT64);
    if (variance == NULL) {
        goto out;
    }
    Py_BEGIN_ALLOW_THREADS
    err = tsd_conditional_coalescent_variance((tsd_size_t) num_tips, DOUBLES(variance));
    Py_END_ALLOW_THREADS
    if (err != 0) {
        handle_library_error(err);
        goto out;
    }
    ret = (PyObject *) variance;
    variance = NULL;
out:
    Py_XDECREF(variance);
    return ret;
}

static PyObject *
tsdate_count_mutations(PyObject *Py_UNUSED(self), PyObject *args, PyObject *kwds)
{
    PyObject *ret = NULL;
    static char *kwlist[] = { "nodes_sample", "mutations_node", "mutations_position",
        "edges_parent", "edges_child", "edges_left", "edges_right", "indexes_insert",
        "indexes_remove", "sequence_length", "size_biased", NULL };
    PyObject *nodes_sample_obj = NULL;
    PyObject *mutations_node_obj = NULL;
    PyObject *mutations_position_obj = NULL;
    PyObject *edges_parent_obj = NULL;
    PyObject *edges_child_obj = NULL;
    PyObject *edges_left_obj = NULL;
    PyObject *edges_right_obj = NULL;
    PyObject *indexes_insert_obj = NULL;
    PyObject *indexes_remove_obj = NULL;
    PyArrayObject *nodes_sample = NULL;
    PyArrayObject *mutations_node = NULL;
    PyArrayObject *mutations_position = NULL;
    PyArrayObject *edges_parent = NULL;
    PyArrayObject *edges_child = NULL;
    PyArrayObject *edges_left = NULL;
    PyArrayObject *edges_right = NULL;
    PyArrayObject *indexes_insert = NULL;
    PyArrayObject *indexes_remove = NULL;
    PyArrayObject *edges_stats = NULL;
    PyArrayObject *mutations_edge = NULL;
    double sequence_length;
    int size_biased;
    tsd_size_t num_edges, num_mutations;
    npy_intp dims[2];
    int err;

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "OOOOOOOOOdp", kwlist,
            &nodes_sample_obj, &mutations_node_obj, &mutations_position_obj,
            &edges_parent_obj, &edges_child_obj, &edges_left_obj, &edges_right_obj,
            &indexes_insert_obj, &indexes_remove_obj, &sequence_length, &size_biased)) {
        goto out;
    }
    nodes_sample = (PyArrayObject *) PyArray_FROMANY(
        nodes_sample_obj, NPY_BOOL, 1, 1, NPY_ARRAY_IN_ARRAY);
    if (nodes_sample == NULL) {
        goto out;
    }
    mutations_node = (PyArrayObject *) PyArray_FROMANY(
        mutations_node_obj, NPY_INT32, 1, 1, NPY_ARRAY_IN_ARRAY);
    if (mutations_node == NULL) {
        goto out;
    }
    mutations_position = (PyArrayObject *) PyArray_FROMANY(
        mutations_position_obj, NPY_FLOAT64, 1, 1, NPY_ARRAY_IN_ARRAY);
    if (mutations_position == NULL) {
        goto out;
    }
    edges_parent = (PyArrayObject *) PyArray_FROMANY(
        edges_parent_obj, NPY_INT32, 1, 1, NPY_ARRAY_IN_ARRAY);
    if (edges_parent == NULL) {
        goto out;
    }
    edges_child = (PyArrayObject *) PyArray_FROMANY(
        edges_child_obj, NPY_INT32, 1, 1, NPY_ARRAY_IN_ARRAY);
    if (edges_child == NULL) {
        goto out;
    }
    edges_left = (PyArrayObject *) PyArray_FROMANY(
        edges_left_obj, NPY_FLOAT64, 1, 1, NPY_ARRAY_IN_ARRAY);
    if (edges_left == NULL) {
        goto out;
    }
    edges_right = (PyArrayObject *) PyArray_FROMANY(
        edges_right_obj, NPY_FLOAT64, 1, 1, NPY_ARRAY_IN_ARRAY);
    if (edges_right == NULL) {
        goto out;
    }
    indexes_insert = (PyArrayObject *) PyArray_FROMANY(
        indexes_insert_obj, NPY_INT32, 1, 1, NPY_ARRAY_IN_ARRAY);
    if (indexes_insert == NULL) {
        goto out;
    }
    indexes_remove = (PyArrayObject *) PyArray_FROMANY(
        indexes_remove_obj, NPY_INT32, 1, 1, NPY_ARRAY_IN_ARRAY);
    if (indexes_remove == NULL) {
        goto out;
    }
    num_edges = ARRAY_LEN(edges_parent);
    num_mutations = ARRAY_LEN(mutations_node);
    if (ARRAY_LEN(edges_child) != num_edges || ARRAY_LEN(edges_left) != num_edges
        || ARRAY_LEN(edges_right) != num_edges || ARRAY_LEN(indexes_insert) != num_edges
        || ARRAY_LEN(indexes_remove) != num_edges) {
        PyErr_SetString(PyExc_ValueError, "Edge arrays must have the same size");
        goto out;
    }
    if (ARRAY_LEN(mutations_position) != num_mutations) {
        PyErr_SetString(PyExc_ValueError, "Mutation arrays must have the same size");
        goto out;
    }
    dims[0] = (npy_intp) num_edges;
    dims[1] = 2;
    edges_stats = (PyArrayObject *) PyArray_SimpleNew(2, dims, NPY_FLOAT64);
    if (edges_stats == NULL) {
        goto out;
    }
    dims[0] = (npy_intp) num_mutations;
    mutations_edge = (PyArrayObject *) PyArray_SimpleNew(1, dims, NPY_INT32);
    if (mutations_edge == NULL) {
        goto out;
    }
    Py_BEGIN_ALLOW_THREADS
    err = tsd_count_mutations(ARRAY_LEN(nodes_sample), FLAGS(nodes_sample),
        num_mutations, IDS(mutations_node), DOUBLES(mutations_position), num_edges,
        IDS(edges_parent), IDS(edges_child), DOUBLES(edges_left), DOUBLES(edges_right),
        IDS(indexes_insert), IDS(indexes_remove), sequence_length, size_biased,
        DOUBLES(edges_stats), IDS(mutations_edge));
    Py_END_ALLOW_THREADS
    if (err != 0) {
        handle_library_error(err);
        goto out;
    }
    ret = Py_BuildValue("OO", edges_stats, mutations_edge);
out:
    Py_XDECREF(nodes_sample);
    Py_XDECREF(mutations_node);
    Py_XDECREF(mutations_position);
    Py_XDECREF(edges_parent);
    Py_XDECREF(edges_child);
    Py_XDECREF(edges_left);
    Py_XDECREF(edges_right);
    Py_XDECREF(indexes_insert);
    Py_XDECREF(indexes_remove);
    Py_XDECREF(edges_stats);
    Py_XDECREF(mutations_edge);
    return ret;
}

static PyObject *
tsdate_mutational_area(PyObject *Py_UNUSED(self), PyObject *args, PyObject *kwds)
{
    PyObject *ret = NULL;
    static char *kwlist[]
        = { "nodes_time", "edges_likelihood", "edges_parent", "edges_child", NULL };
    PyObject *nodes_time_obj = NULL;
    PyObject *edges_likelihood_obj = NULL;
    PyObject *edges_parent_obj = NULL;
    PyObject *edges_child_obj = NULL;
    PyArrayObject *nodes_time = NULL;
    PyArrayObject *edges_likelihood = NULL;
    PyArrayObject *edges_parent = NULL;
    PyArrayObject *edges_child = NULL;
    PyArrayObject *counts = NULL;
    PyArrayObject *offset = NULL;
    PyArrayObject *duration = NULL;
    PyArrayObject *nodes_index = NULL;
    tsd_epoch_stats_t stats = { 0, NULL, NULL, NULL };
    tsd_size_t num_nodes, num_edges;
    npy_intp dims[1];
    int err;

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "OOOO", kwlist, &nodes_time_obj,
            &edges_likelihood_obj, &edges_parent_obj, &edges_child_obj)) {
        goto out;
    }
    nodes_time = (PyArrayObject *) PyArray_FROMANY(
        nodes_time_obj, NPY_FLOAT64, 1, 1, NPY_ARRAY_IN_ARRAY);
    if (nodes_time == NULL) {
        goto out;
    }
    edges_likelihood = (PyArrayObject *) PyArray_FROMANY(
        edges_likelihood_obj, NPY_FLOAT64, 2, 2, NPY_ARRAY_IN_ARRAY);
    if (edges_likelihood == NULL) {
        goto out;
    }
    edges_parent = (PyArrayObject *) PyArray_FROMANY(
        edges_parent_obj, NPY_INT32, 1, 1, NPY_ARRAY_IN_ARRAY);
    if (edges_parent == NULL) {
        goto out;
    }
    edges_child = (PyArrayObject *) PyArray_FROMANY(
        edges_child_obj, NPY_INT32, 1, 1, NPY_ARRAY_IN_ARRAY);
    if (edges_child == NULL) {
        goto out;
    }
    num_nodes = ARRAY_LEN(nodes_time);
    num_edges = ARRAY_LEN(edges_parent);
    if (ARRAY_LEN(edges_child) != num_edges) {
        PyErr_SetString(PyExc_ValueError, "Edge arrays must have the same size");
        goto out;
    }
    if (ARRAY_LEN(edges_likelihood) != num_edges
        || PyArray_DIM(edges_likelihood, 1) != 2) {
        PyErr_SetString(
            PyExc_ValueError, "Likelihoods must have one row of two columns per edge");
        goto out;
    }
    dims[0] = (npy_intp) num_nodes;
    nodes_index = (PyArrayObject *) PyArray_SimpleNew(1, dims, NPY_INT32);
    if (nodes_index == NULL) {
        goto out;
    }
    Py_BEGIN_ALLOW_THREADS
    err = tsd_mutational_area(num_nodes, DOUBLES(nodes_time), num_edges,
        DOUBLES(edges_likelihood), IDS(edges_parent), IDS(edges_child), &stats,
        IDS(nodes_index));
    Py_END_ALLOW_THREADS
    if (err != 0) {
        handle_library_error(err);
        goto out;
    }
    dims[0] = (npy_intp) stats.num_epochs;
    counts = new_array_from_data(NPY_FLOAT64, 1, dims, stats.counts);
    if (counts == NULL) {
        goto out;
    }
    offset = new_array_from_data(NPY_FLOAT64, 1, dims, stats.offset);
    if (offset == NULL) {
        goto out;
    }
    duration = new_array_from_data(NPY_FLOAT64, 1, dims, stats.duration);
    if (duration == NULL) {
        goto out;
    }
    ret = Py_BuildValue("OOOO", counts, offset, duration, nodes_index);
out:
    tsd_epoch_stats_free(&stats);
    Py_XDECREF(nodes_time);
    Py_XDECREF(edges_likelihood);
    Py_XDECREF(edges_parent);
    Py_XDECREF(edges_child);
    Py_XDECREF(counts);
    Py_XDECREF(offset);
    Py_XDECREF(duration);
    Py_XDECREF(nodes_index);
    return ret;
}

static PyObject *
tsdate_mutational_timescale(PyObject *Py_UNUSED(self), PyObject *args, PyObject *kwds)
{
    PyObject *ret = NULL;
    static char *kwlist[] = { "nodes_time", "edges_likelihood", "nodes_fixed",
        "edges_parent", "edges_child", "max_intervals", NULL };
    PyObject *nodes_time_obj = NULL;
    PyObject *edges_likelihood_obj = NULL;
    PyObject *nodes_fixed_obj = NULL;
    PyObject *edges_parent_obj = NULL;
    PyObject *edges_child_obj = NULL;
    PyArrayObject *nodes_time = NULL;
    PyArrayObject *edges_likelihood = NULL;
    PyArrayObject *nodes_fixed = NULL;
    PyArrayObject *edges_parent = NULL;
    PyArrayObject *edges_child = NULL;
    PyArrayObject *origin = NULL;
    PyArrayObject *adjust = NULL;
    double *origin_data = NULL;
    double *adjust_data = NULL;
    long long max_intervals;
    tsd_size_t num_edges, num_breaks = 0;
    npy_intp dims[1];
    int err;

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "OOOOOL", kwlist, &nodes_time_obj,
            &edges_likelihood_obj, &nodes_fixed_obj, &edges_parent_obj, &edges_child_obj,
            &max_intervals)) {
        goto out;
    }
    nodes_time = (PyArrayObject *) PyArray_FROMANY(
        nodes_time_obj, NPY_FLOAT64, 1, 1, NPY_ARRAY_IN_ARRAY);
    if (nodes_time == NULL) {
        goto out;
    }
    edges_likelihood = (PyArrayObject *) PyArray_FROMANY(
        edges_likelihood_obj, NPY_FLOAT64, 2, 2, NPY_ARRAY_IN_ARRAY);
    if (edges_likelihood == NULL) {
        goto out;
    }
    nodes_fixed = (PyArrayObject *) PyArray_FROMANY(
        nodes_fixed_obj, NPY_BOOL, 1, 1, NPY_ARRAY_IN_ARRAY);
    if (nodes_fixed == NULL) {
        goto out;
    }
    edges_parent = (PyArrayObject *) PyArray_FROMANY(
        edges_parent_obj, NPY_INT32, 1, 1, NPY_ARRAY_IN_ARRAY);
    if (edges_parent == NULL) {
        goto out;
    }
    edges_child = (PyArrayObject *) PyArray_FROMANY(
        edges_child_obj, NPY_INT32, 1, 1, NPY_ARRAY_IN_ARRAY);
    if (edges_child == NULL) {
        goto out;
    }
    num_edges = ARRAY_LEN(edges_parent);
    if (ARRAY_LEN(edges_child) != num_edges) {
        PyErr_SetString(PyExc_ValueError, "Edge arrays must have the same size");
        goto out;
    }
    if (ARRAY_LEN(edges_likelihood) != num_edges
        || PyArray_DIM(edges_likelihood, 1) != 2) {
        PyErr_SetString(
            PyExc_ValueError, "Likelihoods must have one row of two columns per edge");
        goto out;
    }
    if (ARRAY_LEN(nodes_fixed) != ARRAY_LEN(nodes_time)) {
        PyErr_SetString(
            PyExc_ValueError, "nodes_fixed and nodes_time must have the same size");
        goto out;
    }
    Py_BEGIN_ALLOW_THREADS
    err = tsd_mutational_timescale(ARRAY_LEN(nodes_time), DOUBLES(nodes_time), num_edges,
        DOUBLES(edges_likelihood), IDS(edges_parent), IDS(edges_child),
        (int64_t) max_intervals, &num_breaks, &origin_data, &adjust_data);
    Py_END_ALLOW_THREADS
    if (err != 0) {
        handle_library_error(err);
        goto out;
    }
    dims[0] = (npy_intp) num_breaks;
    origin = new_array_from_data(NPY_FLOAT64, 1, dims, origin_data);
    if (origin == NULL) {
        goto out;
    }
    adjust = new_array_from_data(NPY_FLOAT64, 1, dims, adjust_data);
    if (adjust == NULL) {
        goto out;
    }
    ret = Py_BuildValue("OO", origin, adjust);
out:
    tsd_safe_free(origin_data);
    tsd_safe_free(adjust_data);
    Py_XDECREF(nodes_time);
    Py_XDECREF(edges_likelihood);
    Py_XDECREF(nodes_fixed);
    Py_XDECREF(edges_parent);
    Py_XDECREF(edges_child);
    Py_XDECREF(origin);
    Py_XDECREF(adjust);
    return ret;
}

static PyObject *
tsdate_piecewise_scale_point_estimate(
    PyObject *Py_UNUSED(self), PyObject *args, PyObject *kwds)
{
    PyObject *ret = NULL;
    static char *kwlist[] = { "point_estimate", "point_fixed", "original_breaks",
        "rescaled_breaks", NULL };
    PyObject *point_estimate_obj = NULL;
    PyObject *point_fixed_obj = NULL;
    PyObject *original_breaks_obj = NULL;
    PyObject *rescaled_breaks_obj = NULL;
    PyArrayObject *point_estimate = NULL;
    PyArrayObject *point_fixed = NULL;
    PyArrayObject *original_breaks = NULL;
    PyArrayObject *rescaled_breaks = NULL;
    PyArrayObject *rescaled = NULL;
    tsd_size_t num_points, num_breaks;
    npy_intp dims[1];
    int err;

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "OOOO", kwlist, &point_estimate_obj,
            &point_fixed_obj, &original_breaks_obj, &rescaled_breaks_obj)) {
        goto out;
    }
    point_estimate = (PyArrayObject *) PyArray_FROMANY(
        point_estimate_obj, NPY_FLOAT64, 1, 1, NPY_ARRAY_IN_ARRAY);
    if (point_estimate == NULL) {
        goto out;
    }
    point_fixed = (PyArrayObject *) PyArray_FROMANY(
        point_fixed_obj, NPY_BOOL, 1, 1, NPY_ARRAY_IN_ARRAY);
    if (point_fixed == NULL) {
        goto out;
    }
    original_breaks = (PyArrayObject *) PyArray_FROMANY(
        original_breaks_obj, NPY_FLOAT64, 1, 1, NPY_ARRAY_IN_ARRAY);
    if (original_breaks == NULL) {
        goto out;
    }
    rescaled_breaks = (PyArrayObject *) PyArray_FROMANY(
        rescaled_breaks_obj, NPY_FLOAT64, 1, 1, NPY_ARRAY_IN_ARRAY);
    if (rescaled_breaks == NULL) {
        goto out;
    }
    num_points = ARRAY_LEN(point_estimate);
    num_breaks = ARRAY_LEN(original_breaks);
    if (ARRAY_LEN(point_fixed) != num_points) {
        PyErr_SetString(
            PyExc_ValueError, "point_fixed and point_estimate must have the same size");
        goto out;
    }
    if (ARRAY_LEN(rescaled_breaks) != num_breaks) {
        PyErr_SetString(PyExc_ValueError, "Breakpoints must be of the same size");
        goto out;
    }
    dims[0] = (npy_intp) num_points;
    rescaled = (PyArrayObject *) PyArray_SimpleNew(1, dims, NPY_FLOAT64);
    if (rescaled == NULL) {
        goto out;
    }
    Py_BEGIN_ALLOW_THREADS
    err = tsd_piecewise_scale_point_estimate(num_points, DOUBLES(point_estimate),
        FLAGS(point_fixed), num_breaks, DOUBLES(original_breaks),
        DOUBLES(rescaled_breaks), DOUBLES(rescaled));
    Py_END_ALLOW_THREADS
    if (err != 0) {
        handle_library_error(err);
        goto out;
    }
    ret = (PyObject *) rescaled;
    rescaled = NULL;
out:
    Py_XDECREF(point_estimate);
    Py_XDECREF(point_fixed);
    Py_XDECREF(original_breaks);
    Py_XDECREF(rescaled_breaks);
    Py_XDECREF(rescaled);
    return ret;
}

static PyObject *
tsdate_piecewise_scale_posterior(
    PyObject *Py_UNUSED(self), PyObject *args, PyObject *kwds)
{
    PyObject *ret = NULL;
    static char *kwlist[] = { "posteriors", "posteriors_fixed", "original_breaks",
        "rescaled_breaks", "quantile_width", "max_shape", NULL };
    PyObject *posteriors_obj = NULL;
    PyObject *posteriors_fixed_obj = NULL;
    PyObject *original_breaks_obj = NULL;
    PyObject *rescaled_breaks_obj = NULL;
    PyArrayObject *posteriors = NULL;
    PyArrayObject *posteriors_fixed = NULL;
    PyArrayObject *original_breaks = NULL;
    PyArrayObject *rescaled_breaks = NULL;
    PyArrayObject *new_posteriors = NULL;
    double quantile_width, max_shape;
    tsd_size_t num_posteriors, num_breaks;
    npy_intp dims[2];
    int err;

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "OOOOdd", kwlist, &posteriors_obj,
            &posteriors_fixed_obj, &original_breaks_obj, &rescaled_breaks_obj,
            &quantile_width, &max_shape)) {
        goto out;
    }
    posteriors = (PyArrayObject *) PyArray_FROMANY(
        posteriors_obj, NPY_FLOAT64, 2, 2, NPY_ARRAY_IN_ARRAY);
    if (posteriors == NULL) {
        goto out;
    }
    posteriors_fixed = (PyArrayObject *) PyArray_FROMANY(
        posteriors_fixed_obj, NPY_BOOL, 1, 1, NPY_ARRAY_IN_ARRAY);
    if (posteriors_fixed == NULL) {
        goto out;
    }
    original_breaks = (PyArrayObject *) PyArray_FROMANY(
        original_breaks_obj, NPY_FLOAT64, 1, 1, NPY_ARRAY_IN_ARRAY);
    if (original_breaks == NULL) {
        goto out;
    }
    rescaled_breaks = (PyArrayObject *) PyArray_FROMANY(
        rescaled_breaks_obj, NPY_FLOAT64, 1, 1, NPY_ARRAY_IN_ARRAY);
    if (rescaled_breaks == NULL) {
        goto out;
    }
    num_posteriors = ARRAY_LEN(posteriors);
    num_breaks = ARRAY_LEN(original_breaks);
    if (PyArray_DIM(posteriors, 1) != 2
        || ARRAY_LEN(posteriors_fixed) != num_posteriors) {
        PyErr_SetString(PyExc_ValueError,
            "Posteriors must have two columns and one row per fixed flag");
        goto out;
    }
    if (ARRAY_LEN(rescaled_breaks) != num_breaks) {
        PyErr_SetString(PyExc_ValueError, "Breakpoints must be of the same size");
        goto out;
    }
    dims[0] = (npy_intp) num_posteriors;
    dims[1] = 2;
    new_posteriors = (PyArrayObject *) PyArray_SimpleNew(2, dims, NPY_FLOAT64);
    if (new_posteriors == NULL) {
        goto out;
    }
    Py_BEGIN_ALLOW_THREADS
    err = tsd_piecewise_scale_posterior(num_posteriors, DOUBLES(posteriors),
        FLAGS(posteriors_fixed), num_breaks, DOUBLES(original_breaks),
        DOUBLES(rescaled_breaks), quantile_width, max_shape, DOUBLES(new_posteriors));
    Py_END_ALLOW_THREADS
    if (err != 0) {
        handle_library_error(err);
        goto out;
    }
    ret = (PyObject *) new_posteriors;
    new_posteriors = NULL;
out:
    Py_XDECREF(posteriors);
    Py_XDECREF(posteriors_fixed);
    Py_XDECREF(original_breaks);
    Py_XDECREF(rescaled_breaks);
    Py_XDECREF(new_posteriors);
    return ret;
}

static PyObject *
tsdate_constraints_are_consistent(
    PyObject *Py_UNUSED(self), PyObject *args, PyObject *kwds)
{
    PyObject *ret = NULL;
    static char *kwlist[] = { "constraints", "edges_parent", "edges_child", NULL };
    PyObject *constraints_obj = NULL;
    PyObject *edges_parent_obj = NULL;
    PyObject *edges_child_obj = NULL;
    PyArrayObject *constraints = NULL;
    PyArrayObject *edges_parent = NULL;
    PyArrayObject *edges_child = NULL;
    tsd_size_t num_edges;
    int consistent = 0;
    int err;

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "OOO", kwlist, &constraints_obj,
            &edges_parent_obj, &edges_child_obj)) {
        goto out;
    }
    constraints = (PyArrayObject *) PyArray_FROMANY(
        constraints_obj, NPY_FLOAT64, 2, 2, NPY_ARRAY_IN_ARRAY);
    if (constraints == NULL) {
        goto out;
    }
    edges_parent = (PyArrayObject *) PyArray_FROMANY(
        edges_parent_obj, NPY_INT32, 1, 1, NPY_ARRAY_IN_ARRAY);
    if (edges_parent == NULL) {
        goto out;
    }
    edges_child = (PyArrayObject *) PyArray_FROMANY(
        edges_child_obj, NPY_INT32, 1, 1, NPY_ARRAY_IN_ARRAY);
    if (edges_child == NULL) {
        goto out;
    }
    num_edges = ARRAY_LEN(edges_parent);
    if (PyArray_DIM(constraints, 1) != 2) {
        PyErr_SetString(PyExc_ValueError, "constraints must have two columns");
        goto out;
    }
    if (ARRAY_LEN(edges_child) != num_edges) {
        PyErr_SetString(PyExc_ValueError, "Edge arrays must have the same size");
        goto out;
    }
    Py_BEGIN_ALLOW_THREADS
    err = tsd_constraints_are_consistent(ARRAY_LEN(constraints), num_edges,
        DOUBLES(constraints), IDS(edges_parent), IDS(edges_child), &consistent);
    Py_END_ALLOW_THREADS
    if (err != 0) {
        handle_library_error(err);
        goto out;
    }
    ret = PyBool_FromLong(consistent);
out:
    Py_XDECREF(constraints);
    Py_XDECREF(edges_parent);
    Py_XDECREF(edges_child);
    return ret;
}

/*===================================================================
 * ExpectationPropagation
 *===================================================================
 */

typedef struct {
    PyObject_HEAD
    tsd_ep_t *ep;
} ExpectationPropagation;

static int
ExpectationPropagation_check_state(ExpectationPropagation *self)
{
    int ret = 0;

    if (self->ep == NULL) {
        PyErr_SetString(PyExc_SystemError, "ExpectationPropagation not initialised");
        ret = -1;
    }
    return ret;
}

static void
ExpectationPropagation_dealloc(ExpectationPropagation *self)
{
    if (self->ep != NULL) {
        tsd_ep_free(self->ep);
        PyMem_Free(self->ep);
        self->ep = NULL;
    }
    Py_TYPE(self)->tp_free((PyObject *) self);
}

static int
ExpectationPropagation_init(ExpectationPropagation *self, PyObject *args, PyObject *kwds)
{
    int ret = -1;
    static char *kwlist[] = { "constraints", "edges_parent", "edges_child",
        "edges_likelihood", "blocks_left", "blocks_right", "blocks_likelihood", NULL };
    PyObject *constraints_obj = NULL;
    PyObject *edges_parent_obj = NULL;
    PyObject *edges_child_obj = NULL;
    PyObject *edges_likelihood_obj = NULL;
    PyObject *blocks_left_obj = NULL;
    PyObject *blocks_right_obj = NULL;
    PyObject *blocks_likelihood_obj = NULL;
    PyArrayObject *constraints = NULL;
    PyArrayObject *edges_parent = NULL;
    PyArrayObject *edges_child = NULL;
    PyArrayObject *edges_likelihood = NULL;
    PyArrayObject *blocks_left = NULL;
    PyArrayObject *blocks_right = NULL;
    PyArrayObject *blocks_likelihood = NULL;
    tsd_size_t num_edges, num_blocks;
    int err;

    if (self->ep != NULL) {
        tsd_ep_free(self->ep);
        PyMem_Free(self->ep);
        self->ep = NULL;
    }
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "OOOOOOO", kwlist, &constraints_obj,
            &edges_parent_obj, &edges_child_obj, &edges_likelihood_obj, &blocks_left_obj,
            &blocks_right_obj, &blocks_likelihood_obj)) {
        goto out;
    }
    constraints = (PyArrayObject *) PyArray_FROMANY(
        constraints_obj, NPY_FLOAT64, 2, 2, NPY_ARRAY_IN_ARRAY);
    if (constraints == NULL) {
        goto out;
    }
    edges_parent = (PyArrayObject *) PyArray_FROMANY(
        edges_parent_obj, NPY_INT32, 1, 1, NPY_ARRAY_IN_ARRAY);
    if (edges_parent == NULL) {
        goto out;
    }
    edges_child = (PyArrayObject *) PyArray_FROMANY(
        edges_child_obj, NPY_INT32, 1, 1, NPY_ARRAY_IN_ARRAY);
    if (edges_child == NULL) {
        goto out;
    }
    edges_likelihood = (PyArrayObject *) PyArray_FROMANY(
        edges_likelihood_obj, NPY_FLOAT64, 2, 2, NPY_ARRAY_IN_ARRAY);
    if (edges_likelihood == NULL) {
        goto out;
    }
    blocks_left = (PyArrayObject *) PyArray_FROMANY(
        blocks_left_obj, NPY_INT32, 1, 1, NPY_ARRAY_IN_ARRAY);
    if (blocks_left == NULL) {
        goto out;
    }
    blocks_right = (PyArrayObject *) PyArray_FROMANY(
        blocks_right_obj, NPY_INT32, 1, 1, NPY_ARRAY_IN_ARRAY);
    if (blocks_right == NULL) {
        goto out;
    }
    blocks_likelihood = (PyArrayObject *) PyArray_FROMANY(
        blocks_likelihood_obj, NPY_FLOAT64, 2, 2, NPY_ARRAY_IN_ARRAY);
    if (blocks_likelihood == NULL) {
        goto out;
    }
    num_edges = ARRAY_LEN(edges_parent);
    num_blocks = ARRAY_LEN(blocks_left);
    if (PyArray_DIM(constraints, 1) != 2 || PyArray_DIM(edges_likelihood, 1) != 2
        || PyArray_DIM(blocks_likelihood, 1) != 2) {
        PyErr_SetString(PyExc_ValueError,
            "constraints, edges_likelihood and blocks_likelihood must have two columns");
        goto out;
    }
    if (ARRAY_LEN(edges_child) != num_edges
        || ARRAY_LEN(edges_likelihood) != num_edges) {
        PyErr_SetString(PyExc_ValueError, "Edge arrays must have the same size");
        goto out;
    }
    if (ARRAY_LEN(blocks_right) != num_blocks
        || ARRAY_LEN(blocks_likelihood) != num_blocks) {
        PyErr_SetString(PyExc_ValueError, "Block arrays must have the same size");
        goto out;
    }
    self->ep = PyMem_Malloc(sizeof(*self->ep));
    if (self->ep == NULL) {
        PyErr_NoMemory();
        goto out;
    }
    err = tsd_ep_init(self->ep, ARRAY_LEN(constraints), num_edges, num_blocks,
        DOUBLES(constraints), IDS(edges_parent), IDS(edges_child),
        DOUBLES(edges_likelihood), IDS(blocks_left), IDS(blocks_right),
        DOUBLES(blocks_likelihood));
    if (err != 0) {
        handle_library_error(err);
        tsd_ep_free(self->ep);
        PyMem_Free(self->ep);
        self->ep = NULL;
        goto out;
    }
    ret = 0;
out:
    Py_XDECREF(constraints);
    Py_XDECREF(edges_parent);
    Py_XDECREF(edges_child);
    Py_XDECREF(edges_likelihood);
    Py_XDECREF(blocks_left);
    Py_XDECREF(blocks_right);
    Py_XDECREF(blocks_likelihood);
    return ret;
}

static PyObject *
ExpectationPropagation_propagate_likelihood(
    ExpectationPropagation *self, PyObject *args, PyObject *kwds)
{
    PyObject *ret = NULL;
    static char *kwlist[] = { "order", "unphased", "max_shape", "min_step", NULL };
    PyObject *order_obj = NULL;
    PyArrayObject *order = NULL;
    int unphased;
    double max_shape, min_step;
    int err;

    if (ExpectationPropagation_check_state(self) != 0) {
        goto out;
    }
    if (!PyArg_ParseTupleAndKeywords(
            args, kwds, "Opdd", kwlist, &order_obj, &unphased, &max_shape, &min_step)) {
        goto out;
    }
    order = (PyArrayObject *) PyArray_FROMANY(
        order_obj, NPY_INT32, 1, 1, NPY_ARRAY_IN_ARRAY);
    if (order == NULL) {
        goto out;
    }
    Py_BEGIN_ALLOW_THREADS
    err = tsd_ep_propagate_likelihood(
        self->ep, ARRAY_LEN(order), IDS(order), unphased, max_shape, min_step);
    Py_END_ALLOW_THREADS
    if (err != 0) {
        handle_library_error(err);
        goto out;
    }
    ret = Py_BuildValue("");
out:
    Py_XDECREF(order);
    return ret;
}

static PyObject *
ExpectationPropagation_propagate_prior(
    ExpectationPropagation *self, PyObject *args, PyObject *kwds)
{
    PyObject *ret = NULL;
    static char *kwlist[] = { "free", "max_shape", "em_maxitt", "em_reltol", NULL };
    PyObject *free_obj = NULL;
    PyArrayObject *free = NULL;
    double max_shape, em_reltol;
    long long em_maxitt;
    int err;

    if (ExpectationPropagation_check_state(self) != 0) {
        goto out;
    }
    if (!PyArg_ParseTupleAndKeywords(
            args, kwds, "OdLd", kwlist, &free_obj, &max_shape, &em_maxitt, &em_reltol)) {
        goto out;
    }
    free = (PyArrayObject *) PyArray_FROMANY(
        free_obj, NPY_BOOL, 1, 1, NPY_ARRAY_IN_ARRAY);
    if (free == NULL) {
        goto out;
    }
    if (ARRAY_LEN(free) != self->ep->num_nodes) {
        PyErr_SetString(PyExc_ValueError, "free must have one entry per node");
        goto out;
    }
    Py_BEGIN_ALLOW_THREADS
    err = tsd_ep_propagate_prior(
        self->ep, FLAGS(free), max_shape, (int64_t) em_maxitt, em_reltol);
    Py_END_ALLOW_THREADS
    if (err != 0) {
        handle_library_error(err);
        goto out;
    }
    ret = Py_BuildValue("");
out:
    Py_XDECREF(free);
    return ret;
}

static PyObject *
ExpectationPropagation_propagate_mutations(
    ExpectationPropagation *self, PyObject *args, PyObject *kwds)
{
    PyObject *ret = NULL;
    static char *kwlist[] = { "mutations_edge", "unphased", NULL };
    PyObject *mutations_edge_obj = NULL;
    PyArrayObject *mutations_edge = NULL;
    PyArrayObject *mutations_posterior = NULL;
    PyArrayObject *mutations_phase = NULL;
    int unphased;
    npy_intp dims[2];
    int err;

    if (ExpectationPropagation_check_state(self) != 0) {
        goto out;
    }
    if (!PyArg_ParseTupleAndKeywords(
            args, kwds, "Op", kwlist, &mutations_edge_obj, &unphased)) {
        goto out;
    }
    mutations_edge = (PyArrayObject *) PyArray_FROMANY(
        mutations_edge_obj, NPY_INT32, 1, 1, NPY_ARRAY_IN_ARRAY);
    if (mutations_edge == NULL) {
        goto out;
    }
    dims[0] = PyArray_DIM(mutations_edge, 0);
    dims[1] = 2;
    mutations_posterior = (PyArrayObject *) PyArray_SimpleNew(2, dims, NPY_FLOAT64);
    if (mutations_posterior == NULL) {
        goto out;
    }
    mutations_phase = (PyArrayObject *) PyArray_SimpleNew(1, dims, NPY_FLOAT64);
    if (mutations_phase == NULL) {
        goto out;
    }
    Py_BEGIN_ALLOW_THREADS
    err = tsd_ep_propagate_mutations(self->ep, ARRAY_LEN(mutations_edge),
        IDS(mutations_edge), unphased, DOUBLES(mutations_posterior),
        DOUBLES(mutations_phase));
    Py_END_ALLOW_THREADS
    if (err != 0) {
        handle_library_error(err);
        goto out;
    }
    ret = Py_BuildValue("OO", mutations_posterior, mutations_phase);
out:
    Py_XDECREF(mutations_edge);
    Py_XDECREF(mutations_posterior);
    Py_XDECREF(mutations_phase);
    return ret;
}

static PyObject *
ExpectationPropagation_rescale_factors(
    ExpectationPropagation *self, PyObject *Py_UNUSED(args))
{
    PyObject *ret = NULL;

    if (ExpectationPropagation_check_state(self) != 0) {
        goto out;
    }
    tsd_ep_rescale_factors(self->ep);
    ret = Py_BuildValue("");
out:
    return ret;
}

static PyObject *
ExpectationPropagation_assemble_factors(
    ExpectationPropagation *self, PyObject *Py_UNUSED(args))
{
    PyObject *ret = NULL;
    PyArrayObject *posterior = NULL;
    npy_intp dims[2];

    if (ExpectationPropagation_check_state(self) != 0) {
        goto out;
    }
    dims[0] = (npy_intp) self->ep->num_nodes;
    dims[1] = 2;
    posterior = (PyArrayObject *) PyArray_SimpleNew(2, dims, NPY_FLOAT64);
    if (posterior == NULL) {
        goto out;
    }
    tsd_ep_assemble_factors(self->ep, DOUBLES(posterior));
    ret = (PyObject *) posterior;
    posterior = NULL;
out:
    Py_XDECREF(posterior);
    return ret;
}

static PyObject *
ExpectationPropagation_get_posterior(
    ExpectationPropagation *self, void *Py_UNUSED(closure))
{
    PyObject *ret = NULL;
    npy_intp dims[2];

    if (ExpectationPropagation_check_state(self) != 0) {
        goto out;
    }
    dims[0] = (npy_intp) self->ep->num_nodes;
    dims[1] = 2;
    ret = (PyObject *) new_array_from_data(NPY_FLOAT64, 2, dims, self->ep->posterior);
out:
    return ret;
}

static PyObject *
ExpectationPropagation_get_edges_lognorm(
    ExpectationPropagation *self, void *Py_UNUSED(closure))
{
    PyObject *ret = NULL;
    npy_intp dims[1];

    if (ExpectationPropagation_check_state(self) != 0) {
        goto out;
    }
    dims[0] = (npy_intp) self->ep->num_edges;
    ret = (PyObject *) new_array_from_data(
        NPY_FLOAT64, 1, dims, self->ep->edges_lognorm);
out:
    return ret;
}

static PyObject *
ExpectationPropagation_get_blocks_lognorm(
    ExpectationPropagation *self, void *Py_UNUSED(closure))
{
    PyObject *ret = NULL;
    npy_intp dims[1];

    if (ExpectationPropagation_check_state(self) != 0) {
        goto out;
    }
    dims[0] = (npy_intp) self->ep->num_blocks;
    ret = (PyObject *) new_array_from_data(
        NPY_FLOAT64, 1, dims, self->ep->blocks_lognorm);
out:
    return ret;
}

static PyMethodDef ExpectationPropagation_methods[] = {
    { "propagate_likelihood",
        (PyCFunction) (void (*)(void)) ExpectationPropagation_propagate_likelihood,
        METH_VARARGS | METH_KEYWORDS,
        "Update the factors of the edges (blocks, if unphased) in the given order." },
    { "propagate_prior",
        (PyCFunction) (void (*)(void)) ExpectationPropagation_propagate_prior,
        METH_VARARGS | METH_KEYWORDS,
        "Fit an exponential prior to the free nodes and update their factors." },
    { "propagate_mutations",
        (PyCFunction) (void (*)(void)) ExpectationPropagation_propagate_mutations,
        METH_VARARGS | METH_KEYWORDS,
        "The posteriors and phases of mutations above the given edges (blocks)." },
    { "rescale_factors", (PyCFunction) ExpectationPropagation_rescale_factors,
        METH_NOARGS, "Multiply the factors by the scales of their nodes." },
    { "assemble_factors", (PyCFunction) ExpectationPropagation_assemble_factors,
        METH_NOARGS, "The posteriors implied by the factors." },
    { NULL } /* Sentinel */
};

static PyGetSetDef ExpectationPropagation_getsetters[] = {
    { "posterior", (getter) ExpectationPropagation_get_posterior, NULL,
        "A copy of the posteriors of the nodes (natural parameters).", NULL },
    { "edges_lognorm", (getter) ExpectationPropagation_get_edges_lognorm, NULL,
        "A copy of the log normalising constants of the edges.", NULL },
    { "blocks_lognorm", (getter) ExpectationPropagation_get_blocks_lognorm, NULL,
        "A copy of the log normalising constants of the blocks.", NULL },
    { NULL } /* Sentinel */
};

static PyTypeObject ExpectationPropagationType = {
    // clang-format off
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "_tsdate.ExpectationPropagation",
    // clang-format on
    .tp_basicsize = sizeof(ExpectationPropagation),
    .tp_dealloc = (destructor) ExpectationPropagation_dealloc,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_doc = "The state of expectation propagation.",
    .tp_methods = ExpectationPropagation_methods,
    .tp_getset = ExpectationPropagation_getsetters,
    .tp_init = (initproc) ExpectationPropagation_init,
    .tp_new = PyType_GenericNew,
};

#define ARRAY_DIM(array, axis) ((tsd_size_t) PyArray_DIM(array, axis))

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

/*
 * Rescaling functions (lib/rescaling.c). Array sizes and shapes are checked here and
 * raise ValueError.
 */

/* A one-dimensional array of tsd_id_t (int32), copied from `data`. */
static PyObject *
new_id_array(tsd_size_t size, const tsd_id_t *data)
{
    npy_intp dim = (npy_intp) size;
    PyObject *ret = PyArray_SimpleNew(1, &dim, NPY_INT32);

    if (ret != NULL && size > 0) {
        memcpy(PyArray_DATA((PyArrayObject *) ret), data, size * sizeof(*data));
    }
    return ret;
}

/* A one-dimensional array of doubles, copied from `data`. */
static PyObject *
new_double_array(tsd_size_t size, const double *data)
{
    npy_intp dim = (npy_intp) size;
    PyObject *ret = PyArray_SimpleNew(1, &dim, NPY_FLOAT64);

    if (ret != NULL && size > 0) {
        memcpy(PyArray_DATA((PyArrayObject *) ret), data, size * sizeof(*data));
    }
    return ret;
}

/* A C-contiguous array of the given type and dimension from `obj`, or NULL. */
static PyArrayObject *
get_array(PyObject *obj, int typenum, int ndim)
{
    return (PyArrayObject *) PyArray_FROMANY(
        obj, typenum, ndim, ndim, NPY_ARRAY_IN_ARRAY);
}

/* Raise ValueError and return -1 unless `ok`. */
static int
check_sizes(int ok, const char *message)
{
    if (!ok) {
        PyErr_SetString(PyExc_ValueError, message);
        return -1;
    }
    return 0;
}

static PyObject *
tsdate_fixed_changepoints(PyObject *Py_UNUSED(self), PyObject *args)
{
    PyObject *ret = NULL;
    PyObject *counts_obj = NULL;
    PyArrayObject *counts = NULL;
    long long epochs;
    tsd_id_t *breaks = NULL;
    tsd_size_t num_breaks = 0;
    int err;

    if (!PyArg_ParseTuple(args, "OL", &counts_obj, &epochs)) {
        goto out;
    }
    counts = get_array(counts_obj, NPY_FLOAT64, 1);
    if (counts == NULL) {
        goto out;
    }
    Py_BEGIN_ALLOW_THREADS
    err = tsd_fixed_changepoints(
        ARRAY_LEN(counts), DOUBLES(counts), (int64_t) epochs, &breaks, &num_breaks);
    Py_END_ALLOW_THREADS
    if (err != 0) {
        handle_library_error(err);
        goto out;
    }
    ret = new_id_array(num_breaks, breaks);
out:
    tsd_safe_free(breaks);
    Py_XDECREF(counts);
    return ret;
}

static PyObject *
tsdate_poisson_changepoints(PyObject *Py_UNUSED(self), PyObject *args)
{
    PyObject *ret = NULL;
    PyObject *counts_obj = NULL;
    PyObject *offset_obj = NULL;
    PyArrayObject *counts = NULL;
    PyArrayObject *offset = NULL;
    double penalty, min_counts, min_offset;
    tsd_id_t *breaks = NULL;
    tsd_size_t num_breaks = 0;
    int err;

    if (!PyArg_ParseTuple(args, "OOddd", &counts_obj, &offset_obj, &penalty, &min_counts,
            &min_offset)) {
        goto out;
    }
    counts = get_array(counts_obj, NPY_FLOAT64, 1);
    offset = get_array(offset_obj, NPY_FLOAT64, 1);
    if (counts == NULL || offset == NULL) {
        goto out;
    }
    if (check_sizes(ARRAY_LEN(counts) == ARRAY_LEN(offset),
            "counts and offset must have the same size")
        != 0) {
        goto out;
    }
    Py_BEGIN_ALLOW_THREADS
    err = tsd_poisson_changepoints(ARRAY_LEN(counts), DOUBLES(counts), DOUBLES(offset),
        penalty, min_counts, min_offset, &breaks, &num_breaks);
    Py_END_ALLOW_THREADS
    if (err != 0) {
        handle_library_error(err);
        goto out;
    }
    ret = new_id_array(num_breaks, breaks);
out:
    tsd_safe_free(breaks);
    Py_XDECREF(counts);
    Py_XDECREF(offset);
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
    { "marginalize_over_ancestors", (PyCFunction) tsdate_marginalize_over_ancestors,
        METH_VARARGS, "Integrate an expectation over counts of extant ancestors." },
    { "fixed_changepoints", (PyCFunction) tsdate_fixed_changepoints, METH_VARARGS,
        "Breakpoints dividing counts roughly equally between epochs." },
    { "poisson_changepoints", (PyCFunction) tsdate_poisson_changepoints, METH_VARARGS,
        "Changepoints of a Poisson rate under a linear penalty." },
    { "block_singletons", (PyCFunction) (void (*)(void)) tsdate_block_singletons,
        METH_VARARGS | METH_KEYWORDS,
        "Group the singletons of unphased individuals into blocks." },
    { "reallocate_unphased", (PyCFunction) (void (*)(void)) tsdate_reallocate_unphased,
        METH_VARARGS | METH_KEYWORDS,
        "The edge likelihoods with those of unphased blocks reallocated by phase." },
    { "mutation_frequency", (PyCFunction) (void (*)(void)) tsdate_mutation_frequency,
        METH_VARARGS | METH_KEYWORDS,
        "The number of members of each sample set below each mutation." },
    { "split_disjoint_nodes", (PyCFunction) (void (*)(void)) tsdate_split_disjoint_nodes,
        METH_VARARGS | METH_KEYWORDS,
        "Split disconnected regions of nodes into separate nodes." },
    { "relabel_mutations_node",
        (PyCFunction) (void (*)(void)) tsdate_relabel_mutations_node,
        METH_VARARGS | METH_KEYWORDS,
        "The new node of each mutation, after splitting nodes." },
    { "constrain_ages", (PyCFunction) (void (*)(void)) tsdate_constrain_ages,
        METH_VARARGS | METH_KEYWORDS,
        "Constrain node ages to be consistent with the edges." },
    { "contains_unary_nodes", (PyCFunction) (void (*)(void)) tsdate_contains_unary_nodes,
        METH_VARARGS | METH_KEYWORDS,
        "Whether any unmasked node has a single child in some tree." },
    { "conditional_coalescent_variance",
        (PyCFunction) (void (*)(void)) tsdate_conditional_coalescent_variance,
        METH_VARARGS | METH_KEYWORDS,
        "Variance of node age given the number of descendant leaves." },
    { "count_mutations", (PyCFunction) (void (*)(void)) tsdate_count_mutations,
        METH_VARARGS | METH_KEYWORDS,
        "Mutation counts and spans per edge, and the edge of each mutation." },
    { "mutational_area", (PyCFunction) (void (*)(void)) tsdate_mutational_area,
        METH_VARARGS | METH_KEYWORDS,
        "Mutation counts and area per inter-node interval." },
    { "mutational_timescale", (PyCFunction) (void (*)(void)) tsdate_mutational_timescale,
        METH_VARARGS | METH_KEYWORDS,
        "Breakpoints for rescaling time so that the mutation rate is constant." },
    { "piecewise_scale_point_estimate",
        (PyCFunction) (void (*)(void)) tsdate_piecewise_scale_point_estimate,
        METH_VARARGS | METH_KEYWORDS, "Rescale point estimates of node ages." },
    { "piecewise_scale_posterior",
        (PyCFunction) (void (*)(void)) tsdate_piecewise_scale_posterior,
        METH_VARARGS | METH_KEYWORDS, "Rescale gamma posteriors of node ages." },
    { "constraints_are_consistent",
        (PyCFunction) (void (*)(void)) tsdate_constraints_are_consistent,
        METH_VARARGS | METH_KEYWORDS,
        "Whether node age constraints are consistent with the edges." },
    { NULL } /* Sentinel */
};

static struct PyModuleDef tsdatemodule = { PyModuleDef_HEAD_INIT, "_tsdate", MODULE_DOC,
    -1, tsdate_methods, NULL, NULL, NULL, NULL };

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

    if (PyType_Ready(&ExpectationPropagationType) < 0) {
        Py_DECREF(module);
        return NULL;
    }
    Py_INCREF(&ExpectationPropagationType);
    if (PyModule_AddObject(
            module, "ExpectationPropagation", (PyObject *) &ExpectationPropagationType)
        != 0) {
        Py_DECREF(&ExpectationPropagationType);
        Py_DECREF(module);
        return NULL;
    }

    if (add_scalar_functions(module) != 0) {
        Py_DECREF(module);
        return NULL;
    }

    return module;
}
