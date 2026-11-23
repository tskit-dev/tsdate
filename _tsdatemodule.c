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

static PyMethodDef tsdate_methods[] = {
    { "strerror", (PyCFunction) tsdate_strerror, METH_VARARGS,
        "Return the message for a library error code." },
    { "raise_library_error", (PyCFunction) tsdate_raise_library_error, METH_VARARGS,
        "Raise the Python exception corresponding to a library error code." },
    { "average_gammas", (PyCFunction) tsdate_average_gammas, METH_VARARGS,
        "Natural parameters of the gamma averaging the sufficient statistics of an "
        "array of gammas, given as natural parameters." },
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
