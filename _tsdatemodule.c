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

static PyMethodDef tsdate_methods[] = {
    { "strerror", (PyCFunction) tsdate_strerror, METH_VARARGS,
        "Return the message for a library error code." },
    { "raise_library_error", (PyCFunction) tsdate_raise_library_error, METH_VARARGS,
        "Raise the Python exception corresponding to a library error code." },
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
