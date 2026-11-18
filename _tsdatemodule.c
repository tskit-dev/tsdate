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

static PyMethodDef tsdate_methods[] = {
    { "strerror", (PyCFunction) tsdate_strerror, METH_VARARGS,
        "Return the message for a library error code." },
    { "raise_library_error", (PyCFunction) tsdate_raise_library_error, METH_VARARGS,
        "Raise the Python exception corresponding to a library error code." },
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

    return module;
}
