# MIT License
#
# Copyright (c) 2020 University of Oxford
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""
Versions of important dependencies and environment.
"""

import json
import platform

import tskit

__version__ = "undefined"
try:
    from . import _version

    __version__ = _version.version
except ImportError:  # pragma: nocover
    try:
        from setuptools_scm import get_version

        __version__ = get_version(root="..", relative_to=__file__)
    except ImportError:
        pass


def get_environment():
    """
    Returns a dictionary encoding provenance information
    """
    env = {
        "os": {
            "system": platform.system(),
            "node": platform.node(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
        },
        "python": {
            "implementation": platform.python_implementation(),
            "version": platform.python_version(),
        },
    }
    libs = {"tskit": {"version": tskit.__version__}}
    env["libraries"] = libs
    return env


def get_provenance_dict(command, **kwargs):
    """
    Returns a dictionary encoding an execution of tsdate conforming to the
    tskit provenance schema.
    """
    if command is None:
        raise ValueError("`command` cannot be None")
    parameters = dict(kwargs)
    parameters["command"] = command
    document = {
        "schema_version": "1.0.0",
        "software": {"name": "tsdate", "version": __version__},
        "parameters": parameters,
        "environment": get_environment(),
    }
    return document


def record_provenance(tables, command=None, **kwargs):
    """
    Adds provenance information to this table collection using the
    tskit provenances schema.
    """
    record = get_provenance_dict(command=command, **kwargs)
    tables.provenances.add_row(record=json.dumps(record))
