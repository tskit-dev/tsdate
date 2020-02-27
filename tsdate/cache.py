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
Handle cache for precalculated prior
"""

import os
import pathlib
import logging

import appdirs

import tsdate


__version__ = "undefined"
try:
    from . import _version
    __version__ = _version.version
except ImportError:
    try:
        __version__ = tsdate.__version__
    except ModuleNotFoundError:
        pass


logger = logging.getLogger(__name__)

_cache_dir = None


def set_cache_dir(cache_dir=None):
    """
    The cache_dir is the directory in which tsdate stores and checks for
    downloaded data. If the specified cache_dir is not None, this value is
    converted to a pathlib.Path instance, which is used as the cache directory.
    If cache_dir is None (the default) the cache is set to the
    default location using the :mod:`appdirs` module.
    No checks for existance, writability, etc. are performed by this function.
    """
    if cache_dir is None:
        cache_dir = appdirs.user_cache_dir("tsdate", "tsdate")
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
    global _cache_dir
    _cache_dir = pathlib.Path(cache_dir)
    logger.info(f"Set cache_dir to {_cache_dir}")


def get_cache_dir():
    """
    Returns the directory used to cache material precalculated and stored by tsdate as a
    pathlib.Path instance. Defaults to a directory 'tsdate' in a user cache directory
    (e.g., ~/.cache/tsdate on Unix flavours). See the :func:`.set_cache_dir` function
    for how this value can be set.
    """
    return _cache_dir
