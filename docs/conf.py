# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# http://www.sphinx-doc.org/en/master/config


from unittest.mock import MagicMock
import os
import sys
import pkg_resources

class Mock(MagicMock):
    @classmethod
    def __getattr__(cls, name):
        return MagicMock()

MOCK_MODULES = [
    'numpy', 'tskit', 'tqdm', 'appdirs', 'numba', 'scipy', 'scipy.stats', 'scipy.special'
]
sys.modules.update((mod_name, Mock()) for mod_name in MOCK_MODULES)

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#

sys.path.insert(0, os.path.abspath('..'))

# The master document
master_doc = 'index'

# -- Project information -----------------------------------------------------

project = 'tsdate'
copyright = '2020, University of Oxford'
author = 'Anthony Wilder Wohns and Yan Wong'

# The full version, including alpha/beta/rc tags
try:
    from setuptools_scm import get_version
    release = get_version(root='..', relative_to=__file__)
    version = release[:3]
except pkg_resources.DistributionNotFound:
    release = "0.0.0"
    version = "0.0.0"


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.intersphinx',
    'sphinxarg.ext',
    'sphinx_issues',
    'recommonmark'
]

# Github repo
issues_github_path = "awohns/tsdate"

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'alabaster'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = []

# -- Extension configuration -------------------------------------------------

# -- Options for intersphinx extension ---------------------------------------

# Example configuration for intersphinx: refer to the Python standard library.
intersphinx_mapping = {
    'https://docs.python.org/': None,
    'https://docs.scipy.org/doc/numpy/': None,
    'https://tsdate.readthedocs.io/en/stable/': None,
}
