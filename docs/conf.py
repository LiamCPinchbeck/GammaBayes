# Configuration file for the Sphinx documentation builder.
#
# Build:   cd docs && make html          (see docs/README.md)
# The API reference is generated with autodoc, so the package must be importable:
# ``../src`` is put on sys.path.  Heavy optional dependencies that are not installed
# in the documentation environment are mocked (see ``autodoc_mock_imports`` below).

import importlib.util
import os
import sys

sys.path.insert(0, os.path.abspath('../src'))

# -- Project information -----------------------------------------------------
project = 'GammaBayes'
copyright = '2023-2026, Liam Pinchbeck'
author = 'Liam Pinchbeck'

try:
    from gammabayes import __version__ as release          # keep in sync with the package
except Exception:                                          # pragma: no cover - docs env without deps
    release = '2.1.0'
version = '.'.join(release.split('.')[:2])

# -- General configuration ---------------------------------------------------
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.napoleon',
    'sphinx.ext.mathjax',
    'sphinx.ext.viewcode',
    'sphinx.ext.intersphinx',
    'nbsphinx',
    'myst_parser',
]

master_doc = 'index'
source_suffix = {'.rst': 'restructuredtext', '.md': 'markdown'}
templates_path = ['_templates']
exclude_patterns = ['_build', '**.ipynb_checkpoints', 'Thumbs.db', '.DS_Store',
                    'legacy_tutorials/*.ipynb',        # v0.x notebooks: kept for reference, not built
                    'README.md']

# -- autodoc / autosummary ---------------------------------------------------
autodoc_default_options = {
    'members': True,
    'undoc-members': True,
    'show-inheritance': True,
    'member-order': 'bysource',
}
autodoc_typehints = 'description'
autodoc_class_signature = 'separated'
autosummary_generate = False
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_use_param = True
napoleon_use_rtype = True

# Mock whatever heavy dependency is missing from the docs environment so that the API pages still
# build (signatures / docstrings only).  In the analysis environment nothing is mocked.
_OPTIONAL = ['torch', 'pyro', 'gammapy', 'healpy', 'h5py', 'dynesty', 'corner', 'icecream', 'tqdm',
             'seaborn', 'requests', 'astropy', 'scipy', 'pandas', 'yaml', 'matplotlib']
autodoc_mock_imports = [m for m in _OPTIONAL if importlib.util.find_spec(m) is None]

intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'torch': ('https://pytorch.org/docs/stable/', None),
    'astropy': ('https://docs.astropy.org/en/stable/', None),
    'gammapy': ('https://docs.gammapy.org/2.0/', None),
}

# -- myst -------------------------------------------------------------------
myst_enable_extensions = ['dollarmath', 'amsmath', 'colon_fence', 'deflist']
myst_heading_anchors = 3

# -- nbsphinx ---------------------------------------------------------------
# Notebooks are stored EXECUTED (run docs/run_notebooks.sh on a machine with the IRF files and a GPU
# or a patient CPU); the HTML build never executes them.
nbsphinx_execute = 'never'
nbsphinx_allow_errors = False
nbsphinx_prolog = r"""
.. raw:: html

   <div class="admonition note">
   <p class="admonition-title">Note</p>
   <p>This page was generated from the Jupyter notebook
   <code>docs/{{ env.doc2path(env.docname, base=None) }}</code> in the
   <a href="https://github.com/lpin0002/GammaBayes">GammaBayes repository</a>.</p>
   </div>
"""

# -- Options for HTML output -------------------------------------------------
html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
html_theme_options = {
    'navigation_depth': 4,
    'collapse_navigation': False,
}
