# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys

# Add package to path for autodoc
sys.path.insert(0, os.path.abspath('..'))

# -- Project information -----------------------------------------------------

project = 'svmix'
copyright = '2026, Tiaan Viviers'
author = 'Tiaan Viviers'
release = '1.0.0'
version = '1.0.0'

# -- General configuration ---------------------------------------------------

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.intersphinx',
    'sphinx.ext.mathjax',
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# -- Options for autodoc -----------------------------------------------------

autodoc_default_options = {
    'members': True,
    'member-order': 'bysource',
    'special-members': '__init__',
    'undoc-members': False,
    'exclude-members': '__weakref__',
    'show-inheritance': True,
}

autodoc_typehints = 'description'
autodoc_class_signature = 'separated'

# Mock the native module since it requires the C library
autodoc_mock_imports = ['svmix._native']

# -- Options for Napoleon (Google/NumPy docstrings) --------------------------

napoleon_google_docstrings = True
napoleon_numpy_docstrings = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = True
napoleon_use_admonition_for_notes = True
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_use_keyword = True
napoleon_attr_annotations = True

# -- Options for HTML output -------------------------------------------------

html_theme = 'furo'
html_title = 'svmix'
html_static_path = ['_static']

html_theme_options = {
    "light_css_variables": {
        "color-brand-primary": "#2563eb",
        "color-brand-content": "#2563eb",
    },
    "dark_css_variables": {
        "color-brand-primary": "#60a5fa",
        "color-brand-content": "#60a5fa",
    },
    "sidebar_hide_name": False,
    "navigation_with_keys": True,
}

# -- Options for intersphinx -------------------------------------------------

intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
}

# -- Options for MathJax -----------------------------------------------------

mathjax3_config = {
    'tex': {
        'inlineMath': [['$', '$'], ['\\(', '\\)']],
        'displayMath': [['$$', '$$'], ['\\[', '\\]']],
    },
}
