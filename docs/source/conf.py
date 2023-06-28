# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
import os
import sys

sys.path.insert(0, os.path.abspath(".."))

project = "Pycollo"
copyright = "2023, Sam Brockie"  # noqa: A001
author = "Sam Brockie"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "nbsphinx"
    
]


templates_path = ["source/_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store","**.ipynb_checkpoints"]

napoleon_numpy_docstring = True

intersphinx_mapping = {
    "cyipopt": ("https://cyipopt.readthedocs.io/en/stable/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
    "numba": ("https://numba.readthedocs.io/en/stable/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "sympy": ("https://docs.sympy.org/latest/", None),
}


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_material"
html_theme_options = {
    "base_url": "https://brocksam.github.io/pycollo/",
    "color_primary": "teal",
    "color_accent": "deep-orange",  # hover color of hyperlinks
    "repo_name": "Pycollo",
    "repo_url": "https://github.com/brocksam/pycollo/",
    "logo_icon": "&#xe52f",
    "master_doc": False,  # Doesn't show duplicate title
    "nav_links": [{"href": "index", "internal": True, "title": "Home"}],
}

html_sidebars = {
    "**": ["logo-text.html", "globaltoc.html", "localtoc.html", "searchbox.html"]
}

html_static_path = ["_static"]
