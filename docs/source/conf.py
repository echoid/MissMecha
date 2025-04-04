# Configuration file for the Sphinx documentation builder.

import os
import sys
sys.path.insert(0, os.path.abspath("../.."))
sys.path.insert(0, os.path.abspath("../../missmecha"))

# -- Project information -----------------------------------------------------

project = "MissMecha"
copyright = "2025, Youran Zhou"
author = "Youran Zhou"
release = "0.0.1"

# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
]

templates_path = ["_templates"]
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------

html_theme = "sphinx_rtd_theme"
#html_theme = "pydata_sphinx_theme"
