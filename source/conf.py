# Configuration file for the Sphinx documentation builder.
import os
import sys

sys.path.insert(0, os.path.abspath("../"))

#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "quantminer"
copyright = "2024, Jerry Inyang"
author = "Jerry Inyang"
release = "1.0.0"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ["sphinx.ext.autodoc", "sphinx.ext.napoleon", "sphinx_autodoc_typehints"]

templates_path = ["_templates"]
exclude_patterns = []

# The suffix of source filenames
source_suffix = ".rst"

# The master toctree document
master_doc = "index"

# Autodoc settings
autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "private-members": True,
    "special-members": True,
    "inherited-members": True,
    "show-inheritance": True,
}


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "alabaster"
html_static_path = ["_static"]
