# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
import sys
import os

sys.path.insert(0, os.path.abspath('../..'))
print(sys.path)
project = 'fiddler-auditor'
copyright = '2023, Fiddler Labs'
author = 'Fiddler Labs'
release = '0.0.1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'nbsphinx',
]

templates_path = ['_templates']
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# html_theme = 'alabaster'
# html_theme = 'sphinx_rtd_theme'
html_theme = 'sphinx_book_theme'
html_static_path = ['_static']

html_logo = 'images/fiddler-logo_black.png'
html_theme_options = {
    'logo_only': False
}

autosummary_generate = True

# def setup(app):
#     import modelauditor
    # need to assign the names here, otherwise autodoc won't document these classes,
    # # and will instead just say 'alias of ...'
    # modelauditor.perturbations.PerturbText.__name__ = "PerturbText"
    # modelauditor.perturbations.PerturbText.__module__ = "modelauditor.perturbations"
