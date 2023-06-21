# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys

sys.path.insert(0,os.path.abspath("../../src"))
sys.path.insert(0,os.path.abspath("../../src/Train"))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'Industrial_Vision'
copyright = '2023, Alex6Crbt'
author = 'Alex6Crbt'
release = '1.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon", 
    "sphinx_copybutton",
    ]

templates_path = ['_templates']
exclude_patterns = []

autodoc_mock_imports = ["pyueye","serial.tools.list_ports","Serial",]
#"PyQt5","numpy","sklearn","PIL","scipy","matplotlib","pandas","numpy.core","numpy"

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'furo'
html_static_path = ['_static']

# html_logo = "logo.png"

html_theme_options = {
    "light_css_variables": {
        "color-brand-primary": "#7C4DFF",
        "color-brand-content": "#7C4DFF",
    },
    "dark_css_variables": {
        "color-brand-primary": "#F7DC6F",
        "color-brand-content": "#F7DC6F",
    },
}
