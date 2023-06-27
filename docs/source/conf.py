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
    "sphinx_design",
    "sphinx_favicon",
    "sphinx_automodapi.automodapi",
    "sphinx_automodapi.smart_resolver",
    ]




templates_path = ['_templates']
exclude_patterns = []

autodoc_mock_imports = ["pyueye","serial.tools.list_ports","Serial","cv2","webcolors","serial"]
#"PyQt5","numpy","sklearn","PIL","scipy","matplotlib","pandas","numpy.core","numpy"


favicons = [
   {
      "sizes": "16x16",
      "href": "_static/logo.png",
   },
   {
      "sizes": "32x32",
      "href": "_static/logo.png",
   },
   {
      "rel": "apple-touch-icon",
      "sizes": "180x180",
      "href": "_static/logo.png",  # use a local file in _static
   },
]

autoapi_dirs = [
    'your-module/',
]


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'furo'
html_static_path = ['_static']

html_logo = "_static/logo.png"

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
