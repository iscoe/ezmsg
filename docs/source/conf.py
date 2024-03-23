import os
import sys
import importlib
import inspect

sys.path.insert(0, os.path.abspath('../../src'))
sys.path.insert(0, os.path.abspath('../../extensions/ezmsg-sigproc/src'))
# sys.path.insert(0, os.path.abspath('../../extensions/ezmsg-websocket/src'))
# sys.path.insert(0, os.path.abspath('../../extensions/ezmsg-zmq/src'))

# Configuration file for the Sphinx documentation builder.

# -- Project information

project = "ezmsg"
copyright = "2022, JHU/APL"
author = "JHU/APL"

release = "3.3.4"
version = "3.3.4"

# -- General configuration

extensions = [
    "sphinx.ext.duration",
    "sphinx.ext.doctest",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.linkcode",
    "sphinx.ext.napoleon"
]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "sphinx": ("https://www.sphinx-doc.org/en/master/", None),
    "numpy": ("https://docs.scipy.org/doc/numpy", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/reference", None),
}
intersphinx_disabled_domains = ["std"]

templates_path = ["_templates"]

# -- Options for HTML output

html_theme = "sphinx_rtd_theme"

# -- Options for EPUB output
epub_show_urls = "footnote"

add_module_names = False


code_url = f"https://github.com/iscoe/ezmsg/blob/dev/"


def linkcode_resolve(domain, info):
    if domain != 'py':
        return None
    if not info['module']:
        return None
    filename = info['module'].replace('.', '/')
    if "sigproc" in filename:
        return f"{code_url}extensions/ezmsg-sigproc/src/{filename}.py"
    elif "core" in filename:
        return f"{code_url}src/ezmsg/core/__init__.py"
    else:
        return f"{code_url}src/{filename}.py"

