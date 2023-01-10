# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

import logging as pylogging

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
from pathlib import Path

import dask.dataframe
import xarray
from sphinx.util import logging

sys.path.insert(0, os.path.abspath(".."))

print("python exec:", sys.executable)
print("sys.path:", sys.path)

HERE = Path(__file__).parent
sys.path[:0] = [str(HERE.parent), str(HERE / "extensions")]

import sgkit  # noqa: F401 isort:skip

# -- Project information -----------------------------------------------------

project = "sgkit"
copyright = "2020, sgkit developers"
author = "sgkit developers"


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx_autodoc_typehints",  # needs to be after napoleon
    "scanpydoc.elegant_typehints",
    "sphinx.ext.autosummary",
    "sphinx.ext.githubpages",
    "sphinx.ext.intersphinx",
    "sphinx.ext.extlinks",
    "scanpydoc",
    "IPython.sphinxext.ipython_console_highlighting",
    "IPython.sphinxext.ipython_directive",
    "myst_nb",
    "ablog",
    *[p.stem for p in (HERE / "extensions").glob("*.py")],
]

extlinks = {
    "issue": ("https://github.com/pystatgen/sgkit/issues/%s", "GH %s"),
    "pr": ("https://github.com/pystatgen/sgkit/pull/%s", "PR %s"),
    "user": ("https://github.com/%s", "%s"),
}


# Workaround https://github.com/agronholm/sphinx-autodoc-typehints/issues/123
# When this https://github.com/agronholm/sphinx-autodoc-typehints/pull/153
# gets merged, we can remove this
class FilterForIssue123(pylogging.Filter):
    def filter(self, record: pylogging.LogRecord) -> bool:
        msg = record.getMessage()
        return not (
            msg.startswith("Cannot treat a function")
            and any(
                s in msg
                for s in ["sgkit.variables.Spec", "sgkit.variables.ArrayLikeSpec"]
            )
        )


logging.getLogger("sphinx_autodoc_typehints").logger.addFilter(FilterForIssue123())
# End of workaround


# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

html_context = dict(
    display_github=False,  # Integrate GitHub
    github_user="pystatgen",  # Username
    github_repo="sgkit",  # Repo name
    github_version="main",  # Version
    conf_py_path="/docs/",  # Path in the checkout to the docs root
)

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

autosummary_generate = True

nitpick_ignore = [("py:class", "sgkit.display.GenotypeDisplay")]


# FIXME: Workaround for linking xarray module
# For some reason, intersphinx is not able to to link xarray objects.
# https://github.com/pydata/xarray/issues/4279
xarray.Dataset.__module__ = "xarray"
xarray.DataArray.__module__ = "xarray"

dask.dataframe.DataFrame.__module__ = "dask.dataframe"

intersphinx_mapping = dict(
    dask=("https://docs.dask.org/en/stable/", None),
    xarray=("https://xarray.pydata.org/en/stable/", None),
    zarr=("https://zarr.readthedocs.io/en/stable", None),
    numpy=("https://numpy.org/doc/stable/", None),
    python=("https://docs.python.org/3", None),
    sklearn=("https://scikit-learn.org/stable/", None),
)

# -- Options for HTML output -------------------------------------------------

version = sgkit.__version__
if "dev" in version or version == "unknown":
    version = "latest"

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "pydata_sphinx_theme"

# https://pydata-sphinx-theme.readthedocs.io/en/latest/user_guide/configuring.html
html_theme_options = {
    "github_url": "https://github.com/pystatgen/sgkit",
    "logo": {
        "image_light": "sgkit_trnsprnt.png",
        "image_dark": "sgkit_blue_trnsprnt.png",
    },
    "navbar_end": ["version-switcher", "theme-switcher", "navbar-icon-links"],
    "switcher": {
        "json_url": "_static/switcher.json",
        "version_match": version,
    },
}

html_css_files = [
    "docsearch.sbt.css",
    "https://cdn.jsdelivr.net/npm/docsearch.js@2/dist/cdn/docsearch.min.css",
]

html_js_files = [
    (
        "https://cdn.jsdelivr.net/npm/docsearch.js@2/dist/cdn/docsearch.min.js",
        {"defer": "defer"},
    ),
    ("docsearch.sbt.js", {"defer": "defer"}),
]

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]
