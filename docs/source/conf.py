# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.

import os
import sys

sys.path.insert(0, os.path.abspath("../.."))

# -- Project information -----------------------------------------------------

project = "pytsviz"
author = "xtream"
copyright = "2021, " + author

# The full version, including alpha/beta/rc tags
release = "0.1.0"

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "myst_parser",
    "nbsphinx",
    "sphinx.ext.autodoc",
    "sphinx_autodoc_typehints",
    "sphinx.ext.intersphinx",
]

# nbsphinx settings
nbsphinx_execute = 'always'

# Intersphinx settings
intersphinx_mapping = {
    "numpy": ("https://numpy.org/doc/stable/", None),
    "matplotlib": ("https://matplotlib.org", None),
    "pandas": ("https://pandas.pydata.org/pandas-docs/dev", None),
    "python": ("https://docs.python.org/3", None),
    "plotly": ("https://plotly.com/python-api-reference/", None),
}

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------

# Set page favicon
html_favicon = "images/favicon.ico"

# Set link name generated in the top bar.
html_title = project

FORCE_RTD = os.environ.get("SPHINX_FORCE_RTD_THEME", False)
FORCE_RTD = FORCE_RTD in ("1", "true")
if FORCE_RTD:
    print("Using RTD theme")
    html_theme = "sphinx_rtd_theme"
    html_sidebars = {
        "**": ["globaltoc.html", "localtoc.html", "searchbox.html"]
    }
    html_theme_options = {
        # 'analytics_id': 'G-XXXXXXXXXX',
        # 'analytics_anonymize_ip': False,
        "logo_only": False,
        "display_version": True,
        "prev_next_buttons_location": "bottom",
        "style_external_links": False,
        # Toc options
        "collapse_navigation": False,
        "sticky_navigation": True,
        "navigation_depth": 4,
        "includehidden": True,
        "titles_only": False,
    }
else:
    import sphinx_material

    print("Using material theme")
    extensions.append("sphinx_material")
    html_theme_path = sphinx_material.html_theme_path()
    html_context = sphinx_material.get_html_context()
    html_theme = "sphinx_material"
    html_sidebars = {"**": ["globaltoc.html", "searchbox.html"]}
    html_theme_options = {
        # Set you GA account ID to enable tracking
        # 'google_analytics_account': 'UA-XXXXX',
        # Set the color and the accent color
        "color_primary": "blue",
        "color_accent": "cyan",
        "html_minify": False,
        "html_prettify": True,
        "css_minify": True,
        "logo_icon": "&#xe869",
        "repo_type": "github",
        # Set the repo location to get a badge with stats
        "repo_url": "https://github.com/xtreamsrl/pytsviz",
        "repo_name": "pytsviz",
        # Navbar config
        "master_doc": False,
        "nav_links": [
            {"href": "index", "internal": True, "title": "Home"},
            {
                "href": "_autodoc/pytsviz",
                "internal": True,
                "title": "Documentation",
            },
            {
                "href": "notebooks/GO_01_Displaying_Plots",
                "internal": True,
                "title": "Examples",
            },
            {"href": "support", "internal": True, "title": "Support"},
        ],
        # 'version_dropdown': True,
        # TOC Tree generation
        # The maximum depth of the global TOC; set it to -1 to allow unlimited depth
        "globaltoc_depth": -1,
        # If true, TOC entries that are not ancestors of the current page are collapsed
        "globaltoc_collapse": True,
        # If true, the global TOC tree will also contain hidden entries
        "globaltoc_includehidden": True,
    }

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

language = "en"


def setup(app):
    app.add_css_file("my_theme.css")
