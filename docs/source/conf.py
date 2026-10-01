"""Sphinx configuration for the catchment_simulation documentation.

The package API pages only need ``catchment_simulation`` to be importable.
The pages of the Django app (``cs_app_*.rst``) also need Django and the web
dependency group; without them they are left out and the package pages still
build.
"""

import os
import sys
from pathlib import Path

import catchment_simulation

CS_APP = Path(__file__).resolve().parents[2] / "cs_app"

project = "Catchment Simulation"
author = "Rafał Buczyński"
copyright = "2022-2026, Rafał Buczyński"
release = catchment_simulation.__version__
version = ".".join(release.split(".")[:2])

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.coverage",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "myst_parser",
]

source_suffix = {".rst": "restructuredtext", ".md": "markdown"}
exclude_patterns: list[str] = []

napoleon_google_docstring = False
napoleon_numpy_docstring = True

autodoc_typehints = "description"
autodoc_member_order = "bysource"

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
}

html_theme = "sphinx_rtd_theme"

try:
    import django
except ImportError:
    # Without the web group only the package documentation is built.
    exclude_patterns.append("cs_app_*.rst")
    suppress_warnings = ["toc.excluded"]
else:
    sys.path.insert(0, str(CS_APP))
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "cs_app.test_settings")
    django.setup()
