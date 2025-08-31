# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

# pylint: skip-file

import os
import os.path as osp
import shutil
import sys

import guidata.config as gcfg
from docutils import nodes
from docutils.parsers.rst import Directive
from docutils.statemachine import StringList


class OptionsTableDirective(Directive):
    """Custom directive to include dynamically generated options table."""

    has_content = False

    def run(self):
        """Generate and include the options table."""
        from sigima.config import options

        # Get the RST content
        rst_content = options.generate_rst_doc()

        # Create a container node
        container = nodes.container()

        # Parse the RST content and add it to the container
        rst_lines = rst_content.splitlines()
        string_list = StringList(rst_lines)
        self.state.nested_parse(string_list, self.content_offset, container)

        return [container]


sys.path.insert(0, os.path.abspath(".."))

import sigima

# Turn off validation of guidata config
# (documentation build is not the right place for validation)
gcfg.set_validation_mode(gcfg.ValidationMode.DISABLED)


def copy_changelog(app):
    """Copy CHANGELOG.md to doc folder."""
    docpath = osp.abspath(osp.dirname(__file__))
    dest_fname = osp.join(docpath, "changelog.md")
    if osp.exists(dest_fname):
        os.remove(dest_fname)
    shutil.copyfile(osp.join(docpath, "..", "CHANGELOG.md"), dest_fname)
    app.env.temp_changelog_path = dest_fname


def cleanup_changelog(app, exception):
    """Remove CHANGELOG.md from doc folder."""
    try:
        path = getattr(app.env, "temp_changelog_path", None)
        if path and osp.exists(path):
            os.remove(path)
    except Exception as exc:
        print(f"Warning: failed to remove {path}: {exc}")
    finally:
        if hasattr(app.env, "temp_changelog_path"):
            del app.env.temp_changelog_path


def setup(app):
    """Setup function for Sphinx."""
    app.connect("builder-inited", copy_changelog)
    app.connect("build-finished", cleanup_changelog)
    app.add_directive("options-table", OptionsTableDirective)


# -- Project information -----------------------------------------------------

project = "Sigima"
author = ""
copyright = "2025, DataLab Platform Developers"
release = sigima.__version__

# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "sphinx.ext.mathjax",
    "sphinx.ext.githubpages",
    "sphinx.ext.viewcode",
    "myst_parser",
    "sphinx_design",
    "sphinx_copybutton",
    "guidata.dataset.autodoc",
]
templates_path = ["_templates"]
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------
html_theme = "pydata_sphinx_theme"
html_logo = "images/Sigima-Banner.svg"
html_title = project
html_favicon = "_static/favicon.ico"
html_show_sourcelink = False
templates_path = ["_templates"]
if "language=fr" in sys.argv:
    ann = "Sigima est en cours de développement 🚧"  # noqa: E501
else:
    ann = "Sigima is under development 🚧"  # noqa: E501
html_theme_options = {
    "show_toc_level": 2,
    "github_url": "https://github.com/DataLab-Platform/Sigima/",
    "logo": {
        "text": f"v{sigima.__version__}",
    },
    "icon_links": [
        {
            "name": "PyPI",
            "url": "https://pypi.org/project/sigima",
            "icon": "_static/pypi.svg",
            "type": "local",
            "attributes": {"target": "_blank"},
        },
        {
            "name": "Codra",
            "url": "https://codra.net",
            "icon": "_static/codra.png",
            "type": "local",
            "attributes": {"target": "_blank"},
        },
        {
            "name": "DataLab",
            "url": "https://datalab-platform.com",
            "icon": "_static/DataLab.svg",
            "type": "local",
            "attributes": {"target": "_blank"},
        },
    ],
    "announcement": ann,
}
html_static_path = ["_static"]

# -- Options for LaTeX output ------------------------------------------------
latex_logo = "_static/Sigima-Frontpage.png"

# -- Options for sphinx-intl package -----------------------------------------
locale_dirs = ["locale/"]  # path is example but recommended.
gettext_compact = False
gettext_location = False

# -- Options for autodoc extension -------------------------------------------
autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
}

# -- Options for intersphinx extension ---------------------------------------
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "scikit-image": ("https://scikit-image.org/docs/stable/", None),
    "guidata": ("https://guidata.readthedocs.io/en/latest/", None),
}
