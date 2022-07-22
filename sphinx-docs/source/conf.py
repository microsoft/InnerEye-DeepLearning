#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------

# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, make it absolute.
#
import sys
import shutil
from pathlib import Path
repo_dir = Path(__file__).absolute().parents[2]
sys.path.insert(0, str(repo_dir))
from InnerEye.Common import fixed_paths
fixed_paths.add_submodules_to_path()


# -- Imports -----------------------------------------------------------------
from recommonmark.parser import CommonMarkParser

# -- Project information -----------------------------------------------------

project = 'InnerEye-DeepLearning'
copyright = 'Microsoft Corporation'
author = 'Microsoft'

# The full version, including alpha/beta/rc tags
release = '1.0.0'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx_rtd_theme',
    'recommonmark',
    'sphinx.ext.viewcode'
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []  # type: ignore


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
# html_static_path = ['_static']

source_parsers = {
    '.md': CommonMarkParser,
}

source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}

# Autodoc options

autodoc_default_options = {
    'members': True,
    'undoc-members': True,
}


# -- Copy markdown files to source directory --------------------------------

def replace_in_file(filepath: Path, original_str: str, replace_str: str) -> None:
    """
    Replace all occurences of the original_str with replace_str in the file provided.
    """
    text = filepath.read_text()
    text = text.replace(original_str, replace_str)
    filepath.write_text(text)

sphinx_root = Path(__file__).absolute().parent
docs_path = Path(sphinx_root / "docs")
repository_root = sphinx_root.parent.parent
repository_url = "https://github.com/microsoft/InnerEye-DeepLearning"

# Copy all markdown files to the markdown directory
shutil.copy(repository_root / "README.md", docs_path)
shutil.copy(repository_root / "CHANGELOG.md", docs_path)

# replace links to files in repository with urls
md_files = docs_path.rglob("*.md")
print(f"MY FILES: {[f for f in md_files]}")
for filepath in md_files:
    replace_in_file(filepath, "](/", f"]({repository_url}/blob/main/")
