# Configuration file for the Sphinx documentation builder.

import os
import sys
sys.path.insert(0, os.path.abspath('..'))

project = 'pyLEAFS'
copyright = '2026, Damian Sowinski'
author = 'Damian Sowinski'
release = '0.1.0'

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.mathjax',
    'sphinx.ext.viewcode',
    'sphinx_copybutton',
    'myst_nb',
]

copybutton_prompt_text = r'>>> |\.\.\. '
copybutton_prompt_is_regexp = True
# Copy buttons go on code, not on the outputs of notebook cells.
copybutton_selector = 'div:not(.output) > div.highlight pre'

# Pages written as Jupyter notebooks (docs/*.ipynb) are run from top to
# bottom on every build, so the outputs a reader sees always come from the
# code beside them and the repository stores the notebooks without outputs.
# A cell that raises fails the build, just like a warning does.
nb_execution_mode = 'force'
nb_execution_raise_on_error = True
nb_execution_timeout = 900

# Notebook text writes math as $...$ and $$...$$, which Jupyter and Colab
# render too, where the {math} role would show up as raw markup.
myst_enable_extensions = ['dollarmath']

# Every notebook page gets buttons that open it, runnable, on Binder and on
# Google Colab (see docs/_templates/page.html).  Both services read the
# notebook from this public repository and branch.
html_context = {
    'launch_repo': 'EternalTime/LEAFS',
    'launch_branch': 'master',
    'launch_path': 'docs',
}

templates_path = ['_templates']
exclude_patterns = ['_build', '**.ipynb_checkpoints']

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
html_css_files = ['custom.css']
