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
    'sphinx.ext.intersphinx',
    'sphinx.ext.napoleon',
    'sphinx.ext.mathjax',
    'sphinx.ext.viewcode',
    'sphinx_copybutton',
    'myst_nb',
    'matplotlib.sphinxext.plot_directive',
]

# Report every cross-reference whose target cannot be found.  With the -W
# in docs/Makefile, a dead reference fails the build instead of rendering as
# plain text.
nitpicky = True

# Docstring types such as "ndarray, shape (n, D), optional" are split into
# words and each word is looked up as a class.  The shape placeholders,
# "optional" and "array_like" are not classes, so they are not reported as
# dead references.
nitpick_ignore = [
    ('py:class', name)
    for name in ('optional', 'array_like', 'n', 'n_active', '3**D')
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

# Figures on the reStructuredText pages come from ``.. plot::`` blocks, which
# run their code on every build and show it above the figure.  Every run is
# seeded, so each build draws the same figures.  They are saved the way the
# notebook pages save theirs, as one PNG at 100 dpi trimmed to the drawing.
# All of their text - labels, ticks, legends and titles - is typeset by LaTeX
# in Computer Modern, as the notebook pages' is (see their first cell), so
# building the manual needs a LaTeX install with dvipng.
plot_include_source = True
plot_html_show_source_link = False
plot_html_show_formats = False
plot_formats = [('png', 100)]
plot_apply_rcparams = True
plot_rcparams = {
    'savefig.bbox': 'tight',
    'text.usetex': True,
    'font.family': 'serif',
    'font.serif': ['Computer Modern Roman'],
}

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

# Resolve references to the Python standard library and to NumPy.
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable', None),
}


# -- Plain dash between a parameter's type and its description -------------
# Sphinx and napoleon join them with " -- ", which smart quotes draw as an
# en dash.

from docutils import nodes
from sphinx.transforms import SphinxTransform


class PlainDashFieldSeparator(SphinxTransform):
    default_priority = 700  # before smart quotes, which run at 750

    def apply(self, **kwargs):
        for body in self.document.findall(nodes.field_body):
            for text in list(body.findall(nodes.Text)):
                if " -- " in text and not isinstance(text.parent, nodes.literal):
                    plain = nodes.Text(str(text).replace(" -- ", " - "))
                    text.parent.replace(text, plain)


def setup(app):
    app.add_transform(PlainDashFieldSeparator)
