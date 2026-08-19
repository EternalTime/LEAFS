# Minimal makefile for Sphinx documentation

# -W turns Sphinx warnings into errors, so a broken cross-reference or a page
# missing from a toctree fails the build instead of passing quietly; ?= keeps
# it overridable while drafting.
SPHINXOPTS    ?= -W
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = .
BUILDDIR      = _build

help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS)

.PHONY: help Makefile

%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS)
