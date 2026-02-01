# Minimal makefile for Sphinx documentation

SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = .
BUILDDIR      = _build

.PHONY: help clean html view

help:
	@echo "Usage: make <target>"
	@echo ""
	@echo "Targets:"
	@echo "  html     Build HTML documentation"
	@echo "  view     Build and open documentation in browser"
	@echo "  clean    Remove build artifacts"

html:
	@$(SPHINXBUILD) -b html "$(SOURCEDIR)" "$(BUILDDIR)/html" $(SPHINXOPTS)
	@echo "Build finished. Open $(BUILDDIR)/html/index.html"

view: html
	@google-chrome "$(BUILDDIR)/html/index.html" 2>/dev/null || \
	 chromium-browser "$(BUILDDIR)/html/index.html" 2>/dev/null || \
	 xdg-open "$(BUILDDIR)/html/index.html" 2>/dev/null || \
	 echo "file://$(PWD)/$(BUILDDIR)/html/index.html"

clean:
	rm -rf $(BUILDDIR)
