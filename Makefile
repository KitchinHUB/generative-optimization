# Makefile for executing Jupyter notebooks in place
# Usage: make all          - Execute all notebooks
#        make gmm          - Execute GMM notebooks
#        make fm           - Execute Flow Matching notebooks
#        make results      - Execute results notebooks
#        make pdf          - Build the manuscript PDF
#        make diff         - Build a latexdiff PDF vs DIFFBASE
#        make clean        - Remove checkpoints, build/ and LaTeX scratch

# Jupyter command for executing notebooks in place
#
# Prefer the project venv: a bare `jupyter` picks up whatever is on PATH,
# which does not have gmr installed and fails with ModuleNotFoundError.
#
# Invoked as `python -m jupyter` rather than through .venv/bin/jupyter,
# whose shebang still points at the path this project had before it was
# moved into @archive/.  Run `uv sync` to rewrite those console scripts.
JUPYTER = $(if $(wildcard .venv/bin/python),.venv/bin/python -m jupyter,jupyter)
#
# The notebooks record their kernel as "python3", which jupyter resolves to
# the first python3 kernelspec on the search path -- currently an unrelated
# venv without gmr.  KERNEL pins execution to this project's venv instead.
KERNEL ?= genopt-venv
JUPYTER_EXECUTE = $(JUPYTER) nbconvert --execute --to notebook --inplace \
                  --ExecutePreprocessor.kernel_name=$(KERNEL)

# GMM (00*) notebooks
GMM_NOTEBOOKS = \
	00a_root_finding.ipynb \
	00aa_newton_raphson.ipynb \
	00ab_complex_roots.ipynb \
	00b_optimization.ipynb \
	00c_equality_constraints.ipynb \
	00d_inequality_constraints.ipynb \
	00e_parameter_estimation.ipynb \
	00h_space_mapping.ipynb

# Flow Matching (01*) notebooks
FM_NOTEBOOKS = \
	01a_fm_root_finding.ipynb \
	01b_fm_optimization.ipynb \
	01c_fm_equality_constraints.ipynb \
	01d_fm_inequality_constraints.ipynb \
	01e_fm_parameter_estimation.ipynb \
	01f_fm_space_mapping.ipynb \
	01g_fm_eos_fitting.ipynb

# Other example notebooks
OTHER_NOTEBOOKS = \
	02_flow_field_visualization.ipynb \
	03_space_mapping_flow_matching.ipynb \
	04_advanced_optimization_examples.ipynb

# Results notebooks (combine GMM and FM results)
RESULTS_NOTEBOOKS = \
	results-00-root-finding-preos.ipynb \
	results-01-optimization.ipynb \
	results-02-equality-constrained-blending.ipynb \
	results-03-reaction-equilibrium.ipynb \
	results-04-parameter-estimation.ipynb \
	results-05-space-mapping.ipynb

# All notebooks (excluding results-and-figures which is deprecated)
ALL_NOTEBOOKS = $(GMM_NOTEBOOKS) $(FM_NOTEBOOKS) $(OTHER_NOTEBOOKS) $(RESULTS_NOTEBOOKS)

# Default target
.PHONY: all
all: gmm fm other results
	@echo "All notebooks executed successfully!"

# Execute GMM notebooks
.PHONY: gmm
gmm: $(GMM_NOTEBOOKS)
	@echo "GMM notebooks executed successfully!"

# Execute Flow Matching notebooks
.PHONY: fm
fm: $(FM_NOTEBOOKS)
	@echo "Flow Matching notebooks executed successfully!"

# Execute other example notebooks
.PHONY: other
other: $(OTHER_NOTEBOOKS)
	@echo "Other notebooks executed successfully!"

# Execute results notebooks (depend on GMM and FM being done)
.PHONY: results
results: gmm fm $(RESULTS_NOTEBOOKS)
	@echo "Results notebooks executed successfully!"

# Pattern rule for executing individual notebooks
%.ipynb: FORCE
	@echo "Executing $@..."
	@$(JUPYTER_EXECUTE) $@

# Execute readme separately if needed
.PHONY: readme
readme:
	@echo "Executing readme.ipynb..."
	@$(JUPYTER_EXECUTE) readme.ipynb

# ------------------------------------------------------------------
# Manuscript figures
#
# Figures are discovered from \includegraphics{...} in manuscript.tex so
# this list never goes stale when figures are added or removed.  It is
# defined up here, ahead of every rule that uses it, because make expands
# a rule's prerequisites at the moment the rule is read.
# ------------------------------------------------------------------
FIGURES = $(shell sed -n 's/.*\\includegraphics[^{]*{\([^}]*\)}.*/\1/p' manuscript.tex | sed 's|^\./||' | sort -u)

# ------------------------------------------------------------------
# Build directory
#
# This directory is inside Dropbox.  A LaTeX run rewrites manuscript.aux,
# .out and .blg several times, and achemso regenerates acs-manuscript.bib
# on every bibtex pass, so Dropbox ends up uploading files that latexmk is
# concurrently rewriting or deleting.  When it cannot reconcile the two it
# forks the file, which is where this directory's "conflicted copy" bib and
# aux files came from.
#
# So every intermediate goes to $(BUILDDIR), which is marked with Dropbox's
# com.dropbox.ignored extended attribute and never syncs.  Only the two
# finished products, manuscript.pdf and manuscript.bbl, are copied back out
# (the zip target ships the .bbl so recipients need not run bibtex).
# ------------------------------------------------------------------
BUILDDIR = build

$(BUILDDIR):
	@mkdir -p $(BUILDDIR)
	@xattr -w com.dropbox.ignored 1 $(BUILDDIR) 2>/dev/null || \
	  echo "note: could not mark $(BUILDDIR) as Dropbox-ignored"

# Scratch files from before the build directory existed, plus the conflicted
# copies Dropbox made of them.  Nothing should land here any more; this runs
# after each build to catch strays from a hand-run latexmk.
TEXEXT = aux log blg fls fdb_latexmk out toc lof lot spl synctex.gz xdv
TEXTRASH = $(foreach e,$(TEXEXT),manuscript.$(e) manuscript-diff.$(e)) \
	   manuscript.bbl manuscript-diff.bbl acs-manuscript.bib acs-manuscript-diff.bib

.PHONY: texclean
texclean:
	@rm -f $(TEXTRASH)
	@rm -f *"conflicted copy"*.aux *"conflicted copy"*.log *"conflicted copy"*.blg \
	       *"conflicted copy"*.fls *"conflicted copy"*.out *"conflicted copy"*.bbl \
	       *"conflicted copy"*.fdb_latexmk acs-manuscript*"conflicted copy"*.bib

# Build PDF from LaTeX
#
# The PDF depends on the bibliography and the figures as well as the tex,
# so regenerating a figure or editing manuscript.bib triggers a rebuild.
.PHONY: pdf
pdf: manuscript.pdf

manuscript.pdf: manuscript.tex manuscript.bib $(FIGURES) | $(BUILDDIR)
	@echo "Building PDF from manuscript.tex..."
	latexmk -C -outdir=$(BUILDDIR) manuscript.tex
	latexmk -pdf -outdir=$(BUILDDIR) manuscript.tex
	@cp $(BUILDDIR)/manuscript.pdf .
	@$(MAKE) --no-print-directory texclean
	@echo "PDF built successfully!"

# ------------------------------------------------------------------
# Marked-up diff against a previous revision
#
# DIFFBASE is the git ref the current manuscript is compared against.
# It defaults to the commit holding the version that was circulated, so
# "make diff" reproduces the markup reviewers were sent; override it for
# any other comparison:
#
#     make diff DIFFBASE=HEAD~3
# ------------------------------------------------------------------
DIFFBASE ?= 70d36a2

.PHONY: diff
diff: manuscript-diff.pdf

manuscript-diff.pdf: manuscript.tex manuscript.bib $(FIGURES) | $(BUILDDIR)
	@command -v latexdiff >/dev/null || \
	  { echo "ERROR: latexdiff not found (brew install latexdiff)"; exit 1; }
	@echo "Diffing manuscript.tex against $(DIFFBASE)..."
	@git show $(DIFFBASE):manuscript.tex > $(BUILDDIR)/manuscript-base.tex
	@latexdiff $(BUILDDIR)/manuscript-base.tex manuscript.tex > manuscript-diff.tex
	latexmk -pdf -outdir=$(BUILDDIR) manuscript-diff.tex
	@cp $(BUILDDIR)/manuscript-diff.pdf .
	@$(MAKE) --no-print-directory texclean
	@echo "Built manuscript-diff.pdf (vs $(DIFFBASE))"

# ------------------------------------------------------------------
# Package the manuscript source + figures for submission / sharing
# ------------------------------------------------------------------
ZIPNAME = manuscript-source.zip

# manuscript.bbl is included so the recipient can build without running bibtex.
# It is taken from $(BUILDDIR); zip -j flattens that path, which is safe
# because every figure is already at the top level of this directory.
TEX_SOURCES = manuscript.tex manuscript.bib $(BUILDDIR)/manuscript.bbl

ZIP_CONTENTS = $(TEX_SOURCES) $(FIGURES)

# Rebuilt unconditionally: packaging is cheap and a stale archive is worse
# than a redundant one.
.PHONY: zip
zip:
	@missing=""; for f in $(ZIP_CONTENTS); do \
	  [ -f "$$f" ] || missing="$$missing $$f"; \
	done; \
	if [ -n "$$missing" ]; then \
	  echo "ERROR: missing required file(s):$$missing"; \
	  echo "(manuscript.bbl is created by 'make pdf')"; \
	  exit 1; \
	fi
	@rm -f $(ZIPNAME)
	@zip -q -j $(ZIPNAME) $(ZIP_CONTENTS)
	@echo "Created $(ZIPNAME) with $(words $(ZIP_CONTENTS)) files:"
	@printf '  %s\n' $(ZIP_CONTENTS)

# Verify the zip actually builds a PDF in a clean directory
.PHONY: zip-check
zip-check: zip
	@tmp=$$(mktemp -d) && \
	unzip -q $(ZIPNAME) -d "$$tmp" && \
	(cd "$$tmp" && latexmk -pdf -interaction=nonstopmode manuscript.tex > build.log 2>&1) && \
	echo "OK: $(ZIPNAME) builds manuscript.pdf standalone" || \
	{ echo "FAILED: see $$tmp/build.log"; exit 1; }; \
	rm -rf "$$tmp"

# Clean checkpoint files
.PHONY: clean
clean: texclean
	@echo "Removing checkpoint files..."
	@rm -rf .ipynb_checkpoints $(BUILDDIR)
	@rm -f $(ZIPNAME)
	@echo "Checkpoint, build and LaTeX scratch files removed!"

# List all notebooks
.PHONY: list
list:
	@echo "GMM Notebooks:"
	@printf '  %s\n' $(GMM_NOTEBOOKS)
	@echo ""
	@echo "Flow Matching Notebooks:"
	@printf '  %s\n' $(FM_NOTEBOOKS)
	@echo ""
	@echo "Other Notebooks:"
	@printf '  %s\n' $(OTHER_NOTEBOOKS)
	@echo ""
	@echo "Results Notebooks:"
	@printf '  %s\n' $(RESULTS_NOTEBOOKS)

# Help target
.PHONY: help
help:
	@echo "Available targets:"
	@echo "  all       - Execute all notebooks (GMM, FM, other, then results)"
	@echo "  gmm       - Execute GMM notebooks (00*.ipynb)"
	@echo "  fm        - Execute Flow Matching notebooks (01*.ipynb)"
	@echo "  other     - Execute other example notebooks"
	@echo "  results   - Execute results notebooks (after GMM and FM)"
	@echo "  readme    - Execute readme.ipynb"
	@echo "  pdf       - Build PDF from manuscript.tex"
	@echo "  diff      - Build manuscript-diff.pdf vs DIFFBASE (default $(DIFFBASE))"
	@echo "  zip       - Package tex, bib, bbl + figures into $(ZIPNAME)"
	@echo "  zip-check - Verify the zip builds a PDF in a clean temp dir"
	@echo "  list      - List all notebooks"
	@echo "  texclean  - Remove stray LaTeX scratch files from the source dir"
	@echo "  clean     - Remove checkpoint files, $(BUILDDIR) and LaTeX scratch files"
	@echo "  help      - Show this help message"
	@echo ""
	@echo "Execute individual notebooks:"
	@echo "  make <notebook.ipynb>"

# Force execution of notebooks even if they haven't changed
.PHONY: FORCE
FORCE:
