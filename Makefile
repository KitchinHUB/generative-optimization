# Makefile for executing Jupyter notebooks in place
# Usage: make all          - Execute all notebooks
#        make gmm          - Execute GMM notebooks
#        make fm           - Execute Flow Matching notebooks
#        make results      - Execute results notebooks
#        make clean        - Remove checkpoint files

# Jupyter command for executing notebooks in place
JUPYTER_EXECUTE = jupyter nbconvert --execute --to notebook --inplace

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

# Clean checkpoint files
.PHONY: clean
clean:
	@echo "Removing checkpoint files..."
	@rm -rf .ipynb_checkpoints
	@echo "Checkpoint files removed!"

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
	@echo "  list      - List all notebooks"
	@echo "  clean     - Remove checkpoint files"
	@echo "  help      - Show this help message"
	@echo ""
	@echo "Execute individual notebooks:"
	@echo "  make <notebook.ipynb>"

# Force execution of notebooks even if they haven't changed
.PHONY: FORCE
FORCE:
