# Makefile for MSR 2026 Mining Challenge Paper
# "Ghosting in the Machine"
#
# Usage:
#   make features    - Generate feature matrix
#   make train       - Train models
#   make eval        - Evaluate and generate metrics
#   make figs        - Generate all figures
#   make paper       - Compile LaTeX paper
#   make reproduce   - Full end-to-end reproduction

PYTHON = python
CONDA_ENV = msr26_aidev
PAPER_DIR = paper
OUTPUT_DIR = outputs
DATA_DIR = data

.PHONY: all features train eval figs paper reproduce clean

all: reproduce

# Activate conda environment (run manually if needed)
conda-check:
	@echo "Ensure conda environment '$(CONDA_ENV)' is active"

# Generate feature matrix from raw data
features:
	$(PYTHON) scripts/01_data_exploration.py
	$(PYTHON) scripts/03_feature_engineering.py
	@echo "Features generated in $(DATA_DIR)/processed/"

# Train models
train:
	$(PYTHON) scripts/04_modeling.py
	@echo "Models saved in models/"

# Evaluate models and generate metrics
eval:
	$(PYTHON) scripts/05_evaluation.py
	$(PYTHON) scripts/snapshot_vs_full.py
	$(PYTHON) scripts/audit_ghosting.py
	@echo "Evaluation complete. Results in $(OUTPUT_DIR)/"

# Generate all figures
figs:
	$(PYTHON) scripts/02_visualizations.py
	$(PYTHON) scripts/06_shap_analysis.py
	$(PYTHON) scripts/audit_ghosting.py
	@echo "Figures generated in $(OUTPUT_DIR)/figures/ and $(OUTPUT_DIR)/audit/"

# Compile LaTeX paper
paper:
	cd $(PAPER_DIR) && xelatex -interaction=nonstopmode paper_draft.tex
	cd $(PAPER_DIR) && bibtex paper_draft
	cd $(PAPER_DIR) && xelatex -interaction=nonstopmode paper_draft.tex
	cd $(PAPER_DIR) && xelatex -interaction=nonstopmode paper_draft.tex
	@echo "Paper compiled: $(PAPER_DIR)/paper_draft.pdf"

# Full reproduction pipeline
reproduce: features train eval figs paper
	@echo "=============================================="
	@echo "REPRODUCTION COMPLETE"
	@echo "=============================================="
	@echo "Key outputs:"
	@echo "  - Features: $(DATA_DIR)/processed/valid_features.csv"
	@echo "  - Audit: $(OUTPUT_DIR)/audit/ghosting_audit_sample.csv"
	@echo "  - Leakage: $(OUTPUT_DIR)/leakage_check/leakage_check_results.csv"
	@echo "  - Paper: $(PAPER_DIR)/paper_draft.pdf"

# Clean generated files
clean:
	rm -rf $(OUTPUT_DIR)/*
	rm -f $(PAPER_DIR)/*.aux $(PAPER_DIR)/*.bbl $(PAPER_DIR)/*.blg
	rm -f $(PAPER_DIR)/*.log $(PAPER_DIR)/*.out $(PAPER_DIR)/*.pdf
	@echo "Cleaned generated files"

# Help
help:
	@echo "Available targets:"
	@echo "  features  - Generate feature matrix"
	@echo "  train     - Train models"
	@echo "  eval      - Evaluate models"
	@echo "  figs      - Generate figures"
	@echo "  paper     - Compile LaTeX paper"
	@echo "  reproduce - Full end-to-end reproduction"
	@echo "  clean     - Remove generated files"
