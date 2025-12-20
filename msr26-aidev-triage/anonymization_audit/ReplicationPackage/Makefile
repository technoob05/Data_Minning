
# MSR 2026 Mining Challenge - AI Dev Triage Replication
# Environment: Python 3.9+

PYTHON = python
PIP = pip

.PHONY: help install features train eval figures audit paper reproduce clean

help:
	@echo "Available targets:"
	@echo "  install    : Install dependencies"
	@echo "  features   : Generate Snapshot and Full features"
	@echo "  audit      : Run Ghosting Audit (CSV + Plot)"
	@echo "  eval       : Train models and bootstrap CIs"
	@echo "  figures    : Generate paper figures"
	@echo "  paper      : Compile LaTeX paper"
	@echo "  reproduce  : Run full pipeline (features -> audit -> eval -> figures)"
	@echo "  clean      : Remove artifacts"

install:
	$(PIP) install -r requirements.txt

features:
	$(PYTHON) scripts/build_snapshot_features.py

audit:
	$(PYTHON) scripts/audit_ghosting_enhanced.py
	$(PYTHON) scripts/effort_definition.py

eval:
	$(PYTHON) scripts/bootstrap_results.py

figures:
	$(PYTHON) scripts/plot_ghost_rate_cis.py

paper:
	cd paper && xelatex -interaction=nonstopmode paper_draft.tex
	cd paper && bibtex paper_draft
	cd paper && xelatex -interaction=nonstopmode paper_draft.tex
	cd paper && xelatex -interaction=nonstopmode paper_draft.tex

reproduce: features audit eval figures

clean:
	rm -rf output/metrics/*
	rm -rf data/processed/features_*.parquet
