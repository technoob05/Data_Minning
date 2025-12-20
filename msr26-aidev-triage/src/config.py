from pathlib import Path

# Project Root (calculated relative to this file)
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Data Paths
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

# Ensure directories exist
TABLES_DIR = OUTPUTS_DIR / "tables"
FIGURES_DIR = OUTPUTS_DIR / "figures"

for d in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, ARTIFACTS_DIR, OUTPUTS_DIR, TABLES_DIR, FIGURES_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# File Paths
PR_PARQUET = DATA_DIR / "pull_request.parquet"
COMMITS_PARQUET = DATA_DIR / "pr_commits.parquet"
TIMELINE_PARQUET = DATA_DIR / "pr_timeline.parquet"

FEATURES_SNAPSHOT = PROCESSED_DATA_DIR / "features_snapshot.parquet"
FEATURES_FULL = PROCESSED_DATA_DIR / "features_full.parquet"

# Constants
RANDOM_SEED = 42
HIGH_COST_PERCENTILE = 0.80
GHOSTING_THRESHOLD_DAYS = 14
