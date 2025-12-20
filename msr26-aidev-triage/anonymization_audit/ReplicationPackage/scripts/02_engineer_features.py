import sys
from pathlib import Path
import pandas as pd

# Add Project Root to Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.config import PROCESSED_DATA_DIR, RAW_DATA_DIR, FEATURES_SNAPSHOT, FEATURES_FULL
from src.utils import setup_logger, load_data, save_data
from src.features import engineer_features, get_feature_columns

logger = setup_logger("02_engineer_features")

def main():
    logger.info("Step 2: Engineer Features")
    
    # Input Path
    # Ideally should process from raw, but reusing valid_features.csv for continuity with previous steps if raw not ready
    input_path = PROCESSED_DATA_DIR / "valid_features.csv"
    if not input_path.exists():
        logger.warning(f"{input_path} not found. Checking artifacts...")
        # Fallback to pr_features.parquet if exists
        fallback = ARTIFACTS_DIR / "pr_features.parquet"
        if fallback.exists():
            input_path = fallback
        else:
            logger.error("No input features found.")
            return

    logger.info(f"Loading data from {input_path}")
    df_raw = load_data(input_path, file_type="csv" if str(input_path).endswith('.csv') else "parquet")
    
    # CRITICAL FIX: Actually generate the features and targets!
    logger.info("Running Feature Engineering...")
    df = engineer_features(df_raw)
    
    # 1. Snapshot Features (Strict)
    snapshot_cols = get_feature_columns(strict=True)
    # Add Targets/Meta for training
    targets = ['is_high_cost', 'is_ghosted', 'is_merged', 'effort_score']
    meta = ['id', 'number', 'repo_full_name', 'created_at']
    
    # Filter available
    snap_final = [c for c in snapshot_cols + targets + meta if c in df.columns]
    df_snap = df[snap_final].copy()
    
    # 2. Full Features
    # Just take everything available
    df_full = df.copy()
    
    # Save
    logger.info(f"Saving Snapshot Features to {FEATURES_SNAPSHOT}")
    save_data(df_snap, FEATURES_SNAPSHOT)
    
    logger.info(f"Saving Full Features to {FEATURES_FULL}")
    save_data(df_full, FEATURES_FULL)
    
    # Verify
    logger.info("Verification - Snapshot Columns:")
    logger.info(df_snap.columns.tolist())
    
    if 'is_high_cost' not in df_snap.columns:
        logger.error("CRITICAL: 'is_high_cost' target missing from Snapshot features!")
    else:
        logger.info(f"Target 'is_high_cost' present. Positive rate: {df_snap['is_high_cost'].mean():.2%}")
    
    logger.info("Done.")

if __name__ == "__main__":
    main()
