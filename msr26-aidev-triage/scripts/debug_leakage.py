
import sys
from pathlib import Path
import pandas as pd
from sklearn.metrics import roc_auc_score
import logging

# Add Project Root to Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.config import FEATURES_SNAPSHOT
from src.utils import load_data, setup_logger

logger = setup_logger("debug_leakage")

def check_leakage():
    logger.info("Checking for data leakage in Snapshot features...")
    
    try:
        df = load_data(FEATURES_SNAPSHOT)
    except FileNotFoundError:
        logger.error(f"File not found: {FEATURES_SNAPSHOT}")
        return

    target = 'is_high_cost'
    if target not in df.columns:
        logger.error(f"Target {target} not found in dataframe")
        return

    logger.info(f"Analyzing {len(df)} rows. Target: {target}")
    
    # Calculate univariate AUC for each feature
    leakage_suspects = []
    
    for col in df.columns:
        if col in [target, 'effort_score', 'is_ghosted', 'is_merged', 'id', 'repo_full_name', 'created_at', 'merged_at']:
            continue
            
        try:
            # Handle categoricals basically
            if df[col].dtype == 'object' or df[col].dtype.name == 'category':
                # Skip high cardinality strings or encode them
                if df[col].nunique() > 100:
                    continue
                series = df[col].astype('category').cat.codes
            else:
                series = df[col]
                
            # Drop NaNs just for this check
            valid_idx = series.notna() & df[target].notna()
            if valid_idx.sum() < 10:
                continue
                
            auc = roc_auc_score(df.loc[valid_idx, target], series.loc[valid_idx])
            
            # Flip if inverse correlation
            if auc < 0.5:
                auc = 1 - auc
                
            logger.info(f"Feature: {col:<30} AUC: {auc:.4f}")
            
            if auc > 0.95:
                leakage_suspects.append((col, auc))
                
        except Exception as e:
            logger.warning(f"Could not calc AUC for {col}: {e}")

    if leakage_suspects:
        logger.warning("\nPOSSIBLE LEAKAGE FOUND (AUC > 0.95):")
        for col, auc in leakage_suspects:
            logger.warning(f"{col}: {auc:.4f}")
    else:
        logger.info("\nNo single feature has AUC > 0.95. Leakage unlikely via direct proxy.")

if __name__ == "__main__":
    check_leakage()
