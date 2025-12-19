#!/usr/bin/env python3
"""
Snapshot-Only vs Full-PR Comparison (Phase 1: Leakage Shield)

This script trains two versions of the model:
1. Snapshot-Only: Uses only features available at PR creation (title, body, first commit)
2. Full-PR: Uses all commits (current approach - potential leakage)

Compares AUC to validate the claim of "submission-time prediction".
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

# Paths
DATA_DIR = Path("data")
PROCESSED_DIR = DATA_DIR / "processed"
OUTPUT_DIR = Path("outputs/leakage_check")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def load_features():
    """Load feature matrix."""
    features_path = PROCESSED_DIR / "valid_features.csv"
    df = pd.read_csv(features_path)
    return df

def define_feature_groups(df):
    """Define snapshot-only vs full features."""
    
    all_cols = df.columns.tolist()
    
    # Snapshot features: available at PR creation (title/body only, no commit stats)
    snapshot_features = [
        # Intent (from title/body at creation)
        'has_plan', 'title_len', 'body_len',
        # Agent identity
        'author_agent',
        # Time features
        'created_hour', 'created_day_of_week',
        # Language
        'lang_extension',
    ]
    
    # Filter to columns that exist
    snapshot_features = [c for c in snapshot_features if c in all_cols]
    
    # Full features: includes aggregate commit stats (potential leakage)
    full_features = snapshot_features + [
        'additions', 'deletions', 'changed_files',
        'touches_ci', 'touches_tests', 'touches_deps',
    ]
    full_features = [c for c in full_features if c in all_cols]
    
    print(f"Snapshot features ({len(snapshot_features)}): {snapshot_features}")
    print(f"Full features ({len(full_features)}): {full_features}")
    
    return snapshot_features, full_features

def prepare_data(df, target_col='is_high_cost'):
    """Prepare train/test split with repo-disjoint protocol."""
    
    # Create high cost target from comments + reviews
    if 'comments' in df.columns and 'review_comments' in df.columns:
        df['effort_score'] = df['comments'].fillna(0) + df['review_comments'].fillna(0)
        threshold = df['effort_score'].quantile(0.80)
        df['is_high_cost'] = (df['effort_score'] >= threshold).astype(int)
        print(f"Created is_high_cost target (threshold={threshold:.0f}, positives={df['is_high_cost'].sum():,})")
    
    # Use is_ghost if available for ghosting analysis
    if 'is_ghost' in df.columns:
        df['is_ghost'] = df['is_ghost'].astype(int)
        print(f"Using is_ghost target (positives={df['is_ghost'].sum():,})")
    
    # Repo-disjoint split
    repo_col = 'repo_full_name' if 'repo_full_name' in df.columns else 'repo_name'
    if repo_col not in df.columns:
        print("Warning: No repo column, using random split")
        from sklearn.model_selection import train_test_split
        return train_test_split(df, test_size=0.2, random_state=42)
    
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(splitter.split(df, groups=df[repo_col]))
    
    return df.iloc[train_idx], df.iloc[test_idx]

def train_and_evaluate(train_df, test_df, features, target_col='is_high_cost'):
    """Train LightGBM and evaluate AUC."""
    
    # Filter to only numeric/categorical features
    X_train = train_df[features].copy()
    X_test = test_df[features].copy()
    y_train = train_df[target_col]
    y_test = test_df[target_col]
    
    # Encode categoricals
    cat_cols = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
    for col in cat_cols:
        X_train[col] = X_train[col].astype('category')
        X_test[col] = X_test[col].astype('category')
    
    # Train
    model = lgb.LGBMClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        class_weight='balanced',
        random_state=42,
        verbose=-1
    )
    
    model.fit(X_train, y_train, categorical_feature=cat_cols)
    
    # Evaluate
    y_pred = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_pred)
    
    return auc, model

def main():
    print("="*60)
    print("SNAPSHOT-ONLY vs FULL-PR LEAKAGE CHECK")
    print("="*60)
    
    # Load data
    df = load_features()
    print(f"Loaded {len(df):,} PRs with {len(df.columns)} features")
    
    # Define feature groups
    snapshot_features, full_features = define_feature_groups(df)
    
    if len(snapshot_features) < 3:
        print("Error: Not enough snapshot features available")
        return
    
    # Prepare data
    train_df, test_df = prepare_data(df)
    print(f"Train: {len(train_df):,}, Test: {len(test_df):,}")
    
    # Train both versions
    print("\n[Training Snapshot-Only Model]")
    snapshot_auc, _ = train_and_evaluate(train_df, test_df, snapshot_features)
    print(f"  Snapshot-Only AUC: {snapshot_auc:.3f}")
    
    print("\n[Training Full-PR Model]")
    full_auc, _ = train_and_evaluate(train_df, test_df, full_features)
    print(f"  Full-PR AUC: {full_auc:.3f}")
    
    # Compare
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)
    print(f"  Snapshot-Only AUC: {snapshot_auc:.3f}")
    print(f"  Full-PR AUC:       {full_auc:.3f}")
    print(f"  Difference:        {full_auc - snapshot_auc:+.3f}")
    
    if full_auc - snapshot_auc > 0.05:
        print("\n  [WARNING] Full-PR significantly outperforms Snapshot-Only.")
        print("     This suggests potential leakage from post-submission features.")
    else:
        print("\n  [OK] Snapshot-Only is competitive with Full-PR.")
        print("    The model is deployable at submission time.")
    
    # Save results
    results = pd.DataFrame({
        'Model': ['Snapshot-Only', 'Full-PR'],
        'AUC': [snapshot_auc, full_auc],
        'Features': [len(snapshot_features), len(full_features)]
    })
    results.to_csv(OUTPUT_DIR / "leakage_check_results.csv", index=False)
    print(f"\nSaved results to {OUTPUT_DIR / 'leakage_check_results.csv'}")

if __name__ == "__main__":
    main()
