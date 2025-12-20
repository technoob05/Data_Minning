#!/usr/bin/env python3
"""
Leave-One-Agent-Out (LOAO) Evaluation
Tests model generalization to unseen agents by holding out each agent and training on others.
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, average_precision_score
from pathlib import Path
import sys

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.config import FEATURES_SNAPSHOT, TABLES_DIR, RANDOM_SEED
from src.features import get_feature_columns

def run_loao():
    print("="*80)
    print("LEAVE-ONE-AGENT-OUT (LOAO) EVALUATION")
    print("="*80)
    
    print("\n[1/4] Loading data...")
    df = pd.read_parquet(FEATURES_SNAPSHOT)
    print(f"Loaded {len(df)} PRs from FEATURES_SNAPSHOT")
    
    # Check available columns
    print(f"Available columns: {df.columns.tolist()[:10]}... (+{len(df.columns)-10} more)")
    
    # Filter instant merges if exists
    if "is_instant_merge" in df.columns:
        before = len(df)
        df = df[df["is_instant_merge"] == 0].copy()
        print(f"Filtered instant merges: {before} -> {len(df)} PRs")
    
    # Check required columns
    required = ['agent', 'is_high_cost']
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"ERROR: Missing required columns: {missing}")
        print(f"Available columns: {df.columns.tolist()}")
        return
    
    print("\n[2/4] Preparing features...")
    # Use T0 features (strict=True for creation-time only)
    feature_cols = get_feature_columns(strict=True)
    
    # Filter to available features
    available_features = [f for f in feature_cols if f in df.columns]
    print(f"Using {len(available_features)}/{len(feature_cols)} T0 features")
    
    if len(available_features) < 5:
        print(f"ERROR: Too few features available ({len(available_features)})")
        return
    
    feature_cols = available_features
    
    # Get unique agents
    agents = df["agent"].unique()
    print(f"\n[3/4] Found {len(agents)} unique agents:")
    for i, agent in enumerate(agents, 1):
        count = len(df[df["agent"] == agent])
        pos_rate = df[df["agent"] == agent]["is_high_cost"].mean()
        print(f"  {i}. {agent:20s}: {count:5d} PRs (High Cost: {pos_rate:.1%})")
    
    print("\n[4/4] Running LOAO evaluation...")
    results = []
    
    for idx, test_agent in enumerate(agents, 1):
        print(f"\n[{idx}/{len(agents)}] Holding out: {test_agent}")
        
        # Split train/test
        train_df = df[df["agent"] != test_agent].copy()
        test_df = df[df["agent"] == test_agent].copy()
        
        # Skip if test set too small
        if len(test_df) < 50:
            print(f"  ⚠️  Skipping (test size {len(test_df)} < 50)")
            continue
        
        print(f"  Train: {len(train_df)} PRs, Test: {len(test_df)} PRs")
        
        # Prepare features
        X_train = train_df[feature_cols].copy()
        y_train = train_df["is_high_cost"].values
        
        X_test = test_df[feature_cols].copy()
        y_test = test_df["is_high_cost"].values
        
        # Handle missing values
        for col in feature_cols:
            X_train[col] = X_train[col].fillna(0)
            X_test[col] = X_test[col].fillna(0)
        
        # Train model (same config as main model)
        model = lgb.LGBMClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            class_weight='balanced',
            random_state=RANDOM_SEED,
            verbose=-1
        )
        
        model.fit(X_train, y_train)
        
        # Predict
        y_prob = model.predict_proba(X_test)[:, 1]
        
        # Metrics
        auc = roc_auc_score(y_test, y_prob)
        pr_auc = average_precision_score(y_test, y_prob)
        
        print(f"  ✓ AUC: {auc:.4f}, PR-AUC: {pr_auc:.4f}")
        
        results.append({
            "Left_Out_Agent": test_agent,
            "Train_Size": len(train_df),
            "Test_Size": len(test_df),
            "Test_High_Cost_Rate": y_test.mean(),
            "AUC": auc,
            "PR_AUC": pr_auc
        })
    
    # Summary
    print("\n" + "="*80)
    print("LOAO RESULTS SUMMARY")
    print("="*80)
    
    res_df = pd.DataFrame(results)
    print(res_df.to_string(index=False))
    
    if len(results) > 0:
        print(f"\n📊 STATISTICS:")
        print(f"  AUC Range:  {res_df['AUC'].min():.4f} - {res_df['AUC'].max():.4f}")
        print(f"  AUC Mean:   {res_df['AUC'].mean():.4f} ± {res_df['AUC'].std():.4f}")
        print(f"  AUC Median: {res_df['AUC'].median():.4f}")
        
        # Check paper claim: "0.66--0.80"
        paper_min, paper_max = 0.66, 0.80
        actual_min, actual_max = res_df['AUC'].min(), res_df['AUC'].max()
        
        print(f"\n✅ PAPER VERIFICATION:")
        print(f"  Paper claim: AUC 0.66--0.80")
        print(f"  Actual range: AUC {actual_min:.2f}--{actual_max:.2f}")
        
        if abs(actual_min - paper_min) < 0.05 and abs(actual_max - paper_max) < 0.05:
            print(f"  ✓ VERIFIED: Numbers match paper!")
        else:
            print(f"  ⚠️  MISMATCH: Paper needs update!")
    
    # Save results
    output_path = TABLES_DIR / "loao_results_fresh.csv"
    res_df.to_csv(output_path, index=False)
    print(f"\n💾 Saved to: {output_path}")
    
    return res_df

if __name__ == "__main__":
    results = run_loao()
