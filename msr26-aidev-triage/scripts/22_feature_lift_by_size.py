#!/usr/bin/env python3
"""
Feature Lift Analysis by Size Quartile
Answers: "In each size bin, which features add value beyond raw size?"
Computes Precision@20% for size-only vs full model within each quartile.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import precision_score, recall_score
from sklearn.linear_model import LogisticRegression
import lightgbm as lgb
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.config import ARTIFACTS_DIR, TABLES_DIR

# Splits directory
SPLITS_DIR = Path(__file__).resolve().parents[1] / "scripts" / "splits"

def compute_size_quartiles(df):
    """Divide PRs into size quartiles."""
    df['size'] = df['additions'].fillna(0) + df['deletions'].fillna(0)
    df['size_quartile'] = pd.qcut(df['size'], q=4, labels=['Q1_Small', 'Q2_Medium', 'Q3_Large', 'Q4_XL'], duplicates='drop')
    return df

def train_models_within_quartile(df_quartile, feature_sets):
    """Train size-only vs full model within a single quartile."""
    
    # Ensure target exists
    if 'is_high_cost' not in df_quartile.columns:
        print("  Error: is_high_cost not found")
        return None
    
    # Split train/test
    from sklearn.model_selection import train_test_split
    train, test = train_test_split(df_quartile, test_size=0.3, random_state=42, stratify=df_quartile['is_high_cost'])
    
    results = {}
    
    for name, features in feature_sets.items():
        # Check features exist
        available_features = [f for f in features if f in train.columns]
        if not available_features:
            print(f"  Warning: No features available for {name}")
            continue
        
        X_train = train[available_features].fillna(0)
        X_test = test[available_features].fillna(0)
        y_train = train['is_high_cost']
        y_test = test['is_high_cost']
        
        # Train model
        if name == 'size_only':
            model = LogisticRegression(max_iter=1000, random_state=42)
        else:
            model = lgb.LGBMClassifier(n_estimators=50, max_depth=4, random_state=42, verbose=-1)
        
        model.fit(X_train, y_train)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Get top 20%
        top_k = int(len(y_test) * 0.2)
        top_indices = np.argsort(y_pred_proba)[::-1][:top_k]
        y_pred_top_k = np.zeros(len(y_test))
        y_pred_top_k[top_indices] = 1
        
        # Metrics
        precision = precision_score(y_test, y_pred_top_k, zero_division=0)
        recall = recall_score(y_test, y_pred_top_k, zero_division=0)
        
        results[name] = {
            'precision_at_20': precision,
            'recall_at_20': recall
        }
    
    return results

def main():
    print("=" * 70)
    print("FEATURE LIFT ANALYSIS BY SIZE QUARTILE")
    print("=" * 70)
    
    # Load features
    feat_path = ARTIFACTS_DIR / "pr_features.parquet"
    if not feat_path.exists():
        print(f"Error: {feat_path} not found")
        return
    
    df = pd.read_parquet(feat_path)
    print(f"Loaded {len(df)} PRs")
    
    # Ensure High Cost label
    if 'is_high_cost' not in df.columns:
        print("Creating High Cost labels...")
        if 'effort_score' not in df.columns:
            df['effort_score'] = df.get('num_comments', 0).fillna(0) + df.get('num_reviews', 0).fillna(0)
        threshold = df['effort_score'].quantile(0.8)
        df['is_high_cost'] = (df['effort_score'] > threshold).astype(int)
    
    # Compute size quartiles
    print("\n[1/3] Computing size quartiles...")
    df = compute_size_quartiles(df)
    print(df.groupby('size_quartile')['size'].agg(['count', 'min', 'max', 'median']))
    
    # Define feature sets
    print("\n[2/3] Defining feature sets...")
    
    size_features = ['additions', 'deletions', 'total_changes', 'files_changed']
    
    # Full feature set (subset of T0 features)
    full_features = size_features + [
        'has_plan',
        'body_length',
        'title_length',
        'touches_ci',
        'touches_tests',
        'touches_docs',
        'touches_deps',
        'touches_src',
        'entropy'
    ]
    
    # Filter to available features
    available_full = [f for f in full_features if f in df.columns]
    available_size = [f for f in size_features if f in df.columns]
    
    feature_sets = {
        'size_only': available_size,
        'full_model': available_full
    }
    
    print(f"  Size-only features: {len(available_size)}")
    print(f"  Full model features: {len(available_full)}")
    
    # Train within each quartile
    print("\n[3/3] Training models within each quartile...")
    
    all_results = []
    
    for quartile in ['Q1_Small', 'Q2_Medium', 'Q3_Large', 'Q4_XL']:
        print(f"\n  Processing {quartile}...")
        
        df_q = df[df['size_quartile'] == quartile].copy()
        
        if len(df_q) < 100:
            print(f"  Skipping {quartile} (insufficient data: {len(df_q)} PRs)")
            continue
        
        results = train_models_within_quartile(df_q, feature_sets)
        
        if results:
            size_prec = results['size_only']['precision_at_20']
            full_prec = results['full_model']['precision_at_20']
            lift = full_prec - size_prec
            
            all_results.append({
                'quartile': quartile,
                'n_prs': len(df_q),
                'size_only_precision': size_prec,
                'full_model_precision': full_prec,
                'lift': lift,
                'lift_pct': (lift / size_prec * 100) if size_prec > 0 else 0,
                'size_only_recall': results['size_only']['recall_at_20'],
                'full_model_recall': results['full_model']['recall_at_20']
            })
            
            print(f"    Size-only Prec@20%: {size_prec:.3f}")
            print(f"    Full model Prec@20%: {full_prec:.3f}")
            print(f"    Lift: {lift:+.3f} ({lift/size_prec*100:+.1f}%)")
    
    # Save results
    results_df = pd.DataFrame(all_results)
    out_path = TABLES_DIR / "feature_lift_by_quartile.csv"
    results_df.to_csv(out_path, index=False)
    
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(results_df.to_string(index=False))
    print(f"\n[OK] Saved to {out_path}")
    
    # Interpretation
    print("\n" + "=" * 70)
    print("KEY FINDINGS")
    print("=" * 70)
    
    avg_lift = results_df['lift'].mean()
    positive_lift_count = (results_df['lift'] > 0).sum()
    
    print(f"\nAverage Lift: {avg_lift:+.3f} precision points")
    print(f"Quartiles with positive lift: {positive_lift_count}/{len(results_df)}")
    
    if avg_lift > 0.02:
        print("\n[OK] Full model provides MEANINGFUL lift beyond size (>2pp average)")
        print("  -> Justifies inclusion of plan, file-type, context features")
    elif avg_lift > 0:
        print("\n[WARN] Full model provides MODEST lift beyond size (<2pp)")
        print("  -> Size dominates, but additional features help")
    else:
        print("\n[ERROR] Full model shows NO consistent lift")
        print("  -> Size-only may be sufficient")
    
    # Best performing quartile
    best_quartile = results_df.loc[results_df['lift'].idxmax()]
    print(f"\nBest lift in: {best_quartile['quartile']} ({best_quartile['lift']:+.3f} precision)")
    print(f"  Interpretation: {'Plan/context features' if 'Q1' in best_quartile['quartile'] else 'File diversity'} matter most here")

if __name__ == "__main__":
    main()
