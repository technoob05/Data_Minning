#!/usr/bin/env python3
"""
Leave-One-Agent-Out (LOAO) Evaluation - FIXED VERSION
Tests model generalization to unseen agents.
Uses agent_encoded as proxy since agent names not in FEATURES_SNAPSHOT.
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, average_precision_score
from pathlib import Path
import sys

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.config import FEATURES_SNAPSHOT, TABLES_DIR, RANDOM_SEED, DATA_DIR
from src.features import get_feature_columns

def run_loao():
    print("="*80)
    print("LEAVE-ONE-AGENT-OUT (LOAO) EVALUATION")
    print("="*80)
    
    print("\n[1/5] Loading data...")
    df = pd.read_parquet(FEATURES_SNAPSHOT)
    print(f"Loaded {len(df)} PRs from FEATURES_SNAPSHOT")
    
    # Load agent names from pull_request.parquet
    pr_base_path = DATA_DIR / "pull_request.parquet"
    if pr_base_path.exists():
        print("Loading agent names from pull_request.parquet...")
        pr_base = pd.read_parquet(pr_base_path, columns=['id', 'agent'])
        
        # Create mapping from agent_encoded to agent name
        # We need to figure out the mapping
        # Load full features that had both
        print("Creating agent_encoded -> agent name mapping...")
        
        # Simple approach: use agent_encoded as is (0,1,2,3,4 map to 5 agents)
        # But we want names. Let's check artifacts
        artifacts_pr = Path('artifacts/pr_features.parquet')
        if artifacts_pr.exists():
            temp_df = pd.read_parquet(artifacts_pr, columns=['agent', 'agent_encoded'] if 'agent' in pd.read_parquet(artifacts_pr, nrows=1).columns else ['agent_encoded'])
            if 'agent' in temp_df.columns:
                mapping = temp_df[['agent_encoded', 'agent']].drop_duplicates().set_index('agent_encoded')['agent'].to_dict()
                df['agent'] = df['agent_encoded'].map(mapping)
                print(f"  ✓ Mapped {df['agent'].notna().sum()}/{len(df)} agents")
            else:
                # Fallback: use pr_base
                # Try merging if we have common id
                pass
        
        if 'agent' not in df.columns:
            # Last resort: manually map based on known agents from old results
            # From loao_results.csv we know: Claude_Code, Copilot, OpenAI_Codex, Cursor, Devin
            agent_names = {0: 'Claude_Code', 1: 'Copilot', 2: 'Cursor', 3: 'Devin', 4: 'OpenAI_Codex'}
            df['agent'] = df['agent_encoded'].map(agent_names)
            print(f"  ✓ Manual mapping applied: {df['agent'].notna().sum()}/{len(df)} agents")
    
    # Fallback if still no agent
    if 'agent' not in df.columns or df['agent'].isna().all():
        print("  Using agent_encoded directly as agent identifier")
        df['agent'] = 'Agent_' + df['agent_encoded'].astype(str)
    
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
        return
    
    print("\n[2/5] Preparing features...")
    # Use T0 features (strict=True)
    feature_cols = get_feature_columns(strict=True)
    
    # Remove agent_encoded from features to test pure structure generalization
    if 'agent_encoded' in feature_cols:
        feature_cols.remove('agent_encoded')
        print("  ℹ️  Removed 'agent_encoded' to test structure-only generalization")
    
    # Filter to available features
    available_features = [f for f in feature_cols if f in df.columns]
    print(f"Using {len(available_features)}/{len(feature_cols)} T0 features")
    
    if len(available_features) < 5:
        print(f"ERROR: Too few features ({len(available_features)})")
        return
    
    feature_cols = available_features
    
    # Get unique agents
    agents = sorted(df["agent"].unique())
    print(f"\n[3/5] Found {len(agents)} unique agents:")
    for i, agent in enumerate(agents, 1):
        agent_df = df[df["agent"] == agent]
        count = len(agent_df)
        pos_rate = agent_df["is_high_cost"].mean()
        print(f"  {i}. {agent:20s}: {count:5d} PRs (High Cost: {pos_rate:5.1%})")
    
    print(f"\n[4/5] Running LOAO evaluation (features: {feature_cols[:5]}...)")
    results = []
    
    for idx, test_agent in enumerate(agents, 1):
        print(f"\n[{idx}/{len(agents)}] Holding out: {test_agent}")
        
        # Split
        train_df = df[df["agent"] != test_agent].copy()
        test_df = df[df["agent"] == test_agent].copy()
        
        # Skip if too small
        if len(test_df) < 50:
            print(f"  Skip (test size {len(test_df)} < 50)")
            continue
        
        print(f"  Train: {len(train_df):5d} | Test: {len(test_df):4d}")
        
        # Features
        X_train = train_df[feature_cols].copy()
        y_train = train_df["is_high_cost"].values
        X_test = test_df[feature_cols].copy()
        y_test = test_df["is_high_cost"].values
        
        # Fill NaN
        for col in feature_cols:
            X_train[col] = X_train[col].fillna(0)
            X_test[col] = X_test[col].fillna(0)
        
        # Train (same config as paper)
        model = lgb.LGBMClassifier(
            n_estimators=100, max_depth=6, learning_rate=0.1,
            class_weight='balanced', random_state=RANDOM_SEED, verbose=-1
        )
        model.fit(X_train, y_train)
        
        # Predict
        y_prob = model.predict_proba(X_test)[:, 1]
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
    print("LO AO RESULTS SUMMARY")
    print("="*80)
    
    res_df = pd.DataFrame(results)
    print(res_df.to_string(index=False))
    
    if len(results) > 0:
        print(f"\n📊 STATISTICS:")
        print(f"  AUC Range:  {res_df['AUC'].min():.4f} -- {res_df['AUC'].max():.4f}")
        print(f"  AUC Mean:   {res_df['AUC'].mean():.4f} ± {res_df['AUC'].std():.4f}")
        print(f"  AUC Median: {res_df['AUC'].median():.4f}")
        
        # Paper verification
        print(f"\n✅ PAPER VERIFICATION:")
        print(f"  Paper claim: 'AUC 0.66--0.80'")
        print(f"  Actual:      'AUC {res_df['AUC'].min():.2f}--{res_df['AUC'].max():.2f}'")
        
        if 0.64 <= res_df['AUC'].min() <= 0.68 and 0.78 <= res_df['AUC'].max() <= 0.82:
            print(f"  ✓ MATCH: Paper claim verified!")
        else:
            print(f"  ⚠️  MISMATCH: Update paper to actual range")
    
    # Save
    output_path = TABLES_DIR / "loao_results_fresh.csv"
    res_df.to_csv(output_path, index=False)
    print(f"\n💾 Saved: {output_path}")
    
    return res_df

if __name__ == "__main__":
    results = run_loao()
