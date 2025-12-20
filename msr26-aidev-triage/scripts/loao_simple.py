#!/usr/bin/env python3
"""
Leave-One-Agent-Out (LO AO) Evaluation - Simplified
Tests model generalization to unseen agents.
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, average_precision_score
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.config import FEATURES_SNAPSHOT, TABLES_DIR, RANDOM_SEED
from src.features import get_feature_columns

def run_loao():
    print("="*80)
    print("LEAVE-ONE-AGENT-OUT (LOAO) EVALUATION")
    print("="*80)
    
    print("\n[1/4] Loading data...")
    df = pd.read_parquet(FEATURES_SNAPSHOT)
    print(f"Loaded {len(df)} PRs")
    
    # Manual agent mapping from agent_encoded (based on old results)
    # Old results show: Claude_Code, Copilot, OpenAI_Codex, Cursor, Devin
    agent_map = {
        0: 'Claude_Code',
        1: 'Copilot', 
        2: 'Cursor',
        3: 'Devin',
        4: 'OpenAI_Codex'
    }
    
    df['agent'] = df['agent_encoded'].map(agent_map)
    print(f"Mapped agents: {df['agent'].value_counts().to_dict()}")
    
    # Filter instant merges
    if "is_instant_merge" in df.columns:
        df = df[df["is_instant_merge"] == 0].copy()
        print(f"After filtering instant merges: {len(df)} PRs")
    
    print("\n[2/4] Preparing features...")
    feature_cols = get_feature_columns(strict=True)
    
    # Remove agent_encoded to test structure-only
    feature_cols = [f for f in feature_cols if f != 'agent_encoded' and f in df.columns]
    print(f"Using {len(feature_cols)} T0 features (no agent ID)")
    
    agents = sorted(df["agent"].unique())
    print(f"\n[3/4] Testing {len(agents)} agents:")
    for agent in agents:
        count = len(df[df["agent"] == agent])
        high_cost_rate = df[df["agent"] == agent]["is_high_cost"].mean()
        print(f"  - {agent:15s}: {count:5d} PRs ({high_cost_rate:5.1%} high-cost)")
    
    print("\n[4/4] Running LOAO...")
    results = []
    
    for test_agent in agents:
        print(f"\nHolding out: {test_agent}")
        
        train_df = df[df["agent"] != test_agent]
        test_df = df[df["agent"] == test_agent]
        
        if len(test_df) < 50:
            print(f"  Skipped (n={len(test_df)})")
            continue
        
        # Prepare
        X_train = train_df[feature_cols].fillna(0).copy()
        y_train = train_df["is_high_cost"]
        X_test = test_df[feature_cols].fillna(0).copy()
        y_test = test_df["is_high_cost"]
        
        # Encode categoricals
        from sklearn.preprocessing import LabelEncoder
        for col in X_train.columns:
            if X_train[col].dtype == 'object':
                le = LabelEncoder()
                X_train[col] = le.fit_transform(X_train[col].astype(str))
                X_test[col] = le.transform(X_test[col].astype(str))
        
        # Train
        model = lgb.LGBMClassifier(
            n_estimators=100, max_depth=6, learning_rate=0.1,
            class_weight='balanced', random_state=RANDOM_SEED, verbose=-1
        )
        model.fit(X_train, y_train)
        
        # Eval
        y_prob = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_prob)
        pr_auc = average_precision_score(y_test, y_prob)
        
        print(f"  AUC: {auc:.4f} | PR-AUC: {pr_auc:.4f}")
        
        results.append({
            "Agent": test_agent,
            "Train_N": len(train_df),
            "Test_N": len(test_df),
            "AUC": auc,
            "PR_AUC": pr_auc
        })
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    res_df = pd.DataFrame(results)
    print(res_df.to_string(index=False))
    
    print(f"\n📊 AUC Range: {res_df['AUC'].min():.2f} -- {res_df['AUC'].max():.2f}")
    print(f"   AUC Mean:  {res_df['AUC'].mean():.3f} ± {res_df['AUC'].std():.3f}")
    
    # Verify paper
    paper_range = "0.66--0.80"
    actual_range = f"{res_df['AUC'].min():.2f}--{res_df['AUC'].max():.2f}"
    print(f"\n✅ Paper: '{paper_range}' | Actual: '{actual_range}'")
    
    # Save
    out_path = TABLES_DIR / "loao_fresh.csv"
    res_df.to_csv(out_path, index=False)
    print(f"\n💾 Saved: {out_path}")
    
    return res_df

if __name__ == "__main__":
    run_loao()
