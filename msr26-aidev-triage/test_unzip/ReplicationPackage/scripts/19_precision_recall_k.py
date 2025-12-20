"""
Calculate Precision@K and Recall@K for Policy Table
"""
import pandas as pd
import numpy as np
import joblib
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.features import get_feature_columns
from sklearn.model_selection import GroupShuffleSplit

def calculate_precision_recall_at_k():
    print("Calculating Precision@K and Recall@K...")
    
    df = pd.read_parquet("msr26-aidev-triage/artifacts/pr_features.parquet")
    df = df[df["status"].isin(["merged", "rejected"])].copy()
    if "is_instant_merge" in df.columns:
        df = df[df["is_instant_merge"] == 0].copy()
    
    # Split (same as training)
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(gss.split(df, groups=df["repo_id"]))
    test_df = df.iloc[test_idx].copy()
    
    # Load model and predict
    model = joblib.load("msr26-aidev-triage/artifacts/triage_model_high_cost.pkl")
    feat_cols = get_feature_columns()
    X_test = test_df[feat_cols]
    y_prob = model.predict(X_test)
    test_df["risk_score"] = y_prob
    
    # Sort by risk
    test_df = test_df.sort_values("risk_score", ascending=False)
    
    # Ground truth
    test_df["true_high_cost"] = test_df["is_high_cost"]
    test_df["true_ghosted"] = test_df["is_ghosted"].fillna(0).astype(int)
    
    total_high_cost = test_df["true_high_cost"].sum()
    total_ghosted = test_df["true_ghosted"].sum()
    total_effort = test_df["effort_score"].sum()
    n_total = len(test_df)
    
    print(f"\nTest Set Stats:")
    print(f"  Total PRs: {n_total}")
    print(f"  Total High Cost: {int(total_high_cost)}")
    print(f"  Total Ghosted: {int(total_ghosted)}")
    print(f"  Total Effort: {total_effort:.0f}")
    
    results = []
    print("\n" + "=" * 80)
    print(f"{'K%':<10}{'N Flagged':<12}{'Prec@K (HC)':<15}{'Rec@K (HC)':<15}{'Rec@K (Ghost)':<15}{'Effort%':<12}")
    print("=" * 80)
    
    for k_pct in [10, 20, 30]:
        k = int(n_total * k_pct / 100)
        top_k = test_df.head(k)
        
        # Precision@K for High Cost (% of flagged that are truly high cost)
        tp_hc = top_k["true_high_cost"].sum()
        precision_k = tp_hc / k * 100
        
        # Recall@K for High Cost (% of all high cost that are captured)
        recall_hc = tp_hc / total_high_cost * 100 if total_high_cost > 0 else 0
        
        # Recall@K for Ghosting
        tp_ghost = top_k["true_ghosted"].sum()
        recall_ghost = tp_ghost / total_ghosted * 100 if total_ghosted > 0 else 0
        
        # Effort captured
        effort_captured = top_k["effort_score"].sum() / total_effort * 100
        
        print(f"{k_pct}%{' '*6}{k:<12}{precision_k:<15.1f}{recall_hc:<15.1f}{recall_ghost:<15.1f}{effort_captured:<12.1f}")
        
        results.append({
            "Budget": f"Top {k_pct}%",
            "N_Flagged": k,
            "Precision_HC": f"{precision_k:.1f}%",
            "Recall_HC": f"{recall_hc:.1f}%",
            "Recall_Ghost": f"{recall_ghost:.1f}%",
            "Effort_Saved": f"{effort_captured:.1f}%"
        })
    
    res_df = pd.DataFrame(results)
    res_df.to_csv("msr26-aidev-triage/outputs/tables/policy_precision_recall.csv", index=False)
    print("\nSaved policy_precision_recall.csv")

if __name__ == "__main__":
    calculate_precision_recall_at_k()
