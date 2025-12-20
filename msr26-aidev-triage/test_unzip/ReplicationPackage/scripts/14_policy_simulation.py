
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parents[1]))

def run_policy_simulation():
    print("Loading data for Policy Simulation...")
    from src.config import FEATURES_FULL, RANDOM_SEED, PROJECT_ROOT
    df = pd.read_parquet(FEATURES_FULL)
    from src.features import get_feature_columns
    from sklearn.model_selection import GroupShuffleSplit
    
    # Define stats
    df["is_merged"] = df["merged_at"].notna()
    df["status"] = df["is_merged"].map({True: "merged", False: "rejected"})
    
    # Filter for relevant (already done but safe)
    df = df[df["status"].isin(["merged", "rejected"])].copy()
    
    if "is_instant_merge" in df.columns:
        df = df[df["is_instant_merge"] == 0].copy()

    # Split (Same seed as always)
    # Use repo_full_name if repo_id missing
    group_col = "repo_full_name"
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(gss.split(df, groups=df[group_col]))
    
    test_df = df.iloc[test_idx].copy()
    
    # Load Model
    import joblib
    model_path = Path("artifacts/triage_model_high_cost.pkl")
    if not model_path.exists():
        model_path = PROJECT_ROOT / "artifacts" / "triage_model_high_cost.pkl"

    model = joblib.load(model_path)
    
    # Predict
    feature_cols = get_feature_columns(strict=False)
    available_feats = [f for f in feature_cols if f in test_df.columns]
    
    X_test = test_df[available_feats]
    
    # Category handling
    for col in X_test.select_dtypes(include=['object']).columns:
        X_test[col] = X_test[col].astype('category')
        
    y_prob = model.predict_proba(X_test)[:, 1]
    test_df["risk_score"] = y_prob

    
    # Simulation
    # Sort by risk (High to Low)
    test_df = test_df.sort_values("risk_score", ascending=False)

    
    total_effort = test_df["effort_score"].sum()
    total_ghosts = test_df["is_ghosted"].fillna(0).sum()
    total_prs = len(test_df)
    
    print("\n=== Policy Simulation Table ===")
    results = []
    
    for k_pct in [10, 20, 30]:
        k = int(total_prs * (k_pct / 100.0))
        top_k = test_df.head(k)
        
        # 1. Effort Captured (of total)
        effort_captured = top_k["effort_score"].sum() / total_effort * 100
        
        # 2. Ghosting Recall (of total ghosts)
        # Note: Ghosting is only defined for rejected PRs. 
        # But we want to know: of ALL ghosts in the test set, how many did we catch?
        ghosts_caught = top_k["is_ghosted"].fillna(0).sum()
        ghosting_recall = ghosts_caught / total_ghosts * 100 if total_ghosts > 0 else 0
        
        # 3. False Alarm Cost (Merged PRs flagged)
        # "Noise" = Merged PRs inside Top K / Total K
        merged_in_top_k = top_k[top_k["status"] == "merged"].shape[0]
        noise_rate = merged_in_top_k / k * 100
        
        print(f"Top {k_pct}%: Effort={effort_captured:.1f}%, Ghosting Recall={ghosting_recall:.1f}%, Noise={noise_rate:.1f}%")
        
        results.append({
            "Budget": f"Top {k_pct}%",
            "Effort Saved": f"{effort_captured:.1f}%",
            "Ghosting Caught": f"{ghosting_recall:.1f}%",
            "False Alarm Rate": f"{noise_rate:.1f}%"
        })
        
    res_df = pd.DataFrame(results)
    from src.config import TABLES_DIR
    res_df.to_csv(TABLES_DIR / "policy_simulation.csv", index=False)

if __name__ == "__main__":
    run_policy_simulation()
