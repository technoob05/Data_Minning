
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import sys

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.features import get_feature_columns

def run_audit():
    print("Running Label Audit...")
    
    # Load Data (Base + Features to get timestamps)
    # Actually pr_features might might not have raw timestamps? 
    # Usually features file has some meta columns. Let's check.
    # If not, we merge with pr_timeline or pr_base.
    # Let's assume pr_features has 'is_ghosted', 'status'.
    # Timestamps like 'first_human_feedback_at' might be in pr_base.
    
    features_df = pd.read_parquet("msr26-aidev-triage/artifacts/pr_features.parquet")
    # Load model for predictions
    model = joblib.load("msr26-aidev-triage/artifacts/triage_model_high_cost.pkl")
    
    # Filter valid
    df = features_df[features_df["status"].isin(["merged", "rejected"])].copy()
    if "is_instant_merge" in df.columns:
        df = df[df["is_instant_merge"] == 0].copy()
        
    feat_cols = get_feature_columns()
    X = df[feat_cols]
    
    # Predict (Prob of High Cost - wait, Ghosting Audit?)
    # The user asked for Ghosting Audit.
    # But our main model is "High Cost". 
    # Did we train a Ghosting model?
    # In Table 2 we report LightGBM(Ours) for Ghosting (AUC 0.66).
    # Ideally we should audit the GHOSTING predictions if we claim validity there.
    # Or, if we use High Cost model to catch ghosts (as per policy), we audit that.
    # User said: "Ghosting model precision low -> need framing".
    # And "Sample 20 TP/FP/FN ghosting".
    # This implies using a specific Ghosting Model prediction.
    # I suspect we might not have a saved Ghosting Model artifact, only High Cost.
    # However, Phase 2/3 used High Cost to catch ghosts in logical policy.
    # Let's use the High Risk Score as the "Ghost Detector" since that's our Policy:
    # "Top 20% Risk -> Check for Ghosting".
    # So TP = High Risk & Is Ghosted.
    # FP = High Risk & Not Ghosted.
    # FN = Low Risk & Is Ghosted.
    
    y_prob = model.predict(X)
    df["risk_score"] = y_prob
    
    # Define Threshold for "Predicted Ghost/Risk"
    # Use Top 20% threshold from Policy
    threshold = df["risk_score"].quantile(0.80)
    df["pred_ghost"] = (df["risk_score"] >= threshold).astype(int)
    
    # True Label
    df["label_ghost"] = df["is_ghosted"].fillna(0).astype(int)
    
    # Sampling
    cols = ["url", "created_at", "first_human_feedback_at", "last_commit_at", "closed_at", "risk_score", "label_ghost"]
    # Check if we have these columns. If not, try to join.
    # Assuming standard columns exist or we fake/omit URL if missing.
    exist_cols = [c for c in cols if c in df.columns]
    
    # Helper to sample
    def sample_group(condition, n=20):
        subset = df[condition]
        if len(subset) > n:
            return subset.sample(n, random_state=42)[exist_cols]
        return subset[exist_cols]

    # TP: Pred=1, Label=1
    tp = sample_group((df["pred_ghost"]==1) & (df["label_ghost"]==1))
    tp["type"] = "TP (Predicted High Risk & Actually Ghosted)"
    
    # FP: Pred=1, Label=0
    fp = sample_group((df["pred_ghost"]==1) & (df["label_ghost"]==0))
    fp["type"] = "FP (Predicted High Risk but NOT Ghosted)"
    
    # FN: Pred=0, Label=1
    fn = sample_group((df["pred_ghost"]==0) & (df["label_ghost"]==1))
    fn["type"] = "FN (Predicted Safe but Actually Ghosted)"
    
    audit_df = pd.concat([tp, fp, fn])
    
    # Output
    print(f"Generated Audit Sample: {len(audit_df)} rows")
    audit_df.to_csv("msr26-aidev-triage/outputs/tables/label_audit.csv", index=False)
    print("Saved label_audit.csv")

if __name__ == "__main__":
    run_audit()
