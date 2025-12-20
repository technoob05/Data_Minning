import pandas as pd
import numpy as np
import lightgbm as lgb
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupShuffleSplit
from pathlib import Path
import sys

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parents[1]))

def plot_topk_coverage():
    print("Loading data for Top-K Coverage...")
    from src.config import FEATURES_FULL, ARTIFACTS_DIR, FIGURES_DIR
    df = pd.read_parquet(FEATURES_FULL)
    
    # Target Construction
    if "effort_score" not in df.columns:
        df['effort_score'] = df['num_comments'].fillna(0) + df['num_reviews'].fillna(0)
    
    cost_threshold = df['is_high_cost'] if 'is_high_cost' in df.columns else (df['effort_score'] >= df['effort_score'].quantile(0.80)).astype(int)
    if 'is_high_cost' not in df.columns:
        df['is_high_cost'] = cost_threshold

    # Split
    print("Splitting data (Repo-Disjoint)...")
    from src.features import get_feature_columns
    feature_cols = get_feature_columns(strict=False)
    available_feats = [f for f in feature_cols if f in df.columns]
    
    group_col = "repo_full_name"
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(gss.split(df, groups=df[group_col]))
    
    test_df = df.iloc[test_idx].copy()
    X_test = test_df[available_feats].copy()
    y_test = test_df["is_high_cost"]
    test_effort = test_df["effort_score"]

    # Predictions
    print("Loading Model & Predicting...")
    # Categorical handling
    for col in X_test.select_dtypes(include=['object']).columns:
        X_test[col] = X_test[col].astype('category')
        
    model_path = ARTIFACTS_DIR / "triage_model_high_cost.pkl"
    import joblib
    model = joblib.load(model_path)
    
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
    else:
        y_prob = model.predict(X_test)
        
    # Analysis
    res_df = pd.DataFrame({
        "prob": y_prob,
        "actual": y_test,
        "effort": test_effort
    })
    res_df = res_df.sort_values("prob", ascending=False)
    
    total_effort = res_df["effort"].sum()
    res_df["cum_effort"] = res_df["effort"].cumsum()
    res_df["cum_effort_pct"] = res_df["cum_effort"] / total_effort
    res_df["pct_pr_reviewed"] = np.arange(1, len(res_df) + 1) / len(res_df)
    
    # Metrics
    k_20 = int(len(res_df) * 0.20)
    recall_20_effort = res_df.iloc[:k_20]["effort"].sum() / total_effort
    
    # Ideal
    ideal_df = res_df.sort_values("effort", ascending=False)
    ideal_recall_20 = ideal_df.iloc[:k_20]["effort"].sum() / total_effort
    
    print(f"\n>>> RECALL@20% EFFORT (MODEL): {recall_20_effort:.4f} <<<")
    print(f">>> RECALL@20% EFFORT (IDEAL): {ideal_recall_20:.4f} <<<")
    
    # Plot
    plt.figure(figsize=(8, 6))
    plt.plot(res_df["pct_pr_reviewed"], res_df["cum_effort_pct"], label="Model Triage", linewidth=2, color="blue")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random")
    plt.title("Top-K Triage Utility: High Cost PRs")
    plt.xlabel("Fraction of PRs Reviewed (Sorted by Risk)")
    plt.ylabel("Fraction of Total Effort Covered")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    output_path = FIGURES_DIR / "topk_coverage.png"
    plt.savefig(output_path)
    print(f"Saved {output_path}")


if __name__ == "__main__":
    plot_topk_coverage()
