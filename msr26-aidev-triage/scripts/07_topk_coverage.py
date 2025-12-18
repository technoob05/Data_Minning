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
    features_path = "msr26-aidev-triage/artifacts/pr_features.parquet"
    if not Path(features_path).exists():
        print(f"Error: {features_path} not found.")
        return

    df = pd.read_parquet(features_path, engine="fastparquet")
    
    # Filter instant merges
    if "is_instant_merge" in df.columns:
        df = df[df["is_instant_merge"] == 0].copy()
        
    target_col = "is_high_cost"
    if target_col not in df.columns:
        print(f"Target {target_col} not found.")
        return

    # Features
    # Features - Use same as training
    from src.features import get_feature_columns
    feature_cols = get_feature_columns()
    # Add any missing features from the list used in 05
    # (Simplified for now, ensuring key ones are present)
    available_feats = [f for f in feature_cols if f in df.columns]
    X = df[available_feats]
    y = df[target_col]
    
    # Repo-disjoint split (Same as 05)
    print("Splitting data (Repo-Disjoint)...")
    if "repo_id" in df.columns:
        gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        train_idx, test_idx = next(gss.split(X, y, groups=df["repo_id"]))
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        # We also need effort scores for the test set to calculate coverage
        if "effort_score" in df.columns:
             test_effort = df.iloc[test_idx]["effort_score"].fillna(0)
        else:
             test_effort = (df.iloc[test_idx]["num_comments"].fillna(0) + df.iloc[test_idx]["num_reviews"].fillna(0))
    else:
        print("Repo ID missing, skipping.")
        return

    print("Loading High Cost Model...")
    model_path = "msr26-aidev-triage/artifacts/triage_model_high_cost.pkl"
    if not Path(model_path).exists():
        print(f"Error: {model_path} not found. Run 05_train_triage.py first.")
        # Fallback to training if model not found, but warn
        print("Falling back to training new model (Consistency Warning!)")
        train_data = lgb.Dataset(X_train, label=y_train)
        test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)
        params = {
            "objective": "binary", "metric": "auc", "boosting_type": "gbdt",
            "num_leaves": 31, "learning_rate": 0.05, "feature_fraction": 0.9, "verbose": -1
        }
        model = lgb.train(params, train_data, num_boost_round=500, valid_sets=[test_data], 
                          callbacks=[lgb.early_stopping(stopping_rounds=20)])
    else:
        import joblib
        model = joblib.load(model_path)

    
    # Predictions
    print("Generating Top-K Plot...")
    y_prob = model.predict(X_test)
    
    # Create DF for sorting
    res_df = pd.DataFrame({
        "prob": y_prob,
        "actual": y_test,
        "effort": test_effort
    })
    
    # Random Baseline
    total_effort = res_df["effort"].sum()
    res_df = res_df.sort_values("prob", ascending=False)
    
    # Calculate cumulative effort
    res_df["cum_effort"] = res_df["effort"].cumsum()
    res_df["cum_effort_pct"] = res_df["cum_effort"] / total_effort
    res_df["pct_pr_reviewed"] = np.arange(1, len(res_df) + 1) / len(res_df)
    
    # Ideal Curve (Sort by actual effort)
    ideal_df = res_df.sort_values("effort", ascending=False).copy()
    ideal_df["cum_effort"] = ideal_df["effort"].cumsum()
    ideal_df["cum_effort_pct"] = ideal_df["cum_effort"] / total_effort
    ideal_df["pct_pr_reviewed"] = np.arange(1, len(ideal_df) + 1) / len(ideal_df)
    
    # Plot
    plt.figure(figsize=(8, 6))
    plt.plot(res_df["pct_pr_reviewed"], res_df["cum_effort_pct"], label="Model Triage", linewidth=2, color="blue")
    plt.plot(ideal_df["pct_pr_reviewed"], ideal_df["cum_effort_pct"], label="Ideal (Oracle)", linewidth=2, linestyle=":", color="green")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random")
    
    plt.title("Top-K Triage Utility: High Cost PRs")
    plt.xlabel("Fraction of PRs Reviewed (Sorted by Risk)")
    plt.ylabel("Fraction of Total Effort Covered")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    output_path = "msr26-aidev-triage/outputs/figures/topk_coverage.png"
    plt.savefig(output_path)
    print(f"Saved {output_path}")

if __name__ == "__main__":
    plot_topk_coverage()
