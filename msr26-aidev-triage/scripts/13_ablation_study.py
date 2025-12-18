
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.features import get_feature_columns

def run_ablation():
    print("Loading features...")
    df = pd.read_parquet("msr26-aidev-triage/artifacts/pr_features.parquet")
    
    # Filter valid
    df = df[df["status"].isin(["merged", "rejected"])].copy()
    if "is_instant_merge" in df.columns:
        df = df[df["is_instant_merge"] == 0].copy() # Analysis on Normal PRs
    
    all_feats = get_feature_columns()
    
    # Feature Groups
    groups = {
        "Full Model": all_feats,
        "No Agent ID": [f for f in all_feats if "agent" not in f],
        "No Text/Intent": [f for f in all_feats if f not in ["title_len", "body_len", "has_plan", "checklist", "mentions_tests", "links_issue"]],
        "No Complexity": [f for f in all_feats if "touches" not in f and "lines" not in f and "files" not in f and "additions" not in f] # Approximated based on naming
    }
    
    # Refine "No Complexity" list manually to be precise based on features.py
    # complexity: additions, deletions, changed_files, total_changes, touches_*
    complexity_feats = ["additions", "deletions", "changed_files", "total_changes", 
                        "touches_tests", "touches_docs", "touches_ci", "touches_deps", 
                        "touches_config", "num_commits"]
    groups["No Complexity"] = [f for f in all_feats if f not in complexity_feats]
    
    # Split
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(gss.split(df, groups=df["repo_id"]))
    
    # Target: High Cost
    target_col = "is_high_cost"
    y_train = df.iloc[train_idx][target_col]
    y_test = df.iloc[test_idx][target_col]
    
    results = []
    
    print("Running Ablation...")
    for name, feats in groups.items():
        # Ensure feats exist
        valid_feats = [f for f in feats if f in df.columns]
        
        X_train = df.iloc[train_idx][valid_feats]
        X_test = df.iloc[test_idx][valid_feats]
        
        train_data = lgb.Dataset(X_train, label=y_train)
        valid_data = lgb.Dataset(X_test, label=y_test, reference=train_data)
        
        params = {
            "objective": "binary", "metric": "auc", "boosting_type": "gbdt",
            "num_leaves": 31, "learning_rate": 0.05, "feature_fraction": 0.9, "verbose": -1
        }
        
        model = lgb.train(params, train_data, num_boost_round=100, valid_sets=[valid_data], 
                          callbacks=[lgb.early_stopping(stopping_rounds=10)])
        
        y_prob = model.predict(X_test)
        auc = roc_auc_score(y_test, y_prob)
        print(f"Model: {name}, AUC: {auc:.4f}")
        results.append({"Model": name, "AUC": auc})
        
    res_df = pd.DataFrame(results)
    
    # Calculate Drop
    full_auc = res_df.loc[res_df["Model"]=="Full Model", "AUC"].values[0]
    res_df["Delta AUC"] = res_df["AUC"] - full_auc
    
    print("\nAblation Results:")
    print(res_df)
    res_df.to_csv("msr26-aidev-triage/outputs/tables/ablation_results.csv", index=False)
    
    # Plot
    plt.figure(figsize=(8, 5))
    sns.barplot(data=res_df, x="Model", y="AUC")
    plt.ylim(0.5, 0.9)
    plt.title("Ablation Study: Feature Contribution")
    plt.tight_layout()
    plt.savefig("msr26-aidev-triage/outputs/figures/ablation_plot.png")
    print("Saved ablation_plot.png")

if __name__ == "__main__":
    run_ablation()
