
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import roc_auc_score, jaccard_score
from pathlib import Path
import sys

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.features import get_feature_columns

def check_robustness():
    print("Loading features...")
    df = pd.read_parquet("msr26-aidev-triage/artifacts/pr_features.parquet")
    
    # Filter valid
    df = df[df["status"].isin(["merged", "rejected"])].copy()
    if "is_instant_merge" in df.columns:
        df = df[df["is_instant_merge"] == 0].copy()
        
    print(f"N (Non-instant) = {len(df)}")
    
    feature_cols = get_feature_columns()
    
    # Define Effort Variants
    # E0: Original (Comments + Reviews)
    df["E0"] = df["num_comments"] + df["num_reviews"]
    
    # E1: Reviews Only
    df["E1"] = df["num_reviews"]
    
    # E2: Comments Only
    df["E2"] = df["num_comments"]
    
    # E3: Weighted
    df["E3"] = 2 * df["num_reviews"] + df["num_comments"]
    
    results = []
    
    # Split once for fairness
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(gss.split(df, groups=df["repo_id"]))
    
    # Identify Top 20% indices for Jaccard Overlap Analysis
    # We do this on the WHOLE set or TEST set? Let's do Test set to be predictive.
    test_df = df.iloc[test_idx].copy()
    
    # Get Top 20% sets
    sets = {}
    for e in ["E0", "E1", "E2", "E3"]:
        thresh = test_df[e].quantile(0.80)
        sets[e] = set(test_df[test_df[e] >= thresh].index)
        
    # Calculate Jaccard with E0
    print("\nOverlap Analysis (Jaccard with E0 - Original):")
    overlaps = {}
    for e in ["E1", "E2", "E3"]:
        j = len(sets["E0"].intersection(sets[e])) / len(sets["E0"].union(sets[e]))
        overlaps[e] = j
        print(f"{e} vs E0: {j:.3f}")
        
    # Train Models for each
    print("\nModel Stability Analysis (AUC):")
    X_train = df.iloc[train_idx][feature_cols]
    X_test = test_df[feature_cols]
    
    for e in ["E0", "E1", "E2", "E3"]:
        # Create binary target manually
        thresh = df.iloc[train_idx][e].quantile(0.80)
        y_train = (df.iloc[train_idx][e] >= thresh).astype(int)
        y_test = (test_df[e] >= test_df[e].quantile(0.80)).astype(int) # robust quantiling
        
        # Train
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
        
        res = {
            "Effort_Def": e,
            "AUC": auc,
            "Jaccard_w_E0": overlaps.get(e, 1.0) # E0 is 1.0
        }
        results.append(res)
        print(f"Target {e} AUC: {auc:.3f}")
        
    res_df = pd.DataFrame(results)
    res_df.to_csv("msr26-aidev-triage/outputs/tables/robustness_effort.csv", index=False)

if __name__ == "__main__":
    check_robustness()
