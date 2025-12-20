import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, brier_score_loss
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

def run_baselines_and_analysis():
    print("Loading data...")
    base_path = "msr26-aidev-triage/artifacts/pr_base.parquet"
    feat_path = "msr26-aidev-triage/artifacts/pr_features.parquet"
    
    df_base = pd.read_parquet(base_path, engine="fastparquet")
    print("Base columns:", df_base.columns.tolist())
    df_feat = pd.read_parquet(feat_path, engine="fastparquet")
    print("Feat columns:", df_feat.columns.tolist())
    
    # Merge base metadata into features
    # df_feat usually has 'id'. Check if both have 'id'
    if "id" in df_base.columns and "id" in df_feat.columns:
        df_model = pd.merge(df_feat, df_base, on="id", suffixes=("", "_base"))
    else:
        print("Warning: 'id' column missing, cannot merge. Using feat only.")
        df_model = df_feat.copy()

    # Filter instant merges for modeling
    if "is_instant_merge" in df_model.columns:
        df_model = df_model[df_model["is_instant_merge"] == 0].copy()
    
    # Definition of High Cost (Top 20% effort) is already in is_high_cost feature if run previously
    # If not, let's assume it is.
    
    targets = ["is_high_cost", "is_ghosted"]
    feature_cols = [
        "num_files", "num_lines_added", "num_lines_removed", 
        "lines_per_file", "has_tests", "has_docs", "links_issue",
        "has_plan", "touches_tests", "touches_docs", "touches_ci", 
        "touches_deps", "touches_config"
    ]
    
    # Ensure columns exist
    available_feats = [f for f in feature_cols if f in df_model.columns]
    
    # Split
    if "repo_id" not in df_model.columns:
        print("Warning: repo_id not found, using random split")
        df_model["repo_id"] = np.random.randint(0, 100, len(df_model))
        
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(gss.split(df_model, df_model["is_high_cost"], groups=df_model["repo_id"]))
    
    train_df = df_model.iloc[train_idx]
    test_df = df_model.iloc[test_idx]
    
    results = []
    
    for target in targets:
        if target not in df_model.columns:
            continue
            
        print(f"\nrunning baselines for {target}...")
        y_train = train_df[target]
        y_test = test_df[target]
        
        # 1. Logistic Regression Baseline
        pipe = Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler()),
            ('logreg', LogisticRegression(class_weight='balanced', max_iter=1000))
        ])
        
        pipe.fit(train_df[available_feats], y_train)
        y_prob_lr = pipe.predict_proba(test_df[available_feats])[:, 1]
        
        auc_lr = roc_auc_score(y_test, y_prob_lr)
        precision, recall, _ = precision_recall_curve(y_test, y_prob_lr)
        prauc_lr = auc(recall, precision)
        
        results.append({
            "target": target,
            "model": "Logistic Regression",
            "auc": auc_lr,
            "pr_auc": prauc_lr
        })
        
        # 2. Rule-based Baseline
        # Rule: If touches_ci OR touches_deps -> Predict Positive
        # This is a heuristic for "High Risk"
        if "touches_ci" in available_feats and "touches_deps" in available_feats:
            y_rule = (test_df["touches_ci"] == 1) | (test_df["touches_deps"] == 1)
            y_rule = y_rule.astype(int)
            
            # AUC for binary prediction is just (Sensitivity + Specificity)/2 basically
            auc_rule = roc_auc_score(y_test, y_rule)
            precision, recall, _ = precision_recall_curve(y_test, y_rule)
            prauc_rule = auc(recall, precision)
            
            results.append({
                "target": target,
                "model": "Rule: CI or Deps",
                "auc": auc_rule,
                "pr_auc": prauc_rule
            })
            
    # Save Baselines
    res_df = pd.DataFrame(results)
    print("Baseline Results:")
    print(res_df)
    res_df.to_csv("msr26-aidev-triage/outputs/tables/baselines.csv", index=False)
    
    # 3. Error Analysis for Ghosting
    print("\nError Analysis for Ghosting...")
    # We need predictions from the BEST model (LightGBM). TopK script likely saved it?
    # Or we can just retrain efficiently here or assume we have predictions.
    # For simplicity, let's use the LR predictions we just made as a proxy OR better, 
    # Use the LightGBM predictions if we can easily run it. 
    # Actually, let's use the LR baseline for error analysis vs "High Quality" if we want, 
    # BUT user asked for "Ghosting model" (LightGBM) failures.
    # To avoid re-implementing full LGBM here, let's look for saved predictions or re-train quickly.
    # Re-training LGBM quickly:
    
    import lightgbm as lgb
    lgb_train = lgb.Dataset(train_df[available_feats], label=train_df["is_ghosted"])
    lgb_test = lgb.Dataset(test_df[available_feats], label=test_df["is_ghosted"], reference=lgb_train)
    params = {'objective': 'binary', 'metric': 'auc', 'is_unbalance': True, 'verbose': -1}
    bst = lgb.train(params, lgb_train, num_boost_round=100)
    y_prob_lgbm = bst.predict(test_df[available_feats])
    
    # Identify FP and FN
    # Threshold = 0.5 for simplicity (or calibrated)
    test_df = test_df.copy()
    test_df["prob_ghost"] = y_prob_lgbm
    test_df["pred_ghost"] = (test_df["prob_ghost"] > 0.5).astype(int)
    
    fp = test_df[(test_df["pred_ghost"] == 1) & (test_df["is_ghosted"] == 0)]
    fn = test_df[(test_df["pred_ghost"] == 0) & (test_df["is_ghosted"] == 1)]
    
    print(f"Found {len(fp)} FPs and {len(fn)} FNs")
    
    # Save sample
    desired_cols = ["id", "prob_ghost", "touches_ci", "touches_deps", "has_plan", "num_files"]
    out_cols = [c for c in desired_cols if c in test_df.columns]
    
    error_df = pd.concat([
        fp.head(20).assign(error_type="False Positive"),
        fn.head(20).assign(error_type="False Negative")
    ])
    
    # Ensure we select columns that exist
    final_cols = out_cols + ["error_type"]
    error_df = error_df[final_cols]
    
    error_df.to_csv("msr26-aidev-triage/outputs/tables/error_analysis.csv", index=False)
    
    try:
        # Instant Merge Chart
        print("\nGenerating Instant Merges Chart...")
        if "is_instant_merge" in df_base.columns:
            agent_counts = df_base.groupby("agent")["is_instant_merge"].sum().sort_values(ascending=False)
            
            plt.figure(figsize=(10, 6))
            bars = plt.bar(agent_counts.index, agent_counts.values, color='skyblue', edgecolor='black')
            plt.title("Instant Merges (< 1 min) by Agent")
            plt.xlabel("Agent")
            plt.ylabel("Count of Instant Merges")
            plt.grid(axis='y', alpha=0.3)
            
            # Add labels
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height,
                         f'{int(height)}',
                         ha='center', va='bottom')
                         
            plt.tight_layout()
            plt.savefig("msr26-aidev-triage/outputs/figures/instant_merges.png")
            print("Saved instant_merges.png")
        else:
            print("Skipping Instant Merges Chart: Column missing")
    except Exception as e:
        print(f"Error in Instant Merges Chart: {e}")

if __name__ == "__main__":
    run_baselines_and_analysis()
