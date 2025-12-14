import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score, brier_score_loss
import shap

def train_lgbm_model(df: pd.DataFrame, feature_cols: list, target_col: str = "is_merged", filter_instant: bool = True):
    """
    Trains a LightGBM model to predict PR acceptance.
    """
    print(f"Training LightGBM model for target: {target_col}...")
    
    # Filter instant merges if requested (to avoid learning from bots)
    if filter_instant and "is_instant_merge" in df.columns:
        n_instant = df["is_instant_merge"].sum()
        print(f"Filtering out {n_instant} instant merges (< 1 min)...")
        df = df[df["is_instant_merge"] == 0].copy()
    
    X = df[feature_cols]
    y = df[target_col]
    
    # Repo-disjoint split
    if "repo_id" in df.columns:
        print("Using Repo-Disjoint Split (Train/Test on different repos)...")
        gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        train_idx, test_idx = next(gss.split(X, y, groups=df["repo_id"]))
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    else:
        print("Warning: repo_id not found. Using random split (potential leakage).")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
    
    # Create dataset
    train_data = lgb.Dataset(X_train, label=y_train)
    test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)
    
    # Parameters
    params = {
        "objective": "binary",
        "metric": "auc",
        "boosting_type": "gbdt",
        "num_leaves": 31,
        "learning_rate": 0.05,
        "feature_fraction": 0.9,
        "verbose": -1
    }
    
    # Train
    model = lgb.train(
        params,
        train_data,
        num_boost_round=1000,
        valid_sets=[test_data],
        callbacks=[lgb.early_stopping(stopping_rounds=50), lgb.log_evaluation(100)]
    )
    
    # Evaluate
    y_pred_prob = model.predict(X_test, num_iteration=model.best_iteration)
    y_pred = (y_pred_prob > 0.5).astype(int)
    
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "auc": roc_auc_score(y_test, y_pred_prob),
        "pr_auc": average_precision_score(y_test, y_pred_prob),
        "brier_score": brier_score_loss(y_test, y_pred_prob)
    }
    
    print("Model Metrics:", metrics)
    
    return model, metrics, X_test

def explain_model(model, X_test):
    """
    Generates SHAP values for model explanation.
    """
    print("Generating SHAP values...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    
    # For binary classification, shap_values is a list of arrays. 
    # Usually index 1 corresponds to the positive class (merged).
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
        
    return explainer, shap_values
