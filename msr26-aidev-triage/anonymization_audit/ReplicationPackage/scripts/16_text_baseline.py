
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
import numpy as np

def run_text_baseline():
    print("Loading data for Text Baseline...")
    df = pd.read_parquet("msr26-aidev-triage/artifacts/pr_features.parquet")
    
    # Filter valid
    df = df[df["status"].isin(["merged", "rejected"])].copy()
    if "is_instant_merge" in df.columns:
        df = df[df["is_instant_merge"] == 0].copy()
        
    # Prepare Text
    df["text"] = df["title"].fillna("") + " " + df["body"].fillna("")
    
    # Setup Split
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(gss.split(df, groups=df["repo_id"]))
    
    X_train = df.iloc[train_idx]["text"]
    y_train_ghost = df.iloc[train_idx]["is_ghosted"].fillna(0) # Careful with this, ghosting is only defined for rejected/feedback
    # Actually, for baseline comparison in Table 2, we usually compare on the same TARGET definition.
    # The paper says: "True Ghosting: PR Status = Rejected AND Received Human Feedback AND No follow-up..."
    # So we should only train/eval on the subset where ghosting is possible? 
    # Or is it a global prediction "Will this be ghosted?" at submission time?
    # Usually it's: Predict "Will it be a Ghosted PR?" (Binary).
    # If a PR is Merged, it is NOT Ghosted (0).
    # If a PR is Rejected + Followup, it is NOT Ghosted (0).
    # If a PR is Rejected + No Followup, it is Ghosted (1).
    # So we define target globally.
    
    df["target"] = 0
    df.loc[(df["status"]=="rejected") & (df["is_ghosted"]==1), "target"] = 1
    
    y_train = df.iloc[train_idx]["target"]
    y_test = df.iloc[test_idx]["target"]
    
    print(f"Training Text Baseline (TF-IDF + LR) on {len(X_train)} samples...")
    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=1000, stop_words="english")),
        ("lr", LogisticRegression(class_weight="balanced", max_iter=1000))
    ])
    
    pipe.fit(X_train, y_train)
    y_prob = pipe.predict_proba(df.iloc[test_idx]["text"])[:, 1]
    
    auc = roc_auc_score(y_test, y_prob)
    print(f"Text Baseline AUC (Ghosting): {auc:.4f}")
    
    # Save result for paper
    with open("msr26-aidev-triage/outputs/tables/text_baseline.txt", "w") as f:
        f.write(f"{auc:.4f}")

if __name__ == "__main__":
    run_text_baseline()
