import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineers features for the triage model.
    """
    print("Engineering features...")
    
    # Copy to avoid modifying original
    feat_df = df.copy()
    
    # 1. Temporal Features
    feat_df["created_hour"] = feat_df["created_at"].dt.hour
    feat_df["created_dayofweek"] = feat_df["created_at"].dt.dayofweek
    feat_df["is_weekend"] = feat_df["created_dayofweek"].isin([5, 6]).astype(int)
    
    # 2. Text Features (Title/Body)
    feat_df["title"] = feat_df["title"].fillna("")
    feat_df["body"] = feat_df["body"].fillna("")
    
    feat_df["title_len"] = feat_df["title"].str.len()
    feat_df["body_len"] = feat_df["body"].str.len()
    
    # NLP Cues
    feat_df["mentions_tests"] = feat_df["body"].str.contains("test", case=False).astype(int)
    feat_df["has_checklist"] = feat_df["body"].str.contains("- \[ \]", regex=False).astype(int) | \
                               feat_df["body"].str.contains("- \[x\]", regex=False).astype(int)
    feat_df["links_issue"] = feat_df["body"].str.contains("#\d+", regex=True).astype(int)
    feat_df["has_plan"] = feat_df["body"].str.contains(r"plan|steps|reproduce|expected|why|context", regex=True, case=False).astype(int)
    
    # 3. Interaction Features
    # (Already have num_commits, num_reviews, num_comments, num_review_comments)
    
    # 4. Size Features
    # (Already have additions, deletions, changed_files)
    feat_df["total_changes"] = feat_df["additions"] + feat_df["deletions"]
    
    # 5. Agent Encoding (One-Hot or Label Encoding)
    # For tree-based models, Label Encoding is often sufficient or categorical support.
    # Let's use Label Encoding for simplicity in this baseline.
    le = LabelEncoder()
    feat_df["agent_encoded"] = le.fit_transform(feat_df["agent"].astype(str))
    
    # 6. Task Type Encoding
    # Handle missing task types
    feat_df["task_type"] = feat_df["task_type"].fillna("unknown")
    le_task = LabelEncoder()
    feat_df["task_type_encoded"] = le_task.fit_transform(feat_df["task_type"].astype(str))
    
    # 7. Target Construction
    # Filter out 'open' PRs for training.
    feat_df = feat_df[feat_df["status"].isin(["merged", "rejected"])].copy()
    
    # Target 1: High Cost (Top 20% of effort)
    # Effort = num_comments + num_reviews
    feat_df["effort_score"] = feat_df["num_comments"] + feat_df["num_reviews"]
    effort_threshold = feat_df["effort_score"].quantile(0.80)
    feat_df["is_high_cost"] = (feat_df["effort_score"] >= effort_threshold).astype(int)
    
    # Target 2: Rejected with Effort (Old Ghosting)
    # Definition: Rejected AND (num_comments > 0 OR num_reviews > 0)
    feat_df["rejected_with_effort"] = ((feat_df["status"] == "rejected") & (feat_df["effort_score"] > 0)).astype(int)

    # Target 3: True Ghosting
    # Definition: Rejected AND (Has Human Feedback) AND (No Follow-up OR Follow-up > 14 days later)
    if "first_human_feedback_at" in feat_df.columns:
        rejected = (feat_df["status"] == "rejected")
        has_feedback = feat_df["first_human_feedback_at"].notna()
        
        # Check follow-up
        if "first_followup_commit_at" in feat_df.columns:
             no_followup = feat_df["first_followup_commit_at"].isna()
             
             # Time delta
             delta = (feat_df["first_followup_commit_at"] - feat_df["first_human_feedback_at"]).dt.total_seconds() / 86400.0
             late_followup = delta > 14
             
             feat_df["is_ghosted"] = (rejected & has_feedback & (no_followup | late_followup)).astype(int)
        else:
             # Fallback if no follow-up info
             feat_df["is_ghosted"] = 0
             print("Warning: first_followup_commit_at not found. is_ghosted set to 0.")
    else:
        feat_df["is_ghosted"] = 0
        print("Warning: first_human_feedback_at not found. is_ghosted set to 0.")
    
    # Legacy Target: is_merged
    feat_df["is_merged"] = (feat_df["status"] == "merged").astype(int)
    
    # 8. Sanity Filter: Remove instant merges (< 1 min) from training
    # Calculate duration if not present
    if "duration_hours" not in feat_df.columns:
        feat_df["end_date"] = feat_df["merged_at"].combine_first(feat_df["closed_at"])
        feat_df["duration_hours"] = (feat_df["end_date"] - feat_df["created_at"]).dt.total_seconds() / 3600.0
        
    # Flag instant merges
    feat_df["is_instant_merge"] = (feat_df["duration_hours"] < 1/60).astype(int)
    
    print(f"Features engineered. Training set size: {len(feat_df)}")
    
    return feat_df

def get_feature_columns():
    """
    Returns the list of feature columns to be used for training.
    """
    return [
        "agent_encoded",
        "task_type_encoded",
        "task_confidence",
        "num_commits",
        "additions",
        "deletions",
        "changed_files",
        "total_changes",
        "title_len",
        "body_len",
        "created_hour",
        "created_dayofweek",
        "is_weekend",
        "mentions_tests",
        "has_checklist",
        "links_issue",
        "has_plan",
        # New Diff Features
        "touches_tests",
        "touches_docs",
        "touches_ci",
        "touches_deps",
        "touches_config"
    ]
