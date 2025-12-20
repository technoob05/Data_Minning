"""
Feature engineering module for AI-Generated PR Triage.

File Pattern Taxonomy (for touches_* features):
------------------------------------------------
touches_ci:     .github/workflows/*, circleci/*, jenkins/*, gitlab-ci.*, .travis.yml
touches_deps:   package.json, package-lock.json, requirements.txt, pom.xml, 
                build.gradle, go.mod, cargo.lock
touches_config: *.conf, *.yml, *.yaml, *.toml, *.ini
touches_tests:  Computed from is_test flag (test/*, *_test.py, spec/*, etc.)
touches_docs:   Computed from is_doc flag (docs/*, README.*, *.md)

These patterns are applied to file paths in commit details to create binary flags
indicating whether a PR touches each file category.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from typing import List, Tuple, Dict
import logging

from src.config import HIGH_COST_PERCENTILE, GHOSTING_THRESHOLD_DAYS
from src.utils import setup_logger

logger = setup_logger("features")

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineers features for the triage model from raw PR data.
    
    Args:
        df: DataFrame containing raw or semi-processed PR metadata.
        
    Returns:
        DataFrame with engineered features and targets.
    """
    logger.info("Starting feature engineering...")
    
    # Copy to avoid modifying original
    feat_df = df.copy()
    
    # --- Column Handling & Aliases ---
    # Map 'comments' -> 'num_comments' if needed
    if "num_comments" not in feat_df.columns and "comments" in feat_df.columns:
        feat_df["num_comments"] = feat_df["comments"]
    if "num_reviews" not in feat_df.columns and "review_comments" in feat_df.columns:
        # Note: pr_reviews usually means review count, review_comments is distinct.
        # But if we lack 'num_reviews', we might have to assume 0 or use what we have.
        # Check if 'review_comments' acts as proxy or if we have 'pr_reviews'.
        pass 

    # Rename 'author_agent' -> 'agent' if needed
    if "agent" not in feat_df.columns and "author_agent" in feat_df.columns:
        feat_df["agent"] = feat_df["author_agent"]

    # --- 1. Temporal Features ---
    if "created_at" in feat_df.columns:
        feat_df["created_at"] = pd.to_datetime(feat_df["created_at"])
        feat_df["created_hour"] = feat_df["created_at"].dt.hour
        feat_df["created_dayofweek"] = feat_df["created_at"].dt.dayofweek
        feat_df["is_weekend"] = feat_df["created_dayofweek"].isin([5, 6]).astype(int)
    
    # --- 2. Text Features ---
    # Only process if raw text is available
    if "title" in feat_df.columns:
        feat_df["title"] = feat_df["title"].fillna("")
        feat_df["title_len"] = feat_df["title"].str.len()
    
    if "body" in feat_df.columns:
        feat_df["body"] = feat_df["body"].fillna("")
        feat_df["body_len"] = feat_df["body"].str.len()
        
        # NLP Cues
        feat_df["mentions_tests"] = feat_df["body"].str.contains("test", case=False).astype(int)
        feat_df["has_checklist"] = (feat_df["body"].str.contains(r"- \[ \]", regex=True) | 
                                    feat_df["body"].str.contains(r"- \[x\]", regex=True)).astype(int)
        feat_df["links_issue"] = feat_df["body"].str.contains(r"#\d+", regex=True).astype(int)
        feat_df["has_plan"] = feat_df["body"].str.contains(r"plan|steps|reproduce|expected|why|context", regex=True, case=False).astype(int)

    # --- 3. Interaction & Size (Pass-through if present) ---
    # additions, deletions, comments, review_comments are usually present in raw data
    if "additions" in feat_df.columns and "deletions" in feat_df.columns:
        feat_df["total_changes"] = feat_df["additions"] + feat_df["deletions"]
    
    # --- 4. Agent Encoding ---
    if "agent" in feat_df.columns:
        le = LabelEncoder()
        feat_df["agent_encoded"] = le.fit_transform(feat_df["agent"].astype(str))
    
    # --- 5. Target Construction ---
    # Filter for training (Merged/Rejected only) where status is known
    # Note: We keep all rows here; filtering should happen at training time to allow for inference
    
    # High Cost Target
    # Ensure we have components
    if "num_comments" not in feat_df.columns:
         feat_df["num_comments"] = feat_df.get("comments", 0)
    if "num_reviews" not in feat_df.columns:
         feat_df["num_reviews"] = feat_df.get("review_comments", 0) 
    
    feat_df["effort_score"] = feat_df["num_comments"] + feat_df["num_reviews"]
    threshold = feat_df["effort_score"].quantile(HIGH_COST_PERCENTILE)
    feat_df["is_high_cost"] = (feat_df["effort_score"] >= threshold).astype(int)
    
    # Ghosting Target
    # Logic: Rejected AND Feedback AND No Follow-up > 14 days
    # If is_ghost already exists (from pre-calc), use it
    if "is_ghost" in feat_df.columns and "is_ghosted" not in feat_df.columns:
        feat_df["is_ghosted"] = feat_df["is_ghost"]
    
    if "first_human_feedback_at" in feat_df.columns:
        feat_df["first_human_feedback_at"] = pd.to_datetime(feat_df["first_human_feedback_at"])
        
        is_rejected = (feat_df["state"] == "closed") & (feat_df["merged_at"].isna())
        has_feedback = feat_df["first_human_feedback_at"].notna()
        
        if "first_followup_commit_at" in feat_df.columns:
             feat_df["first_followup_commit_at"] = pd.to_datetime(feat_df["first_followup_commit_at"])
             
             no_followup = feat_df["first_followup_commit_at"].isna()
             delta = (feat_df["first_followup_commit_at"] - feat_df["first_human_feedback_at"]).dt.total_seconds() / 86400.0
             late_followup = delta > GHOSTING_THRESHOLD_DAYS
             
             feat_df["is_ghosted"] = (is_rejected & has_feedback & (no_followup | late_followup)).astype(int)
    
    # Instant Merge Sanity Check
    if "merged_at" in feat_df.columns and "created_at" in feat_df.columns:
        feat_df["merged_at"] = pd.to_datetime(feat_df["merged_at"])
        # Use merged_at specific logic
        duration = (feat_df["merged_at"] - feat_df["created_at"]).dt.total_seconds() / 3600.0
        feat_df["is_instant_merge"] = (duration < 1/60).astype(int) # < 1 min

    logger.info(f"Features engineered. Shape: {feat_df.shape}")
    return feat_df

def get_feature_columns(strict: bool = False) -> List[str]:
    """
    Returns the list of feature columns.
    
    Args:
        strict: If True, returns T0 features (Metadata/Text + Static Complexity).
                If False, returns T1 features (T0 + Dynamic Interaction signals).
                
    Note: Per reviewer feedback, static complexity features (additions, deletions)
          ARE available at PR creation and should be included in T0.
    """
    # T0 Features (Available at creation: Metadata + Text + Static Complexity)
    t0_features = [
        # Metadata
        "agent_encoded",
        "created_hour", "created_dayofweek", "is_weekend",
        "lang_extension",
        # Text
        "title_len", "body_len",
        "mentions_tests", "has_checklist", "links_issue", "has_plan",
        # Static Complexity (Available at creation)
        "additions", "deletions", "changed_files", "total_changes",
        "touches_tests", "touches_docs", "touches_ci", "touches_deps", "touches_config"
    ]
    
    # T1 Features (Dynamic/Interaction signals accumulated before human review)
    t1_additional_features = [
        "num_commits",  # Number of agent self-correction commits before human review
    ]
    
    if strict:
        return t0_features
    else:
        return t0_features + t1_additional_features


