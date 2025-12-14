import pandas as pd
import numpy as np
from .load import preprocess_dates

def build_pr_base_table(dfs: dict) -> pd.DataFrame:
    """
    Constructs the base PR table by merging and aggregating data from various sources.
    """
    print("Building PR base table...")
    
    pr_df = dfs["pr"].copy()
    pr_df = preprocess_dates(pr_df)
    
    # 1. Merge Task Type
    if dfs["pr_task_type"] is not None:
        task_type_df = dfs["pr_task_type"][["id", "type", "confidence"]].copy()
        task_type_df = task_type_df.rename(columns={"type": "task_type", "confidence": "task_confidence"})
        # Ensure ID types match
        pr_df["id"] = pr_df["id"].astype(str)
        task_type_df["id"] = task_type_df["id"].astype(str)
        
        pr_df = pr_df.merge(task_type_df, on="id", how="left")
    
    # 2. Aggregate Commit Details
    if dfs["pr_commit_details"] is not None:
        commits_df = dfs["pr_commit_details"].copy()
        # Assuming 'pr_id' links to PRs
        commits_df["pr_id"] = commits_df["pr_id"].astype(str)
        
        # Check available columns
        agg_dict = {
            "additions": "sum",
            "deletions": "sum",
            "sha": "nunique" # Number of unique commits
        }
        
        # If changed_files column exists, sum it. Otherwise, count rows as proxy for changed files (if file-level)
        if "changed_files" in commits_df.columns:
            agg_dict["changed_files"] = "sum"
        else:
            # Assuming table is file-level, count rows
            agg_dict["filename"] = "count"
            
        # New: Identify file types
        if "filename" in commits_df.columns:
            commits_df["is_test"] = commits_df["filename"].str.contains("test", case=False, na=False).astype(int)
            commits_df["is_doc"] = commits_df["filename"].str.contains("doc|readme", case=False, na=False).astype(int)
            agg_dict["is_test"] = "max"
            agg_dict["is_doc"] = "max"
        
        # Aggregations
        commit_stats = commits_df.groupby("pr_id").agg(agg_dict)
        
        # Rename columns
        rename_dict = {"sha": "num_commits", "is_test": "touches_tests", "is_doc": "touches_docs"}
        if "filename" in agg_dict:
            rename_dict["filename"] = "changed_files"
            
        commit_stats = commit_stats.rename(columns=rename_dict)
        
        pr_df = pr_df.merge(commit_stats, left_on="id", right_on="pr_id", how="left")
        
        # Fill NaNs for PRs with no commits (or no details)
        for col in ["additions", "deletions", "changed_files", "num_commits", "touches_tests", "touches_docs"]:
            if col in pr_df.columns:
                pr_df[col] = pr_df[col].fillna(0)

    # 3. Aggregate Reviews
    if dfs["pr_reviews"] is not None:
        reviews_df = dfs["pr_reviews"].copy()
        reviews_df["pr_id"] = reviews_df["pr_id"].astype(str)
        
        review_stats = reviews_df.groupby("pr_id").size().reset_index(name="num_reviews")
        pr_df = pr_df.merge(review_stats, left_on="id", right_on="pr_id", how="left")
        pr_df["num_reviews"] = pr_df["num_reviews"].fillna(0)
        pr_df = pr_df.drop(columns=["pr_id"], errors="ignore")

    # 4. Aggregate Comments (General PR comments)
    if dfs["pr_comments"] is not None:
        comments_df = dfs["pr_comments"].copy()
        
        # Use pr_id if available
        if "pr_id" in comments_df.columns:
             comments_df["pr_id"] = comments_df["pr_id"].astype(str)
             comment_stats = comments_df.groupby("pr_id").size().reset_index(name="num_comments")
             pr_df = pr_df.merge(comment_stats, left_on="id", right_on="pr_id", how="left")
             pr_df["num_comments"] = pr_df["num_comments"].fillna(0)
             pr_df = pr_df.drop(columns=["pr_id"], errors="ignore")
        elif "pull_request_id" in comments_df.columns:
             comments_df["pull_request_id"] = comments_df["pull_request_id"].astype(str)
             comment_stats = comments_df.groupby("pull_request_id").size().reset_index(name="num_comments")
             pr_df = pr_df.merge(comment_stats, left_on="id", right_on="pull_request_id", how="left")
             pr_df["num_comments"] = pr_df["num_comments"].fillna(0)
             pr_df = pr_df.drop(columns=["pull_request_id"], errors="ignore")

    # 5. Aggregate Review Comments (Inline code comments)
    if dfs["pr_review_comments"] is not None:
        rv_comments_df = dfs["pr_review_comments"].copy()
        
        # Try to get pr_id from reviews if not present
        if "pr_id" not in rv_comments_df.columns and "pull_request_id" not in rv_comments_df.columns:
             if "pull_request_review_id" in rv_comments_df.columns and dfs["pr_reviews"] is not None:
                 reviews_lookup = dfs["pr_reviews"][["id", "pr_id"]].rename(columns={"id": "pull_request_review_id"})
                 rv_comments_df = rv_comments_df.merge(reviews_lookup, on="pull_request_review_id", how="left")

        if "pr_id" in rv_comments_df.columns:
            rv_comments_df["pr_id"] = rv_comments_df["pr_id"].astype(str)
            rv_comment_stats = rv_comments_df.groupby("pr_id").size().reset_index(name="num_review_comments")
            pr_df = pr_df.merge(rv_comment_stats, left_on="id", right_on="pr_id", how="left")
            pr_df["num_review_comments"] = pr_df["num_review_comments"].fillna(0)
            pr_df = pr_df.drop(columns=["pr_id"], errors="ignore")
        elif "pull_request_id" in rv_comments_df.columns:
            rv_comments_df["pull_request_id"] = rv_comments_df["pull_request_id"].astype(str)
            rv_comment_stats = rv_comments_df.groupby("pull_request_id").size().reset_index(name="num_review_comments")
            pr_df = pr_df.merge(rv_comment_stats, left_on="id", right_on="pull_request_id", how="left")
            pr_df["num_review_comments"] = pr_df["num_review_comments"].fillna(0)
            pr_df = pr_df.drop(columns=["pull_request_id"], errors="ignore")

    # 6. Calculate Turnaround Time (in hours)
    # merged_at if merged, closed_at if closed/rejected.
    pr_df["end_date"] = pr_df["merged_at"].combine_first(pr_df["closed_at"])
    # If still open, end_date is NaT
    
    pr_df["turnaround_time_hours"] = (pr_df["end_date"] - pr_df["created_at"]).dt.total_seconds() / 3600.0
    
    # 7. Determine Status (Merged, Rejected, Open)
    conditions = [
        pr_df["merged_at"].notna(),
        (pr_df["merged_at"].isna()) & (pr_df["state"] == "closed"),
        pr_df["state"] == "open"
    ]
    choices = ["merged", "rejected", "open"]
    pr_df["status"] = np.select(conditions, choices, default="unknown")

    print(f"Base table built with {len(pr_df)} rows.")
    return pr_df
