
import pandas as pd
import numpy as np

def audit_ghosting():
    print("Loading pr_base...")
    try:
        df = pd.read_parquet("msr26-aidev-triage/artifacts/pr_base.parquet")
    except Exception as e:
        print(f"Error loading pr_base: {e}")
        return

    # Define Pool: Rejected + Has Feedback
    # Note: 'first_human_feedback_at' might be in pr_base or we need to check features.
    # Based on 08_paper_stats.py, it expects it in pr_base or merges it.
    
    cols = df.columns.tolist()
    print(f"Columns in pr_base: {cols}")
    
    if "first_human_feedback_at" not in cols:
        print("MISSING: first_human_feedback_at in pr_base")
        return
        
    rejected = (df["status"] == "rejected")
    has_feedback = df["first_human_feedback_at"].notna()
    pool = df[rejected & has_feedback]
    
    n_pool = len(pool)
    print(f"Ghosting Analysis Pool (Rejected + Feedback): {n_pool}")
    
    if "first_followup_commit_at" in cols:
        n_followup = pool["first_followup_commit_at"].notna().sum()
        pct_followup = (n_followup / n_pool) * 100 if n_pool > 0 else 0
        missing = n_pool - n_followup
        print(f"Has Follow-up Timestamp: {n_followup} ({pct_followup:.2f}%)")
        print(f"Missing Follow-up Timestamp: {missing} ({100-pct_followup:.2f}%)")
        
        if n_pool > 0 and missing == n_pool:
            print("CRITICAL: 100% Missing Follow-up Timestamps! Ghosting rate will be artificially 100%.")
    else:
        print("MISSING: first_followup_commit_at in pr_base")

if __name__ == "__main__":
    audit_ghosting()
