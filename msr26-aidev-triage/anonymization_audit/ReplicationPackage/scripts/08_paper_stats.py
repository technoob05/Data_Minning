import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parents[1]))

def calculate_paper_stats():
    print("Loading data for Paper Stats...")
    from src.config import ARTIFACTS_DIR, TABLES_DIR
    base_path = ARTIFACTS_DIR / "pr_base.parquet"
    feat_path = ARTIFACTS_DIR / "pr_features.parquet"
    
    df_base = pd.read_parquet(base_path)
    df_feat = pd.read_parquet(feat_path)
    
    # 1. Instant Merges
    print("\n=== Instant Merges ===")
    if "is_instant_merge" in df_base.columns:
        n_total = len(df_base)
        n_instant = df_base["is_instant_merge"].sum()
        pct_instant = (n_instant / n_total) * 100
        print(f"Total PRs: {n_total}")
        print(f"Instant Merges (<1m): {n_instant} ({pct_instant:.2f}%)")
        
        # Breakdown by agent
        agent_instant = df_base.groupby("agent")["is_instant_merge"].agg(["count", "sum", "mean"])
        agent_instant.columns = ["total_prs", "instant_merges", "rate"]
        agent_instant["rate_pct"] = agent_instant["rate"] * 100
        print(agent_instant)
        from src.config import TABLES_DIR
        agent_instant.to_csv(TABLES_DIR / "instant_merges.csv")
    
    # 2. Ghosting Sensitivity
    print("\n=== Ghosting Sensitivity ===")
    # Reconstruct timeline from base (or check if it's in feat)
    # feat usually has 'first_human_feedback_at' if we passed it.
    # Let's check columns.
    
    # If not in feat, we might need to merge from base.
    if "first_human_feedback_at" not in df_feat.columns and "first_human_feedback_at" in df_base.columns:
        df_feat = df_feat.merge(df_base[["id", "first_human_feedback_at", "first_followup_commit_at"]], left_on="id", right_on="id", how="left")
    
    if "first_human_feedback_at" in df_feat.columns:
        thresholds = [7, 14, 30]
        results = []
        
        rejected = (df_feat["status"] == "rejected")
        has_feedback = df_feat["first_human_feedback_at"].notna()
        base_pool = rejected & has_feedback
        n_pool = base_pool.sum()
        print(f"Pool for Ghosting (Rejected + Feedback): {n_pool}")
        
        for d in thresholds:
            if "first_followup_commit_at" in df_feat.columns:
                 no_followup = df_feat["first_followup_commit_at"].isna()
                 delta = (df_feat["first_followup_commit_at"] - df_feat["first_human_feedback_at"]).dt.total_seconds() / 86400.0
                 late_followup = delta > d
                 
                 is_ghosted = (base_pool & (no_followup | late_followup))
                 n_ghosted = is_ghosted.sum()
                 rate = (n_ghosted / n_pool) * 100 if n_pool > 0 else 0
                 
                 results.append({"threshold_days": d, "ghosted_count": n_ghosted, "rate_pct": rate})
        
        res_df = pd.DataFrame(results)
        print(res_df)
        res_df.to_csv(TABLES_DIR / "ghosting_sensitivity.csv", index=False)
        
    # 3. Ghosting Cost
    print("\n=== Ghosting Cost Distribution ===")
    # Define current ghosting (14d)
    # We can perform the logic again or use 'is_ghosted' column if it matches 14d
    if "is_ghosted" in df_feat.columns:
        # Ghosted
        ghosted = df_feat[df_feat["is_ghosted"] == 1]
        
        # Non-Ghosted Rejected (The rest of rejected)
        # Status rejected but not ghosted
        rejected_non_ghost = df_feat[(df_feat["status"] == "rejected") & (df_feat["is_ghosted"] == 0)]
        
        # Merged
        merged = df_feat[df_feat["status"] == "merged"]
        
        # Calculate median effort
        if "effort_score" not in df_feat.columns:
             df_feat["effort_score"] = df_feat["num_comments"].fillna(0) + df_feat["num_reviews"].fillna(0)
             
        # Helper stats
        def get_stats(sub, name):
             e = sub["effort_score"]
             return {"type": name, "count": len(sub), "mean_effort": e.mean(), "median_effort": e.median()}
             
        stats = [
             get_stats(ghosted, "Ghosted"),
             get_stats(rejected_non_ghost, "Rejected (Non-Ghost)"),
             get_stats(merged, "Merged")
        ]
        stats_df = pd.DataFrame(stats)
        print(stats_df)
        stats_df.to_csv(TABLES_DIR / "ghosting_effort_stats.csv", index=False)

if __name__ == "__main__":
    calculate_paper_stats()
