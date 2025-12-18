"""
Comprehensive Label Audit for Best Paper Qualification
This script audits ghosting rates across different pools to reconcile 100% vs 64.1% claims.
"""
import pandas as pd
import numpy as np

def comprehensive_audit():
    print("=" * 60)
    print("COMPREHENSIVE LABEL AUDIT")
    print("=" * 60)
    
    # Load data
    df = pd.read_parquet("msr26-aidev-triage/artifacts/pr_features.parquet")
    print(f"\nTotal PRs loaded: {len(df)}")
    print(f"Columns: {len(df.columns)}")
    
    # --- AUDIT 1: Pool Sizes ---
    print("\n" + "=" * 60)
    print("AUDIT 1: Pool Sizes by Status & Regime")
    print("=" * 60)
    
    # Status breakdown
    status_counts = df["status"].value_counts()
    print("\nBy Status:")
    for s, c in status_counts.items():
        print(f"  {s}: {c} ({c/len(df)*100:.1f}%)")
    
    # Regime breakdown
    if "is_instant_merge" in df.columns:
        instant_counts = df["is_instant_merge"].value_counts()
        n_instant = instant_counts.get(1, 0)
        n_normal = instant_counts.get(0, 0)
        print(f"\nBy Regime:")
        print(f"  Instant Merges (<1m): {n_instant} ({n_instant/len(df)*100:.1f}%)")
        print(f"  Normal PRs: {n_normal} ({n_normal/len(df)*100:.1f}%)")
    
    # --- AUDIT 2: Ghosting Definition Components ---
    print("\n" + "=" * 60)
    print("AUDIT 2: Ghosting Definition Components")
    print("=" * 60)
    
    # Check key columns
    cols_to_check = ["first_human_feedback_at", "first_followup_commit_at", "is_ghosted"]
    for col in cols_to_check:
        if col in df.columns:
            n_missing = df[col].isna().sum()
            n_present = df[col].notna().sum()
            print(f"\n{col}:")
            print(f"  Present: {n_present} ({n_present/len(df)*100:.1f}%)")
            print(f"  Missing: {n_missing} ({n_missing/len(df)*100:.1f}%)")
    
    # --- AUDIT 3: Ghosting Rate by Pool ---
    print("\n" + "=" * 60)
    print("AUDIT 3: Ghosting Rates by Pool")
    print("=" * 60)
    
    def calc_ghosting_rate(pool_df, pool_name):
        n = len(pool_df)
        if n == 0:
            print(f"\n{pool_name}: N=0 (empty pool)")
            return
        
        # Count ghosted
        if "is_ghosted" in pool_df.columns:
            n_ghosted = pool_df["is_ghosted"].fillna(0).sum()
            rate = n_ghosted / n * 100
            print(f"\n{pool_name}:")
            print(f"  N = {n}")
            print(f"  Ghosted = {int(n_ghosted)}")
            print(f"  Rate = {rate:.1f}%")
            
            # Check for ALL ghosted (100% case)
            if rate > 99:
                print(f"  ⚠️ WARNING: Near 100% rate - check for data issues!")
                
            # Breakdown by missing followup
            if "first_followup_commit_at" in pool_df.columns:
                n_no_followup = pool_df["first_followup_commit_at"].isna().sum()
                print(f"  Missing followup_commit_at: {n_no_followup} ({n_no_followup/n*100:.1f}%)")
    
    # Pool 1: ALL PRs
    calc_ghosting_rate(df, "ALL PRs")
    
    # Pool 2: Non-instant PRs only
    if "is_instant_merge" in df.columns:
        non_instant = df[df["is_instant_merge"] == 0]
        calc_ghosting_rate(non_instant, "Non-Instant PRs Only")
    
    # Pool 3: Rejected PRs
    rejected = df[df["status"] == "rejected"]
    calc_ghosting_rate(rejected, "Rejected PRs (All)")
    
    # Pool 4: Rejected + Has Human Feedback
    rejected_feedback = df[(df["status"] == "rejected") & (df["first_human_feedback_at"].notna())]
    calc_ghosting_rate(rejected_feedback, "Rejected + Human Feedback (The Correct Pool)")
    
    # Pool 5: Rejected + Non-Instant + Feedback
    if "is_instant_merge" in df.columns:
        rejected_normal_feedback = df[
            (df["status"] == "rejected") & 
            (df["is_instant_merge"] == 0) & 
            (df["first_human_feedback_at"].notna())
        ]
        calc_ghosting_rate(rejected_normal_feedback, "Rejected + Non-Instant + Feedback")
    
    # --- AUDIT 4: Where does "100%" come from? ---
    print("\n" + "=" * 60)
    print("AUDIT 4: Investigating '100%' Claim Source")
    print("=" * 60)
    
    # Check if 100% is from instant merges (non-sensical for ghosting)
    if "is_instant_merge" in df.columns:
        instant = df[df["is_instant_merge"] == 1]
        instant_rejected = instant[instant["status"] == "rejected"]
        if len(instant_rejected) > 0:
            instant_ghost_rate = instant_rejected["is_ghosted"].fillna(0).mean() * 100
            print(f"\nInstant Merges + Rejected:")
            print(f"  N = {len(instant_rejected)}")
            print(f"  Ghosting Rate = {instant_ghost_rate:.1f}%")
    
    # --- AUDIT 5: Sensitivity to Threshold ---
    print("\n" + "=" * 60)
    print("AUDIT 5: Ghosting Sensitivity to Threshold (7/14/30 days)")
    print("=" * 60)
    
    pool = rejected_feedback.copy()
    if "first_followup_commit_at" in pool.columns and "first_human_feedback_at" in pool.columns:
        for days in [7, 14, 30]:
            no_followup = pool["first_followup_commit_at"].isna()
            delta = (pool["first_followup_commit_at"] - pool["first_human_feedback_at"]).dt.total_seconds() / 86400.0
            late_followup = delta > days
            ghosted = no_followup | late_followup
            rate = ghosted.sum() / len(pool) * 100
            print(f"  Threshold {days} days: {rate:.1f}% ghosted")
    
    # --- FINAL RECOMMENDATION ---
    print("\n" + "=" * 60)
    print("RECOMMENDATION FOR PAPER")
    print("=" * 60)
    print("""
1. USE the 'Rejected + Human Feedback' pool for ghosting claims.
2. REMOVE or CLARIFY 'near 100%' - it's likely from a bug or wrong pool.
3. ADD missingness stats to Threats: X% of rejected PRs lack feedback timestamp.
4. DEFINE ghosting clearly: 'No follow-up commit within 14 days of human feedback.'
""")

if __name__ == "__main__":
    comprehensive_audit()
