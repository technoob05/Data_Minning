"""
M3: Ghosting Label Missingness Report
"""
import pandas as pd

def missingness_report():
    print("=" * 60)
    print("M3: GHOSTING LABEL MISSINGNESS REPORT")
    print("=" * 60)
    
    df = pd.read_parquet("msr26-aidev-triage/artifacts/pr_features.parquet")
    
    print(f"\nTotal PRs: {len(df)}")
    
    # Key timestamps for ghosting
    cols = ["first_human_feedback_at", "first_followup_commit_at", "closed_at"]
    
    print("\n--- Timestamp Missingness ---")
    for col in cols:
        if col in df.columns:
            missing = df[col].isna().sum()
            present = df[col].notna().sum()
            pct = missing / len(df) * 100
            print(f"  {col}: {missing} missing ({pct:.1f}%)")
    
    # Ghosting-specific pool
    print("\n--- Ghosting Pool Analysis ---")
    rejected = df[df["status"] == "rejected"]
    print(f"  Rejected PRs: {len(rejected)}")
    
    rejected_with_feedback = rejected[rejected["first_human_feedback_at"].notna()]
    print(f"  Rejected + Has Feedback Timestamp: {len(rejected_with_feedback)}")
    
    # Of those, how many have followup timestamp?
    has_followup_ts = rejected_with_feedback["first_followup_commit_at"].notna().sum()
    missing_followup_ts = rejected_with_feedback["first_followup_commit_at"].isna().sum()
    print(f"    - Has followup timestamp: {has_followup_ts}")
    print(f"    - Missing followup timestamp: {missing_followup_ts} ({missing_followup_ts/len(rejected_with_feedback)*100:.1f}%)")
    
    # What does missing followup mean?
    print("""
    Interpretation:
    - Missing followup_commit_at = NO FOLLOW-UP COMMIT FOUND
    - This is treated as "ghosted" (no response after feedback)
    - This is semantically correct: absence = abandonment
    """)
    
    # Sensitivity to threshold
    print("\n--- Ghosting Rate by Threshold (Sensitivity) ---")
    pool = rejected_with_feedback.copy()
    for days in [7, 14, 30]:
        no_followup = pool["first_followup_commit_at"].isna()
        delta = (pool["first_followup_commit_at"] - pool["first_human_feedback_at"]).dt.total_seconds() / 86400.0
        late_followup = delta > days
        ghosted = no_followup | late_followup
        rate = ghosted.sum() / len(pool) * 100
        print(f"  {days}-day threshold: {rate:.1f}% ghosted ({int(ghosted.sum())}/{len(pool)})")
    
    # Save
    summary = {
        "Pool": ["All PRs", "Rejected", "Rejected + Feedback", "Missing Followup TS"],
        "Count": [len(df), len(rejected), len(rejected_with_feedback), missing_followup_ts],
        "Pct": [100, len(rejected)/len(df)*100, len(rejected_with_feedback)/len(df)*100, 
                missing_followup_ts/len(rejected_with_feedback)*100 if len(rejected_with_feedback) > 0 else 0]
    }
    pd.DataFrame(summary).to_csv("msr26-aidev-triage/outputs/tables/missingness_stats.csv", index=False)
    print("\nSaved missingness_stats.csv")

if __name__ == "__main__":
    missingness_report()
