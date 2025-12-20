
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from lifelines import KaplanMeierFitter

# Paths
DATA_DIR = Path("data")
OUTPUT_DIR = Path("outputs/audit")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def main():
    print("Loading data...")
    prs = pd.read_parquet(DATA_DIR / "pull_request.parquet")
    timeline = pd.read_parquet(DATA_DIR / "pr_timeline.parquet")
    commits = pd.read_parquet(DATA_DIR / "pr_commits.parquet")
    
    # 1. Feedback Timestamps
    feedback_mask = timeline['event'].isin([
        'commented', 'reviewed', 'review_requested', 
        'IssueComment', 'PullRequestReviewComment', 'PullRequestReview'
    ])
    feedback_events = timeline[feedback_mask].copy()
    
    if 'created_at' in feedback_events.columns:
        feedback_events['created_at'] = pd.to_datetime(feedback_events['created_at'], errors='coerce', utc=True)
    
    first_feedback = feedback_events.groupby('pr_id')['created_at'].min().reset_index()
    first_feedback.columns = ['pr_id', 'first_feedback_at']
    
    # 2. Check Commit Timestamps (Audit)
    # Check if 'stats' or 'created_at' exists in commits
    has_commit_ts = False
    if 'created_at' in commits.columns:
        valid_ts = commits['created_at'].notna().sum()
        total_commits = len(commits)
        print(f"Commit Timestamp Audit: {valid_ts}/{total_commits} valid timestamps ({(100*valid_ts/total_commits):.1f}%)")
        has_commit_ts = (valid_ts > 0)
    else:
        print("Commit Timestamp Audit: 'created_at' column MISSING in pr_commits.parquet")
        
    # Check timeline 'committed' events
    committed_events = timeline[timeline['event'] == 'committed']
    if 'created_at' in committed_events.columns:
        valid_ts = committed_events['created_at'].notna().sum()
        total = len(committed_events)
        pct = (100*valid_ts/total) if total > 0 else 0
        print(f"Timeline 'committed' Timestamp Audit: {valid_ts}/{total} valid ({(pct):.1f}%)")
    
    # 3. Create Audit Dataset (Rejected + Feedback)
    audit = prs[['id', 'state', 'merged_at', 'created_at', 'closed_at', 'agent']].copy()
    audit = audit.rename(columns={'id': 'pr_id'})
    # Parse dates
    for col in ['created_at', 'closed_at', 'merged_at']:
        audit[col] = pd.to_datetime(audit[col], errors='coerce', utc=True)
        
    audit = audit.merge(first_feedback, on='pr_id', how='left')
    
    # Filter: Rejected (Closed & Not Merged) AND Has Feedback
    audit['is_merged'] = audit['merged_at'].notna()
    audit['is_closed'] = audit['state'] == 'closed'
    audit['has_feedback'] = audit['first_feedback_at'].notna()
    
    # Commit counts
    commit_counts = commits.groupby('pr_id').size().reset_index(name='num_commits')
    audit = audit.merge(commit_counts, on='pr_id', how='left')
    audit['num_commits'] = audit['num_commits'].fillna(0)
    
    # Target Pool
    pool = audit[audit['is_closed'] & (~audit['is_merged']) & audit['has_feedback']].copy()
    print(f"Audit Pool (Rejected + Feedback): {len(pool)} PRs")
    
    # Calculate Time to Resolution (Close) from Feedback
    pool['hours_to_close'] = (pool['closed_at'] - pool['first_feedback_at']).dt.total_seconds() / 3600.0
    pool['days_to_close'] = pool['hours_to_close'] / 24.0
    
    # Negative time check (Feedback after close?)
    neg_time = (pool['hours_to_close'] < 0).sum()
    if neg_time > 0:
        print(f"Warning: {neg_time} PRs have feedback AFTER close. Setting duration to 0.")
        pool.loc[pool['hours_to_close'] < 0, 'hours_to_close'] = 0
        pool.loc[pool['days_to_close'] < 0, 'days_to_close'] = 0
    
    # Export CSV
    out_csv = OUTPUT_DIR / "audit_ghosting.csv"
    export_cols = ['pr_id', 'agent', 'first_feedback_at', 'closed_at', 'num_commits', 'days_to_close']
    pool[export_cols].sample(min(50, len(pool))).to_csv(out_csv, index=False)
    print(f"Exported audit sample to {out_csv}")
    
    # PLOT: Kaplan-Meier Time-to-Resolution
    kmf = KaplanMeierFitter()
    T = pool['days_to_close'].dropna()
    E = np.ones(len(T)) # Event observed (All are closed)
    
    plt.figure(figsize=(8, 5))
    kmf.fit(T, event_observed=E, label='Time to Close (Rejected PRs)')
    kmf.plot_survival_function()
    plt.title("Time-to-Resolution after Feedback (Rejected PRs)")
    plt.xlabel("Days after Feedback")
    plt.ylabel("Probability Open")
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 30) # 30 day window
    
    out_plot = OUTPUT_DIR / "time_to_close_km.png"
    plt.savefig(out_plot, dpi=300)
    print(f"Saved plot to {out_plot}")
    
    # Report Follow-up within X days
    # Since we lack follow-up timestamp, we use "Closed within X days" as proxy for "Resolution"
    # Ghosting = Open but specific "No response"?
    # Here we report "Resolution Speed".
    d7 = (pool['days_to_close'] <= 7).mean()
    d14 = (pool['days_to_close'] <= 14).mean()
    d30 = (pool['days_to_close'] <= 30).mean()
    
    print(f"Resolution Stats:")
    print(f"  Closed within 7 days: {d7*100:.1f}%")
    print(f"  Closed within 14 days: {d14*100:.1f}%")
    print(f"  Closed within 30 days: {d30*100:.1f}%")
    print(f"  (Note: Missing follow-up commit timestamps prevent 'Time-to-Reply' analysis)")

if __name__ == "__main__":
    main()
