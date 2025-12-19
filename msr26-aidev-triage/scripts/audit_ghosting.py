#!/usr/bin/env python3
"""
Ghosting Label Audit Script (Phase 1: Validity Hardening)

Validates the "True Ghosting" label by sampling PRs and computing follow-up timing.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Paths
DATA_DIR = Path("data")
OUTPUT_DIR = Path("outputs/audit")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def load_data():
    """Load required dataframes."""
    print("Loading data...")
    prs = pd.read_parquet(DATA_DIR / "pull_request.parquet")
    timeline = pd.read_parquet(DATA_DIR / "pr_timeline.parquet")
    commits = pd.read_parquet(DATA_DIR / "pr_commits.parquet")
    
    # Parse datetime columns
    for col in ['created_at', 'closed_at', 'merged_at']:
        if col in prs.columns:
            prs[col] = pd.to_datetime(prs[col], errors='coerce')
    
    if 'created_at' in timeline.columns:
        timeline['created_at'] = pd.to_datetime(timeline['created_at'], errors='coerce')
    
    return prs, timeline, commits

def compute_feedback_timestamps(prs, timeline):
    """Compute first human feedback timestamp per PR."""
    # Column is 'event' 
    feedback_events = timeline[
        timeline['event'].isin(['commented', 'reviewed', 'review_requested', 
                                'IssueComment', 'PullRequestReviewComment', 
                                'PullRequestReview'])
    ].copy()
    
    if len(feedback_events) == 0:
        print(f"Warning: No feedback events found. Available events: {timeline['event'].unique()[:10]}")
        return pd.DataFrame(columns=['pr_id', 'first_human_feedback_at'])
    
    feedback_events['event_time'] = pd.to_datetime(feedback_events['created_at'], errors='coerce')
    
    # Get first feedback per PR
    first_feedback = feedback_events.groupby('pr_id')['event_time'].min().reset_index()
    first_feedback.columns = ['pr_id', 'first_human_feedback_at']
    
    return first_feedback

def compute_commit_timestamps(commits):
    """Get commit count and timing per PR."""
    # Commits don't have timestamp, so we'll just count
    commit_counts = commits.groupby('pr_id').size().reset_index(name='num_commits')
    return commit_counts

def create_audit_dataset(prs, first_feedback, commit_counts):
    """Create audit dataset."""
    
    # Normalize PR id column
    audit = prs[['id', 'number', 'title', 'state', 'agent', 'created_at', 'closed_at', 'merged_at']].copy()
    audit = audit.rename(columns={'id': 'pr_id'})
    
    # Merge feedback data
    audit = audit.merge(first_feedback, on='pr_id', how='left')
    audit = audit.merge(commit_counts, on='pr_id', how='left')
    
    # Compute derived columns
    audit['is_merged'] = audit['merged_at'].notna()
    audit['is_rejected'] = (audit['state'] == 'closed') & (~audit['is_merged'])
    audit['has_feedback'] = audit['first_human_feedback_at'].notna()
    
    # Time from feedback to close
    audit['feedback_to_close_days'] = (
        audit['closed_at'] - audit['first_human_feedback_at']
    ).dt.total_seconds() / 86400
    
    return audit

def generate_audit_report(audit):
    """Generate audit statistics."""
    
    print("\n" + "="*60)
    print("GHOSTING LABEL AUDIT REPORT")
    print("="*60)
    
    print(f"\n[Dataset Overview]")
    print(f"  Total PRs: {len(audit):,}")
    print(f"  Merged PRs: {audit['is_merged'].sum():,} ({100*audit['is_merged'].mean():.1f}%)")
    print(f"  Rejected PRs: {audit['is_rejected'].sum():,} ({100*audit['is_rejected'].mean():.1f}%)")
    
    # Feedback analysis
    print(f"\n[Feedback Coverage]")
    print(f"  PRs with feedback: {audit['has_feedback'].sum():,} ({100*audit['has_feedback'].mean():.1f}%)")
    print(f"  PRs without feedback: {(~audit['has_feedback']).sum():,}")
    
    # Focus on rejected+feedback pool
    pool = audit[(audit['is_rejected']) & (audit['has_feedback'])].copy()
    print(f"\n[Rejected + Feedback Pool]")
    print(f"  Count: {len(pool):,}")
    
    if len(pool) > 0:
        # Count single-commit PRs (indicative of no follow-up)
        single_commit = (pool['num_commits'] == 1).sum()
        print(f"  Single-commit PRs (no follow-up): {single_commit:,} ({100*single_commit/len(pool):.1f}%)")
        
        # Time from feedback to close
        valid_time = pool['feedback_to_close_days'].dropna()
        if len(valid_time) > 0:
            print(f"\n[Feedback-to-Close Timing (for closed PRs)]")
            print(f"  Median: {valid_time.median():.1f} days")
            print(f"  Mean: {valid_time.mean():.1f} days")
            print(f"  % closed within 1 day of feedback: {100*(valid_time < 1).mean():.1f}%")
            print(f"  % closed within 7 days of feedback: {100*(valid_time < 7).mean():.1f}%")
            print(f"  % closed within 14 days of feedback: {100*(valid_time < 14).mean():.1f}%")
    
    # Agent breakdown
    print(f"\n[Ghosting by Agent (in Rejected+Feedback pool)]")
    if len(pool) > 0:
        agent_stats = pool.groupby('agent').agg({
            'pr_id': 'count',
            'num_commits': lambda x: (x == 1).mean()  # % single-commit
        }).reset_index()
        agent_stats.columns = ['Agent', 'Count', 'Single-Commit %']
        agent_stats['Single-Commit %'] = agent_stats['Single-Commit %'] * 100
        print(agent_stats.to_string(index=False))
    
    return pool

def plot_feedback_timing(pool):
    """Plot feedback-to-close timing."""
    
    valid_time = pool['feedback_to_close_days'].dropna()
    valid_time = valid_time[valid_time >= 0]
    
    if len(valid_time) < 5:
        print("Warning: Not enough data for timing plot")
        return
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # ECDF
    sorted_vals = np.sort(valid_time)
    ecdf = np.arange(1, len(sorted_vals)+1) / len(sorted_vals)
    ax1.plot(sorted_vals, ecdf, 'b-', linewidth=2)
    ax1.axvline(1, color='green', linestyle='--', label='1 day')
    ax1.axvline(7, color='orange', linestyle='--', label='7 days')
    ax1.axvline(14, color='red', linestyle='--', label='14 days')
    ax1.set_xlabel('Days from Feedback to Close')
    ax1.set_ylabel('Cumulative Probability')
    ax1.set_title('ECDF: Time from Feedback to PR Close')
    ax1.legend()
    ax1.set_xlim(0, 30)
    ax1.grid(True, alpha=0.3)
    
    # Histogram
    ax2.hist(valid_time.clip(upper=60), bins=30, edgecolor='black', alpha=0.7)
    ax2.axvline(1, color='green', linestyle='--', label='1 day')
    ax2.axvline(7, color='orange', linestyle='--', label='7 days')
    ax2.axvline(14, color='red', linestyle='--', label='14 days')
    ax2.set_xlabel('Days from Feedback to Close')
    ax2.set_ylabel('Count')
    ax2.set_title('Histogram: Feedback-to-Close Timing')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "followup_ecdf.png", dpi=150, bbox_inches='tight')
    print(f"\nSaved timing plot to {OUTPUT_DIR / 'followup_ecdf.png'}")
    plt.close()

def export_sample(pool, n=50):
    """Export sample for manual inspection."""
    if len(pool) == 0:
        return
    
    sample = pool.sample(min(n, len(pool)), random_state=42)
    sample_cols = ['pr_id', 'number', 'title', 'agent', 'state', 'created_at', 
                   'closed_at', 'first_human_feedback_at', 'num_commits',
                   'feedback_to_close_days']
    sample_cols = [c for c in sample_cols if c in sample.columns]
    sample[sample_cols].to_csv(OUTPUT_DIR / "ghosting_audit_sample.csv", index=False)
    print(f"Saved audit sample ({len(sample)} PRs) to {OUTPUT_DIR / 'ghosting_audit_sample.csv'}")

def main():
    prs, timeline, commits = load_data()
    print(f"Loaded {len(prs):,} PRs, {len(timeline):,} timeline events, {len(commits):,} commits")
    
    # Show event types
    print(f"Event types: {timeline['event'].unique()[:15]}")
    
    first_feedback = compute_feedback_timestamps(prs, timeline)
    commit_counts = compute_commit_timestamps(commits)
    
    audit = create_audit_dataset(prs, first_feedback, commit_counts)
    pool = generate_audit_report(audit)
    
    if len(pool) > 0:
        plot_feedback_timing(pool)
        export_sample(pool)
    
    print("\n" + "="*60)
    print("AUDIT COMPLETE")
    print("="*60)

if __name__ == "__main__":
    main()
