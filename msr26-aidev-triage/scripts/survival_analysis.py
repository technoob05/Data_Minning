#!/usr/bin/env python3
"""
Survival Analysis of Agent PRs after Feedback
(Kaplan-Meier of Time-to-Resolution)

Due to missing timestamps for 'committed' events, we analyze:
Time from First Human Feedback -> PR Close/Merge.

This proxies interactions:
- Fast drop: Quick resolution (merge or quick reject).
- Long tail: Lingering/Ghosting PRs that waste maintainer attention.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter
from pathlib import Path

# Paths
DATA_DIR = Path("data")
OUTPUT_DIR = Path("outputs/analysis")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def load_data():
    """Load PR and Timeline data."""
    prs = pd.read_parquet(DATA_DIR / "pull_request.parquet")
    timeline = pd.read_parquet(DATA_DIR / "pr_timeline.parquet")
    
    # Timestamps
    for col in ['created_at', 'closed_at', 'merged_at']:
        if col in prs.columns:
            prs[col] = pd.to_datetime(prs[col], errors='coerce', utc=True)
            
    if 'created_at' in timeline.columns:
        timeline['created_at'] = pd.to_datetime(timeline['created_at'], errors='coerce', utc=True)
        
    return prs, timeline

def prepare_data(prs, timeline):
    """Compute feedback times and merge."""
    
    # 1. Identify First Human Feedback
    timeline['actor'] = timeline['actor'].astype(str)
    feedback_mask = (
        (timeline['event'].isin(['commented', 'reviewed'])) & 
        (~timeline['actor'].str.lower().str.contains('bot', na=False)) &
        (~timeline['actor'].str.contains(r'\[bot\]', regex=True))
    )
    feedback_events = timeline[feedback_mask].copy()
    
    # First feedback per PR
    first_feedback = feedback_events.groupby('pr_id')['created_at'].min().reset_index()
    first_feedback = first_feedback.rename(columns={'created_at': 'feedback_at'})
    
    # 2. Merge with PR metadata
    # Map PR ID
    df = prs.merge(first_feedback, left_on='id', right_on='pr_id', how='inner')
    
    # 3. Calculate Duration
    # Duration = (Closed/Merged - Feedback)
    # Event = 1 (Observed) if Closed/Merged. 0 (Censored) if still Open.
    
    # Use closed_at (merged PRs also have closed_at usually, or use max(closed, merged))
    # If closed_at is NaT (Open), separate handling.
    
    df['end_time'] = df['closed_at']
    df['event_observed'] = df['end_time'].notna().astype(int)
    
    # If still open, use 'now' (or max dataset time) for censoring
    max_time = df['end_time'].max()
    if pd.isna(max_time):
        max_time = pd.Timestamp.now(tz='UTC')
    
    df['end_time'] = df['end_time'].fillna(max_time)
    
    df['duration_hours'] = (df['end_time'] - df['feedback_at']).dt.total_seconds() / 3600
    df['duration_days'] = df['duration_hours'] / 24
    
    # Filter non-negative (race conditions)
    df = df[df['duration_days'] >= 0].copy()
    
    return df

def plot_km(df):
    """Plot Kaplan-Meier curves."""
    plt.figure(figsize=(10, 6))
    ax = plt.subplot(111)
    
    top_agents = df['agent'].value_counts().nlargest(5).index.tolist()
    
    kmf = KaplanMeierFitter()
    
    for agent in top_agents:
        cohort = df[df['agent'] == agent]
        kmf.fit(
            cohort['duration_days'],
            event_observed=cohort['event_observed'],
            label=f"{agent} (n={len(cohort)})"
        )
        kmf.plot_survival_function(ax=ax, ci_show=False, linewidth=2)
        
    plt.title("Post-Feedback Survival Analysis (Time-to-Resolution)")
    plt.xlabel("Days since First Human Feedback")
    plt.ylabel("Probability of PR Remaining Open")
    plt.xlim(0, 14)
    plt.ylim(0, 1.0)
    plt.grid(True, alpha=0.3)
    plt.legend(title="Agent")
    
    # Add annotation
    plt.axvline(7, color='r', linestyle='--', alpha=0.3)
    plt.text(7.2, 0.05, '7 Days', color='r')
    
    out_path = OUTPUT_DIR / "survival_km.png"
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Copy to paper
    import shutil
    shutil.copy(out_path, Path("paper/survival_km.png"))
    print(f"Saved figure to {out_path}")

def main():
    print("Loading data...")
    prs, timeline = load_data()
    
    print("Preparing data...")
    df = prepare_data(prs, timeline)
    print(f"Analysis dataset: {len(df)} PRs with feedback")
    
    plot_km(df)
    print("Survival analysis complete.")

if __name__ == "__main__":
    main()
