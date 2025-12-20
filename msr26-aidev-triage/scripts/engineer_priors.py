
import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = Path("data/processed")
if not DATA_DIR.exists(): DATA_DIR.mkdir(parents=True)

def main():
    print("Loading valid features...")
    # Load canonical set
    try:
        df = pd.read_csv("data/processed/valid_features.csv")
    except:
        df = pd.read_parquet("msr26-aidev-triage/artifacts/pr_features.parquet")
    
    # Ensure datetime
    if 'created_at' in df.columns:
        df['created_at'] = pd.to_datetime(df['created_at'], utc=True)
    
    # Ensure ID (Standardize with build_snapshot logic)
    if 'id' not in df.columns:
        if 'repo_full_name' in df.columns and 'pr_number' in df.columns:
            df['id'] = df['repo_full_name'] + '_' + df['pr_number'].astype(str)
        else:
            print("Error: Cannot generate ID (missing repo or number)")
            return

    # Ensure target
    if 'ghosted' not in df.columns:
        print("Calculating ghosted target...")
        # Re-calc ghost logic if missing (simplified)
        # Assuming 'is_ghost' logic from audit
        # But for features, we need 'is_high_cost' mostly.
        # Paper says: "High Cost = Top 20%".
        # Let's target 'is_high_cost' for priors too?
        # Or 'ghosted'?
        # The Ghosting rate is a key signal.
        # Let's calculate 'is_ghost' if missing.
        # Ghost = Rejected + Feedback + No Followup?
        # We don't have 'No Followup' easily here without joins.
        # We'll use 'is_high_cost' as the main target for priors.
        # Or just use 'merged' rate?
        # 'merged_at' exists.
        df['is_merged'] = df['merged_at'].notna().astype(int)
        target = 'is_merged' # Success rate is a good prior.
        
    else:
        target = 'ghosted'

    print("Columns:", df.columns.tolist())
    
    # Time-based split for priors
    # We want prior = average of *past* PRs.
    # Sort by time
    df = df.sort_values('created_at')
    
    # Calculate expanding mean (cumulative)
    # Group by Agent
    # shift(1) ensures we don't include current PR (leakage prevention)
    
    # Determine agent col
    if 'author_agent' in df.columns:
        agent_col = 'author_agent'
    elif 'agent' in df.columns:
        agent_col = 'agent'
    else:
        agent_col = 'agent_id'
    
    print(f"Calculating Agent Priors (col: {agent_col})...")
    # Success Rate
    df['agent_success_rate'] = df.groupby(agent_col)['is_merged'].transform(
        lambda x: x.shift(1).expanding().mean()
    ).fillna(0.5) # Default to 0.5 or global mean
    
    # Count (Confidence)
    df['agent_pr_count'] = df.groupby(agent_col)['is_merged'].cumcount()
    
    print("Calculating Repo Priors...")
    # Repo Success Rate
    df['repo_success_rate'] = df.groupby('repo_full_name')['is_merged'].transform(
        lambda x: x.shift(1).expanding().mean()
    ).fillna(0.5)
    
    # Save as new snapshot source
    # We only keep ID + derived priors
    priors = df[['id', 'agent_success_rate', 'agent_pr_count', 'repo_success_rate']].copy()
    
    # Merge with existing snapshot
    snap_path = DATA_DIR / "features_snapshot.parquet"
    if snap_path.exists():
        snap = pd.read_parquet(snap_path)
        # Merge
        # Check ID col
        if 'id' in snap.columns:
            snap = snap.merge(priors, on='id', how='left')
            # Save
            out_path = DATA_DIR / "features_snapshot_with_priors.parquet"
            snap.to_parquet(out_path)
            print(f"Saved Snapshot+Priors to {out_path}")
            print("Preview:", snap[['agent_success_rate', 'repo_success_rate']].head())
        else:
            print("Error: Snapshot missing 'id' column")
    else:
        print("Error: features_snapshot.parquet not found")

if __name__ == "__main__":
    main()
