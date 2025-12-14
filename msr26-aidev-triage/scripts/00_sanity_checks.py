import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parents[1]))

def run_sanity_checks():
    print("Running Sanity Checks...")
    
    # Load PR Base
    pr_base_path = Path("msr26-aidev-triage/artifacts/pr_base.parquet")
    if not pr_base_path.exists():
        print("Error: pr_base.parquet not found. Run 02_build_pr_base.py first.")
        return

    df = pd.read_parquet(pr_base_path)
    print(f"Loaded {len(df)} rows from pr_base.")

    # 1. Timezone & Date Checks
    print("\n--- 1. Timezone & Date Consistency ---")
    date_cols = ["created_at", "closed_at", "merged_at"]
    for col in date_cols:
        if col in df.columns:
            # Check if timezone aware
            is_tz_aware = df[col].dt.tz is not None
            print(f"{col}: Timezone Aware? {is_tz_aware}")
            # Ensure UTC for calculation
            if not is_tz_aware:
                print(f"Warning: {col} is not timezone aware. Converting to UTC...")
                df[col] = pd.to_datetime(df[col], utc=True)
            else:
                df[col] = df[col].dt.tz_convert("UTC")

    # 2. Duration Logic Checks
    print("\n--- 2. Duration Logic (Negative & Zero) ---")
    
    # Calculate end_date
    df["end_date"] = df["merged_at"].combine_first(df["closed_at"])
    df["duration_hours"] = (df["end_date"] - df["created_at"]).dt.total_seconds() / 3600.0
    
    # Check for negative durations
    neg_duration = df[df["duration_hours"] < 0]
    print(f"Negative durations: {len(neg_duration)} ({len(neg_duration)/len(df)*100:.2f}%)")
    if len(neg_duration) > 0:
        print("Sample negative duration:")
        print(neg_duration[["created_at", "merged_at", "closed_at", "duration_hours"]].head())

    # Check for near-zero durations (< 1 minute)
    zero_duration = df[(df["duration_hours"] >= 0) & (df["duration_hours"] < 1/60)]
    print(f"Near-zero durations (< 1 min): {len(zero_duration)} ({len(zero_duration)/len(df)*100:.2f}%)")
    
    # Breakdown by Agent for Zero Durations
    if len(zero_duration) > 0:
        print("\nNear-zero duration breakdown by Agent:")
        print(zero_duration["agent"].value_counts())

    # 3. Missing Data
    print("\n--- 3. Missing Data Patterns ---")
    print(f"Missing merged_at (Open/Closed unmerged): {df['merged_at'].isna().sum()} ({df['merged_at'].isna().mean()*100:.1f}%)")
    print(f"Missing closed_at: {df['closed_at'].isna().sum()}")
    
    # Check if we have repo info (join success)
    if "repo_name" in df.columns:
        missing_repo = df["repo_name"].isna().sum()
    elif "full_name" in df.columns: # Assuming repo table has full_name
        missing_repo = df["full_name"].isna().sum()
    else:
        missing_repo = "Column not found"
    print(f"Missing Repo Info: {missing_repo}")

    # 4. OpenAI Codex Specific Investigation
    print("\n--- 4. OpenAI Codex Deep Dive ---")
    codex_df = df[df["agent"] == "OpenAI_Codex"]
    if not codex_df.empty:
        merged_codex = codex_df[codex_df["merged_at"].notna()]
        print(f"Codex Total: {len(codex_df)}")
        print(f"Codex Merged: {len(merged_codex)}")
        if not merged_codex.empty:
            print(f"Codex Median Merge Time (hours): {merged_codex['duration_hours'].median():.4f}")
            print(f"Codex Mean Merge Time (hours): {merged_codex['duration_hours'].mean():.4f}")
            print("Distribution of Codex Merge Times (First 10 sorted):")
            print(merged_codex["duration_hours"].sort_values().head(10).values)
    
    # 5. Leakage Check (Basic)
    print("\n--- 5. Potential Leakage Indicators ---")
    # If we have features like 'num_commits' or 'num_reviews', check if they are 0 for instant merges
    # Instant merges shouldn't logically have many reviews if human review is required.
    if "num_reviews" in df.columns:
        instant_merges_with_reviews = df[
            (df["duration_hours"] < 1/60) & 
            (df["num_reviews"] > 0)
        ]
        print(f"Instant merges (<1min) with >0 reviews: {len(instant_merges_with_reviews)}")

if __name__ == "__main__":
    run_sanity_checks()
