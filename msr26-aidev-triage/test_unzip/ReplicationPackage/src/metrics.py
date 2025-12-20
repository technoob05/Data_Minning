import pandas as pd
import numpy as np

def calculate_acceptance_rate(df: pd.DataFrame, group_by: str = "agent") -> pd.DataFrame:
    """
    Calculates the acceptance rate (merged / (merged + rejected)) grouped by the specified column.
    Excludes open PRs.
    """
    # Filter for closed PRs (merged or rejected)
    closed_df = df[df["status"].isin(["merged", "rejected"])].copy()
    
    if closed_df.empty:
        return pd.DataFrame()

    # Group and calculate
    stats = closed_df.groupby(group_by).agg(
        total_closed=("id", "count"),
        merged_count=("status", lambda x: (x == "merged").sum()),
        rejected_count=("status", lambda x: (x == "rejected").sum())
    ).reset_index()
    
    stats["acceptance_rate"] = stats["merged_count"] / stats["total_closed"]
    
    return stats

def calculate_turnaround_time(df: pd.DataFrame, group_by: str = "agent") -> pd.DataFrame:
    """
    Calculates median turnaround time for merged and rejected PRs.
    """
    closed_df = df[df["status"].isin(["merged", "rejected"])].copy()
    
    if closed_df.empty:
        return pd.DataFrame()
        
    stats = closed_df.groupby([group_by, "status"])["turnaround_time_hours"].median().unstack().reset_index()
    
    # Rename columns for clarity
    stats = stats.rename(columns={"merged": "median_turnaround_merged_hours", "rejected": "median_turnaround_rejected_hours"})
    
    return stats

def calculate_comprehensive_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates a comprehensive set of metrics per agent.
    """
    # Acceptance Rate
    acc_rate = calculate_acceptance_rate(df, group_by="agent")
    
    # Turnaround Time
    turnaround = calculate_turnaround_time(df, group_by="agent")
    
    # Merge
    metrics = acc_rate.merge(turnaround, on="agent", how="outer")
    
    return metrics
