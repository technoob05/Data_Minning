
import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = Path("data")
OUT_DIR = Path("outputs/metrics")
OUT_DIR.mkdir(parents=True, exist_ok=True)

def main():
    print("Loading data...")
    # Load from processed/valid_features.csv for consistency
    # But usually validity lock means raw data.
    # I'll use valid_features.csv as canonical.
    path = Path("data/processed/valid_features.csv")
    if not path.exists():
        path = Path("msr26-aidev-triage/artifacts/pr_features.parquet")
        
    if str(path).endswith('.csv'):
        df = pd.read_csv(path)
    else:
        df = pd.read_parquet(path)
        
    print(f"Loaded {len(df)} records")
    
    # Check dependencies
    if 'comments' not in df.columns or 'review_comments' not in df.columns:
        print("Error: Missing 'comments' or 'review_comments'")
        return

    # 1. Define Effort Metrics
    df['E0_Total'] = df['comments'].fillna(0) + df['review_comments'].fillna(0)
    df['E1_Reviews'] = df['review_comments'].fillna(0) # Proxy if 'reviews' count not avail
    df['E2_Comments'] = df['comments'].fillna(0)
    df['E3_Weighted'] = 2 * df['review_comments'].fillna(0) + df['comments'].fillna(0)
    
    # 2. Define Wasted Effort
    # Assume 'merged_at' indicates merge. If missing -> non-merged.
    # 'state' -> closed/open.
    # Wasted = Effort on Closed & Unmerged PRs.
    # Need merged status. 'is_merged' usually in feature set?
    if 'merged_at' in df.columns:
        is_merged = df['merged_at'].notna()
    elif 'state' in df.columns:
        # If merged_at missing, assume 'merged' state?
        # valid_features.csv might not have merged_at.
        # I'll check.
        is_merged = (df['state'] == 'merged') # If state has 'merged'
    else:
        print("Warning: Cannot determine merge status. Wasted Effort calc skipped.")
        is_merged = pd.Series([False]*len(df)) # Conservative
        
    df['is_wasted'] = (~is_merged).astype(int)
    
    for metric in ['E0_Total', 'E1_Reviews', 'E2_Comments', 'E3_Weighted']:
        df[f'Wasted_{metric}'] = df[metric] * df['is_wasted']
        
    # 3. Correlation Matrix
    metrics = ['E0_Total', 'E1_Reviews', 'E2_Comments', 'E3_Weighted']
    corr = df[metrics].corr()
    print("\nEffort Metric Correlation:")
    print(corr.round(2))
    
    # 4. Wasted Stats
    total_wasted = df[f'Wasted_E0_Total'].sum()
    total_effort = df['E0_Total'].sum()
    pct_wasted = (total_wasted / total_effort) * 100 if total_effort > 0 else 0
    
    print(f"\nTotal Effort (E0): {total_effort:,.0f}")
    print(f"Total Wasted Effort: {total_wasted:,.0f} ({pct_wasted:.1f}%)")
    
    # Export
    out_path = OUT_DIR / "effort_stats.csv"
    df[metrics + [f'Wasted_{m}' for m in metrics]].describe().to_csv(out_path)
    print(f"Exported stats to {out_path}")

if __name__ == "__main__":
    main()
