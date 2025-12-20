#!/usr/bin/env python3
"""Calculate EXACT ghosting rate matching paper definition"""
import pandas as pd

# Load data
features = pd.read_parquet("data/processed/features_snapshot.parquet")
pr = pd.read_parquet("data/pull_request.parquet")

print("="*60)
print("GHOSTING RATE CALCULATION")
print("="*60)

# Method 1: From features (all PRs)
total_prs = len(features)
ghosted_all = features['is_ghosted'].sum()
rate_all = features['is_ghosted'].mean()

print(f"\n[METHOD 1: % of All PRs]")
print(f"  Total PRs: {total_prs:,}")
print(f"  Ghosted: {ghosted_all:,}")
print(f"  Rate: {rate_all:.1%}")

# Method 2: From features (rejected PRs only)
# Merge with PR state
features_with_state = features.merge(
    pr[['id', 'state', 'merged_at']], 
    on='id', 
    how='inner'
)

rejected = features_with_state[
    (features_with_state['state'] == 'closed') & 
    (features_with_state['merged_at'].isna())
]

ghost_rejected = rejected['is_ghosted'].sum()
total_rejected = len(rejected)
rate_rejected = rejected['is_ghosted'].mean() if total_rejected > 0 else 0

print(f"\n[METHOD 2: % of Rejected PRs]")
print(f"  Total Rejected: {total_rejected:,}")
print(f"  Ghosted (rejected): {ghost_rejected:,}")
print(f"  Rate: {rate_rejected:.1%}")

# Method 3: Cross-check with audit script results
# From audit: 4969 rejected PRs with feedback
# 90.6% closed within 14 days = NOT ghosted
# So ~9.4% ghosted???

print(f"\n[METHOD 3: Audit Cross-Check]")
print(f"  Rejected + Feedback (audit): 4,969")
print(f"  Closed within 14d: 90.6%")
print(f"  IMPLIED ghosting: ~9.4% of rejected+feedback")

# Paper verification
paper_claim = 0.645
print(f"\n[PAPER CLAIM]")
print(f"  Paper: 64.5%")
print(f"  Matches Method 1: {'NO - '+str(abs(rate_all-paper_claim)*100)[:3]+'pp diff'}")
print(f"  Matches Method 2: {'YES!' if abs(rate_rejected-paper_claim) < 0.02 else 'NO - '+str(abs(rate_rejected-paper_claim)*100)[:4]+'pp diff'}")

# Final recommendation
print(f"\n[RECOMMENDATION]")
if abs(rate_rejected - paper_claim) < 0.02:
    print(f"  ✓ Paper claim CORRECT (64.5% of REJECTED PRs)")
elif abs(rate_all - paper_claim) < 0.02:
    print(f"  ? Paper may be using all PRs ({rate_all:.1%})")
else:
    print(f"  ! UPDATE NEEDED")
    print(f"    Best match: {rate_rejected:.1%} of rejected PRs")
    print(f"    OR: {rate_all:.1%} of all PRs")

print("="*60)
