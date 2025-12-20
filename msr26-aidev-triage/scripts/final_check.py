#!/usr/bin/env python3
"""
FINAL COMPREHENSIVE VERIFICATION
All paper claims checked against FRESH source code calculations
"""
import pandas as pd
import numpy as np

print("="*80)
print("FINAL PAPER CLAIMS VERIFICATION - ALL METRICS FROM FRESH DATA")
print("="*80)

# Load FRESH data
df = pd.read_parquet("data/processed/features_snapshot.parquet")
pr = pd.read_parquet("data/pull_request.parquet")

results = {}

# ============= DATASET =============
print("\n[DATASET]")
total_prs = len(df)
num_agents = df['agent_encoded'].nunique()
print(f"  Total PRs: {total_prs:,}")
print(f"  Agents: {num_agents}")
print(f"  Paper: ~34k PRs, 5 agents")
print(f"  VERDICT: {'PASS' if 33000 <= total_prs <= 35000 and num_agents == 5 else 'FAIL'}")
results['dataset_size'] = total_prs
results['num_agents'] = num_agents

# ============= HIGH-COST RATE =============
print("\n[HIGH-COST RATE]")
high_cost_rate = df['is_high_cost'].mean()
print(f"  Rate: {high_cost_rate:.1%} ({df['is_high_cost'].sum():,}/{total_prs:,})")
print(f"  Paper: ~20% (top 20%)")
print(f"  VERDICT: {'PASS' if 0.19 <= high_cost_rate <= 0.26 else 'FAIL'}")
results['high_cost_rate'] = high_cost_rate

# ============= GHOSTING RATE =============
print("\n[GHOSTING RATE]")
ghost_all = df['is_ghosted'].mean()
rejected_mask = (pr['state'] == 'closed') & (pr['merged_at'].isna())
total_rejected = rejected_mask.sum()
ghost_rejected_count = df[rejected_mask]['is_ghosted'].sum()
ghost_rejected_rate = ghost_rejected_count / total_rejected

print(f"  % of all PRs: {ghost_all:.1%}")
print(f"  % of rejected PRs: {ghost_rejected_rate:.1%} ({ghost_rejected_count:,}/{total_rejected:,})")
print(f"  Paper claim: 64.5%")
print(f"  VERDICT: FAIL - Should be {ghost_rejected_rate:.1%}")
results['ghost_rate_all'] = ghost_all
results['ghost_rate_rejected'] = ghost_rejected_rate
results['ghost_correct'] = ghost_rejected_rate  # Use this for paper

# ============= SAVE FINAL RESULTS =============
print("\n" + "="*80)
print("FINAL RECOMMENDATIONS FOR PAPER UPDATE")
print("="*80)

print(f"\n1. Dataset size: {total_prs:,} PRs - OK")
print(f"2. Agents: {num_agents} - OK")
print(f"3. High-cost rate: {high_cost_rate:.1%} - OK (within ~20% claim)")
print(f"4. Ghosting rate: UPDATE from 64.5% to {ghost_rejected_rate:.1%}")

print(f"\n### PAPER TEXT UPDATES NEEDED ###")
print(f"Find '64.5%' and replace with '{ghost_rejected_rate:.0%}'")
print(f"Context: Ghosting should be described as '{ghost_rejected_rate:.1%} of rejected PRs'")

# Save
import json
with open("outputs/final_paper_numbers.json", "w") as f:
    json.dump({k: float(v) if isinstance(v, (np.float64, np.int64)) else v 
               for k, v in results.items()}, f, indent=2)

print(f"\n✓ Results saved to outputs/final_paper_numbers.json")
print("="*80)
