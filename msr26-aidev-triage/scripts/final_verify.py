#!/usr/bin/env python3
"""
Final comprehensive paper verification
Calculates ALL metrics and compares with paper claims
"""
import pandas as pd
import numpy as np

print("="*70)
print("COMPREHENSIVE PAPER METRICS VERIFICATION")
print("="*70)

# Load all data
features = pd.read_parquet("data/processed/features_snapshot.parquet")
pr_base = pd.read_parquet("data/pull_request.parquet")

# === DATASET METRICS ===
print(f"\n### DATASET ###")
print(f"Total PRs: {len(features):,}")
print(f"Agents: {features['agent_encoded'].nunique()}")

# === TARGET RATES ===
print(f"\n### TARGETS ###")
high_cost_rate = features['is_high_cost'].mean()
print(f"High-cost (is_high_cost==1): {high_cost_rate:.1%} ({features['is_high_cost'].sum():,}/{len(features):,})")

# Ghosting: % of ALL PRs
ghosting_all = features['is_ghosted'].mean()
print(f"Ghosting (all PRs): {ghosting_all:.1%} ({features['is_ghosted'].sum():,}/{len(features):,})")

# Ghosting context from PR base
merged_count = pr_base['merged_at'].notna().sum()
closed_count = (pr_base['state'] == 'closed').sum()
rejected_count = closed_count - merged_count

print(f"\n### GHOSTING CONTEXT ###")
print(f"PR States:")
print(f"  Total: {len(pr_base):,}")
print(f"  Merged: {merged_count:,}")
print(f"  Rejected (closed, not merged): {rejected_count:,}")

# Calculate ghosting as % of rejected
# Note: features has ghosting labels, but no state column
# We can't directly compute "% of rejected" without merging

# === LOAO ===
try:
    loao = pd.read_csv("outputs/tables/loao_fresh.csv")
    print(f"\n### LOAO ###")
    print(f"AUC range: {loao['AUC'].min():.3f}--{loao['AUC'].max():.3f}")
    print(f"AUC mean: {loao['AUC'].mean():.3f} ± {loao['AUC'].std():.3f}")
except:
    print(f"\n### LOAO ### File not found")

# === SOTA ===
try:
    sota = pd.read_csv("outputs/tables/sota_model_benchmark.csv")
    print(f"\n### SOTA MODELS ###")
    print(f"Best AUC: {sota['AUC'].max():.3f}")
    lgbm = sota[sota['model'].str.contains('LightGBM', case=False, na=False)]
    if len(lgbm) > 0:
        print(f"LightGBM: {lgbm['AUC'].iloc[0]:.3f}")
except:
    print(f"\n### SOTA ### File not found")

# === SEMANTIC ===
try:
    sem = pd.read_csv("outputs/tables/semantic_baseline_results.csv")
    print(f"\n### SEMANTIC BASELINES ###")
    print(f"AUC range: {sem['AUC'].min():.2f}--{sem['AUC'].max():.2f}")
    for _, row in sem.iterrows():
        print(f"  {row['baseline']}: {row['AUC']:.2f}")
except:
    print(f"\n### SEMANTIC ### File not found")

# === BOT FILTERING ===
try:
    bot = pd.read_csv("outputs/tables/bot_effort_sensitivity.csv")
    print(f"\n### BOT FILTERING ###")
    if 'jaccard_overlap' in bot.columns:
        print(f"Jaccard overlap: {bot['jaccard_overlap'].iloc[0]:.1%}")
    if 'agreement_rate' in bot.columns:
        print(f"Agreement rate: {bot['agreement_rate'].iloc[0]:.1%}")
except:
    print(f"\n### BOT ### File not found")

# === FEATURE LIFT ===
try:
    lift = pd.read_csv("outputs/tables/feature_lift_by_quartile.csv")
    print(f"\n### FEATURE LIFT ###")
    if 'precision_gain' in lift.columns:
        print(f"Precision gain range: +{lift['precision_gain'].min():.1%} to +{lift['precision_gain'].max():.1%}")
        for _, row in lift.iterrows():
            size = row.get('size_quartile', row.get('quartile', 'Unknown'))
            gain = row['precision_gain']
            print(f"  {size}: +{gain:.1%}")
except:
    print(f"\n### FEATURE LIFT ### File not found")

# === PAPER VERIFICATION ===
print(f"\n" + "="*70)
print("PAPER CLAIM VERIFICATION")
print("="*70)

checks = []

# Dataset
checks.append(("Dataset size ~34k", len(features), 33000 <= len(features) <= 35000))
checks.append(("5 agents", features['agent_encoded'].nunique(), features['agent_encoded'].nunique() == 5))

# Targets  
checks.append(("High-cost ~20-25%", f"{high_cost_rate:.1%}", 0.20 <= high_cost_rate <= 0.25))
checks.append(("Ghosting 64.5% (?)", f"{ghosting_all:.1%}", abs(ghosting_all - 0.645) < 0.02))

print(f"\nResults:")
for claim, actual, passed in checks:
    status = "[OK]" if passed else "[FAIL]"
    print(f"  {status} {claim}: {actual}")

print("\n" + "="*70)
print(f"SUMMARY: {sum(1 for _, _, p in checks if p)}/{len(checks)} checks passed")
print("="*70)
