#!/usr/bin/env python3
"""Check all key statistics for paper"""
import pandas as pd
from pathlib import Path

print("="*60)
print("PAPER STATISTICS VERIFICATION")
print("="*60)

# 1. Load features
df = pd.read_parquet("data/processed/features_snapshot.parquet")
print(f"\n[DATASET]")
print(f"  Total PRs: {len(df):,}")
print(f"  Agents: {df['agent_encoded'].nunique()}")

# 2. Target rates
print(f"\n[TARGETS - All PRs]")
print(f"  High-cost: {df['is_high_cost'].mean():.1%} ({df['is_high_cost'].sum():,}/{len(df):,})")
print(f"  Ghosted: {df['is_ghosted'].mean():.1%} ({df['is_ghosted'].sum():,}/{len(df):,})")

# 3. Load raw to check ghosting context
pr = pd.read_parquet("data/pull_request.parquet")
closed = pr[pr["state"] == "closed"]
print(f"\n[GHOSTING CONTEXT]")
print(f"  Total PRs: {len(pr):,}")
print(f"  Closed (not merged): {len(closed):,}")
print(f"  If ghosting is % of CLOSED: {df['is_ghosted'].sum()/len(closed)*100:.1%}")

# 4. LOAO
try:
    loao = pd.read_csv("outputs/tables/loao_fresh.csv")
    print(f"\n[LOAO]")
    print(f"  AUC range: {loao['AUC'].min():.3f} -- {loao['AUC'].max():.3f}")
    print(f"  AUC mean: {loao['AUC'].mean():.3f}")
except:
    print(f"\n[LOAO] Not found")

# 5. SOTA
try:
    sota = pd.read_csv("outputs/tables/sota_model_benchmark.csv")
    print(f"\n[SOTA]")
    print(f"  Best AUC: {sota['AUC'].max():.3f}")
    lgbm = sota[sota['model'].str.contains('LightGBM', case=False, na=False)]
    if len(lgbm) > 0:
        print(f"  LightGBM AUC: {lgbm['AUC'].iloc[0]:.3f}")
except:
    print(f"\n[SOTA] Not found")

# 6. Semantic
try:
    sem = pd.read_csv("outputs/tables/semantic_baseline_results.csv")
    print(f"\n[SEMANTIC BASELINES]")
    print(f"  AUC range: {sem['AUC'].min():.2f} -- {sem['AUC'].max():.2f}")
except:
    print(f"\n[SEMANTIC] Not found")

# 7. Bot
try:
    bot = pd.read_csv("outputs/tables/bot_effort_sensitivity.csv")
    print(f"\n[BOT FILTERING]")
    if 'jaccard_overlap' in bot.columns:
        print(f"  Jaccard: {bot['jaccard_overlap'].iloc[0]:.1%}")
except:
    print(f"\n[BOT] Not found")

# 8. Feature lift
try:
    lift = pd.read_csv("outputs/tables/feature_lift_by_quartile.csv")
    print(f"\n[FEATURE LIFT]")
    if 'precision_gain' in lift.columns:
        print(f"  Gain range: +{lift['precision_gain'].min():.1%} to +{lift['precision_gain'].max():.1%}")
except:
    print(f"\n[LIFT] Not found")

print("\n" + "="*60)
