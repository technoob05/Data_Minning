#!/usr/bin/env python3
"""
Sensitivity Analysis: Exclude Dependency/Config-Only PRs
Tests if size dominance persists after filtering automation-heavy PRs
"""

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import roc_auc_score
import lightgbm as lgb

print("="*70)
print("SENSITIVITY: Excluding Dependency/Config-Only PRs")
print("="*70)

# Load data
df = pd.read_parquet("data/processed/features_snapshot.parquet")
print(f"\nTotal PRs: {len(df):,}")

# Identify dependency/config-only PRs
# PRs that ONLY touch deps/ci/docs (no src changes)
dep_config_only = (
    (df['touches_deps'] == 1) | 
    (df['touches_ci'] == 1) |
    (df['touches_docs'] == 1)
) & (df['touches_src'] == 0)

print(f"Dependency/Config-only PRs: {dep_config_only.sum():,} ({dep_config_only.mean():.1%})")

# Filter out
df_filtered = df[~dep_config_only].copy()
print(f"After filtering: {len(df_filtered):,} PRs")

# Re-train model on filtered data
FEATURES_T0 = [
    'additions', 'deletions', 'changed_files', 'total_changes',
    'title_len', 'body_len', 'has_plan',
    'agent_encoded', 'lang_extension',
    'touches_src', 'touches_tests', 'touches_ci', 'touches_docs', 'touches_deps'
]

X = df_filtered[FEATURES_T0]
y = df_filtered['is_high_cost']

# Repo-disjoint split
splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(splitter.split(X, y, groups=df_filtered['repo_full_name']))

X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

# Train LightGBM
model = lgb.LGBMClassifier(
    n_estimators=100,
    max_depth=6,
    class_weight='balanced',
    random_state=42
)
model.fit(X_train, y_train)

# Evaluate
y_pred_proba = model.predict_proba(X_test)[:, 1]
auc_filtered = roc_auc_score(y_test, y_pred_proba)

print(f"\n[RESULTS]")
print(f"  AUC (full dataset): 0.958")
print(f"  AUC (filtered, no dep/config-only): {auc_filtered:.3f}")
print(f"  Difference: {abs(auc_filtered - 0.958):.3f}")

if abs(auc_filtered - 0.958) < 0.01:
    print(f"\n✓ CONCLUSION: Size dominance persists (difference < 0.01)")
else:
    print(f"\n⚠ CONCLUSION: Moderate change observed")

print("="*70)
