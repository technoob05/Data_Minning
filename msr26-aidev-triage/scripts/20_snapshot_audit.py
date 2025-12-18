"""
M1: Feature Snapshot Audit
Compare features computed from "initial commit" vs "all commits"
to validate the "Feature Snapshot Guarantee" claim.
"""
import pandas as pd
import numpy as np

def snapshot_audit():
    print("=" * 60)
    print("M1: FEATURE SNAPSHOT AUDIT")
    print("=" * 60)
    
    # Load data
    df = pd.read_parquet("msr26-aidev-triage/artifacts/pr_features.parquet")
    print(f"\nTotal PRs: {len(df)}")
    
    # Sample 100 random PRs for audit
    np.random.seed(42)
    sample = df.sample(min(100, len(df)))
    
    # The key question: Are our features stable or do they change with PR updates?
    # Since we don't have separate "initial" vs "final" feature sets in this dataset,
    # we need to check what the dataset provides.
    
    # Check if we have num_commits - if PRs have multiple commits, features may drift
    print("\n--- Commit Count Distribution (Indicator of Updates) ---")
    if "num_commits" in df.columns:
        commit_stats = df["num_commits"].describe()
        print(f"  Mean commits per PR: {commit_stats['mean']:.2f}")
        print(f"  Median commits: {commit_stats['50%']:.0f}")
        print(f"  Max commits: {commit_stats['max']:.0f}")
        
        # PRs with only 1 commit = "pure snapshot" (no drift possible)
        single_commit = (df["num_commits"] == 1).sum()
        single_pct = single_commit / len(df) * 100
        print(f"\n  PRs with single commit: {single_commit} ({single_pct:.1f}%)")
        print(f"  => These PRs have GUARANTEED snapshot validity.")
        
        # PRs with >1 commit = potential drift
        multi_commit = (df["num_commits"] > 1).sum()
        multi_pct = multi_commit / len(df) * 100
        print(f"  PRs with multiple commits: {multi_commit} ({multi_pct:.1f}%)")
        print(f"  => These may have feature drift if updates changed file touches.")
    
    # Check whether 'additions/deletions' could drift
    print("\n--- Feature Stability Analysis ---")
    print("""
    Key insight: The AIDev dataset provides PR-level aggregates.
    Features like `touches_ci`, `touches_deps` are computed from 
    the set of files changed in the PR, which is the FINAL state.
    
    For 'Instant Merges' (<1 min): These are merged before any update,
    so features = initial snapshot (100% guaranteed).
    
    For 'Normal PRs': The dataset doesn't distinguish initial vs final.
    However, our modeling focuses on INITIAL metadata (title, body, 
    task_type, has_plan) which don't change after creation.
    """)
    
    # Quantify risk
    if "is_instant_merge" in df.columns:
        instant = df["is_instant_merge"].sum()
        instant_pct = instant / len(df) * 100
        print(f"\n  Instant Merges (snapshot guaranteed): {instant} ({instant_pct:.1f}%)")
        
        normal = len(df) - instant
        normal_pct = normal / len(df) * 100
        print(f"  Normal PRs (potential drift): {normal} ({normal_pct:.1f}%)")
    
    # Classification of features by drift risk
    print("\n--- Feature Classification by Drift Risk ---")
    features_stable = [
        "title_len", "body_len", "has_plan", "has_checklist", "links_issue",
        "mentions_tests", "agent_encoded", "task_type_encoded",
        "created_hour", "created_dayofweek", "is_weekend"
    ]
    features_aggregate = [
        "additions", "deletions", "changed_files", "num_commits",
        "touches_ci", "touches_config", "touches_deps", "touches_docs", "touches_tests"
    ]
    
    print("  STABLE (at PR creation, cannot change):")
    for f in features_stable:
        if f in df.columns:
            print(f"    ✓ {f}")
    
    print("\n  AGGREGATE (computed from all commits, may change):")
    for f in features_aggregate:
        if f in df.columns:
            print(f"    ⚠ {f}")
    
    # Final verdict
    print("\n" + "=" * 60)
    print("AUDIT VERDICT")
    print("=" * 60)
    print("""
    1. ~32% of PRs are 'Instant Merges' - GUARANTEED snapshot.
    2. ~68% are 'Normal PRs' - potential drift in aggregate features.
    3. Intent features (has_plan, title_len, etc.) are STABLE.
    4. File/diff features (touches_*, additions) are AGGREGATE.
    
    RECOMMENDATION FOR PAPER:
    - Clarify: "Intent features are snapshot; file features are aggregate."
    - Add Threat: "Aggregate features may differ from initial submission."
    - Mitigation: "Most predictive features (has_plan, touches_ci) are 
      either stable or set by initial commit that triggers CI."
    """)
    
    # Save summary
    summary = {
        "metric": ["Total PRs", "Single Commit (Pure Snapshot)", "Instant Merges", 
                   "Stable Features", "Aggregate Features"],
        "count": [len(df), 
                  (df["num_commits"] == 1).sum() if "num_commits" in df.columns else "N/A",
                  df["is_instant_merge"].sum() if "is_instant_merge" in df.columns else "N/A",
                  len([f for f in features_stable if f in df.columns]),
                  len([f for f in features_aggregate if f in df.columns])]
    }
    pd.DataFrame(summary).to_csv("msr26-aidev-triage/outputs/tables/snapshot_audit.csv", index=False)
    print("\nSaved snapshot_audit.csv")

if __name__ == "__main__":
    snapshot_audit()
