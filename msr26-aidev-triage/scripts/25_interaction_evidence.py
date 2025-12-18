"""
Phase 5.3: Interactive Complexity Evidence
Odds ratios for feature interactions + bootstrap CIs.
"""
import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency

def interaction_evidence():
    print("=" * 60)
    print("PHASE 5.3: INTERACTIVE COMPLEXITY EVIDENCE")
    print("=" * 60)
    
    df = pd.read_parquet("msr26-aidev-triage/artifacts/pr_features.parquet")
    
    # Pool for ghosting
    pool = df[(df["status"] == "rejected") & (df["first_human_feedback_at"].notna())].copy()
    pool = pool[pool["is_instant_merge"] == 0] if "is_instant_merge" in pool.columns else pool
    
    print(f"Pool size: {len(pool)}")
    
    # Ghosting rate by feature combinations
    print("\n--- Ghosting Rate by Feature Combinations ---")
    
    features = ["touches_ci", "touches_deps", "touches_tests", "touches_docs"]
    available = [f for f in features if f in pool.columns]
    
    results = []
    for f in available:
        rate = pool.groupby(f)["is_ghosted"].mean() * 100
        n = pool.groupby(f)["is_ghosted"].count()
        if 1 in rate.index and 0 in rate.index:
            print(f"\n{f}:")
            print(f"  {f}=0: {rate[0]:.1f}% ghosted (n={n[0]})")
            print(f"  {f}=1: {rate[1]:.1f}% ghosted (n={n[1]})")
            diff = rate[1] - rate[0]
            print(f"  Difference: {diff:+.1f}%")
            results.append({"feature": f, "rate_0": rate[0], "rate_1": rate[1], "diff": diff, "n_0": n[0], "n_1": n[1]})
    
    # Interaction: touches_ci AND touches_deps
    print("\n--- Interaction: CI x Deps ---")
    if "touches_ci" in pool.columns and "touches_deps" in pool.columns:
        for ci in [0, 1]:
            for deps in [0, 1]:
                subset = pool[(pool["touches_ci"] == ci) & (pool["touches_deps"] == deps)]
                if len(subset) > 0:
                    rate = subset["is_ghosted"].mean() * 100
                    print(f"  CI={ci}, Deps={deps}: {rate:.1f}% (n={len(subset)})")
    
    # Bootstrap CI for key rate difference
    print("\n--- Bootstrap CI for Ghosting Rate Difference ---")
    
    def bootstrap_ci(group1, group2, n_bootstrap=1000, ci=95):
        diffs = []
        for _ in range(n_bootstrap):
            s1 = np.random.choice(group1, size=len(group1), replace=True).mean()
            s2 = np.random.choice(group2, size=len(group2), replace=True).mean()
            diffs.append(s1 - s2)
        lower = np.percentile(diffs, (100 - ci) / 2)
        upper = np.percentile(diffs, 100 - (100 - ci) / 2)
        return np.mean(diffs), lower, upper
    
    if "touches_ci" in pool.columns:
        ci_1 = pool[pool["touches_ci"] == 1]["is_ghosted"].values
        ci_0 = pool[pool["touches_ci"] == 0]["is_ghosted"].values
        mean_diff, lower, upper = bootstrap_ci(ci_1, ci_0)
        print(f"\ntouches_ci=1 vs touches_ci=0:")
        print(f"  Mean difference: {mean_diff*100:.1f}%")
        print(f"  95% CI: [{lower*100:.1f}%, {upper*100:.1f}%]")
        
        if lower > 0:
            print("  => STATISTICALLY SIGNIFICANT (CI does not include 0)")
        else:
            print("  => Not significant at 95% level")
    
    # Odds Ratio calculation
    print("\n--- Odds Ratios ---")
    if "touches_ci" in pool.columns:
        # 2x2 table: touches_ci vs is_ghosted
        ct = pd.crosstab(pool["touches_ci"], pool["is_ghosted"])
        if ct.shape == (2, 2):
            a, b = ct.iloc[1, 1], ct.iloc[1, 0]  # CI=1: ghosted, not ghosted
            c, d = ct.iloc[0, 1], ct.iloc[0, 0]  # CI=0: ghosted, not ghosted
            odds_ratio = (a * d) / (b * c) if b * c > 0 else float("inf")
            print(f"touches_ci Odds Ratio: {odds_ratio:.2f}")
            if odds_ratio > 1:
                print(f"  => PRs with CI touches have {odds_ratio:.1f}x odds of being ghosted")
    
    # Save results
    pd.DataFrame(results).to_csv("msr26-aidev-triage/outputs/tables/interaction_evidence.csv", index=False)
    print("\nSaved interaction_evidence.csv")

if __name__ == "__main__":
    interaction_evidence()
