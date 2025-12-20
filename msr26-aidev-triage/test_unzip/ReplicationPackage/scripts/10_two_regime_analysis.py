
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

def analyze_two_regimes():
    print("Loading features...")
    df = pd.read_parquet("msr26-aidev-triage/artifacts/pr_features.parquet")
    
    # Define Regimes
    # Instant < 1 minute (already calculated as is_instant_merge in features)
    if "is_instant_merge" not in df.columns:
        # Recalculate if missing
        df["end_date"] = df["merged_at"].combine_first(df["closed_at"])
        df["duration_hours"] = (df["end_date"] - df["created_at"]).dt.total_seconds() / 3600.0
        df["is_instant_merge"] = (df["duration_hours"] < 1/60).astype(int)

    instant = df[df["is_instant_merge"] == 1]
    normal = df[df["is_instant_merge"] == 0]
    
    print(f"Total PRs: {len(df)}")
    print(f"Instant PRs: {len(instant)} ({len(instant)/len(df)*100:.1f}%)")
    print(f"Normal PRs: {len(normal)} ({len(normal)/len(df)*100:.1f}%)")
    
    # 1. Feature Prevalence Comparison
    print("Comparing Feature Prevalence...")
    features = ["touches_tests", "touches_docs", "touches_ci", "touches_deps", "touches_config"]
    
    stats = []
    for feat in features:
        if feat in df.columns:
            i_mean = instant[feat].mean()
            n_mean = normal[feat].mean()
            stats.append({"Feature": feat, "Regime": "Instant", "Prevalence": i_mean})
            stats.append({"Feature": feat, "Regime": "Normal", "Prevalence": n_mean})
            
    stats_df = pd.DataFrame(stats)
    
    # Plot
    plt.figure(figsize=(10, 6))
    sns.barplot(data=stats_df, x="Feature", y="Prevalence", hue="Regime")
    plt.title("Feature Prevalence: Instant vs Normal PRs")
    plt.ylabel("Proportion of PRs")
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig("msr26-aidev-triage/outputs/figures/instant_vs_normal_dist.png")
    print("Saved instant_vs_normal_dist.png")
    
    # 2. Key Metrics Comparison
    metrics = []
    
    # Quantify "Trivial": Config/Deps dominance & Median Changes
    # % touches_config / touches_deps
    for name, d in [("Instant", instant), ("Normal", normal)]:
        config_pct = d["touches_config"].mean() * 100
        deps_pct = d["touches_deps"].mean() * 100
        median_changes = d["total_changes"].median() if "total_changes" in d.columns else 0
        median_additions = d["additions"].median()
        
        print(f"[{name}] Config: {config_pct:.1f}%, Deps: {deps_pct:.1f}%, Median Changes: {median_changes}, Median Additions: {median_additions}")
        
    # Acceptance Rate
    metrics.append({
        "Metric": "Acceptance Rate",
        "Instant": instant[instant["status"]=="merged"].shape[0] / len(instant),
        "Normal": normal[normal["status"]=="merged"].shape[0] / len(normal)
    })
    
    # Ghosting Rate (Rejected + Feedback only)
    def calc_ghosting(d):
        pool = d[(d["status"]=="rejected") & (d["first_human_feedback_at"].notna())]
        if len(pool) == 0: return 0
        return pool["is_ghosted"].mean()

    metrics.append({
        "Metric": "Ghosting Rate",
        "Instant": calc_ghosting(instant),
        "Normal": calc_ghosting(normal)
    })
    
    metrics_df = pd.DataFrame(metrics)
    print("\nKey Metrics:")
    print(metrics_df)
    metrics_df.to_csv("msr26-aidev-triage/outputs/tables/two_regime_stats.csv", index=False)

if __name__ == "__main__":
    analyze_two_regimes()
