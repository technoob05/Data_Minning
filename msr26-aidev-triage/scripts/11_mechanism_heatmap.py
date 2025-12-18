
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

def analyze_mechanism():
    print("Loading features...")
    df = pd.read_parquet("msr26-aidev-triage/artifacts/pr_features.parquet")
    
    # Filter for valid PRs (Merged or Rejected)
    df = df[df["status"].isin(["merged", "rejected"])].copy()
    
    # Create Interaction Feature: Complexity Score
    # Simple proxies: tests, docs, ci
    # We want to see how effort or ghosting varies with these
    
    print("Generating Complexity Heatmap...")
    # Group by touches_tests and touches_docs
    # Calculate Mean Effort Score
    
    heatmap_data = df.groupby(["touches_tests", "touches_docs"])["effort_score"].mean().unstack()
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(heatmap_data, annot=True, fmt=".1f", cmap="YlOrRd", cbar_kws={'label': 'Mean Effort Score'})
    plt.title("Complexity Heatmap: Effort by Components")
    plt.xlabel("Touches Docs")
    plt.ylabel("Touches Tests")
    plt.tight_layout()
    plt.savefig("msr26-aidev-triage/outputs/figures/complexity_heatmap_effort.png")
    print("Saved complexity_heatmap_effort.png")
    
    # Ghosting Heatmap (for rejected pool)
    rejected_pool = df[(df["status"]=="rejected") & (df["first_human_feedback_at"].notna())].copy()
    if len(rejected_pool) > 0:
        heatmap_ghost = rejected_pool.groupby(["touches_tests", "touches_docs"])["is_ghosted"].mean().unstack()
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(heatmap_ghost, annot=True, fmt=".2f", cmap="Purples", cbar_kws={'label': 'Ghosting Rate'})
        plt.title("Risk Heatmap: Ghosting Rate by Components")
        plt.xlabel("Touches Docs")
        plt.ylabel("Touches Tests")
        plt.tight_layout()
        plt.savefig("msr26-aidev-triage/outputs/figures/complexity_heatmap_ghosting.png")
        print("Saved complexity_heatmap_ghosting.png")
    else:
        print("No rejected PRs suitable for ghosting heatmap.")

if __name__ == "__main__":
    analyze_mechanism()
