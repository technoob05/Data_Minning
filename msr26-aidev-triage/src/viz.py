import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap

def plot_acceptance_rate(metrics_df: pd.DataFrame, output_path: str):
    """
    Plots acceptance rate by agent.
    """
    plt.figure(figsize=(10, 6))
    sns.barplot(data=metrics_df, x="agent", y="acceptance_rate")
    plt.title("Acceptance Rate by Agent")
    plt.ylabel("Acceptance Rate")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def plot_turnaround_time(metrics_df: pd.DataFrame, output_path: str):
    """
    Plots median turnaround time by agent.
    """
    # Melt for easier plotting
    melted = metrics_df.melt(id_vars=["agent"], value_vars=["median_turnaround_merged_hours", "median_turnaround_rejected_hours"], 
                             var_name="status", value_name="hours")
    
    plt.figure(figsize=(10, 6))
    sns.barplot(data=melted, x="agent", y="hours", hue="status")
    plt.title("Median Turnaround Time by Agent")
    plt.ylabel("Hours")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def plot_shap_summary(shap_values, X_test, output_path: str):
    """
    Plots SHAP summary plot.
    """
    plt.figure()
    shap.summary_plot(shap_values, X_test, show=False)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def plot_pareto_effort(df: pd.DataFrame, output_path: str):
    """
    Plots Pareto chart of effort (comments + reviews) vs PR count.
    """
    # Calculate effort score
    if "effort_score" not in df.columns:
        df["effort_score"] = df["num_comments"].fillna(0) + df["num_reviews"].fillna(0)
        
    # Sort by effort descending
    sorted_df = df.sort_values("effort_score", ascending=False)
    
    # Cumulative effort
    total_effort = sorted_df["effort_score"].sum()
    if total_effort == 0:
        print("Warning: Total effort is 0. Skipping Pareto plot.")
        return

    sorted_df["cum_effort"] = sorted_df["effort_score"].cumsum()
    sorted_df["cum_effort_pct"] = sorted_df["cum_effort"] / total_effort
    
    # Cumulative PR count
    sorted_df["pr_rank"] = range(1, len(sorted_df) + 1)
    sorted_df["pr_pct"] = sorted_df["pr_rank"] / len(sorted_df)
    
    plt.figure(figsize=(8, 6))
    plt.plot(sorted_df["pr_pct"], sorted_df["cum_effort_pct"], label="Effort Curve", linewidth=2)
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random")
    
    # Mark 20% PRs
    mask_20 = sorted_df["pr_pct"] <= 0.2
    if mask_20.any():
        top_20_effort = sorted_df.loc[mask_20, "cum_effort_pct"].max()
        plt.scatter([0.2], [top_20_effort], color="red", zorder=5)
        plt.text(0.25, top_20_effort - 0.05, f"Top 20% PRs = {top_20_effort:.1%} Effort", color="red", fontsize=10)
    
    plt.title("Pareto Chart: Human Effort Concentration")
    plt.xlabel("% of PRs (Sorted by Effort)")
    plt.ylabel("% of Total Human Effort")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def plot_ghosting_rate(df: pd.DataFrame, output_path: str):
    """
    Plots ghosting rate by agent.
    Ghosting = Rejected but had effort (num_comments + num_reviews > 0)
    """
    if "effort_score" not in df.columns:
        df["effort_score"] = df["num_comments"].fillna(0) + df["num_reviews"].fillna(0)
        
    # Define ghosted: Closed (not merged) AND Effort > 0
    # Assuming 'state' and 'merged_at' are available or 'status' column
    if "status" in df.columns:
        df["is_ghosted"] = ((df["status"] == "rejected") & (df["effort_score"] > 0)).astype(int)
    else:
        # Fallback if status not pre-calculated
        df["is_ghosted"] = ((df["state"] == "closed") & (df["merged_at"].isna()) & (df["effort_score"] > 0)).astype(int)
    
    # Group by agent
    ghost_stats = df.groupby("agent")["is_ghosted"].mean().reset_index()
    ghost_stats = ghost_stats.sort_values("is_ghosted", ascending=False)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(data=ghost_stats, x="agent", y="is_ghosted")
    plt.title("Ghosting Rate by Agent (Wasted Effort)")
    plt.ylabel("Ghosting Rate (% of PRs)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
