"""
Phase 5.2: Survival Analysis
Kaplan-Meier curves for ghosting dynamics by agent and complexity.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def survival_analysis():
    print("=" * 60)
    print("PHASE 5.2: SURVIVAL ANALYSIS (Ghosting Dynamics)")
    print("=" * 60)
    
    df = pd.read_parquet("msr26-aidev-triage/artifacts/pr_features.parquet")
    
    # Pool: Rejected + Human Feedback (same as ghosting analysis)
    pool = df[(df["status"] == "rejected") & (df["first_human_feedback_at"].notna())].copy()
    print(f"Pool size: {len(pool)}")
    
    # Time to event: days from feedback to follow-up commit
    pool["feedback_ts"] = pd.to_datetime(pool["first_human_feedback_at"])
    pool["followup_ts"] = pd.to_datetime(pool["first_followup_commit_at"])
    pool["time_to_followup"] = (pool["followup_ts"] - pool["feedback_ts"]).dt.total_seconds() / 86400.0
    
    # Event: 1 if follow-up occurred, 0 if censored (ghosted)
    pool["event"] = pool["first_followup_commit_at"].notna().astype(int)
    
    # For censored cases, use 30 days as max observation time
    pool.loc[pool["event"] == 0, "time_to_followup"] = 30
    pool.loc[pool["time_to_followup"] < 0, "time_to_followup"] = 0
    pool.loc[pool["time_to_followup"] > 30, "time_to_followup"] = 30
    
    print(f"Events (follow-up): {pool['event'].sum()}")
    print(f"Censored (ghosted): {(pool['event'] == 0).sum()}")
    
    # Simple Kaplan-Meier by agent
    print("\n--- Survival by Agent ---")
    plt.figure(figsize=(10, 6))
    
    for agent in pool["agent"].unique():
        agent_pool = pool[pool["agent"] == agent]
        if len(agent_pool) < 10:
            continue
        
        # Calculate survival curve (simplified KM)
        times = sorted(agent_pool[agent_pool["event"] == 1]["time_to_followup"].dropna().unique())
        survival = []
        at_risk = len(agent_pool)
        s = 1.0
        prev_t = 0
        
        for t in times:
            events_at_t = ((agent_pool["time_to_followup"] == t) & (agent_pool["event"] == 1)).sum()
            s = s * (1 - events_at_t / at_risk)
            survival.append((t, s))
            at_risk -= events_at_t
        
        if survival:
            ts, ss = zip(*survival)
            plt.step(ts, ss, where='post', label=f"{agent} (n={len(agent_pool)})")
    
    plt.xlabel("Days since first human feedback")
    plt.ylabel("Probability of NOT yet responding")
    plt.title("Kaplan-Meier Survival: Time to First Follow-up Commit")
    plt.legend(loc="lower left")
    plt.xlim(0, 30)
    plt.ylim(0, 1)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("msr26-aidev-triage/paper/survival_by_agent.png", dpi=150)
    plt.close()
    print("Saved survival_by_agent.png")
    
    # Survival by complexity (touches_ci)
    print("\n--- Survival by Complexity ---")
    plt.figure(figsize=(10, 6))
    
    for complexity, label in [(0, "No CI touches"), (1, "Touches CI")]:
        if "touches_ci" not in pool.columns:
            break
        subset = pool[pool["touches_ci"] == complexity]
        if len(subset) < 10:
            continue
        
        times = sorted(subset[subset["event"] == 1]["time_to_followup"].dropna().unique())
        survival = []
        at_risk = len(subset)
        s = 1.0
        
        for t in times:
            events_at_t = ((subset["time_to_followup"] == t) & (subset["event"] == 1)).sum()
            s = s * (1 - events_at_t / at_risk)
            survival.append((t, s))
            at_risk -= events_at_t
        
        if survival:
            ts, ss = zip(*survival)
            plt.step(ts, ss, where='post', label=f"{label} (n={len(subset)})")
    
    plt.xlabel("Days since first human feedback")
    plt.ylabel("Probability of NOT yet responding")
    plt.title("Kaplan-Meier Survival: CI Complexity Effect")
    plt.legend(loc="lower left")
    plt.xlim(0, 30)
    plt.ylim(0, 1)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("msr26-aidev-triage/paper/survival_by_complexity.png", dpi=150)
    plt.close()
    print("Saved survival_by_complexity.png")
    
    print("\nSURVIVAL ANALYSIS COMPLETE")

if __name__ == "__main__":
    survival_analysis()
