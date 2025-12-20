import sys
from pathlib import Path
import pandas as pd

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.viz import plot_acceptance_rate, plot_turnaround_time, plot_pareto_effort, plot_ghosting_rate

def main():
    metrics_path = "msr26-aidev-triage/outputs/tables/agent_metrics.csv"
    features_path = "msr26-aidev-triage/artifacts/pr_features.parquet"
    output_dir = "msr26-aidev-triage/outputs/figures"
    
    print(f"Loading metrics from {metrics_path}...")
    if not Path(metrics_path).exists():
        print("Error: Metrics table not found. Run 03_metrics.py first.")
        return

    metrics_df = pd.read_csv(metrics_path)
    
    # Plot Acceptance Rate
    acc_path = Path(output_dir) / "acceptance_rate.png"
    print(f"Saving acceptance rate plot to {acc_path}...")
    plot_acceptance_rate(metrics_df, str(acc_path))
    
    # Plot Turnaround Time
    turn_path = Path(output_dir) / "turnaround_time.png"
    print(f"Saving turnaround time plot to {turn_path}...")
    plot_turnaround_time(metrics_df, str(turn_path))
    
    # Load features for detailed plots (Pareto & Ghosting)
    print(f"Loading features from {features_path}...")
    if Path(features_path).exists():
        feat_df = pd.read_parquet(features_path)
        
        # Plot Pareto
        pareto_path = Path(output_dir) / "pareto_effort.png"
        print(f"Saving Pareto plot to {pareto_path}...")
        plot_pareto_effort(feat_df, str(pareto_path))
        
        # Plot Ghosting
        ghost_path = Path(output_dir) / "ghosting_rate.png"
        print(f"Saving Ghosting plot to {ghost_path}...")
        plot_ghosting_rate(feat_df, str(ghost_path))
    else:
        print("Warning: Features table not found. Skipping Pareto and Ghosting plots.")
    
    print("Done.")

if __name__ == "__main__":
    main()
