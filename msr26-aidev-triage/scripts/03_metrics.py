import sys
from pathlib import Path
import pandas as pd

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.metrics import calculate_comprehensive_metrics

def main():
    input_path = "msr26-aidev-triage/artifacts/pr_base.parquet"
    output_dir = "msr26-aidev-triage/outputs/tables"
    
    print(f"Loading PR base table from {input_path}...")
    if not Path(input_path).exists():
        print("Error: PR base table not found. Run 02_build_pr_base.py first.")
        return

    pr_df = pd.read_parquet(input_path)
    
    # Calculate metrics
    print("Calculating metrics...")
    metrics_df = calculate_comprehensive_metrics(pr_df)
    
    # Save
    output_path = Path(output_dir) / "agent_metrics.csv"
    print(f"Saving metrics to {output_path}...")
    metrics_df.to_csv(output_path, index=False)
    
    print("Metrics:")
    print(metrics_df)

if __name__ == "__main__":
    main()
