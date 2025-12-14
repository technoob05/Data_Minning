import sys
from pathlib import Path
import pandas as pd

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.features import engineer_features

def main():
    input_path = "msr26-aidev-triage/artifacts/pr_base.parquet"
    output_dir = "msr26-aidev-triage/artifacts"
    
    print(f"Loading PR base table from {input_path}...")
    if not Path(input_path).exists():
        print("Error: PR base table not found. Run 02_build_pr_base.py first.")
        return

    pr_df = pd.read_parquet(input_path)
    
    # Engineer features
    feat_df = engineer_features(pr_df)
    
    # Save
    output_path = Path(output_dir) / "pr_features.parquet"
    print(f"Saving features to {output_path}...")
    feat_df.to_parquet(output_path)
    print("Done.")

if __name__ == "__main__":
    main()
