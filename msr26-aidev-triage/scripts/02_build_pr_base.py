import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.load import load_raw_data
from src.build_tables import build_pr_base_table

def main():
    data_dir = "msr26-aidev-triage/data"
    output_dir = "msr26-aidev-triage/artifacts"
    
    # Load raw data
    dfs = load_raw_data(data_dir)
    
    # Build base table
    pr_base_df = build_pr_base_table(dfs)
    
    # Save
    output_path = Path(output_dir) / "pr_base.parquet"
    print(f"Saving PR base table to {output_path}...")
    pr_base_df.to_parquet(output_path)
    print("Done.")

if __name__ == "__main__":
    main()
