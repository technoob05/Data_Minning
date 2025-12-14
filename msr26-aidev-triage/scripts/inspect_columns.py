import pandas as pd
from pathlib import Path

data_path = Path("msr26-aidev-triage/data/pr_commit_details.parquet")
if data_path.exists():
    df = pd.read_parquet(data_path)
    print("Columns in pr_commit_details.parquet:")
    print(df.columns.tolist())
    print("\nFirst few rows:")
    print(df.head())
else:
    print(f"{data_path} does not exist.")
