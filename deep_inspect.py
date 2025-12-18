
import pandas as pd
import numpy as np

def deep_inspect():
    print("Inspecting pr_commits.parquet...")
    try:
        df = pd.read_parquet("msr26-aidev-triage/data/pr_commits.parquet")
        print("Columns:", df.columns.tolist())
        print("Dtypes:")
        print(df.dtypes)
        print("\nFirst row as dict:")
        print(df.iloc[0].to_dict())
        
        sample = df["committer"].iloc[0]
        print(f"\nCommitter Type: {type(sample)}")
        print(f"Committer Value: {sample}")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    deep_inspect()
