
import pandas as pd

def inspect_committer():
    try:
        df = pd.read_parquet("msr26-aidev-triage/data/pr_commits.parquet")
        print("First 5 committer entries:")
        print(df["committer"].head().to_list())
        
        # Check if it's a dict or struct and try to access keys
        sample = df["committer"].iloc[0]
        if isinstance(sample, dict):
            print(f"Keys: {sample.keys()}")
        elif hasattr(sample, "as_py"): # Arrow scalar
             print(f"Arrow keys: {sample.as_py().keys()}")
             
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    inspect_committer()
