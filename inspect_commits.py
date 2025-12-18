
import pandas as pd
import os

def check_commits_schema():
    path = "msr26-aidev-triage/data/pr_commits.parquet"
    if os.path.exists(path):
        try:
            df = pd.read_parquet(path)
            print(f"pr_commits columns: {df.columns.tolist()}")
            if "committed_date" in df.columns:
                print("Found 'committed_date'")
            elif "committer" in df.columns:
                print("Found 'committer' struct (might need extraction)")
        except Exception as e:
            print(f"Error reading pr_commits: {e}")
    else:
        print("pr_commits.parquet not found in data/")
        
    path2 = "msr26-aidev-triage/data/pr_commit_details.parquet"
    if os.path.exists(path2):
        try:
            df = pd.read_parquet(path2)
            print(f"pr_commit_details columns: {df.columns.tolist()}")
        except Exception as e:
            print(f"Error reading pr_commit_details: {e}")

if __name__ == "__main__":
    check_commits_schema()
