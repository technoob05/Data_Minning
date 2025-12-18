
import pandas as pd

def inspect_timeline():
    print("Inspecting pr_timeline.parquet...")
    try:
        df = pd.read_parquet("msr26-aidev-triage/data/pr_timeline.parquet")
        print("Columns:", df.columns.tolist())
        print("Event types:", df["event"].unique())
        
        commits = df[df["event"] == "committed"]
        print(f"Total 'committed' events: {len(commits)}")
        if not commits.empty:
            print("First committed event:", commits.iloc[0].to_dict())
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    inspect_timeline()
