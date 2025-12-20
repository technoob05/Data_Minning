import pandas as pd
from pathlib import Path

def load_raw_data(data_dir: str = "data"):
    """
    Loads the raw parquet files from the data directory.
    Returns a dictionary of DataFrames.
    """
    data_path = Path(data_dir)
    
    # Define file paths
    files = {
        "pr": data_path / "pull_request.parquet",
        "repo": data_path / "repository.parquet",
        "user": data_path / "user.parquet",
        "pr_comments": data_path / "pr_comments.parquet",
        "pr_reviews": data_path / "pr_reviews.parquet",
        "pr_review_comments": data_path / "pr_review_comments_v2.parquet",
        "pr_commits": data_path / "pr_commits.parquet",
        "pr_commit_details": data_path / "pr_commit_details.parquet",
        "related_issue": data_path / "related_issue.parquet",
        "issue": data_path / "issue.parquet",
        "pr_timeline": data_path / "pr_timeline.parquet",
        "pr_task_type": data_path / "pr_task_type.parquet",
    }
    
    dfs = {}
    for name, path in files.items():
        if path.exists():
            print(f"Loading {name} from {path}...")
            try:
                dfs[name] = pd.read_parquet(path)
            except Exception as e:
                print(f"Error loading {name} with default engine: {e}")
                print("Trying with engine='fastparquet'...")
                try:
                    dfs[name] = pd.read_parquet(path, engine="fastparquet")
                except Exception as e2:
                    print(f"Error loading {name} with fastparquet: {e2}")
                    dfs[name] = None
        else:
            print(f"Warning: {path} does not exist. Skipping.")
            dfs[name] = None
            
    return dfs

def preprocess_dates(df: pd.DataFrame, date_cols: list = None):
    """
    Converts specified columns to datetime objects.
    """
    if date_cols is None:
        date_cols = ["created_at", "closed_at", "merged_at", "updated_at"]
        
    for col in date_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], utc=True, errors="coerce")
    return df
