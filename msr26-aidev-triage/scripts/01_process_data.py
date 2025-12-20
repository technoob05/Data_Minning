import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parents[1]))

# Since we are in a restricted environment where we might not be able to download large files easily 
# or the user might already have them, this script is a placeholder or a checker.
# However, the user provided context implies the data might be available or needs to be downloaded.
# The user's prompt mentioned "AI_Teammates_in_SE3" folder which seems to contain analysis scripts but maybe not the raw parquet files.
# The `load_AIDev.ipynb` suggests loading from "hf://datasets/hao-li/AIDev/...".
# We can try to use pandas to read directly from HF if internet is available, or assume files are in `data/`.

# For this script, let's assume we want to download them to `data/` if they don't exist.
# Using pandas read_parquet with hf:// url works if libraries are installed.

import pandas as pd
import os

from src.config import DATA_DIR

# DATA_DIR is already defined in config as absolute path
DATA_DIR.mkdir(parents=True, exist_ok=True)

FILES = {
    "pull_request.parquet": "hf://datasets/hao-li/AIDev/pull_request.parquet",
    "repository.parquet": "hf://datasets/hao-li/AIDev/repository.parquet",
    "user.parquet": "hf://datasets/hao-li/AIDev/user.parquet",
    "pr_comments.parquet": "hf://datasets/hao-li/AIDev/pr_comments.parquet",
    "pr_reviews.parquet": "hf://datasets/hao-li/AIDev/pr_reviews.parquet",
    "pr_review_comments_v2.parquet": "hf://datasets/hao-li/AIDev/pr_review_comments_v2.parquet",
    "pr_commits.parquet": "hf://datasets/hao-li/AIDev/pr_commits.parquet",
    "pr_commit_details.parquet": "hf://datasets/hao-li/AIDev/pr_commit_details.parquet",
    "related_issue.parquet": "hf://datasets/hao-li/AIDev/related_issue.parquet",
    "issue.parquet": "hf://datasets/hao-li/AIDev/issue.parquet",
    "pr_timeline.parquet": "hf://datasets/hao-li/AIDev/pr_timeline.parquet",
    "pr_task_type.parquet": "hf://datasets/hao-li/AIDev/pr_task_type.parquet",
}

def download_data():
    print("Checking/Downloading data...")
    for filename, url in FILES.items():
        filepath = DATA_DIR / filename
        if not filepath.exists():
            print(f"Downloading {filename}...")
            try:
                df = pd.read_parquet(url)
                df.to_parquet(filepath)
                print(f"Saved {filename}")
            except Exception as e:
                print(f"Failed to download {filename}: {e}")
        else:
            print(f"{filename} already exists.")

if __name__ == "__main__":
    download_data()
