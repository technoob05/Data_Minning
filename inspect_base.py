
import pandas as pd
import os

try:
    df = pd.read_parquet("msr26-aidev-triage/artifacts/pr_base.parquet")
    print("Columns:", df.columns.tolist())
    if "is_instant_merge" in df.columns:
        print("is_instant_merge exists")
    else:
        print("is_instant_merge MISSING")
        
    # Check for timestamps if we need to calculate it
    print("Time columns:", [c for c in df.columns if "time" in c.lower() or "at" in c.lower()])
except Exception as e:
    print(e)
