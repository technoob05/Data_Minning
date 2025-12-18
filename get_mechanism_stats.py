
import pandas as pd

def get_mechanism_stats():
    print("Loading features...")
    df = pd.read_parquet("msr26-aidev-triage/artifacts/pr_features.parquet")
    
    # Filter for Ghosting Analysis (Rejected + Feedback)
    pool = df[(df["status"]=="rejected") & (df["first_human_feedback_at"].notna())].copy()
    
    print(f"Ghosting Pool Size: {len(pool)}")
    
    # Docs Only: touches_docs=1, touches_tests=0
    docs_only = pool[(pool["touches_docs"]==1) & (pool["touches_tests"]==0)]
    docs_ghost = docs_only["is_ghosted"].mean() * 100
    print(f"Docs Only Ghosting Rate: {docs_ghost:.1f}% (n={len(docs_only)})")
    
    # Tests + Docs: touches_docs=1, touches_tests=1
    both = pool[(pool["touches_docs"]==1) & (pool["touches_tests"]==1)]
    both_ghost = both["is_ghosted"].mean() * 100
    print(f"Tests + Docs Ghosting Rate: {both_ghost:.1f}% (n={len(both)})")
    
    # Tests Only: touches_docs=0, touches_tests=1
    tests_only = pool[(pool["touches_docs"]==0) & (pool["touches_tests"]==1)]
    tests_ghost = tests_only["is_ghosted"].mean() * 100
    print(f"Tests Only Ghosting Rate: {tests_ghost:.1f}% (n={len(tests_only)})")

if __name__ == "__main__":
    get_mechanism_stats()
