#!/usr/bin/env python3
"""
Confounder Analysis: Ghosting vs CI Touches
Controls for PR size and Agent ID to validate the claim that
"CI feedback reduces ghosting" is not just a proxy for "Small PRs reduce ghosting".
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from pathlib import Path

DATA_DIR = Path("data/processed")
OUTPUT_DIR = Path("outputs/stats")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def load_data():
    path = DATA_DIR / "valid_features.csv"
    if not path.exists():
        # Fallback to creating frame from raw if valid_features doesn't exist
        # For now assume it exists as per previous context
        raise FileNotFoundError(f"{path} not found. Please run feature generation first.")
    return pd.read_csv(path)

def analyze_confounders(df):
    print("="*60)
    print("CONFOUNDER ANALYSIS: GHOSTING MECHANISM")
    print("="*60)
    
    # 1. Filter to Rejected PRs (Ghosting is only defined for rejected/open subset usually, 
    # but paper says "64.5% ghosting rate in rejected PRs").
    # We should look at relevant population. 
    # Let's use the whole dataset or just those that received feedback?
    # Paper "Ghosting" definition: Rejected + Feedback + No Followup.
    # So we probably want to model probability of ghosting GIVEN it was rejected?
    # Or just use the 'is_ghosted' flag on the whole set?
    # The 'is_ghosted' flag is 0 for Merged.
    # Let's perform analysis on the subset where Ghosting is possible (Rejected PRs).
    
    if 'is_ghost' not in df.columns:
        print("Error: 'is_ghost' column missing.")
        return

    # Filter to Rejected PRs for mechanism analysis
    # (Merged PRs by definition aren't ghosted in this context)
    analysis_df = df[df['state'] == 'closed'].copy() # Assuming 'state' or 'status'
    # Actually, let's look at the implementation of is_ghosted in features.py
    # It seems to be 1 or 0.
    
    # Let's try to replicate the paper's claim on the broadest meaningful set.
    # "PRs touching CI files have a lower ghosting rate".
    
    # Features
    # Log transform size features to handle skew
    analysis_df['log_changes'] = np.log1p(analysis_df['additions'] + analysis_df['deletions'])
    
    # Formula
    # is_ghost ~ touches_ci + log_changes + agent
    
    # We need to make sure 'agent' is categorical
    if 'author_agent' in analysis_df.columns:
         analysis_df['agent'] = analysis_df['author_agent'].astype(str)
    else:
         analysis_df['agent'] = analysis_df['agent'].astype(str)
    
    print(f"N = {len(analysis_df)} (Closed PRs)")
    
    model = smf.logit("is_ghost ~ touches_ci + log_changes + C(agent)", data=analysis_df)
    result = model.fit()
    
    print(result.summary())
    
    # Extract Odds Ratios
    params = result.params
    conf = result.conf_int()
    conf['OR'] = params
    conf.columns = ['2.5%', '97.5%', 'OR']
    odds_ratios = np.exp(conf)
    
    print("\nODDS RATIOS (95% CI):")
    print(odds_ratios.loc[['touches_ci', 'log_changes']])
    
    ci_or = odds_ratios.loc['touches_ci', 'OR']
    ci_lower = odds_ratios.loc['touches_ci', '2.5%']
    ci_upper = odds_ratios.loc['touches_ci', '97.5%']
    
    print(f"\nEffect of CI Touches: OR = {ci_or:.2f} [{ci_lower:.2f}, {ci_upper:.2f}]")
    if ci_upper < 1.0:
        print("CONCLUSION: Touching CI significantly REDUCES ghosting risk, independent of size/agent.")
    else:
        print("CONCLUSION: Result is not significant after control.")
        
    return result

if __name__ == "__main__":
    try:
        df = load_data()
        analyze_confounders(df)
    except Exception as e:
        print(f"Analysis failed: {e}")
