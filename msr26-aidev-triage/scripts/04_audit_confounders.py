import sys
from pathlib import Path
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf

# Add Project Root to Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.config import FEATURES_FULL, FEATURES_SNAPSHOT
from src.utils import setup_logger, load_data

logger = setup_logger("04_audit_confounders")

def main():
    logger.info("Step 4: Audit Confounders (Ghosting Analysis)")
    
    # Load Data (Full features has the interactions)
    # If full feature doesn't exist, try snapshot (but snapshot lacks touches_ci)
    path = FEATURES_FULL if FEATURES_FULL.exists() else FEATURES_SNAPSHOT
    if not path.exists():
        logger.error("No feature data found.")
        return
        
    df = load_data(path)
    logger.info(f"Loaded {len(df)} rows from {path.name}")
    
    # Check required columns
    req_cols = ['is_ghosted', 'touches_ci', 'additions', 'agent', 'repo_full_name']
    # Note: 'author_agent' or 'agent'? 'agent_encoded'?
    # 'features.py' creates 'agent_encoded'. Let's check raw columns if available or encoded.
    
    # Fix column names if needed based on what engineer_features outputted
    # In engineer_features, we passed through 'agent_encoded' but not necessarily raw 'agent' unless in Full columns.
    # df_full = df.copy() which implies input df columns are preserved.
    # If input was valid_features.csv, it has 'author_agent'.
    
    if 'author_agent' in df.columns:
        df['agent'] = df['author_agent']
    
    missing = [c for c in req_cols if c not in df.columns]
    if missing:
        logger.warning(f"Missing columns for audit: {missing}. Attempting to proceed with available...")
    
    # Prepare Regression Data
    # Target: is_ghosted
    if 'is_ghosted' not in df.columns and 'is_ghost' in df.columns:
        df['is_ghosted'] = df['is_ghost']
        
    if 'touches_ci' not in df.columns:
        logger.error("Crucial column 'touches_ci' missing. Cannot run confounder check.")
        return

    # Transformations
    df['log_additions'] = np.log1p(df['additions'].fillna(0))
    
    # Top K Categories
    top_agents = df['agent'].value_counts().nlargest(4).index
    df['agent_cat'] = df['agent'].apply(lambda x: x if x in top_agents else 'Other')
    
    top_repos = df['repo_full_name'].value_counts().nlargest(9).index
    df['repo_cat'] = df['repo_full_name'].apply(lambda x: x if x in top_repos else 'Other')
    
    # Formula
    formula = "is_ghosted ~ touches_ci + log_additions + C(agent_cat) + C(repo_cat)"
    logger.info(f"Formula: {formula}")
    
    try:
        model = smf.logit(formula=formula, data=df).fit(disp=0)
        logger.info("\n" + str(model.summary()))
        
        # Extract OR
        params = model.params
        ci_coeff = params.get("touches_ci", 0)
        or_val = np.exp(ci_coeff)
        
        logger.info(f"Odds Ratio for touches_ci: {or_val:.4f}")
        if or_val < 1:
            logger.info("Result: CI touches associated with REDUCED ghosting.")
        else:
            logger.info("Result: CI touches associated with INCREASED/NEUTRAL ghosting.")
            
    except Exception as e:
        logger.error(f"Regression failed: {e}")

if __name__ == "__main__":
    main()
