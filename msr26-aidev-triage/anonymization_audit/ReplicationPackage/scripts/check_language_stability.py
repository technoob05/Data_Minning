
import sys
from pathlib import Path
import pandas as pd
from sklearn.metrics import roc_auc_score
import logging

# Add Project Root
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.config import FEATURES_SNAPSHOT
from src.utils import load_data, setup_logger

logger = setup_logger("check_lang_stability")

def check_language_stability():
    logger.info("Checking Model Stability across Languages...")
    
    try:
        df = load_data(FEATURES_SNAPSHOT)
    except FileNotFoundError:
        logger.error(f"File not found: {FEATURES_SNAPSHOT}")
        return

    target = 'is_high_cost'
    pred_col = 'effort_score' # Using effort score as proxy or we can train a model. 
    # Actually, to be accurate, we should use the model's predictions. 
    # But for a quick check, checking the predictive power of 'total_changes' or 'additions' (our main signal) 
    # across languages is a strong proxy for "Model Stability" since the model is size-dominated.
    # Alternatively, if we saved predictions, we could use them. We likely didn't save per-row preds.
    # Let's train a quick lightgbm model to get fresh predictions.
    
    from sklearn.model_selection import train_test_split
    import lightgbm as lgb
    
    # Quick Train
    features = [
        "created_hour", "created_dayofweek", "is_weekend",
        "title_len", "body_len",
        "has_plan",
        "additions", "deletions", "changed_files", "total_changes",
        "touches_tests", "touches_ci", "touches_deps"
    ]
    
    # Filter valid
    valid_cols = [c for c in features + [target, 'lang_extension'] if c in df.columns]
    df = df[valid_cols].dropna()
    
    X = df[features]
    y = df[target]
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = lgb.LGBMClassifier(n_estimators=100, max_depth=6, class_weight='balanced', verbose=-1)
    model.fit(X_train, y_train)
    
    # Predict on ALL data (for slicing) or just test? 
    # Ideally test set slicing.
    
    # Get test indices to map back to languages
    test_indices = X_test.index
    test_langs = df.loc[test_indices, 'lang_extension']
    preds = model.predict_proba(X_test)[:, 1]
    
    # Create Analysis DF
    res_df = pd.DataFrame({'lang': test_langs, 'y_true': y_test, 'y_pred': preds})
    
    # Filter top languages
    top_langs = res_df['lang'].value_counts().head(8).index.tolist()
    
    logger.info(f"{'Language':<15} | {'N':<6} | {'AUC':<6}")
    logger.info("-" * 35)
    
    results = {}
    for lang in top_langs:
        subset = res_df[res_df['lang'] == lang]
        if len(subset) < 50:
            continue
            
        try:
            auc = roc_auc_score(subset['y_true'], subset['y_pred'])
            logger.info(f"{lang:<15} | {len(subset):<6} | {auc:.3f}")
            results[lang] = auc
        except:
            pass

if __name__ == "__main__":
    check_language_stability()
