import sys
from pathlib import Path
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import roc_auc_score
import warnings

# Add Project Root to Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.config import FEATURES_SNAPSHOT, FEATURES_FULL, OUTPUTS_DIR, RANDOM_SEED, ARTIFACTS_DIR
from src.utils import setup_logger, load_data
from src.features import get_feature_columns

logger = setup_logger("03_train_models")
warnings.filterwarnings('ignore')

def prepare_data(df: pd.DataFrame, features: list, target_col: str = 'is_high_cost'):
    """Prepare train/test split with repo-disjoint protocol."""
    # Features + Target
    cols = features + [target_col]
    if 'repo_full_name' in df.columns:
         cols += ['repo_full_name']
    
    # Filter valid
    valid_cols = [c for c in cols if c in df.columns]
    X = df[valid_cols].copy()
    y = df[target_col]
    
    # Repo-disjoint split
    repo_col = 'repo_full_name'
    if repo_col in df.columns:
        splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_SEED)
        train_idx, test_idx = next(splitter.split(X, y, groups=df[repo_col]))
        return X.iloc[train_idx], X.iloc[test_idx], features
    else:
        logger.warning("No repo column. Using random split.")
        from sklearn.model_selection import train_test_split
        return train_test_split(X, test_size=0.2, random_state=RANDOM_SEED) + (features,)

def train(train_df, test_df, features, target_col='is_high_cost'):
    """Train LightGBM."""
    X_train = train_df[features]
    y_train = train_df[target_col]
    X_test = test_df[features]
    y_test = test_df[target_col]
    
    # Encode categoricals safely
    for col in X_train.select_dtypes(include=['object']).columns:
        X_train[col] = X_train[col].astype('category')
        X_test[col] = X_test[col].astype('category')
        
    model = lgb.LGBMClassifier(
        n_estimators=100, max_depth=6, learning_rate=0.1,
        class_weight='balanced', random_state=RANDOM_SEED, verbose=-1
    )
    
    model.fit(X_train, y_train)
    preds = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, preds)
    return auc, model

def main():
    logger.info("Step 3: Train Models (Snapshot vs Full)")
    
    # Load Data
    try:
        df_snap = load_data(FEATURES_SNAPSHOT)
        df_full = load_data(FEATURES_FULL)
    except FileNotFoundError:
        logger.error("Feature files not found. Run '02_engineer_features.py' first.")
        return

    logger.info(f"Loaded Snapshot: {len(df_snap)} rows, Full: {len(df_full)} rows")

    # Features
    feats_snap = get_feature_columns(strict=True)
    feats_full = get_feature_columns(strict=False)
    
    # Align features with dataframe columns
    feats_snap = [f for f in feats_snap if f in df_snap.columns]
    feats_full = [f for f in feats_full if f in df_full.columns]

    # Train Snapshot
    logger.info("[Snapshot Model]")
    train_s, test_s, fs = prepare_data(df_snap, feats_snap)
    auc_s, _ = train(train_s, test_s, fs)
    logger.info(f"Snapshot AUC: {auc_s:.3f}")
    
    # Train Full
    logger.info("[Full Model]")
    train_f, test_f, ff = prepare_data(df_full, feats_full)
    auc_f, model_f = train(train_f, test_f, ff)
    logger.info(f"Full AUC: {auc_f:.3f}")
    
    # Results
    diff = auc_f - auc_s
    logger.info(f"Difference: {diff:+.3f}")
    
    # Save model and results
    import joblib
    model_path = ARTIFACTS_DIR / "triage_model_high_cost.pkl"
    joblib.dump(model_f, model_path)
    logger.info(f"Saved Full model to {model_path}")

    out_file = OUTPUTS_DIR / "leakage_check_results.csv"
    pd.DataFrame({
        "Model": ["Snapshot", "Full"],
        "AUC": [auc_s, auc_f],
        "Features": [len(fs), len(ff)]
    }).to_csv(out_file, index=False)
    logger.info(f"Saved results to {out_file}")

if __name__ == "__main__":
    main()
