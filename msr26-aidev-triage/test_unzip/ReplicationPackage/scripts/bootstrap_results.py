
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import roc_auc_score, average_precision_score, precision_score
from pathlib import Path
import json

DATA_DIR = Path("data/processed")
OUT_DIR = Path("outputs/results")
OUT_DIR.mkdir(parents=True, exist_ok=True)

def bootstrap_metric(y_true, y_pred, metric_fn, n_boot=1000, rng=None):
    if rng is None: rng = np.random.RandomState(42)
    stats = []
    indices = np.arange(len(y_true))
    for _ in range(n_boot):
        boot_idx = rng.choice(indices, size=len(indices), replace=True)
        score = metric_fn(y_true[boot_idx], y_pred[boot_idx])
        stats.append(score)
    return np.mean(stats), np.percentile(stats, [2.5, 97.5])

def precision_at_k(y_true, y_pred, k_fraction=0.2):
    # k_fraction = 20% budget
    k = int(len(y_true) * k_fraction)
    # Sort by pred desc
    sorted_indices = np.argsort(y_pred)[::-1]
    top_k_indices = sorted_indices[:k]
    # Precision: Fraction of top k that are positive
    return np.mean(y_true[top_k_indices])

def main():
    print("Loading Snapshot+Priors Features...")
    df = pd.read_parquet(DATA_DIR / "features_snapshot_with_priors.parquet")
    
    # Target
    target = 'is_high_cost'
    if target not in df.columns:
        # Re-calc similar to before if needed, or assume it's there
        # Valid features has it? build_snapshot saves it.
        # But engineered_priors loaded snapshot which has it.
        pass

    # Split
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(splitter.split(df, df[target], groups=df['repo_full_name']))
    
    train = df.iloc[train_idx]
    test = df.iloc[test_idx]
    
    # Train Model (Full T0 Features)
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from src.features import get_feature_columns
    
    # Get T0 features (strict=True for creation-time only)
    features = get_feature_columns(strict=True)
    
    # Filter to available columns
    features = [f for f in features if f in df.columns]
    
    print(f"Using {len(features)} T0 features: {features}")
    
    X_train = train[features]
    y_train = train[target]
    X_test = test[features]
    y_test = test[target]
    
    print(f"Training Features ({len(X_train.columns)}):", X_train.columns.tolist())
    
    print(f"Training LightGBM on {len(X_train)} samples...")
    model = lgb.LGBMClassifier(random_state=42, verbose=-1)
    model.fit(X_train, y_train)
    
    # Predict
    y_pred = model.predict_proba(X_test)[:, 1]
    
    # Bootsrap
    n_boot = 100 
    print(f"Bootstrapping {n_boot} times...")
    
    metrics = {
        'AUC': roc_auc_score,
        'AUPRC': average_precision_score,
        'Prec@20': lambda yt, yp: precision_at_k(yt, yp, 0.2)
    }
    
    results = {}
    rng = np.random.RandomState(42)
    
    for name, func in metrics.items():
        mean, (low, high) = bootstrap_metric(y_test.values, y_pred, func, n_boot, rng)
        results[name] = {'mean': mean, 'ci_low': low, 'ci_high': high}
        print(f"{name}: {mean:.3f} [{low:.3f}, {high:.3f}]")
        
    # Slice Analysis
    print("\nSlice Analysis (AUC):")
    slices = {}
    
    # Agent Slice
    agent_col = 'agent' if 'agent' in test.columns else 'agent_id'
    if agent_col in test.columns:
        agents = test[agent_col].unique()
        for ag in agents:
            mask = test[agent_col] == ag
            if mask.sum() > 50:
                sliced_y = y_test[mask]
                sliced_pred = y_pred[mask]
                try:
                    auc = roc_auc_score(sliced_y, sliced_pred)
                    slices[f"Agent_{ag}"] = auc
                    print(f"  {ag}: {auc:.3f} (n={mask.sum()})")
                except:
                    pass
    else:
        print("Warning: Agent column not found for slicing.")
                
    # Save
    with open(OUT_DIR / "bootstrap_results.json", "w") as f:
        json.dump({'overall': results, 'slices': slices}, f, indent=2)

if __name__ == "__main__":
    main()
