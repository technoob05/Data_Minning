#!/usr/bin/env python3
"""
SOTA Model Benchmark for PR Effort Prediction
Tests comprehensive set of state-of-the-art models to maximize modeling novelty.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc as pr_auc_calc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import sys
import warnings
import time
warnings.filterwarnings('ignore')

sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.config import ARTIFACTS_DIR, TABLES_DIR

# ============================================================================
# MODEL IMPORTS
# ============================================================================

# Gradient Boosting Models
import lightgbm as lgb
try:
    import xgboost as xgb
    HAS_XGB = True
except:
    HAS_XGB = False
    print("XGBoost not available")

try:
    import catboost as cb
    HAS_CATBOOST = True
except:
    HAS_CATBOOST = False
    print("CatBoost not available")

# Scikit-learn models
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    VotingClassifier,
    StackingClassifier
)
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression

# Deep Learning (if available)
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
    HAS_TORCH = True
except:
    HAS_TORCH = False
    print("PyTorch not available")

# TabNet (if available)
try:
    from pytorch_tabnet.tab_model import TabNetClassifier
    HAS_TABNET = True
except:
    HAS_TABNET = False
    print("TabNet not available")

# AutoML (if available)
try:
    from autogluon.tabular import TabularPredictor
    HAS_AUTOGLUON = True
except:
    HAS_AUTOGLUON = False
    print("AutoGluon not available")

# TabPFN (few-shot SOTA)
try:
    from tabpfn import TabPFNClassifier
    HAS_TABPFN = True
except:
    HAS_TABPFN = False
    print("TabPFN not available")

# ============================================================================
# DEEP LEARNING MODELS
# ============================================================================

# PyTorch models (only define if available)
if HAS_TORCH:
    class DeepTabularNet(nn.Module):
        """Deep MLP with BatchNorm and Dropout for tabular data"""
        def __init__(self, input_dim, hidden_dims=[256, 128, 64, 32], dropout=0.3):
            super().__init__()
            layers = []
            prev_dim = input_dim
            
            for hidden_dim in hidden_dims:
                layers.extend([
                    nn.Linear(prev_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout)
                ])
                prev_dim = hidden_dim
            
            layers.append(nn.Linear(prev_dim, 2))
            self.network = nn.Sequential(*layers)
        
        def forward(self, x):
            return self.network(x)

    class ResNetTabular(nn.Module):
        """ResNet-style architecture for tabular data"""
        def __init__(self, input_dim, hidden_dim=128, num_blocks=3, dropout=0.3):
            super().__init__()
            self.input_proj = nn.Linear(input_dim, hidden_dim)
            
            self.blocks = nn.ModuleList([
                ResBlock(hidden_dim, dropout) for _ in range(num_blocks)
            ])
            
            self.output = nn.Sequential(
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 2)
            )
        
        def forward(self, x):
            x = self.input_proj(x)
            for block in self.blocks:
                x = block(x)
            return self.output(x)

    class ResBlock(nn.Module):
        def __init__(self, dim, dropout=0.3):
            super().__init__()
            self.block = nn.Sequential(
                nn.BatchNorm1d(dim),
                nn.ReLU(),
                nn.Linear(dim, dim),
                nn.Dropout(dropout),
                nn.BatchNorm1d(dim),
                nn.ReLU(),
                nn.Linear(dim, dim),
                nn.Dropout(dropout)
            )
        
        def forward(self, x):
            return x + self.block(x)

# ============================================================================
# BENCHMARK FUNCTIONS
# ============================================================================

def prepare_data():
    """Load and prepare data with EXACT paper methodology"""
    print("Loading data...")
    
    # Load from FEATURES_SNAPSHOT (pre-engineered features)
    from src.config import FEATURES_SNAPSHOT
    from src.features import get_feature_columns
    
    feat_path = FEATURES_SNAPSHOT
    if not feat_path.exists():
        print("WARNING: FEATURES_SNAPSHOT not found, falling back to pr_features.parquet")
        feat_path = ARTIFACTS_DIR / "pr_features.parquet"
    
    df = pd.read_parquet(feat_path)
    print(f"Loaded {len(df)} PRs")
    
    # Get T0 features (strict=True for creation-time only)
    features = get_feature_columns(strict=True)
    
    # Filter to available features (some may be missing)
    features = [f for f in features if f in df.columns]
    print(f"Using {len(features)} T0 features: {features[:5]}... (+{len(features)-5} more)")
    
    # Ensure target exists
    if 'is_high_cost' not in df.columns:
        if 'effort_score' not in df.columns:
            df['effort_score'] = df.get('num_comments', 0).fillna(0) + df.get('num_reviews', 0).fillna(0)
        threshold = df['effort_score'].quantile(0.8)
        df['is_high_cost'] = (df['effort_score'] > threshold).astype(int)
    
    # Prepare X, y
    X = df[features + ['repo_full_name']].copy() if 'repo_full_name' in df.columns else df[features].copy()
    y = df['is_high_cost'].values
    
    # Handle missing values and encode categoricals
    from sklearn.preprocessing import LabelEncoder
    
    for col in features:
        # Fill missing first
        X[col] = X[col].fillna(0 if X[col].dtype in ['float64', 'int64'] else 'unknown')
        
        # Encode categoricals (object/string dtype)
        if X[col].dtype == 'object' or col in ['lang_extension']:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
        # For numeric, ensure no NaN
        elif X[col].dtype in ['float64', 'int64']:
            X[col] = X[col].fillna(0)
    
    # REPO-DISJOINT SPLIT (matching paper exactly)
    if 'repo_full_name' in X.columns:
        print("Using repo-disjoint split (GroupShuffleSplit)")
        from sklearn.model_selection import GroupShuffleSplit
        
        splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        train_idx, test_idx = next(splitter.split(X, y, groups=df['repo_full_name']))
        
        X_train = X.iloc[train_idx][features]
        X_test = X.iloc[test_idx][features]
        y_train = y[train_idx]
        y_test = y[test_idx]
    else:
        print("WARNING: repo_full_name not found, using random split")
        X = X[features]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
    
    print(f"Train: {len(X_train)}, Test: {len(X_test)}")
    print(f"Positive rate: {y_train.mean():.3f}")
    
    # Return feature names for model training
    return X_train, X_test, y_train, y_test, features

def train_pytorch_model(model, X_train, y_train, X_test, epochs=50, batch_size=256, lr=0.001):
    """Train PyTorch model - only called if HAS_TORCH is True"""
    if not HAS_TORCH:
        return None
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Convert to tensors
    X_train_t = torch.FloatTensor(X_train.values).to(device)
    y_train_t = torch.LongTensor(y_train).to(device)
    X_test_t = torch.FloatTensor(X_test.values).to(device)
    
    # DataLoader
    train_dataset = torch.utils.data.TensorDataset(X_train_t, y_train_t)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # Training setup
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    # Train
    model.train()
    for epoch in range(epochs):
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
        scheduler.step()
    
    # Predict
    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test_t)
        y_pred_proba = torch.softmax(test_outputs, dim=1)[:, 1].cpu().numpy()
    
    return y_pred_proba

def benchmark_all_models(X_train, X_test, y_train, y_test, features):
    """Benchmark all available SOTA models"""
    results = []
    
    print(f"\nFeature count: {len(features)}")
    print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
    
    # Standardize for neural networks
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=features)
    X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=features)
    
    # ========================================================================
    # 1. BASELINE: LightGBM (current)
    # ========================================================================
    print("\n[1/N] LightGBM (Baseline)")
    start = time.time()
    lgbm = lgb.LGBMClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        class_weight='balanced',
        random_state=42,
        verbose=-1
    )
    lgbm.fit(X_train, y_train)
    y_pred = lgbm.predict_proba(X_test)[:, 1]
    
    auc_score = roc_auc_score(y_test, y_pred)
    precision, recall, _ = precision_recall_curve(y_test, y_pred)
    pr_auc_score = pr_auc_calc(recall, precision)
    
    results.append({
        'model': 'LightGBM (Baseline)',
        'AUC': auc_score,
        'PR-AUC': pr_auc_score,
        'time_sec': time.time() - start
    })
    print(f"  AUC: {auc_score:.4f}, PR-AUC: {pr_auc_score:.4f}")
    
    # ========================================================================
    # 2. XGBoost
    # ========================================================================
    if HAS_XGB:
        print("\n[2/N] XGBoost")
        start = time.time()
        xgb_model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            scale_pos_weight=len(y_train[y_train==0]) / len(y_train[y_train==1]),
            random_state=42,
            eval_metric='logloss'
        )
        xgb_model.fit(X_train, y_train)
        y_pred = xgb_model.predict_proba(X_test)[:, 1]
        
        auc_score = roc_auc_score(y_test, y_pred)
        precision, recall, _ = precision_recall_curve(y_test, y_pred)
        pr_auc_score = pr_auc_calc(recall, precision)
        
        results.append({
            'model': 'XGBoost',
            'AUC': auc_score,
            'PR-AUC': pr_auc_score,
            'time_sec': time.time() - start
        })
        print(f"  AUC: {auc_score:.4f}, PR-AUC: {pr_auc_score:.4f}")
    
    # ========================================================================
    # 3. CatBoost
    # ========================================================================
    if HAS_CATBOOST:
        print("\n[3/N] CatBoost")
        start = time.time()
        cat_model = cb.CatBoostClassifier(
            iterations=100,
            depth=6,
            learning_rate=0.1,
            auto_class_weights='Balanced',
            random_state=42,
            verbose=0
        )
        cat_model.fit(X_train, y_train)
        y_pred = cat_model.predict_proba(X_test)[:, 1]
        
        auc_score = roc_auc_score(y_test, y_pred)
        precision, recall, _ = precision_recall_curve(y_test, y_pred)
        pr_auc_score = pr_auc_calc(recall, precision)
        
        results.append({
            'model': 'CatBoost',
            'AUC': auc_score,
            'PR-AUC': pr_auc_score,
            'time_sec': time.time() - start
        })
        print(f"  AUC: {auc_score:.4f}, PR-AUC: {pr_auc_score:.4f}")
    
    # ========================================================================
    # 4. HistGradientBoosting (sklearn, very fast)
    # ========================================================================
    print("\n[4/N] HistGradientBoosting")
    start = time.time()
    hist_gb = HistGradientBoostingClassifier(
        max_iter=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=42
    )
    hist_gb.fit(X_train, y_train)
    y_pred = hist_gb.predict_proba(X_test)[:, 1]
    
    auc_score = roc_auc_score(y_test, y_pred)
    precision, recall, _ = precision_recall_curve(y_test, y_pred)
    pr_auc_score = pr_auc_calc(recall, precision)
    
    results.append({
        'model': 'HistGradientBoosting',
        'AUC': auc_score,
        'PR-AUC': pr_auc_score,
        'time_sec': time.time() - start
    })
    print(f"  AUC: {auc_score:.4f}, PR-AUC: {pr_auc_score:.4f}")
    
    # ========================================================================
    # 5. Random Forest
    # ========================================================================
    print("\n[5/N] Random Forest")
    start = time.time()
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)
    y_pred = rf.predict_proba(X_test)[:, 1]
    
    auc_score = roc_auc_score(y_test, y_pred)
    precision, recall, _ = precision_recall_curve(y_test, y_pred)
    pr_auc_score = pr_auc_calc(recall, precision)
    
    results.append({
        'model': 'Random Forest',
        'AUC': auc_score,
        'PR-AUC': pr_auc_score,
        'time_sec': time.time() - start
    })
    print(f"  AUC: {auc_score:.4f}, PR-AUC: {pr_auc_score:.4f}")
    
    # ========================================================================
    # 6. Deep MLP
    # ========================================================================
    print("\n[6/N] Deep MLP (Neural Network)")
    start = time.time()
    mlp = MLPClassifier(
        hidden_layer_sizes=(256, 128, 64, 32),
        activation='relu',
        solver='adam',
        alpha=0.0001,
        batch_size=256,
        learning_rate='adaptive',
        learning_rate_init=0.001,
        max_iter=100,
        early_stopping=True,
        validation_fraction=0.1,
        random_state=42
    )
    mlp.fit(X_train_scaled, y_train)
    y_pred = mlp.predict_proba(X_test_scaled)[:, 1]
    
    auc_score = roc_auc_score(y_test, y_pred)
    precision, recall, _ = precision_recall_curve(y_test, y_pred)
    pr_auc_score = pr_auc_calc(recall, precision)
    
    results.append({
        'model': 'Deep MLP',
        'AUC': auc_score,
        'PR-AUC': pr_auc_score,
        'time_sec': time.time() - start
    })
    print(f"  AUC: {auc_score:.4f}, PR-AUC: {pr_auc_score:.4f}")
    
    # ========================================================================
    # 7. PyTorch Deep Tabular (if available)
    # ========================================================================
    if HAS_TORCH:
        print("\n[7/N] PyTorch Deep Tabular")
        start = time.time()
        model = DeepTabularNet(input_dim=X_train.shape[1])
        y_pred = train_pytorch_model(model, X_train_scaled_df, y_train, X_test_scaled_df, epochs=30)
        
        auc_score = roc_auc_score(y_test, y_pred)
        precision, recall, _ = precision_recall_curve(y_test, y_pred)
        pr_auc_score = pr_auc_calc(recall, precision)
        
        results.append({
            'model': 'PyTorch DeepTabular',
            'AUC': auc_score,
            'PR-AUC': pr_auc_score,
            'time_sec': time.time() - start
        })
        print(f"  AUC: {auc_score:.4f}, PR-AUC: {pr_auc_score:.4f}")
        
        # ========================================================================
        # 8. PyTorch ResNet Tabular
        # ========================================================================
        print("\n[8/N] PyTorch ResNet Tabular")
        start = time.time()
        model = ResNetTabular(input_dim=X_train.shape[1])
        y_pred = train_pytorch_model(model, X_train_scaled_df, y_train, X_test_scaled_df, epochs=30)
        
        auc_score = roc_auc_score(y_test, y_pred)
        precision, recall, _ = precision_recall_curve(y_test, y_pred)
        pr_auc_score = pr_auc_calc(recall, precision)
        
        results.append({
            'model': 'PyTorch ResNet',
            'AUC': auc_score,
            'PR-AUC': pr_auc_score,
            'time_sec': time.time() - start
        })
        print(f"  AUC: {auc_score:.4f}, PR-AUC: {pr_auc_score:.4f}")
    
    # ========================================================================
    # 9. TabNet (if available)
    # ========================================================================
    if HAS_TABNET:
        print("\n[9/N] TabNet")
        start = time.time()
        tabnet = TabNetClassifier(
            n_d=64, n_a=64,
            n_steps=5,
            gamma=1.5,
            n_independent=2,
            n_shared=2,
            seed=42,
            verbose=0
        )
        tabnet.fit(
            X_train.values, y_train,
            eval_set=[(X_test.values, y_test)],
            max_epochs=50,
            patience=10,
            batch_size=256,
            virtual_batch_size=128
        )
        y_pred = tabnet.predict_proba(X_test.values)[:, 1]
        
        auc_score = roc_auc_score(y_test, y_pred)
        precision, recall, _ = precision_recall_curve(y_test, y_pred)
        pr_auc_score = pr_auc_calc(recall, precision)
        
        results.append({
            'model': 'TabNet',
            'AUC': auc_score,
            'PR-AUC': pr_auc_score,
            'time_sec': time.time() - start
        })
        print(f"  AUC: {auc_score:.4f}, PR-AUC: {pr_auc_score:.4f}")
    
    # ========================================================================
    # 10. TabPFN (few-shot SOTA, if available)
    # ========================================================================
    if HAS_TABPFN:
        print("\n[10/N] TabPFN (Few-Shot SOTA)")
        # TabPFN has limit on training size (~10k)
        if len(X_train) > 10000:
            print("  Sampling 10k for TabPFN...")
            indices = np.random.choice(len(X_train), 10000, replace=False)
            X_train_pf= X_train.iloc[indices]
            y_train_pfn = y_train[indices]
        else:
            X_train_pfn = X_train
            y_train_pfn = y_train
        
        start = time.time()
        try:
            tabpfn = TabPFNClassifier(device='cpu', N_ensemble_configurations=8)
            tabpfn.fit(X_train_pfn.values, y_train_pfn)
            y_pred = tabpfn.predict_proba(X_test.values)[:, 1]
            
            auc_score = roc_auc_score(y_test, y_pred)
            precision, recall, _ = precision_recall_curve(y_test, y_pred)
            pr_auc_score = pr_auc_calc(recall, precision)
            
            results.append({
                'model': 'TabPFN (Few-Shot)',
                'AUC': auc_score,
                'PR-AUC': pr_auc_score,
                'time_sec': time.time() - start
            })
            print(f"  AUC: {auc_score:.4f}, PR-AUC: {pr_auc_score:.4f}")
        except Exception as e:
            print(f"  TabPFN failed: {e}")
    
    # ========================================================================
    # 11. Ensemble: Voting
    # ========================================================================
    print("\n[11/N] Voting Ensemble")
    start = time.time()
    estimators = [
        ('lgbm', lgb.LGBMClassifier(n_estimators=100, max_depth=6, random_state=42, verbose=-1)),
        ('rf', RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1))
    ]
    if HAS_XGB:
        estimators.append(('xgb', xgb.XGBClassifier(n_estimators=100, max_depth=6, random_state=42, eval_metric='logloss')))
    
    voting = VotingClassifier(estimators=estimators, voting='soft', n_jobs=-1)
    voting.fit(X_train, y_train)
    y_pred = voting.predict_proba(X_test)[:, 1]
    
    auc_score = roc_auc_score(y_test, y_pred)
    precision, recall, _ = precision_recall_curve(y_test, y_pred)
    pr_auc_score = pr_auc_calc(recall, precision)
    
    results.append({
        'model': 'Voting Ensemble',
        'AUC': auc_score,
        'PR-AUC': pr_auc_score,
        'time_sec': time.time() - start
    })
    print(f"  AUC: {auc_score:.4f}, PR-AUC: {pr_auc_score:.4f}")
    
    # ========================================================================
    # 12. Ensemble: Stacking
    # ========================================================================
    print("\n[12/N] Stacking Ensemble")
    start = time.time()
    estimators = [
        ('lgbm', lgb.LGBMClassifier(n_estimators=50, max_depth=5, random_state=42, verbose=-1)),
        ('rf', RandomForestClassifier(n_estimators=50, max_depth=8, random_state=42, n_jobs=-1)),
        ('hist', HistGradientBoostingClassifier(max_iter=50, random_state=42))
    ]
    
    stacking = StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(max_iter=1000),
        cv=3,
        n_jobs=-1
    )
    stacking.fit(X_train, y_train)
    y_pred = stacking.predict_proba(X_test)[:, 1]
    
    auc_score = roc_auc_score(y_test, y_pred)
    precision, recall, _ = precision_recall_curve(y_test, y_pred)
    pr_auc_score = pr_auc_calc(recall, precision)
    
    results.append({
        'model': 'Stacking Ensemble',
        'AUC': auc_score,
        'PR-AUC': pr_auc_score,
        'time_sec': time.time() - start
    })
    print(f"  AUC: {auc_score:.4f}, PR-AUC: {pr_auc_score:.4f}")
    
    return pd.DataFrame(results)

# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 80)
    print("SOTA MODEL BENCHMARK")
    print("Testing comprehensive set of state-of-the-art models")
    print("=" * 80)
    
    # Prepare data
    X_train, X_test, y_train, y_test, features = prepare_data()
    
    # Benchmark all models
    print("\n" + "=" * 80)
    print("TRAINING ALL MODELS")
    print("=" * 80)
    
    results = benchmark_all_models(X_train, X_test, y_train, y_test, features)
    
    # Sort by AUC
    results = results.sort_values('AUC', ascending=False)
    
    # Save results
    out_path = TABLES_DIR / "sota_model_benchmark.csv"
    results.to_csv(out_path, index=False)
    
    print("\n" + "=" * 80)
    print("FINAL RESULTS (sorted by AUC)")
    print("=" * 80)
    print(results.to_string(index=False))
    print(f"\n[OK] Saved to {out_path}")
    
    # Highlight best model
    best_model = results.iloc[0]
    print("\n" + "=" * 80)
    print("BEST MODEL")
    print("=" * 80)
    print(f"Model: {best_model['model']}")
    print(f"AUC: {best_model['AUC']:.4f}")
    print(f"PR-AUC: {best_model['PR-AUC']:.4f}")
    print(f"Training time: {best_model['time_sec']:.1f}s")
    
    # Compare to baseline
    baseline_auc = results[results['model'] == 'LightGBM (Baseline)']['AUC'].values[0]
    improvement = best_model['AUC'] - baseline_auc
    print(f"\nImprovement over baseline: {improvement:+.4f} AUC ({improvement/baseline_auc*100:+.2f}%)")

if __name__ == "__main__":
    main()
