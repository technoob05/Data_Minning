
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, average_precision_score
from pathlib import Path
import sys

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.features import get_feature_columns

def run_loao():
    print("Loading data for LOAO...")
    df = pd.read_parquet("msr26-aidev-triage/artifacts/pr_features.parquet")
    df = df[df["status"].isin(["merged", "rejected"])].copy()
    if "is_instant_merge" in df.columns:
        df = df[df["is_instant_merge"] == 0].copy()
        
    feature_cols = get_feature_columns()
    # Remove agent-specific features to be fair/true LOAO?
    # Actually, we want to know if the model generalizes. If we include agent features, 
    # the model might learn "Agent A is good". If we test on Agent B, that feature is useless or informative (if Agent B is known?).
    # In LOAO, the test agent is UNSEEN. So `agent_encoded` for that agent should probably be handled carefully or just rely on other features.
    # LightGBM handles categoricals. If a new category appears, it might be handled as default.
    # Ideally, we remove agent identifiers to prove "Structure > Identity". 
    # The ablation showed "No Agent ID" dropped AUC only slightly. 
    # Let's KEEP agent features but accept they won't help for the new agent.
    
    # Identify Agents
    # We need the original 'agent' column. 
    # It was 'agent' in prior scripts.
    if "agent" not in df.columns:
        # We might need to reload pr_base to get agent names if they were dropped?
        # But wait, pr_features usually keeps meta columns.
        pass

    agents = df["agent"].unique()
    print(f"Agents found: {agents}")
    
    results = []
    
    for test_agent in agents:
        print(f"Testing on Left-Out Agent: {test_agent}...")
        
        train_df = df[df["agent"] != test_agent]
        test_df = df[df["agent"] == test_agent]
        
        if len(test_df) < 50:
            print("Skipping (too small)")
            continue
            
        X_train = train_df[feature_cols]
        y_train = train_df["is_high_cost"]
        
        X_test = test_df[feature_cols]
        y_test = test_df["is_high_cost"]
        
        train_data = lgb.Dataset(X_train, label=y_train)
        valid_data = lgb.Dataset(X_test, label=y_test, reference=train_data)
        
        params = {
            "objective": "binary", "metric": "auc", "boosting_type": "gbdt",
            "num_leaves": 31, "learning_rate": 0.05, "feature_fraction": 0.9, "verbose": -1
        }
        
        model = lgb.train(params, train_data, num_boost_round=100, valid_sets=[valid_data], 
                          callbacks=[lgb.early_stopping(stopping_rounds=10)])
        
        y_prob = model.predict(X_test)
        auc = roc_auc_score(y_test, y_prob)
        # AP for Ghosting (if we did ghosting LOAO, but high cost is main target per request)
        
        print(f"Agent: {test_agent}, AUC: {auc:.4f}")
        results.append({
            "Left_Out_Agent": test_agent,
            "Train_Size": len(train_df),
            "Test_Size": len(test_df),
            "AUC": auc
        })
        
    res_df = pd.DataFrame(results)
    print("\nLOAO Results:")
    print(res_df)
    res_df.to_csv("msr26-aidev-triage/outputs/tables/loao_results.csv", index=False)

if __name__ == "__main__":
    run_loao()
