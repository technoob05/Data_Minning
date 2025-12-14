import sys
from pathlib import Path
import pandas as pd
import joblib

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.features import get_feature_columns
from src.models import train_lgbm_model, explain_model
from src.viz import plot_shap_summary

def main():
    input_path = "msr26-aidev-triage/artifacts/pr_features.parquet"
    output_dir = "msr26-aidev-triage/outputs"
    model_dir = "msr26-aidev-triage/artifacts"
    
    print(f"Loading features from {input_path}...")
    if not Path(input_path).exists():
        print("Error: Features table not found. Run 04_features.py first.")
        return

    feat_df = pd.read_parquet(input_path)
    
    # Get feature columns
    feature_cols = get_feature_columns()
    
    # 1. Train High Cost Model (Main Triage Model)
    print("\n=== Training High Cost Model (Top 20% Effort) ===")
    model_cost, metrics_cost, X_test_cost = train_lgbm_model(feat_df, feature_cols, target_col="is_high_cost")
    
    # 2. Train Ghosting Model (Wasted Effort)
    print("\n=== Training Ghosting Model (Rejected with Effort) ===")
    model_ghost, metrics_ghost, X_test_ghost = train_lgbm_model(feat_df, feature_cols, target_col="is_ghosted")
    
    # Save models
    joblib.dump(model_cost, Path(model_dir) / "triage_model_high_cost.pkl")
    joblib.dump(model_ghost, Path(model_dir) / "triage_model_ghosting.pkl")
    
    # Save metrics
    metrics_df = pd.DataFrame([metrics_cost, metrics_ghost], index=["high_cost", "ghosting"])
    metrics_df["target"] = metrics_df.index
    metrics_path = Path(output_dir) / "tables" / "model_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)
    print(f"Saved metrics to {metrics_path}")
    
    # Explain High Cost Model
    print("\nExplaining High Cost Model...")
    explainer_cost, shap_values_cost = explain_model(model_cost, X_test_cost)
    shap_path_cost = Path(output_dir) / "figures" / "shap_summary_high_cost.png"
    plot_shap_summary(shap_values_cost, X_test_cost, str(shap_path_cost))
    
    # Explain Ghosting Model
    print("\nExplaining Ghosting Model...")
    explainer_ghost, shap_values_ghost = explain_model(model_ghost, X_test_ghost)
    shap_path_ghost = Path(output_dir) / "figures" / "shap_summary_ghosting.png"
    plot_shap_summary(shap_values_ghost, X_test_ghost, str(shap_path_ghost))
    
    print("Done.")

if __name__ == "__main__":
    main()
