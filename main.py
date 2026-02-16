from src.preprocessing import create_master_table
from src.feature_eng import prepare_features
from src.model_trainer import train_models
import pandas as pd

if __name__ == "__main__":
    master_df = create_master_table("data")
    X_train, X_test, y_train, y_test, feature_names = prepare_features(master_df)
    
    # Save feature names for SHAP visualization
    pd.Series(feature_names).to_csv("models/feature_names.csv", index=False)
    
    train_models(X_train, X_test, y_train, y_test)
    print("\n--- PIPELINE COMPLETE: MODELS SAVED ---")