from src.preprocessing import create_master_table
from src.feature_eng import prepare_features
from src.model_trainer import train_models
import pandas as pd
import joblib

if __name__ == "__main__":
    RAW_DATA_DIR = "data"
    
    # 1. Ingestion & Preprocessing (Fill all NaNs)
    master_df = create_master_table(RAW_DATA_DIR)
    
    # 2. Feature Engineering (Encode Categories & SMOTE)
    X_train, X_test, y_train, y_test, feature_names = prepare_features(master_df)
    
    # 3. Save feature names for XAI layout
    pd.Series(feature_names).to_csv("models/feature_names.csv", index=False)
    
    # 4. Train and Evaluation (94%+ Accuracy target)
    train_models(X_train, X_test, y_train, y_test)
    
    print("\n--- SYSTEM READY ---")