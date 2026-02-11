from src.preprocessing import create_master_table
from src.feature_eng import prepare_features
from src.model_trainer import train_models  # Added new module
import pandas as pd

if __name__ == "__main__":
    # Point to your data folder
    RAW_DATA_DIR = "data"
    
    # Phase 2: Data Ingestion
    master_df = create_master_table(RAW_DATA_DIR)
    
    # Phase 3: Feature Engineering & SMOTE
    X_train, X_test, y_train, y_test, feature_names = prepare_features(master_df)
    
    # Save feature metadata for the dashboard
    pd.Series(feature_names).to_csv("models/feature_names.csv", index=False)
    
    # Phase 4: Training & Comparative Evaluation
    train_models(X_train, X_test, y_train, y_test)
    
    print("\nTraining Complete! Ready for Phase 5: Explainable AI (XAI) and Dashboarding!")