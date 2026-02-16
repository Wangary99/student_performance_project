from src.preprocessing import create_master_table
from src.feature_eng import prepare_features
from src.model_trainer import train_models
import pandas as pd

if __name__ == "__main__":
    # Point to the OULAD raw data folder
    RAW_DATA_DIR = "data"
    
    # Step 1: Create Master Data
    master_df = create_master_table(RAW_DATA_DIR)
    
    # Step 2: Engineer Features
    X_train, X_test, y_train, y_test, feature_names = prepare_features(master_df)
    
    # Step 3: Save Assets for Dashboard
    master_df.to_csv("data/master_student_data.csv", index=False)
    pd.Series(feature_names).to_csv("models/feature_names.csv", index=False)
    
    # Step 4: Train Models (Target 94%+ Accuracy)
    train_models(X_train, X_test, y_train, y_test)
    
    print("\n--- PIPELINE SUCCESSFUL: SYSTEM READY ---")