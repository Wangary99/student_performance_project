from src.preprocessing import create_master_table
from src.feature_eng import prepare_features
from src.model_trainer import train_models
import pandas as pd

if __name__ == "__main__":
    # Conduct Phase 1 & 2
    master_df = create_master_table("data")
    X_train, X_test, y_train, y_test, feature_names = prepare_features(master_df)
    
    # Save student data and feature names for dashboard use
    master_df.to_csv("data/master_student_data.csv", index=False)
    pd.Series(feature_names).to_csv("models/feature_names.csv", index=False)
    
    # Conduct Phase 3: Training (RF and XGBoost)
    train_models(X_train, X_test, y_train, y_test)
    
    print("\n--- SYSTEM READY FOR DASHBOARD ---")