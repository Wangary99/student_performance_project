from src.preprocessing import create_master_table
from src.feature_eng import prepare_features
from src.model_trainer import train_models
import pandas as pd

if __name__ == "__main__":
    master_df = create_master_table("data")
    X_train, X_test, y_train, y_test, feature_names = prepare_features(master_df)
    
    # Crucial: Save the clean master data so the dashboard doesn't have to re-read OULAD
    master_df.to_csv("data/master_student_data.csv", index=False)
    pd.Series(feature_names).to_csv("models/feature_names.csv", index=False)
    
    train_models(X_train, X_test, y_train, y_test)
    print("\n--- PIPELINE SUCCESSFUL: SYSTEM READY ---")