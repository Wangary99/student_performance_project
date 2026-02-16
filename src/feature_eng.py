import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

def prepare_features(df):
    print("--- Phase 2: Advanced Feature Engineering ---")
    
    # Map results to binary target: 1 = At-Risk (Fail/Withdrawn), 0 = Success
    target_map = {'Pass': 0, 'Distinction': 0, 'Fail': 1, 'Withdrawn': 1}
    df['target'] = df['final_result'].map(target_map)

    # Remove identifiers and redundant data
    df_clean = df.drop(columns=['id_student', 'code_module', 'code_presentation', 
                                'final_result', 'date_unregistration'], errors='ignore')

    # Encode categorical text into numbers and SAVE the encoders
    encoders = {}
    cat_cols = df_clean.select_dtypes(include=['object']).columns
    for col in cat_cols:
        le = LabelEncoder()
        df_clean[col] = le.fit_transform(df_clean[col].astype(str))
        encoders[col] = le
    joblib.dump(encoders, "models/categorical_encoders.pkl")

    X = df_clean.drop('target', axis=1)
    y = df_clean['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale the numbers and SAVE the scaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    joblib.dump(scaler, "models/standard_scaler.pkl")

    # Apply SMOTE to balance the dataset
    X_res, y_res = SMOTE(random_state=42).fit_resample(X_train_scaled, y_train)
    
    print(f"Ready for training! Post-SMOTE shape: {X_res.shape}")
    return X_res, X_test_scaled, y_res, y_test, X.columns