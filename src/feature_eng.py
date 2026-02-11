import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

def prepare_features(df):
    print("--- Starting Feature Engineering ---")
    
    # 1. Target Variable Mapping
    target_map = {'Pass': 0, 'Distinction': 0, 'Fail': 1, 'Withdrawn': 1}
    df['target'] = df['final_result'].map(target_map)

    # 2. Drop non-predictive columns
    cols_to_drop = ['id_student', 'code_module', 'code_presentation', 
                    'final_result', 'date_unregistration']
    df_clean = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

    # 3. Handle Categorical Columns (Label Encoding)
    # We save the encoders in a dictionary to reuse them in the dashboard
    encoders = {}
    cat_cols = df_clean.select_dtypes(include=['object']).columns
    for col in cat_cols:
        le = LabelEncoder()
        df_clean[col] = le.fit_transform(df_clean[col].astype(str))
        encoders[col] = le
    
    # Save encoders for the dashboard
    joblib.dump(encoders, "models/categorical_encoders.pkl")

    # 4. Split Features and Target
    X = df_clean.drop('target', axis=1)
    y = df_clean['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 5. Standardization (Scaling)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Save scaler for the dashboard
    joblib.dump(scaler, "models/standard_scaler.pkl")

    # 6. SMOTE for class balance
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X_train_scaled, y_train)
    
    return X_res, X_test_scaled, y_res, y_test, X.columns