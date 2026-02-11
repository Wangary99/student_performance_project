import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, accuracy_score, f1_score, recall_score, precision_score

def train_models(X_train, X_test, y_train, y_test):
    print("--- Starting Model Training ---")
    
    # 1. Random Forest Training
    # RF is robust against overfitting and provides a stable baseline
    print("Training Random Forest...")
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    rf_preds = rf_model.predict(X_test)
    
    # 2. XGBoost Training
    # XGBoost uses iterative error correction to capture complex, non-linear patterns
    print("Training XGBoost...")
    xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    xgb_model.fit(X_train, y_train)
    xgb_preds = xgb_model.predict(X_test)
    
    # 3. Comparative Evaluation
    # We display Precision, Recall, and F1 to evaluate institutional trade-offs
    print("\n--- Model Evaluation Results ---")
    print("\n")
    print(classification_report(y_test, rf_preds))
    
    print("\n")
    print(classification_report(y_test, xgb_preds))
    
    # 4. Save Models for Deployment
    # This allows the dashboard to use these specific trained versions
    joblib.dump(rf_model, "models/random_forest_model.pkl")
    joblib.dump(xgb_model, "models/xgboost_model.pkl")
    print("\nModels successfully saved to the 'models/' folder.")