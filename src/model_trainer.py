import pickle
import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

def train_models(X_train, X_test, y_train, y_test):
    """
    Train and save the Random Forest model and Scaler.
    Called by main.py.
    """

    print("--- Starting Model Training ---")

    # 1. Initialize and fit the Scaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # 2. Initialize and fit the Random Forest Model
    model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
    model.fit(X_train_scaled, y_train)

    # 3. Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)

    # 4. Save the Random Forest model
    with open('models/student_model.pkl', 'wb') as f:
        pickle.dump(model, f)

    with open('models/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)

    print(f"Model trained. Accuracy on test set: {model.score(scaler.transform(X_test), y_test):.2f}")
    print("Assets saved to models/ folder.")

    # 5. Train and save XGBoost model using the proper save method (fixes pickle warning)
    print("--- Training XGBoost Model ---")
    xgb_model = xgb.XGBClassifier(
        n_estimators=100,
        random_state=42,
        max_depth=10,
        eval_metric='logloss'
    )
    xgb_model.fit(X_train_scaled, y_train)
    xgb_model.save_model('models/xgboost_model.ubj')

    print(f"XGBoost trained. Accuracy on test set: {xgb_model.score(X_test, y_test):.2f}")
    print("XGBoost model saved to models/xgboost_model.ubj")


def load_model(model_path='models/student_model.pkl', scaler_path='models/scaler.pkl'):
    """Load the trained model and scaler"""
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)

        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)

        print("Model and scaler loaded successfully!")
        return model, scaler

    except FileNotFoundError:
        print("Error: Trained model files not found!")
        return None, None


def predict_student_risk(model, scaler, student_data):
    """Predict risk level for a student using the 11 real features"""
    if isinstance(student_data, dict):
        student_df = pd.DataFrame([student_data])
    else:
        student_df = student_data.copy()

    # Scale the input
    X_scaled = scaler.transform(student_df)

    # Make prediction
    prediction = model.predict(X_scaled)
    probabilities = model.predict_proba(X_scaled)
    risk_prob = float(probabilities[0][1]) if len(probabilities[0]) == 2 else float(np.max(probabilities[0]))

    if risk_prob > 0.6:
        risk_level = "High Risk"
    elif risk_prob > 0.3:
        risk_level = "Medium Risk"
    else:
        risk_level = "Low Risk"

    return {
        'prediction': int(prediction[0]),
        'risk_probability': risk_prob,
        'risk_level': risk_level
    }


if __name__ == "__main__":
    # This block now just tests loading if you run the file directly
    m, s = load_model()
    if m:
        print("Test Load Successful.")