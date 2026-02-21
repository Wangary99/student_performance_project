import pickle
import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, f1_score, recall_score, precision_score, accuracy_score
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
    X_test_scaled = scaler.transform(X_test)

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

    # 5. Train and save XGBoost model using the proper save method
    print("--- Training XGBoost Model ---")
    xgb_model = xgb.XGBClassifier(
        n_estimators=100,
        random_state=42,
        max_depth=10,
        eval_metric='logloss'
    )
    xgb_model.fit(X_train_scaled, y_train)
    xgb_model.save_model('models/xgboost_model.ubj')
    print("XGBoost model saved to models/xgboost_model.ubj")

    # 6. K-Fold Cross Validation for both models
    print("--- Running K-Fold Cross Validation (5-Fold) ---")
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    rf_cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=kfold, scoring='f1')
    xgb_cv_scores = cross_val_score(xgb_model, X_train_scaled, y_train, cv=kfold, scoring='f1')

    print(f"RF   K-Fold F1 Scores: {rf_cv_scores.round(3)} | Mean: {rf_cv_scores.mean():.3f}")
    print(f"XGB  K-Fold F1 Scores: {xgb_cv_scores.round(3)} | Mean: {xgb_cv_scores.mean():.3f}")

    # 7. Full evaluation metrics on test set
    rf_preds  = model.predict(X_test_scaled)
    xgb_preds = xgb_model.predict(X_test_scaled)

    rf_metrics = {
        'accuracy':  accuracy_score(y_test, rf_preds),
        'precision': precision_score(y_test, rf_preds),
        'recall':    recall_score(y_test, rf_preds),
        'f1':        f1_score(y_test, rf_preds),
        'cv_f1_mean': rf_cv_scores.mean(),
        'cv_f1_std':  rf_cv_scores.std()
    }

    xgb_metrics = {
        'accuracy':  accuracy_score(y_test, xgb_preds),
        'precision': precision_score(y_test, xgb_preds),
        'recall':    recall_score(y_test, xgb_preds),
        'f1':        f1_score(y_test, xgb_preds),
        'cv_f1_mean': xgb_cv_scores.mean(),
        'cv_f1_std':  xgb_cv_scores.std()
    }

    # 8. Print and save validation report
    print("\n--- VALIDATION REPORT ---")
    print("\nRandom Forest:")
    print(classification_report(y_test, rf_preds, target_names=['Pass', 'At Risk']))
    print("\nXGBoost:")
    print(classification_report(y_test, xgb_preds, target_names=['Pass', 'At Risk']))

    report_lines = []
    report_lines.append("=" * 55)
    report_lines.append("   STUDENT PERFORMANCE PREDICTION - VALIDATION REPORT")
    report_lines.append("=" * 55)
    report_lines.append("")
    report_lines.append("RANDOM FOREST")
    report_lines.append("-" * 35)
    report_lines.append(f"  Accuracy:          {rf_metrics['accuracy']:.4f}")
    report_lines.append(f"  Precision:         {rf_metrics['precision']:.4f}")
    report_lines.append(f"  Recall:            {rf_metrics['recall']:.4f}")
    report_lines.append(f"  F1-Score:          {rf_metrics['f1']:.4f}")
    report_lines.append(f"  K-Fold CV F1 Mean: {rf_metrics['cv_f1_mean']:.4f}")
    report_lines.append(f"  K-Fold CV F1 Std:  {rf_metrics['cv_f1_std']:.4f}")
    report_lines.append("")
    report_lines.append("XGBOOST")
    report_lines.append("-" * 35)
    report_lines.append(f"  Accuracy:          {xgb_metrics['accuracy']:.4f}")
    report_lines.append(f"  Precision:         {xgb_metrics['precision']:.4f}")
    report_lines.append(f"  Recall:            {xgb_metrics['recall']:.4f}")
    report_lines.append(f"  F1-Score:          {xgb_metrics['f1']:.4f}")
    report_lines.append(f"  K-Fold CV F1 Mean: {xgb_metrics['cv_f1_mean']:.4f}")
    report_lines.append(f"  K-Fold CV F1 Std:  {xgb_metrics['cv_f1_std']:.4f}")
    report_lines.append("")
    report_lines.append("=" * 55)

    # Determine which model performed better
    if rf_metrics['recall'] >= xgb_metrics['recall']:
        report_lines.append("CONCLUSION: Random Forest has higher Recall.")
        report_lines.append("Better at catching at-risk students (fewer missed cases).")
    else:
        report_lines.append("CONCLUSION: XGBoost has higher Recall.")
        report_lines.append("Better at catching at-risk students (fewer missed cases).")
    report_lines.append("=" * 55)

    os.makedirs('reports', exist_ok=True)
    report_path = 'reports/validation_report.txt'
    with open(report_path, 'w') as f:
        f.write('\n'.join(report_lines))

    print(f"\nValidation report saved to {report_path}")
    print("Assets saved to models/ folder.")


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
    m, s = load_model()
    if m:
        print("Test Load Successful.")