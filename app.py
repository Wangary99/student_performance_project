import gradio as gr
import pickle
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder, StandardScaler

# --- 1. DATA & MODEL LOADING ---
try:
    master_data = pd.read_csv("data/master_student_data.csv")

    with open('models/student_model.pkl', 'rb') as f:
        rf_model = pickle.load(f)

    xgb_model = xgb.XGBClassifier()
    xgb_model.load_model('models/xgboost_model.ubj')

    with open('models/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open('models/categorical_encoders.pkl', 'rb') as f:
        encoders = pickle.load(f)

    print("Dashboard assets loaded successfully!")
except Exception as e:
    print(f"Startup Error: {e}")

# --- 2. THE LOOKUP, PREDICT & SHAP FUNCTION ---
def predict_by_id(student_id, model_choice):
    try:
        sid = int(student_id)
        student_row = master_data[master_data['id_student'] == sid]

        if student_row.empty:
            return f"❌ Student ID {student_id} not found.", None

        model = rf_model if model_choice == "Random Forest" else xgb_model

        cols = ['gender', 'region', 'highest_education', 'imd_band', 'age_band',
                'num_of_prev_attempts', 'studied_credits', 'disability',
                'date_registration', 'avg_score', 'total_clicks']

        input_df = student_row[cols].copy()

        # Apply encoding
        for col in ['gender', 'region', 'highest_education', 'imd_band', 'age_band', 'disability']:
            if col in encoders:
                input_df[col] = encoders[col].transform(input_df[col])

        # Scale
        scaled_data = scaler.transform(input_df.values)
        prediction = model.predict(scaled_data)[0]

        status = "✅ PASS" if prediction == 1 else "⚠️ AT RISK"

        # Risk probability
        prob = model.predict_proba(scaled_data)[0]
        risk_prob = prob[1] if len(prob) == 2 else max(prob)
        risk_level = "High Risk" if risk_prob > 0.6 else "Medium Risk" if risk_prob > 0.3 else "Low Risk"

        result_text = (
            f"Student ID:    {student_id}\n"
            f"Model:         {model_choice}\n"
            f"Prediction:    {status}\n"
            f"Risk Level:    {risk_level}\n"
            f"Risk Score:    {risk_prob * 100:.1f}%"
        )

        # --- SHAP Explanation ---
        explainer = shap.TreeExplainer(model)
        scaled_df = pd.DataFrame(scaled_data, columns=cols)
        shap_values = explainer(scaled_df)

        fig, ax = plt.subplots(figsize=(10, 5))

        # For binary classification, pick the 'At Risk' class explanation
        if hasattr(shap_values, 'values') and len(shap_values.values.shape) == 3:
            shap.plots.waterfall(shap_values[0, :, 1], show=False)
        else:
            shap.plots.waterfall(shap_values[0], show=False)

        plt.title(f"SHAP Explanation — Why is Student {student_id} predicted as {status}?", 
                  fontsize=11, pad=15)
        plt.tight_layout()

        return result_text, fig

    except Exception as e:
        import traceback
        return f"Prediction Error: {str(e)}\n\nFull details:\n{traceback.format_exc()}", None

# --- 3. GRADIO UI ---
demo = gr.Interface(
    fn=predict_by_id,
    inputs=[
        gr.Textbox(label="Lecturer Access: Enter Student ID (e.g., 11391)"),
        gr.Dropdown(["Random Forest", "XGBoost"], label="Select Model", value="Random Forest")
    ],
    outputs=[
        gr.Textbox(label="Prediction Result"),
        gr.Plot(label="SHAP Explanation: Top Factors Influencing This Prediction")
    ],
    title="🎓 Lecturer Dashboard: Early Warning System",
    description="Automatically retrieves student behavioral and academic data for real-time risk prediction with explainable AI."
)

if __name__ == "__main__":
    demo.launch()