import gradio as gr
import pickle
import pandas as pd
import numpy as np
import joblib
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder, StandardScaler

# --- 1. DATA & MODEL LOADING ---
try:
    # This is the master record visible in your 'data/' folder
    master_data = pd.read_csv("data/master_student_data.csv")
    
    # Matching filenames exactly as they appear in your sidebar
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

# --- 2. THE LOOKUP & PREDICT FUNCTION ---
def predict_by_id(student_id, model_choice):
    try:
        sid = int(student_id)
        # Search using 'id_student' column from your master CSV
        student_row = master_data[master_data['id_student'] == sid]
        
        if student_row.empty:
            return f"❌ Student ID {student_id} not found."

        model = rf_model if model_choice == "Random Forest" else xgb_model
        
        # Features required for the model as defined in your proposal
        cols = ['gender', 'region', 'highest_education', 'imd_band', 'age_band', 
                'num_of_prev_attempts', 'studied_credits', 'disability', 
                'date_registration', 'avg_score', 'total_clicks']
        
        input_df = student_row[cols].copy()
        
        # Apply encoding (text to numbers)
        for col in ['gender', 'region', 'highest_education', 'imd_band', 'age_band', 'disability']:
            if col in encoders:
                input_df[col] = encoders[col].transform(input_df[col])
        
        # Scale and Predict
        scaled_data = scaler.transform(input_df.values)
        prediction = model.predict(scaled_data)[0]
        
        status = "✅ PASS" if prediction == 1 else "⚠️ AT RISK"
        return f"Student ID: {student_id}\nModel: {model_choice}\nPrediction: {status}"

    except Exception as e:
        import traceback
        return f"Prediction Error: {str(e)}\n\nFull details:\n{traceback.format_exc()}"

# --- 3. GRADIO UI (Proposal Alignment) ---
demo = gr.Interface(
    fn=predict_by_id,
    inputs=[
        gr.Textbox(label="Lecturer Access: Enter Student ID (e.g., 11391)"),
        gr.Dropdown(["Random Forest", "XGBoost"], label="Select Model", value="Random Forest")
    ],
    outputs="text",
    title="🎓 Lecturer Dashboard: Early Warning System",
    description="Automatically retrieves student behavioral and academic data for real-time risk prediction."
)

if __name__ == "__main__":
    demo.launch()