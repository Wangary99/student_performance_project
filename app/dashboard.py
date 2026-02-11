import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

st.set_page_config(page_title="Student Early Warning System", layout="wide")
st.title("🎓 Student Performance Early Warning Dashboard")

@st.cache_resource
def load_assets():
    model = joblib.load("models/xgboost_model.pkl")
    encoders = joblib.load("models/categorical_encoders.pkl")
    scaler = joblib.load("models/standard_scaler.pkl")
    data = pd.read_csv("data/master_student_data.csv")
    return model, encoders, scaler, data

try:
    model, encoders, scaler, df_master = load_assets()
except Exception as e:
    st.error(f"Models missing. Run 'python main.py' first. Error: {e}")
    st.stop()

student_id = st.sidebar.selectbox("Select Student ID", df_master['id_student'].unique())
student_row = df_master[df_master['id_student'] == student_id].copy()

# --- STEP 1: CONVERT TEXT TO NUMBERS ---
X_input = student_row.drop(columns=['id_student', 'code_module', 'code_presentation', 'final_result', 'date_unregistration'], errors='ignore')

# Use saved encoders to ensure categories match training
for col, le in encoders.items():
    if col in X_input.columns:
        X_input[col] = le.transform(X_input[col].astype(str))

# Scale features exactly as the model expects
X_input_scaled = scaler.transform(X_input)

# --- STEP 2: PREDICT (FIXED INDEXING) ---
#  retrieves row 0 (the student) and column 1 (Risk Probability)
risk_prob_val = model.predict_proba(X_input_scaled)
risk_prob = float(risk_prob_val)

risk_level = "High Risk" if risk_prob > 0.6 else "Medium Risk" if risk_prob > 0.3 else "Low Risk"

# --- STEP 3: DISPLAY ---
c1, c2, c3 = st.columns(3)
c1.metric("Predicted Status", risk_level)
c2.metric("Risk Probability", f"{risk_prob*100:.1f}%")
c3.metric("Engagement (Clicks)", int(student_row['total_clicks'].values))

st.subheader("🔍 Why is this student at risk? (SHAP Analysis)")
# We use DataFrame to keep feature names in the plot
X_explain = pd.DataFrame(X_input_scaled, columns=X_input.columns)
explainer = shap.TreeExplainer(model)
shap_values = explainer(X_explain)

fig, ax = plt.subplots(figsize=(10, 4))
#  tells SHAP to explain only the first row (the selected student)
shap.plots.waterfall(shap_values, show=False)
st.pyplot(fig)