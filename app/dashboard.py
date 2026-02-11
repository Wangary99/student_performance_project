import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

# 1. Page Configuration
st.set_page_config(page_title="Student Early Warning System", layout="wide")
st.title("🎓 Student Performance Early Warning Dashboard")
st.markdown("---")

# 2. Load Models and Assets
@st.cache_resource
def load_assets():
    # Loading the fixed files we created in Phase 3 & 4
    model = joblib.load("models/xgboost_model.pkl")
    encoders = joblib.load("models/categorical_encoders.pkl")
    scaler = joblib.load("models/standard_scaler.pkl")
    # Load the master data to select students from
    data = pd.read_csv("data/master_student_data.csv")
    return model, encoders, scaler, data

model, encoders, scaler, df_master = load_assets()

# 3. Sidebar - Student Selection
st.sidebar.header("Navigation")
student_id = st.sidebar.selectbox("Select Student ID", df_master['id_student'].unique())

# Filter data for the selected student
student_row = df_master[df_master['id_student'] == student_id].copy()

# --- PREPROCESSING FOR PREDICTION ---
# 1. Drop identifier columns that weren't used in training
X_input = student_row.drop(columns=['id_student', 'code_module', 'code_presentation', 'final_result', 'date_unregistration'], errors='ignore')

# 2. Apply saved Encoders (converts text to the numbers the model learned)
for col, le in encoders.items():
    if col in X_input.columns:
        X_input[col] = le.transform(X_input[col].astype(str))

# 3. Apply saved Scaler
X_input_scaled = scaler.transform(X_input)

# --- 4. PREDICT (FIXED LINE 37) ---
#  means: Get the 1st student (index 0) and their At-Risk probability (index 1)
risk_prob = model.predict_proba(X_input_scaled)
risk_level = "High Risk" if risk_prob > 0.6 else "Medium Risk" if risk_prob > 0.3 else "Low Risk"

# --- DISPLAY RESULTS ---
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Predicted Status", risk_level)
with col2:
    st.metric("Risk Probability", f"{risk_prob*100:.1f}%")
with col3:
    st.metric("Engagement (Clicks)", int(student_row['total_clicks'].values))

# --- 5. EXPLAINABLE AI (FIXED LINE 51) ---
st.subheader("🔍 Why is this student at risk? (SHAP Analysis)")
explainer = shap.TreeExplainer(model)
shap_values = explainer(X_input_scaled)

fig, ax = plt.subplots()
#  tells SHAP to explain only the 1st student selected
shap.plots.waterfall(shap_values, show=False)
st.pyplot(fig)

# 6. Pedagogical Recommendations
st.markdown("---")
st.subheader("📋 Targeted Intervention Plan")
if risk_level == "High Risk":
    st.error("🚨 **Immediate Action Required:** Schedule a 1-on-1 counseling session. Provide supplemental materials for identified weak modules.")
elif risk_level == "Medium Risk":
    st.warning("⚠️ **Monitor Engagement:** Send a personalized encouragement email. Check if the student has accessed recent quiz feedback.")
else:
    st.success("✅ **Support Success:** No immediate intervention needed. Continue standard monitoring.")