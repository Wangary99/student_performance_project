import sys
import os

# Add current directory to Python path to ensure imports work
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
import pandas as pd
import numpy as np
from model_trainer import load_model
import shap
import matplotlib.pyplot as plt

# Load the trained model
model, scaler = load_model()

# Streamlit app configuration
st.set_page_config(page_title="Student Performance Dashboard", layout="wide")
st.title("🎓 Student Performance Prediction Dashboard")
st.markdown("### Predict student academic risk and get actionable insights")

# Sidebar for user inputs
st.sidebar.header("Student Information")

# Input fields
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
age = st.sidebar.number_input("Age", min_value=15, max_value=25, value=18)
study_hours = st.sidebar.slider("Weekly Study Hours", 0, 40, 10)
attendance = st.sidebar.slider("Attendance (%)", 0, 100, 80)
parental_education = st.sidebar.selectbox(
    "Parental Education Level",
    ["High School", "Some College", "Bachelor's", "Master's", "PhD"]
)
internet_access = st.sidebar.selectbox("Internet Access", ["Yes", "No"])
extracurricular = st.sidebar.selectbox("Extracurricular Activities", ["Yes", "No"])

# Encode categorical variables
gender_encoded = 1 if gender == "Male" else 0
parental_edu_map = {
    "High School": 0, 
    "Some College": 1, 
    "Bachelor's": 2, 
    "Master's": 3, 
    "PhD": 4
}
parental_education_encoded = parental_edu_map[parental_education]
internet_encoded = 1 if internet_access == "Yes" else 0
extracurricular_encoded = 1 if extracurricular == "Yes" else 0

# Create input dataframe
input_data = pd.DataFrame({
    'gender': [gender_encoded],
    'age': [age],
    'study_hours_per_week': [study_hours],
    'attendance_percentage': [attendance],
    'parental_education_level': [parental_education_encoded],
    'internet_access': [internet_encoded],
    'extracurricular_activities': [extracurricular_encoded]
})

# Predict button
if st.sidebar.button("Predict Performance"):
    # STEP 1: Scale the input
    X_input_scaled = scaler.transform(input_data)
    
    # STEP 2: Make prediction
    prediction = model.predict(X_input_scaled)
    risk_prob_array = model.predict_proba(X_input_scaled)
    
    # FIX: Extract the probability value correctly
    # risk_prob_array is shape (1, n_classes), we need the probability of the positive class
    # Assuming binary classification: class 0 = Low Risk, class 1 = High Risk
    # We want the probability of class 1 (High Risk)
    if len(risk_prob_array[0]) == 2:
        risk_prob = float(risk_prob_array[0][1])  # Probability of positive class (High Risk)
    else:
        # If multiclass, take the max probability
        risk_prob = float(np.max(risk_prob_array[0]))
    
    # STEP 3: Determine risk level
    if risk_prob > 0.6:
        risk_level = "High Risk"
    elif risk_prob > 0.3:
        risk_level = "Medium Risk"
    else:
        risk_level = "Low Risk"
    
    # STEP 4: Display results
    st.success("## Prediction Results")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Predicted Status", "At Risk" if prediction[0] == 1 else "On Track")
    
    with col2:
        st.metric("Risk Probability", f"{risk_prob*100:.1f}%")
    
    with col3:
        st.metric("Engagement (Clicks)", int(study_hours * 10 + attendance))
    
    # Risk level with color coding
    if risk_level == "High Risk":
        st.error(f"### ⚠️ {risk_level}")
    elif risk_level == "Medium Risk":
        st.warning(f"### ⚠️ {risk_level}")
    else:
        st.success(f"### ✅ {risk_level}")
    
    # STEP 5: SHAP Explainability
    st.subheader("🔍 Why is this student at risk? (SHAP Analysis)")
    
    try:
        # Create SHAP explainer
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_input_scaled)
        
        # Create SHAP waterfall plot
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # For binary classification, shap_values might be a list with 2 elements
        if isinstance(shap_values, list):
            shap_values_to_plot = shap_values[1][0]  # Use positive class SHAP values
        else:
            shap_values_to_plot = shap_values[0]
        
        # Create feature names
        feature_names = list(input_data.columns)
        
        # Manual waterfall-style visualization
        base_value = explainer.expected_value
        if isinstance(base_value, (list, np.ndarray)):
            base_value = base_value[1] if len(base_value) > 1 else base_value[0]
        
        # Sort features by absolute SHAP value
        feature_importance = list(zip(feature_names, shap_values_to_plot))
        feature_importance.sort(key=lambda x: abs(x[1]), reverse=True)
        
        # Plot horizontal bar chart
        features = [f[0] for f in feature_importance]
        values = [f[1] for f in feature_importance]
        colors = ['red' if v > 0 else 'blue' for v in values]
        
        ax.barh(features, values, color=colors)
        ax.set_xlabel('SHAP Value (Impact on Prediction)')
        ax.set_title('Feature Importance for Prediction')
        ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
        
        st.pyplot(fig)
        plt.close()
        
        # Display interpretations
        st.markdown("### 📊 Key Factors:")
        for feature, value in feature_importance[:3]:
            if value > 0:
                st.markdown(f"- **{feature}**: Increases risk (SHAP: {value:.3f})")
            else:
                st.markdown(f"- **{feature}**: Decreases risk (SHAP: {value:.3f})")
    
    except Exception as e:
        st.warning(f"SHAP analysis unavailable: {str(e)}")
        st.info("Displaying basic feature importance instead.")
        
        # Fallback: Show input values
        st.markdown("### Input Values:")
        for col in input_data.columns:
            st.markdown(f"- **{col}**: {input_data[col].values[0]}")
    
    # STEP 6: Recommendations
    st.subheader("💡 Recommendations")
    
    if risk_level == "High Risk":
        st.markdown("""
        - **Immediate Intervention Required**
        - Schedule one-on-one tutoring sessions
        - Contact parents/guardians
        - Monitor attendance closely
        - Provide additional study resources
        """)
    elif risk_level == "Medium Risk":
        st.markdown("""
        - **Preventive Measures Recommended**
        - Encourage study groups
        - Provide time management guidance
        - Regular check-ins with teachers
        - Consider peer tutoring
        """)
    else:
        st.markdown("""
        - **Continue Current Support**
        - Maintain engagement levels
        - Encourage continued participation
        - Recognize achievements
        - Provide enrichment opportunities
        """)

else:
    # Display welcome message
    st.info("👈 Enter student information in the sidebar and click 'Predict Performance' to see results.")
    
    # Display sample statistics or information
    st.subheader("About This Dashboard")
    st.markdown("""
    This dashboard uses machine learning to predict student academic performance and identify students at risk.
    
    **Features:**
    - Real-time risk assessment
    - SHAP-based explainability
    - Personalized recommendations
    - Interactive visualizations
    
    **How to use:**
    1. Enter student information in the sidebar
    2. Click "Predict Performance"
    3. Review the risk assessment and recommendations
    """)