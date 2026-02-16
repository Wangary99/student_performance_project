import pickle
import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

def load_model(model_path='models/student_model.pkl', scaler_path='models/scaler.pkl'):
    """
    Load the trained model and scaler
    """
    try:
        # Load model
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        # Load scaler
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        
        print("Model and scaler loaded successfully!")
        return model, scaler
    
    except FileNotFoundError:
        print("Model files not found. Creating default model...")
        print("Please run 'python main.py' first to train the model.")
        
        # Create a default model and scaler as fallback
        model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
        scaler = StandardScaler()
        
        # Create dummy data to fit the scaler
        dummy_data = np.random.randn(100, 7)
        dummy_labels = np.random.choice([0, 1], 100)
        
        scaler.fit(dummy_data)
        model.fit(dummy_data, dummy_labels)
        
        # Save for next time
        os.makedirs('models', exist_ok=True)
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler, f)
        
        print("Default model created and saved.")
        return model, scaler


def predict_student_risk(model, scaler, student_data):
    """
    Predict risk level for a student
    
    Parameters:
    -----------
    model : trained classifier
    scaler : fitted StandardScaler
    student_data : dict or DataFrame with student features
    
    Returns:
    --------
    dict with prediction results
    """
    # Convert to DataFrame if dict
    if isinstance(student_data, dict):
        student_df = pd.DataFrame([student_data])
    else:
        student_df = student_data.copy()
    
    # Expected features
    expected_features = [
        'gender', 'age', 'study_hours_per_week', 'attendance_percentage',
        'parental_education_level', 'internet_access', 'extracurricular_activities'
    ]
    
    # Ensure all features are present
    for feature in expected_features:
        if feature not in student_df.columns:
            student_df[feature] = 0
    
    # Select only expected features
    student_df = student_df[expected_features]
    
    # Scale the input
    X_scaled = scaler.transform(student_df)
    
    # Make prediction
    prediction = model.predict(X_scaled)
    probabilities = model.predict_proba(X_scaled)
    
    # Extract risk probability (probability of positive class)
    risk_prob = float(probabilities[0][1]) if len(probabilities[0]) == 2 else float(np.max(probabilities[0]))
    
    # Determine risk level
    if risk_prob > 0.6:
        risk_level = "High Risk"
    elif risk_prob > 0.3:
        risk_level = "Medium Risk"
    else:
        risk_level = "Low Risk"
    
    return {
        'prediction': int(prediction[0]),
        'risk_probability': risk_prob,
        'risk_level': risk_level,
        'all_probabilities': probabilities[0].tolist()
    }


def batch_predict(model, scaler, students_df):
    """
    Predict risk for multiple students
    
    Parameters:
    -----------
    model : trained classifier
    scaler : fitted StandardScaler
    students_df : DataFrame with student features
    
    Returns:
    --------
    DataFrame with predictions
    """
    results = []
    
    for idx, row in students_df.iterrows():
        student_dict = row.to_dict()
        result = predict_student_risk(model, scaler, student_dict)
        result['student_id'] = idx
        results.append(result)
    
    return pd.DataFrame(results)


if __name__ == "__main__":
    # Test the model loading
    print("Testing model loading...")
    model, scaler = load_model()
    
    # Test prediction
    print("\nTesting prediction with sample data...")
    sample_student = {
        'gender': 1,
        'age': 18,
        'study_hours_per_week': 10,
        'attendance_percentage': 75,
        'parental_education_level': 2,
        'internet_access': 1,
        'extracurricular_activities': 1
    }
    
    result = predict_student_risk(model, scaler, sample_student)
    
    print("\nPrediction Results:")
    print(f"Prediction: {'At Risk' if result['prediction'] == 1 else 'On Track'}")
    print(f"Risk Probability: {result['risk_probability']:.2%}")
    print(f"Risk Level: {result['risk_level']}")