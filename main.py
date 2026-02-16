import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import pickle
import os

def load_and_prepare_data(filepath='student_data.csv'):
    """
    Load and prepare the student performance data
    """
    try:
        # Load data
        df = pd.read_csv(filepath)
        
        print(f"Data loaded successfully: {df.shape}")
        print(f"Columns: {df.columns.tolist()}")
        
        return df
    
    except FileNotFoundError:
        print(f"Error: File {filepath} not found.")
        print("Creating sample data for demonstration...")
        
        # Create sample data
        np.random.seed(42)
        n_samples = 1000
        
        sample_data = {
            'gender': np.random.choice([0, 1], n_samples),
            'age': np.random.randint(15, 25, n_samples),
            'study_hours_per_week': np.random.randint(0, 40, n_samples),
            'attendance_percentage': np.random.randint(50, 100, n_samples),
            'parental_education_level': np.random.randint(0, 5, n_samples),
            'internet_access': np.random.choice([0, 1], n_samples),
            'extracurricular_activities': np.random.choice([0, 1], n_samples),
            'at_risk': np.random.choice([0, 1], n_samples)
        }
        
        df = pd.DataFrame(sample_data)
        df.to_csv('student_data.csv', index=False)
        print("Sample data created and saved as 'student_data.csv'")
        
        return df


def preprocess_data(df):
    """
    Preprocess the data: handle missing values, encode categoricals, etc.
    """
    # Make a copy
    df_processed = df.copy()
    
    # Handle missing values
    if df_processed.isnull().sum().sum() > 0:
        print("Handling missing values...")
        df_processed = df_processed.fillna(df_processed.mean(numeric_only=True))
    
    # Identify target variable
    if 'at_risk' in df_processed.columns:
        target = 'at_risk'
    elif 'performance' in df_processed.columns:
        target = 'performance'
    elif 'risk_level' in df_processed.columns:
        target = 'risk_level'
    else:
        # Create a synthetic target based on attendance and study hours
        print("No target variable found. Creating synthetic target...")
        df_processed['at_risk'] = ((df_processed['attendance_percentage'] < 70) | 
                                   (df_processed.get('study_hours_per_week', 10) < 5)).astype(int)
        target = 'at_risk'
    
    # Separate features and target
    X = df_processed.drop(columns=[target])
    y = df_processed[target]
    
    # Ensure all features are numeric
    for col in X.columns:
        if X[col].dtype == 'object':
            # Simple label encoding for categorical variables
            X[col] = pd.Categorical(X[col]).codes
    
    print(f"Features: {X.columns.tolist()}")
    print(f"Target: {target}")
    print(f"Target distribution:\n{y.value_counts()}")
    
    return X, y


def train_models(X, y):
    """
    Train multiple models and select the best one
    """
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Initialize models
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42, max_depth=5),
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000)
    }
    
    results = {}
    
    print("\n" + "="*50)
    print("Training Models...")
    print("="*50)
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        
        # Train
        model.fit(X_train_scaled, y_train)
        
        # Predict
        y_pred = model.predict(X_test_scaled)
        
        # Evaluate
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"Accuracy: {accuracy:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        results[name] = {
            'model': model,
            'accuracy': accuracy,
            'predictions': y_pred
        }
    
    # Select best model
    best_model_name = max(results, key=lambda x: results[x]['accuracy'])
    best_model = results[best_model_name]['model']
    best_accuracy = results[best_model_name]['accuracy']
    
    print("\n" + "="*50)
    print(f"Best Model: {best_model_name}")
    print(f"Best Accuracy: {best_accuracy:.4f}")
    print("="*50)
    
    return best_model, scaler, X_test_scaled, y_test


def save_model(model, scaler, filepath='models/'):
    """
    Save the trained model and scaler
    """
    # Create directory if it doesn't exist
    os.makedirs(filepath, exist_ok=True)
    
    # Save model
    model_path = os.path.join(filepath, 'student_model.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    
    # Save scaler
    scaler_path = os.path.join(filepath, 'scaler.pkl')
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    
    print(f"\nModel saved to: {model_path}")
    print(f"Scaler saved to: {scaler_path}")


def main():
    """
    Main execution function
    """
    print("Starting Student Performance Prediction Pipeline...")
    print("="*50)
    
    # Load data
    df = load_and_prepare_data()
    
    # Preprocess
    X, y = preprocess_data(df)
    
    # Train models
    best_model, scaler, X_test_scaled, y_test = train_models(X, y)
    
    # Save model
    save_model(best_model, scaler)
    
    print("\n" + "="*50)
    print("Pipeline completed successfully!")
    print("="*50)
    print("\nNext steps:")
    print("1. Run 'streamlit run dashboard.py' to launch the dashboard")
    print("2. Or use model_trainer.py to make predictions")


if __name__ == "__main__":
    main()