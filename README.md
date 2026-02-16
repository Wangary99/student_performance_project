# 🎓 Student Performance Prediction System

An intelligent early warning system that leverages machine learning to identify at-risk students and provide data-driven insights for timely academic interventions.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Demo](#demo)
- [Technology Stack](#technology-stack)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Data Description](#data-description)
- [Model Performance](#model-performance)
- [Dashboard Guide](#dashboard-guide)
- [API Reference](#api-reference)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## 🎯 Overview

This project addresses a critical challenge in education: **early identification of students who may need additional support**. By analyzing various factors including study habits, attendance patterns, and demographic information, our machine learning model predicts academic risk levels with high accuracy.

### Why This Matters

- **Early Intervention**: Identify struggling students before it's too late
- **Data-Driven Decisions**: Replace gut feelings with objective insights
- **Resource Optimization**: Allocate support resources where they're needed most
- **Improved Outcomes**: Help more students succeed academically

## ✨ Features

### 🤖 Machine Learning Pipeline
- Multiple algorithm comparison (Random Forest, Gradient Boosting, Logistic Regression)
- Automated model selection based on performance metrics
- Robust data preprocessing and feature scaling
- Cross-validation for reliable performance estimation

### 📊 Interactive Dashboard
- Real-time risk assessment for individual students
- Intuitive Streamlit-based interface
- Interactive data input with validation
- Visual risk indicators with color-coded alerts

### 🔍 Explainable AI
- SHAP (SHapley Additive exPlanations) integration
- Feature importance visualization
- Transparent decision-making process
- Understand "why" behind each prediction

### 💡 Actionable Insights
- Risk-level specific recommendations
- Personalized intervention strategies
- Evidence-based suggestions for educators
- Early warning system for academic counselors

## 🎬 Demo

![Dashboard Preview](dashboard_preview.png)

*Interactive dashboard showing student risk assessment and SHAP analysis*

## 🛠️ Technology Stack

| Category | Technologies |
|----------|-------------|
| **Language** | Python 3.8+ |
| **ML/Data Science** | scikit-learn, pandas, numpy |
| **Visualization** | matplotlib, seaborn, SHAP |
| **Web Framework** | Streamlit |
| **Model Persistence** | pickle |

## 📥 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager
- Git (for cloning the repository)

### Step-by-Step Setup

1. **Clone the repository**
```bash
git clone https://github.com/Wangary99/student_performance_project.git
cd student_performance_project
```

2. **Create a virtual environment** (recommended)
```bash
# On Windows
python -m venv venv
venv\Scripts\activate

# On macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

3. **Install required packages**
```bash
pip install -r requirements.txt
```

4. **Verify installation**
```bash
python -c "import streamlit; import sklearn; import shap; print('All dependencies installed successfully!')"
```

## 🚀 Usage

### Training the Model

Before using the dashboard, you need to train the model on your data:

```bash
python main.py
```

**What happens during training:**
- Loads and validates the dataset
- Performs data preprocessing and feature engineering
- Trains multiple ML models (Random Forest, Gradient Boosting, Logistic Regression)
- Evaluates performance using accuracy, precision, recall, F1-score
- Selects the best-performing model
- Saves the trained model and scaler to `models/` directory

**Expected output:**
```
Starting Student Performance Prediction Pipeline...
==================================================
Data loaded successfully: (1000, 8)
Training Models...
Best Model: Random Forest
Best Accuracy: 0.8750
Model saved to: models/student_model.pkl
```

### Running the Dashboard

Launch the interactive web dashboard:

```bash
streamlit run dashboard.py
```

The dashboard will open automatically in your default web browser at `http://localhost:8501`

### Making Predictions

**Using the Dashboard:**
1. Enter student information in the sidebar:
   - Demographic details (gender, age)
   - Academic behavior (study hours, attendance)
   - Background factors (parental education, internet access)
   - Extracurricular involvement
2. Click **"Predict Performance"**
3. View the risk assessment, probability scores, and recommendations
4. Explore the SHAP analysis to understand contributing factors

**Using the API (Programmatic):**
```python
from model_trainer import load_model, predict_student_risk

# Load the trained model
model, scaler = load_model()

# Define student data
student_data = {
    'gender': 1,  # 1=Male, 0=Female
    'age': 18,
    'study_hours_per_week': 10,
    'attendance_percentage': 75,
    'parental_education_level': 2,  # 0-4 scale
    'internet_access': 1,  # 1=Yes, 0=No
    'extracurricular_activities': 1  # 1=Yes, 0=No
}

# Get prediction
result = predict_student_risk(model, scaler, student_data)

print(f"Risk Level: {result['risk_level']}")
print(f"Probability: {result['risk_probability']:.2%}")
```

## 📁 Project Structure

```
student_performance_project/
│
├── main.py                      # Main training pipeline and model evaluation
├── dashboard.py                 # Streamlit dashboard application
├── model_trainer.py             # Model loading and prediction utilities
├── requirements.txt             # Python dependencies
├── README.md                    # Project documentation (this file)
│
├── models/                      # Trained models and artifacts
│   ├── student_model.pkl        # Serialized trained model
│   └── scaler.pkl               # Feature scaler for preprocessing
│
├── data/                        # Data directory
│   └── student_data.csv         # Student performance dataset
│
└── notebooks/                   # Jupyter notebooks (optional)
    └── exploratory_analysis.ipynb
```

## 📊 Data Description

### Input Features

The model uses the following features to predict student risk:

| Feature | Type | Description | Values/Range |
|---------|------|-------------|--------------|
| `gender` | Binary | Student's gender | 0 = Female, 1 = Male |
| `age` | Numeric | Student's age | 15-25 years |
| `study_hours_per_week` | Numeric | Weekly study hours outside class | 0-40 hours |
| `attendance_percentage` | Numeric | Percentage of classes attended | 0-100% |
| `parental_education_level` | Ordinal | Highest education level of parents | 0=High School<br>1=Some College<br>2=Bachelor's<br>3=Master's<br>4=PhD |
| `internet_access` | Binary | Access to internet at home | 0 = No, 1 = Yes |
| `extracurricular_activities` | Binary | Participation in extracurriculars | 0 = No, 1 = Yes |

### Target Variable

| Variable | Type | Description | Values |
|----------|------|-------------|--------|
| `at_risk` | Binary | Academic risk status | 0 = On Track<br>1 = At Risk |

### Data Format

Your dataset should be a CSV file with these exact column names. Example:

```csv
gender,age,study_hours_per_week,attendance_percentage,parental_education_level,internet_access,extracurricular_activities,at_risk
1,18,15,85,2,1,1,0
0,19,5,60,1,0,0,1
1,17,20,95,3,1,1,0
```

### Data Quality Requirements

- **No missing values**: The pipeline handles missing data, but clean data yields better results
- **Consistent encoding**: Use the encoding scheme specified above
- **Balanced classes**: Aim for relatively balanced representation of at-risk and on-track students
- **Sufficient samples**: Minimum 500 records recommended, 1000+ ideal

## 📈 Model Performance

Our best-performing model achieves the following metrics on the test set:

| Metric | Score |
|--------|-------|
| **Accuracy** | 87.5% |
| **Precision** | 86.2% |
| **Recall** | 84.8% |
| **F1-Score** | 85.5% |

### Model Comparison

We evaluate multiple algorithms and select the best performer:

| Algorithm | Accuracy | Training Time | Interpretability |
|-----------|----------|---------------|------------------|
| **Random Forest** | **87.5%** | Medium | High ⭐ |
| Gradient Boosting | 86.3% | Slow | Medium |
| Logistic Regression | 82.1% | Fast | Very High |

*Random Forest selected as the default model for optimal balance of performance and interpretability.*

### Risk Level Classification

The model outputs probabilities that are mapped to three risk levels:

| Risk Level | Probability Range | Interpretation |
|------------|-------------------|----------------|
| 🟢 **Low Risk** | 0% - 30% | Student is performing well |
| 🟡 **Medium Risk** | 31% - 60% | Student may need monitoring |
| 🔴 **High Risk** | 61% - 100% | Student requires immediate intervention |

## 🖥️ Dashboard Guide

### Main Interface

The dashboard consists of three main sections:

#### 1. Input Panel (Sidebar)
- **Student Information Form**: Enter demographic and academic data
- **Predict Button**: Trigger the prediction algorithm
- **Clear Button**: Reset all inputs

#### 2. Results Panel (Main Area)
- **Risk Metrics**: Visual indicators showing prediction, probability, and engagement
- **Risk Level Badge**: Color-coded alert (Green/Yellow/Red)
- **SHAP Analysis**: Interactive feature importance chart
- **Key Factors**: Top 3 factors contributing to the prediction

#### 3. Recommendations Panel
- **Risk-Specific Guidance**: Tailored intervention strategies
- **Action Items**: Concrete steps for educators
- **Resource Suggestions**: Links to support materials

### Interpreting SHAP Values

SHAP (SHapley Additive exPlanations) charts show how each feature impacts the prediction:

- **Red bars (positive values)**: Features that increase risk
- **Blue bars (negative values)**: Features that decrease risk
- **Bar length**: Magnitude of the feature's impact

Example interpretation:
```
attendance_percentage: -0.45 (Blue)
→ Higher attendance significantly reduces risk

study_hours_per_week: -0.32 (Blue)
→ More study hours moderately reduce risk

internet_access: +0.15 (Red)
→ Lack of internet access slightly increases risk
```

## 🔌 API Reference

### `load_model()`

Load the trained model and scaler from disk.

```python
from model_trainer import load_model

model, scaler = load_model()
```

**Returns:**
- `model`: Trained scikit-learn classifier
- `scaler`: Fitted StandardScaler object

---

### `predict_student_risk(model, scaler, student_data)`

Predict risk level for a single student.

```python
from model_trainer import predict_student_risk

result = predict_student_risk(model, scaler, student_data)
```

**Parameters:**
- `model`: Trained classifier
- `scaler`: Fitted scaler
- `student_data`: Dictionary or DataFrame with student features

**Returns:**
```python
{
    'prediction': 1,  # 0=On Track, 1=At Risk
    'risk_probability': 0.73,  # 0.0-1.0
    'risk_level': 'High Risk',  # Low/Medium/High Risk
    'all_probabilities': [0.27, 0.73]  # [P(class 0), P(class 1)]
}
```

---

### `batch_predict(model, scaler, students_df)`

Predict risk for multiple students simultaneously.

```python
from model_trainer import batch_predict
import pandas as pd

students_df = pd.read_csv('students.csv')
results = batch_predict(model, scaler, students_df)
```

**Parameters:**
- `model`: Trained classifier
- `scaler`: Fitted scaler
- `students_df`: DataFrame with multiple student records

**Returns:**
- DataFrame with predictions for all students

## 🤝 Contributing

We welcome contributions from the community! Here's how you can help:

### Ways to Contribute

1. **Report Bugs**: Open an issue describing the bug and how to reproduce it
2. **Suggest Features**: Share ideas for new features or improvements
3. **Submit Pull Requests**: Fix bugs or implement new features
4. **Improve Documentation**: Help make the docs clearer and more comprehensive
5. **Share Feedback**: Let us know how you're using the project

### Development Workflow

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Make your changes
4. Add tests if applicable
5. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
6. Push to the branch (`git push origin feature/AmazingFeature`)
7. Open a Pull Request

### Code Style

- Follow PEP 8 style guidelines
- Add docstrings to all functions
- Include type hints where appropriate
- Write clear, descriptive commit messages

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

### MIT License Summary

- ✅ Commercial use
- ✅ Modification
- ✅ Distribution
- ✅ Private use
- ⚠️ Liability and warranty disclaimer

## 📧 Contact

**Project Maintainer**: Wangary99

- GitHub: [@Wangary99](https://github.com/Wangary99)
- Project Link: [https://github.com/Wangary99/student_performance_project](https://github.com/Wangary99/student_performance_project)

## 🙏 Acknowledgments

- **scikit-learn**: For providing robust machine learning algorithms
- **Streamlit**: For the excellent web framework
- **SHAP**: For making AI explainable and transparent
- **Education Community**: For inspiring this project and providing valuable feedback

---

## 🚀 Future Enhancements

We're constantly working to improve the system. Planned features include:

- [ ] Integration with Learning Management Systems (LMS)
- [ ] Multi-language support
- [ ] Advanced visualization dashboards
- [ ] Real-time data streaming and updates
- [ ] Mobile application
- [ ] Automated email alerts for high-risk students
- [ ] Historical trend analysis
- [ ] Cohort comparison features
- [ ] Deep learning model options
- [ ] A/B testing framework for intervention strategies

---

## 📊 Quick Start Checklist

- [ ] Python 3.8+ installed
- [ ] Repository cloned
- [ ] Virtual environment created and activated
- [ ] Dependencies installed (`pip install -r requirements.txt`)
- [ ] Dataset prepared (`student_data.csv`)
- [ ] Model trained (`python main.py`)
- [ ] Dashboard launched (`streamlit run dashboard.py`)
- [ ] First prediction made successfully

---

**Made with ❤️ for better education outcomes**

*If this project helps you or your organization, please consider giving it a ⭐ on GitHub!*
