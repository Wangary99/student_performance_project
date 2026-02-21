---
title: Student Performance Early Warning System
emoji: 🎓
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: "6.6.0"
python_version: "3.13"
app_file: app.py
pinned: false
---

# 🎓 Student Performance Early Warning System

An intelligent early warning system that uses machine learning to identify at-risk students and provide explainable, data-driven insights for timely academic interventions.

## 🎯 Overview

This project addresses a critical challenge in education: **early identification of students who may need additional support**. By analyzing behavioral, demographic, and academic factors, the system predicts whether a student is likely to pass or be at risk — and explains *why* using SHAP (Explainable AI).

## ✨ Features

- 🤖 **Two ML Models**: Random Forest and XGBoost for comparison
- 📊 **Interactive Gradio Dashboard**: Lecturers enter a Student ID and get instant predictions
- 🔍 **SHAP Explainability**: Waterfall chart showing the top factors behind each prediction
- ⚖️ **Class Imbalance Handling**: SMOTE applied during training
- 📈 **Validation Report**: F1-Score, Recall, Precision, and K-Fold Cross Validation comparison

## 🛠️ Technology Stack

| Category | Technologies |
|----------|-------------|
| Language | Python 3.13 |
| ML Libraries | scikit-learn, XGBoost, imbalanced-learn |
| Explainability | SHAP |
| Dashboard | Gradio |
| Data Processing | pandas, numpy |
| Model Persistence | pickle, joblib |

## 🚀 Usage

### Step 1 — Train the models
```bash
python main.py
```
This will generate all model files and save a validation report to `reports/validation_report.txt`

### Step 2 — Launch the dashboard
```bash
python app.py
```

### Step 3 — Make a prediction
1. Enter a Student ID (e.g. `11391`)
2. Select a model (Random Forest or XGBoost)
3. Click Submit
4. View the prediction, risk score, and SHAP explanation chart

## 📁 Project Structure
```
student_performance_project/
│
├── app.py                        # Gradio dashboard (main app)
├── main.py                       # Training pipeline entry point
├── requirements.txt              # Dependencies
├── README.md                     # This file
│
├── src/
│   ├── preprocessing.py          # Data ingestion and merging
│   ├── feature_eng.py            # Feature engineering and SMOTE
│   └── model_trainer.py          # Model training, K-Fold, metrics
│
├── models/
│   ├── student_model.pkl         # Trained Random Forest model
│   ├── xgboost_model.ubj         # Trained XGBoost model
│   ├── scaler.pkl                # Feature scaler
│   └── categorical_encoders.pkl  # Label encoders
│
├── data/
│   └── master_student_data.csv   # Processed student dataset
│
└── reports/
    └── validation_report.txt     # Model comparison report
```

## 📈 Model Performance

| Metric | Random Forest | XGBoost |
|--------|--------------|---------|
| Accuracy | 80% | 75% |
| Evaluated using | K-Fold Cross Validation | K-Fold Cross Validation |

Full detailed metrics available in `reports/validation_report.txt`

## 🔍 Risk Level Classification

| Risk Level | Probability Range |
|------------|------------------|
| 🟢 Low Risk | 0% - 30% |
| 🟡 Medium Risk | 31% - 60% |
| 🔴 High Risk | 61% - 100% |

## 📧 Contact

**Author**: Ann Muthoni Wangari
- GitHub: [@Wangary99](https://github.com/Wangary99)

---
**Made with ❤️ for better education outcomes**
```