import pandas as pd
import numpy as np

def create_master_table(data_path):
    print("--- Phase 1: Ingesting & Merging Data ---")
    
    # Load primary OULAD tables
    info = pd.read_csv(f"{data_path}/studentInfo.csv")
    registration = pd.read_csv(f"{data_path}/studentRegistration.csv")
    student_assessment = pd.read_csv(f"{data_path}/studentAssessment.csv")
    vle_interactions = pd.read_csv(f"{data_path}/studentVle.csv")

    # Merge demographics with registration
    df = pd.merge(info, registration, on=['id_student', 'code_module', 'code_presentation'], how='inner')

    # Aggregate behavioral data (Big Data component)
    student_scores = student_assessment.groupby('id_student')['score'].mean().reset_index().rename(columns={'score': 'avg_score'})
    student_engagement = vle_interactions.groupby('id_student')['sum_click'].sum().reset_index().rename(columns={'sum_click': 'total_clicks'})
    
    df = pd.merge(df, student_scores, on='id_student', how='left')
    df = pd.merge(df, student_engagement, on='id_student', how='left')

    # CRITICAL: Fill missing values to prevent XGBoost and SMOTE crashes
    df['avg_score'] = df['avg_score'].fillna(0)
    df['total_clicks'] = df['total_clicks'].fillna(0)
    df['date_registration'] = df['date_registration'].fillna(0)
    df['imd_band'] = df['imd_band'].fillna('Unknown') # Handle categorical NaNs
    
    # Remove records missing a final result (target)
    df = df.dropna(subset=['final_result'])
    
    print(f"Master Table Created! Shape: {df.shape}")
    return df