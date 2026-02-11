import pandas as pd
import numpy as np

def create_master_table(data_path):
    print("--- Starting Data Ingestion & Merging ---")
    
    # 1. Load Data
    info = pd.read_csv(f"{data_path}/studentInfo.csv")
    registration = pd.read_csv(f"{data_path}/studentRegistration.csv")
    assessments = pd.read_csv(f"{data_path}/assessments.csv")
    student_assessment = pd.read_csv(f"{data_path}/studentAssessment.csv")
    vle_interactions = pd.read_csv(f"{data_path}/studentVle.csv")

    # 2. Merge Demographics & Registration
    df = pd.merge(info, registration, on=['id_student', 'code_module', 'code_presentation'], how='inner')

    # 3. Aggregate Assessment Scores
    student_scores = student_assessment.groupby('id_student')['score'].mean().reset_index()
    student_scores.rename(columns={'score': 'avg_score'}, inplace=True)
    
    # 4. Aggregate VLE (Big Data Clickstream)
    student_engagement = vle_interactions.groupby('id_student')['sum_click'].sum().reset_index()
    student_engagement.rename(columns={'sum_click': 'total_clicks'}, inplace=True)

    # 5. Final Master Join
    df = pd.merge(df, student_scores, on='id_student', how='left')
    df = pd.merge(df, student_engagement, on='id_student', how='left')

    # --- NEW: CRITICAL DATA CLEANING ---
    # Fill numeric columns with 0
    df['avg_score'] = df['avg_score'].fillna(0)
    df['total_clicks'] = df['total_clicks'].fillna(0)
    df['date_registration'] = df['date_registration'].fillna(0)
    
    # Fill categorical columns (like imd_band) with a placeholder
    df['imd_band'] = df['imd_band'].fillna('Unknown')
    
    # Drop rows that don't have a final result (cannot predict if we don't have the answer)
    df = df.dropna(subset=['final_result'])

    print(f"Master Table Created! Shape: {df.shape}")
    return df