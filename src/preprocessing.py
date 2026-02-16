import pandas as pd
import numpy as np

def create_master_table(data_path):
    print("--- Phase 1: Ingesting Data (Memory-Safe Mode) ---")
    
    info = pd.read_csv(f"{data_path}/studentInfo.csv")
    registration = pd.read_csv(f"{data_path}/studentRegistration.csv")
    student_assessment = pd.read_csv(f"{data_path}/studentAssessment.csv")

    # FIX: Process studentVle.csv in chunks to prevent ArrayMemoryError
    print("Aggregating Clickstream (10M+ rows)...")
    vle_agg = pd.DataFrame()
    chunk_iter = pd.read_csv(f"{data_path}/studentVle.csv", usecols=['id_student', 'sum_click'], chunksize=500000)
    for chunk in chunk_iter:
        aggregated_chunk = chunk.groupby('id_student')['sum_click'].sum().reset_index()
        vle_agg = pd.concat([vle_agg, aggregated_chunk]).groupby('id_student')['sum_click'].sum().reset_index()

    # Merge demographics with registration
    df = pd.merge(info, registration, on=['id_student', 'code_module', 'code_presentation'], how='inner')

    # Merge Assessment Scores
    student_scores = student_assessment.groupby('id_student')['score'].mean().reset_index().rename(columns={'score': 'avg_score'})
    df = pd.merge(df, student_scores, on='id_student', how='left')
    
    # Merge VLE Clickstream
    df = pd.merge(df, vle_agg.rename(columns={'sum_click': 'total_clicks'}), on='id_student', how='left')

    # Fill NaNs to prevent model crashes (Fix for Turn 11 error)
    df['avg_score'] = df['avg_score'].fillna(0)
    df['total_clicks'] = df['total_clicks'].fillna(0)
    df['date_registration'] = df['date_registration'].fillna(0)
    df['imd_band'] = df['imd_band'].fillna('Unknown')
    
    return df.dropna(subset=['final_result'])