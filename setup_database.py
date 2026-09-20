import pandas as pd
import sqlite3
import os
import numpy as np

print("Building Mastery-Level SQL Database...")

db_path = "churn_production.db"
conn = sqlite3.connect(db_path)

try:
    df = pd.read_csv("data/raw/churn.csv")
except FileNotFoundError:
    print("❌ Error: Could not find data/raw/churn.csv. Check your path.")
    exit()

if 'TotalCharges' in df.columns:
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce').fillna(0)

reference_df = df.sample(frac=0.7, random_state=42)
current_batch_df = df.drop(reference_df.index).copy()

print("Injecting MASSIVE artificial drift to force retraining...")

# 1. TotalCharges & MonthlyCharges go up
if 'TotalCharges' in current_batch_df.columns:
    current_batch_df['TotalCharges'] = current_batch_df['TotalCharges'] * 1.5
if 'MonthlyCharges' in current_batch_df.columns:
    current_batch_df['MonthlyCharges'] = current_batch_df['MonthlyCharges'] * 1.3

# 2. Tenure drops (simulating a flood of brand new users)
if 'tenure' in current_batch_df.columns:
    current_batch_df['tenure'] = current_batch_df['tenure'].apply(lambda x: max(1, int(x * 0.3)))

# 3. Massive shift in categorical preferences
if 'PaymentMethod' in current_batch_df.columns:
    current_batch_df['PaymentMethod'] = 'Electronic check'
if 'InternetService' in current_batch_df.columns:
    current_batch_df['InternetService'] = 'Fiber optic'
if 'PaperlessBilling' in current_batch_df.columns:
    current_batch_df['PaperlessBilling'] = 'Yes'

# 4. Concept Drift: Force a high churn rate among these new profiles
if 'Contract' in current_batch_df.columns and 'Churn' in current_batch_df.columns:
    mask = (current_batch_df['Contract'] == 'Month-to-month') & (np.random.rand(len(current_batch_df)) < 0.6)
    current_batch_df.loc[mask, 'Churn'] = 'Yes'

reference_df.to_sql("historical_logs", conn, if_exists="replace", index=False)
current_batch_df.to_sql("recent_production_logs", conn, if_exists="replace", index=False)

conn.close()
print(f"✅ Real-world database created at: {os.path.abspath(db_path)}")
print("🚀 MASSIVE Drift injected! Run pipelines/retrain.py again.")