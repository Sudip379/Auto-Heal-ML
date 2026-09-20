import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler # Added StandardScaler
from sklearn.impute import SimpleImputer
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
import pickle
import os

def load_data(filepath):
    print(f"Loading data from {filepath}...")
    return pd.read_csv(filepath)

def clean_base_data(df):
    """Handles basic formatting and strict target validation."""
    print("Preparing base data...")
    
    if 'customerID' in df.columns:
        df = df.drop('customerID', axis=1)
    elif 'customer_id' in df.columns:
        df = df.drop('customer_id', axis=1)
    
    if 'TotalCharges' in df.columns:
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    
    target_col = 'Churn' if 'Churn' in df.columns else 'churn'
    
    # Target Validation: Added '1' and '0' strings to valid targets
    valid_targets = {'Yes', 'No', 1, 0, '1', '0'}
    actual_targets = set(df[target_col].dropna().unique())
    if not actual_targets.issubset(valid_targets):
        raise ValueError(f"Target column contains unexpected values: {actual_targets}. Expected Yes/No or 1/0.")
    
    # Map target safely: Added string numbers to mapping
    df[target_col] = df[target_col].map({'Yes': 1, 'No': 0, 1: 1, 0: 0, '1': 1, '0': 0})
    df = df.dropna(subset=[target_col])
    
    return df, target_col

def build_and_train_pipeline(X_train, y_train, n_estimators, max_depth):
    print("Building Scikit-Learn Pipeline...")
    
    numeric_cols = X_train.select_dtypes(include=['int64', 'float64']).columns
    categorical_cols = X_train.select_dtypes(include=['object', 'category']).columns

    # Upgraded: Fills with 0 (ideal for TotalCharges) and scales the data
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value=0)),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_cols),
            ('cat', categorical_transformer, categorical_cols)
        ])

    # Upgraded: Added class_weight='balanced' to handle imbalanced churn data
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(
            n_estimators=n_estimators, 
            max_depth=max_depth, 
            random_state=42, 
            class_weight='balanced'
        ))
    ])

    print("Training full pipeline (Preprocessing + Model)...")
    pipeline.fit(X_train, y_train)
    return pipeline

def evaluate_model(model, X_test, y_test):
    print("Evaluating pipeline...")
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]
    
    metrics = {
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions),
        "recall": recall_score(y_test, predictions),
        "f1_score": f1_score(y_test, predictions),
        "roc_auc": roc_auc_score(y_test, probabilities)
    }
    
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
        
    return metrics

def save_model(model, output_path):
    print(f"Saving local backup to {output_path}...")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'wb') as f:
        pickle.dump(model, f)

if __name__ == "__main__":
    DATA_PATH = "data/raw/churn.csv"
    MODEL_PATH = "models/model_v1.pkl"
    
    n_estimators = 100
    max_depth = 10
    test_size = 0.2
    
    df = load_data(DATA_PATH)
    df_clean, target_col = clean_base_data(df)
    
    X = df_clean.drop(target_col, axis=1)
    y = df_clean[target_col]
    
    # Maintains class balance in the split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    
    mlflow.set_experiment("AutoHealML_Churn_Experiment")
    
    with mlflow.start_run() as run:
        print(f"Starting MLflow Run ID: {run.info.run_id}")
        
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("test_size", test_size)
        mlflow.log_param("class_weight", "balanced")
        
        full_pipeline = build_and_train_pipeline(X_train, y_train, n_estimators, max_depth)
        metrics = evaluate_model(full_pipeline, X_test, y_test)
        
        for metric_name, value in metrics.items():
            mlflow.log_metric(metric_name, value)
            
        # Infers signature and creates an input example for serving validation
        signature = infer_signature(X_train, full_pipeline.predict(X_train))
        input_example = X_train.head(3)
            
        mlflow.sklearn.log_model(
            sk_model=full_pipeline,
            artifact_path="model",
            registered_model_name="AutoHealChurnModel",
            signature=signature,
            input_example=input_example
        )
        
        save_model(full_pipeline, MODEL_PATH)
        print("✅ Success: Production-ready Pipeline tracked and registered with signature.")