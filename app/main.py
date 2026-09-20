import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import mlflow.sklearn

app = FastAPI(
    title="AutoHealML Prediction Service",
    description="Serves the current production churn model tracked via MLflow.",
    version="1.0"
)

# Global model variable
model = None

class ChurnPredictionRequest(BaseModel):
    tenure: int = Field(..., example=12)
    MonthlyCharges: float = Field(..., example=70.35)
    TotalCharges: float = Field(..., example=840.2)
    contract: str = Field(..., example="Month-to-month")
    payment_method: str = Field(..., example="Electronic check")
    gender: str = Field(..., example="Female")
    age: int = Field(..., example=35)
    support_calls: int = Field(..., example=1)
    # Add other required fields or use a flexible payload approach depending on your strict feature alignment

@app.on_event("startup")
def load_production_model():
    global model
    model_uri = "models:/AutoHealChurnModel/1"
    print(f"Loading model from MLflow Registry: {model_uri}...")
    try:
        model = mlflow.sklearn.load_model(model_uri)
        print("Model loaded successfully into FastAPI.")
    except Exception as e:
        print(f"Warning: Could not load model from registry ({e}). Ensure MLflow tracking server or local store is accessible.")

@app.get("/")
def read_root():
    return {"status": "Healthy", "system": "AutoHealML API"}

@app.post("/predict")
def predict(payload: ChurnPredictionRequest):
    if model is None:
        raise HTTPException(status_code=500, detail="Model is not loaded.")
    
    # Convert input payload to DataFrame
    input_data = pd.DataFrame([payload.dict()])
    
    # Preprocessing pipeline alignment (Ensure it matches training preprocessing steps)
    # For a robust setup, feature engineering functions should be imported from src/preprocessing.py
    input_data['TotalCharges'] = pd.to_numeric(input_data['TotalCharges'], errors='coerce').fillna(0)
    
    # Perform dummy encoding to mirror training data features
    # Note: In production, ensure columns match the exact training feature space schema
    input_encoded = pd.get_dummies(input_data)
    
    try:
        prediction = int(model.predict(input_encoded)[0])
        probability = float(model.predict_proba(input_encoded)[0][1])
        
        return {
            "prediction": prediction,
            "probability": round(probability, 4),
            "model_version": "1"
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")