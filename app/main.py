import os
import sqlite3
from datetime import datetime, timezone

import pandas as pd

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

import mlflow
import mlflow.sklearn

from mlflow import MlflowClient


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_NAME = "AutoHealChurnModel"

MODEL_ALIAS = "champion"

# ISE UPDATE KAREIN: File path ki jagah Docker ka host address use karein
MLFLOW_TRACKING_URI = "http://10.79.157.18:5000"

MODEL_URI = (
    f"models:/{MODEL_NAME}@{MODEL_ALIAS}"
)

DB_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "churn_production.db"
)


# ============================================================
# GLOBAL MODEL STATE
# ============================================================

model = None

loaded_model_version = None


# ============================================================
# LOAD PRODUCTION MODEL
# ============================================================

def load_production_model():
    """
    Load the model currently assigned to the
    @champion alias in MLflow Model Registry.
    """

    global model
    global loaded_model_version

    print(
        f"Loading production model from MLflow: "
        f"{MODEL_URI}"
    )

    # --------------------------------------------------------
    # Connect to local MLflow tracking database
    # --------------------------------------------------------

    mlflow.set_tracking_uri(
        MLFLOW_TRACKING_URI
    )

    client = MlflowClient(
        tracking_uri=MLFLOW_TRACKING_URI
    )

    # --------------------------------------------------------
    # Find the model version currently assigned
    # to the @champion alias
    # --------------------------------------------------------

    champion_info = (
        client.get_model_version_by_alias(
            MODEL_NAME,
            MODEL_ALIAS
        )
    )

    loaded_model_version = (
        champion_info.version
    )

    print(
        f"Current @champion version: "
        f"v{loaded_model_version}"
    )

    # --------------------------------------------------------
    # Load the complete sklearn pipeline
    # --------------------------------------------------------

    model = mlflow.sklearn.load_model(
        MODEL_URI
    )

    print(
        "✅ Production model loaded successfully."
    )


# ============================================================
# PREDICTION LOGGING
# ============================================================

def init_prediction_log_table():
    """
    Create the prediction_logs table if it does not already exist.
    Uses the same churn_production.db that historical_logs and
    recent_production_logs live in (see setup_database.py).
    """

    conn = sqlite3.connect(DB_PATH)

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS prediction_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            customerID TEXT,
            model_name TEXT NOT NULL,
            model_alias TEXT NOT NULL,
            model_version TEXT NOT NULL,
            prediction INTEGER NOT NULL,
            probability REAL
        )
        """
    )

    conn.commit()
    conn.close()


def log_prediction(customer_id, prediction, probability):
    """
    Best-effort write of one prediction to prediction_logs.

    Deliberately never raises: a logging failure must not turn a
    successful prediction into a failed API response.
    """

    try:

        conn = sqlite3.connect(DB_PATH)

        conn.execute(
            """
            INSERT INTO prediction_logs
                (timestamp, customerID, model_name, model_alias,
                 model_version, prediction, probability)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                datetime.now(timezone.utc).isoformat(),
                customer_id,
                MODEL_NAME,
                MODEL_ALIAS,
                str(loaded_model_version),
                prediction,
                probability,
            ),
        )

        conn.commit()
        conn.close()

    except Exception as e:

        print(
            "⚠️ Prediction logging failed "
            f"(prediction was still returned): {e}"
        )


# ============================================================
# FASTAPI LIFESPAN
# ============================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application startup and shutdown lifecycle.

    Startup:
        Load the current @champion MLflow model.

    Shutdown:
        Release the model from memory.
    """

    global model
    global loaded_model_version

    # --------------------------------------------------------
    # STARTUP
    # --------------------------------------------------------

    try:

        init_prediction_log_table()

    except Exception as e:

        print(
            "⚠️ Could not initialise prediction_logs table: "
            f"{e}"
        )

    try:

        load_production_model()

    except Exception as e:

        model = None
        loaded_model_version = None

        print(
            "⚠️ FastAPI started, but the production "
            f"model could not be loaded: {e}"
        )

    # --------------------------------------------------------
    # Application runs here
    # --------------------------------------------------------

    yield

    # --------------------------------------------------------
    # SHUTDOWN
    # --------------------------------------------------------

    print(
        "Shutting down AutoHealML API..."
    )

    model = None
    loaded_model_version = None

    print(
        "✅ Model resources released."
    )


# ============================================================
# FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="AutoHealML Prediction Service",

    description=(
        "Production prediction API for the current "
        "@champion churn model managed by MLflow."
    ),

    version="1.0",

    lifespan=lifespan
)


# ============================================================
# REQUEST SCHEMA
# ============================================================

class ChurnPredictionRequest(BaseModel):
    """
    Prediction request.

    Send the feature values required by the trained
    churn model.

    Do NOT include the Churn target.
    """

    features: dict = Field(
        ...,
        description=(
            "Feature values required by the trained "
            "churn model. Do not include Churn."
        )
    )


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/")
def read_root():
    """
    Basic API health check.
    """

    return {
        "status": "Healthy",
        "system": "AutoHealML API",
        "model": MODEL_NAME,
        "alias": MODEL_ALIAS,
        "model_version": (
            str(loaded_model_version)
            if loaded_model_version is not None
            else "Unavailable"
        )
    }


# ============================================================
# MODEL STATUS
# ============================================================

@app.get("/model")
def model_status():
    """
    Return the currently loaded production model.
    """

    if model is None:

        raise HTTPException(
            status_code=503,
            detail="Production model is not loaded."
        )

    return {
        "model_name": MODEL_NAME,
        "alias": MODEL_ALIAS,
        "version": str(
            loaded_model_version
        ),
        "status": "loaded"
    }


# ============================================================
# PREDICTION ENDPOINT
# ============================================================

@app.post("/predict")
def predict(
    payload: ChurnPredictionRequest
):
    """
    Generate churn prediction using the
    currently loaded @champion model.
    """

    # --------------------------------------------------------
    # Check model availability
    # --------------------------------------------------------

    if model is None:

        raise HTTPException(
            status_code=503,
            detail="Production model is not loaded."
        )

    try:

        # ----------------------------------------------------
        # Convert incoming JSON features to DataFrame
        # ----------------------------------------------------

        input_data = pd.DataFrame(
            [payload.features]
        )

        # ----------------------------------------------------
        # Remove target if accidentally supplied
        # ----------------------------------------------------

        if "Churn" in input_data.columns:

            input_data = input_data.drop(
                "Churn",
                axis=1
            )

        # ----------------------------------------------------
        # Remove customer identifier if supplied
        #
        # The training pipeline removes these identifiers,
        # so they should not reach the model. Captured first
        # so it can still be recorded in prediction_logs.
        # ----------------------------------------------------

        customer_id_value = None

        for id_column in [
            "customerID",
            "customer_id"
        ]:

            if id_column in input_data.columns:

                customer_id_value = input_data[id_column].iloc[0]

                input_data = input_data.drop(
                    id_column,
                    axis=1
                )

        # ----------------------------------------------------
        # Convert TotalCharges safely
        # ----------------------------------------------------

        if "TotalCharges" in input_data.columns:

            input_data["TotalCharges"] = (
                pd.to_numeric(
                    input_data["TotalCharges"],
                    errors="coerce"
                )
            )

        # ----------------------------------------------------
        # MODEL PREDICTION
        #
        # IMPORTANT:
        #
        # Do NOT use pd.get_dummies() here.
        #
        # The MLflow model contains the complete
        # sklearn Pipeline used during training:
        #
        # Raw Data
        #     ↓
        # SimpleImputer
        #     ↓
        # OneHotEncoder
        #     ↓
        # RandomForest
        #
        # Therefore the exact same preprocessing
        # is automatically applied during inference.
        # ----------------------------------------------------

        prediction = int(
            model.predict(
                input_data
            )[0]
        )

        # ----------------------------------------------------
        # Prediction probability
        # ----------------------------------------------------

        probability = None

        if hasattr(
            model,
            "predict_proba"
        ):

            probability = float(
                model.predict_proba(
                    input_data
                )[0][1]
            )

        # ----------------------------------------------------
        # Log this prediction (best-effort, never blocks
        # the response)
        # ----------------------------------------------------

        log_prediction(
            customer_id_value,
            prediction,
            probability
        )

        # ----------------------------------------------------
        # Return prediction response
        # ----------------------------------------------------

        return {

            "prediction": prediction,

            "churn": (
                "Yes"
                if prediction == 1
                else "No"
            ),

            "probability": (
                round(
                    probability,
                    4
                )
                if probability is not None
                else None
            ),

            "model_name": MODEL_NAME,

            "model_alias": MODEL_ALIAS,

            "model_version": str(
                loaded_model_version
            )
        }

    except Exception as e:

        raise HTTPException(
            status_code=400,

            detail=(
                "Prediction failed. "
                "Check that all required model "
                f"features are provided. Error: {str(e)}"
            )
        )