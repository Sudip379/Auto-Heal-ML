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

MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"

MODEL_URI = (
    f"models:/{MODEL_NAME}@{MODEL_ALIAS}"
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
        # so they should not reach the model.
        # ----------------------------------------------------

        for id_column in [
            "customerID",
            "customer_id"
        ]:

            if id_column in input_data.columns:

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