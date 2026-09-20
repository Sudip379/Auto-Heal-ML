import sys
import os
import sqlite3
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    f1_score,
    accuracy_score,
    recall_score,
    roc_auc_score,
    precision_score
)

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer

import mlflow
import mlflow.sklearn
import mlflow.pyfunc

from mlflow.client import MlflowClient
from mlflow.models import infer_signature


# ============================================================
# PROJECT IMPORTS
# ============================================================

sys.path.append(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

from monitoring.drift_detector import detect_drift


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_NAME = "AutoHealChurnModel"

MIN_NEW_SAMPLES = 500

# Candidate must improve F1 by at least 0.01
MIN_F1_IMPROVEMENT = 0.01

# Secondary metrics are allowed to decrease by at most 3%
SAFETY_TOLERANCE = 0.03

# Training data ratio
HISTORICAL_WEIGHT = 0.70
RECENT_WEIGHT = 0.30

# Holdout percentage
HOLDOUT_SIZE = 0.15

# Reproducibility
RANDOM_STATE = 42


# ============================================================
# 1. FETCH DATA FROM SQLITE
# ============================================================

def fetch_data_from_sql():
    """
    Fetch historical reference data and recent production data
    from the simulated production SQLite database.
    """

    print("Connecting to simulated production database...")

    db_path = os.path.join(
        os.path.dirname(
            os.path.dirname(
                os.path.abspath(__file__)
            )
        ),
        "churn_production.db"
    )

    if not os.path.exists(db_path):
        print(
            f"❌ Error: Database not found at {db_path}. "
            "Run setup_database.py first."
        )
        return None, None

    conn = sqlite3.connect(db_path)

    try:
        reference_df = pd.read_sql(
            "SELECT * FROM historical_logs",
            conn
        )

        recent_df = pd.read_sql(
            "SELECT * FROM recent_production_logs",
            conn
        )

        return reference_df, recent_df

    except Exception as e:
        print(f"❌ Database error: {e}")
        return None, None

    finally:
        conn.close()


# ============================================================
# 2. SCHEMA VALIDATION
# ============================================================

def validate_schema(reference_df, recent_df):
    """
    Validate that recent production data contains the
    required schema.

    Extra columns are removed so both datasets remain aligned.
    """

    reference_cols = set(reference_df.columns)
    recent_cols = set(recent_df.columns)

    missing_in_recent = reference_cols - recent_cols
    extra_in_recent = recent_cols - reference_cols

    # Critical failure
    if missing_in_recent:
        raise ValueError(
            "CRITICAL: Missing columns in recent data: "
            f"{missing_in_recent}"
        )

    # Extra columns are not necessarily fatal.
    if extra_in_recent:
        print(
            f"⚠️ Extra columns detected: "
            f"{extra_in_recent}. Ignoring them."
        )

    # Keep recent data aligned to reference schema
    recent_df = recent_df[
        [col for col in reference_df.columns
         if col in recent_df.columns]
    ].copy()

    print("✅ Schema validation passed.")

    return recent_df


# ============================================================
# 3. CLEAN DATA
# ============================================================

def clean_and_split(df):
    """
    Clean churn dataset and separate X and y.
    """

    df = df.copy()

    # --------------------------------------------------------
    # Remove duplicate customers
    # --------------------------------------------------------

    if "customerID" in df.columns:

        df = df.drop_duplicates(
            subset=["customerID"]
        )

        df = df.drop(
            "customerID",
            axis=1
        )

    elif "customer_id" in df.columns:

        df = df.drop_duplicates(
            subset=["customer_id"]
        )

        df = df.drop(
            "customer_id",
            axis=1
        )

    # --------------------------------------------------------
    # Convert TotalCharges safely
    # --------------------------------------------------------

    if "TotalCharges" in df.columns:

        df["TotalCharges"] = pd.to_numeric(
            df["TotalCharges"],
            errors="coerce"
        )

    # --------------------------------------------------------
    # Validate Churn target
    # --------------------------------------------------------

    valid_values = {
        "Yes",
        "No",
        "1",
        "0",
        1,
        0
    }

    unexpected = (
        set(df["Churn"].dropna().unique())
        - valid_values
    )

    if unexpected:

        raise ValueError(
            "CRITICAL: Unexpected Churn values detected: "
            f"{unexpected}"
        )

    # --------------------------------------------------------
    # Convert target to binary
    # --------------------------------------------------------

    df["Churn"] = df["Churn"].map({
        "Yes": 1,
        "No": 0,
        "1": 1,
        "0": 0,
        1: 1,
        0: 0
    })

    # Remove rows without valid target
    df = df.dropna(
        subset=["Churn"]
    )

    X = df.drop(
        "Churn",
        axis=1
    )

    y = df["Churn"].astype(int)

    return X, y


# ============================================================
# 4. BUILD MODEL PIPELINE
# ============================================================

def build_candidate_pipeline(X_train):
    """
    Build complete preprocessing + Random Forest pipeline.
    """

    numeric_cols = X_train.select_dtypes(
        include=["int64", "float64"]
    ).columns

    categorical_cols = X_train.select_dtypes(
        include=["object", "category"]
    ).columns

    preprocessor = ColumnTransformer(
        transformers=[

            # Numeric features
            (
                "num",

                SimpleImputer(
                    strategy="median"
                ),

                numeric_cols
            ),

            # Categorical features
            (
                "cat",

                Pipeline(
                    steps=[
                        (
                            "imputer",
                            SimpleImputer(
                                strategy="most_frequent"
                            )
                        ),

                        (
                            "onehot",
                            OneHotEncoder(
                                handle_unknown="ignore"
                            )
                        )
                    ]
                ),

                categorical_cols
            )
        ]
    )

    candidate_pipeline = Pipeline(
        steps=[

            (
                "preprocessor",
                preprocessor
            ),

            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=100,
                    max_depth=10,
                    random_state=RANDOM_STATE,
                    class_weight="balanced"
                )
            )
        ]
    )

    return candidate_pipeline


# ============================================================
# 5. EVALUATE MODEL
# ============================================================

def evaluate_model(model, X_test, y_test):
    """
    Calculate all important classification metrics.
    """

    predictions = model.predict(X_test)

    probabilities = model.predict_proba(
        X_test
    )[:, 1]

    return {

        "f1_score": f1_score(
            y_test,
            predictions,
            zero_division=0
        ),

        "precision": precision_score(
            y_test,
            predictions,
            zero_division=0
        ),

        "recall": recall_score(
            y_test,
            predictions,
            zero_division=0
        ),

        "accuracy": accuracy_score(
            y_test,
            predictions
        ),

        "roc_auc": roc_auc_score(
            y_test,
            probabilities
        )
    }


# ============================================================
# 6. PROMOTE CANDIDATE TO CHAMPION
# ============================================================

def promote_to_champion(
    run_id,
    current_champion_version
):
    """
    Register candidate model and assign the @champion alias.

    The previous champion version is stored as a tag
    for rollback.
    """

    client = MlflowClient()

    model_uri = (
        f"runs:/{run_id}/model"
    )

    model_details = mlflow.register_model(
        model_uri=model_uri,
        name=MODEL_NAME
    )

    new_version = model_details.version

    # Store previous champion for rollback
    client.set_model_version_tag(
        MODEL_NAME,
        new_version,
        "previous_champion_version",
        str(current_champion_version)
    )

    # Assign new champion
    client.set_registered_model_alias(
        MODEL_NAME,
        "champion",
        new_version
    )

    print(
        f"🚀 Success: Version {new_version} "
        "promoted to @champion."
    )


# ============================================================
# 7. ROLLBACK
# ============================================================

def execute_rollback():
    """
    Restore the previous champion version using
    the previous_champion_version tag.
    """

    print(
        "\n🚨 INITIATING ROLLBACK SEQUENCE..."
    )

    client = MlflowClient()

    try:

        current_champion = (
            client.get_model_version_by_alias(
                MODEL_NAME,
                "champion"
            )
        )

        previous_version = (
            current_champion.tags.get(
                "previous_champion_version"
            )
        )

        if not previous_version:

            print(
                "❌ Rollback failed: "
                "No previous champion version recorded."
            )

            return

        client.set_registered_model_alias(
            MODEL_NAME,
            "champion",
            previous_version
        )

        print(
            f"✅ Rollback successful. "
            f"Restored version {previous_version}."
        )

    except Exception as e:

        print(
            f"❌ Rollback error: {e}"
        )


# ============================================================
# 8. MAIN ORCHESTRATOR
# ============================================================

def run_orchestrator():

    print(
        "\n================================================"
    )
    print(
        "       AUTOHEAL ML RETRAINING PIPELINE"
    )
    print(
        "================================================"
    )

    # --------------------------------------------------------
    # STEP 1: FETCH DATA
    # --------------------------------------------------------

    print(
        "\n--- STEP 1: Fetching Data ---"
    )

    reference_df, recent_df = (
        fetch_data_from_sql()
    )

    if (
        reference_df is None
        or recent_df is None
    ):
        return

    # --------------------------------------------------------
    # STEP 2: SCHEMA VALIDATION
    # --------------------------------------------------------

    print(
        "\n--- STEP 2: Validating Schema ---"
    )

    try:

        recent_df = validate_schema(
            reference_df,
            recent_df
        )

    except ValueError as e:

        print(f"❌ {e}")
        return

    # --------------------------------------------------------
    # STEP 3: MINIMUM DATA GATE
    # --------------------------------------------------------

    if len(recent_df) < MIN_NEW_SAMPLES:

        print(
            f"⚠️ Not enough new data "
            f"({len(recent_df)} samples). "
            f"Minimum required: {MIN_NEW_SAMPLES}."
        )

        return

    print(
        f"✅ Recent production data: "
        f"{len(recent_df)} samples"
    )

    # --------------------------------------------------------
    # STEP 4: DRIFT DETECTION
    # --------------------------------------------------------

    print(
        "\n--- STEP 3: Checking Data Drift ---"
    )

    try:

        drift_metrics = detect_drift(
            reference_df,
            recent_df
        )

    except Exception as e:

        print(
            f"❌ Drift detection failed: {e}"
        )

        return

    if not drift_metrics["drift_detected"]:

        print(
            "🟢 No significant drift detected."
        )

        print(
            "Retraining skipped."
        )

        return

    print(
        "\n⚠️ DATA DRIFT DETECTED!"
    )

    print(
        "Initiating automated retraining..."
    )

    # --------------------------------------------------------
    # STEP 5: CREATE UNSEEN HOLDOUT
    # --------------------------------------------------------

    print(
        "\n--- STEP 4: Creating Unseen Holdout ---"
    )

    # Keep historical and recent holdouts separate first.
    train_ref_df, holdout_ref_df = (
        train_test_split(
            reference_df,
            test_size=HOLDOUT_SIZE,
            random_state=RANDOM_STATE,
            stratify=reference_df["Churn"]
            if "Churn" in reference_df.columns
            else None
        )
    )

    train_recent_df, holdout_recent_df = (
        train_test_split(
            recent_df,
            test_size=HOLDOUT_SIZE,
            random_state=RANDOM_STATE,
            stratify=recent_df["Churn"]
            if "Churn" in recent_df.columns
            else None
        )
    )

    # Combine both holdouts.
    mixed_holdout_df = pd.concat(
        [
            holdout_ref_df,
            holdout_recent_df
        ],
        ignore_index=True
    )

    X_holdout, y_holdout = (
        clean_and_split(
            mixed_holdout_df
        )
    )

    # --------------------------------------------------------
    # STEP 6: CREATE HISTORICAL + RECENT TRAINING DATA
    # --------------------------------------------------------

    print(
        "\n--- STEP 5: Building Training Dataset ---"
    )

    historical_target = int(
        len(train_recent_df)
        * (
            HISTORICAL_WEIGHT
            / RECENT_WEIGHT
        )
    )

    sample_size = min(
        historical_target,
        len(train_ref_df)
    )

    sampled_ref_df = train_ref_df.sample(
        n=sample_size,
        random_state=RANDOM_STATE,
        replace=False
    )

    mixed_train_df = pd.concat(
        [
            sampled_ref_df,
            train_recent_df
        ],
        ignore_index=True
    )

    # Actual proportions after sampling
    actual_historical_weight = (
        len(sampled_ref_df)
        / len(mixed_train_df)
    )

    actual_recent_weight = (
        len(train_recent_df)
        / len(mixed_train_df)
    )

    print(
        f"Historical training samples: "
        f"{len(sampled_ref_df)}"
    )

    print(
        f"Recent training samples: "
        f"{len(train_recent_df)}"
    )

    print(
        f"Actual historical weight: "
        f"{actual_historical_weight:.2%}"
    )

    print(
        f"Actual recent weight: "
        f"{actual_recent_weight:.2%}"
    )

    # --------------------------------------------------------
    # STEP 7: CLEAN TRAINING DATA
    # --------------------------------------------------------

    X_train, y_train = (
        clean_and_split(
            mixed_train_df
        )
    )

    # --------------------------------------------------------
    # STEP 8: BUILD CANDIDATE PIPELINE
    # --------------------------------------------------------

    print(
        "\n--- STEP 6: Building Candidate Model ---"
    )

    candidate_pipeline = (
        build_candidate_pipeline(
            X_train
        )
    )

    # --------------------------------------------------------
    # STEP 9: TRAIN
    # --------------------------------------------------------

    print(
        "Training candidate pipeline..."
    )

    candidate_pipeline.fit(
        X_train,
        y_train
    )

    # --------------------------------------------------------
    # STEP 10: EVALUATE CANDIDATE
    # --------------------------------------------------------

    print(
        "\n--- STEP 7: Evaluating Candidate ---"
    )

    candidate_metrics = (
        evaluate_model(
            candidate_pipeline,
            X_holdout,
            y_holdout
        )
    )

    # --------------------------------------------------------
    # STEP 11: LOAD CURRENT CHAMPION
    # --------------------------------------------------------

    print(
        "\n--- STEP 8: Loading Current Champion ---"
    )

    client = MlflowClient()

    try:

        champion_version_info = (
            client.get_model_version_by_alias(
                MODEL_NAME,
                "champion"
            )
        )

        champ_version = (
            champion_version_info.version
        )

        champion_model = (
            mlflow.pyfunc.load_model(
                f"models:/{MODEL_NAME}@champion"
            )
        )

        champ_metrics = evaluate_model(
            champion_model,
            X_holdout,
            y_holdout
        )

        print(
            f"Current Champion: v{champ_version}"
        )

    except Exception as e:

        print(
            "⚠️ No existing @champion found."
        )

        print(
            f"Reason: {e}"
        )

        print(
            "Running cold-start evaluation."
        )

        champ_metrics = {
            "f1_score": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "accuracy": 0.0,
            "roc_auc": 0.0
        }

        champ_version = "None"

    # --------------------------------------------------------
    # STEP 12: SHOWDOWN
    # --------------------------------------------------------

    print(
        "\n================================================"
    )

    print(
        f"Champion (v{champ_version}) "
        "vs Candidate"
    )

    print(
        "ON UNSEEN MIXED HOLDOUT DATA"
    )

    print(
        "================================================"
    )

    print(
        f"F1 Score : "
        f"{champ_metrics['f1_score']:.4f} "
        f"vs "
        f"{candidate_metrics['f1_score']:.4f}"
    )

    print(
        f"Precision: "
        f"{champ_metrics['precision']:.4f} "
        f"vs "
        f"{candidate_metrics['precision']:.4f}"
    )

    print(
        f"Recall   : "
        f"{champ_metrics['recall']:.4f} "
        f"vs "
        f"{candidate_metrics['recall']:.4f}"
    )

    print(
        f"Accuracy : "
        f"{champ_metrics['accuracy']:.4f} "
        f"vs "
        f"{candidate_metrics['accuracy']:.4f}"
    )

    print(
        f"ROC-AUC  : "
        f"{champ_metrics['roc_auc']:.4f} "
        f"vs "
        f"{candidate_metrics['roc_auc']:.4f}"
    )

    # --------------------------------------------------------
    # STEP 13: QUALITY GATE
    # --------------------------------------------------------

    f1_improvement = (
        candidate_metrics["f1_score"]
        - champ_metrics["f1_score"]
    )

    gate_passed = (

        # Primary requirement
        f1_improvement
        >= MIN_F1_IMPROVEMENT

        and

        # Safety checks
        candidate_metrics["recall"]
        >= (
            champ_metrics["recall"]
            - SAFETY_TOLERANCE
        )

        and

        candidate_metrics["precision"]
        >= (
            champ_metrics["precision"]
            - SAFETY_TOLERANCE
        )

        and

        candidate_metrics["roc_auc"]
        >= (
            champ_metrics["roc_auc"]
            - SAFETY_TOLERANCE
        )

        and

        candidate_metrics["accuracy"]
        >= (
            champ_metrics["accuracy"]
            - SAFETY_TOLERANCE
        )
    )

    print(
        "\n--- QUALITY GATE ---"
    )

    print(
        f"Required F1 improvement: "
        f"+{MIN_F1_IMPROVEMENT:.2f}"
    )

    print(
        f"Actual F1 improvement: "
        f"{f1_improvement:+.4f}"
    )

    if gate_passed:

        print(
            "✅ QUALITY GATE PASSED"
        )

    else:

        print(
            "❌ QUALITY GATE FAILED"
        )

   # --------------------------------------------------------
    # STEP 14: MLflow LOGGING
    # --------------------------------------------------------

    mlflow.set_tracking_uri("sqlite:///mlflow.db")

    mlflow.set_experiment(
        "AutoHealML_Churn_Experiment"
    )

    with mlflow.start_run(
        run_name="automated_retrain_run"
    ) as run:

        # ----------------------------
        # Parameters
        # ----------------------------

        mlflow.log_param(
            "trigger",
            "data_drift"
        )

        mlflow.log_param(
            "training_source",
            "historical_plus_recent"
        )

        mlflow.log_param(
            "target_historical_weight",
            HISTORICAL_WEIGHT
        )

        mlflow.log_param(
            "target_recent_weight",
            RECENT_WEIGHT
        )

        mlflow.log_param(
            "actual_historical_weight",
            actual_historical_weight
        )

        mlflow.log_param(
            "actual_recent_weight",
            actual_recent_weight
        )

        mlflow.log_param(
            "train_dataset_size",
            len(mixed_train_df)
        )

        mlflow.log_param(
            "holdout_dataset_size",
            len(mixed_holdout_df)
        )

        mlflow.log_param(
            "holdout_size",
            HOLDOUT_SIZE
        )

        mlflow.log_param(
            "min_new_samples",
            MIN_NEW_SAMPLES
        )

        mlflow.log_param(
            "min_f1_improvement",
            MIN_F1_IMPROVEMENT
        )

        mlflow.log_param(
            "safety_tolerance",
            SAFETY_TOLERANCE
        )

        mlflow.log_param(
            "quality_gate_passed",
            gate_passed
        )

        # ----------------------------
        # Drift metrics
        # ----------------------------

        for key, value in (
            drift_metrics.items()
        ):

            if isinstance(value, bool):

                value = int(value)

            mlflow.log_metric(
                f"drift_{key}",
                value
            )

        # ----------------------------
        # Candidate metrics
        # ----------------------------

        for metric_name, value in (
            candidate_metrics.items()
        ):

            mlflow.log_metric(
                f"candidate_{metric_name}",
                value
            )

        # ----------------------------
        # Champion metrics
        # ----------------------------

        for metric_name, value in (
            champ_metrics.items()
        ):

            if metric_name != "version":

                mlflow.log_metric(
                    f"champion_{metric_name}",
                    value
                )

        # ----------------------------
        # Comparison
        # ----------------------------

        mlflow.log_metric(
            "f1_improvement",
            f1_improvement
        )

        # ----------------------------
        # Model signature
        # ----------------------------

        sample_input = X_train.head(3)

        sample_output = (
            candidate_pipeline.predict(
                sample_input
            )
        )

        signature = infer_signature(
            sample_input,
            sample_output
        )

        # ----------------------------
        # Log candidate model
        # ----------------------------

        mlflow.sklearn.log_model(
            sk_model=candidate_pipeline,
            artifact_path="model",
            signature=signature,
            input_example=sample_input,
            serialization_format="cloudpickle"
        )

        # ----------------------------------------------------
        # STEP 15: PROMOTE OR REJECT
        # ----------------------------------------------------

        if gate_passed:

            print(
                "\n🚀 Promoting candidate "
                "to @champion..."
            )

            promote_to_champion(
                run.info.run_id,
                champ_version
            )

        else:

            print(
                "\n🛑 Candidate rejected."
            )

            print(
                "Current champion remains unchanged."
            )

    print(
        "\n================================================"
    )

    print(
        "AUTOHEAL PIPELINE EXECUTION COMPLETE"
    )

    print(
        "================================================"
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    run_orchestrator()