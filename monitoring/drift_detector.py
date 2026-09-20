from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
import warnings

warnings.filterwarnings("ignore")

# If 30% or more feature columns drift,
# the pipeline will trigger retraining.
DRIFT_SHARE_THRESHOLD = 0.30

# These columns should NOT participate in feature drift detection.
# customerID is an identifier, while Churn is the target label.
EXCLUDED_COLUMNS = {"customerID", "customer_id", "Churn", "churn"}


def detect_drift(reference_data, current_data):
    """
    Compare reference data with recent production data.

    Returns a structured dictionary containing:
    - whether drift was detected
    - drift share
    - number of drifted columns
    - total columns checked
    """

    print("Analyzing statistical distributions with Evidently AI...")

    # ---------------------------------------------------------
    # 1. Select only columns useful for feature drift detection
    # ---------------------------------------------------------
    common_columns = [
        col
        for col in reference_data.columns
        if col in current_data.columns
        and col not in EXCLUDED_COLUMNS
    ]

    if not common_columns:
        raise ValueError(
            "No common feature columns available for drift detection."
        )

    reference_features = reference_data[common_columns].copy()
    current_features = current_data[common_columns].copy()

    # ---------------------------------------------------------
    # 2. Build Evidently drift report
    # ---------------------------------------------------------
    report = Report(
        metrics=[
            DataDriftPreset(
                columns=common_columns,
                drift_share=DRIFT_SHARE_THRESHOLD
            )
        ]
    )

    report.run(
        reference_data=reference_features,
        current_data=current_features
    )

    results = report.as_dict()

    # ---------------------------------------------------------
    # 3. Safely find DatasetDriftMetric
    # ---------------------------------------------------------
    drift_result = None

    for metric in results.get("metrics", []):
        if metric.get("metric") == "DatasetDriftMetric":
            drift_result = metric.get("result")
            break

    if drift_result is None:
        raise RuntimeError(
            "Could not extract DatasetDriftMetric "
            "from Evidently report."
        )

    # ---------------------------------------------------------
    # 4. Extract metrics
    # ---------------------------------------------------------
    drift_share = float(
        drift_result.get("share_of_drifted_columns", 0.0)
    )

    drifted_columns = int(
        drift_result.get("number_of_drifted_columns", 0)
    )

    total_columns = int(
        drift_result.get("number_of_columns", 0)
    )

    # Prefer Evidently's own dataset-level decision.
    drift_detected = bool(
        drift_result.get(
            "dataset_drift",
            drift_share >= DRIFT_SHARE_THRESHOLD
        )
    )

    # ---------------------------------------------------------
    # 5. Structured result
    # ---------------------------------------------------------
    metrics = {
        "drift_detected": drift_detected,
        "drift_share": drift_share,
        "drifted_columns": drifted_columns,
        "total_columns": total_columns
    }

    # ---------------------------------------------------------
    # 6. Console output
    # ---------------------------------------------------------
    if drift_detected:
        print(
            f"🚨 Drift detected: "
            f"{drifted_columns}/{total_columns} columns "
            f"({drift_share:.2%})"
        )
    else:
        print(
            f"🟢 No significant drift: "
            f"{drift_share:.2%} of feature columns drifted."
        )

    return metrics