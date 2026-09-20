import os

from evidently.report import Report
from evidently.metric_preset import DataDriftPreset


# ============================================================
# CONFIGURATION
# ============================================================

# Percentage of monitored features that must drift
# before the system considers the dataset significantly drifted.
DRIFT_SHARE_THRESHOLD = 0.30


# Columns that should NOT participate in drift analysis.
# customerID is an identifier, while Churn is the target.
EXCLUDED_COLUMNS = {
    "customerID",
    "customer_id",
    "Churn"
}


# ============================================================
# DRIFT DETECTION
# ============================================================

def detect_drift(reference_df, current_df):
    """
    Compare historical reference data against recent production
    data using Evidently Data Drift analysis.

    Returns:
        {
            "drift_share": float,
            "number_of_columns": int,
            "number_of_drifted_columns": int,
            "drift_detected": bool
        }
    """

    print(
        "Analyzing statistical distributions with Evidently AI..."
    )

    # --------------------------------------------------------
    # 1. Remove identifiers and target variable
    # --------------------------------------------------------

    reference_monitor = reference_df.drop(
        columns=list(EXCLUDED_COLUMNS),
        errors="ignore"
    ).copy()

    current_monitor = current_df.drop(
        columns=list(EXCLUDED_COLUMNS),
        errors="ignore"
    ).copy()

    # --------------------------------------------------------
    # 2. Ensure both datasets use the same columns
    # --------------------------------------------------------

    common_columns = [
        column
        for column in reference_monitor.columns
        if column in current_monitor.columns
    ]

    reference_monitor = reference_monitor[
        common_columns
    ]

    current_monitor = current_monitor[
        common_columns
    ]

    if not common_columns:

        raise ValueError(
            "No common feature columns available for drift detection."
        )

    # --------------------------------------------------------
    # 3. Run Evidently Data Drift analysis
    # --------------------------------------------------------

    drift_report = Report(
        metrics=[
            DataDriftPreset()
        ]
    )

    drift_report.run(
        reference_data=reference_monitor,
        current_data=current_monitor
    )

    # --------------------------------------------------------
    # 4. Save HTML report
    # --------------------------------------------------------

    os.makedirs(
        "monitoring",
        exist_ok=True
    )

    report_path = os.path.join(
        "monitoring",
        "drift_report.html"
    )

    drift_report.save_html(
        report_path
    )

    # --------------------------------------------------------
    # 5. Extract dataset drift result safely
    # --------------------------------------------------------

    report_dict = drift_report.as_dict()

    dataset_drift = None

    for metric in report_dict.get("metrics", []):

        result = metric.get(
            "result",
            {}
        )

        if "dataset_drift" in result:

            dataset_drift = result

            break

    if dataset_drift is None:

        raise ValueError(
            "Could not extract dataset drift metrics from Evidently report."
        )

    # --------------------------------------------------------
    # 6. Extract metrics
    # --------------------------------------------------------

    drift_share = float(
        dataset_drift.get(
            "share_of_drifted_columns",
            0.0
        )
    )

    number_of_columns = int(
        dataset_drift.get(
            "number_of_columns",
            len(common_columns)
        )
    )

    number_of_drifted_columns = int(
        dataset_drift.get(
            "number_of_drifted_columns",
            0
        )
    )

    # --------------------------------------------------------
    # 7. Apply OUR explicit project threshold
    # --------------------------------------------------------

    drift_detected = (
        drift_share
        >= DRIFT_SHARE_THRESHOLD
    )

    drift_metrics = {

        "drift_share":
            drift_share,

        "number_of_columns":
            number_of_columns,

        "number_of_drifted_columns":
            number_of_drifted_columns,

        "drift_detected":
            drift_detected
    }

    # --------------------------------------------------------
    # 8. Console output
    # --------------------------------------------------------

    if drift_detected:

        print(
            f"🚨 Drift detected: "
            f"{number_of_drifted_columns}/"
            f"{number_of_columns} columns "
            f"({drift_share * 100:.2f}%)"
        )

        print(
            f"Threshold: "
            f"{DRIFT_SHARE_THRESHOLD * 100:.0f}%"
        )

    else:

        print(
            f"🟢 No significant drift detected: "
            f"{number_of_drifted_columns}/"
            f"{number_of_columns} columns "
            f"({drift_share * 100:.2f}%)"
        )

    print(
        f"📄 Evidently report saved to: "
        f"{report_path}"
    )

    return drift_metrics