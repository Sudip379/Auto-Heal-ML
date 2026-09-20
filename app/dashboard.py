import streamlit as st
import pandas as pd
import mlflow

from mlflow.client import MlflowClient
from pathlib import Path


# ============================================================
# 1. UI CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AutoHealML Control Center",
    page_icon="⚡",
    layout="wide"
)

st.title(
    "⚡ AutoHealML: Drift-Aware Continuous Training System"
)

st.markdown(
    "Automated Monitoring, Retraining, and Model Promotion Control Panel"
)

st.markdown("---")


# ============================================================
# 2. CONFIGURATION
# ============================================================

MLFLOW_DB_URI = (
    "sqlite:///mlflow.db"
)

MODEL_NAME = (
    "AutoHealChurnModel"
)

EXPERIMENT_NAME = (
    "AutoHealML_Churn_Experiment"
)


# ============================================================
# 3. DEFAULT VALUES
# ============================================================

champ_version = "None"
champ_f1 = 0.0
champ_f1_improvement = 0.0

champ_run_id = "N/A"
champ_training_source = "N/A"
prev_champ_version = "None"

champ_validation_status = "N/A"

drift_share = 0.0
drifted_cols = 0
total_cols = 0
drift_detected = False

last_trigger = "N/A"

train_samples = 0
hist_samples = 0
recent_samples = 0

historical_holdout_samples = 0
recent_holdout_samples = 0

gate_passed = False

recent_recall_diff = 0.0
recent_roc_auc_diff = 0.0

historical_f1_change = 0.0
historical_recall_diff = 0.0
historical_roc_auc_diff = 0.0


# ============================================================
# 4. MLFLOW CONNECTION
# ============================================================

try:

    mlflow.set_tracking_uri(
        MLFLOW_DB_URI
    )

    client = MlflowClient(
        tracking_uri=MLFLOW_DB_URI
    )

    # --------------------------------------------------------
    # A. Champion
    # --------------------------------------------------------

    champion_info = (
        client.get_model_version_by_alias(
            MODEL_NAME,
            "champion"
        )
    )

    champ_version = (
        f"v{champion_info.version}"
    )

    champ_run_id = (
        champion_info.run_id
    )

    prev_champ_version = (
        champion_info.tags.get(
            "previous_champion_version",
            "None"
        )
    )

    if prev_champ_version != "None":

        prev_champ_version = (
            f"v{prev_champ_version}"
        )

    champ_run = client.get_run(
        champ_run_id
    )

    champ_f1 = (
        champ_run.data.metrics.get(
            "candidate_recent_f1_score",
            champ_run.data.metrics.get(
                "candidate_f1_score",
                0.0
            )
        )
    )

    champ_f1_improvement = (
        champ_run.data.metrics.get(
            "f1_improvement",
            0.0
        )
    )

    champ_training_source = (
        champ_run.data.params.get(
            "training_source",
            "N/A"
        )
    )

    champ_validation_status = (
        champ_run.data.params.get(
            "validation_status",
            champion_info.tags.get(
                "validation_status",
                "N/A"
            )
        )
        .upper()
    )

    # --------------------------------------------------------
    # B. Latest automated retraining run
    # --------------------------------------------------------

    experiment = (
        client.get_experiment_by_name(
            EXPERIMENT_NAME
        )
    )

    if experiment:

        latest_runs = client.search_runs(

            experiment_ids=[
                experiment.experiment_id
            ],

            filter_string=(
                "tags.mlflow.runName = "
                "'automated_retrain_run'"
            ),

            order_by=[
                "attributes.start_time DESC"
            ],

            max_results=1
        )

        if latest_runs:

            latest_run = (
                latest_runs[0]
            )

            # ------------------------------------------------
            # Drift
            # ------------------------------------------------

            drift_share = (
                latest_run.data.metrics.get(
                    "drift_drift_share",
                    0.0
                )
            )

            drifted_cols = int(
                latest_run.data.metrics.get(
                    "drift_number_of_drifted_columns",
                    0
                )
            )

            total_cols = int(
                latest_run.data.metrics.get(
                    "drift_number_of_columns",
                    0
                )
            )

            drift_detected = bool(
                latest_run.data.metrics.get(
                    "drift_drift_detected",
                    0
                )
            )

            # ------------------------------------------------
            # Retraining information
            # ------------------------------------------------

            last_trigger = (
                latest_run.data.params.get(
                    "trigger",
                    "N/A"
                )
            )

            train_samples = int(
                latest_run.data.params.get(
                    "total_train_samples",
                    0
                )
            )

            hist_samples = int(
                latest_run.data.params.get(
                    "historical_train_samples",
                    0
                )
            )

            recent_samples = int(
                latest_run.data.params.get(
                    "recent_train_samples",
                    0
                )
            )

            historical_holdout_samples = int(
                latest_run.data.params.get(
                    "historical_holdout_samples",
                    0
                )
            )

            recent_holdout_samples = int(
                latest_run.data.params.get(
                    "recent_holdout_samples",
                    0
                )
            )

            gate_passed_str = str(
                latest_run.data.params.get(
                    "quality_gate_passed",
                    "False"
                )
            )

            gate_passed = (
                gate_passed_str.lower()
                == "true"
            )

            # ------------------------------------------------
            # Recent production changes
            # ------------------------------------------------

            recent_recall_diff = (
                latest_run.data.metrics.get(
                    "recent_recall_change",
                    0.0
                )
            )

            recent_roc_auc_diff = (
                latest_run.data.metrics.get(
                    "recent_roc_auc_change",
                    0.0
                )
            )

            # ------------------------------------------------
            # Historical protection
            # ------------------------------------------------

            historical_f1_change = (
                latest_run.data.metrics.get(
                    "historical_f1_change",
                    0.0
                )
            )

            historical_recall_diff = (
                latest_run.data.metrics.get(
                    "historical_recall_change",
                    0.0
                )
            )

            historical_roc_auc_diff = (
                latest_run.data.metrics.get(
                    "historical_roc_auc_change",
                    0.0
                )
            )


except Exception as e:

    st.warning(
        "Note: Some MLflow data is unavailable "
        f"(Database may be empty). Error: {e}"
    )


# ============================================================
# 5. TOP KPIs
# ============================================================

col1, col2, col3 = st.columns(3)


with col1:

    st.metric(
        label="Champion Model",

        value=(
            f"{champ_version} 🟢"
            if champ_version != "None"
            else "None"
        ),

        delta="Active Production"
    )


with col2:

    f1_delta = (

        f"+{champ_f1_improvement * 100:.2f}%"
        if champ_f1_improvement > 0
        else
        f"{champ_f1_improvement * 100:.2f}%"
    )

    st.metric(
        label="Production F1",

        value=(
            f"{champ_f1 * 100:.1f}%"
        ),

        delta=f1_delta
    )


with col3:

    drift_status_icon = (
        "⚠️"
        if drift_detected
        else
        "🟢"
    )

    st.metric(
        label="Data Drift",

        value=(
            f"{drift_share * 100:.1f}% "
            f"{drift_status_icon}"
        ),

        delta=(
            f"{drifted_cols}/{total_cols} features"
        ),

        delta_color=(
            "inverse"
            if drift_detected
            else
            "normal"
        )
    )


st.markdown("---")


# ============================================================
# 6. MAIN DASHBOARD
# ============================================================

left_col, right_col = st.columns(2)


# ============================================================
# LEFT COLUMN
# ============================================================

with left_col:

    # --------------------------------------------------------
    # MODEL REGISTRY LINEAGE
    # --------------------------------------------------------

    st.subheader(
        "📊 MODEL REGISTRY LINEAGE"
    )

    try:

        versions = client.search_model_versions(
            f"name='{MODEL_NAME}'"
        )

        lineage_data = []

        for version in sorted(
            versions,
            key=lambda x: int(x.version),
            reverse=True
        ):

            is_champion = (
                version.aliases
                and
                "champion"
                in version.aliases
            )

            version_run = client.get_run(
                version.run_id
            )

            version_f1 = (
                version_run.data.metrics.get(
                    "candidate_recent_f1_score",
                    version_run.data.metrics.get(
                        "candidate_f1_score",
                        0.0
                    )
                )
            )

            version_source = (
                version_run.data.params.get(
                    "training_source",
                    "historical"
                )
            )

            status_label = (
                "🟢 Champion"
                if is_champion
                else
                "📁 Archived"
            )

            if (
                not is_champion
                and
                f"v{version.version}"
                == prev_champ_version
            ):

                status_label = (
                    "📁 Previous"
                )

            elif version.version == "1":

                status_label = (
                    "📁 Baseline"
                )

            lineage_data.append(
                {
                    "Version":
                        f"v{version.version}",

                    "Status":
                        status_label,

                    "Training Source":
                        version_source
                        .replace(
                            "_plus_",
                            " + "
                        )
                        .title(),

                    "F1 Score":
                        f"{version_f1 * 100:.1f}%"
                }
            )

        if lineage_data:

            st.dataframe(
                pd.DataFrame(
                    lineage_data
                ),
                use_container_width=True,
                hide_index=True
            )

    except Exception as e:

        st.info(
            f"Lineage data unavailable: {e}"
        )

    # --------------------------------------------------------
    # MODEL DETAILS
    # --------------------------------------------------------

    st.markdown(
        "<br>",
        unsafe_allow_html=True
    )

    st.subheader(
        "🧬 MODEL DETAILS"
    )

    st.markdown(
        f"""
- **Champion**: `{champ_version}`
- **Previous Champion**: `{prev_champ_version}`
- **MLflow Run**: `{champ_run_id}`
- **Training Source**: `{champ_training_source.replace("_plus_", " + ").title()}`
- **Validation**: `{champ_validation_status}`
"""
    )


# ============================================================
# RIGHT COLUMN
# ============================================================

with right_col:

    # --------------------------------------------------------
    # DRIFT MONITORING
    # --------------------------------------------------------

    st.subheader(
        "🔍 DRIFT MONITORING"
    )

    st.markdown(
        f"""
- **Drift Share**: `{drift_share * 100:.1f}%`
- **Drifted Features**: `{drifted_cols} / {total_cols}`
- **Threshold**: `30.0%`
- **Status**: {
    "🚨 **RETRAINING TRIGGERED**"
    if drift_detected
    else
    "🟢 **STABLE**"
}
"""
    )

    # --------------------------------------------------------
    # RETRAINING
    # --------------------------------------------------------

    st.markdown(
        "<br>",
        unsafe_allow_html=True
    )

    st.subheader(
        "🤖 RETRAINMENT"
    )

    gate_status = (
        "✅ PASSED"
        if gate_passed
        else
        "❌ FAILED"
    )

    st.markdown(
        f"""
- **Last Trigger**: `{last_trigger.replace("_", " ").title()}`
- **Training Samples**: `{train_samples:,}`
  - *Historical Samples*: `{hist_samples:,}`
  - *Recent Samples*: `{recent_samples:,}`
- **Historical Holdout**: `{historical_holdout_samples:,}`
- **Recent Holdout**: `{recent_holdout_samples:,}`
- **Quality Gate**: {gate_status}
- **Recent F1 Improvement**: `{champ_f1_improvement * 100:+.2f}%`
- **Recent Recall Change**: `{recent_recall_diff * 100:+.2f}%`
- **Recent ROC-AUC Change**: `{recent_roc_auc_diff * 100:+.2f}%`
"""
    )


st.markdown("---")


# ============================================================
# 7. SAFETY VALIDATION
# ============================================================

st.subheader(
    "🛡️ SAFETY VALIDATION"
)

safety_col1, safety_col2, safety_col3 = (
    st.columns(3)
)


with safety_col1:

    st.metric(
        "Historical F1 Change",
        f"{historical_f1_change * 100:+.2f}%"
    )


with safety_col2:

    st.metric(
        "Historical Recall Change",
        f"{historical_recall_diff * 100:+.2f}%"
    )


with safety_col3:

    st.metric(
        "Historical ROC-AUC Change",
        f"{historical_roc_auc_diff * 100:+.2f}%"
    )


st.markdown("---")


# ============================================================
# 8. EVIDENTLY REPORT
# ============================================================

if st.checkbox(
    "Show Detailed Evidently HTML Drift Report"
):

    drift_report_path = Path(
        "monitoring/drift_report.html"
    )

    if drift_report_path.exists():

        st.components.v1.html(
            drift_report_path.read_text(
                encoding="utf-8"
            ),
            height=800,
            scrolling=True
        )

    else:

        st.info(
            "No drift report found on disk."
        )