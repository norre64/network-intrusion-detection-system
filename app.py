from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
import sklearn
import streamlit as st

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATHS = {
    "Random Forest": BASE_DIR / "models" / "random_forest_final_balanced_binary.joblib.gz",
    "MLP (ANN)": BASE_DIR / "models" / "mlp_ann_final_balanced_binary.joblib.gz",
    "Decision Tree": BASE_DIR / "models" / "decision_tree_final_balanced_binary.joblib.gz",
    "Logistic Regression": BASE_DIR / "models" / "logistic_regression_final_balanced_binary.joblib.gz",
}

FEATURE_PATH = BASE_DIR / "config" / "retained_features.json"
RESULTS_PATH = BASE_DIR / "data" / "balanced_final_model_results.csv"
TRAINING_SKLEARN_VERSION = "1.6.1"

st.set_page_config(
    page_title="IoT Intrusion Detection Research Prototype",
    page_icon="🛡️",
    layout="wide",
)

st.markdown(
    """
    <style>
    .block-container {padding-top: 1.5rem; padding-bottom: 2rem;}
    div[data-testid="stMetric"] {
        border: 1px solid rgba(49, 51, 63, 0.15);
        padding: 0.8rem;
        border-radius: 0.6rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def load_features():
    return json.loads(FEATURE_PATH.read_text(encoding="utf-8"))


@st.cache_data
def load_research_results():
    return pd.read_csv(RESULTS_PATH)


@st.cache_resource(show_spinner=False)
def load_model(model_name):
    return joblib.load(MODEL_PATHS[model_name])


def read_uploaded_csv(uploaded_file):
    compression = "gzip" if uploaded_file.name.lower().endswith(".gz") else "infer"
    return pd.read_csv(uploaded_file, compression=compression)


def prepare_predictors(data, required_features):
    missing_features = [feature for feature in required_features if feature not in data.columns]
    if missing_features:
        return None, missing_features, None

    original = data[required_features]
    numeric = original.apply(pd.to_numeric, errors="coerce")

    invalid_mask = original.notna() & numeric.isna()
    invalid_counts = invalid_mask.sum()
    invalid_counts = invalid_counts[invalid_counts > 0].sort_values(ascending=False)

    numeric = numeric.replace([np.inf, -np.inf], np.nan)
    return numeric.astype("float32"), [], invalid_counts


def probability_columns(model, probabilities):
    classes = list(model.classes_)
    if 0 not in classes or 1 not in classes:
        raise ValueError(f"Expected model classes [0, 1], but found {classes}.")

    benign_index = classes.index(0)
    attack_index = classes.index(1)
    return probabilities[:, benign_index], probabilities[:, attack_index]


def evaluation_table(y_true, prediction, attack_probability):
    tn, fp, fn, tp = confusion_matrix(y_true, prediction, labels=[0, 1]).ravel()

    values = {
        "Accuracy": accuracy_score(y_true, prediction),
        "Attack Precision": precision_score(y_true, prediction, pos_label=1, zero_division=0),
        "Attack Recall": recall_score(y_true, prediction, pos_label=1, zero_division=0),
        "Attack F1": f1_score(y_true, prediction, pos_label=1, zero_division=0),
        "Benign Precision": precision_score(y_true, prediction, pos_label=0, zero_division=0),
        "Benign Recall": recall_score(y_true, prediction, pos_label=0, zero_division=0),
        "Benign F1": f1_score(y_true, prediction, pos_label=0, zero_division=0),
        "Macro F1": f1_score(y_true, prediction, average="macro", zero_division=0),
        "Weighted F1": f1_score(y_true, prediction, average="weighted", zero_division=0),
        "Balanced Accuracy": balanced_accuracy_score(y_true, prediction),
        "Benign FPR": fp / (fp + tn) if (fp + tn) else np.nan,
        "Attack FNR": fn / (fn + tp) if (fn + tp) else np.nan,
    }

    if pd.Series(y_true).nunique() == 2:
        values["ROC-AUC"] = roc_auc_score(y_true, attack_probability)
        values["PR-AUC"] = average_precision_score(y_true, attack_probability)
    else:
        values["ROC-AUC"] = np.nan
        values["PR-AUC"] = np.nan

    return pd.DataFrame(
        {"Metric": list(values.keys()), "Value": list(values.values())}
    )


features = load_features()
research_results = load_research_results()

st.title("🛡️ IoT Network Intrusion Detection Research Prototype")
st.caption(
    "Master's research artefact — binary classification of pre-extracted IoT network flows as Benign or Attack."
)

if sklearn.__version__ != TRAINING_SKLEARN_VERSION:
    st.error(
        f"The supplied models were trained with scikit-learn {TRAINING_SKLEARN_VERSION}, "
        f"but this environment is using {sklearn.__version__}. "
        "Install the versions in requirements.txt before running predictions."
    )
    st.stop()

selected_model = st.sidebar.selectbox(
    "Classification model",
    list(MODEL_PATHS.keys()),
    index=0,
)

st.sidebar.markdown("**Recommended research model:** Random Forest")
st.sidebar.caption(
    "Random Forest achieved the highest Macro F1 in the controlled balanced final-test comparison."
)

tab_overview, tab_detection, tab_results, tab_methodology = st.tabs(
    ["Overview", "Traffic Detection", "Research Results", "Methodology"]
)

with tab_overview:
    st.subheader("Research objective")
    st.write(
        "This prototype demonstrates the machine-learning artefact developed to compare "
        "traditional supervised classifiers with an MLP artificial neural network for "
        "binary IoT anomaly detection."
    )

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Source flows", "27.81M")
    col2.metric("Model features", len(features))
    col3.metric("Final task", "Benign vs Attack")
    col4.metric("Best balanced model", "Random Forest")

    st.subheader("Why two experiments were used")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### Experiment 1 — Natural distribution")
        st.write(
            "Preserved the dataset's severe class imbalance. The selected Decision Tree "
            "reached 97.82% accuracy, but only 60.02% Macro F1 and an 80.65% Benign "
            "false-positive rate. This showed why accuracy alone was misleading."
        )
    with c2:
        st.markdown("#### Experiment 2 — Balanced distribution")
        st.write(
            "Used equal Benign and Attack representation to provide a controlled comparison "
            "of Logistic Regression, Decision Tree, Random Forest and MLP. Random Forest "
            "achieved the strongest final Macro F1."
        )

    st.info(
        "This application performs offline classification of pre-extracted network-flow records. "
        "It is not a real-time packet-capture or live network-monitoring system."
    )

with tab_detection:
    st.subheader("Classify network-flow records")
    st.write(
        f"Selected model: **{selected_model}**. Upload a CSV containing the "
        f"{len(features)} required predictor columns. A raw dataset may contain additional "
        "columns; only the retained predictor columns are passed to the model."
    )

    template_bytes = (BASE_DIR / "sample_input_template.csv").read_bytes()
    st.download_button(
        "Download CSV feature template",
        data=template_bytes,
        file_name="iot_ids_feature_template.csv",
        mime="text/csv",
    )

    uploaded_file = st.file_uploader(
        "Upload network-flow CSV",
        type=["csv", "gz"],
        help="CSV or gzip-compressed CSV. For a smooth demonstration, use a manageable sample rather than millions of rows.",
    )

    if uploaded_file is not None:
        try:
            data = read_uploaded_csv(uploaded_file)
        except Exception as exc:
            st.error(f"The uploaded file could not be read as CSV: {exc}")
            st.stop()

        if len(data) == 0:
            st.warning("The uploaded CSV contains column headers but no data rows.")
            st.stop()

        if len(data) > 250_000:
            st.warning(
                f"The file contains {len(data):,} rows. It can be processed, but large uploads "
                "may exceed memory limits on hosted Streamlit environments."
            )

        st.success(f"Loaded {len(data):,} network-flow records with {len(data.columns)} columns.")

        with st.expander("Preview uploaded data"):
            st.dataframe(data.head(25), use_container_width=True)

        X, missing_features, invalid_counts = prepare_predictors(data, features)

        if missing_features:
            st.error(
                f"The file is missing {len(missing_features)} of the {len(features)} required model features."
            )
            st.dataframe(pd.DataFrame({"Missing feature": missing_features}), hide_index=True)
            st.stop()

        if invalid_counts is not None and not invalid_counts.empty:
            st.error(
                "Some required feature values contain non-numeric text. Correct these values before prediction."
            )
            st.dataframe(
                invalid_counts.rename("Invalid value count").reset_index(names="Feature"),
                hide_index=True,
            )
            st.stop()

        st.success(f"All {len(features)} required model features were found and validated.")

        missing_numeric_values = int(X.isna().sum().sum())
        if missing_numeric_values:
            st.info(
                f"{missing_numeric_values:,} missing/infinite numeric values will be handled by "
                "the median imputer stored inside the trained model pipeline."
            )

        if st.button("Run intrusion detection", type="primary", use_container_width=True):
            try:
                with st.spinner(f"Loading {selected_model} and classifying flows..."):
                    model = load_model(selected_model)
                    prediction = model.predict(X).astype(int)
                    probabilities = model.predict_proba(X)
                    benign_probability, attack_probability = probability_columns(model, probabilities)
            except Exception as exc:
                st.error(f"Prediction failed: {exc}")
                st.stop()

            prediction_name = np.where(prediction == 1, "Attack", "Benign")
            top_probability = np.maximum(benign_probability, attack_probability)

            results = data.copy()
            results["Predicted binary_label"] = prediction
            results["Prediction"] = prediction_name
            results["Benign Probability"] = benign_probability
            results["Attack Probability"] = attack_probability
            results["Top Class Probability"] = top_probability

            benign_count = int((prediction == 0).sum())
            attack_count = int((prediction == 1).sum())
            total_count = len(prediction)
            attack_rate = attack_count / total_count

            st.subheader("Prediction summary")
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Total flows", f"{total_count:,}")
            c2.metric("Benign", f"{benign_count:,}")
            c3.metric("Attack", f"{attack_count:,}")
            c4.metric("Attack rate", f"{attack_rate:.2%}")
            c5.metric("Mean attack probability", f"{attack_probability.mean():.2%}")

            distribution = pd.DataFrame(
                {"Class": ["Benign", "Attack"], "Count": [benign_count, attack_count]}
            ).set_index("Class")

            chart_col1, chart_col2 = st.columns(2)
            with chart_col1:
                st.markdown("#### Predicted class distribution")
                st.bar_chart(distribution)

            with chart_col2:
                st.markdown("#### Attack-probability distribution")
                counts, edges = np.histogram(attack_probability, bins=np.linspace(0, 1, 11))
                labels = [f"{edges[i]:.1f}–{edges[i+1]:.1f}" for i in range(len(counts))]
                histogram = pd.DataFrame(
                    {"Probability range": labels, "Flows": counts}
                ).set_index("Probability range")
                st.bar_chart(histogram)

            st.caption(
                "Probabilities are model-estimated values from predict_proba and were not "
                "formally calibrated. They should not be interpreted as guaranteed certainty."
            )

            st.subheader("Prediction results")
            preview_columns = [
                "Prediction",
                "Predicted binary_label",
                "Benign Probability",
                "Attack Probability",
                "Top Class Probability",
            ]
            st.dataframe(
                results[preview_columns].head(1_000),
                use_container_width=True,
                hide_index=True,
            )
            if len(results) > 1_000:
                st.caption("Showing the first 1,000 rows. The download contains every prediction.")

            if "binary_label" in data.columns:
                st.subheader("Evaluation against supplied labels")
                labels = pd.to_numeric(data["binary_label"], errors="coerce")
                labels_valid = labels.notna().all() and labels.isin([0, 1]).all()

                if labels_valid:
                    y_true = labels.astype(int).to_numpy()
                    metrics = evaluation_table(y_true, prediction, attack_probability)

                    metric_lookup = dict(zip(metrics["Metric"], metrics["Value"]))
                    e1, e2, e3, e4 = st.columns(4)
                    e1.metric("Accuracy", f"{metric_lookup['Accuracy']:.2%}")
                    e2.metric("Macro F1", f"{metric_lookup['Macro F1']:.2%}")
                    e3.metric("Balanced Accuracy", f"{metric_lookup['Balanced Accuracy']:.2%}")
                    e4.metric("Attack Recall", f"{metric_lookup['Attack Recall']:.2%}")

                    display_metrics = metrics.copy()
                    display_metrics["Value"] = display_metrics["Value"].map(
                        lambda value: "" if pd.isna(value) else f"{value:.4f}"
                    )
                    st.dataframe(display_metrics, hide_index=True, use_container_width=True)

                    matrix = confusion_matrix(y_true, prediction, labels=[0, 1])
                    matrix_df = pd.DataFrame(
                        matrix,
                        index=["Actual Benign", "Actual Attack"],
                        columns=["Predicted Benign", "Predicted Attack"],
                    )
                    st.markdown("#### Confusion matrix")
                    st.dataframe(matrix_df, use_container_width=True)

                    report = classification_report(
                        y_true,
                        prediction,
                        labels=[0, 1],
                        target_names=["Benign", "Attack"],
                        output_dict=True,
                        zero_division=0,
                    )
                    report_df = pd.DataFrame(report).transpose().round(4)
                    st.markdown("#### Classification report")
                    st.dataframe(report_df, use_container_width=True)
                else:
                    st.warning(
                        "A binary_label column was detected, but it contains values other than "
                        "0 and 1 (or missing labels). Evaluation was skipped."
                    )

            csv_bytes = results.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download all predictions",
                data=csv_bytes,
                file_name=f"iot_ids_{selected_model.lower().replace(' ', '_').replace('(', '').replace(')', '')}_predictions.csv",
                mime="text/csv",
                use_container_width=True,
            )

with tab_results:
    st.subheader("Completed balanced final-test comparison")
    st.write(
        "Each model family's best hyperparameter configuration was refitted on the same "
        "200,000-row balanced development dataset and evaluated on the same fixed "
        "100,000-row balanced test set."
    )

    display = research_results.copy()
    percent_columns = [
        "Accuracy", "Attack Precision", "Attack Recall", "Attack F1",
        "Benign Precision", "Benign Recall", "Benign F1", "Macro F1",
        "Balanced Accuracy", "ROC-AUC", "PR-AUC", "Benign FPR", "Attack FNR"
    ]
    for column in percent_columns:
        display[column] = display[column].map(lambda value: f"{value:.2%}")

    st.dataframe(display, hide_index=True, use_container_width=True)

    rf = research_results.loc[research_results["Model"] == "Random Forest"].iloc[0]
    mlp = research_results.loc[research_results["Model"] == "MLP (ANN)"].iloc[0]

    c1, c2, c3 = st.columns(3)
    c1.metric("RF Macro F1", f"{rf['Macro F1']:.2%}")
    c2.metric("MLP Macro F1", f"{mlp['Macro F1']:.2%}")
    c3.metric(
        "RF advantage",
        f"{(rf['Macro F1'] - mlp['Macro F1']) * 100:.2f} percentage points",
    )

    st.write(
        "Random Forest achieved the highest final Macro F1, while MLP was a close second. "
        "No formal significance test was conducted, so the observed performance difference "
        "should not be described as statistically significant."
    )

with tab_methodology:
    st.subheader("Experimental pipeline")
    st.markdown(
        """
        1. **CIC IoT-DIAD 2024:** 27,810,004 labelled flow observations.
        2. **Main holdout boundary:** source-file-aware separation where possible, with an early/late fallback for single-file attack types.
        3. **Training pool:** 22,209,556 rows.
        4. **Held-out test pool:** 5,600,448 rows.
        5. **Balanced development data:** 100,000 Benign + 100,000 Attack.
        6. **Development split:** 160,000 training + 40,000 validation, stratified by binary label.
        7. **Tuning:** GridSearchCV with three-fold StratifiedKFold and Macro F1.
        8. **Preprocessing:** median imputation for all models; StandardScaler additionally for Logistic Regression and MLP.
        9. **Final comparison:** each best estimator refitted on the full 200,000 development rows and evaluated on the same 100,000-row balanced test.
        """
    )

    st.subheader("Feature handling")
    st.write(
        f"The deployed pipelines expect {len(features)} numeric flow predictors. "
        "Flow ID, source/destination IP addresses, Timestamp, label-derived fields and the "
        "binary target were excluded from the predictor matrix during model development."
    )

    with st.expander("Show retained model features"):
        st.dataframe(pd.DataFrame({"Feature": features}), hide_index=True, use_container_width=True)

    st.subheader("Deployment boundary")
    st.write(
        "The Streamlit interface is an application layer around the completed offline "
        "machine-learning pipelines. It does not change the models, retrain them, fit new "
        "imputers/scalers, or convert the research prototype into a true live packet-monitoring IDS."
    )
