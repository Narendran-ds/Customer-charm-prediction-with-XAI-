import os
import sys

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, BASE_DIR)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import streamlit as st

from src.features import align_to_schema, build_features, load_feature_schema, unseen_columns
from src.utils import load_model_and_scaler, preprocess_single_input

MODEL_PATH = os.path.join(BASE_DIR, "models", "churn_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "models", "scaler.pkl")
SCHEMA_PATH = os.path.join(BASE_DIR, "models", "feature_schema.json")

# The "no churn" / "churn" cutoff and the Low/Medium/High risk buckets used to be
# independent magic numbers that could disagree (e.g. a 0.45 probability showing
# "Medium" risk while being labeled "No churn"). Medium now ends exactly at the
# classification threshold, so "High" risk and a "Churn" prediction always agree.
CHURN_THRESHOLD = 0.5
RISK_BINS = [0, 0.3, CHURN_THRESHOLD, 1.0]
RISK_LABELS = ["Low", "Medium", "High"]

REQUIRED_BATCH_COLUMNS = ["tenure", "MonthlyCharges"]

# Known category domain for the Telco churn dataset -- used both to build the
# manual-entry form and to keep it in sync with what the model was trained on.
CATEGORY_OPTIONS = {
    "gender": ["Female", "Male"],
    "SeniorCitizen": [0, 1],
    "Partner": ["Yes", "No"],
    "Dependents": ["Yes", "No"],
    "PhoneService": ["Yes", "No"],
    "MultipleLines": ["No", "Yes", "No phone service"],
    "InternetService": ["DSL", "Fiber optic", "No"],
    "OnlineSecurity": ["No", "Yes", "No internet service"],
    "OnlineBackup": ["No", "Yes", "No internet service"],
    "DeviceProtection": ["No", "Yes", "No internet service"],
    "TechSupport": ["No", "Yes", "No internet service"],
    "StreamingTV": ["No", "Yes", "No internet service"],
    "StreamingMovies": ["No", "Yes", "No internet service"],
    "Contract": ["Month-to-month", "One year", "Two year"],
    "PaperlessBilling": ["Yes", "No"],
    "PaymentMethod": [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ],
}

st.set_page_config(page_title="Customer Churn Prediction Dashboard", layout="wide")
st.title("Customer Churn Prediction & Explainability Dashboard")


@st.cache_resource
def get_model_artifacts():
    model, scaler = load_model_and_scaler(MODEL_PATH, SCALER_PATH)
    schema = load_feature_schema(SCHEMA_PATH)
    return model, scaler, schema


def build_shap_explainer(model, background, feature_names):
    try:
        return shap.Explainer(model, background, feature_names=feature_names)
    except Exception as e:
        st.warning(f"Could not build a SHAP explainer for this model: {e}")
        return None


try:
    model, scaler, schema = get_model_artifacts()
except Exception as e:
    st.error(
        f"Could not load the trained model/scaler/schema from `models/`: {e}\n\n"
        "Run `python -m src.train_model` first to produce them."
    )
    st.stop()

feature_columns = schema["columns"]

uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])
menu = st.sidebar.radio("Navigation", ["Manual Input Form", "Settings"])


def predict_batch(raw_df):
    X_raw = raw_df.drop(columns=["customerID", "Churn"], errors="ignore")
    encoded = build_features(X_raw)
    unseen = unseen_columns(encoded, schema)
    aligned = align_to_schema(encoded, schema)
    X_scaled = scaler.transform(aligned)
    probs = model.predict_proba(X_scaled)[:, 1]
    preds = np.where(probs > CHURN_THRESHOLD, 1, 0)
    return preds, probs, X_scaled, unseen


if uploaded_file:
    data = pd.read_csv(uploaded_file)

    if data.empty:
        st.warning("The uploaded CSV has no rows.")
    else:
        missing = [c for c in REQUIRED_BATCH_COLUMNS if c not in data.columns]
        if missing:
            st.error(f"Uploaded CSV is missing required column(s): {', '.join(missing)}")
        else:
            # tenure/MonthlyCharges are read directly (not through build_features)
            # for the KPI cards below, so they need the same numeric coercion.
            data["tenure"] = pd.to_numeric(data["tenure"], errors="coerce")
            data["MonthlyCharges"] = pd.to_numeric(data["MonthlyCharges"], errors="coerce")
            bad_rows = data["tenure"].isna() | data["MonthlyCharges"].isna()
            if bad_rows.any():
                st.warning(
                    f"{bad_rows.sum()} row(s) had a non-numeric tenure or "
                    "MonthlyCharges value and were excluded from the KPI cards."
                )
                data = data.loc[~bad_rows].reset_index(drop=True)

            if data.empty:
                st.warning("No rows had valid tenure/MonthlyCharges values.")
            else:
                try:
                    preds, probs, X_scaled, unseen = predict_batch(data)
                    if unseen:
                        st.warning(
                            "Some uploaded values don't match any category seen "
                            "during training and were treated as missing (filled "
                            f"with the training-average value) for: {', '.join(unseen)}"
                        )

                    churn_risk = pd.DataFrame({
                        "customerID": data.get("customerID", pd.Series(range(len(probs)))),
                        "Churn Probability": probs,
                        "Risk Level": pd.cut(probs, bins=RISK_BINS, labels=RISK_LABELS),
                    })

                    kpi1, kpi2, kpi3 = st.columns(3)
                    kpi1.metric("% Churn Risk", f"{100 * np.mean(preds):.2f}%")
                    kpi2.metric("Average Tenure", f"{np.mean(data['tenure']):.1f} months")
                    kpi3.metric(
                        "Expected Revenue Loss",
                        f"${(np.mean(probs) * np.mean(data['MonthlyCharges'])):.2f}",
                    )

                    st.subheader("Top 5 High-Risk Customers")
                    st.dataframe(
                        churn_risk.sort_values(by="Churn Probability", ascending=False).head(5)
                    )

                    pie_fig, ax = plt.subplots()
                    ax.pie(
                        [np.mean(preds), 1 - np.mean(preds)],
                        labels=["Churn", "No Churn"],
                        autopct="%1.1f%%",
                    )
                    st.pyplot(pie_fig)

                    st.subheader("Feature Importance (SHAP)")
                    explainer = build_shap_explainer(model, X_scaled, feature_columns)
                    if explainer is not None:
                        shap_values = explainer(X_scaled)
                        shap.summary_plot(
                            shap_values,
                            features=X_scaled,
                            feature_names=feature_columns,
                            show=False,
                        )
                        st.pyplot(plt.gcf())
                        plt.clf()
                except Exception as e:
                    st.error(f"Could not generate batch predictions for this file: {e}")

if menu == "Manual Input Form":
    st.subheader("Manual Customer Entry")
    st.caption("Every field below feeds the prediction -- nothing is silently zeroed out.")

    col_a, col_b = st.columns(2)
    raw_input = {}
    for i, (field, options) in enumerate(CATEGORY_OPTIONS.items()):
        target_col = col_a if i % 2 == 0 else col_b
        raw_input[field] = target_col.selectbox(field, options)

    raw_input["tenure"] = st.slider("Tenure (months)", 0, 72, 12)
    raw_input["MonthlyCharges"] = st.slider("Monthly Charges ($)", 18.0, 120.0, 70.0)
    raw_input["TotalCharges"] = st.number_input(
        "Total Charges ($)",
        min_value=0.0,
        value=float(raw_input["tenure"] * raw_input["MonthlyCharges"]),
    )

    try:
        scaled_input = preprocess_single_input(raw_input, schema, scaler)
        pred = model.predict(scaled_input)[0]
        prob = model.predict_proba(scaled_input)[0, 1]

        st.metric("Churn Probability", f"{100 * prob:.2f}%")

        explainer = build_shap_explainer(model, scaled_input, feature_columns)
        if explainer is not None:
            shap_values = explainer(scaled_input)
            shap.plots.waterfall(shap_values[0], show=False)
            st.pyplot(plt.gcf())
            plt.clf()

        pie_fig2, ax2 = plt.subplots()
        ax2.pie([prob, 1 - prob], labels=["Churn", "No Churn"], autopct="%1.1f%%")
        st.pyplot(pie_fig2)
    except Exception as e:
        st.error(f"Could not generate a prediction for this input: {e}")

elif menu == "Settings":
    st.subheader("Settings")
    st.write("Adjust thresholds, view logs, or download prediction reports here.")
