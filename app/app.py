import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
import pickle

from src.utils import load_model_and_scaler, preprocess_single_input

st.set_page_config(page_title="Customer Churn Prediction Dashboard", layout="wide")
st.title("Customer Churn Prediction & Explainability Dashboard")

model, scaler = load_model_and_scaler("models/churn_model.pkl", "models/scaler.pkl")
feature_columns = scaler.feature_names_in_

uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])
menu = st.sidebar.radio("Navigation", ["Manual Input Form", "Settings"])

@st.cache_data
def predict_batch(df):
    X = df.drop(columns=["customerID", "Churn"], errors='ignore')
    X = pd.get_dummies(X).reindex(columns=feature_columns, fill_value=0)
    X_scaled = scaler.transform(X)
    probs = model.predict_proba(X_scaled)[:,1]
    preds = np.where(probs > 0.5, 1, 0)
    return preds, probs, X_scaled

if uploaded_file:
    data = pd.read_csv(uploaded_file)
    preds, probs, X_scaled = predict_batch(data)

    churn_risk = pd.DataFrame({
        "customerID": data.get("customerID", pd.Series(range(len(probs)))),
        "Churn Probability": probs,
        "Risk Level": pd.cut(probs, bins=[0,0.4,0.7,1], labels=["Low", "Medium", "High"])
    })

    kpi1, kpi2, kpi3 = st.columns(3)
    kpi1.metric("% Churn Risk", f"{100*np.mean(preds):.2f}%")
    kpi2.metric("Average Tenure", f"{np.mean(data['tenure']):.1f} months")
    kpi3.metric("Expected Revenue Loss", f"${(np.mean(probs) * np.mean(data['MonthlyCharges'])):.2f}")

    st.subheader("Top 5 High-Risk Customers")
    st.dataframe(churn_risk.sort_values(by="Churn Probability", ascending=False).head(5))

    pie_fig, ax = plt.subplots()
    ax.pie([np.mean(preds), 1-np.mean(preds)], labels=["Churn","No Churn"], autopct='%1.1f%%')
    st.pyplot(pie_fig)

    st.subheader("Feature Importance (SHAP)")
    explainer = shap.Explainer(model, X_scaled, feature_names=feature_columns)
    shap_values = explainer(X_scaled)
    shap.summary_plot(shap_values, features=X_scaled, feature_names=feature_columns, show=False)
    st.pyplot(plt.gcf())
    plt.clf()

if menu == "Manual Input Form":
    st.subheader("Manual Customer Entry")
    input_data = {"tenure": st.slider("Tenure", 0, 72, 12), "MonthlyCharges": st.slider("Monthly Charges", 20, 120, 70)}
    for col in feature_columns:
        if col not in input_data:
            input_data[col] = 0

    scaled_input = preprocess_single_input(input_data, feature_columns, scaler)
    pred = model.predict(scaled_input)[0]
    prob = model.predict_proba(scaled_input)[0,1]

    st.metric("Churn Probability", f"{100*prob:.2f}%")

    explainer = shap.Explainer(model, scaled_input, feature_names=feature_columns)
    shap_values = explainer(scaled_input)
    shap.plots.waterfall(shap_values[0], show=False)
    st.pyplot(plt.gcf())
    plt.clf()

    pie_fig2, ax2 = plt.subplots()
    ax2.pie([prob,1-prob], labels=["Churn","No Churn"], autopct='%1.1f%%')
    st.pyplot(pie_fig2)

elif menu == "Settings":
    st.subheader("Settings")
    st.write("Adjust thresholds, view logs, or download prediction reports here.")
