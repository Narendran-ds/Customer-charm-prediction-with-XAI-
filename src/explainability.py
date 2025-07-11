import os
import pandas as pd
import numpy as np
import pickle
import shap
import lime.lime_tabular
import matplotlib.pyplot as plt
import webbrowser

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "churn_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "models", "scaler.pkl")
DATA_PATH = os.path.join(BASE_DIR, "data", "cleaned_telco_data.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "explainability_reports")
os.makedirs(OUTPUT_DIR, exist_ok=True)

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)
with open(SCALER_PATH, "rb") as f:
    scaler = pickle.load(f)

df = pd.read_csv(DATA_PATH)
X = df.drop(columns=["customerID", "Churn"])
y = df["Churn"]

X_scaled = scaler.transform(X)

try:
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_scaled)
    plt.figure()
    shap.summary_plot(shap_values, X, feature_names=X.columns, max_display=10, show=False)
    plt.savefig(os.path.join(OUTPUT_DIR, "shap_summary.png"), bbox_inches='tight')
    plt.close()
except:
    explainer = shap.Explainer(model.predict, X_scaled)
    shap_values = explainer(X_scaled)
    plt.figure()
    shap.plots.beeswarm(shap_values, max_display=10, show=False)
    plt.savefig(os.path.join(OUTPUT_DIR, "shap_summary.png"), bbox_inches='tight')
    plt.close()

idx = 5
plt.figure()
try:
    shap.plots.waterfall(shap_values[idx], max_display=10, show=False)
except:
    shap.waterfall_plot(shap_values[idx], max_display=10, show=False)
plt.savefig(os.path.join(OUTPUT_DIR, f"shap_waterfall_instance_{idx}.png"), bbox_inches='tight')
plt.close()

lime_explainer = lime.lime_tabular.LimeTabularExplainer(
    training_data=np.array(X_scaled),
    mode="classification",
    feature_names=X.columns.tolist(),
    class_names=["No Churn", "Churn"],
    verbose=False,
    random_state=42
)
lime_exp = lime_explainer.explain_instance(
    data_row=X_scaled[idx],
    predict_fn=model.predict_proba,
    num_features=8,
    top_labels=1
)

lime_html_path = os.path.join(OUTPUT_DIR, f"lime_explanation_instance_{idx}.html")
lime_exp.save_to_file(lime_html_path)

webbrowser.open(lime_html_path)
