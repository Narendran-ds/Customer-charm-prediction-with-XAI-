import argparse
import json
import os
import pickle
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

import lime.lime_tabular
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

MODEL_PATH = os.path.join(BASE_DIR, "models", "churn_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "models", "scaler.pkl")
DATA_PATH = os.path.join(BASE_DIR, "data", "cleaned_telco_data.csv")
TEST_INDICES_PATH = os.path.join(BASE_DIR, "models", "test_indices.json")
OUTPUT_DIR = os.path.join(BASE_DIR, "explainability_reports")

INSTANCE_INDEX = 5  # position within the held-out test set, not the full CSV


def load_artifacts():
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)
    return model, scaler


def load_test_set():
    """Explain only the held-out test set persisted by train_model.py, not the
    full dataset the model was trained on -- falls back to the full dataset
    with a warning if train_model.py hasn't been re-run yet to produce it."""
    df = pd.read_csv(DATA_PATH)
    X_full = df.drop(columns=["customerID", "Churn"])
    if os.path.exists(TEST_INDICES_PATH):
        with open(TEST_INDICES_PATH) as f:
            test_indices = json.load(f)
        return X_full.loc[X_full.index.intersection(test_indices)]
    print(
        "Warning: no models/test_indices.json found; explaining the full dataset. "
        "Re-run train_model.py to persist a held-out test set."
    )
    return X_full


def compute_shap_values(model, X_scaled, feature_names):
    """A single unified explainer (the callable API) instead of a
    TreeExplainer/generic-Explainer branch with a bare except -- shap.Explainer
    already dispatches to the right algorithm for the given model type and
    always returns an Explanation object usable by both plots.beeswarm and
    plots.waterfall."""
    explainer = shap.Explainer(model, X_scaled, feature_names=feature_names)
    return explainer(X_scaled)


def save_summary_plot(shap_values, path):
    plt.figure()
    shap.plots.beeswarm(shap_values, max_display=10, show=False)
    plt.savefig(path, bbox_inches="tight")
    plt.close()


def save_waterfall_plot(shap_values, idx, path):
    plt.figure()
    shap.plots.waterfall(shap_values[idx], max_display=10, show=False)
    plt.savefig(path, bbox_inches="tight")
    plt.close()


def main(open_browser=False):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    model, scaler = load_artifacts()
    X = load_test_set()

    if len(X) == 0:
        raise ValueError("Test set is empty -- re-run train_model.py to regenerate it.")
    idx = min(INSTANCE_INDEX, len(X) - 1)

    X_scaled = scaler.transform(X)

    shap_values = compute_shap_values(model, X_scaled, X.columns)
    save_summary_plot(shap_values, os.path.join(OUTPUT_DIR, "shap_summary.png"))
    save_waterfall_plot(
        shap_values, idx, os.path.join(OUTPUT_DIR, f"shap_waterfall_instance_{idx}.png")
    )

    lime_explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=np.array(X_scaled),
        mode="classification",
        feature_names=X.columns.tolist(),
        class_names=["No Churn", "Churn"],
        verbose=False,
        random_state=42,
    )
    lime_exp = lime_explainer.explain_instance(
        data_row=X_scaled[idx],
        predict_fn=model.predict_proba,
        num_features=8,
        top_labels=1,
    )
    lime_html_path = os.path.join(OUTPUT_DIR, f"lime_explanation_instance_{idx}.html")
    lime_exp.save_to_file(lime_html_path)

    print(f"Saved SHAP + LIME explainability reports to {OUTPUT_DIR}")

    if open_browser:
        import webbrowser

        webbrowser.open(lime_html_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate SHAP/LIME explainability reports.")
    parser.add_argument(
        "--open-browser",
        action="store_true",
        help="Open the generated LIME report in the default browser (skip for headless runs).",
    )
    args = parser.parse_args()
    main(open_browser=args.open_browser)
