import pickle

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

from .features import align_to_schema, build_features


def load_model_and_scaler(model_path, scaler_path):
    try:
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)
        print("Model and scaler loaded successfully.")
        return model, scaler
    except Exception as e:
        print(f"Error loading model/scaler: {e}")
        raise


def evaluate_classification_metrics(y_true, y_pred, y_prob):
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred),
        "Recall": recall_score(y_true, y_pred),
        "F1": f1_score(y_true, y_pred),
        "ROC-AUC": roc_auc_score(y_true, y_prob),
    }


def print_metrics(metrics_dict):
    print("\nEvaluation Metrics:")
    for key, value in metrics_dict.items():
        print(f"{key}: {value:.4f}")


def preprocess_single_input(raw_input, schema, scaler):
    """raw_input is a dict of RAW (pre-encoding) customer fields, e.g.
    {'gender': 'Male', 'tenure': 12, 'MonthlyCharges': 70.0, ...}. Goes through
    the same build_features/align_to_schema path as training and batch
    predictions, so missing fields are filled from the persisted training-time
    medians/modes instead of a blind 0.
    """
    try:
        df_input = pd.DataFrame([raw_input])
        encoded = build_features(df_input)
        aligned = align_to_schema(encoded, schema)
        return scaler.transform(aligned)
    except Exception as e:
        print(f"Error preprocessing input: {e}")
        raise


def save_object(obj, filepath):
    """Raises on failure -- a caller must be able to detect a failed save
    rather than have it silently swallowed."""
    with open(filepath, "wb") as f:
        pickle.dump(obj, f)
    print(f"Object saved to {filepath}")


def load_object(filepath):
    try:
        with open(filepath, "rb") as f:
            obj = pickle.load(f)
        print(f"Object loaded from {filepath}")
        return obj
    except Exception as e:
        print(f"Failed to load object: {e}")
        raise


def summary_statistics(dataframe):
    print("\nData Summary:")
    print(dataframe.describe())
    print("\nMissing Values:")
    print(dataframe.isnull().sum())
    print("\nCardinality:")
    print(dataframe.nunique())
    return dataframe.describe()
