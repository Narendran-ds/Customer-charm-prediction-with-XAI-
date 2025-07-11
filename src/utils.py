import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

def load_model_and_scaler(model_path, scaler_path):
    try:
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)
        print("✅ Model and scaler loaded successfully.")
        return model, scaler
    except Exception as e:
        print(f"❌ Error loading model/scaler: {e}")
        raise

def evaluate_classification_metrics(y_true, y_pred, y_prob):
    metrics = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred),
        "Recall": recall_score(y_true, y_pred),
        "F1": f1_score(y_true, y_pred),
        "ROC-AUC": roc_auc_score(y_true, y_prob)
    }
    return metrics

def print_metrics(metrics_dict):
    print("\n📊 Evaluation Metrics:")
    for key, value in metrics_dict.items():
        print(f"{key}: {value:.4f}")

def preprocess_single_input(input_dict, feature_columns, scaler):
    try:
        df_input = pd.DataFrame([input_dict])
        df_input = df_input.reindex(columns=feature_columns, fill_value=0)
        scaled_input = scaler.transform(df_input)
        return scaled_input
    except Exception as e:
        print(f"❌ Error preprocessing input: {e}")
        raise

def save_object(obj, filepath):
    try:
        with open(filepath, 'wb') as f:
            pickle.dump(obj, f)
        print(f"✅ Object saved to {filepath}")
    except Exception as e:
        print(f"❌ Failed to save object: {e}")

def load_object(filepath):
    try:
        with open(filepath, 'rb') as f:
            obj = pickle.load(f)
        print(f"✅ Object loaded from {filepath}")
        return obj
    except Exception as e:
        print(f"❌ Failed to load object: {e}")
        raise

def summary_statistics(dataframe):
    print("\n📝 Data Summary:")
    print(dataframe.describe())
    print("\nMissing Values:")
    print(dataframe.isnull().sum())
    print("\nCardinality:")
    print(dataframe.nunique())