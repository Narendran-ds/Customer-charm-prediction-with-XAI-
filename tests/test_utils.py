import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.features import build_features
from src.utils import evaluate_classification_metrics, preprocess_single_input


def test_evaluate_classification_metrics_perfect_predictions():
    y_true = [0, 1, 0, 1]
    y_pred = [0, 1, 0, 1]
    y_prob = [0.1, 0.9, 0.2, 0.8]
    metrics = evaluate_classification_metrics(y_true, y_pred, y_prob)
    assert metrics["Accuracy"] == 1.0
    assert metrics["ROC-AUC"] == 1.0


def test_preprocess_single_input_matches_schema_shape_and_uses_defaults():
    training_rows = [
        {"gender": "Female", "tenure": 1, "MonthlyCharges": 70.0},
        {"gender": "Male", "tenure": 24, "MonthlyCharges": 50.0},
        {"gender": "Male", "tenure": 60, "MonthlyCharges": 30.0},
    ]
    encoded = build_features(pd.DataFrame(training_rows))
    scaler = StandardScaler().fit(encoded)
    schema = {
        "columns": list(encoded.columns),
        "defaults": {"gender_Male": 1, "tenure": 24.0, "MonthlyCharges": 50.0},
    }

    # Caller only supplies tenure -- MonthlyCharges and gender should be
    # filled from the schema's training-time defaults, not a blind 0.
    scaled = preprocess_single_input({"tenure": 12}, schema, scaler)
    assert scaled.shape == (1, len(schema["columns"]))
    assert np.isfinite(scaled).all()
