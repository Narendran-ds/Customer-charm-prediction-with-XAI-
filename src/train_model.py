import json
import os
import pickle
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from src.features import compute_defaults, save_feature_schema

DATA_PATH = os.path.join(BASE_DIR, "data", "cleaned_telco_data.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")
RANDOM_STATE = 42


def load_training_data(path):
    df = pd.read_csv(path)
    X = df.drop(columns=["customerID", "Churn"])
    y = df["Churn"]
    return X, y


def select_best_model(X_train_scaled, y_train):
    """Cross-validated model selection on the training fold only -- the held-out
    test set is reserved for a single, final, unbiased evaluation of whichever
    model wins here, instead of being used for both selection and reporting."""
    candidates = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE),
        "XGBoost": XGBClassifier(eval_metric="logloss", random_state=RANDOM_STATE),
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    cv_scores = {}
    print("\nCross-validated model selection (5-fold, ROC-AUC):")
    for name, model in candidates.items():
        scores = cross_val_score(model, X_train_scaled, y_train, cv=cv, scoring="roc_auc")
        cv_scores[name] = scores.mean()
        print(f"   {name}: {scores.mean():.4f} (+/- {scores.std():.4f})")

    best_name = max(cv_scores, key=cv_scores.get)
    best_model = candidates[best_name]
    best_model.fit(X_train_scaled, y_train)
    print(f"\nBest model by cross-validated ROC-AUC: {best_name} ({cv_scores[best_name]:.4f})")
    return best_name, best_model


def evaluate(model, X_test_scaled, y_test):
    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:, 1]
    return {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1": f1_score(y_test, y_pred),
        "ROC-AUC": roc_auc_score(y_test, y_prob),
    }


def main():
    print(f"Loading cleaned data from {DATA_PATH}")
    X, y = load_training_data(DATA_PATH)
    print(f"Loaded {X.shape[0]} rows, {X.shape[1]} features")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )
    print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    best_name, best_model = select_best_model(X_train_scaled, y_train)

    metrics = evaluate(best_model, X_test_scaled, y_test)
    print(f"\nHeld-out test performance for {best_name} (not used for model selection):")
    for metric, score in metrics.items():
        print(f"   {metric}: {score:.4f}")

    os.makedirs(MODELS_DIR, exist_ok=True)

    with open(os.path.join(MODELS_DIR, "churn_model.pkl"), "wb") as f:
        pickle.dump(best_model, f)
    with open(os.path.join(MODELS_DIR, "scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)

    schema_defaults = compute_defaults(X_train)
    save_feature_schema(
        X_train.columns, schema_defaults, os.path.join(MODELS_DIR, "feature_schema.json")
    )

    with open(os.path.join(MODELS_DIR, "test_indices.json"), "w") as f:
        json.dump(X_test.index.tolist(), f)

    print(f"\nSaved model, scaler, feature schema, and test-set indices to {MODELS_DIR}")


if __name__ == "__main__":
    main()
