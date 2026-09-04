"""Canonical raw-to-model-feature encoding.

This is the single place that turns a raw/cleaned customer dataframe into the
numeric, one-hot-encoded feature frame the models expect. Training
(data_preprocessing.py), the Streamlit app's batch mode, and single-row manual
predictions (utils.preprocess_single_input) all go through this module instead
of each re-implementing pd.get_dummies/reindex logic independently.
"""

import json

import pandas as pd

# The fixed category domain for this dataset's categorical columns, listed in
# alphabetical order to match pandas' default get_dummies/drop_first behavior.
# Encoding a column against this known domain (instead of letting
# pd.get_dummies infer categories from whatever's present in the current
# dataframe) is what makes a single manually-entered row encode correctly:
# with only one row, get_dummies would otherwise see only one category and,
# with drop_first=True, produce *no* dummy column for that field at all --
# silently discarding the caller's actual selection. SeniorCitizen is
# deliberately excluded: it's already numeric (0/1) in the raw data, not a
# category to one-hot encode.
CATEGORICAL_DOMAINS = {
    "gender": ["Female", "Male"],
    "Partner": ["No", "Yes"],
    "Dependents": ["No", "Yes"],
    "PhoneService": ["No", "Yes"],
    "MultipleLines": ["No", "No phone service", "Yes"],
    "InternetService": ["DSL", "Fiber optic", "No"],
    "OnlineSecurity": ["No", "No internet service", "Yes"],
    "OnlineBackup": ["No", "No internet service", "Yes"],
    "DeviceProtection": ["No", "No internet service", "Yes"],
    "TechSupport": ["No", "No internet service", "Yes"],
    "StreamingTV": ["No", "No internet service", "Yes"],
    "StreamingMovies": ["No", "No internet service", "Yes"],
    "Contract": ["Month-to-month", "One year", "Two year"],
    "PaperlessBilling": ["No", "Yes"],
    "PaymentMethod": [
        "Bank transfer (automatic)",
        "Credit card (automatic)",
        "Electronic check",
        "Mailed check",
    ],
}


NUMERIC_COLUMNS = ["tenure", "MonthlyCharges", "TotalCharges"]


def coerce_known_numeric(df):
    """Coerce columns that are numeric in the source dataset but can arrive as
    strings -- TotalCharges famously has blank-string values for zero-tenure
    customers in the raw Telco file, and any of these can arrive malformed in
    an ad hoc CSV upload. If left as object dtype, pd.get_dummies would treat
    such a column as categorical and one-hot-explode it instead of raising,
    silently discarding the feature (this previously happened for TotalCharges
    specifically; generalized here to every known numeric column)."""
    df = df.copy()
    for col in NUMERIC_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def apply_known_categories(df):
    """Fix each known categorical column to its full domain before one-hot
    encoding, so get_dummies always produces the same dummy columns
    regardless of how many rows -- or how many distinct categories -- are
    actually present in `df`."""
    df = df.copy()
    for col, categories in CATEGORICAL_DOMAINS.items():
        if col in df.columns:
            df[col] = pd.Categorical(df[col], categories=categories)
    return df


def build_features(df, drop_first=True):
    """Turn a raw/cleaned customer dataframe into the one-hot-encoded feature
    frame the models are trained on."""
    df = coerce_known_numeric(df)
    df = apply_known_categories(df)
    encoded = pd.get_dummies(df, drop_first=drop_first)
    bool_cols = encoded.select_dtypes(include="bool").columns
    encoded[bool_cols] = encoded[bool_cols].astype(int)
    return encoded


def compute_defaults(df):
    """Per-column fill values for features a caller doesn't supply: median for
    continuous columns, mode for 0/1 dummy columns -- never a blind 0, which
    would silently misrepresent a continuous feature like TotalCharges."""
    defaults = {}
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            defaults[col] = float(df[col].median())
        else:
            defaults[col] = df[col].mode().iloc[0]
    return defaults


def save_feature_schema(columns, defaults, path):
    schema = {"columns": list(columns), "defaults": defaults}
    with open(path, "w") as f:
        json.dump(schema, f, indent=2)


def load_feature_schema(path):
    with open(path) as f:
        return json.load(f)


def unseen_columns(encoded_df, schema):
    """Columns build_features produced that aren't part of the persisted
    training schema -- i.e. categorical values the model never saw. Surface
    these to the caller rather than letting align_to_schema drop them silently."""
    return [c for c in encoded_df.columns if c not in schema["columns"]]


def align_to_schema(encoded_df, schema):
    """Reindex an encoded dataframe to the persisted training column order,
    filling any column the caller didn't supply with its training-time
    median/mode default (not a raw 0), and dropping any unseen columns."""
    columns = schema["columns"]
    defaults = schema.get("defaults", {})
    aligned = encoded_df.reindex(columns=columns)
    for col in columns:
        if aligned[col].isna().any():
            aligned[col] = aligned[col].fillna(defaults.get(col, 0))
    return aligned
