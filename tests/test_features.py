import pandas as pd

from src.features import align_to_schema, build_features, compute_defaults, unseen_columns


def make_training_frame():
    return pd.DataFrame({
        "gender": ["Female", "Male", "Male", "Female"],
        "Contract": ["Month-to-month", "One year", "Two year", "Month-to-month"],
        "SeniorCitizen": [0, 1, 0, 0],
        "tenure": [1, 24, 60, 5],
        "MonthlyCharges": [70.0, 50.0, 30.0, 90.0],
        "TotalCharges": ["70.0", "1200.0", "1800.0", " "],
    })


def test_build_features_coerces_total_charges_to_numeric():
    df = make_training_frame()
    encoded = build_features(df)
    assert pd.api.types.is_numeric_dtype(encoded["TotalCharges"])
    # the blank string becomes NaN, not a bogus dummy category
    assert encoded["TotalCharges"].isna().sum() == 1


def test_build_features_drop_first_one_hot_encoding():
    df = make_training_frame()
    encoded = build_features(df)
    assert "gender_Male" in encoded.columns
    assert "gender_Female" not in encoded.columns  # baseline category dropped
    assert "Contract_One year" in encoded.columns
    assert "Contract_Two year" in encoded.columns
    assert "Contract_Month-to-month" not in encoded.columns
    # SeniorCitizen is already numeric 0/1 -- must not be one-hot encoded
    assert "SeniorCitizen" in encoded.columns
    assert "SeniorCitizen_1" not in encoded.columns


def test_build_features_single_row_encodes_full_category_domain():
    """Regression test for a real bug found during this audit: encoding a lone
    row with plain pd.get_dummies (no fixed category domain) sees only one
    category per field and, with drop_first=True, silently drops the dummy
    column entirely -- discarding the caller's actual selection. This must not
    happen once build_features fixes each column to its known domain first."""
    row = pd.DataFrame([{
        "gender": "Male",
        "Contract": "Two year",
        "InternetService": "Fiber optic",
        "PaymentMethod": "Mailed check",
        "SeniorCitizen": 1,
        "tenure": 12,
        "MonthlyCharges": 70.0,
        "TotalCharges": 840.0,
    }])
    encoded = build_features(row)
    assert encoded["gender_Male"].iloc[0] == 1
    assert encoded["Contract_Two year"].iloc[0] == 1
    assert encoded["InternetService_Fiber optic"].iloc[0] == 1
    assert encoded["PaymentMethod_Mailed check"].iloc[0] == 1

    baseline_row = pd.DataFrame([{**row.iloc[0].to_dict(), "gender": "Female"}])
    baseline_encoded = build_features(baseline_row)
    assert baseline_encoded["gender_Male"].iloc[0] == 0


def test_compute_defaults_uses_median_for_numeric_and_mode_for_dummy_columns():
    df = pd.DataFrame({
        "TotalCharges": [10.0, 20.0, 30.0],
        "gender_Male": [1, 1, 0],
    })
    defaults = compute_defaults(df)
    assert defaults["TotalCharges"] == 20.0
    assert defaults["gender_Male"] == 1


def test_align_to_schema_fills_missing_columns_with_schema_default_not_zero():
    schema = {
        "columns": ["tenure", "TotalCharges", "gender_Male"],
        "defaults": {"TotalCharges": 500.0, "gender_Male": 1},
    }
    df = pd.DataFrame({"tenure": [12]})  # TotalCharges and gender_Male missing
    aligned = align_to_schema(df, schema)
    assert aligned["TotalCharges"].iloc[0] == 500.0
    assert aligned["gender_Male"].iloc[0] == 1
    assert list(aligned.columns) == schema["columns"]


def test_unseen_columns_flags_categories_not_in_training_schema():
    schema = {"columns": ["gender_Male"], "defaults": {}}
    encoded = pd.DataFrame({"gender_Male": [1], "PaymentMethod_Crypto": [1]})
    assert unseen_columns(encoded, schema) == ["PaymentMethod_Crypto"]
