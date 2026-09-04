import pandas as pd

from src.data_preprocessing import clean_data


def make_raw_frame():
    return pd.DataFrame({
        "customerID": ["C1", "C2", "C3"],
        "gender": ["Female", "Male", "Male"],
        "Contract": ["Month-to-month", "One year", "Two year"],
        "TotalCharges": ["29.85", " ", "1500.0"],  # blank string like real Telco data
        "Churn": ["Yes", "No", "No"],
    })


def test_clean_data_maps_churn_to_binary():
    cleaned = clean_data(make_raw_frame())
    assert set(cleaned["Churn"].unique()) <= {0, 1}
    assert cleaned.loc[cleaned["customerID"] == "C1", "Churn"].iloc[0] == 1
    assert cleaned.loc[cleaned["customerID"] == "C2", "Churn"].iloc[0] == 0


def test_clean_data_imputes_blank_total_charges():
    cleaned = clean_data(make_raw_frame())
    assert cleaned["TotalCharges"].isna().sum() == 0


def test_clean_data_preserves_customer_ids_and_encodes_categoricals():
    cleaned = clean_data(make_raw_frame())
    assert list(cleaned["customerID"]) == ["C1", "C2", "C3"]
    assert "gender_Male" in cleaned.columns
    assert "Contract_One year" in cleaned.columns
