import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

import pandas as pd

from src.features import build_features

DATA_PATH = os.path.join(BASE_DIR, "data", "WA_Fn-UseC_-Telco-Customer-Churn.xlsx")
OUTPUT_PATH = os.path.join(BASE_DIR, "data", "cleaned_telco_data.csv")


def load_data(path):
    df = pd.read_excel(path)
    print(f"Data loaded successfully. Initial shape: {df.shape}")
    return df


def clean_data(df):
    df = df.copy()
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())
    df["Churn"] = df["Churn"].map({"No": 0, "Yes": 1})

    before = df.shape[0]
    df = df.drop_duplicates()
    after = df.shape[0]
    if before != after:
        print(f"Dropped {before - after} duplicate rows")

    customer_ids = df["customerID"]
    encoded = build_features(df.drop(columns=["customerID"]))
    encoded.insert(0, "customerID", customer_ids.loc[encoded.index].values)

    print(f"Data cleaned and encoded successfully. Shape after cleaning: {encoded.shape}")
    return encoded


def save_data(df, output_path):
    df.to_csv(output_path, index=False)
    print(f"Cleaned data saved to {output_path}")


if __name__ == "__main__":
    raw = load_data(DATA_PATH)
    cleaned = clean_data(raw)
    save_data(cleaned, OUTPUT_PATH)
    print("\nPreview of cleaned data:")
    print(cleaned.head())
    print("\nDataFrame info:")
    print(cleaned.info())
