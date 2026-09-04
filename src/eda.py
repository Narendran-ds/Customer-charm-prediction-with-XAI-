import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import chi2_contingency
from sklearn.feature_selection import mutual_info_classif
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools import add_constant

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "WA_Fn-UseC_-Telco-Customer-Churn.xlsx")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")

RANDOM_STATE = 42


def load_data():
    df = pd.read_excel(DATA_PATH)
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())
    return df


def print_overview(df):
    print(df.info())
    print(df.describe())
    print("Missing values:\n", df.isnull().sum())
    print("Churn rate: {:.2f}%".format(100 * df["Churn"].mean()))


def plot_distributions(df, reports_dir):
    for col in ["tenure", "MonthlyCharges", "TotalCharges"]:
        plt.figure(figsize=(8, 5))
        sns.histplot(df[col], kde=True)
        plt.title(f"{col} Distribution")
        plt.savefig(os.path.join(reports_dir, f"{col}_distribution.png"))
        plt.close()


def plot_churn_by_category(df, categorical, reports_dir):
    for col in categorical:
        plt.figure(figsize=(8, 4))
        sns.countplot(x=col, data=df, hue="Churn")
        plt.title(f"Churn by {col}")
        plt.savefig(os.path.join(reports_dir, f"churn_by_{col}.png"))
        plt.close()


def plot_boxplots(df, reports_dir):
    for col in ["tenure", "MonthlyCharges", "TotalCharges"]:
        plt.figure(figsize=(8, 4))
        sns.boxplot(x="Churn", y=col, data=df)
        plt.title(f"{col} vs Churn")
        plt.savefig(os.path.join(reports_dir, f"{col}_vs_churn_boxplot.png"))
        plt.close()


def plot_correlation_heatmap(df, reports_dir):
    plt.figure(figsize=(12, 10))
    sns.heatmap(df.select_dtypes(include=[np.number]).corr(), annot=True, cmap="coolwarm")
    plt.title("Correlation Heatmap")
    plt.savefig(os.path.join(reports_dir, "correlation_heatmap.png"))
    plt.close()


def print_vif(df):
    """VIF requires a design matrix with an intercept column, otherwise the
    auxiliary regressions are forced through the origin and the values are
    not statistically meaningful."""
    numeric = df.select_dtypes(include=[np.number]).drop(columns=["Churn"])
    numeric_with_const = add_constant(numeric)
    vif_data = pd.DataFrame()
    vif_data["feature"] = numeric_with_const.columns
    vif_data["VIF"] = [
        variance_inflation_factor(numeric_with_const.values, i)
        for i in range(numeric_with_const.shape[1])
    ]
    print("\nVariance Inflation Factor (VIF):")
    print(vif_data[vif_data["feature"] != "const"])


def plot_mutual_info(df, reports_dir):
    X = df.drop(["customerID", "Churn"], axis=1)
    X = pd.get_dummies(X, drop_first=True)
    y = df["Churn"]
    mi = mutual_info_classif(X, y, discrete_features="auto", random_state=RANDOM_STATE)
    mi_df = pd.DataFrame({"Feature": X.columns, "Mutual Info": mi})
    mi_df.sort_values(by="Mutual Info", ascending=False, inplace=True)

    plt.figure(figsize=(10, 6))
    sns.barplot(x="Mutual Info", y="Feature", data=mi_df)
    plt.title("Mutual Information with Churn")
    plt.savefig(os.path.join(reports_dir, "mutual_info.png"))
    plt.close()


def cramers_v(confusion_matrix):
    chi2 = chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum()
    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    return np.sqrt(phi2 / min(k - 1, r - 1))


def print_cramers_v(df, categorical):
    """Cramer's V is symmetric -- each unordered pair is only computed once."""
    for i, col1 in enumerate(categorical):
        for col2 in categorical[i + 1:]:
            matrix = pd.crosstab(df[col1], df[col2]).values
            cv = cramers_v(matrix)
            if cv > 0.2:
                print(f"Cramer's V between {col1} and {col2}: {cv:.2f}")


def main():
    sns.set_theme(style="whitegrid")
    pd.set_option("display.max_columns", None)
    os.makedirs(REPORTS_DIR, exist_ok=True)

    df = load_data()
    print_overview(df)

    categorical = [col for col in df.columns if df[col].nunique() < 10 and col != "Churn"]

    plot_distributions(df, REPORTS_DIR)
    plot_churn_by_category(df, categorical, REPORTS_DIR)
    plot_boxplots(df, REPORTS_DIR)
    plot_correlation_heatmap(df, REPORTS_DIR)
    print_vif(df)
    plot_mutual_info(df, REPORTS_DIR)
    print_cramers_v(df, categorical)

    print("Detailed EDA completed and plots saved to reports/")


if __name__ == "__main__":
    main()
