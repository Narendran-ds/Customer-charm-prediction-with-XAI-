import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import mutual_info_classif
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy.stats import chi2_contingency

sns.set(style="whitegrid")
pd.set_option('display.max_columns', None)

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_path = os.path.join(base_dir, "data", "WA_Fn-UseC_-Telco-Customer-Churn.xlsx")
reports_dir = os.path.join(base_dir, "reports")

if not os.path.exists(reports_dir):
    os.makedirs(reports_dir)

df = pd.read_excel(data_path)
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median())

print(df.info())
print(df.describe())
print("Missing values:\n", df.isnull().sum())
print("Churn rate: {:.2f}%".format(100 * df['Churn'].mean()))

for col in ['tenure', 'MonthlyCharges', 'TotalCharges']:
    plt.figure(figsize=(8,5))
    sns.histplot(df[col], kde=True)
    plt.title(f"{col} Distribution")
    plt.savefig(os.path.join(reports_dir, f"{col}_distribution.png"))
    plt.close()

categorical = [col for col in df.columns if df[col].nunique() < 10 and col != 'Churn']
for col in categorical:
    plt.figure(figsize=(8,4))
    sns.countplot(x=col, data=df, hue='Churn')
    plt.title(f"Churn by {col}")
    plt.savefig(os.path.join(reports_dir, f"churn_by_{col}.png"))
    plt.close()

for col in ['tenure', 'MonthlyCharges', 'TotalCharges']:
    plt.figure(figsize=(8,4))
    sns.boxplot(x='Churn', y=col, data=df)
    plt.title(f"{col} vs Churn")
    plt.savefig(os.path.join(reports_dir, f"{col}_vs_churn_boxplot.png"))
    plt.close()

plt.figure(figsize=(12,10))
sns.heatmap(df.select_dtypes(include=[np.number]).corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.savefig(os.path.join(reports_dir, "correlation_heatmap.png"))
plt.close()

numeric = df.select_dtypes(include=[np.number]).drop(columns=['Churn'])
vif_data = pd.DataFrame()
vif_data["feature"] = numeric.columns
vif_data["VIF"] = [variance_inflation_factor(numeric.values, i) for i in range(len(numeric.columns))]
print("\nVariance Inflation Factor (VIF):")
print(vif_data)

X = df.drop(['customerID','Churn'], axis=1)
X = pd.get_dummies(X, drop_first=True)
y = df['Churn']
mi = mutual_info_classif(X, y, discrete_features='auto')
mi_df = pd.DataFrame({'Feature': X.columns, 'Mutual Info': mi})
mi_df.sort_values(by='Mutual Info', ascending=False, inplace=True)

plt.figure(figsize=(10,6))
sns.barplot(x='Mutual Info', y='Feature', data=mi_df)
plt.title("Mutual Information with Churn")
plt.savefig(os.path.join(reports_dir, "mutual_info.png"))
plt.close()

def cramers_v(confusion_matrix):
    chi2 = chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum()
    phi2 = chi2/n
    r,k = confusion_matrix.shape
    return np.sqrt(phi2 / min(k-1, r-1))

for col1 in categorical:
    for col2 in categorical:
        if col1 != col2:
            matrix = pd.crosstab(df[col1], df[col2]).values
            cv = cramers_v(matrix)
            if cv > 0.2:
                print(f"Cramér's V between {col1} and {col2}: {cv:.2f}")

print("✅ Detailed EDA completed and plots saved to reports/")