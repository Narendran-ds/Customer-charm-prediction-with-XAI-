import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import pickle

print("\n🔍 Loading cleaned data...")
data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "cleaned_telco_data.csv")
df = pd.read_csv(data_path)
print(f"✅ Loaded data with shape: {df.shape}")

X = df.drop(['customerID', 'Churn'], axis=1)
y = df['Churn']

print("\n✂️ Splitting into train & test sets with stratification...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")

print("\n⚖️ Scaling features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("✅ Scaling complete.")

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
}

results = []

for name, model in models.items():
    print(f"\n🚀 Training {name}...")
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:,1]

    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1": f1_score(y_test, y_pred),
        "ROC-AUC": roc_auc_score(y_test, y_prob)
    }
    
    print(f"✅ Evaluation for {name}:")
    for metric, score in metrics.items():
        print(f"   {metric}: {score:.4f}")

    results.append({
        "Model": name,
        **metrics,
        "Trained_Model": model
    })

results_df = pd.DataFrame(results).drop(columns="Trained_Model")
print("\n📊 Summary of all models:")
print(results_df.sort_values(by="ROC-AUC", ascending=False))

best_row = results_df.sort_values(by="ROC-AUC", ascending=False).iloc[0]
best_model_name = best_row["Model"]
best_model = next(item["Trained_Model"] for item in results if item["Model"] == best_model_name)

print(f"\n🏆 Best model selected: {best_model_name} with ROC-AUC: {best_row['ROC-AUC']:.4f}")

models_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
os.makedirs(models_dir, exist_ok=True)

with open(os.path.join(models_dir, "churn_model.pkl"), "wb") as f:
    pickle.dump(best_model, f)
with open(os.path.join(models_dir, "scaler.pkl"), "wb") as f:
    pickle.dump(scaler, f)

print(f"\n✅ Saved best model to 'models/churn_model.pkl'")
print(f"✅ Saved scaler to 'models/scaler.pkl'")

