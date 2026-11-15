# Experiment 5: Pima Indians Diabetes Classification using Gaussian Naive Bayes
# Aim: Train and evaluate a Naive Bayes classifier on the public Pima dataset.
# Educational demonstration only; not a medical diagnostic tool.

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

url = "https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv"
try:
    diabetes = pd.read_csv(url)
except Exception as exc:
    raise RuntimeError(
        "Could not download the dataset. In Colab, upload diabetes.csv "
        "and replace pd.read_csv(url) with pd.read_csv('/content/diabetes.csv')."
    ) from exc

print("Dataset shape:", diabetes.shape)
print(diabetes.head())

X = diabetes.drop(columns=["Outcome"])
y = diabetes["Outcome"]

# Zero values in these measurements are commonly treated as missing.
zero_as_missing = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
X[zero_as_missing] = X[zero_as_missing].replace(0, np.nan)
X[zero_as_missing] = X[zero_as_missing].fillna(X[zero_as_missing].median())

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = GaussianNB()
model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)

print("\nAccuracy:", round(accuracy_score(y_test, y_pred) * 100, 2), "%")
print("\nConfusion matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification report:")
print(classification_report(y_test, y_pred, zero_division=0))
