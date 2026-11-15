# Experiment 1: Data Cleaning and Exploratory Data Analysis (EDA)
# Aim: Load a public dataset into a DataFrame and explore its features and statistics.

import pandas as pd
from sklearn.datasets import load_iris

iris = load_iris(as_frame=True)
df = iris.frame.copy()
df["species"] = df["target"].map(dict(enumerate(iris.target_names)))

print("First five rows:")
print(df.head())
print("\nDataset shape (rows, columns):", df.shape)
print("Number of input features:", len(iris.feature_names))
print("Feature names:", iris.feature_names)
print("\nData types:")
print(df.dtypes)
print("\nDataset information:")
df.info()

numeric = df[iris.feature_names]
summary = pd.DataFrame({
    "Minimum": numeric.min(),
    "Maximum": numeric.max(),
    "Mean": numeric.mean(),
    "Range": numeric.max() - numeric.min(),
    "Standard Deviation": numeric.std(),
    "Variance": numeric.var(),
    "25th Percentile": numeric.quantile(0.25),
    "50th Percentile": numeric.quantile(0.50),
    "75th Percentile": numeric.quantile(0.75)
})
print("\nStatistical summary:")
print(summary)
