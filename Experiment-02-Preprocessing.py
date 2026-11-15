# Experiment 2: Missing Data Handling and Min-Max Normalization
# Aim: Impute missing values and scale numeric features to [0, 1].

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler

iris = load_iris(as_frame=True)
data = iris.data.copy()

# Insert a few missing values to demonstrate preprocessing.
data.iloc[0, 0] = np.nan
data.iloc[5, 1] = np.nan
data.iloc[10, 2] = np.nan

print("Missing values before treatment:")
print(data.isna().sum())

imputer = SimpleImputer(strategy="mean")
clean_data = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)

print("\nMissing values after mean imputation:")
print(clean_data.isna().sum())

scaler = MinMaxScaler()
normalized_data = pd.DataFrame(
    scaler.fit_transform(clean_data), columns=clean_data.columns
)

print("\nFirst five normalized rows:")
print(normalized_data.head())
print("\nFeature minima after scaling:")
print(normalized_data.min())
print("\nFeature maxima after scaling:")
print(normalized_data.max())
