# Experiment 6: Simple Linear Regression
# Aim: Predict the diabetes progression target from one input feature (BMI).

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

dataset = load_diabetes(as_frame=True)
X = dataset.data[["bmi"]]
y = dataset.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("Intercept:", round(model.intercept_, 3))
print("Slope:", round(model.coef_[0], 3))
print("Mean Squared Error:", round(mean_squared_error(y_test, y_pred), 3))
print("R-squared:", round(r2_score(y_test, y_pred), 3))

order = np.argsort(X_test["bmi"].values)
plt.figure(figsize=(8, 5))
plt.scatter(X_test["bmi"], y_test, label="Actual values")
plt.plot(X_test["bmi"].values[order], y_pred[order],
         color="red", label="Regression line")
plt.xlabel("BMI feature (standardized)")
plt.ylabel("Disease progression target")
plt.title("Simple Linear Regression")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
