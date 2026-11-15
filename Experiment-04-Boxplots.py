# Experiment 4: Boxplots and IQR Outlier Detection
# Aim: Compare feature distributions and identify potential outliers.

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris

iris = load_iris(as_frame=True)
data = iris.data

plt.figure(figsize=(12, 6))
sns.boxplot(data=data)
plt.title("Boxplots of Iris Dataset Features")
plt.xticks(rotation=20)
plt.tight_layout()
plt.show()

print("Potential outliers using the 1.5 x IQR rule:")
for column in data.columns:
    q1 = data[column].quantile(0.25)
    q3 = data[column].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    outliers = data[(data[column] < lower) | (data[column] > upper)][column]
    print(f"{column}: {len(outliers)} outlier(s)")
    if not outliers.empty:
        print("  Values:", outliers.tolist())
