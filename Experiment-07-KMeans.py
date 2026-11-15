# Experiment 7: K-Means Clustering
# Aim: Divide Iris observations into three clusters and visualize them.

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

iris = load_iris(as_frame=True)
X = iris.data[["petal length (cm)", "petal width (cm)"]]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = KMeans(n_clusters=3, random_state=42, n_init=10)
labels = model.fit_predict(X_scaled)

print("Cluster labels for first 10 records:", labels[:10])
print("\nNumber of records in each cluster:")
print(pd.Series(labels).value_counts().sort_index())

plt.figure(figsize=(8, 5))
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=labels, cmap="viridis")
centers = model.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], marker="X", s=200,
            edgecolor="black", label="Centroids")
plt.xlabel("Petal length (scaled)")
plt.ylabel("Petal width (scaled)")
plt.title("K-Means Clustering on Iris Dataset")
plt.legend()
plt.tight_layout()
plt.show()
