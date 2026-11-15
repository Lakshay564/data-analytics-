# Experiment 8: DBSCAN Clustering
# Aim: Discover density-based clusters and identify noise points.

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

X, _ = make_moons(n_samples=300, noise=0.08, random_state=42)
X_scaled = StandardScaler().fit_transform(X)

model = DBSCAN(eps=0.25, min_samples=5)
labels = model.fit_predict(X_scaled)

unique_labels = sorted(set(labels))
print("Cluster labels (-1 means noise):", unique_labels)
print("\nNumber of points in each cluster:")
for label in unique_labels:
    print(f"Cluster {label}: {(labels == label).sum()} points")

plt.figure(figsize=(8, 5))
for label in unique_labels:
    mask = labels == label
    name = "Noise" if label == -1 else f"Cluster {label}"
    plt.scatter(X[mask, 0], X[mask, 1], label=name, s=25)
plt.title("DBSCAN Clustering on Make-Moons Dataset")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()
plt.tight_layout()
plt.show()
