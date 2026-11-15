# Experiment 3: Histograms for all numeric features
# Aim: Visualize the distributions of Iris dataset features.

import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

iris = load_iris(as_frame=True)
data = iris.data

data.hist(figsize=(12, 8), bins=15, edgecolor="black")
plt.suptitle("Histograms of Iris Dataset Features", fontsize=16)
plt.tight_layout()
plt.show()
