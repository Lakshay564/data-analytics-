# Data Analytics & Machine Learning Practicals

A beginner-friendly collection of 9 Python practicals for Google Colab and local Python.

## Experiments
1. Data Cleaning and Exploratory Data Analysis (Iris dataset)
2. Missing-value handling and Min-Max normalization
3. Histograms for every Iris feature
4. Boxplots and IQR outlier detection
5. Pima Indians Diabetes classification using Gaussian Naive Bayes
6. Simple Linear Regression using one feature from the Scikit-learn Diabetes dataset
7. K-Means clustering using Iris
8. DBSCAN clustering using a Make-Moons dataset
9. Apriori association rule mining using sample transactions

## Run in Google Colab
- Open https://colab.research.google.com/
- Select **File → Upload notebook** only if you have an .ipynb notebook; otherwise create a new notebook.
- Open each `.py` file from this repository, copy its code into a Colab code cell, and run it.
- Experiment 5 downloads a public CSV. Other datasets are loaded from scikit-learn or generated in code.
- Experiment 9 installs `mlxtend` automatically.

## Requirements
Install dependencies with:
```bash
pip install -r requirements.txt
```

Outputs (statistics, metrics, and charts) are generated when the scripts run. Exact ML scores may vary by library version or data source.

**Note:** These are educational demonstrations, not medical diagnostic tools. Association rules are assessed using support, confidence, and lift rather than conventional classification accuracy.
