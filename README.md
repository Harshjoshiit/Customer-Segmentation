# Customer Segmentation (K‑Means + PCA)

## Project overview

This notebook performs customer segmentation using K‑Means clustering on a tabular customer dataset (`segmentation data.csv`). The pipeline applies feature standardization, principal component analysis (PCA) for dimensionality reduction, uses the elbow method to choose the number of clusters, fits K‑Means, profiles each cluster, visualizes results, and saves preprocessing + model artifacts.
---

## Table of contents

* [Dataset](#dataset)
* [What the notebook does](#what-the-notebook-does)
* [Key choices & results (from notebook)](#key-choices--results-from-notebook)
* [How to run](#how-to-run)
* [Quick usage / inference example](#quick-usage--inference-example)
* [File structure](#file-structure)
* [Improvements & next steps](#improvements--next-steps)
* [Dependencies](#dependencies)

---
## What the notebook does (high level)

1. Imports standard data science libraries (pandas, numpy, matplotlib, seaborn) and scikit-learn (`StandardScaler`, `PCA`, `KMeans`).
2. Loads `segmentation data.csv` into `df_segmentation` and removes the `ID` column.
3. Performs exploratory data analysis: `head()`, `describe()`, correlation heatmap and distribution plots to understand features and relationships.
4. Standardizes numerical features using `StandardScaler` into `segmentation_std`.
5. Runs PCA to inspect explained variance and chooses `n_components = 3` for dimensionality reduction. Visualizes principal component structure (biplot / heatmap of PCA components).
6. Uses the elbow method (WCSS / inertia) over `k=1..10` to choose the number of clusters, then trains `KMeans(n_clusters=4)` on the standardized data and on PCA scores.
7. Creates cluster labels, maps them to descriptive segment names (`fall-off`, `fewer-opportunities`, `standard`, `career-focused`) and computes cluster-level summaries (means of features per cluster, N observations, probability per cluster).
8. Visualizes clustering on the first two PCA axes and exports the preprocessing and model artifacts with `pickle`:

   * `scaler.pickle`
   * `pca.pickle`
   * `kmeans_pca.pickle`

---

## Key choices & results (from the notebook code)

* **Dimensionality reduction:** PCA was fitted and `n_components = 3` chosen.
* **Cluster selection:** Elbow method (WCSS) was computed for `k` in 1..10; the notebook selected **k = 4** for K‑Means.
* **Cluster labels:** The notebook assigns human-friendly names to clusters: `fall-off`, `fewer-opportunities`, `standard`, `career-focused`.
* **Artifacts exported:** `scaler.pickle`, `pca.pickle`, `kmeans_pca.pickle` (ready to be reused for scoring new data).

---

## How to run (local)

1. Ensure the CSV `segmentation data.csv` is in the same folder as the notebook.
3. Open `Customer_analytics.ipynb` in Jupyter or Google Colab and run cells in order.

**Suggested `requirements.txt`**

```
pandas
numpy
scikit-learn
matplotlib
seaborn
```

---

## Quick usage / scoring example (concept)

```python
import pickle
import pandas as pd

# load artifacts
scaler = pickle.load(open('scaler.pickle','rb'))
pca = pickle.load(open('pca.pickle','rb'))
kmeans = pickle.load(open('kmeans_pca.pickle','rb'))

# prepare new sample (same columns as training order)
sample = pd.DataFrame([{
  # fill keys same as df_segmentation columns (except ID)
}])

# apply scaler -> pca -> predict
X_scaled = scaler.transform(sample)
X_pca = pca.transform(X_scaled)
label = kmeans.predict(X_pca)

print('Cluster label:', label)
```

---

## Improvements & next steps

* Add cluster-quality metrics (silhouette score, Calinski-Harabasz) and show values alongside elbow plots.
* Build a reproducible `sklearn` pipeline that includes scaling → PCA → KMeans and serialize the entire pipeline (e.g., `joblib`) to simplify deployment.
* Consider alternative clustering (Gaussian Mixture, Agglomerative, DBSCAN) and compare with internal/external metrics.
* If transactional data is available, integrate RFM (Recency, Frequency, Monetary) features before clustering for more business‑centric segments.
* Expose an interactive dashboard (Plotly Dash / Streamlit) to let business users slice segments and inspect customers.

---

## License & contact

This repository may be licensed under MIT. For questions or edits, add your contact details in the notebook metadata or README.

---

*README generated after direct inspection of `Customer_analytics.ipynb`. If you want, I can also run the notebook (if you upload `segmentation data.csv`) and update the README with exact counts and numeric metrics.*
