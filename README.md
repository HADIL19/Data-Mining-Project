# 🔬 FD1 — Data Mining Interface

> **Interactive data mining pipeline** built with Streamlit — preprocessing, unsupervised clustering, PCA visualization, and evaluation metrics.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32%2B-FF4B4B?logo=streamlit)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3%2B-F7931E?logo=scikit-learn)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 📋 Overview

This app was developed as a mini-project for the **FD1 (Fouille de Données 1)** module at the **Faculty of Computer Science, USTHB** (M1 Bioinformatics, 2025–2026).

It provides a complete, interactive data mining pipeline covering:

- **Volet 1 — Preprocessing**: load, explore, clean, normalize, and visualize your dataset
- **Volet 2 — Unsupervised Clustering**: run 5 algorithms side-by-side, evaluate with silhouette score, and project results onto a 2D PCA plot

---

## ✨ Features

### 📂 Volet 1 — Preprocessing
| Feature | Details |
|---|---|
| File import | CSV and Excel (`.xlsx`, `.xls`) |
| Overview metrics | Rows, columns, numeric cols, missing values, duplicates |
| Five-number summary | Min, Q1, Median, Q3, Max + Mean, Std, Mode |
| Missing value handling | Mean / Median / Mode imputation or row removal |
| Duplicate removal | One-click deduplication |
| Normalization | Min-Max `[0,1]` or Z-score `(μ=0, σ=1)` |
| Visualizations | Boxplot, Histograms, Correlation Heatmap, Scatter Matrix |

### 🧩 Volet 2 — Clustering
| Algorithm | Type | Notes |
|---|---|---|
| **K-Means** | Partitioning | Fast, centroid-based |
| **K-Medoids** | Partitioning | Manual implementation — uses real data points, robust to outliers |
| **AGNES** | Hierarchical (bottom-up) | Configurable linkage (Ward, Complete, Average, Single) + Dendrogram |
| **DIANA** | Hierarchical (top-down) | Divisive counterpart to AGNES |
| **DBSCAN** | Density-based | No k required, detects noise points, k-NN distance helper plot |

Additional features: Elbow curve, Silhouette score comparison, inertia bar chart, PCA 2D projections for all algorithms simultaneously.

---

## 🚀 Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/fd1-data-mining.git
cd fd1-data-mining
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the app

```bash
streamlit run app.py
```

The interface opens automatically at **http://localhost:8501**

---

## ☁️ Deploy on Streamlit Community Cloud (free)

1. Push this repository to GitHub (make sure `app.py` and `requirements.txt` are at the root).
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
3. Click **"New app"** → select your repo → set the main file to `app.py`.
4. Click **Deploy** — your app will be live in ~2 minutes at a public URL.

> **Tip:** The `.streamlit/config.toml` file in this repo pre-configures the light theme and headless server mode for cloud deployment.

---

## 📁 Project Structure

```
fd1-data-mining/
├── app.py                  # Main Streamlit application
├── requirements.txt        # Python dependencies
├── .streamlit/
│   └── config.toml         # Streamlit theme & server config
└── README.md
```

---

## 🗂️ Recommended Datasets

| Dataset | Rows | Columns | Target |
|---|---|---|---|
| [Heart Disease UCI](https://archive.ics.uci.edu/dataset/45/heart+disease) | 303 | 14 | `target` (0/1) |
| [Pima Indians Diabetes](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database) | 768 | 9 | `Outcome` |
| [Iris](https://scikit-learn.org/stable/auto_examples/datasets/plot_iris_dataset.html) | 150 | 5 | `species` |

---

## 🔧 Tech Stack

| Library | Version | Purpose |
|---|---|---|
| Python | 3.10+ | Core language |
| Streamlit | ≥ 1.32 | Web UI framework |
| Pandas | ≥ 2.0 | Data manipulation |
| NumPy | ≥ 1.24 | Numerical computing |
| Scikit-learn | ≥ 1.3 | Clustering & preprocessing |
| Matplotlib | ≥ 3.7 | Charts & plots |
| Seaborn | ≥ 0.12 | Heatmaps |
| SciPy | ≥ 1.11 | Hierarchical linkage & dendrogram |

---

## 📐 Implemented Algorithms

### K-Medoids (manual implementation)
The K-Medoids algorithm was implemented **from scratch** without any external library, selecting real data points as cluster representatives to maximize robustness to outliers:

```python
class KMedoids:
    def fit_predict(self, X):
        # Select initial medoids randomly
        # Assign each point to nearest medoid
        # Update medoids: pick point minimizing intra-cluster distances
        # Repeat until convergence
```

### Normalization formulas
- **Min-Max:** `x' = (x - x_min) / (x_max - x_min)`
- **Z-score:** `x' = (x - μ) / σ`

### Silhouette Score
`s(i) = (b(i) - a(i)) / max(a(i), b(i))` — ranges from −1 to +1, higher is better.

---

## 📸 Interface Preview

The app has 3 tabs:

- **🏠 Home** — feature overview and usage guide
- **📂 Preprocessing** — data loading, cleaning, normalization, visualization
- **🧩 Clustering** — algorithm configuration, metrics, PCA projections

---

## 👤 Author

**KHELIF Hadil** — M1 Bioinformatics, USTHB  
AI & Data Science Department
Academic year: 2025–2026

---

## 📜 License

This project is released under the [MIT License](LICENSE).
