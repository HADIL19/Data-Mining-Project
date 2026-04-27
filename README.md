# 🔍 Data Mining Interface 



An interactive Streamlit application covering the complete data mining pipeline:
##
**Preprocessing → Clustering → Supervised Classification**

---

## 📁 Project Structure

```
projet-fd1/
├── app.py          # Main Streamlit application
├── rapport.tex     # LaTeX report
└── README.md       # This file
```

---

## ⚙️ Installation

### 1. Place the files in a folder

```bash
mkdir projet-fd1 && cd projet-fd1
# Copy app.py here
```

### 2. Install dependencies

```bash
pip install streamlit pandas numpy scikit-learn matplotlib seaborn scipy
```

### 3. Run the application

```bash
streamlit run app.py
```

The interface opens automatically at **http://localhost:8501**

---

## 🗂️ Recommended Dataset

The project was designed to work with **heart.csv** (UCI Heart Disease Dataset).

| Property       | Value               |
|----------------|---------------------|
| Instances      | 303 rows            |
| Attributes     | 14 columns          |
| Target column  | `target` (0 or 1)   |
| Format         | CSV                 |

Download from [Kaggle](https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset) or the UCI ML Repository.

> The interface accepts **any CSV or Excel file** — not just heart.csv.

---

## 🧩 Interface Panels

### Panel 1 — Preprocessing

| Feature | Details |
|---|---|
| Import | CSV / Excel via file uploader |
| Exploration | Shape, data types, 5-number summary, descriptive stats |
| Cleaning | Fill NaN (mean / median / mode / drop rows), remove duplicates |
| Normalization | Min-Max Scaling or Z-score Standardization |
| Visualization | Boxplot, Scatter matrix, Histograms, Correlation heatmap |

### Panel 2 — Clustering

| Feature | Details |
|---|---|
| Elbow Method | Inertia vs k curve (adjustable k range) |
| K-Means | `sklearn.cluster.KMeans` |
| K-Medoids | Manual implementation (no external library) |
| AGNES | `AgglomerativeClustering` Ward linkage + dendrogram |
| DIANA | `AgglomerativeClustering` Complete linkage |
| DBSCAN | `eps` and `min_samples` adjustable via sliders |
| Evaluation | Silhouette Score, Davies-Bouldin Index, Calinski-Harabasz |
| Visualization | PCA 2D projection, inertia comparison bar chart |

### Panel 3 — Supervised Classification

| Feature | Details |
|---|---|
| Split | Stratified Train/Test split (adjustable ratio) |
| K-NN | Adjustable K |
| Decision Tree | + feature importance chart |
| Naive Bayes | Gaussian |
| SVM | RBF kernel |
| Logistic Regression | Multi-class |
| Random Forest | 100 trees + feature importance chart |
| Evaluation | Confusion matrix, Accuracy, Precision, Recall, F1-score |
| Comparison | All models benchmarked in one click |

---

## 🔄 Recommended Workflow

```
1. Load the CSV/Excel file (Panel 1)
        ↓
2. Explore → Clean → Normalize (Panel 1)
        ↓
3. Plot Elbow → Choose k → Run clustering (Panel 2)
        ↓
4. Select a model → Train → Analyze results (Panel 3)
```

> ⚠️ **Normalization must be done first** (Panel 1) before accessing Panels 2 and 3.

---

## 📦 Dependencies

```
streamlit>=1.30
pandas>=2.0
numpy>=1.24
scikit-learn>=1.3
matplotlib>=3.7
seaborn>=0.12
scipy>=1.11
```

---

## 📐 Manually Implemented Algorithms

### K-Medoids

```python
class KMedoids:
    def fit_predict(self, X):
        # 1. Randomly initialize k medoids
        # 2. Assign each point to the nearest medoid
        # 3. Update: new medoid = point that minimizes total intra-cluster distance
        # 4. Repeat until convergence
```

---

## 📊 Evaluation Metrics

| Metric | Formula | Interpretation |
|---|---|---|
| Silhouette | s = (b - a) / max(a, b) | closer to 1 = better |
| Davies-Bouldin | DB = (1/k) * sum(max((si+sj)/d(ci,cj))) | closer to 0 = better |
| Calinski-Harabasz | CH = SSB*(k-1) / SSW*(n-k) | larger = better |
| Accuracy | (TP+TN) / (TP+TN+FP+FN) | closer to 1 = better |
| F1-score | 2 * P * R / (P + R) | closer to 1 = better |

---

## 👨‍💻 Project Info

- **Module** : Data Mining 1 (FD1)
- **Program** : M1 Bioinformatics
- **Department** : AI & Data Science
- **Academic Year** : 2025 – 2026
