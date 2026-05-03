import io, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import streamlit as st

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import (
    silhouette_score,
    confusion_matrix, classification_report,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from scipy.cluster.hierarchy import dendrogram, linkage

# ── Colour palette — clean light theme ──────────────────────────
BG       = "#f6f8fc"
SURFACE  = "#ffffff"
SURFACE2 = "#f1f5f9"
BORDER   = "#dde3ed"
TEXT     = "#0f1c2e"
TEXT2    = "#5a6a82"
ACCENT   = "#2563eb"
GREEN    = "#16a34a"
AMBER    = "#d97706"
RED      = "#dc2626"
PURPLE   = "#7c3aed"
CYAN     = "#0891b2"

CLUSTER_PALETTE = [ACCENT, GREEN, AMBER, RED, PURPLE, CYAN,
                   "#db2777", "#ea580c", "#0284c7", "#15803d"]

# ── Matplotlib light theme ───────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor":  "#ffffff",
    "axes.facecolor":    "#f8fafc",
    "axes.edgecolor":    "#dde3ed",
    "axes.labelcolor":   TEXT2,
    "axes.titlecolor":   TEXT,
    "axes.titlesize":    12,
    "axes.labelsize":    10,
    "xtick.color":       TEXT2,
    "ytick.color":       TEXT2,
    "xtick.labelsize":   9,
    "ytick.labelsize":   9,
    "text.color":        TEXT,
    "grid.color":        "#e8edf5",
    "grid.alpha":        0.8,
    "legend.facecolor":  "#ffffff",
    "legend.edgecolor":  "#dde3ed",
    "legend.fontsize":   9,
    "figure.dpi":        120,
    "savefig.facecolor": "#ffffff",
})

# ═══════════════════════════════════════════════════════════════
#  PAGE CONFIG + CSS
# ═══════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="FD1 · Fouille de Données",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

html, body { background: #f0f4f9 !important; color: #0f1c2e !important; }

.stApp, .stApp > div,
[data-testid="stAppViewContainer"],
[data-testid="stAppViewBlockContainer"],
[data-testid="stMain"],
[data-testid="stMainBlockContainer"],
section[data-testid="stSidebar"] ~ div,
.main, .main > div {
    background: #f0f4f9 !important;
    color: #0f1c2e !important;
}

div:not([class*="stDataFrame"]):not([class*="dvn"]):not(canvas),
section, article, aside, main {
    font-family: 'DM Sans', sans-serif !important;
}

#MainMenu, footer, header { visibility: hidden; }
.block-container {
    padding: 1.5rem 2.5rem 3rem !important;
    max-width: 1400px !important;
    background: transparent !important;
}

[data-testid="stSidebar"],
[data-testid="stSidebar"] > div,
[data-testid="stSidebar"] section {
    background: #ffffff !important;
    color: #0f1c2e !important;
    border-right: 1px solid #dde3ed !important;
}

[data-baseweb="tab-list"] {
    background: #ffffff !important;
    border-radius: 12px !important;
    padding: 5px !important;
    border: 1px solid #dde3ed !important;
    box-shadow: 0 1px 6px rgba(15,28,46,0.07) !important;
}
[data-baseweb="tab"] {
    background: transparent !important;
    border-radius: 8px !important;
    color: #5a6a82 !important;
    font-weight: 500 !important;
    font-size: 13px !important;
    padding: 7px 20px !important;
}
[aria-selected="true"][data-baseweb="tab"] {
    background: #2563eb !important;
    color: #ffffff !important;
    box-shadow: 0 2px 8px rgba(37,99,235,.3) !important;
}
[data-baseweb="tab-highlight"], [data-baseweb="tab-border"] { display: none !important; }

.stButton > button {
    background: #2563eb !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 13px !important;
    font-weight: 600 !important;
    padding: 8px 20px !important;
    box-shadow: 0 2px 6px rgba(37,99,235,.25) !important;
    transition: all .15s ease !important;
}
.stButton > button:hover {
    background: #1d4ed8 !important;
    box-shadow: 0 4px 14px rgba(37,99,235,.4) !important;
    transform: translateY(-1px) !important;
}

[data-testid="stFileUploader"] {
    background: #ffffff !important;
    border: 2px dashed #c7d2e0 !important;
    border-radius: 12px !important;
}
[data-testid="stFileUploader"] * { color: #0f1c2e !important; background: transparent !important; }

[data-baseweb="select"] > div,
[data-baseweb="input"] > div,
.stTextInput > div > div,
.stSelectbox > div > div,
.stMultiSelect > div > div,
[data-baseweb="input"],
input, select, textarea {
    background: #ffffff !important;
    border-color: #dde3ed !important;
    border-radius: 8px !important;
    color: #0f1c2e !important;
    font-size: 13px !important;
}
[data-baseweb="select"] [data-testid="stMarkdownContainer"],
[data-testid="stRadio"] label,
[data-testid="stCheckbox"] label,
[data-testid="stSelectbox"] label,
[data-testid="stMultiSelect"] label,
[data-testid="stSlider"] label,
[data-testid="stNumberInput"] label,
[data-testid="stNumberInput"] input {
    color: #0f1c2e !important;
    background: transparent !important;
}
[data-testid="stNumberInput"] input { background: #ffffff !important; }
[data-baseweb="tag"] {
    background: rgba(37,99,235,.10) !important;
    border-color: rgba(37,99,235,.25) !important;
    color: #2563eb !important;
    border-radius: 6px !important;
    font-size: 12px !important;
}

[data-testid="stMetric"] {
    background: #ffffff !important;
    border: 1px solid #dde3ed !important;
    border-radius: 10px !important;
    padding: 14px 18px !important;
    box-shadow: 0 1px 4px rgba(15,28,46,0.05) !important;
}
[data-testid="stMetricLabel"] {
    color: #5a6a82 !important;
    font-size: 11px !important;
    text-transform: uppercase;
    letter-spacing: .06em;
}
[data-testid="stMetricValue"] {
    color: #0f1c2e !important;
    font-size: 24px !important;
    font-weight: 700 !important;
}

[data-testid="stExpander"] {
    background: #ffffff !important;
    border: 1px solid #dde3ed !important;
    border-radius: 12px !important;
    box-shadow: 0 1px 4px rgba(15,28,46,0.05) !important;
    overflow: hidden !important;
}
[data-testid="stExpander"] > details,
[data-testid="stExpander"] details,
[data-testid="stExpander"] details > summary,
[data-testid="stExpander"] details > div,
[data-testid="stExpander"] summary,
[data-testid="stExpander"] > div,
.streamlit-expanderContent,
.streamlit-expanderContent > div {
    background: #ffffff !important;
    color: #0f1c2e !important;
}
[data-testid="stExpander"] summary {
    font-size: 13px !important;
    font-weight: 600 !important;
    color: #0f1c2e !important;
    padding: 12px 16px !important;
    border-radius: 12px !important;
}
[data-testid="stExpander"] summary:hover { color: #2563eb !important; }
[data-testid="stExpander"] summary svg { fill: #0f1c2e !important; }

[data-testid="stDataFrame"] {
    border: 1px solid #dde3ed !important;
    border-radius: 10px !important;
    overflow: visible !important;
    box-shadow: 0 1px 4px rgba(15,28,46,0.04) !important;
}
.dvn-scroller { background: transparent !important; }

[data-testid="stProgress"] > div > div { background: #2563eb !important; }
[data-testid="stProgress"] > div { background: #e2e8f0 !important; }

[data-testid="stSpinner"] { color: #2563eb !important; }

.stAlert { border-radius: 10px !important; font-size: 13px !important; }

[data-baseweb="radio"] { gap: 8px !important; }
[data-testid="stRadio"] label { font-size: 13px !important; }
[data-testid="stRadio"] label:has(input:checked) { font-weight: 600 !important; }
[data-testid="stRadio"] input { accent-color: #2563eb !important; }

hr { border-color: #dde3ed !important; margin: 1.2rem 0 !important; }
code {
    background: #f1f5f9 !important;
    border: 1px solid #dde3ed !important;
    border-radius: 5px !important;
    padding: 1px 5px !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 12px !important;
    color: #2563eb !important;
}

[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] span,
[data-testid="stMarkdownContainer"] li,
[data-testid="stMarkdownContainer"] h1,
[data-testid="stMarkdownContainer"] h2,
[data-testid="stMarkdownContainer"] h3 {
    color: #0f1c2e !important;
}

[data-baseweb="popover"],
[data-baseweb="menu"],
[data-baseweb="menu"] ul,
[data-baseweb="menu"] li {
    background: #ffffff !important;
    color: #0f1c2e !important;
    border-color: #dde3ed !important;
}
[data-baseweb="option"]:hover { background: #eff6ff !important; }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
#  HELPER COMPONENTS
# ═══════════════════════════════════════════════════════════════

def section_title(icon: str, title: str, subtitle: str = ""):
    sub_html = f'<span style="color:#5a6a82;font-size:12px;font-weight:400;margin-left:8px;">{subtitle}</span>' if subtitle else ""
    st.markdown(f"""
    <div style="display:flex;align-items:center;gap:10px;margin:1.4rem 0 .9rem;">
        <div style="width:32px;height:32px;background:rgba(37,99,235,.10);border:1px solid rgba(37,99,235,.2);
                    border-radius:8px;display:flex;align-items:center;justify-content:center;font-size:15px;">{icon}</div>
        <span style="font-size:15px;font-weight:700;color:#0f1c2e;">{title}</span>{sub_html}
    </div>
    """, unsafe_allow_html=True)


def metric_row(metrics: list):
    cols = st.columns(len(metrics))
    for col, m in zip(cols, metrics):
        color = m.get("color", ACCENT)
        delta_html = f'<div style="font-size:11px;color:{GREEN};margin-top:2px;">{m["delta"]}</div>' if "delta" in m else ""
        with col:
            st.markdown(f"""
            <div style="background:#ffffff;border:1px solid #dde3ed;border-radius:12px;
                        padding:16px;text-align:center;box-shadow:0 1px 4px rgba(15,28,46,.06);">
                <div style="font-size:10px;color:#5a6a82;text-transform:uppercase;
                            letter-spacing:.09em;margin-bottom:6px;font-weight:500;">{m['label']}</div>
                <div style="font-size:24px;font-weight:700;color:{color};font-family:'JetBrains Mono',monospace;">{m['value']}</div>
                {delta_html}
            </div>
            """, unsafe_allow_html=True)


def info_box(text: str, kind: str = "info"):
    styles = {
        "info":    (ACCENT,  "rgba(37,99,235,.07)",  "rgba(37,99,235,.18)", "ℹ️"),
        "success": (GREEN,   "rgba(22,163,74,.07)",   "rgba(22,163,74,.18)", "✅"),
        "warning": (AMBER,   "rgba(217,119,6,.07)",   "rgba(217,119,6,.18)", "⚠️"),
        "error":   (RED,     "rgba(220,38,38,.07)",   "rgba(220,38,38,.18)", "❌"),
    }
    c, bg, border, icon = styles.get(kind, styles["info"])
    st.markdown(f"""
    <div style="background:{bg};border:1px solid {border};border-radius:10px;
                padding:11px 14px;margin:6px 0;font-size:13px;
                display:flex;gap:8px;align-items:flex-start;">
        <span style="margin-top:1px;flex-shrink:0;">{icon}</span>
        <span style="color:#0f1c2e;">{text}</span>
    </div>
    """, unsafe_allow_html=True)


def page_header():
    st.markdown("""
    <div style="display:flex;align-items:center;justify-content:space-between;
                padding:1rem 0 1.4rem;border-bottom:2px solid #dde3ed;margin-bottom:1.8rem;">
        <div>
            <div style="display:flex;align-items:center;gap:12px;margin-bottom:4px;">
                <div style="width:38px;height:38px;background:linear-gradient(135deg,#2563eb,#7c3aed);
                            border-radius:10px;display:flex;align-items:center;justify-content:center;
                            font-size:20px;box-shadow:0 4px 12px rgba(37,99,235,.3);">🔬</div>
                <span style="font-size:22px;font-weight:700;color:#0f1c2e;letter-spacing:-.4px;">Interface Fouille de Données</span>
                <span style="background:rgba(37,99,235,.10);border:1px solid rgba(37,99,235,.25);
                             color:#2563eb;border-radius:999px;padding:3px 12px;font-size:11px;font-weight:700;">FD1</span>
            </div>
            <div style="color:#5a6a82;font-size:12px;margin-left:50px;">
                Prétraitement · Clustering &nbsp;·&nbsp; M1 Bioinformatique 2025–2026
            </div>
        </div>
        <div style="display:flex;gap:8px;align-items:center;background:#f0fdf4;
                    border:1px solid rgba(22,163,74,.25);border-radius:8px;padding:6px 12px;">
            <div style="width:8px;height:8px;background:#16a34a;border-radius:50%;box-shadow:0 0 6px #16a34a55;"></div>
            <span style="color:#16a34a;font-size:12px;font-weight:600;">Prêt</span>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
#  K-MEDOIDS (manuel)
# ═══════════════════════════════════════════════════════════════
class KMedoids:
    def __init__(self, n_clusters=3, max_iter=300, random_state=42):
        self.n_clusters, self.max_iter, self.random_state = n_clusters, max_iter, random_state

    def fit_predict(self, X):
        rng = np.random.default_rng(self.random_state)
        n   = X.shape[0]
        med = rng.choice(n, self.n_clusters, replace=False)
        for _ in range(self.max_iter):
            labels  = self._assign(X, med)
            new_med = med.copy()
            for k in range(self.n_clusters):
                members = np.where(labels == k)[0]
                if len(members) == 0: continue
                sub  = X[members]
                dist = np.sum(np.linalg.norm(sub[:, None] - sub[None, :], axis=2), axis=1)
                new_med[k] = members[np.argmin(dist)]
            if np.all(new_med == med): break
            med = new_med
        self.medoid_indices_ = med
        return self._assign(X, med)

    def _assign(self, X, med):
        dist = np.linalg.norm(X[:, None] - X[med][None, :], axis=2)
        return np.argmin(dist, axis=1)


# ═══════════════════════════════════════════════════════════════
#  SESSION STATE
# ═══════════════════════════════════════════════════════════════
for key, val in [("df_raw", None), ("df_clean", None), ("X_scaled", None),
                 ("X_features", None), ("y_target", None), ("scaler_name", None)]:
    if key not in st.session_state:
        st.session_state[key] = val


# ═══════════════════════════════════════════════════════════════
#  SIDEBAR
# ═══════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style="padding:.2rem 0 1.4rem;">
        <div style="display:flex;align-items:center;gap:8px;margin-bottom:4px;">
            <div style="width:32px;height:32px;background:linear-gradient(135deg,#2563eb,#7c3aed);
                        border-radius:8px;display:flex;align-items:center;justify-content:center;font-size:16px;">🔬</div>
            <div>
                <div style="font-size:14px;font-weight:700;color:#0f1c2e;">FD1 · Data Mining</div>
                <div style="font-size:10px;color:#5a6a82;">Pipeline interactif</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div style="font-size:10px;text-transform:uppercase;letter-spacing:.12em;color:#5a6a82;margin-bottom:8px;font-weight:600;">Navigation</div>', unsafe_allow_html=True)
    for icon, label in [("🏠", "Accueil"), ("📂", "Prétraitement"), ("🧩", "Clustering")]:
        st.markdown(f'<div style="display:flex;align-items:center;gap:10px;padding:9px 12px;border-radius:8px;margin-bottom:3px;color:#5a6a82;font-size:13px;font-weight:500;">{icon} {label}</div>', unsafe_allow_html=True)

    st.markdown('<hr style="margin:12px 0;">', unsafe_allow_html=True)
    st.markdown('<div style="font-size:10px;text-transform:uppercase;letter-spacing:.12em;color:#5a6a82;margin-bottom:8px;font-weight:600;">Jeu de données</div>', unsafe_allow_html=True)

    if st.session_state.df_raw is not None:
        df_info = st.session_state.df_raw
        missing = int(df_info.isnull().sum().sum())
        scaled_ok = st.session_state.X_scaled is not None
        st.markdown(f"""
        <div style="background:#f6f8fc;border:1px solid #dde3ed;border-radius:10px;padding:12px 14px;">
            <div style="font-size:12px;color:#0f1c2e;font-weight:600;margin-bottom:8px;">
                📄 {st.session_state.get("fname","dataset.csv")}
            </div>
            <div style="display:grid;grid-template-columns:1fr 1fr;gap:8px;">
                <div style="background:#fff;border:1px solid #dde3ed;border-radius:7px;padding:8px;text-align:center;">
                    <div style="font-size:10px;color:#5a6a82;font-weight:500;">Lignes</div>
                    <div style="font-size:16px;font-weight:700;color:#2563eb;">{df_info.shape[0]:,}</div>
                </div>
                <div style="background:#fff;border:1px solid #dde3ed;border-radius:7px;padding:8px;text-align:center;">
                    <div style="font-size:10px;color:#5a6a82;font-weight:500;">Colonnes</div>
                    <div style="font-size:16px;font-weight:700;color:#2563eb;">{df_info.shape[1]}</div>
                </div>
                <div style="background:#fff;border:1px solid #dde3ed;border-radius:7px;padding:8px;text-align:center;">
                    <div style="font-size:10px;color:#5a6a82;font-weight:500;">Manquants</div>
                    <div style="font-size:16px;font-weight:700;color:{'#dc2626' if missing>0 else '#16a34a'};">{missing}</div>
                </div>
                <div style="background:#fff;border:1px solid #dde3ed;border-radius:7px;padding:8px;text-align:center;">
                    <div style="font-size:10px;color:#5a6a82;font-weight:500;">Normalisé</div>
                    <div style="font-size:16px;font-weight:700;color:{'#16a34a' if scaled_ok else '#5a6a82'};">{'✓' if scaled_ok else '–'}</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="background:#f6f8fc;border:2px dashed #c7d2e0;border-radius:10px;
                    padding:18px;text-align:center;color:#5a6a82;font-size:12px;">
            <div style="font-size:24px;margin-bottom:6px;">📂</div>Aucun dataset chargé
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<hr style="margin:12px 0;">', unsafe_allow_html=True)
    st.markdown("""
    <div style="font-size:11px;color:#5a6a82;text-align:center;line-height:1.7;">
        M1 BioInformatique · FD1<br>Faculté d'Informatique<br>Année 2025–2026
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════
page_header()

tab0, tab1, tab2 = st.tabs([
    "  🏠  Accueil  ",
    "  📂  Prétraitement  ",
    "  🧩  Clustering  ",
])


# ───────────────────────────────────────────────────────────────
#  VOLET 0 : PAGE D'ACCUEIL
# ───────────────────────────────────────────────────────────────
with tab0:
    st.markdown("""
    <div style="background:linear-gradient(135deg,#eff6ff 0%,#f5f3ff 60%,#faf5ff 100%);
                border:1px solid #c7d7fd;border-radius:20px;padding:3rem 3rem 2.5rem;margin-bottom:2rem;position:relative;overflow:hidden;">
        <div style="position:absolute;top:-50px;right:-50px;width:220px;height:220px;
                    background:radial-gradient(circle,rgba(37,99,235,.13),transparent 70%);border-radius:50%;"></div>
        <div style="position:absolute;bottom:-40px;left:35%;width:170px;height:170px;
                    background:radial-gradient(circle,rgba(124,58,237,.10),transparent 70%);border-radius:50%;"></div>
        <div style="position:relative;z-index:1;">
            <div style="display:flex;align-items:center;gap:14px;margin-bottom:1rem;">
                <div style="width:60px;height:60px;background:linear-gradient(135deg,#2563eb,#7c3aed);
                            border-radius:18px;display:flex;align-items:center;justify-content:center;
                            font-size:30px;box-shadow:0 8px 24px rgba(37,99,235,.35);">🔬</div>
                <div>
                    <div style="font-size:30px;font-weight:800;color:#0f1c2e;letter-spacing:-.6px;line-height:1.1;">Interface Fouille de Données</div>
                    <div style="font-size:14px;color:#5a6a82;margin-top:3px;">Mini-Projet · Module FD1 · Faculté d'Informatique · 2025–2026</div>
                </div>
            </div>
            <p style="font-size:15px;color:#334155;max-width:700px;line-height:1.8;margin:0 0 1.5rem;">
                Application interactive de <strong style="color:#2563eb;">fouille de données</strong> automatisant
                l'intégralité du pipeline — du chargement des données brutes à l'évaluation de modèles
                de machine learning — sans écrire une seule ligne de code.
            </p>
            <div style="display:flex;gap:10px;flex-wrap:wrap;">
                <span style="background:#2563eb;color:#fff;padding:6px 16px;border-radius:8px;font-size:13px;font-weight:600;box-shadow:0 2px 8px rgba(37,99,235,.3);">📂 Prétraitement</span>
                <span style="background:#7c3aed;color:#fff;padding:6px 16px;border-radius:8px;font-size:13px;font-weight:600;box-shadow:0 2px 8px rgba(124,58,237,.3);">🧩 Clustering</span>
                <span style="background:rgba(22,163,74,.12);color:#16a34a;border:1px solid rgba(22,163,74,.3);padding:6px 16px;border-radius:8px;font-size:13px;font-weight:600;">✅ CSV / Excel</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div style="font-size:18px;font-weight:700;color:#0f1c2e;margin-bottom:1rem;">Fonctionnalités principales</div>', unsafe_allow_html=True)
    cv1, cv2 = st.columns(2)

    for col, border_color, icon, vtitle, vdesc, items in [
        (cv1, "#2563eb", "📂", "Volet 1 — Prétraitement",
         "Préparez vos données étape par étape avant toute analyse.",
         [("→","#2563eb","Importation CSV / Excel"),("→","#2563eb","Statistiques descriptives"),
          ("→","#2563eb","Gestion des valeurs manquantes"),("→","#2563eb","Normalisation Min-Max / Z-score"),
          ("→","#2563eb","Boxplot · Histogramme · Heatmap")]),
        (cv2, "#7c3aed", "🧩", "Volet 2 — Clustering",
         "Explorez les structures cachées dans vos données.",
         [("→","#7c3aed","K-Means & K-Médoïdes"),("→","#7c3aed","AGNES (ascendant) & DIANA (divisif)"),
          ("→","#7c3aed","DBSCAN (densité)"),("→","#7c3aed","Courbe d'Elbow + Score Silhouette"),
          ("→","#7c3aed","Projection PCA 2D + Dendrogramme")]),
    ]:
        with col:
            items_html = "".join([
                f'<div style="display:flex;align-items:center;gap:8px;font-size:13px;color:#334155;">'
                f'<span style="color:{ic};font-weight:700;">{ar}</span> {txt}</div>'
                for ar, ic, txt in items
            ])
            st.markdown(f"""
            <div style="background:#ffffff;border:1px solid #dde3ed;border-radius:16px;padding:22px;
                        box-shadow:0 2px 8px rgba(15,28,46,.06);border-top:4px solid {border_color};height:100%;">
                <div style="font-size:28px;margin-bottom:10px;">{icon}</div>
                <div style="font-size:16px;font-weight:700;color:#0f1c2e;margin-bottom:6px;">{vtitle}</div>
                <div style="font-size:13px;color:#5a6a82;line-height:1.7;margin-bottom:14px;">{vdesc}</div>
                <div style="display:flex;flex-direction:column;gap:6px;">{items_html}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown('<div style="font-size:18px;font-weight:700;color:#0f1c2e;margin-bottom:1rem;">Comment utiliser l\'interface ?</div>', unsafe_allow_html=True)
    steps = [
        ("1","#2563eb","Importer un dataset","Allez dans <strong>Prétraitement</strong> et chargez votre fichier CSV ou Excel."),
        ("2","#16a34a","Préparer les données","Nettoyez (valeurs manquantes, doublons) puis normalisez avec Min-Max ou Z-score."),
        ("3","#7c3aed","Explorer les clusters","Dans <strong>Clustering</strong>, lancez les algorithmes et analysez les scores et projections PCA."),
    ]
    sc = st.columns(3)
    for col, (num, color, title, desc) in zip(sc, steps):
        with col:
            st.markdown(f"""
            <div style="background:#ffffff;border:1px solid #dde3ed;border-radius:14px;padding:20px;
                        box-shadow:0 1px 6px rgba(15,28,46,.05);text-align:center;">
                <div style="width:42px;height:42px;background:{color};color:#fff;border-radius:50%;
                            display:flex;align-items:center;justify-content:center;font-size:19px;font-weight:800;
                            margin:0 auto 12px;box-shadow:0 4px 12px {color}44;">{num}</div>
                <div style="font-size:13px;font-weight:700;color:#0f1c2e;margin-bottom:6px;">{title}</div>
                <div style="font-size:12px;color:#5a6a82;line-height:1.6;">{desc}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    col_ds, col_tech = st.columns(2)

    with col_ds:
        st.markdown("""
        <div style="background:#ffffff;border:1px solid #dde3ed;border-radius:14px;padding:20px;box-shadow:0 1px 6px rgba(15,28,46,.05);">
            <div style="font-size:14px;font-weight:700;color:#0f1c2e;margin-bottom:12px;">📋 Datasets recommandés</div>
            <div style="display:flex;flex-direction:column;gap:8px;">
                <div style="background:#eff6ff;border:1px solid rgba(37,99,235,.2);border-radius:8px;padding:10px 14px;">
                    <div style="font-size:12px;font-weight:600;color:#2563eb;">Pima Indians Diabetes</div>
                    <div style="font-size:11px;color:#5a6a82;">768 lignes · 9 colonnes · Classification binaire</div>
                </div>
                <div style="background:#fdf4ff;border:1px solid rgba(124,58,237,.2);border-radius:8px;padding:10px 14px;">
                    <div style="font-size:12px;font-weight:600;color:#7c3aed;">Iris Dataset</div>
                    <div style="font-size:11px;color:#5a6a82;">150 lignes · 5 colonnes · 3 classes</div>
                </div>
                <div style="background:#fff7ed;border:1px solid rgba(217,119,6,.2);border-radius:8px;padding:10px 14px;">
                    <div style="font-size:12px;font-weight:600;color:#d97706;">Heart Disease UCI</div>
                    <div style="font-size:11px;color:#5a6a82;">303 lignes · 14 colonnes · Classification</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col_tech:
        st.markdown("""
        <div style="background:#ffffff;border:1px solid #dde3ed;border-radius:14px;padding:20px;box-shadow:0 1px 6px rgba(15,28,46,.05);">
            <div style="font-size:14px;font-weight:700;color:#0f1c2e;margin-bottom:12px;">⚙️ Technologies utilisées</div>
            <div style="display:grid;grid-template-columns:1fr 1fr;gap:8px;">
                <div style="background:#eff6ff;border:1px solid rgba(37,99,235,.15);border-radius:8px;padding:12px;text-align:center;">
                    <div style="font-size:22px;">🐍</div>
                    <div style="font-size:11px;font-weight:700;color:#2563eb;margin-top:4px;">Python 3</div>
                </div>
                <div style="background:#fdf4ff;border:1px solid rgba(124,58,237,.15);border-radius:8px;padding:12px;text-align:center;">
                    <div style="font-size:22px;">🎈</div>
                    <div style="font-size:11px;font-weight:700;color:#7c3aed;margin-top:4px;">Streamlit</div>
                </div>
                <div style="background:#f0fdf4;border:1px solid rgba(22,163,74,.15);border-radius:8px;padding:12px;text-align:center;">
                    <div style="font-size:22px;">🤖</div>
                    <div style="font-size:11px;font-weight:700;color:#16a34a;margin-top:4px;">Scikit-learn</div>
                </div>
                <div style="background:#fff7ed;border:1px solid rgba(217,119,6,.15);border-radius:8px;padding:12px;text-align:center;">
                    <div style="font-size:22px;">📊</div>
                    <div style="font-size:11px;font-weight:700;color:#d97706;margin-top:4px;">Pandas · Matplotlib</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)


# ───────────────────────────────────────────────────────────────
#  VOLET 1 : PRÉTRAITEMENT
# ───────────────────────────────────────────────────────────────
with tab1:
    section_title("📂", "Importation du Dataset", "CSV ou Excel")
    uploaded = st.file_uploader("Chargez votre dataset", type=["csv", "xlsx", "xls"], label_visibility="collapsed")

    if uploaded:
        try:
            df = pd.read_csv(uploaded) if uploaded.name.endswith(".csv") else pd.read_excel(uploaded)
            df = df.loc[:, ~df.columns.str.startswith('Unnamed')]
            df = df.dropna(axis=1, how='all')
            st.session_state.df_raw = df.copy()
            st.session_state["fname"] = uploaded.name
            info_box(f"Fichier <strong>{uploaded.name}</strong> chargé — {df.shape[0]:,} lignes × {df.shape[1]} colonnes", "success")
        except Exception as e:
            info_box(str(e), "error")

    if st.session_state.df_raw is None:
        st.markdown("""
        <div style="text-align:center;padding:4rem 2rem;color:#5a6a82;background:#ffffff;
                    border:2px dashed #c7d2e0;border-radius:16px;margin-top:1rem;">
            <div style="font-size:40px;margin-bottom:12px;">📂</div>
            <div style="font-size:16px;font-weight:700;color:#0f1c2e;margin-bottom:6px;">Aucun dataset chargé</div>
            <div style="font-size:13px;">Importez un fichier CSV ou Excel pour commencer</div>
        </div>
        """, unsafe_allow_html=True)
        st.stop()

    df = st.session_state.df_raw.copy()
    section_title("🔍", "Vue d'ensemble du Dataset")

    num_df = df.select_dtypes(include=np.number)
    metric_row([
        {"label": "Lignes",          "value": f"{df.shape[0]:,}",                "color": ACCENT},
        {"label": "Colonnes",        "value": str(df.shape[1]),                   "color": ACCENT},
        {"label": "Cols numériques", "value": str(num_df.shape[1]),               "color": GREEN},
        {"label": "Catégorielles",   "value": str(df.shape[1]-num_df.shape[1]),   "color": PURPLE},
        {"label": "Val. manquantes", "value": str(int(df.isnull().sum().sum())),  "color": RED if df.isnull().sum().sum() > 0 else GREEN},
        {"label": "Doublons",        "value": str(df.duplicated().sum()),          "color": AMBER if df.duplicated().sum() > 0 else GREEN},
    ])

    col1, col2 = st.columns([3, 2])
    with col1:
        with st.expander("📋 Aperçu des données (10 premières lignes)", expanded=True):
            st.dataframe(df.head(10).reset_index(drop=True), use_container_width=True, height=280)
    with col2:
        with st.expander("📊 Types de colonnes", expanded=True):
            dtype_df = pd.DataFrame({
                "Colonne":  df.columns.tolist(),
                "Type":     df.dtypes.astype(str).tolist(),
                "Non-nuls": df.notna().sum().tolist(),
                "Uniques":  df.nunique().tolist(),
            })
            st.dataframe(dtype_df.reset_index(drop=True), use_container_width=True, height=280)

    section_title("📐", "Résumé en 5 chiffres & Statistiques descriptives")

    def compute_five_num(ndf: pd.DataFrame) -> pd.DataFrame:
        if ndf.empty or len(ndf.columns) == 0:
            return pd.DataFrame()
        rows = []
        for col in ndf.columns:
            s = ndf[col].dropna()
            if s.empty:
                continue
            mode_val = s.mode().iloc[0] if not s.mode().empty else np.nan
            rows.append({
                "Colonne":    col,
                "Min":        round(float(s.min()), 4),
                "Q1":         round(float(s.quantile(0.25)), 4),
                "Médiane":    round(float(s.median()), 4),
                "Q3":         round(float(s.quantile(0.75)), 4),
                "Max":        round(float(s.max()), 4),
                "Moyenne":    round(float(s.mean()), 4),
                "Écart-type": round(float(s.std()), 4),
                "Mode":       round(float(mode_val), 4) if not np.isnan(mode_val) else np.nan,
            })
        if not rows:
            return pd.DataFrame()
        return pd.DataFrame(rows).set_index("Colonne")

    if num_df.empty or len(num_df.columns) == 0:
        info_box("Aucune colonne numérique détectée dans le dataset.", "error")
    else:
        five_num = compute_five_num(num_df)
        if five_num.empty:
            info_box("Toutes les colonnes numériques sont vides (que des NaN).", "error")
        else:
            st.dataframe(five_num, use_container_width=True, height=300)

    section_title("🧹", "Nettoyage des données")
    col_a, col_b, col_c = st.columns([2, 2, 1])
    with col_a:
        fill_method = st.selectbox("Stratégie pour valeurs manquantes",
            ["Moyenne", "Médiane", "Mode", "Supprimer les lignes"], key="fill_method")
    with col_b:
        remove_dup = st.checkbox("Supprimer les doublons", value=True, key="rm_dup")
    with col_c:
        st.markdown("<br>", unsafe_allow_html=True)
        apply_clean = st.button("Appliquer", key="clean_btn", use_container_width=True)

    mv = df.isnull().sum(); mv = mv[mv > 0]
    if not mv.empty:
        with st.expander(f"⚠️ Valeurs manquantes dans {len(mv)} colonne(s)"):
            st.dataframe(mv.rename("Nb. manquants").to_frame(), use_container_width=True)

    if apply_clean:
        if remove_dup:
            before = len(df); df = df.drop_duplicates()
            info_box(f"{before - len(df)} ligne(s) dupliquée(s) supprimée(s)", "success")
        if fill_method == "Supprimer les lignes":
            df = df.dropna()
        else:
            num_c = df.select_dtypes(include=np.number).columns
            if fill_method == "Moyenne":
                df[num_c] = df[num_c].fillna(df[num_c].mean())
            elif fill_method == "Médiane":
                df[num_c] = df[num_c].fillna(df[num_c].median())
            elif fill_method == "Mode":
                df[num_c] = df[num_c].fillna(df[num_c].mode().iloc[0])
            for col in df.select_dtypes(include=["object", "category"]).columns:
                df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else "Inconnu")
        remaining_nan = int(df.isnull().sum().sum())
        st.session_state.df_clean = df.copy()
        color = "success" if remaining_nan == 0 else "warning"
        info_box(f"Nettoyage terminé — {df.shape[0]:,} lignes restantes · {remaining_nan} NaN restant(s)", color)

    if st.session_state.df_clean is not None:
        df = st.session_state.df_clean.copy()

    section_title("⚡", "Normalisation")
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    target_candidates = ['target','output','HeartDisease','heart_disease','condition','label','class','num','diagnosis']
    target_col = None
    for c in target_candidates:
        if c in df.columns: target_col = c; break
    if not target_col:
        for c in num_cols:
            if df[c].nunique() <= 5: target_col = c; break
    feature_cols = [c for c in num_cols if c != target_col]

    col_n1, col_n2 = st.columns([2, 3])
    with col_n1:
        scaler_choice = st.radio("Méthode de mise à l'échelle",
            ["Min-Max  [0, 1]", "Z-score  (μ=0, σ=1)"], key="scaler_choice")
    with col_n2:
        if target_col:
            info_box(f"Variable cible détectée : <code>{target_col}</code> — exclue de la normalisation", "info")
        if not feature_cols:
            info_box("Aucune feature numérique disponible.", "error")
        else:
            st.markdown(f'<div style="color:#5a6a82;font-size:12px;margin-top:8px;font-weight:500;">Features sélectionnées : {len(feature_cols)}</div>', unsafe_allow_html=True)
    if st.button("⚡ Normaliser les features", key="norm_btn"):
        if not feature_cols:
            info_box("Aucune feature numérique disponible pour la normalisation.", "error")
            st.stop()
        X = df[feature_cols].values.astype(float)
        nan_count = int(np.isnan(X).sum())
        if nan_count > 0:
            imputer = SimpleImputer(strategy="mean")
            X = imputer.fit_transform(X)
            info_box(f"<strong>{nan_count} valeur(s) manquante(s)</strong> imputées automatiquement.", "warning")
        scaler = MinMaxScaler() if "Min-Max" in scaler_choice else StandardScaler()
        X_scaled = scaler.fit_transform(X)
        st.session_state.X_scaled    = X_scaled
        st.session_state.X_features  = feature_cols[:X_scaled.shape[1]]
        st.session_state.scaler_name = scaler_choice
        st.session_state.y_target    = df[target_col].values if target_col else None
        info_box(f"Normalisation appliquée — {X_scaled.shape[0]:,} échantillons × {X_scaled.shape[1]} features", "success")
        with st.expander("Aperçu des données normalisées"):
            st.dataframe(pd.DataFrame(X_scaled, columns=feature_cols[:X_scaled.shape[1]]).head(8).round(4), use_container_width=True)

    section_title("📊", "Visualisation")
    vis_col1, vis_col2 = st.columns([2, 1])
    with vis_col1:
        vis_type = st.selectbox("Type de graphique",
            ["Boxplot", "Histogrammes", "Heatmap de corrélation", "Matrice de dispersion"], key="vis_type")
    with vis_col2:
        st.markdown("<br>", unsafe_allow_html=True)
        show_vis = st.button("Générer", key="vis_btn", use_container_width=True)

    if show_vis:
        fig_data  = df[feature_cols].select_dtypes(include=np.number)
        data_json = fig_data.to_json()

        @st.cache_data
        def make_boxplot(djson):
            d = pd.read_json(io.StringIO(djson))
            cols_bp = list(d.columns)
            ncols = min(4, len(cols_bp))
            nrows = (len(cols_bp) + ncols - 1) // ncols
            fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3.2, nrows * 3.6))
            axf = axes.flatten() if hasattr(axes, 'flatten') else [axes]
            for i, col in enumerate(cols_bp):
                ax = axf[i]
                series = d[col].dropna().values
                color = CLUSTER_PALETTE[i % len(CLUSTER_PALETTE)]
                bp = ax.boxplot(series, patch_artist=True, notch=False, widths=0.5,
                    medianprops=dict(color="#ffffff", linewidth=2.5),
                    whiskerprops=dict(color=TEXT2, linewidth=1.3),
                    capprops=dict(color=TEXT2, linewidth=1.3),
                    flierprops=dict(marker='o', markerfacecolor=color, markeredgecolor='none',
                                    markersize=3.5, alpha=0.5))
                bp['boxes'][0].set_facecolor(color + "55")
                bp['boxes'][0].set_edgecolor(color)
                bp['boxes'][0].set_linewidth(1.8)
                median_val = float(np.median(series))
                ax.text(1.32, median_val, f"{median_val:.2g}", va='center', ha='left', fontsize=7.5, color=ACCENT, fontweight='600')
                ax.set_title(col, fontsize=9, fontweight='700', color=TEXT, pad=6)
                ax.set_xticks([])
                ax.tick_params(axis='y', labelsize=7.5, colors=TEXT2)
                ax.grid(axis='y', alpha=0.35, linestyle='--')
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.spines['bottom'].set_visible(False)
                for sp in ['left']: ax.spines[sp].set_color(BORDER)
            for j in range(len(cols_bp), len(axf)):
                axf[j].set_visible(False)
            fig.suptitle("Boxplot — Distribution des features", fontsize=13, fontweight='700', color=TEXT, y=1.01)
            fig.tight_layout(h_pad=2.5, w_pad=2.0)
            return fig

        @st.cache_data
        def make_histograms(djson):
            d = pd.read_json(io.StringIO(djson)); cols_h = list(d.columns[:12])
            ncols = 4; nrows = (len(cols_h) + ncols - 1) // ncols
            fig, axes = plt.subplots(nrows, ncols, figsize=(14, nrows * 3))
            axf = axes.flatten() if hasattr(axes, 'flatten') else [axes]
            for i, col in enumerate(cols_h):
                axf[i].hist(d[col].dropna(), bins=20, color=CLUSTER_PALETTE[i % len(CLUSTER_PALETTE)], edgecolor='white', alpha=0.85, linewidth=0.8)
                axf[i].set_title(col, fontsize=9, fontweight='600')
            for j in range(len(cols_h), len(axf)): axf[j].set_visible(False)
            fig.suptitle("Distributions des features", fontsize=13, fontweight='700', y=1.01)
            fig.tight_layout(); return fig

        @st.cache_data
        def make_heatmap(djson):
            d = pd.read_json(io.StringIO(djson)); corr = d.corr()
            fig, ax = plt.subplots(figsize=(10, 8))
            mask = np.triu(np.ones_like(corr, dtype=bool))
            sns.heatmap(corr, mask=mask, annot=True, fmt=".2f",
                        cmap=sns.diverging_palette(220, 20, as_cmap=True), center=0,
                        ax=ax, square=True, linewidths=0.5, linecolor='white',
                        annot_kws={"size": 7}, cbar_kws={"shrink": 0.75})
            ax.set_title("Matrice de corrélation (triangle inférieur)", pad=14, fontweight='700')
            fig.tight_layout(); return fig

        @st.cache_data
        def make_scatter(djson):
            d = pd.read_json(io.StringIO(djson)); cols_sm = list(d.columns[:5]); n = len(cols_sm)
            fig, axes = plt.subplots(n, n, figsize=(10, 10))
            for i, ci in enumerate(cols_sm):
                for j, cj in enumerate(cols_sm):
                    ax = axes[i][j]
                    if i == j: ax.hist(d[ci].dropna(), bins=15, color=CLUSTER_PALETTE[i % len(CLUSTER_PALETTE)], alpha=0.75, edgecolor='white')
                    else: ax.scatter(d[cj], d[ci], s=5, alpha=0.4, color=ACCENT)
                    if i == n-1: ax.set_xlabel(cj, fontsize=7)
                    if j == 0: ax.set_ylabel(ci, fontsize=7)
                    ax.tick_params(labelsize=6)
            fig.suptitle("Matrice de dispersion (5 premières features)", fontsize=12, fontweight='700')
            fig.tight_layout(); return fig

        vis_map = {"Boxplot": make_boxplot, "Histogrammes": make_histograms,
                   "Heatmap de corrélation": make_heatmap, "Matrice de dispersion": make_scatter}
        if vis_type in ["Heatmap de corrélation", "Matrice de dispersion"]:
            with st.spinner("Génération en cours…"): fig = vis_map[vis_type](data_json)
        else: fig = vis_map[vis_type](data_json)
        st.pyplot(fig); plt.close(fig)


# ───────────────────────────────────────────────────────────────
#  VOLET 2 : CLUSTERING
# ───────────────────────────────────────────────────────────────
with tab2:
    if st.session_state.X_scaled is None:
        st.markdown("""
        <div style="text-align:center;padding:4rem 2rem;color:#5a6a82;background:#ffffff;
                    border:2px dashed #c7d2e0;border-radius:16px;margin-top:1rem;">
            <div style="font-size:40px;margin-bottom:12px;">⚠️</div>
            <div style="font-size:16px;font-weight:700;color:#0f1c2e;margin-bottom:6px;">Données non normalisées</div>
            <div style="font-size:13px;">Veuillez compléter l'étape de Prétraitement (onglet 2)</div>
        </div>
        """, unsafe_allow_html=True)
        st.stop()

    X_scaled = st.session_state.X_scaled
    section_title("🧠", "Algorithmes de Clustering", "5 méthodes disponibles")

    acols = st.columns(5)
    algo_info = [
        (ACCENT,  "⚙️", "K-MEANS",    "Partitionne en k clusters en minimisant l'inertie.", "Rapide · Scalable"),
        (PURPLE,  "💎", "K-MÉDOÏDES", "Utilise un point réel (médoïde). Plus robuste aux outliers.", "Robuste · Interprétable"),
        (GREEN,   "🌿", "AGNES",      "Hiérarchique ascendant. Fusionne les clusters les plus proches.", "Dendrogramme · Ward"),
        (AMBER,   "✂️", "DIANA",      "Hiérarchique divisif. Divise le cluster le plus hétérogène.", "Divisif · Complete"),
        (CYAN,    "🔵", "DBSCAN",     "Densité. Détecte les clusters et les points aberrants (−1).", "Bruit · Sans k"),
    ]
    for col, (color, icon, name, desc, tags) in zip(acols, algo_info):
        with col:
            st.markdown(f"""
            <div style="background:#ffffff;border:1px solid #dde3ed;border-radius:12px;padding:16px 14px;
                        border-top:3px solid {color};box-shadow:0 1px 6px rgba(15,28,46,.05);">
                <div style="font-size:11px;font-weight:700;color:{color};margin-bottom:6px;">{icon} {name}</div>
                <div style="font-size:11px;color:#5a6a82;line-height:1.55;">{desc}</div>
                <div style="margin-top:8px;font-size:10px;color:{GREEN};font-weight:600;">{tags}</div>
            </div>
            """, unsafe_allow_html=True)

    section_title("📈", "Méthode du Coude (Elbow)", "sélection du k optimal")
    e1, e2 = st.columns([3, 1])
    with e1: k_max = st.slider("k maximum à tester", 2, 15, 10, key="k_max")
    with e2:
        st.markdown("<br>", unsafe_allow_html=True)
        run_elbow = st.button("Calculer le coude", key="elbow_btn", use_container_width=True)

    if run_elbow:
        inertias = []
        for k_e in range(1, k_max + 1):
            km = KMeans(n_clusters=k_e, random_state=42, n_init=10); km.fit(X_scaled)
            inertias.append(km.inertia_)
        fig, ax = plt.subplots(figsize=(9, 3.5))
        x = list(range(1, k_max + 1))
        ax.plot(x, inertias, color=ACCENT, linewidth=2.5, zorder=3)
        ax.scatter(x, inertias, color=ACCENT, s=65, zorder=4, edgecolors='white', linewidths=1.5)
        ax.fill_between(x, inertias, alpha=0.08, color=ACCENT)
        ax.set_xlabel("Nombre de clusters k"); ax.set_ylabel("Inertie (WCSS)")
        ax.set_title("Courbe du Coude — Somme des carrés intra-cluster", fontweight='700')
        ax.set_xticks(x); ax.grid(axis='y', alpha=0.5); fig.tight_layout()
        st.pyplot(fig); plt.close(fig)
        info_box("Cherchez le <strong>point de coude</strong> où l'inertie cesse de diminuer fortement — c'est votre k optimal.", "info")

    section_title("⚙️", "Configuration des algorithmes")
    cfg1, cfg2 = st.columns([1, 3])
    with cfg1: k = st.number_input("Clusters k (sauf DBSCAN)", min_value=2, max_value=20, value=3, key="k_val")
    with cfg2:
        algos = st.multiselect("Algorithmes à exécuter",
            ["K-Means", "K-Medoids", "AGNES", "DIANA", "DBSCAN"],
            default=["K-Means", "K-Medoids", "AGNES", "DIANA", "DBSCAN"], key="algos_sel")

    col_agnes, col_diana, col_dbscan = st.columns(3)
    with col_agnes:
        st.markdown(f'<div style="background:#ffffff;border:1px solid #dde3ed;border-radius:10px;padding:16px;border-left:3px solid {GREEN};box-shadow:0 1px 4px rgba(15,28,46,.04);"><div style="font-size:12px;font-weight:700;color:{GREEN};margin-bottom:10px;">🌿 Paramètres AGNES</div>', unsafe_allow_html=True)
        agnes_linkage = st.selectbox("Critère de liaison", ["ward", "complete", "average", "single"], key="agnes_link")
        st.markdown(f'<div style="font-size:10px;color:#5a6a82;margin-top:6px;line-height:1.6;"><b style="color:#0f1c2e;">ward</b> : minimise l\'augmentation de variance<br><b style="color:#0f1c2e;">complete</b> : distance max entre clusters<br><b style="color:#0f1c2e;">average</b> : distances paires moyennes<br><b style="color:#0f1c2e;">single</b> : distance min (effet de chaîne)</div></div>', unsafe_allow_html=True)

    with col_diana:
        st.markdown(f'<div style="background:#ffffff;border:1px solid #dde3ed;border-radius:10px;padding:16px;border-left:3px solid {AMBER};box-shadow:0 1px 4px rgba(15,28,46,.04);"><div style="font-size:12px;font-weight:700;color:{AMBER};margin-bottom:10px;">✂️ Paramètres DIANA</div>', unsafe_allow_html=True)
        diana_linkage = st.selectbox("Liaison de division", ["complete", "average", "single"], key="diana_link")
        st.markdown('<div style="font-size:10px;color:#5a6a82;margin-top:6px;line-height:1.6;">DIANA est le pendant divisif d\'AGNES. Commence avec tous les points dans un seul cluster et divise récursivement le plus hétérogène.</div></div>', unsafe_allow_html=True)

    with col_dbscan:
        st.markdown(f'<div style="background:#ffffff;border:1px solid #dde3ed;border-radius:10px;padding:16px;border-left:3px solid {CYAN};box-shadow:0 1px 4px rgba(15,28,46,.04);"><div style="font-size:12px;font-weight:700;color:{CYAN};margin-bottom:10px;">🔵 Paramètres DBSCAN</div>', unsafe_allow_html=True)
        eps_val  = st.slider("ε (epsilon) — rayon de voisinage", 0.1, 10.0, 0.5, 0.05, key="eps")
        min_samp = st.slider("MinPts — min. points dans la boule ε", 2, 30, 5, key="minpts")
        st.markdown(f'<div style="font-size:10px;color:#5a6a82;margin-top:6px;line-height:1.6;"><b style="color:#0f1c2e;">Point core</b> : ≥ MinPts voisins dans ε<br><b style="color:#0f1c2e;">Point border</b> : dans ε d\'un point core<br><b style="color:#0f1c2e;">Bruit</b> : ni l\'un ni l\'autre → étiqueté −1</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    run_cl = st.button("🚀  Lancer tous les algorithmes sélectionnés", key="run_cl")

    if run_cl:
        if not algos:
            info_box("Sélectionnez au moins un algorithme.", "warning")
        else:
            labels_all = {}
            progress_bar = st.progress(0); status_text = st.empty()
            total_steps = len(algos); step = 0

            run_map = {
                "K-Means":    lambda: KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(X_scaled),
                "K-Medoids":  lambda: KMedoids(n_clusters=k, random_state=42).fit_predict(X_scaled),
                "AGNES":      lambda: AgglomerativeClustering(n_clusters=k, linkage=agnes_linkage).fit_predict(X_scaled),
                "DIANA":      lambda: AgglomerativeClustering(n_clusters=k, linkage=diana_linkage).fit_predict(X_scaled),
                "DBSCAN":     lambda: DBSCAN(eps=eps_val, min_samples=min_samp).fit_predict(X_scaled),
            }
            for algo in algos:
                status_text.markdown(f'<span style="color:{TEXT2};font-size:12px;font-weight:500;">⏳ {algo} en cours…</span>', unsafe_allow_html=True)
                labels_all[algo] = run_map[algo]()
                step += 1; progress_bar.progress(step / total_steps)

            status_text.empty(); progress_bar.empty()

            if "DBSCAN" in labels_all:
                nc = len(set(labels_all["DBSCAN"])) - (1 if -1 in labels_all["DBSCAN"] else 0)
                nn = list(labels_all["DBSCAN"]).count(-1)
                noise_pct = nn / len(labels_all["DBSCAN"]) * 100
                col_db1, col_db2, col_db3 = st.columns(3)
                col_db1.metric("Clusters DBSCAN trouvés", nc)
                col_db2.metric("Points de bruit", nn)
                col_db3.metric("% de bruit", f"{noise_pct:.1f}%")
                if nc == 0: info_box("DBSCAN a trouvé 0 clusters. Essayez d'<strong>augmenter ε</strong> ou de <strong>réduire MinPts</strong>.", "warning")
                elif noise_pct > 30: info_box(f"Ratio de bruit élevé ({noise_pct:.1f}%). Envisagez d'<strong>augmenter ε</strong>.", "warning")

            section_title("📊", "Métriques d'évaluation", "Silhouette")
            rows = []
            for name, lbl in labels_all.items():
                mask = lbl != -1; valid = len(set(lbl[mask])) >= 2
                rows.append({
                    "Algorithme":      name,
                    "Silhouette ↑":    f"{silhouette_score(X_scaled[mask], lbl[mask]):.4f}" if valid else "N/A",
                    "Clusters":        str(len(set(lbl[mask]))),
                    "Points de bruit": str(list(lbl).count(-1)) if -1 in lbl else "—",
                })
            st.dataframe(pd.DataFrame(rows).set_index("Algorithme"), use_container_width=True)

            section_title("📉", "Comparaison des inerties")

            def calc_inertia(X, labels):
                return sum(np.sum((X[labels==cid] - X[labels==cid].mean(axis=0))**2) for cid in set(labels) if cid != -1)

            inerties = {n: calc_inertia(X_scaled, l) for n, l in labels_all.items()}
            fig_in, ax_in = plt.subplots(figsize=(9, 3.8))
            bar_colors = [CLUSTER_PALETTE[i % len(CLUSTER_PALETTE)] for i in range(len(inerties))]
            bars = ax_in.bar(list(inerties.keys()), list(inerties.values()), color=bar_colors, edgecolor='white', linewidth=1.2, width=0.55)
            for bar in bars:
                ax_in.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3, f"{bar.get_height():.1f}", ha='center', va='bottom', fontsize=9, color=TEXT2, fontweight='600')
            ax_in.set_title("Comparaison des inerties — Tous les algorithmes", fontweight='700')
            ax_in.set_ylabel("Inertie (WCSS)"); ax_in.grid(axis='y', alpha=0.5); fig_in.tight_layout()
            st.pyplot(fig_in); plt.close(fig_in)

            section_title("🔭", "Projection PCA 2D")
            pca = PCA(n_components=2); X_2d = pca.fit_transform(X_scaled)
            var = pca.explained_variance_ratio_ * 100
            info_box(f"Variance expliquée — PC1 : <strong>{var[0]:.1f}%</strong>  ·  PC2 : <strong>{var[1]:.1f}%</strong>  ·  Total : <strong>{var[0]+var[1]:.1f}%</strong>", "info")

            n_alg = len(labels_all)
            algo_col_map = {"K-Means": ACCENT, "K-Medoids": PURPLE, "AGNES": GREEN, "DIANA": AMBER, "DBSCAN": CYAN}

            ncols_pca = min(2, n_alg)
            nrows_pca = (n_alg + ncols_pca - 1) // ncols_pca
            fig_pca, axes_pca = plt.subplots(nrows_pca, ncols_pca, figsize=(ncols_pca * 4.6, nrows_pca * 4.2), squeeze=False)
            axf_pca = [axes_pca[r][c] for r in range(nrows_pca) for c in range(ncols_pca)]

            for idx, (name, lbl) in enumerate(labels_all.items()):
                ax = axf_pca[idx]
                unique_ids = sorted(set(lbl))
                for cid in unique_ids:
                    mask = lbl == cid
                    is_noise = cid == -1
                    color = "#aab4c0" if is_noise else CLUSTER_PALETTE[cid % len(CLUSTER_PALETTE)]
                    label = "Bruit" if is_noise else f"C{cid} ({mask.sum()})"
                    ax.scatter(X_2d[mask, 0], X_2d[mask, 1], c=color, s=18, alpha=0.75, linewidths=0, label=label, zorder=2)
                for cid in unique_ids:
                    if cid == -1: continue
                    mask = lbl == cid
                    cx, cy = X_2d[mask, 0].mean(), X_2d[mask, 1].mean()
                    ax.scatter(cx, cy, marker='X', s=90, c=CLUSTER_PALETTE[cid % len(CLUSTER_PALETTE)], edgecolors='white', linewidths=1.2, zorder=4)
                accent_c = algo_col_map.get(name, ACCENT)
                ax.set_title(name, fontsize=10, fontweight='700', pad=8, color=accent_c)
                ax.set_xlabel(f"PC1  ({var[0]:.1f}% var.)", fontsize=7.5, color=TEXT2)
                ax.set_ylabel(f"PC2  ({var[1]:.1f}% var.)", fontsize=7.5, color=TEXT2)
                ax.tick_params(labelsize=7, colors=TEXT2)
                ax.grid(alpha=0.25, linestyle='--', zorder=0)
                for spine in ax.spines.values():
                    spine.set_color(BORDER); spine.set_linewidth(0.8)
                leg = ax.legend(fontsize=7, markerscale=1.4, framealpha=0.9, edgecolor=BORDER, loc='best')
                leg.get_frame().set_linewidth(0.6)

            for j in range(n_alg, len(axf_pca)):
                axf_pca[j].set_visible(False)

            fig_pca.suptitle("Projections des clusters — PCA 2D", fontsize=12, fontweight='700', color=TEXT, y=1.01)
            fig_pca.tight_layout(h_pad=2.8, w_pad=2.5)
            st.pyplot(fig_pca); plt.close(fig_pca)

            if "AGNES" in algos:
                section_title("🌿", "AGNES — Dendrogramme", f"Liaison : {agnes_linkage}")
                n_samp = min(250, X_scaled.shape[0])
                link_mat = linkage(X_scaled[:n_samp], method=agnes_linkage)
                threshold = 0.7 * max(link_mat[:, 2])
                fig_d, ax_d = plt.subplots(figsize=(13, 5))
                dendrogram(link_mat, truncate_mode='lastp', p=30, leaf_rotation=40, leaf_font_size=8,
                           color_threshold=threshold, above_threshold_color=TEXT2, ax=ax_d)
                ax_d.axhline(y=threshold, color=AMBER, linewidth=1.5, linestyle='--', label=f"Seuil ({threshold:.2f})")
                ax_d.set_title(f"Dendrogramme AGNES (liaison {agnes_linkage}) — {n_samp} premiers échantillons", fontweight='700')
                ax_d.set_xlabel("Indice / taille du cluster"); ax_d.set_ylabel("Distance de fusion")
                ax_d.legend(fontsize=9); ax_d.grid(axis='y', alpha=0.4); fig_d.tight_layout()
                st.pyplot(fig_d); plt.close(fig_d)

            if "DBSCAN" in algos:
                section_title("🔵", "DBSCAN — Analyse par densité", f"ε={eps_val} · MinPts={min_samp}")
                lbl_db = labels_all["DBSCAN"]
                nc_db = len(set(lbl_db)) - (1 if -1 in lbl_db else 0)
                nn_db = list(lbl_db).count(-1)

                if nc_db > 0:
                    cluster_sizes = [int(np.sum(lbl_db==c)) for c in range(nc_db)]
                    labels_pie = [f"C{c} ({cluster_sizes[c]})" for c in range(nc_db)]
                    if nn_db > 0: cluster_sizes.append(nn_db); labels_pie.append(f"Bruit ({nn_db})")
                    fig_pie, ax_pie = plt.subplots(figsize=(6, 4))
                    colors_pie = [CLUSTER_PALETTE[i%len(CLUSTER_PALETTE)] for i in range(nc_db)] + (["#aab4c0"] if nn_db>0 else [])
                    wedges, texts, autotexts = ax_pie.pie(cluster_sizes, labels=labels_pie, colors=colors_pie,
                        autopct='%1.1f%%', pctdistance=0.82, wedgeprops=dict(edgecolor='white', linewidth=2), startangle=90)
                    for t in texts: t.set_color(TEXT2); t.set_fontsize(9)
                    for at in autotexts: at.set_color(TEXT); at.set_fontsize(8); at.set_fontweight('600')
                    ax_pie.set_title(f"Distribution DBSCAN", fontweight='700'); fig_pie.tight_layout()
                    st.pyplot(fig_pie); plt.close(fig_pie)

                from sklearn.neighbors import NearestNeighbors
                nbrs = NearestNeighbors(n_neighbors=min_samp).fit(X_scaled)
                distances, _ = nbrs.kneighbors(X_scaled)
                k_dists = np.sort(distances[:, -1])[::-1]
                fig_kd, ax_kd = plt.subplots(figsize=(9, 3.2))
                ax_kd.plot(k_dists, color=CYAN, linewidth=2)
                ax_kd.axhline(y=eps_val, color=AMBER, linewidth=1.5, linestyle='--', label=f"ε actuel = {eps_val}")
                ax_kd.set_title(f"Graphe {min_samp}-NN distance (trié décroissant)", fontweight='700')
                ax_kd.set_xlabel("Points (triés par distance)"); ax_kd.set_ylabel(f"Distance {min_samp}-NN")
                ax_kd.legend(fontsize=9); ax_kd.grid(alpha=0.4); fig_kd.tight_layout()
                st.pyplot(fig_kd); plt.close(fig_kd)
                info_box("Le <strong>coude</strong> de la courbe k-distance est une bonne estimation pour ε.", "info")


# ── Footer ──────────────────────────────────────────────────────
st.markdown("""
<div style="margin-top:3rem;padding-top:1.2rem;border-top:1px solid #dde3ed;
            display:flex;justify-content:space-between;align-items:center;">
    <span style="font-size:11px;color:#5a6a82;font-weight:500;">🔬 FD1 · Interface Fouille de Données · M1 Bioinformatique</span>
    <span style="font-size:11px;color:#5a6a82;">Faculté d'Informatique · Année 2025–2026</span>
</div>
""", unsafe_allow_html=True)
