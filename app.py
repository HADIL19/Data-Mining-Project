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
    silhouette_score, davies_bouldin_score, calinski_harabasz_score,
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

# ── Modern SaaS Color Palette — Light Dashboard ──────────────────────
BG       = "#f8fafc"
SURFACE  = "rgba(15, 23, 42, 0.04)"
SURFACE2 = "rgba(15, 23, 42, 0.06)"
BORDER   = "rgba(15, 23, 42, 0.12)"
BORDER_MATPLOTLIB = (0.08, 0.12, 0.18, 0.95)  # For matplotlib compatibility
TEXT     = "#0f172a"
TEXT2    = "#475569"
TEXT2_MATPLOTLIB = (0.28, 0.34, 0.44, 0.95)  # For matplotlib compatibility
ACCENT   = "#2563eb"
GRADIENT = "linear-gradient(135deg, #2563eb 0%, #6366f1 100%)"
GREEN    = "#16a34a"
AMBER    = "#f59e0b"
RED      = "#dc2626"
PURPLE   = "#8b5cf6"
CYAN     = "#06b6d4"

CLUSTER_PALETTE = ["#667eea", "#8b5cf6", "#ec4899", "#f59e0b", "#10b981", "#06b6d4",
                   "#06b6d4", "#f43f5e", "#0ea5e9", "#84cc16"]

# ── Matplotlib Light Theme ───────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor":  "#ffffff",
    "axes.facecolor":    "#ffffff",
    "axes.edgecolor":    "#cbd5e1",
    "axes.labelcolor":   "#475569",
    "axes.titlecolor":   "#0f172a",
    "axes.titlesize":    13,
    "axes.labelsize":    11,
    "xtick.color":       "#475569",
    "ytick.color":       "#475569",
    "xtick.labelsize":   10,
    "ytick.labelsize":   10,
    "text.color":        "#0f172a",
    "grid.color":        "#e2e8f0",
    "grid.alpha":        0.9,
    "legend.facecolor":  "#f8fafc",
    "legend.edgecolor":  "#cbd5e1",
    "legend.fontsize":   10,
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
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

* { transition: all 0.22s ease !important; }

html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    background: #f8fafc !important;
    color: #0f172a !important;
}

#MainMenu, footer, header { visibility: hidden; }
.block-container { 
    padding: 2rem 2.5rem 3rem !important; 
    max-width: 1300px !important;
    background: #f8fafc !important;
}

[data-testid="stSidebar"] {
    background: #ffffff !important;
    border-right: 1px solid rgba(15, 23, 42, 0.08) !important;
    box-shadow: 0 24px 80px rgba(15, 23, 42, 0.05) !important;
}
[data-testid="stSidebar"] > div:first-child { padding-top: 1.5rem; }

[data-baseweb="tab-list"] {
    background: #f1f5f9 !important;
    border-radius: 16px !important;
    padding: 6px !important;
    gap: 4px !important;
    border: 1px solid rgba(15, 23, 42, 0.08) !important;
}
[data-baseweb="tab"] {
    background: transparent !important;
    border-radius: 12px !important;
    color: #475569 !important;
    font-weight: 600 !important;
    font-size: 13px !important;
    padding: 8px 24px !important;
}
[aria-selected="true"][data-baseweb="tab"] {
    background: #2563eb !important;
    color: #ffffff !important;
    box-shadow: 0 12px 30px rgba(37, 99, 235, 0.18) !important;
}
[data-baseweb="tab-highlight"] { display: none !important; }
[data-baseweb="tab-border"]    { display: none !important; }

.stButton > button {
    background: #2563eb !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 14px !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 13px !important;
    font-weight: 700 !important;
    padding: 10px 22px !important;
    cursor: pointer !important;
    box-shadow: 0 12px 24px rgba(37, 99, 235, 0.18) !important;
}
.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 16px 28px rgba(37, 99, 235, 0.2) !important;
}
.stButton > button:active { transform: translateY(0) !important; }

[data-testid="stFileUploader"] {
    background: #ffffff !important;
    border: 2px dashed rgba(15, 23, 42, 0.12) !important;
    border-radius: 16px !important;
    padding: 1.6rem !important;
}
[data-baseweb="select"] > div,
[data-baseweb="input"] > div,
.stTextInput > div > div,
.stSelectbox > div > div,
.stMultiSelect > div > div {
    background: #ffffff !important;
    border-color: rgba(15, 23, 42, 0.12) !important;
    border-radius: 12px !important;
    color: #0f172a !important;
    font-size: 13px !important;
}
[data-testid="stMetric"] {
    background: #ffffff !important;
    border: 1px solid rgba(15, 23, 42, 0.08) !important;
    border-radius: 18px !important;
    padding: 18px 20px !important;
    box-shadow: 0 16px 50px rgba(15, 23, 42, 0.05) !important;
}
[data-testid="stMetricLabel"] { 
    color: #475569 !important; 
    font-size: 11px !important; 
    text-transform: uppercase; 
    letter-spacing: .08em;
    font-weight: 700;
}
[data-testid="stMetricValue"] { 
    color: #0f172a !important; 
    font-size: 28px !important; 
    font-weight: 800 !important;
    background: none !important;
    -webkit-text-fill-color: initial !important;
}

[data-testid="stDataFrame"] {
    border: 1px solid rgba(15, 23, 42, 0.08) !important;
    border-radius: 16px !important;
    overflow: visible !important;
    box-shadow: 0 16px 40px rgba(15, 23, 42, 0.05) !important;
}
[data-testid="stExpander"] {
    background: #ffffff !important;
    border: 1px solid rgba(15, 23, 42, 0.08) !important;
    border-radius: 18px !important;
    box-shadow: 0 12px 32px rgba(15, 23, 42, 0.05) !important;
}
[data-testid="stExpander"] summary {
    font-size: 13px !important;
    font-weight: 700 !important;
    color: #0f172a !important;
    padding: 14px 18px !important;
}
[data-testid="stExpander"] summary:hover { background: rgba(37, 99, 235, 0.05) !important; }

.stAlert { 
    border-radius: 16px !important; 
    font-size: 13px !important; 
    border-left-width: 3px !important;
    background: #ffffff !important;
    border: 1px solid rgba(15, 23, 42, 0.08) !important;
}
[data-baseweb="radio"] { gap: 8px !important; }
[data-testid="stRadio"] label,
[data-testid="stRadio"] label span,
[data-testid="stRadio"] label div,
[data-testid="stRadio"] label * {
    font-size: 13px !important;
    color: #0f172a !important;
    background: transparent !important;
    text-shadow: none !important;
    opacity: 1 !important;
}
[data-testid="stRadio"] label:hover,
[data-testid="stRadio"] label:hover * {
    color: #0f172a !important;
}
[data-testid="stRadio"] label:has(input:checked),
[data-testid="stRadio"] label:has(input:checked) * {
    color: #0f172a !important;
    font-weight: 700 !important;
}
[data-testid="stRadio"] input {
    accent-color: #2563eb !important;
}
[data-testid="stRadio"] label > div { display: inline-flex !important; align-items: center !important; }
[data-testid="stNumberInput"] label,
[data-testid="stSelectbox"] label,
[data-testid="stMultiSelect"] label,
[data-testid="stSlider"] label,
[data-testid="stTextInput"] label,
[data-testid="stCheckbox"] label,
[data-testid="stButton"] button {
    color: #0f172a !important;
}
[data-testid="stNumberInput"] div[data-testid="stMarkdownContainer"] span,
[data-testid="stSelectbox"] div[data-testid="stMarkdownContainer"] span,
[data-testid="stMultiSelect"] div[data-testid="stMarkdownContainer"] span,
[data-testid="stSlider"] div[data-testid="stMarkdownContainer"] span,
[data-testid="stTextInput"] div[data-testid="stMarkdownContainer"] span {
    color: #0f172a !important;
}
[data-testid="stNumberInput"] input,
[data-testid="stNumberInput"] input[type="number"],
[data-testid="stNumberInput"] [role="spinbutton"] input,
[data-testid="stNumberInput"] [role="spinbutton"] {
    color: #0f172a !important;
    background: #ffffff !important;
}
[data-testid="stCheckbox"] label { 
    font-size: 13px !important; 
    color: #475569 !important;
}
[data-baseweb="tag"] {
    background: rgba(37, 99, 235, 0.08) !important;
    border-color: rgba(37, 99, 235, 0.16) !important;
    color: #2563eb !important;
    border-radius: 999px !important; 
    font-size: 12px !important;
}

hr { 
    border-color: rgba(15, 23, 42, 0.08) !important; 
    margin: 1.5rem 0 !important; 
}
code {
    background: rgba(15, 23, 42, 0.06) !important; 
    border: 1px solid rgba(15, 23, 42, 0.08) !important;
    border-radius: 8px !important; 
    padding: 2px 6px !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 12px !important; 
    color: #0f172a !important;
}
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
#  HELPER COMPONENTS
# ═══════════════════════════════════════════════════════════════

def section_title(icon: str, title: str, subtitle: str = ""):
    sub_html = f'<span style="color:{TEXT2};font-size:12px;font-weight:500;margin-left:10px;">{subtitle}</span>' if subtitle else ""
    st.markdown(f"""
    <div style="display:flex;align-items:center;gap:14px;margin:2rem 0 1rem;padding-bottom:0.85rem;
                border-bottom:1px solid rgba(15, 23, 42, 0.08);">
        <div style="width:46px;height:46px;background:#ffffff;border:1px solid rgba(15, 23, 42, 0.08);
                    border-radius:14px;display:flex;align-items:center;justify-content:center;font-size:18px;
                    box-shadow:0 10px 28px rgba(15, 23, 42, 0.05);">{icon}</div>
        <div style="flex:1;">
            <span style="font-size:18px;font-weight:800;color:{TEXT};">{title}</span>{sub_html}
        </div>
    </div>
    """, unsafe_allow_html=True)


def metric_row(metrics: list):
    cols = st.columns(len(metrics))
    for col, m in zip(cols, metrics):
        delta_html = f'<div style="font-size:11px;color:{GREEN};margin-top:4px;font-weight:700;">{m["delta"]}</div>' if "delta" in m else ""
        with col:
            st.markdown(f"""
            <div style="background:#ffffff;border:1px solid rgba(15, 23, 42, 0.08);
                        border-radius:18px;padding:18px 16px;text-align:center;
                        box-shadow:0 16px 40px rgba(15, 23, 42, 0.05);">
                <div style="font-size:11px;color:{TEXT2};text-transform:uppercase;
                            letter-spacing:.1em;margin-bottom:10px;font-weight:700;">{m['label']}</div>
                <div style="font-size:28px;font-weight:800;color:{m.get('color', ACCENT)};line-height:1.1;">{m['value']}</div>
                {delta_html}
            </div>
            """, unsafe_allow_html=True)


def info_box(text: str, kind: str = "info"):
    styles = {
        "info":    ("#2563eb", "#eff6ff", "#bfdbfe", "ℹ️"),
        "success": ("#16a34a", "#ecfdf5", "#bbf7d0", "✅"),
        "warning": ("#f59e0b", "#fffbeb", "#fde68a", "⚠️"),
        "error":   ("#dc2626", "#fef2f2", "#fecaca", "❌"),
    }
    c, bg, border, icon = styles.get(kind, styles["info"])
    st.markdown(f"""
    <div style="background:{bg};border:1px solid {border};border-radius:16px;
                padding:14px 16px;margin:10px 0;font-size:13px;
                display:flex;gap:12px;align-items:flex-start;box-shadow:0 14px 32px rgba(15, 23, 42, 0.06);">
        <span style="margin-top:2px;flex-shrink:0;font-size:18px;">{icon}</span>
        <span style="color:{TEXT};line-height:1.7;">{text}</span>
    </div>
    """, unsafe_allow_html=True)


def page_header():
    st.markdown(f"""
    <div style="background:#ffffff;border:1px solid rgba(15, 23, 42, 0.08);border-radius:24px;padding:2rem;
                box-shadow:0 24px 70px rgba(15, 23, 42, 0.06);margin-bottom:2rem;">
        <div style="display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:24px;">
            <div style="display:flex;align-items:center;gap:18px;min-width:0;">
                <div style="width:62px;height:62px;background:linear-gradient(135deg, #2563eb 0%, #6366f1 100%);
                            border-radius:18px;display:flex;align-items:center;justify-content:center;font-size:30px;color:#ffffff;">
                    🔬
                </div>
                <div style="min-width:0;">
                    <div style="font-size:30px;font-weight:900;color:{TEXT};line-height:1.05;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">
                        Fouille de Données
                    </div>
                    <div style="font-size:14px;color:{TEXT2};margin-top:6px;">
                        FD1 Pipeline — M1 Bioinformatique · 2025–2026
                    </div>
                </div>
            </div>
            <div style="display:flex;gap:10px;align-items:center;flex-wrap:wrap;">
                <div style="display:flex;align-items:center;gap:8px;background:#eff6ff;border:1px solid rgba(37, 99, 235, 0.16);
                            color:#2563eb;padding:10px 16px;border-radius:14px;font-size:13px;font-weight:700;">
                    <span style="width:10px;height:10px;background:#2563eb;border-radius:50%;"></span>
                    En ligne
                </div>
                <div style="color:#475569;font-size:13px;">Interface claire · Données prêtes</div>
            </div>
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
    st.markdown(f"""
    <div style="padding:0.5rem 0 1.8rem;">
        <div style="display:flex;align-items:center;gap:10px;margin-bottom:6px;">
            <div style="width:40px;height:40px;background:{GRADIENT};
                        border-radius:12px;display:flex;align-items:center;justify-content:center;
                        font-size:18px;box-shadow:0 8px 24px rgba(102,126,234,0.3);">🔬</div>
            <div>
                <div style="font-size:14px;font-weight:800;color:{TEXT};">FD1</div>
                <div style="font-size:10px;color:{TEXT2};font-weight:500;">Data Mining Pipeline</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f'<div style="font-size:9px;text-transform:uppercase;letter-spacing:.15em;color:{TEXT2};margin:1.2rem 0 0.8rem;font-weight:800;">Navigation</div>', unsafe_allow_html=True)
    for icon, label in [("🏠", "Accueil"), ("📂", "Prétraitement"), ("🧩", "Clustering"), ("🤖", "Classification")]:
        st.markdown(f'<div style="display:flex;align-items:center;gap:10px;padding:10px 12px;border-radius:10px;margin-bottom:6px;color:{TEXT2};font-size:13px;font-weight:600;cursor:pointer;transition:all 0.3s;hover:background:rgba(102,126,234,0.2);">{icon} {label}</div>', unsafe_allow_html=True)

    st.markdown(f'<hr style="margin:1.4rem 0;border-color:rgba(15,23,42,0.08);">', unsafe_allow_html=True)
    st.markdown(f'<div style="font-size:9px;text-transform:uppercase;letter-spacing:.15em;color:{TEXT2};margin-bottom:0.8rem;font-weight:800;">Jeu de données</div>', unsafe_allow_html=True)

    if st.session_state.df_raw is not None:
        df_info = st.session_state.df_raw
        missing = int(df_info.isnull().sum().sum())
        scaled_ok = st.session_state.X_scaled is not None
        st.markdown(f"""
        <div style="background:#ffffff;border:1px solid rgba(15,23,42,0.08);border-radius:12px;
                    padding:14px 16px;backdrop-filter:blur(10px);">
            <div style="font-size:12px;color:{TEXT};font-weight:800;margin-bottom:10px;">
                📄 {st.session_state.get("fname","dataset.csv")}
            </div>
            <div style="display:grid;grid-template-columns:1fr 1fr;gap:8px;">
                <div style="background:#f8fafc;border:1px solid rgba(15,23,42,0.08);border-radius:8px;padding:10px;text-align:center;">
                    <div style="font-size:10px;color:{TEXT2};font-weight:700;text-transform:uppercase;">Lignes</div>
                    <div style="font-size:16px;font-weight:800;background:{GRADIENT};-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;">{df_info.shape[0]:,}</div>
                </div>
                <div style="background:#f8fafc;border:1px solid rgba(15,23,42,0.08);border-radius:8px;padding:10px;text-align:center;">
                    <div style="font-size:10px;color:{TEXT2};font-weight:700;text-transform:uppercase;">Colonnes</div>
                    <div style="font-size:16px;font-weight:800;background:{GRADIENT};-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;">{df_info.shape[1]}</div>
                </div>
                <div style="background:#f8fafc;border:1px solid rgba(15,23,42,0.08);border-radius:8px;padding:10px;text-align:center;">
                    <div style="font-size:10px;color:{TEXT2};font-weight:700;text-transform:uppercase;">Manquants</div>
                    <div style="font-size:16px;font-weight:800;color:{'#ef4444' if missing>0 else '#10b981'};">{missing}</div>
                </div>
                <div style="background:#f8fafc;border:1px solid rgba(15,23,42,0.08);border-radius:8px;padding:10px;text-align:center;">
                    <div style="font-size:10px;color:{TEXT2};font-weight:700;text-transform:uppercase;">Normalisé</div>
                    <div style="font-size:16px;font-weight:800;color:{'#10b981' if scaled_ok else '#f59e0b'};">{'✓' if scaled_ok else '–'}</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div style="background:#ffffff;border:1px solid rgba(15,23,42,0.08);border-radius:12px;
                    padding:20px;text-align:center;color:{TEXT2};font-size:12px;backdrop-filter:blur(10px);">
            <div style="font-size:28px;margin-bottom:8px;">📂</div>
            <span style="color:{TEXT};font-weight:700;display:block;margin-bottom:2px;">Aucun dataset</span>
            Chargez vos données pour commencer
        </div>
        """, unsafe_allow_html=True)

    st.markdown(f'<hr style="margin:1.4rem 0;border-color:rgba(15,23,42,0.08);">', unsafe_allow_html=True)
    st.markdown(f"""
    <div style="font-size:11px;color:{TEXT2};text-align:center;line-height:1.8;font-weight:500;">
        M1 BioInformatique<br>FD1 · 2025–2026
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════
page_header()

tab0, tab1, tab2, tab3 = st.tabs([
    "  🏠  Accueil  ",
    "  📂  Prétraitement  ",
    "  🧩  Clustering  ",
    "  🤖  Classification  ",
])


# ───────────────────────────────────────────────────────────────
#  VOLET 0 : PAGE D'ACCUEIL
# ───────────────────────────────────────────────────────────────
with tab0:
    st.markdown(f"""
    <div style="background:#ffffff;border:1px solid rgba(15,23,42,0.08);border-radius:24px;padding:3rem;margin-bottom:2.5rem;
                position:relative;overflow:hidden;box-shadow:0 24px 80px rgba(15,23,42,0.08);">
        <div style="position:absolute;top:-100px;right:-80px;width:300px;height:300px;
                    background:radial-gradient(circle,rgba(37,99,235,0.16),transparent 70%);border-radius:50%;"></div>
        <div style="position:absolute;bottom:-80px;left:20%;width:260px;height:260px;
                    background:radial-gradient(circle,rgba(99,102,241,0.14),transparent 70%);border-radius:50%;"></div>
        <div style="position:relative;z-index:1;">
            <div style="display:flex;align-items:flex-start;gap:16px;margin-bottom:1.5rem;flex-wrap:wrap;">
                <div style="width:64px;height:64px;background:{GRADIENT};
                            border-radius:18px;display:flex;align-items:center;justify-content:center;
                            font-size:32px;color:#ffffff;flex-shrink:0;">🔬</div>
                <div>
                    <div style="font-size:32px;font-weight:900;color:{TEXT};letter-spacing:-.6px;line-height:1.1;">Fouille de Données</div>
                    <div style="font-size:14px;color:{TEXT2};margin-top:6px;font-weight:600;">M1 Bioinformatique · 2025–2026</div>
                </div>
            </div>
            <p style="font-size:15px;color:{TEXT2};max-width:760px;line-height:1.8;margin:0 0 1.8rem;">
                Suite complète d'<strong style="color:#2563eb;font-weight:800;">analyse de données</strong>
                sans code — du nettoyage à la classification, tout-en-un.
                Prétraitement · Clustering · Classification supervisée.
            </p>
            <div style="display:flex;gap:10px;flex-wrap:wrap;">
                <span style="background:#eff6ff;color:#2563eb;border:1px solid rgba(37,99,235,0.18);padding:10px 18px;border-radius:12px;font-size:13px;font-weight:700;">📂 Prétraitement</span>
                <span style="background:#f4efff;color:#7c3aed;border:1px solid rgba(124,58,237,0.18);padding:10px 18px;border-radius:12px;font-size:13px;font-weight:700;">🧩 Clustering</span>
                <span style="background:#fef3c7;color:#b45309;border:1px solid rgba(245,158,11,0.24);padding:10px 18px;border-radius:12px;font-size:13px;font-weight:700;">🤖 Classification</span>
                <span style="background:#ecfdf5;color:#15803d;border:1px solid rgba(16,185,129,0.24);padding:10px 18px;border-radius:12px;font-size:13px;font-weight:700;">✅ CSV / Excel</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Volets cards
    st.markdown(f'<div style="font-size:18px;font-weight:900;color:{TEXT};margin:2rem 0 1.2rem;">Fonctionnalités principales</div>', unsafe_allow_html=True)
    cv1, cv2, cv3 = st.columns(3)

    for col, border_color, icon, vtitle, vdesc, items in [
        (cv1, "#667eea", "📂", "Volet 1 — Prétraitement",
         "Préparez vos données étape par étape avant toute analyse.",
         [("→","#667eea","Importation CSV / Excel"),("→","#667eea","Statistiques descriptives"),
          ("→","#667eea","Gestion des valeurs manquantes"),("→","#667eea","Normalisation Min-Max / Z-score"),
          ("→","#667eea","Boxplot · Histogramme · Heatmap")]),
        (cv2, "#8b5cf6", "🧩", "Volet 2 — Clustering",
         "Explorez les structures cachées dans vos données.",
         [("→","#8b5cf6","K-Means & K-Médoïdes"),("→","#8b5cf6","AGNES (ascendant) & DIANA (divisif)"),
          ("→","#8b5cf6","DBSCAN (densité)"),("→","#8b5cf6","Courbe d'Elbow + Score Silhouette"),
          ("→","#8b5cf6","Projection PCA 2D + Dendrogramme")]),
        (cv3, "#f59e0b", "🤖", "Volet 3 — Classification",
         "Construisez et évaluez des modèles supervisés.",
         [("→","#f59e0b","K-NN · Arbre de décision"),("→","#f59e0b","Naïve Bayes · SVM · Régression logistique"),
          ("→","#f59e0b","Random Forest"),("→","#f59e0b","Matrice de confusion"),
          ("→","#f59e0b","Accuracy · Précision · Rappel · F1")]),
    ]:
        with col:
            items_html = "".join([
                f'<div style="display:flex;align-items:center;gap:8px;font-size:13px;color:{TEXT2};">'
                f'<span style="color:{ic};font-weight:800;">{ar}</span> <span style="color:{TEXT};">{txt}</span></div>'
                for ar, ic, txt in items
            ])
            st.markdown(f"""
            <div style="background:#ffffff;border:1px solid rgba(15,23,42,0.08);
                        border-radius:16px;padding:24px;box-shadow:0 16px 40px rgba(15,23,42,0.06);
                        border-top:3px solid {border_color};height:100%;backdrop-filter:blur(10px);
                        transition:all 0.3s;">
                <div style="font-size:32px;margin-bottom:12px;">{icon}</div>
                <div style="font-size:15px;font-weight:800;color:{TEXT};margin-bottom:6px;">{vtitle}</div>
                <div style="font-size:13px;color:{TEXT2};line-height:1.7;margin-bottom:14px;">{vdesc}</div>
                <div style="display:flex;flex-direction:column;gap:7px;">{items_html}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Steps
    st.markdown(f'<div style="font-size:18px;font-weight:900;color:{TEXT};margin:2rem 0 1.2rem;">Comment utiliser l\'interface ?</div>', unsafe_allow_html=True)
    steps = [
        ("1","#667eea","Importer un dataset","Allez dans <strong style=\"color:#667eea;\">Prétraitement</strong> et chargez votre fichier CSV ou Excel."),
        ("2","#10b981","Préparer les données","Nettoyez (valeurs manquantes, doublons) puis normalisez avec Min-Max ou Z-score."),
        ("3","#8b5cf6","Explorer les clusters","Dans <strong style=\"color:#8b5cf6;\">Clustering</strong>, lancez les algorithmes et analysez les scores et projections PCA."),
        ("4","#f59e0b","Classer et évaluer","Dans <strong style=\"color:#f59e0b;\">Classification</strong>, entraînez un modèle et analysez la matrice de confusion."),
    ]
    sc = st.columns(4)
    for col, (num, color, title, desc) in zip(sc, steps):
        with col:
            st.markdown(f"""
            <div style="background:#ffffff;border:1px solid rgba(15,23,42,0.08);
                        border-radius:14px;padding:22px;box-shadow:0 16px 40px rgba(15,23,42,0.06);
                        text-align:center;backdrop-filter:blur(10px);transition:all 0.3s;">
                <div style="width:48px;height:48px;background:{color};color:#fff;border-radius:50%;
                            display:flex;align-items:center;justify-content:center;font-size:20px;font-weight:900;
                            margin:0 auto 14px;box-shadow:0 8px 24px {color}55;">{num}</div>
                <div style="font-size:13px;font-weight:800;color:{TEXT};margin-bottom:6px;">{title}</div>
                <div style="font-size:12px;color:{TEXT2};line-height:1.6;">{desc}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    col_ds, col_tech = st.columns(2)

    with col_ds:
        st.markdown(f"""
        <div style="background:#ffffff;border:1px solid rgba(15,23,42,0.08);
                    border-radius:14px;padding:22px;box-shadow:0 16px 40px rgba(15,23,42,0.06);backdrop-filter:blur(10px);">
            <div style="font-size:14px;font-weight:800;color:{TEXT};margin-bottom:14px;">📋 Datasets recommandés</div>
            <div style="display:flex;flex-direction:column;gap:8px;">
                <div style="background:rgba(102,126,234,0.15);border:1px solid rgba(102,126,234,0.3);border-radius:10px;padding:12px 14px;">
                    <div style="font-size:12px;font-weight:800;color:#667eea;">Pima Indians Diabetes</div>
                    <div style="font-size:11px;color:{TEXT2};margin-top:2px;">768 lignes · 9 colonnes · Classification binaire</div>
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
            st.dataframe(df.head(10).reset_index(drop=True), width='stretch', height=280)
    with col2:
        with st.expander("📊 Types de colonnes", expanded=True):
            dtype_df = pd.DataFrame({
                "Colonne":  df.columns.tolist(),
                "Type":     df.dtypes.astype(str).tolist(),
                "Non-nuls": df.notna().sum().tolist(),
                "Uniques":  df.nunique().tolist(),
            })
            st.dataframe(dtype_df.reset_index(drop=True), width='stretch', height=280)

    section_title("📐", "Résumé en 5 chiffres & Statistiques descriptives")

    @st.cache_data
    def compute_five_num(data_json: str) -> pd.DataFrame:
        ndf = pd.read_json(io.StringIO(data_json))
        return pd.DataFrame({
            "Min":        ndf.min(),
            "Q1":         ndf.quantile(0.25),
            "Médiane":    ndf.median(),
            "Q3":         ndf.quantile(0.75),
            "Max":        ndf.max(),
            "Moyenne":    ndf.mean().round(3),
            "Écart-type": ndf.std().round(3),
            "Mode":       ndf.mode().iloc[0],
        })

    five_num = compute_five_num(num_df.to_json())
    st.dataframe(five_num.T.style.format("{:.3f}"), width='stretch', height=280)

    section_title("🧹", "Nettoyage des données")
    col_a, col_b, col_c = st.columns([2, 2, 1])
    with col_a:
        fill_method = st.selectbox("Stratégie pour valeurs manquantes",
            ["Moyenne", "Médiane", "Mode", "Supprimer les lignes"], key="fill_method")
    with col_b:
        remove_dup = st.checkbox("Supprimer les doublons", value=True, key="rm_dup")
    with col_c:
        st.markdown("<br>", unsafe_allow_html=True)
        apply_clean = st.button("Appliquer", key="clean_btn", width='stretch')

    mv = df.isnull().sum(); mv = mv[mv > 0]
    if not mv.empty:
        with st.expander(f"⚠️ Valeurs manquantes dans {len(mv)} colonne(s)"):
            st.dataframe(mv.rename("Nb. manquants").to_frame(), width='stretch')

    if apply_clean:
        if remove_dup:
            before = len(df); df = df.drop_duplicates()
            info_box(f"{before - len(df)} ligne(s) dupliquée(s) supprimée(s)", "success")
        if fill_method == "Moyenne":    df = df.fillna(df.mean(numeric_only=True))
        elif fill_method == "Médiane":  df = df.fillna(df.median(numeric_only=True))
        elif fill_method == "Mode":     df = df.fillna(df.mode().iloc[0])
        else:                           df = df.dropna()
        for col in df.select_dtypes(include="object").columns:
            df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else "Inconnu")
        st.session_state.df_clean = df.copy()
        info_box(f"Nettoyage terminé — {df.shape[0]:,} lignes restantes · {int(df.isnull().sum().sum())} NaN", "success")

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
        st.markdown(f'<div style="color:#5a6a82;font-size:12px;margin-top:8px;font-weight:500;">Features sélectionnées : {len(feature_cols)}</div>', unsafe_allow_html=True)

    if st.button("⚡ Normaliser les features", key="norm_btn"):
        X = df[feature_cols].values.astype(float)
        imputer = SimpleImputer(strategy="mean")
        X = imputer.fit_transform(X)
        scaler = MinMaxScaler() if "Min-Max" in scaler_choice else StandardScaler()
        X_scaled = scaler.fit_transform(X)
        st.session_state.X_scaled    = X_scaled
        st.session_state.X_features  = feature_cols
        st.session_state.scaler_name = scaler_choice
        st.session_state.y_target    = df[target_col].values if target_col else None
        info_box(f"Normalisation appliquée — {X_scaled.shape[0]:,} échantillons × {X_scaled.shape[1]} features", "success")
        with st.expander("Aperçu des données normalisées"):
            preview_cols = feature_cols[:X_scaled.shape[1]] if len(feature_cols) >= X_scaled.shape[1] else [f"feature_{i+1}" for i in range(X_scaled.shape[1])]
            st.dataframe(pd.DataFrame(X_scaled, columns=preview_cols).head(8).round(4), width='stretch')

    section_title("📊", "Visualisation")
    vis_col1, vis_col2 = st.columns([2, 1])
    with vis_col1:
        vis_type = st.selectbox("Type de graphique",
            ["Boxplot", "Histogrammes", "Heatmap de corrélation", "Matrice de dispersion"], key="vis_type")
    with vis_col2:
        st.markdown("<br>", unsafe_allow_html=True)
        show_vis = st.button("Générer", key="vis_btn", width='stretch')

    if show_vis:
        fig_data  = df[feature_cols].select_dtypes(include=np.number)
        data_json = fig_data.to_json()

        @st.cache_data
        def make_boxplot(djson):
            d = pd.read_json(io.StringIO(djson))
            fig, ax = plt.subplots(figsize=(12, 4))
            bp = ax.boxplot([d[c].dropna().values for c in d.columns], patch_artist=True, notch=False,
                medianprops=dict(color=ACCENT, linewidth=2.5),
                whiskerprops=dict(color=TEXT2_MATPLOTLIB, linewidth=1.2),
                capprops=dict(color=TEXT2_MATPLOTLIB, linewidth=1.2),
                flierprops=dict(marker='o', color=RED, markersize=3, alpha=0.5))
            for i, patch in enumerate(bp['boxes']):
                patch.set_facecolor(CLUSTER_PALETTE[i % len(CLUSTER_PALETTE)] + "28")
                patch.set_edgecolor(CLUSTER_PALETTE[i % len(CLUSTER_PALETTE)]); patch.set_linewidth(1.5)
            ax.set_xticks(range(1, len(d.columns)+1))
            ax.set_xticklabels(d.columns, rotation=35, ha='right', fontsize=8)
            ax.set_title("Boxplot — Distribution des features", pad=14, fontweight='700')
            ax.grid(axis='y', alpha=0.4); fig.tight_layout(); return fig

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
        run_elbow = st.button("Calculer le coude", key="elbow_btn", width='stretch')

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

            section_title("📊", "Métriques d'évaluation", "Silhouette · Davies-Bouldin · Calinski-Harabasz")
            st.markdown(f"""
            <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:10px;margin-bottom:16px;">
              <div style="background:#ffffff;border:1px solid #dde3ed;border-radius:10px;padding:12px 14px;border-top:3px solid {ACCENT};box-shadow:0 1px 4px rgba(15,28,46,.05);">
                <div style="font-size:11px;color:{ACCENT};font-weight:700;text-transform:uppercase;letter-spacing:.07em;">Score Silhouette</div>
                <div style="font-size:11px;color:#5a6a82;margin-top:4px;">Cohésion vs séparation. Plage [−1, 1]. <strong style="color:#0f1c2e;">↑ proche de 1 = meilleur</strong></div>
              </div>
              <div style="background:#ffffff;border:1px solid #dde3ed;border-radius:10px;padding:12px 14px;border-top:3px solid {PURPLE};box-shadow:0 1px 4px rgba(15,28,46,.05);">
                <div style="font-size:11px;color:{PURPLE};font-weight:700;text-transform:uppercase;letter-spacing:.07em;">Indice Davies-Bouldin</div>
                <div style="font-size:11px;color:#5a6a82;margin-top:4px;">Similarité inter-clusters. <strong style="color:#0f1c2e;">↓ proche de 0 = meilleur</strong></div>
              </div>
              <div style="background:#ffffff;border:1px solid #dde3ed;border-radius:10px;padding:12px 14px;border-top:3px solid {GREEN};box-shadow:0 1px 4px rgba(15,28,46,.05);">
                <div style="font-size:11px;color:{GREEN};font-weight:700;text-transform:uppercase;letter-spacing:.07em;">Calinski-Harabasz</div>
                <div style="font-size:11px;color:#5a6a82;margin-top:4px;">Rapport dispersion inter/intra. <strong style="color:#0f1c2e;">↑ plus grand = meilleur</strong></div>
              </div>
            </div>
            """, unsafe_allow_html=True)

            rows = []
            for name, lbl in labels_all.items():
                mask = lbl != -1; valid = len(set(lbl[mask])) >= 2
                rows.append({
                    "Algorithme":          name,
                    "Silhouette ↑":        f"{silhouette_score(X_scaled[mask], lbl[mask]):.4f}"        if valid else "N/A",
                    "Davies-Bouldin ↓":    f"{davies_bouldin_score(X_scaled[mask], lbl[mask]):.4f}"    if valid else "N/A",
                    "Calinski-Harabasz ↑": f"{calinski_harabasz_score(X_scaled[mask], lbl[mask]):.2f}" if valid else "N/A",
                    "Clusters":            str(len(set(lbl[mask]))),
                    "Points de bruit":     str(list(lbl).count(-1)) if -1 in lbl else "—",
                })
            st.dataframe(pd.DataFrame(rows).set_index("Algorithme"), width='stretch')

            section_title("📉", "Comparaison des inerties")

            def calc_inertia(X, labels):
                return sum(np.sum((X[labels==cid] - X[labels==cid].mean(axis=0))**2) for cid in set(labels) if cid != -1)

            inerties = {n: calc_inertia(X_scaled, l) for n, l in labels_all.items()}
            fig_in, ax_in = plt.subplots(figsize=(9, 3.8))
            bar_colors = [CLUSTER_PALETTE[i % len(CLUSTER_PALETTE)] for i in range(len(inerties))]
            bars = ax_in.bar(list(inerties.keys()), list(inerties.values()), color=bar_colors, edgecolor='white', linewidth=1.2, width=0.55)
            for bar in bars:
                ax_in.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3, f"{bar.get_height():.1f}", ha='center', va='bottom', fontsize=9, color=TEXT2_MATPLOTLIB, fontweight='600')
            ax_in.set_title("Comparaison des inerties — Tous les algorithmes", fontweight='700')
            ax_in.set_ylabel("Inertie (WCSS)"); ax_in.grid(axis='y', alpha=0.5); fig_in.tight_layout()
            st.pyplot(fig_in); plt.close(fig_in)

            section_title("🔭", "Projection PCA 2D")
            pca = PCA(n_components=2); X_2d = pca.fit_transform(X_scaled)
            var = pca.explained_variance_ratio_ * 100
            info_box(f"Variance expliquée — PC1 : <strong>{var[0]:.1f}%</strong>  ·  PC2 : <strong>{var[1]:.1f}%</strong>  ·  Total : <strong>{var[0]+var[1]:.1f}%</strong>", "info")

            n_alg = len(labels_all)
            fig_pca, axes = plt.subplots(1, n_alg, figsize=(5*n_alg, 4.8))
            if n_alg == 1: axes = [axes]
            algo_col_map = {"K-Means": ACCENT, "K-Medoids": PURPLE, "AGNES": GREEN, "DIANA": AMBER, "DBSCAN": CYAN}
            for ax, (name, lbl) in zip(axes, labels_all.items()):
                for cid in sorted(set(lbl)):
                    mask = lbl == cid
                    ax.scatter(X_2d[mask,0], X_2d[mask,1], c="#aab4c0" if cid==-1 else CLUSTER_PALETTE[cid%len(CLUSTER_PALETTE)],
                               s=22, alpha=0.80, linewidths=0, label="Bruit" if cid==-1 else f"C{cid}")
                ax.set_title(name, fontsize=11, fontweight='700', pad=8, color=algo_col_map.get(name, ACCENT))
                ax.set_xlabel(f"PC1 ({var[0]:.1f}%)", fontsize=8); ax.set_ylabel(f"PC2 ({var[1]:.1f}%)", fontsize=8)
                ax.legend(fontsize=7, markerscale=1.5)
                for spine in ax.spines.values(): spine.set_color(BORDER_MATPLOTLIB)
            fig_pca.suptitle("Projections des clusters (PCA 2D)", fontsize=13, fontweight='700')
            fig_pca.tight_layout(); st.pyplot(fig_pca); plt.close(fig_pca)

            if "AGNES" in algos:
                section_title("🌿", "AGNES — Dendrogramme", f"Liaison : {agnes_linkage}")
                st.markdown(f"""<div style="background:#f0fdf4;border:1px solid rgba(22,163,74,.25);border-radius:10px;
                            padding:12px 16px;margin-bottom:12px;font-size:12px;color:#5a6a82;line-height:1.6;">
                  Un <strong style="color:#0f1c2e;">dendrogramme</strong> montre comment les clusters sont fusionnés à chaque étape.
                  L'<strong style="color:#0f1c2e;">axe vertical</strong> représente la distance de fusion.
                  Les <strong style="color:{GREEN};">branches colorées</strong> en dessous du seuil = clusters finaux.
                </div>""", unsafe_allow_html=True)
                n_samp = min(250, X_scaled.shape[0])
                link_mat = linkage(X_scaled[:n_samp], method=agnes_linkage)
                threshold = 0.7 * max(link_mat[:, 2])
                fig_d, ax_d = plt.subplots(figsize=(13, 5))
                dendrogram(link_mat, truncate_mode='lastp', p=30, leaf_rotation=40, leaf_font_size=8,
                           color_threshold=threshold, above_threshold_color=TEXT2_MATPLOTLIB, ax=ax_d)
                ax_d.axhline(y=threshold, color=AMBER, linewidth=1.5, linestyle='--', label=f"Seuil ({threshold:.2f})")
                ax_d.set_title(f"Dendrogramme AGNES (liaison {agnes_linkage}) — {n_samp} premiers échantillons", fontweight='700')
                ax_d.set_xlabel("Indice / taille du cluster"); ax_d.set_ylabel("Distance de fusion")
                ax_d.legend(fontsize=9); ax_d.grid(axis='y', alpha=0.4); fig_d.tight_layout()
                st.pyplot(fig_d); plt.close(fig_d)

            if "DIANA" in algos:
                section_title("✂️", "DIANA — Analyse Divisive", f"Liaison : {diana_linkage}")
                st.markdown(f"""<div style="background:#fffbeb;border:1px solid rgba(217,119,6,.25);border-radius:10px;
                            padding:12px 16px;font-size:12px;color:#5a6a82;line-height:1.6;">
                  <strong style="color:#0f1c2e;">DIANA</strong> est le pendant <em>descendant</em> d'AGNES. Il commence avec
                  les {X_scaled.shape[0]} points dans un seul cluster et divise récursivement le plus hétérogène,
                  via la liaison <strong style="color:{AMBER};">{diana_linkage}</strong>. Produit exactement k = {k} clusters.
                </div>""", unsafe_allow_html=True)

            if "DBSCAN" in algos:
                section_title("🔵", "DBSCAN — Analyse par densité", f"ε={eps_val} · MinPts={min_samp}")
                lbl_db = labels_all["DBSCAN"]
                nc_db = len(set(lbl_db)) - (1 if -1 in lbl_db else 0)
                nn_db = list(lbl_db).count(-1)
                st.markdown(f"""<div style="background:#ecfeff;border:1px solid rgba(8,145,178,.25);border-radius:10px;
                            padding:12px 16px;margin-bottom:12px;font-size:12px;color:#5a6a82;line-height:1.6;">
                  <strong style="color:#0f1c2e;">DBSCAN</strong> regroupe les points densément regroupés tout en marquant les épars comme
                  <strong style="color:{RED};">bruit</strong>. Ne nécessite pas k à l'avance.<br>
                  <strong>Résultat :</strong> <strong style="color:#0f1c2e;">{nc_db} cluster(s)</strong> et
                  <strong style="color:{RED};">{nn_db} bruit(s)</strong> ({nn_db/len(lbl_db)*100:.1f}%) avec ε={eps_val}, MinPts={min_samp}.
                </div>""", unsafe_allow_html=True)

                if nc_db > 0:
                    cluster_sizes = [int(np.sum(lbl_db==c)) for c in range(nc_db)]
                    labels_pie = [f"C{c} ({cluster_sizes[c]})" for c in range(nc_db)]
                    if nn_db > 0: cluster_sizes.append(nn_db); labels_pie.append(f"Bruit ({nn_db})")
                    fig_pie, ax_pie = plt.subplots(figsize=(6, 4))
                    colors_pie = [CLUSTER_PALETTE[i%len(CLUSTER_PALETTE)] for i in range(nc_db)] + (["#aab4c0"] if nn_db>0 else [])
                    wedges, texts, autotexts = ax_pie.pie(cluster_sizes, labels=labels_pie, colors=colors_pie,
                        autopct='%1.1f%%', pctdistance=0.82, wedgeprops=dict(edgecolor='white', linewidth=2), startangle=90)
                    for t in texts: t.set_color(TEXT2_MATPLOTLIB); t.set_fontsize(9)
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


# ───────────────────────────────────────────────────────────────
#  VOLET 3 : CLASSIFICATION
# ───────────────────────────────────────────────────────────────
with tab3:
    if st.session_state.X_scaled is None or st.session_state.y_target is None:
        st.markdown("""
        <div style="text-align:center;padding:4rem 2rem;color:#5a6a82;background:#ffffff;
                    border:2px dashed #c7d2e0;border-radius:16px;margin-top:1rem;">
            <div style="font-size:40px;margin-bottom:12px;">⚠️</div>
            <div style="font-size:16px;font-weight:700;color:#0f1c2e;margin-bottom:6px;">Données non préparées</div>
            <div style="font-size:13px;">Normalisez les données dans le Volet 1 (une colonne cible doit être détectée)</div>
        </div>
        """, unsafe_allow_html=True)
        st.stop()

    X = st.session_state.X_scaled
    y = st.session_state.y_target
    unique_classes, class_counts = np.unique(y, return_counts=True)

    section_title("⚙️", "Configuration du modèle")
    metric_row([
        {"label": "Échantillons", "value": f"{len(y):,}",           "color": ACCENT},
        {"label": "Features",     "value": str(X.shape[1]),          "color": GREEN},
        {"label": "Classes",      "value": str(len(unique_classes)), "color": PURPLE},
        {"label": "Classe min.",  "value": f"{class_counts.min()/len(y)*100:.1f}%", "color": AMBER},
    ])

    cfg1, cfg2, cfg3 = st.columns([1, 2, 1])
    with cfg1: test_size = st.slider("Test (%)", 10, 40, 20, key="test_sz") / 100
    with cfg2:
        model_name = st.selectbox("Classificateur", [
            "K-NN", "Arbre de décision", "Naïve Bayes",
            "SVM (RBF)", "Régression logistique", "Random Forest"], key="clf_sel")
    with cfg3:
        k_knn = st.slider("K (K-NN uniquement)", 1, 15, 5, key="k_knn")

    run_clf = st.button("🎯  Entraîner le modèle", key="run_clf")

    if run_clf:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)
        models = {
            "K-NN":                    KNeighborsClassifier(n_neighbors=k_knn),
            "Arbre de décision":       DecisionTreeClassifier(random_state=42),
            "Naïve Bayes":             GaussianNB(),
            "SVM (RBF)":               SVC(random_state=42, probability=True),
            "Régression logistique":   LogisticRegression(max_iter=1000, random_state=42),
            "Random Forest":           RandomForestClassifier(n_estimators=100, random_state=42),
        }
        clf = models[model_name]
        with st.spinner(f"Entraînement de {model_name}…"):
            clf.fit(X_train, y_train); y_pred = clf.predict(X_test)

        report = classification_report(y_test, y_pred, output_dict=True)
        acc = report["accuracy"]; prec = report["weighted avg"]["precision"]
        rec = report["weighted avg"]["recall"]; f1 = report["weighted avg"]["f1-score"]

        section_title("📊", "Résultats", model_name)
        metric_row([
            {"label": "Accuracy",  "value": f"{acc:.4f}",  "color": GREEN if acc  >= 0.8 else AMBER},
            {"label": "Précision", "value": f"{prec:.4f}", "color": GREEN if prec >= 0.8 else AMBER},
            {"label": "Rappel",    "value": f"{rec:.4f}",  "color": GREEN if rec  >= 0.8 else AMBER},
            {"label": "F1-Score",  "value": f"{f1:.4f}",   "color": GREEN if f1   >= 0.8 else AMBER},
            {"label": "Train",     "value": f"{len(X_train):,}", "color": ACCENT},
            {"label": "Test",      "value": f"{len(X_test):,}",  "color": ACCENT},
        ])

        col_r1, col_r2 = st.columns(2)
        with col_r1:
            section_title("🟦", "Matrice de Confusion")
            cm = confusion_matrix(y_test, y_pred)
            fig_cm, ax_cm = plt.subplots(figsize=(5, 4))
            sns.heatmap(cm, annot=True, fmt='d', cmap=sns.light_palette(ACCENT, as_cmap=True),
                        xticklabels=unique_classes, yticklabels=unique_classes,
                        ax=ax_cm, linewidths=0.6, linecolor='white',
                        annot_kws={"size": 11, "weight": "bold"}, cbar_kws={"shrink": 0.7})
            ax_cm.set_xlabel("Prédit", fontsize=10, fontweight='600')
            ax_cm.set_ylabel("Réel", fontsize=10, fontweight='600')
            ax_cm.set_title(f"Matrice de confusion — {model_name}", pad=12, fontweight='700')
            fig_cm.tight_layout(); st.pyplot(fig_cm); plt.close(fig_cm)

        with col_r2:
            section_title("📋", "Rapport de Classification")
            rep_df = pd.DataFrame(report).T.drop(index=["accuracy"], errors="ignore")
            st.dataframe(rep_df.style.format("{:.4f}").background_gradient(
                cmap="Blues", subset=["precision","recall","f1-score"]),
                width='stretch', height=260)

        if model_name in ["Arbre de décision", "Random Forest"]:
            section_title("🌟", "Importance des features")
            fi = pd.DataFrame({"Feature": st.session_state.X_features, "Importance": clf.feature_importances_}).sort_values("Importance", ascending=True).tail(15)
            fig_fi, ax_fi = plt.subplots(figsize=(9, 4))
            ax_fi.barh(fi["Feature"], fi["Importance"], color=[CLUSTER_PALETTE[i%len(CLUSTER_PALETTE)] for i in range(len(fi))], edgecolor='white', linewidth=0.8)
            ax_fi.set_title("Importance des features", fontweight='700')
            ax_fi.set_xlabel("Score d'importance"); ax_fi.grid(axis='x', alpha=0.4); fig_fi.tight_layout()
            st.pyplot(fig_fi); plt.close(fig_fi)

        section_title("⚖️", "Comparaison des classificateurs")
        if st.button("Comparer tous les classificateurs", key="compare_all"):
            rows_cmp = []; prog = st.progress(0)
            for i, (mname, clf_) in enumerate(models.items()):
                try:
                    clf_.fit(X_train, y_train); yp = clf_.predict(X_test)
                    rp = classification_report(yp, y_test, output_dict=True)
                    rows_cmp.append({"Modèle": mname, "Accuracy": round(rp["accuracy"],4),
                        "Précision": round(rp["weighted avg"]["precision"],4),
                        "Rappel": round(rp["weighted avg"]["recall"],4),
                        "F1-Score": round(rp["weighted avg"]["f1-score"],4)})
                except Exception: pass
                prog.progress((i+1)/len(models))

            comp_df = pd.DataFrame(rows_cmp).set_index("Modèle")
            st.dataframe(comp_df.style.highlight_max(axis=0, color='rgba(22,163,74,.20)').format("{:.4f}"), width='stretch')

            fig_cmp, ax_cmp = plt.subplots(figsize=(11, 4))
            x_pos = np.arange(len(comp_df)); width = 0.2
            for i, col in enumerate(["Accuracy","Précision","Rappel","F1-Score"]):
                ax_cmp.bar(x_pos+i*width, comp_df[col], width=width, label=col, color=CLUSTER_PALETTE[i], edgecolor='white', linewidth=0.8)
            ax_cmp.set_xticks(x_pos+width*1.5)
            ax_cmp.set_xticklabels(comp_df.index, rotation=20, ha='right', fontsize=9)
            ax_cmp.set_ylim(0, 1.1); ax_cmp.set_title("Comparaison des classificateurs", fontweight='700')
            ax_cmp.set_ylabel("Score"); ax_cmp.legend(loc='lower right', fontsize=9); ax_cmp.grid(axis='y', alpha=0.4)
            fig_cmp.tight_layout(); st.pyplot(fig_cmp); plt.close(fig_cmp)


# ── Footer ──────────────────────────────────────────────────────
st.markdown("""
<div style="margin-top:3rem;padding-top:1.2rem;border-top:1px solid #dde3ed;
            display:flex;justify-content:space-between;align-items:center;">
    <span style="font-size:11px;color:#5a6a82;font-weight:500;">🔬 FD1 · Interface Fouille de Données · M1 Bioinformatique</span>
    <span style="font-size:11px;color:#5a6a82;">Faculté d'Informatique · Année 2025–2026</span>
</div>
""", unsafe_allow_html=True)