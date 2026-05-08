# ╔══════════════════════════════════════════════════════════════════╗
# ║           NEXUS AutoML Studio  — Complete Premium Edition        ║
# ║   EDA · Missing Values · Feature Engineering · Multi-Model ML   ║
# ║   Outlier Detection · Data Export · Model Comparison · Report   ║
# ╚══════════════════════════════════════════════════════════════════╝

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import warnings, io, base64, time
warnings.filterwarnings("ignore")

# ── Sklearn ──────────────────────────────────────────────────────
from sklearn.model_selection import (
    train_test_split, cross_val_score, StratifiedKFold, KFold
)
from sklearn.preprocessing import (
    LabelEncoder, StandardScaler, MinMaxScaler, RobustScaler
)
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor,
    ExtraTreesClassifier, ExtraTreesRegressor,
    AdaBoostClassifier, AdaBoostRegressor
)
from sklearn.linear_model import (
    LogisticRegression, LinearRegression, Ridge, Lasso, ElasticNet
)
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score, r2_score, mean_squared_error, mean_absolute_error,
    classification_report, confusion_matrix, roc_auc_score,
    precision_score, recall_score, f1_score
)

# ══════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="NEXUS · AutoML Studio",
    page_icon="⬡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ══════════════════════════════════════════════════════════════════
# GLOBAL CSS
# ══════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;800&family=Space+Mono:wght@400;700&family=DM+Mono:wght@300;400;500&display=swap');

:root {
    --bg-base:    #07090d;
    --bg-surface: #0d1117;
    --bg-raised:  #131b23;
    --bg-hover:   #192030;
    --border:     #1c2a38;
    --border-lit: #00c8f033;
    --cyan:       #00c8f0;
    --cyan-d:     #00c8f077;
    --cyan-p:     #00c8f018;
    --green:      #00f5a0;
    --green-d:    #00f5a055;
    --amber:      #f0b429;
    --amber-d:    #f0b42944;
    --red:        #f05779;
    --red-d:      #f0577933;
    --purple:     #b57af5;
    --purple-d:   #b57af544;
    --txt-hi:     #ddeeff;
    --txt-md:     #6e93ae;
    --txt-lo:     #344d61;
    --font-d:     'Syne', sans-serif;
    --font-m:     'Space Mono', monospace;
    --font-b:     'DM Mono', monospace;
}

*, *::before, *::after { box-sizing: border-box; margin: 0; }
html, body, [class*="css"] {
    font-family: var(--font-b) !important;
    background: var(--bg-base) !important;
    color: var(--txt-hi) !important;
}
.stApp {
    background: var(--bg-base) !important;
    background-image: radial-gradient(ellipse 80% 35% at 50% 0%, #00c8f00a 0%, transparent 60%) !important;
}
#MainMenu, footer, header, .stDeployButton { visibility: hidden !important; display: none !important; }
::-webkit-scrollbar { width: 4px; height: 4px; }
::-webkit-scrollbar-track { background: var(--bg-surface); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 2px; }

/* Sidebar */
[data-testid="stSidebar"] {
    background: var(--bg-surface) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] > div:first-child { padding-top: 0 !important; }

/* Headings */
h1 { font-family: var(--font-d) !important; font-weight: 800 !important;
     color: var(--txt-hi) !important; letter-spacing: -0.02em !important; }
h2 { font-family: var(--font-d) !important; font-weight: 600 !important;
     color: var(--cyan) !important; letter-spacing: 0.06em !important;
     text-transform: uppercase !important; }
h3 { font-family: var(--font-m) !important; color: var(--txt-md) !important;
     text-transform: uppercase !important; letter-spacing: 0.1em !important; }

/* Metrics */
[data-testid="metric-container"] {
    background: var(--bg-surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 2px !important;
    padding: 20px 24px !important;
    position: relative !important; overflow: hidden !important;
    transition: box-shadow 0.25s, border-color 0.25s !important;
}
[data-testid="metric-container"]::before {
    content: ''; position: absolute; top: 0; left: 0;
    width: 3px; height: 100%;
    background: linear-gradient(180deg, var(--cyan), transparent);
}
[data-testid="metric-container"]:hover {
    border-color: var(--cyan-d) !important;
    box-shadow: 0 0 24px var(--border-lit) !important;
}
[data-testid="stMetricLabel"] {
    font-family: var(--font-m) !important; font-size: 0.62rem !important;
    color: var(--txt-lo) !important; text-transform: uppercase !important;
    letter-spacing: 0.14em !important;
}
[data-testid="stMetricValue"] {
    font-family: var(--font-d) !important; font-size: 1.9rem !important;
    font-weight: 800 !important; color: var(--txt-hi) !important;
}

/* Buttons */
.stButton > button {
    background: transparent !important; color: var(--cyan) !important;
    border: 1px solid var(--cyan-d) !important; border-radius: 2px !important;
    font-family: var(--font-m) !important; font-size: 0.72rem !important;
    font-weight: 700 !important; letter-spacing: 0.14em !important;
    text-transform: uppercase !important; padding: 10px 28px !important;
    transition: all 0.2s ease !important;
}
.stButton > button:hover {
    background: var(--cyan-p) !important; border-color: var(--cyan) !important;
    box-shadow: 0 0 20px var(--border-lit), inset 0 0 20px var(--border-lit) !important;
    transform: translateY(-1px) !important;
}

/* Inputs */
.stTextInput > div > div > input,
.stNumberInput > div > div > input {
    background: var(--bg-surface) !important; border: 1px solid var(--border) !important;
    border-radius: 2px !important; color: var(--txt-hi) !important;
    font-family: var(--font-b) !important; font-size: 0.85rem !important;
    padding: 10px 14px !important; transition: border-color 0.2s !important;
}
.stTextInput > div > div > input:focus,
.stNumberInput > div > div > input:focus {
    border-color: var(--cyan) !important; box-shadow: 0 0 0 2px var(--cyan-p) !important;
}
.stSelectbox > div > div, .stMultiSelect > div > div {
    background: var(--bg-surface) !important; border: 1px solid var(--border) !important;
    border-radius: 2px !important; color: var(--txt-hi) !important;
}
.stSelectbox label, .stMultiSelect label,
.stTextInput label, .stNumberInput label,
.stSlider label, .stRadio label, .stCheckbox label {
    font-family: var(--font-m) !important; font-size: 0.68rem !important;
    color: var(--txt-lo) !important; text-transform: uppercase !important;
    letter-spacing: 0.1em !important; margin-bottom: 4px !important;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    background: transparent !important;
    border-bottom: 1px solid var(--border) !important;
    gap: 0 !important;
}
.stTabs [data-baseweb="tab"] {
    font-family: var(--font-m) !important; font-size: 0.68rem !important;
    color: var(--txt-lo) !important; letter-spacing: 0.1em !important;
    text-transform: uppercase !important; padding: 14px 20px !important;
    background: transparent !important; border: none !important;
    transition: color 0.2s !important;
}
.stTabs [data-baseweb="tab"]:hover { color: var(--txt-hi) !important; }
.stTabs [aria-selected="true"] {
    color: var(--cyan) !important;
    border-bottom: 2px solid var(--cyan) !important;
}

/* Expander */
[data-testid="stExpander"] {
    background: var(--bg-surface) !important;
    border: 1px solid var(--border) !important; border-radius: 2px !important;
}
[data-testid="stExpander"] summary {
    font-family: var(--font-m) !important; font-size: 0.72rem !important;
    color: var(--txt-md) !important; letter-spacing: 0.06em !important;
}

/* Alerts */
.stSuccess, .stInfo, .stWarning, .stError { border-radius: 0 !important; }
.stSuccess { border-left: 3px solid var(--green) !important; background: #00f5a00d !important; }
.stInfo    { border-left: 3px solid var(--cyan) !important;  background: #00c8f00d !important; }
.stWarning { border-left: 3px solid var(--amber) !important; background: #f0b4290d !important; }
.stError   { border-left: 3px solid var(--red) !important;   background: #f057790d !important; }

hr { border-color: var(--border) !important; margin: 1.5rem 0 !important; }

/* ── Custom Components ── */
.nx-section {
    display: flex; align-items: center; gap: 14px;
    margin: 28px 0 18px 0;
    padding-bottom: 10px;
    border-bottom: 1px solid var(--border);
}
.nx-section-num {
    font-family: var(--font-m); font-size: 0.6rem; color: var(--cyan);
    border: 1px solid var(--cyan-d); padding: 3px 7px; border-radius: 2px;
    letter-spacing: 0.1em;
}
.nx-section-title {
    font-family: var(--font-d); font-size: 1rem; font-weight: 600;
    color: var(--txt-hi); letter-spacing: 0.04em;
}

.nx-badge {
    display: inline-block; font-family: var(--font-m); font-size: 0.6rem;
    letter-spacing: 0.08em; padding: 3px 9px; border-radius: 2px; margin: 2px;
}
.b-num  { background: #00c8f010; color: var(--cyan);   border: 1px solid var(--cyan-d); }
.b-cat  { background: #b57af510; color: var(--purple); border: 1px solid var(--purple-d); }
.b-dt   { background: #f0b42910; color: var(--amber);  border: 1px solid var(--amber-d); }
.b-bool { background: #00f5a010; color: var(--green);  border: 1px solid var(--green-d); }
.b-hc   { background: #f0577910; color: var(--red);    border: 1px solid var(--red-d); }

.nx-score-box {
    background: var(--bg-surface); border: 1px solid var(--border);
    border-radius: 2px; padding: 20px 24px;
}
.nx-score-row {
    display: flex; justify-content: space-between; align-items: center;
    padding: 8px 0; border-bottom: 1px solid var(--border);
    font-family: var(--font-m); font-size: 0.78rem;
}
.nx-score-row:last-child { border-bottom: none; }
.nx-score-label { color: var(--txt-md); letter-spacing: 0.05em; }
.nx-score-val   { color: var(--txt-hi); font-weight: 700; }
.nx-score-val.g { color: var(--green); }
.nx-score-val.r { color: var(--red); }
.nx-score-val.a { color: var(--amber); }

.nx-fi-row {
    display: flex; align-items: center; gap: 12px;
    padding: 6px 0;
}
.nx-fi-name  { font-family: var(--font-b); font-size: 0.75rem; color: var(--txt-md); width: 200px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.nx-fi-track { flex: 1; height: 4px; background: var(--bg-raised); border-radius: 2px; overflow: hidden; }
.nx-fi-fill  { height: 100%; background: linear-gradient(90deg, var(--cyan), var(--green)); border-radius: 2px; }
.nx-fi-pct   { font-family: var(--font-m); font-size: 0.7rem; color: var(--cyan); width: 50px; text-align: right; }

.nx-prob-row {
    display: flex; align-items: center; gap: 12px; padding: 7px 0;
}
.nx-prob-label { font-family: var(--font-b); font-size: 0.78rem; color: var(--txt-md); width: 160px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.nx-prob-track { flex: 1; height: 6px; background: var(--bg-raised); border-radius: 2px; overflow: hidden; }
.nx-prob-fill  { height: 100%; border-radius: 2px; transition: width 0.6s ease; }
.nx-prob-pct   { font-family: var(--font-m); font-size: 0.72rem; width: 52px; text-align: right; }

.nx-result-card {
    background: linear-gradient(135deg, var(--bg-surface), var(--bg-raised));
    border: 1px solid var(--green-d);
    border-radius: 2px; padding: 36px; text-align: center;
    box-shadow: 0 0 48px var(--green-d);
}
.nx-result-label {
    font-family: var(--font-m); font-size: 0.65rem; color: var(--txt-lo);
    letter-spacing: 0.2em; text-transform: uppercase; margin-bottom: 12px;
}
.nx-result-val {
    font-family: var(--font-d); font-size: 3.2rem; color: var(--green);
    font-weight: 800; line-height: 1; text-shadow: 0 0 32px var(--green-d);
}

.nx-model-badge {
    display: flex; gap: 2px;
    background: var(--bg-surface); border: 1px solid var(--border);
    border-radius: 2px; padding: 16px 24px; margin-bottom: 24px;
    flex-wrap: wrap;
}
.nx-badge-item-label { font-family: var(--font-m); font-size: 0.6rem; color: var(--txt-lo);
                        letter-spacing: 0.1em; text-transform: uppercase; margin-bottom: 4px; }
.nx-badge-item-val   { font-family: var(--font-d); font-size: 0.95rem; color: var(--txt-hi); font-weight: 600; }
.nx-badge-item { margin-right: 32px; }

.nx-compare-row {
    display: flex; align-items: center; gap: 12px; padding: 10px 14px;
    border-bottom: 1px solid var(--border); transition: background 0.2s;
}
.nx-compare-row:hover { background: var(--bg-hover); }
.nx-compare-rank { font-family: var(--font-m); font-size: 0.65rem; color: var(--txt-lo); width: 28px; }
.nx-compare-name { font-family: var(--font-b); font-size: 0.82rem; color: var(--txt-hi); flex: 1; }
.nx-compare-score { font-family: var(--font-m); font-size: 0.9rem; font-weight: 700; width: 90px; text-align: right; }
.nx-compare-bar-track { flex: 1; height: 4px; background: var(--bg-raised); border-radius: 2px; }
.nx-compare-bar-fill  { height: 100%; border-radius: 2px; background: linear-gradient(90deg, var(--cyan), var(--green)); }

.nx-outlier-pill {
    display: inline-block; font-family: var(--font-m); font-size: 0.65rem;
    padding: 3px 10px; border-radius: 2px; margin: 2px;
    background: var(--red-d); color: var(--red); border: 1px solid var(--red-d);
}

.nx-download-btn a {
    display: inline-block;
    background: transparent; color: var(--green);
    border: 1px solid var(--green-d); border-radius: 2px;
    font-family: var(--font-m); font-size: 0.68rem; font-weight: 700;
    letter-spacing: 0.12em; text-transform: uppercase;
    padding: 9px 22px; text-decoration: none;
    transition: all 0.2s ease;
}
.nx-download-btn a:hover {
    background: #00f5a015; border-color: var(--green);
    box-shadow: 0 0 16px var(--green-d);
}

.nx-sidebar-logo {
    font-family: var(--font-d); font-size: 1.4rem; font-weight: 800;
    color: var(--cyan); letter-spacing: 0.04em;
    padding: 24px 20px 4px 20px;
}
.nx-sidebar-tag {
    font-family: var(--font-m); font-size: 0.6rem; color: var(--txt-lo);
    letter-spacing: 0.12em; padding: 0 20px 20px 20px;
}
.nx-sidebar-stat {
    font-family: var(--font-m); font-size: 0.68rem; padding: 3px 0;
}
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
# CHART THEME
# ══════════════════════════════════════════════════════════════════
BG    = "#07090d"
SURF  = "#0d1117"
GRID  = "#1c2a38"
TXT   = "#6e93ae"
CYAN  = "#00c8f0"
GREEN = "#00f5a0"
RED   = "#f05779"
AMB   = "#f0b429"
PRP   = "#b57af5"
PAL   = [CYAN, GREEN, PRP, AMB, RED, "#4fc3f7", "#80cbc4", "#ce93d8"]

def sax(ax, title="", xl="", yl=""):
    ax.set_facecolor(SURF)
    ax.tick_params(colors=TXT, labelsize=8)
    for s in ax.spines.values(): s.set_edgecolor(GRID)
    ax.grid(color=GRID, lw=0.4, alpha=0.7); ax.set_axisbelow(True)
    if title: ax.set_title(title, color="#ddeeff", fontsize=10, pad=8, fontfamily="monospace")
    if xl:    ax.set_xlabel(xl, color=TXT, fontsize=8)
    if yl:    ax.set_ylabel(yl, color=TXT, fontsize=8)

def mfig(w=10, h=4.5):
    fig, ax = plt.subplots(figsize=(w, h), facecolor=BG); sax(ax); return fig, ax

def mfigs(r, c, w=14, h=5):
    fig, axes = plt.subplots(r, c, figsize=(w, h), facecolor=BG)
    axs = np.array([axes]).flatten() if not isinstance(axes, np.ndarray) else axes.flatten()
    for a in axs: sax(a)
    return fig, axs

def df_download_link(df, fname="data.csv", label="⬡  Download CSV"):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    return f'<div class="nx-download-btn"><a href="data:file/csv;base64,{b64}" download="{fname}">{label}</a></div>'

# ══════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════
def detect_col_types(df):
    r = {}
    for c in df.columns:
        if pd.api.types.is_bool_dtype(df[c]):                                  r[c] = "boolean"
        elif pd.api.types.is_datetime64_any_dtype(df[c]):                      r[c] = "datetime"
        elif pd.api.types.is_numeric_dtype(df[c]):                             r[c] = "numeric"
        elif df[c].nunique() / max(len(df),1) > 0.85 and df[c].nunique() > 50: r[c] = "high_cardinality"
        else:                                                                   r[c] = "categorical"
    return r

def try_parse_dates(df):
    for c in df.select_dtypes(include="object").columns:
        try:
            pd.to_datetime(df[c].dropna().head(40), infer_datetime_format=True, errors="raise")
            df[c] = pd.to_datetime(df[c], infer_datetime_format=True, errors="coerce")
        except: pass
    return df

def smart_impute(df, strategy_map):
    df = df.copy()
    knn_cols = [c for c,s in strategy_map.items()
                if s == "knn" and pd.api.types.is_numeric_dtype(df[c])]
    if knn_cols:
        df[knn_cols] = KNNImputer(n_neighbors=5).fit_transform(df[knn_cols])
    for col, strat in strategy_map.items():
        if col not in df.columns or strat == "knn": continue
        if   strat == "drop_rows":                   df = df.dropna(subset=[col])
        elif strat == "ffill":                        df[col] = df[col].ffill().bfill()
        elif strat == "bfill":                        df[col] = df[col].bfill().ffill()
        elif strat == "mean"   and pd.api.types.is_numeric_dtype(df[col]): df[col] = df[col].fillna(df[col].mean())
        elif strat == "median" and pd.api.types.is_numeric_dtype(df[col]): df[col] = df[col].fillna(df[col].median())
        elif strat == "mode":
            m = df[col].mode(); df[col] = df[col].fillna(m[0] if len(m) else "MISSING")
        elif strat == "constant":
            df[col] = df[col].fillna("MISSING" if df[col].dtype == "object" else -999)
        else:
            if pd.api.types.is_numeric_dtype(df[col]): df[col] = df[col].fillna(df[col].median())
            else:
                m = df[col].mode()
                if len(m): df[col] = df[col].fillna(m[0])
    return df

def encode_features(df, col_types, target):
    df = df.copy(); encoders = {}
    for col in df.columns:
        if col == target: continue
        ct = col_types.get(col, "categorical")
        if ct == "datetime":
            dt = pd.to_datetime(df[col], errors="coerce")
            df[col+"_year"] = dt.dt.year; df[col+"_month"] = dt.dt.month
            df[col+"_day"]  = dt.dt.day;  df[col+"_dow"]   = dt.dt.dayofweek
            df = df.drop(columns=[col])
        elif ct in ("categorical","boolean","high_cardinality"):
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            encoders[col] = le
    return df, encoders

def detect_outliers_iqr(df, cols):
    outlier_map = {}
    for c in cols:
        q1,q3 = df[c].quantile(0.25), df[c].quantile(0.75)
        iqr = q3 - q1
        mask = (df[c] < q1-1.5*iqr) | (df[c] > q3+1.5*iqr)
        outlier_map[c] = int(mask.sum())
    return outlier_map

def safe_cv_folds(y, problem, n_min=2, n_max=5):
    if problem == "Classification":
        min_class = int(pd.Series(y).value_counts().min())
        return max(n_min, min(n_max, min_class))
    else:
        return max(n_min, min(n_max, len(y)//10))

def scale_features(X_train, X_test, method):
    scalers = {"Standard": StandardScaler(), "MinMax": MinMaxScaler(), "Robust": RobustScaler()}
    if method == "None": return X_train, X_test
    s = scalers[method]; return s.fit_transform(X_train), s.transform(X_test)

MODELS_CLF = {
    "Random Forest":       RandomForestClassifier(n_estimators=100, random_state=42),
    "Extra Trees":         ExtraTreesClassifier(n_estimators=100, random_state=42),
    "Gradient Boosting":   GradientBoostingClassifier(random_state=42),
    "AdaBoost":            AdaBoostClassifier(random_state=42),
    "Decision Tree":       DecisionTreeClassifier(random_state=42),
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "KNN":                 KNeighborsClassifier(),
    "Naive Bayes":         GaussianNB(),
    "SVM":                 SVC(probability=True, random_state=42),
}
MODELS_REG = {
    "Random Forest":      RandomForestRegressor(n_estimators=100, random_state=42),
    "Extra Trees":        ExtraTreesRegressor(n_estimators=100, random_state=42),
    "Gradient Boosting":  GradientBoostingRegressor(random_state=42),
    "AdaBoost":           AdaBoostRegressor(random_state=42),
    "Decision Tree":      DecisionTreeRegressor(random_state=42),
    "Ridge":              Ridge(),
    "Lasso":              Lasso(),
    "ElasticNet":         ElasticNet(),
    "Linear Regression":  LinearRegression(),
    "KNN":                KNeighborsRegressor(),
    "SVR":                SVR(),
}

@st.cache_data
def load_data(file):
    for enc in ["utf-8","latin1","iso-8859-1","cp1252"]:
        for sep in [",","\t",";","|"]:
            try:
                file.seek(0)
                df = pd.read_csv(file, encoding=enc, sep=sep, on_bad_lines="skip", engine="python")
                if df.shape[1] > 1: return df, enc, sep
            except: continue
    file.seek(0)
    return pd.read_csv(file, on_bad_lines="skip"), "utf-8", ","

badge_css = {"numeric":"b-num","categorical":"b-cat","datetime":"b-dt",
             "boolean":"b-bool","high_cardinality":"b-hc"}

def sec(num, title):
    st.markdown(f'<div class="nx-section"><span class="nx-section-num">{num}</span>'
                f'<span class="nx-section-title">{title}</span></div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown('<div class="nx-sidebar-logo">⬡ NEXUS</div>', unsafe_allow_html=True)
    st.markdown('<div class="nx-sidebar-tag">AUTOML STUDIO  ·  v3.0</div>', unsafe_allow_html=True)
    st.markdown("<hr>", unsafe_allow_html=True)

    page = st.radio("NAVIGATION", [
        "01 · Overview & EDA",
        "02 · Missing Values",
        "03 · Outlier Detection",
        "04 · Feature Engineering",
        "05 · Model Training",
        "06 · Model Comparison",
        "07 · Live Prediction",
        "08 · Export & Report",
    ], label_visibility="visible")

    st.markdown("<hr>", unsafe_allow_html=True)

    if "df_clean" in st.session_state:
        ds = st.session_state["df_clean"]
        st.markdown(f"""
        <div style="background:var(--bg-raised);border:1px solid var(--border);
                    border-radius:2px;padding:14px 16px;">
          <div style="font-family:var(--font-m);font-size:0.6rem;color:var(--txt-lo);
                      letter-spacing:0.12em;margin-bottom:8px;">DATASET STATUS</div>
          <div class="nx-sidebar-stat" style="color:var(--cyan);">✓ {ds.shape[0]:,} rows</div>
          <div class="nx-sidebar-stat" style="color:var(--purple);">✓ {ds.shape[1]} columns</div>
          <div class="nx-sidebar-stat" style="color:var(--green);">
            {'✓ 0 missing' if ds.isnull().sum().sum()==0 else f'⚠ {ds.isnull().sum().sum()} missing'}</div>
        </div>
        """, unsafe_allow_html=True)

    if "model" in st.session_state:
        st.markdown(f"""
        <div style="background:var(--bg-raised);border:1px solid var(--green-d);
                    border-radius:2px;padding:14px 16px;margin-top:10px;">
          <div style="font-family:var(--font-m);font-size:0.6rem;color:var(--txt-lo);
                      letter-spacing:0.12em;margin-bottom:8px;">MODEL ACTIVE</div>
          <div class="nx-sidebar-stat" style="color:var(--green);">
            ⬡ {st.session_state.get('model_name','—')}</div>
          <div class="nx-sidebar-stat" style="color:var(--txt-md);">
            {st.session_state.get('problem','')}</div>
        </div>
        """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════════
st.markdown("""
<div style="display:flex;align-items:center;gap:18px;padding:22px 0 14px 0;
            border-bottom:1px solid #1c2a38;margin-bottom:28px;">
  <div style="font-family:'Syne',sans-serif;font-size:2.1rem;color:#00c8f0;
              font-weight:800;letter-spacing:-0.02em;line-height:1;">⬡ NEXUS</div>
  <div>
    <div style="font-family:'Space Mono',monospace;font-size:0.68rem;color:#344d61;
                letter-spacing:0.18em;text-transform:uppercase;">
      Automated Machine Learning &amp; Exploratory Data Analysis Platform</div>
    <div style="font-family:'DM Mono',monospace;font-size:0.82rem;color:#1c2a38;margin-top:3px;">
      Upload · Analyze · Engineer · Train · Compare · Export</div>
  </div>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
# FILE UPLOAD
# ══════════════════════════════════════════════════════════════════
uploaded_file = st.file_uploader(
    "UPLOAD DATASET", type=["csv","tsv","txt"],
    help="Supports CSV, TSV, pipe-delimited. Auto-detects encoding & separator."
)

if not uploaded_file:
    st.markdown("""
    <div style="background:var(--bg-surface);border:2px dashed var(--border);
                border-radius:2px;padding:64px;text-align:center;margin-top:16px;">
      <div style="font-size:2.8rem;margin-bottom:14px;opacity:0.4;">⬡</div>
      <div style="font-family:'Syne',sans-serif;font-size:1.1rem;color:var(--cyan);margin-bottom:8px;">
        No dataset loaded</div>
      <div style="font-family:'DM Mono',monospace;font-size:0.8rem;color:var(--txt-lo);">
        Upload a CSV / TSV file to start automated analysis</div>
    </div>""", unsafe_allow_html=True)
    st.stop()

df_raw, enc_used, sep_used = load_data(uploaded_file)
df_raw.columns = df_raw.columns.str.strip()
df_raw = try_parse_dates(df_raw)
col_types = detect_col_types(df_raw)

st.markdown(f"""
<div style="display:flex;align-items:center;gap:10px;background:#00f5a00a;
            border:1px solid var(--green-d);border-radius:2px;
            padding:11px 18px;margin-bottom:24px;">
  <span style="color:var(--green);font-size:1rem;">✓</span>
  <span style="font-family:'Space Mono',monospace;font-size:0.72rem;color:var(--green);">
    {uploaded_file.name} &nbsp;·&nbsp; {df_raw.shape[0]:,} rows &nbsp;·&nbsp;
    {df_raw.shape[1]} cols &nbsp;·&nbsp; enc:{enc_used} &nbsp;·&nbsp; sep:{repr(sep_used)}
  </span>
</div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
# PAGE 01 — OVERVIEW & EDA
# ══════════════════════════════════════════════════════════════════
if page == "01 · Overview & EDA":

    sec("01","Dataset Overview")
    c1,c2,c3,c4,c5 = st.columns(5)
    c1.metric("ROWS",          f"{df_raw.shape[0]:,}")
    c2.metric("COLUMNS",        df_raw.shape[1])
    c3.metric("MISSING CELLS",  f"{df_raw.isnull().sum().sum():,}")
    c4.metric("DUPLICATE ROWS", f"{df_raw.duplicated().sum():,}")
    c5.metric("MEMORY",         f"{df_raw.memory_usage(deep=True).sum()/1024:.1f} KB")

    sec("02","Column Type Map")
    badges = "".join(
        f'<span class="nx-badge {badge_css.get(t,"b-num")}">{c}</span>'
        for c,t in col_types.items()
    )
    st.markdown(f'<div style="line-height:2.6;">{badges}</div>', unsafe_allow_html=True)

    sec("03","Raw Data Preview")
    st.dataframe(df_raw.head(100), use_container_width=True)

    sec("04","Statistical Summary")
    num_cols = [c for c,t in col_types.items() if t=="numeric"]
    cat_cols = [c for c,t in col_types.items() if t in ("categorical","boolean")]
    tab1,tab2,tab3 = st.tabs(["  NUMERIC  ","  CATEGORICAL  ","  FULL DTYPES  "])
    with tab1:
        if num_cols: st.dataframe(df_raw[num_cols].describe().T.style.format("{:.3f}"), use_container_width=True)
        else: st.info("No numeric columns.")
    with tab2:
        if cat_cols:
            rows=[]
            for c in cat_cols:
                vc=df_raw[c].value_counts()
                rows.append({"Column":c,"Unique":df_raw[c].nunique(),
                             "Top Value":str(vc.index[0]) if len(vc) else "—",
                             "Top Freq":int(vc.iloc[0]) if len(vc) else 0,
                             "Missing %":f"{df_raw[c].isnull().mean()*100:.1f}%"})
            st.dataframe(pd.DataFrame(rows), use_container_width=True)
        else: st.info("No categorical columns.")
    with tab3:
        dtypes_df = pd.DataFrame({"Column":df_raw.columns,"Detected Type":[col_types[c] for c in df_raw.columns],
                                   "Pandas Dtype":df_raw.dtypes.values,"Non-Null":df_raw.count().values,
                                   "Null %":(df_raw.isnull().mean()*100).round(2).values})
        st.dataframe(dtypes_df, use_container_width=True)

    sec("05","Distributions")
    if num_cols:
        vc = st.selectbox("Numeric column", num_cols, key="eda_num")
        fig,axes = mfigs(1,3,w=16,h=4)
        sns.histplot(df_raw[vc].dropna(), kde=True, ax=axes[0], color=CYAN, alpha=0.7, edgecolor="none")
        sax(axes[0], title=f"Histogram · {vc}")
        sns.boxplot(y=df_raw[vc].dropna(), ax=axes[1], color=GREEN, width=0.4,
                    medianprops=dict(color="#fff",lw=2))
        sax(axes[1], title="Box Plot")
        sns.violinplot(y=df_raw[vc].dropna(), ax=axes[2], color=PRP, inner="quartile", linewidth=1)
        sax(axes[2], title="Violin Plot")
        plt.tight_layout(pad=2); st.pyplot(fig,use_container_width=True); plt.close()

    if cat_cols:
        cc = st.selectbox("Categorical column", cat_cols, key="eda_cat")
        vc2 = df_raw[cc].value_counts().head(20)
        fig,ax = mfig(10, max(4,len(vc2)*0.42))
        clrs = [CYAN if i==0 else "#1c2a38" for i in range(len(vc2))]
        bars = ax.barh(range(len(vc2)), vc2.values, color=clrs, edgecolor="none", height=0.62)
        ax.set_yticks(range(len(vc2)))
        ax.set_yticklabels(vc2.index.astype(str), fontsize=8, color=TXT)
        ax.invert_yaxis()
        for bar,val in zip(bars,vc2.values):
            ax.text(bar.get_width()+max(vc2.values)*0.01, bar.get_y()+bar.get_height()/2,
                    f"{val:,}", va="center", ha="left", color=TXT, fontsize=7)
        sax(ax, title=f"Value Counts · {cc}", xl="Count")
        plt.tight_layout(); st.pyplot(fig,use_container_width=True); plt.close()

    sec("06","Correlation Matrix")
    if len(num_cols) > 1:
        corr = df_raw[num_cols].corr()
        n = len(num_cols)
        fig,ax = plt.subplots(figsize=(min(max(n*1.1,8),18), min(max(n*0.9,5),14)), facecolor=BG)
        ax.set_facecolor(SURF)
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(corr, mask=mask, annot=n<=16, fmt=".2f",
                    cmap=sns.diverging_palette(220,20,as_cmap=True),
                    ax=ax, linewidths=0.5, linecolor=BG, center=0,
                    annot_kws={"size":7,"color":"#ddeeff"},
                    cbar_kws={"shrink":0.75})
        ax.tick_params(colors=TXT, labelsize=8)
        ax.set_title("Pearson Correlation — lower triangle", color="#ddeeff", fontsize=10,
                     pad=10, fontfamily="monospace")
        plt.tight_layout(); st.pyplot(fig,use_container_width=True); plt.close()

        pairs = corr.where(~mask).stack().reset_index()
        pairs.columns=["Feature A","Feature B","r"]
        pairs["|r|"]=pairs["r"].abs()
        top = pairs.sort_values("|r|",ascending=False).head(10).reset_index(drop=True)
        st.dataframe(top[["Feature A","Feature B","r"]].style.format({"r":"{:.4f}"}),
                     use_container_width=True)

    sec("07","Pairplot (top 5 numeric)")
    if len(num_cols) >= 2:
        pp_cols = num_cols[:5]
        if st.button("⬡  Generate Pairplot"):
            fig = sns.pairplot(df_raw[pp_cols].dropna(), diag_kind="kde",
                               plot_kws={"alpha":0.4,"color":CYAN,"s":12},
                               diag_kws={"color":CYAN,"fill":True})
            fig.fig.patch.set_facecolor(BG)
            for ax in fig.axes.flatten():
                if ax: ax.set_facecolor(SURF); ax.tick_params(colors=TXT,labelsize=7)
                if ax:
                    for sp in ax.spines.values(): sp.set_edgecolor(GRID)
            st.pyplot(fig.fig, use_container_width=True); plt.close()
    else:
        st.info("Need at least 2 numeric columns for pairplot.")

# ══════════════════════════════════════════════════════════════════
# PAGE 02 — MISSING VALUES
# ══════════════════════════════════════════════════════════════════
elif page == "02 · Missing Values":

    sec("01","Missing Value Analysis")
    miss_df = pd.DataFrame({
        "Column":        df_raw.columns,
        "Type":          [col_types[c] for c in df_raw.columns],
        "Missing Count": df_raw.isnull().sum().values,
        "Missing %":     (df_raw.isnull().mean()*100).round(2).values,
        "Unique Values": df_raw.nunique().values,
    }).sort_values("Missing %", ascending=False).reset_index(drop=True)
    st.dataframe(miss_df.style.format({"Missing %":"{:.2f}%"}), use_container_width=True)

    cols_miss = miss_df[miss_df["Missing Count"]>0]["Column"].tolist()

    if cols_miss:
        sec("02","Missing Value Map")
        n_s = min(500,len(df_raw))
        samp = df_raw[cols_miss].sample(n_s,random_state=42) if len(df_raw)>n_s else df_raw[cols_miss]
        from matplotlib.colors import ListedColormap
        fig,ax = plt.subplots(figsize=(14,max(3,len(cols_miss)*0.45)), facecolor=BG)
        ax.set_facecolor(SURF)
        sns.heatmap(samp.isnull().astype(int).T, ax=ax,
                    cmap=ListedColormap([SURF, RED]),
                    cbar=False, yticklabels=cols_miss, xticklabels=False)
        ax.tick_params(colors=TXT,labelsize=8)
        ax.set_title(f"Missing pattern — {n_s} row sample  ·  Red = Missing",
                     color="#ddeeff", fontsize=9, pad=8, fontfamily="monospace")
        plt.tight_layout(); st.pyplot(fig,use_container_width=True); plt.close()

        # Bar
        fig,ax = mfig(10, max(3,len(cols_miss)*0.42))
        vals = miss_df[miss_df["Missing Count"]>0].set_index("Column")["Missing %"]
        bclrs = [RED if v>50 else CYAN for v in vals.values]
        ax.barh(vals.index, vals.values, color=bclrs, edgecolor="none", height=0.55)
        ax.invert_yaxis()
        ax.axvline(50,color=RED,ls="--",lw=1,alpha=0.5,label="50% threshold")
        for i,(col,val) in enumerate(vals.items()):
            ax.text(val+0.5,i,f"{val:.1f}%",va="center",color=TXT,fontsize=7)
        sax(ax,title="Missing % per Column",xl="Missing %")
        ax.legend(fontsize=7,facecolor=SURF,labelcolor=TXT,edgecolor=GRID)
        plt.tight_layout(); st.pyplot(fig,use_container_width=True); plt.close()

        sec("03","Imputation Strategy")
        strat_opts = {
            "numeric":["median","mean","knn","mode","constant","ffill","bfill","drop_rows"],
            "categorical":["mode","constant","ffill","bfill","drop_rows"],
            "boolean":["mode","constant","drop_rows"],
            "datetime":["ffill","bfill","drop_rows"],
            "high_cardinality":["mode","constant","drop_rows"],
        }
        default_s = {"numeric":"median","categorical":"mode","boolean":"mode",
                     "datetime":"ffill","high_cardinality":"mode"}
        strategy_map = {}
        for col in cols_miss:
            ct = col_types[col]
            mp = miss_df[miss_df["Column"]==col]["Missing %"].values[0]
            opts = strat_opts.get(ct,["mode","constant","drop_rows"])
            dflt = "drop_rows" if mp>60 else default_s.get(ct,"mode")
            warn = "<span style='color:var(--red);font-size:0.65rem;'>⚠ &gt;60%</span>" if mp>60 else ""
            c1,c2,c3 = st.columns([3,1,2])
            with c1:
                st.markdown(f"""<div style="padding:8px 0;">
                  <div style="font-family:var(--font-b);font-size:0.8rem;color:var(--txt-hi);">{col}</div>
                  <div style="font-family:var(--font-b);font-size:0.72rem;color:var(--txt-lo);">{ct} · {mp}%
                    {"&nbsp;"+warn if warn else ""}</div></div>""", unsafe_allow_html=True)
            with c2:
                st.markdown(f'<div style="padding:12px 0;"><span class="nx-badge {badge_css.get(ct,"b-num")}">{ct}</span></div>',
                            unsafe_allow_html=True)
            with c3:
                strategy_map[col] = st.selectbox(f"s_{col}", opts,
                    index=opts.index(dflt) if dflt in opts else 0,
                    key=f"imp_{col}", label_visibility="collapsed")
            st.markdown("<hr style='border-color:#1c2a38;margin:3px 0;'>", unsafe_allow_html=True)

        if st.button("⬡  Apply Imputation"):
            df_imp = smart_impute(df_raw.copy(), strategy_map)
            st.session_state["df_clean"] = df_imp
            rem = df_imp.isnull().sum().sum()
            st.success(f"✓ Imputation applied · {rem} missing remain · {df_imp.shape[0]:,} rows kept")
            st.dataframe(df_imp.head(30), use_container_width=True)
            st.markdown(df_download_link(df_imp,"imputed_data.csv","⬡  Download Imputed CSV"),
                        unsafe_allow_html=True)
    else:
        st.success("No missing values detected — dataset is complete.")
        st.session_state["df_clean"] = df_raw.copy()

# ══════════════════════════════════════════════════════════════════
# PAGE 03 — OUTLIER DETECTION
# ══════════════════════════════════════════════════════════════════
elif page == "03 · Outlier Detection":

    sec("01","Outlier Detection")
    df_o = st.session_state.get("df_clean", df_raw.copy())
    num_cols = [c for c,t in col_types.items() if t=="numeric" and c in df_o.columns]

    if not num_cols:
        st.info("No numeric columns found for outlier detection.")
        st.stop()

    method = st.radio("Method", ["IQR (1.5×)","Z-Score (|z|>3)","Both"], horizontal=True)

    outlier_counts = {}
    for c in num_cols:
        col_data = df_o[c].dropna()
        n_iqr = n_z = 0
        q1,q3 = col_data.quantile(0.25), col_data.quantile(0.75)
        iqr = q3-q1
        iqr_mask = (col_data<q1-1.5*iqr)|(col_data>q3+1.5*iqr)
        n_iqr = int(iqr_mask.sum())
        z_mask = (np.abs((col_data-col_data.mean())/col_data.std())>3)
        n_z = int(z_mask.sum())
        if method=="IQR (1.5×)":       outlier_counts[c] = n_iqr
        elif method=="Z-Score (|z|>3)": outlier_counts[c] = n_z
        else:                           outlier_counts[c] = int((iqr_mask|z_mask).sum())

    out_df = pd.DataFrame({
        "Column": list(outlier_counts.keys()),
        "Outliers": list(outlier_counts.values()),
        "Outlier %": [round(v/len(df_o)*100,2) for v in outlier_counts.values()]
    }).sort_values("Outliers",ascending=False)

    sec("02","Outlier Summary")
    st.dataframe(out_df.style.format({"Outlier %":"{:.2f}%"}), use_container_width=True)

    total_out = out_df["Outliers"].sum()
    c1,c2,c3 = st.columns(3)
    c1.metric("TOTAL OUTLIERS",    f"{total_out:,}")
    c2.metric("COLUMNS AFFECTED",  f"{(out_df['Outliers']>0).sum()}")
    c3.metric("OUTLIER RATE",       f"{total_out/len(df_o)*100:.2f}%")

    sec("03","Box Plots — Outlier Visibility")
    sel_col = st.selectbox("Column to inspect", num_cols)
    fig,axes = mfigs(1,2,w=14,h=4)
    axes[0].hist(df_o[sel_col].dropna(), bins=40, color=CYAN, alpha=0.7, edgecolor="none")
    sax(axes[0], title=f"Distribution · {sel_col}")
    axes[1].boxplot(df_o[sel_col].dropna(), vert=True, patch_artist=True,
                    boxprops=dict(facecolor="#00c8f022",color=CYAN),
                    whiskerprops=dict(color=CYAN),
                    capprops=dict(color=CYAN),
                    medianprops=dict(color=GREEN,lw=2),
                    flierprops=dict(marker="o",color=RED,markersize=3,alpha=0.6))
    sax(axes[1], title=f"Box Plot · {sel_col}")
    plt.tight_layout(); st.pyplot(fig,use_container_width=True); plt.close()

    sec("04","Handle Outliers")
    handle_method = st.selectbox("Treatment method",
        ["Keep as-is","Cap at IQR bounds (Winsorize)","Remove outlier rows","Replace with median"])

    if st.button("⬡  Apply Outlier Treatment"):
        df_out = df_o.copy()
        affected_cols = [c for c,v in outlier_counts.items() if v>0]
        for c in affected_cols:
            q1,q3 = df_out[c].quantile(0.25), df_out[c].quantile(0.75)
            iqr = q3-q1
            lo,hi = q1-1.5*iqr, q3+1.5*iqr
            if handle_method == "Cap at IQR bounds (Winsorize)":
                df_out[c] = df_out[c].clip(lower=lo, upper=hi)
            elif handle_method == "Remove outlier rows":
                df_out = df_out[(df_out[c]>=lo)&(df_out[c]<=hi)]
            elif handle_method == "Replace with median":
                med = df_out[c].median()
                df_out.loc[(df_out[c]<lo)|(df_out[c]>hi), c] = med
        st.session_state["df_clean"] = df_out
        st.success(f"✓ Treatment applied — {df_out.shape[0]:,} rows remain")
        st.markdown(df_download_link(df_out,"outlier_treated.csv","⬡  Download Treated CSV"),
                    unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
# PAGE 04 — FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════════
elif page == "04 · Feature Engineering":

    sec("01","Feature Engineering Studio")
    df_fe = st.session_state.get("df_clean", df_raw.copy())
    num_cols = [c for c,t in col_types.items() if t=="numeric" and c in df_fe.columns]

    tab1,tab2,tab3,tab4 = st.tabs([
        "  INTERACTION FEATURES  ",
        "  POLYNOMIAL FEATURES   ",
        "  BINNING               ",
        "  PCA                   "
    ])

    with tab1:
        sec("","Interaction Features (A op B)")
        if len(num_cols) >= 2:
            c1,c2,c3 = st.columns(3)
            fa = c1.selectbox("Feature A", num_cols, key="fe_a")
            op = c2.selectbox("Operation", ["A + B","A - B","A × B","A / B","A² + B²"], key="fe_op")
            fb = c3.selectbox("Feature B", [c for c in num_cols if c!=fa], key="fe_b")
            new_name = st.text_input("New column name", value=f"{fa}_{op[:3]}_{fb}", key="fe_name")
            if st.button("⬡  Create Feature", key="fe_btn"):
                a,b = df_fe[fa], df_fe[fb]
                if   op=="A + B":      val = a+b
                elif op=="A - B":      val = a-b
                elif op=="A × B":      val = a*b
                elif op=="A / B":      val = a/(b.replace(0,np.nan))
                else:                  val = a**2+b**2
                df_fe[new_name] = val
                st.session_state["df_clean"] = df_fe
                st.success(f"✓ Feature '{new_name}' created. Shape: {df_fe.shape}")
        else:
            st.info("Need at least 2 numeric columns.")

    with tab2:
        sec("","Polynomial Features")
        if num_cols:
            poly_col = st.selectbox("Column", num_cols, key="poly_col")
            deg = st.slider("Max degree", 2, 4, 2, key="poly_deg")
            if st.button("⬡  Add Polynomial Terms", key="poly_btn"):
                for d in range(2, deg+1):
                    df_fe[f"{poly_col}_pow{d}"] = df_fe[poly_col]**d
                st.session_state["df_clean"] = df_fe
                st.success(f"✓ Added degree 2–{deg} for '{poly_col}'. Shape: {df_fe.shape}")

    with tab3:
        sec("","Binning / Discretization")
        if num_cols:
            bin_col = st.selectbox("Column to bin", num_cols, key="bin_col")
            n_bins  = st.slider("Number of bins", 2, 20, 5, key="bin_n")
            bin_strategy = st.radio("Strategy", ["Equal Width","Equal Frequency","Custom"], horizontal=True, key="bin_strat")
            bin_name = st.text_input("New column name", value=f"{bin_col}_bin", key="bin_name")
            if st.button("⬡  Apply Binning", key="bin_btn"):
                if bin_strategy == "Equal Width":
                    df_fe[bin_name] = pd.cut(df_fe[bin_col], bins=n_bins, labels=False)
                elif bin_strategy == "Equal Frequency":
                    df_fe[bin_name] = pd.qcut(df_fe[bin_col], q=n_bins, labels=False, duplicates="drop")
                else:
                    edges_str = st.text_input("Enter bin edges (comma-separated)", key="bin_edges")
                    if edges_str:
                        edges = [float(x.strip()) for x in edges_str.split(",")]
                        df_fe[bin_name] = pd.cut(df_fe[bin_col], bins=edges, labels=False)
                st.session_state["df_clean"] = df_fe
                st.success(f"✓ Binned '{bin_col}' → '{bin_name}'. Shape: {df_fe.shape}")

    with tab4:
        sec("","PCA — Dimensionality Reduction")
        if len(num_cols) >= 2:
            n_comp = st.slider("PCA components", 1, min(len(num_cols),10), 2, key="pca_n")
            if st.button("⬡  Apply PCA", key="pca_btn"):
                X_pca = df_fe[num_cols].fillna(df_fe[num_cols].median())
                X_scaled = StandardScaler().fit_transform(X_pca)
                pca = PCA(n_components=n_comp)
                comps = pca.fit_transform(X_scaled)
                for i in range(n_comp):
                    df_fe[f"PCA_{i+1}"] = comps[:,i]
                st.session_state["df_clean"] = df_fe
                ev = pca.explained_variance_ratio_
                st.success(f"✓ PCA applied. Total variance explained: {ev.sum()*100:.1f}%")
                fig,ax = mfig(8,4)
                ax.bar(range(1,n_comp+1), ev*100, color=CYAN, edgecolor="none", width=0.6)
                sax(ax, title="Explained Variance per Component",xl="Component",yl="Variance %")
                plt.tight_layout(); st.pyplot(fig,use_container_width=True); plt.close()

    sec("02","Current Dataset")
    st.dataframe(st.session_state.get("df_clean",df_fe).head(30), use_container_width=True)
    st.markdown(df_download_link(st.session_state.get("df_clean",df_fe),
                                  "engineered_data.csv","⬡  Download Engineered Dataset"),
                unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
# PAGE 05 — MODEL TRAINING
# ══════════════════════════════════════════════════════════════════
elif page == "05 · Model Training":

    sec("01","Configure Training")
    df_ml = st.session_state.get("df_clean", df_raw.copy())

    # Auto impute remaining
    if df_ml.isnull().sum().sum() > 0:
        st.warning("⚠ Auto-imputing remaining missing values (median/mode). Visit page 02 for custom strategy.")
        for c in df_ml.columns:
            if df_ml[c].isnull().any():
                if pd.api.types.is_numeric_dtype(df_ml[c]): df_ml[c]=df_ml[c].fillna(df_ml[c].median())
                else:
                    m=df_ml[c].mode(); df_ml[c]=df_ml[c].fillna(m[0] if len(m) else "MISSING")

    c1,c2 = st.columns(2)
    with c1: target = st.selectbox("TARGET COLUMN", df_ml.columns)
    with c2:
        y_tmp = df_ml[target].dropna()
        auto_prob = "Regression" if (pd.api.types.is_numeric_dtype(y_tmp) and y_tmp.nunique()>15) else "Classification"
        problem_type = st.radio("PROBLEM TYPE",["Classification","Regression"],
                                index=0 if auto_prob=="Classification" else 1, horizontal=True)

    if df_ml[target].isnull().any():
        df_ml=df_ml.dropna(subset=[target])
        st.info(f"Dropped rows with missing target → {df_ml.shape[0]:,} rows remain")

    c3,c4,c5 = st.columns(3)
    with c3: test_size  = st.slider("TEST SPLIT %",10,40,20)
    with c4: scaler_opt = st.selectbox("FEATURE SCALING",["None","Standard","MinMax","Robust"])
    with c5: run_cv     = st.checkbox("5-FOLD CROSS VALIDATION", value=True)

    feat_cols = [c for c in df_ml.columns if c!=target]
    hc_cols   = [c for c in feat_cols if col_types.get(c)=="high_cardinality"]
    if hc_cols:
        st.warning(f"High-cardinality columns excluded: {hc_cols}")
        feat_cols = [c for c in feat_cols if c not in hc_cols]

    sel_feats = st.multiselect("FEATURES",feat_cols,default=feat_cols)

    models_dict = MODELS_CLF if problem_type=="Classification" else MODELS_REG
    model_name  = st.selectbox("ALGORITHM", list(models_dict.keys()))

    st.markdown("")
    if st.button("⬡  Train Model"):
        if not sel_feats: st.error("Select at least one feature."); st.stop()

        dt = df_ml[sel_feats+[target]].copy().dropna(subset=[target])
        ctl = {c:col_types.get(c,"categorical") for c in dt.columns}
        enc_df, encs = encode_features(dt, ctl, target)
        X = enc_df[[c for c in enc_df.columns if c!=target]]
        y = enc_df[target]

        te = None
        if problem_type=="Classification" and y.dtype=="object":
            te=LabelEncoder(); y=pd.Series(te.fit_transform(y))

        Xtr,Xte,ytr,yte = train_test_split(X,y,test_size=test_size/100,random_state=42)
        Xtr_s, Xte_s = scale_features(Xtr, Xte, scaler_opt)

        mdl = models_dict[model_name]
        t0 = time.time()
        with st.spinner(f"Training {model_name}…"):
            mdl.fit(Xtr_s, ytr)
            preds = mdl.predict(Xte_s)
        train_time = time.time()-t0

        st.session_state.update({
            "model":mdl,"encoders":encs,"target_encoder":te,
            "feature_columns":list(X.columns),"original_features":sel_feats,
            "problem":problem_type,"col_types":ctl,"target":target,
            "model_name":model_name,"scaler_opt":scaler_opt,
            "X_full":X,"y_full":y,"train_time":train_time,
        })

        sec("02","Performance Metrics")
        folds = safe_cv_folds(y, problem_type)

        if problem_type=="Classification":
            acc  = accuracy_score(yte,preds)
            prec = precision_score(yte,preds,average="weighted",zero_division=0)
            rec  = recall_score(yte,preds,average="weighted",zero_division=0)
            f1   = f1_score(yte,preds,average="weighted",zero_division=0)

            cvs = None
            if run_cv:
                if folds < 2:
                    st.warning("CV skipped — classes too small.")
                else:
                    if folds < 5: st.warning(f"CV folds reduced to {folds}.")
                    cvs = cross_val_score(mdl,Xtr_s,ytr,cv=folds,scoring="accuracy")

            cv_html = ""
            if cvs is not None:
                cv_html = f"""
                <div class="nx-score-row"><span class="nx-score-label">CV Mean Accuracy</span>
                <span class="nx-score-val">{cvs.mean()*100:.2f}%</span></div>
                <div class="nx-score-row"><span class="nx-score-label">CV Std Dev</span>
                <span class="nx-score-val">±{cvs.std()*100:.2f}%</span></div>"""

            color = "g" if acc>0.85 else ("a" if acc>0.7 else "r")
            st.markdown(f"""
            <div class="nx-score-box">
              <div class="nx-score-row"><span class="nx-score-label">Accuracy</span>
                <span class="nx-score-val {color}">{acc*100:.2f}%</span></div>
              <div class="nx-score-row"><span class="nx-score-label">Precision (weighted)</span>
                <span class="nx-score-val">{prec:.4f}</span></div>
              <div class="nx-score-row"><span class="nx-score-label">Recall (weighted)</span>
                <span class="nx-score-val">{rec:.4f}</span></div>
              <div class="nx-score-row"><span class="nx-score-label">F1-Score (weighted)</span>
                <span class="nx-score-val">{f1:.4f}</span></div>
              {cv_html}
              <div class="nx-score-row"><span class="nx-score-label">Train Time</span>
                <span class="nx-score-val">{train_time:.2f}s</span></div>
              <div class="nx-score-row"><span class="nx-score-label">Test Samples</span>
                <span class="nx-score-val">{len(yte):,}</span></div>
            </div>""", unsafe_allow_html=True)

            fig,axes = mfigs(1,2,w=14,h=5)
            cm=confusion_matrix(yte,preds)
            sns.heatmap(cm,annot=True,fmt="d",cmap="Blues",ax=axes[0],linewidths=1,
                        linecolor=BG,annot_kws={"size":11,"color":"#ddeeff","weight":"bold"})
            axes[0].set_xlabel("Predicted",color=TXT); axes[0].set_ylabel("Actual",color=TXT)
            sax(axes[0],title="Confusion Matrix")
            if cvs is not None:
                axes[1].bar(range(1,len(cvs)+1),cvs*100,
                            color=[PAL[i%len(PAL)] for i in range(len(cvs))],alpha=0.82,width=0.55)
                axes[1].axhline(cvs.mean()*100,color=GREEN,lw=2,ls="--",
                                label=f"Mean {cvs.mean()*100:.1f}%")
                axes[1].set_ylim(0,110)
                axes[1].legend(fontsize=8,facecolor=SURF,labelcolor=TXT)
                sax(axes[1],title="CV Fold Scores",xl="Fold",yl="Accuracy %")
            plt.tight_layout(); st.pyplot(fig,use_container_width=True); plt.close()
            with st.expander("Full Classification Report"):
                st.code(classification_report(yte,preds),language=None)

        else:
            r2   = r2_score(yte,preds)
            rmse = np.sqrt(mean_squared_error(yte,preds))
            mae  = mean_absolute_error(yte,preds)
            mape = np.mean(np.abs((np.array(yte)-preds)/np.where(np.array(yte)==0,1,yte)))*100

            cvs = None
            if run_cv:
                if folds < 2: st.warning("CV skipped — dataset too small.")
                else: cvs = cross_val_score(mdl,Xtr_s,ytr,cv=folds,scoring="r2")

            rv = "g" if r2>0.8 else ("a" if r2>0.6 else "r")
            cv_html = f"""<div class="nx-score-row"><span class="nx-score-label">CV Mean R²</span>
                <span class="nx-score-val">{cvs.mean():.4f} ± {cvs.std():.4f}</span></div>""" if cvs is not None else ""
            st.markdown(f"""
            <div class="nx-score-box">
              <div class="nx-score-row"><span class="nx-score-label">R² Score</span>
                <span class="nx-score-val {rv}">{r2:.4f}</span></div>
              <div class="nx-score-row"><span class="nx-score-label">RMSE</span>
                <span class="nx-score-val">{rmse:.4f}</span></div>
              <div class="nx-score-row"><span class="nx-score-label">MAE</span>
                <span class="nx-score-val">{mae:.4f}</span></div>
              <div class="nx-score-row"><span class="nx-score-label">MAPE</span>
                <span class="nx-score-val">{mape:.2f}%</span></div>
              {cv_html}
              <div class="nx-score-row"><span class="nx-score-label">Train Time</span>
                <span class="nx-score-val">{train_time:.2f}s</span></div>
            </div>""", unsafe_allow_html=True)

            fig,axes = mfigs(1,2,w=14,h=5)
            axes[0].scatter(yte,preds,alpha=0.4,color=CYAN,s=14,edgecolors="none")
            mn=min(float(yte.min()),float(preds.min())); mx=max(float(yte.max()),float(preds.max()))
            axes[0].plot([mn,mx],[mn,mx],color=RED,lw=2,ls="--",label="Perfect fit")
            axes[0].legend(fontsize=7,facecolor=SURF,labelcolor=TXT)
            sax(axes[0],title="Actual vs Predicted",xl="Actual",yl="Predicted")
            res=np.array(yte)-preds
            axes[1].scatter(preds,res,alpha=0.35,color=GREEN,s=12,edgecolors="none")
            axes[1].axhline(0,color=RED,lw=2,ls="--")
            sax(axes[1],title="Residual Plot",xl="Predicted",yl="Residual")
            plt.tight_layout(); st.pyplot(fig,use_container_width=True); plt.close()

        sec("03","Feature Importance")
        if hasattr(mdl,"feature_importances_"):
            fi=pd.DataFrame({"f":X.columns,"i":mdl.feature_importances_}).sort_values("i",ascending=False).head(20)
            for _,row in fi.iterrows():
                bw=(row["i"]/fi["i"].max())*100
                st.markdown(f"""<div class="nx-fi-row">
                    <div class="nx-fi-name">{row['f']}</div>
                    <div class="nx-fi-track"><div class="nx-fi-fill" style="width:{bw:.1f}%;"></div></div>
                    <div class="nx-fi-pct">{row['i']*100:.1f}%</div></div>""", unsafe_allow_html=True)
        elif hasattr(mdl,"coef_"):
            coef = np.abs(mdl.coef_.flatten()[:len(X.columns)])
            fi=pd.DataFrame({"f":X.columns[:len(coef)],"i":coef}).sort_values("i",ascending=False).head(20)
            for _,row in fi.iterrows():
                bw=(row["i"]/fi["i"].max())*100 if fi["i"].max()>0 else 0
                st.markdown(f"""<div class="nx-fi-row">
                    <div class="nx-fi-name">{row['f']}</div>
                    <div class="nx-fi-track"><div class="nx-fi-fill" style="width:{bw:.1f}%;"></div></div>
                    <div class="nx-fi-pct">{row['i']:.3f}</div></div>""", unsafe_allow_html=True)
        else:
            st.info("Feature importance not available for this model.")

# ══════════════════════════════════════════════════════════════════
# PAGE 06 — MODEL COMPARISON
# ══════════════════════════════════════════════════════════════════
elif page == "06 · Model Comparison":

    sec("01","Model Comparison — All Algorithms")

    if "X_full" not in st.session_state:
        st.warning("Train a model in page 05 first to set up the dataset.")
        st.stop()

    X       = st.session_state["X_full"]
    y       = st.session_state["y_full"]
    problem = st.session_state["problem"]
    scaler_opt = st.session_state.get("scaler_opt","None")
    folds   = safe_cv_folds(y, problem)

    models_dict = MODELS_CLF if problem=="Classification" else MODELS_REG
    scoring     = "accuracy" if problem=="Classification" else "r2"
    score_lbl   = "Accuracy" if problem=="Classification" else "R²"

    test_size = st.slider("Test split % for comparison",10,40,20,key="cmp_split")
    Xtr,Xte,ytr,yte = train_test_split(X,y,test_size=test_size/100,random_state=42)
    Xtr_s,Xte_s = scale_features(Xtr,Xte,scaler_opt)

    if st.button("⬡  Run Full Comparison"):
        results = []
        prog = st.progress(0)
        n = len(models_dict)
        for idx,(mname, mdl) in enumerate(models_dict.items()):
            with st.spinner(f"Training {mname}…"):
                t0=time.time()
                try:
                    mdl.fit(Xtr_s,ytr)
                    preds=mdl.predict(Xte_s)
                    t1=time.time()-t0
                    if problem=="Classification":
                        sc  = accuracy_score(yte,preds)
                        f1v = f1_score(yte,preds,average="weighted",zero_division=0)
                        if folds>=2:
                            cv=cross_val_score(mdl,Xtr_s,ytr,cv=folds,scoring=scoring)
                            cv_m,cv_s=cv.mean(),cv.std()
                        else: cv_m=cv_s=None
                        results.append({"Model":mname,score_lbl:round(sc,4),
                                        "F1-Score":round(f1v,4),
                                        "CV Mean":round(cv_m,4) if cv_m else "N/A",
                                        "CV Std":round(cv_s,4) if cv_s else "N/A",
                                        "Train Time (s)":round(t1,3)})
                    else:
                        r2=r2_score(yte,preds)
                        rmse=np.sqrt(mean_squared_error(yte,preds))
                        mae=mean_absolute_error(yte,preds)
                        if folds>=2:
                            cv=cross_val_score(mdl,Xtr_s,ytr,cv=folds,scoring=scoring)
                            cv_m,cv_s=cv.mean(),cv.std()
                        else: cv_m=cv_s=None
                        results.append({"Model":mname,"R²":round(r2,4),
                                        "RMSE":round(rmse,4),"MAE":round(mae,4),
                                        "CV Mean":round(cv_m,4) if cv_m else "N/A",
                                        "CV Std":round(cv_s,4) if cv_s else "N/A",
                                        "Train Time (s)":round(t1,3)})
                except Exception as e:
                    results.append({"Model":mname,score_lbl:f"Error: {e}","Train Time (s)":"—"})
            prog.progress((idx+1)/n)

        res_df = pd.DataFrame(results)
        sort_col = score_lbl if score_lbl in res_df.columns else "R²"

        try:
            res_df_sorted = res_df.sort_values(sort_col,ascending=False).reset_index(drop=True)
        except: res_df_sorted = res_df.reset_index(drop=True)

        st.session_state["comparison_df"] = res_df_sorted

        sec("02","Leaderboard")
        for i,row in res_df_sorted.iterrows():
            try:
                sc_val = float(row[sort_col])
                best_val = float(res_df_sorted[sort_col].max())
                bw = (sc_val/best_val)*100 if best_val>0 else 0
                sc_clr = GREEN if i==0 else (CYAN if i<3 else TXT)
                sc_disp = f"{sc_val*100:.2f}%" if problem=="Classification" else f"{sc_val:.4f}"
            except: bw=0; sc_clr=TXT; sc_disp=str(row.get(sort_col,"—"))
            rank_clr = ["#ffd700","#c0c0c0","#cd7f32"]
            rank_sym = ["①","②","③"]
            rk = rank_sym[i] if i<3 else f"#{i+1}"
            rk_c = rank_clr[i] if i<3 else TXT
            st.markdown(f"""<div class="nx-compare-row">
              <div class="nx-compare-rank" style="color:{rk_c};">{rk}</div>
              <div class="nx-compare-name">{row['Model']}</div>
              <div class="nx-compare-bar-track"><div class="nx-compare-bar-fill" style="width:{bw:.1f}%;"></div></div>
              <div class="nx-compare-score" style="color:{sc_clr};">{sc_disp}</div>
              <div style="font-family:var(--font-m);font-size:0.65rem;color:var(--txt-lo);
                          width:70px;text-align:right;">{row.get('Train Time (s)','—')}s</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.dataframe(res_df_sorted, use_container_width=True)

        # Score bar chart
        sec("03","Visual Comparison")
        try:
            fig,ax = mfig(12,5)
            vals=[float(v) for v in res_df_sorted[sort_col] if str(v) not in ["N/A","—"] and not str(v).startswith("Error")]
            names=[r for r,v in zip(res_df_sorted["Model"],res_df_sorted[sort_col]) if str(v) not in ["N/A","—"] and not str(v).startswith("Error")]
            clrs=[GREEN if i==0 else (CYAN if i<3 else "#1c2a38") for i in range(len(vals))]
            bars=ax.bar(range(len(vals)),vals,color=clrs,edgecolor="none",width=0.65)
            ax.set_xticks(range(len(names))); ax.set_xticklabels(names,rotation=30,ha="right",fontsize=8,color=TXT)
            for bar,val in zip(bars,vals):
                disp=f"{val*100:.1f}%" if problem=="Classification" else f"{val:.3f}"
                ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.002,
                        disp, ha="center", va="bottom", fontsize=7, color=TXT)
            sax(ax,title=f"Model Comparison — {score_lbl}",yl=score_lbl)
            plt.tight_layout(); st.pyplot(fig,use_container_width=True); plt.close()
        except: st.info("Chart unavailable — check for error results above.")

        st.markdown(df_download_link(res_df_sorted,"model_comparison.csv","⬡  Download Comparison CSV"),
                    unsafe_allow_html=True)

    elif "comparison_df" in st.session_state:
        st.info("Showing previous comparison results. Click the button to re-run.")
        st.dataframe(st.session_state["comparison_df"], use_container_width=True)

# ══════════════════════════════════════════════════════════════════
# PAGE 07 — LIVE PREDICTION
# ══════════════════════════════════════════════════════════════════
elif page == "07 · Live Prediction":

    sec("01","Live Prediction Interface")

    if "model" not in st.session_state:
        st.warning("No model found. Train a model in page 05 first.")
        st.stop()

    st.markdown(f"""
    <div class="nx-model-badge">
      <div class="nx-badge-item"><div class="nx-badge-item-label">Model</div>
        <div class="nx-badge-item-val" style="color:var(--cyan);">{st.session_state['model_name']}</div></div>
      <div class="nx-badge-item"><div class="nx-badge-item-label">Task</div>
        <div class="nx-badge-item-val">{st.session_state['problem']}</div></div>
      <div class="nx-badge-item"><div class="nx-badge-item-label">Target</div>
        <div class="nx-badge-item-val">{st.session_state['target']}</div></div>
      <div class="nx-badge-item"><div class="nx-badge-item-label">Features</div>
        <div class="nx-badge-item-val">{len(st.session_state['original_features'])}</div></div>
      <div class="nx-badge-item"><div class="nx-badge-item-label">Scaling</div>
        <div class="nx-badge-item-val">{st.session_state.get('scaler_opt','None')}</div></div>
    </div>""", unsafe_allow_html=True)

    orig  = st.session_state["original_features"]
    encs  = st.session_state["encoders"]
    ctp   = st.session_state["col_types"]
    dfref = st.session_state.get("df_clean", df_raw)
    user_input = {}

    sec("02","Enter Feature Values")
    for grp in [orig[i:i+3] for i in range(0,len(orig),3)]:
        cols = st.columns(len(grp))
        for i,col in enumerate(grp):
            ct = ctp.get(col,"categorical")
            with cols[i]:
                if ct=="numeric":
                    mn=float(dfref[col].min()) if col in dfref.columns else 0.0
                    mx=float(dfref[col].max()) if col in dfref.columns else 100.0
                    mv=float(dfref[col].mean()) if col in dfref.columns else 0.0
                    user_input[col]=st.number_input(col,min_value=mn,max_value=mx,value=mv,key=f"p_{col}")
                elif col in encs:
                    user_input[col]=st.selectbox(col,list(encs[col].classes_),key=f"p_{col}")
                else:
                    user_input[col]=st.text_input(col,key=f"p_{col}")

    st.markdown("")
    if st.button("⬡  Run Prediction"):
        try:
            idf = pd.DataFrame([user_input])
            for col in idf.columns:
                if col in encs:
                    le=encs[col]; v=str(idf[col].iloc[0])
                    if v not in le.classes_:
                        st.warning(f"Unknown '{v}' for '{col}' → using fallback.")
                        v=le.classes_[0]
                    idf[col]=le.transform([v])
            idf = idf.reindex(columns=st.session_state["feature_columns"],fill_value=0).astype(float)

            # Apply scaling if used during training
            scaler_opt = st.session_state.get("scaler_opt","None")
            if scaler_opt != "None":
                from sklearn.preprocessing import StandardScaler,MinMaxScaler,RobustScaler
                sc_map={"Standard":StandardScaler(),"MinMax":MinMaxScaler(),"Robust":RobustScaler()}
                X_full = st.session_state["X_full"].values
                sc = sc_map[scaler_opt]; sc.fit(X_full)
                idf_vals = sc.transform(idf.values)
                idf = pd.DataFrame(idf_vals, columns=idf.columns)

            pred=st.session_state["model"].predict(idf)
            result=pred[0]
            if st.session_state.get("target_encoder") and st.session_state["problem"]=="Classification":
                result=st.session_state["target_encoder"].inverse_transform([int(result)])[0]

            st.markdown(f"""
            <div class="nx-result-card">
                <div class="nx-result-label">⬡ &nbsp; Prediction Result</div>
                <div class="nx-result-val">{result}</div>
            </div>""", unsafe_allow_html=True)

            if st.session_state["problem"]=="Classification" and hasattr(st.session_state["model"],"predict_proba"):
                proba=st.session_state["model"].predict_proba(idf)[0]
                te_=st.session_state.get("target_encoder")
                classes=te_.inverse_transform(range(len(proba))) if te_ else st.session_state["model"].classes_
                sec("03","Class Probabilities")
                for cls,prob in sorted(zip(classes,proba),key=lambda x:-x[1]):
                    bc=GREEN if prob==max(proba) else CYAN
                    st.markdown(f"""<div class="nx-prob-row">
                        <div class="nx-prob-label">{cls}</div>
                        <div class="nx-prob-track"><div class="nx-prob-fill" style="width:{prob*100:.1f}%;background:{bc};box-shadow:0 0 6px {bc}55;"></div></div>
                        <div class="nx-prob-pct" style="color:{bc};">{prob*100:.1f}%</div>
                    </div>""", unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Prediction failed: {e}"); st.exception(e)

# ══════════════════════════════════════════════════════════════════
# PAGE 08 — EXPORT & REPORT
# ══════════════════════════════════════════════════════════════════
elif page == "08 · Export & Report":

    sec("01","Export & Project Report")
    df_exp = st.session_state.get("df_clean", df_raw)
    num_cols = [c for c,t in col_types.items() if t=="numeric"]

    tab1,tab2,tab3 = st.tabs(["  DATA EXPORTS  ","  MODEL REPORT  ","  FULL EDA REPORT  "])

    with tab1:
        sec("","Download Processed Data")
        st.markdown(df_download_link(df_raw,      "raw_data.csv",       "⬡  Download Raw Data"),        unsafe_allow_html=True)
        st.markdown(df_download_link(df_exp,       "clean_data.csv",     "⬡  Download Clean Data"),      unsafe_allow_html=True)
        if "comparison_df" in st.session_state:
            st.markdown(df_download_link(st.session_state["comparison_df"],
                                          "model_comparison.csv","⬡  Download Model Comparison"), unsafe_allow_html=True)

        sec("","Dataset Statistics Download")
        if num_cols:
            stats_df = df_exp[num_cols].describe().T.reset_index()
            stats_df.columns = ["Feature"] + list(stats_df.columns[1:])
            st.markdown(df_download_link(stats_df,"statistics.csv","⬡  Download Statistics"),
                        unsafe_allow_html=True)

    with tab2:
        sec("","Model Performance Report")
        if "model" not in st.session_state:
            st.info("No model trained yet. Go to page 05.")
        else:
            mdl_info = {
                "Model Name":    st.session_state.get("model_name","—"),
                "Problem Type":  st.session_state.get("problem","—"),
                "Target Column": st.session_state.get("target","—"),
                "Features Used": len(st.session_state.get("original_features",[])),
                "Feature Scaling":st.session_state.get("scaler_opt","None"),
                "Training Time": f"{st.session_state.get('train_time',0):.3f}s",
            }
            for k,v in mdl_info.items():
                st.markdown(f"""<div class="nx-score-row">
                    <span class="nx-score-label">{k}</span>
                    <span class="nx-score-val">{v}</span>
                </div>""", unsafe_allow_html=True)

            report_df = pd.DataFrame([mdl_info])
            st.markdown(df_download_link(report_df,"model_report.csv","⬡  Download Model Report"),
                        unsafe_allow_html=True)

    with tab3:
        sec("","Full EDA Report")
        if st.button("⬡  Generate EDA Report"):
            lines = []
            lines.append("# NEXUS AutoML Studio — EDA Report")
            lines.append(f"\n## Dataset: {uploaded_file.name}")
            lines.append(f"- Rows: {df_raw.shape[0]:,}")
            lines.append(f"- Columns: {df_raw.shape[1]}")
            lines.append(f"- Missing cells: {df_raw.isnull().sum().sum():,}")
            lines.append(f"- Duplicate rows: {df_raw.duplicated().sum():,}")
            lines.append(f"- Memory: {df_raw.memory_usage(deep=True).sum()/1024:.1f} KB")
            lines.append("\n## Column Types")
            for c,t in col_types.items():
                lines.append(f"- **{c}**: {t}")
            # ── tabulate-free markdown table helper ──
            def df_to_md(df):
                df = df.copy().reset_index(drop=True)
                cols = list(df.columns)
                header = "| " + " | ".join(str(c) for c in cols) + " |"
                sep    = "| " + " | ".join("---" for _ in cols) + " |"
                rows   = [
                    "| " + " | ".join(
                        str(round(v, 4)) if isinstance(v, float) else str(v)
                        for v in row
                    ) + " |"
                    for _, row in df.iterrows()
                ]
                return "\n".join([header, sep] + rows)

            lines.append("\n## Numeric Statistics")
            if num_cols:
                stats_tbl = df_raw[num_cols].describe().T.reset_index().rename(columns={"index":"Feature"})
                lines.append(df_to_md(stats_tbl))

            lines.append("\n## Missing Values")
            miss_df = pd.DataFrame({
                "Column":       df_raw.columns,
                "Missing Count":df_raw.isnull().sum().values,
                "Missing %":    (df_raw.isnull().mean()*100).round(2).values
            })
            miss_filtered = miss_df[miss_df["Missing Count"] > 0]
            lines.append(df_to_md(miss_filtered) if len(miss_filtered) > 0 else "_No missing values._")

            if "model" in st.session_state:
                lines.append("\n## Model Summary")
                lines.append(f"- Algorithm: {st.session_state.get('model_name','—')}")
                lines.append(f"- Task: {st.session_state.get('problem','—')}")
                lines.append(f"- Target: {st.session_state.get('target','—')}")
                lines.append(f"- Features: {len(st.session_state.get('original_features',[]))}")
                lines.append(f"- Scaling: {st.session_state.get('scaler_opt','None')}")
            if "comparison_df" in st.session_state:
                lines.append("\n## Model Comparison Leaderboard")
                lines.append(df_to_md(st.session_state["comparison_df"]))

            report_md = "\n".join(lines)
            b64 = base64.b64encode(report_md.encode()).decode()
            st.markdown(f"""
            <div class="nx-download-btn">
              <a href="data:text/markdown;base64,{b64}" download="nexus_eda_report.md">
                ⬡  Download Full EDA Report (.md)
              </a>
            </div>""", unsafe_allow_html=True)
            st.markdown("**Report Preview:**")
            st.markdown(report_md)