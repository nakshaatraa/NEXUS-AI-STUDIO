# Changelog

All notable changes to Nexus AI Studio will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [3.0.0] — 2026-05-08

### 🚀 Added
- **8-page modular pipeline**: Overview & EDA → Missing Values → Outlier Detection → Feature Engineering → Model Training → Model Comparison → Live Prediction → Export & Report
- **20+ ML algorithms**: 9 classifiers + 11 regressors from scikit-learn
- **Smart CSV parser**: Auto-detects encoding (`utf-8`, `latin1`, `iso-8859-1`, `cp1252`) and separators (`,`, `\t`, `;`, `|`)
- **7 imputation strategies**: median, mean, mode, KNN, ffill, bfill, constant with per-column control
- **Outlier detection**: IQR, Z-Score, and combined methods with 4 treatment options
- **Feature engineering studio**: Interaction features, polynomial terms, binning, PCA
- **Model comparison leaderboard**: Ranked visualization with gold/silver/bronze indicators
- **Live prediction interface**: Dynamic form with class probability visualization
- **Full EDA report generation**: Downloadable Markdown reports
- **Premium dark theme**: Custom glassmorphism UI with Syne/Space Mono/DM Mono typography
- **Cross-validation**: Auto-adjusted folds based on dataset characteristics
- **Feature scaling**: Standard, MinMax, and Robust scaler options
- **Data exports**: Raw, cleaned, engineered datasets + model comparison CSVs

### 🎨 Design
- Deep navy color system (`#07090d` base) with cyan/green/amber/red accents
- Custom Streamlit components: metric cards, score boxes, feature bars, probability tracks
- Hover glow effects, smooth transitions, and responsive layouts
- Hidden Streamlit branding for a clean professional look

---

## [Unreleased]

_Nothing yet — contributions welcome!_
