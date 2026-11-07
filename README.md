# 🛰️ Unsupervised Land-Type Grouping of Satellite Image Patches Using Clustering

[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Dataset](https://img.shields.io/badge/dataset-EuroSAT%20(Sentinel--2)-orange.svg)](https://github.com/phelber/eurosat)
[![Modern React Dashboard](https://img.shields.io/badge/dashboard-React%20%2B%20FastAPI-blue.svg)](http://localhost:8000)

---

## 📖 Abstract

Remote sensing imagery from satellite constellations provides crucial data for environmental monitoring, urban development, and disaster management. However, manual annotation of vast satellite archives is labor-intensive and error-prone. This project presents a rigorous, reproducible unsupervised learning framework for grouping 27,000 multi-spectral and RGB Sentinel-2 satellite image patches from the **EuroSAT** benchmark into natural land-cover categories without utilizing ground truth labels during feature modeling.

We evaluate linear dimensionality reduction via **Principal Component Analysis (PCA)**, non-linear manifold projection via **t-Distributed Stochastic Neighbor Embedding (t-SNE)** across a spectrum of perplexities ($p \in [5, 100]$), and **Uniform Manifold Approximation and Projection (UMAP)**. Unsupervised **K-Means clustering** ($k \in [2, 20]$) identifies broad environmental archetypes. Finally, an interactive **React + FastAPI Vision UI dashboard** enables dynamic exploration, cluster inspections, and real-time image upload inference with nearest-neighbor retrieval.

---

## 📑 Table of Contents

- [Overview & Objectives](#-overview--objectives)
- [Project Architecture](#-project-architecture)
- [Mathematical Methodology](#-mathematical-methodology)
- [Key Empirical Findings](#-key-empirical-findings)
- [Installation & Quickstart](#-installation--quickstart)
- [Interactive Dashboard Guide](#-interactive-dashboard-guide)
- [Limitations & Ethical Considerations](#-limitations--ethical-considerations)
- [Unit Testing & Verification](#-unit-testing--verification)
- [Citation & References](#-citation--references)

---

## 🎯 Overview & Objectives

| Objective | Description | Status |
|---|---|:---:|
| **(A) Data Preparation** | Scaled 27,000 EuroSAT 64×64 RGB patches (12,288 flattened features) via `StandardScaler`; ground-truth annotations hidden during modeling. | ✅ Complete |
| **(B) Dimensionality Reduction** | Fitted Randomized PCA capturing >90% variance; multi-perplexity t-SNE ($p \in \{5, 15, 30, 50, 100\}$); UMAP comparison. | ✅ Complete |
| **(C) Unsupervised Grouping** | K-Means clustering across $k \in [2, 20]$; evaluated via Inertia, Silhouette Score, Davies-Bouldin, Calinski-Harabasz, ARI, and NMI. | ✅ Complete |
| **(D) Interactive Dashboard** | Modern Vision UI dashboard in React 19 + Tailwind CSS + FastAPI with interactive Plotly scatterplots, exemplar grids, and live upload inference engine. | ✅ Complete |

---

## 🗂️ Project Architecture

```
EuroSAT_Project/
├── Data/
│   ├── EuroSAT/                     # 27,000 RGB JPG patches (10 land cover classes)
│   └── EuroSATallBands/             # 13-band Sentinel-2 TIF files
├── config.yaml                      # Central pipeline hyperparameters
├── requirements.txt                 # Dependencies
├── pytest.ini                       # Pytest testpath configuration
├── pipeline.py                      # End-to-end unsupervised pipeline (Stages 1–6)
├── classifier.py                    # Inference engine & nearest-neighbor retriever
├── backend/
│   └── api.py                       # High-performance FastAPI service & static server
├── frontend/
│   ├── src/                         # React 19 components & Vision UI dashboard views
│   └── dist/                        # Optimized production frontend bundle
├── generate_figures.py              # Publication-quality static figure generator
├── create_notebook.py               # Notebook generator script
├── analysis_notebook.ipynb          # Self-contained Jupyter research notebook
├── start.bat                        # Modern React + FastAPI 1-click Windows launcher
├── start.sh                         # Linux / macOS automated launcher
├── pipeline_outputs/                # Pre-computed arrays, models, and metrics
│   ├── features_scaled.npy          # Standardized feature matrix (27,000 × 12,288)
│   ├── metadata.csv                 # Dataset manifest with post-hoc evaluation labels
│   ├── scaler.pkl                   # Fitted StandardScaler
│   ├── pca_model.pkl                # Fitted PCA model (220 components)
│   ├── pca_reduced_95.npy           # Projected features (27,000 × 220)
│   ├── pca_2d.npy                   # 2D PCA projection (27,000 × 2)
│   ├── pca_explained_variance.npy   # Variance ratios
│   ├── pca_loadings_top10.json      # Spatial pixel loadings for top 10 PCs
│   ├── tsne_2d_perp*.npy            # t-SNE projections for p = 5, 15, 30, 50, 100
│   ├── tsne_params.json             # t-SNE KL divergence and configuration
│   ├── kmeans_labels_k*.npy         # Cluster labels for k in 2..20
│   ├── kmeans_metrics.json          # Inertia, Silhouette, DB, CH, ARI, NMI
│   ├── kmeans_k10_contingency.csv   # Cluster vs True label contingency table
│   ├── classifier.pkl               # Serialized EuroSATClassifier instance
│   ├── cluster_label_mapping.json   # Majority label and purity mappings
│   ├── cluster_grids/               # 4×4 exemplar grids & centroid images
│   ├── spectral_pca_2d.npy          # Multi-spectral 13-band 2D projection
│   ├── spectral_features.csv        # Sentinel-2 band statistics
│   └── umap_2d.npy                  # UMAP 2D manifold coordinates
├── figures/                         # 300 DPI publication static figures
│   ├── class_distribution.png
│   ├── pca_variance.png
│   ├── pca_2d_scatter.png
│   ├── tsne_comparison.png
│   ├── elbow_curve.png
│   ├── silhouette_scores.png
│   └── confusion_heatmap.png
└── tests/                           # Pytest test suite
    ├── test_pipeline.py
    └── test_dashboard_data.py
```

---

## 📐 Mathematical Methodology

### 1. Principal Component Analysis (PCA)
Given standardized centered matrix $\mathbf{X} \in \mathbb{R}^{N \times D}$ where $D = 12,288$:
$$\max_{\mathbf{w}} \mathbf{w}^T \mathbf{X}^T \mathbf{X} \mathbf{w} \quad \text{s.t.} \quad \|\mathbf{w}\|_2 = 1$$
Solved via randomized singular value decomposition (SVD):
$$\mathbf{X} = \mathbf{U} \mathbf{\Sigma} \mathbf{V}^T$$
Top 220 components account for $>90\%$ of total empirical variance, with the first two components accounting for **65.30%**.

### 2. t-Distributed Stochastic Neighbor Embedding (t-SNE)
High-dimensional joint probabilities:
$$p_{j|i} = \frac{\exp(-\|\mathbf{x}_i - \mathbf{x}_j\|^2 / 2\sigma_i^2)}{\sum_{k \neq i} \exp(-\|\mathbf{x}_i - \mathbf{x}_k\|^2 / 2\sigma_i^2)}, \quad p_{ij} = \frac{p_{j|i} + p_{i|j}}{2N}$$
Low-dimensional Student-t kernel representation:
$$q_{ij} = \frac{(1 + \|\mathbf{y}_i - \mathbf{y}_j\|^2)^{-1}}{\sum_k \sum_{l \neq k} (1 + \|\mathbf{y}_k - \mathbf{y}_l\|^2)^{-1}}$$
Cost function minimized via gradient descent:
$$C = \mathrm{KL}(P \parallel Q) = \sum_i \sum_{j \neq i} p_{ij} \log \frac{p_{ij}}{q_{ij}}$$

### 3. K-Means Clustering Objective
Minimizes within-cluster sum of squares (inertia):
$$J = \sum_{k=1}^K \sum_{\mathbf{x} \in S_k} \|\mathbf{x} - \boldsymbol{\mu}_k\|^2$$

### 4. Post-Hoc Evaluation Metrics
- **Adjusted Rand Index (ARI):**
  $$\mathrm{ARI} = \frac{\sum_{ij} \binom{n_{ij}}{2} - \left[ \sum_i \binom{a_i}{2} \sum_j \binom{b_j}{2} \right] / \binom{n}{2}}{\frac{1}{2}\left[\sum_i \binom{a_i}{2} + \sum_j \binom{b_j}{2}\right] - \left[ \sum_i \binom{a_i}{2} \sum_j \binom{b_j}{2} \right] / \binom{n}{2}}$$
- **Normalized Mutual Information (NMI):**
  $$\mathrm{NMI}(U, V) = \frac{2 \cdot I(U; V)}{H(U) + H(V)}$$

---

## 📊 Key Empirical Findings

1. **Spectral Coherence in Low Dimensions:** The first two principal components capture **65.30%** of total variance, reflecting overall scene luminance and red-vs-green vegetative contrast.
2. **Natural Grouping Regime ($k=4$ to $k=6$):** The Elbow curve and Calinski-Harabasz index indicate natural separation into 4 major macro-classes:
   - **Open Water bodies** (`SeaLake`, deep rivers)
   - **Dense Vegetative Cover** (`Forest`, `Pasture`)
   - **Agricultural / Cultivated land** (`AnnualCrop`, `PermanentCrop`)
   - **Built-up Human Infrastructure** (`Residential`, `Industrial`, `Highway`)
3. **Fine-Grained Class Conflation:** Distinguishing `AnnualCrop` from `PermanentCrop` purely from raw RGB pixels without temporal multi-date phenology proves challenging in unsupervised feature spaces.
4. **Multi-Spectral Red-Edge Leverage:** Analyzing 13-band Sentinel-2 TIFs revealed that Red-Edge (B5–B7) and Short-Wave Infrared (SWIR B11, B12) provide significantly clearer separation between vegetation sub-types than RGB channels alone.

---

## 🚀 Installation & Quickstart

### Prerequisites
- Python 3.10+ (tested on Python 3.13)
- Windows, Linux, or macOS

### 1-Click Launch (Zero-Setup for Anyone Cloning This Repo)

Anyone who clones this repository can run the entire project with a single command:

**Windows:**
Double-click `start.bat` or execute in terminal:
```cmd
start.bat
```

**Linux / macOS:**
Execute in terminal:
```bash
chmod +x start.sh && ./start.sh
```

#### What `start.bat` and `start.sh` handle automatically:
1. **Python Detection:** Verifies Python 3.10+ is available in PATH.
2. **Virtual Environment Creation:** Creates an isolated `.venv` environment if dependencies are not already satisfied.
3. **Smart Dependency Check:** Installs missing requirements once, and **skips re-installation on subsequent launches** for near-instant boot.
4. **Model Verification:** Checks precomputed PCA, t-SNE, and K-Means models in `pipeline_outputs/` (auto-runs `pipeline.py` if absent).
5. **Pre-Built Frontend Serving:** Serves the modern React + Tailwind CSS dashboard from `frontend/dist/` (zero Node.js or npm requirement for end users).
6. **Port Collision Safeguard:** Automatically detects and frees port 8000 if an old instance is running.
7. **Synchronized Browser Launch:** Probes backend health and automatically opens **http://localhost:8000** in the default browser the instant the server is ready.

---

## 🖥️ Interactive Dashboard Guide

The dashboard is structured into 7 functional pages accessible from the sidebar:

1. **🏠 Overview & Dataset Explorer:** High-level project metrics, interactive class distribution bar chart, class breakdown table, and 8-patch preview gallery.
2. **📊 PCA Variance & Loadings:** Scree plot, cumulative variance slider, interactive 2D PCA scatter (smooth 5k or full 27k points), and spatial 64×64 pixel importance activation heatmaps.
3. **🔮 t-SNE Parameter Sensitivity:** Slider to toggle between perplexities 5, 15, 30, 50, and 100 with live KL divergence indicators and side-by-side comparative grid.
4. **🎯 K-Means Clustering Deep-Dive:** 4 evaluation metric curves ($k=2$ to $20$), $k=10$ confusion contingency matrix, per-cluster purity bar chart, and 4×4 exemplar patches closest to cluster centroids.
5. **🔍 Image Upload & Classification:** Drag-and-drop satellite patch uploader or EuroSAT preset selector. Detects predicted cluster, mapped land cover category, purity confidence, centroid distance bar chart, top-5 nearest neighbor historical patches, and highlights position on 2D PCA manifold with a star marker.
6. **📈 Comparative Analysis & Spectral:** Method comparison matrix (PCA vs t-SNE vs UMAP), interactive UMAP 2D scatter, and 13-band Sentinel-2 multi-spectral signature profiles.
7. **📝 Research Report & Governance:** Formatted study report, ethical bias documentation, and download buttons for report markdown, metrics JSON, and contingency CSV.

---

## ⚖️ Limitations & Ethical Considerations

- **European Geographic Bias:** EuroSAT exclusively covers European territory. Applying models directly to tropical, equatorial, or hyper-arid terrain will suffer performance degradation due to differing soil composition and vegetation seasonality.
- **Spatial Resolution Restrictions:** At 10m Ground Sample Distance (GSD), each 64×64 patch spans 640m × 640m. Mixed-pixel phenomena frequently occur where waterways or roadways bisect agricultural or residential zones.
- **t-SNE Non-Metric Distortion:** Euclidean distances between isolated clusters in t-SNE projections do not correlate with physical geographic distance; visual cluster densities should not be mistaken for variance magnitude.
- **Responsible Environmental AI:** Automated land classification should act as screening and decision support for human environmental experts, not as autonomous inputs for punitive zoning or taxation.

---

## 🧪 Unit Testing & Verification

Comprehensive unit test suites verify data integrity, array shapes, and classifier inference:
```bash
python -m pytest tests/ -v
```
**Results:** `10 passed in 3.97s` ✅

---

## 📚 References & Acknowledgments

- Helber, P., Bischke, B., Dengel, A., & Borth, D. (2019). *EuroSAT: A Novel Dataset and Deep Learning Benchmark for Land Use and Land Cover Classification.* IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing.
- Copernicus Open Access Hub / European Space Agency (ESA) Sentinel-2 mission.
