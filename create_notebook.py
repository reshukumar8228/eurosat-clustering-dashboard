"""
Generate the analysis_notebook.ipynb file.
"""

import json
import logging

logging.basicConfig(level=logging.INFO, format="%(message)s")

cells = [
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# 🛰️ Unsupervised Land-Type Grouping of Satellite Image Patches Using Clustering\n",
            "**EuroSAT Sentinel-2 Satellite Dataset**\n",
            "\n",
            "### Project Objectives\n",
            "1. **Preprocessing:** Prepare EuroSAT satellite patches with ground truth labels hidden.\n",
            "2. **Dimensionality Reduction:** Apply PCA and multi-perplexity t-SNE to examine natural groupings and compare retained variance.\n",
            "3. **Clustering & Interpretability:** Cluster PCA-reduced features with K-Means ($k \\in [2, 20]$), assess cluster purity and alignment with true land cover, and interpret ethical implications."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "import os\n",
            "import json\n",
            "import numpy as np\n",
            "import pandas as pd\n",
            "import matplotlib.pyplot as plt\n",
            "import seaborn as sns\n",
            "from PIL import Image\n",
            "import joblib\n",
            "\n",
            "sns.set_theme(style=\"whitegrid\")\n",
            "OUTPUT_DIR = \"pipeline_outputs\"\n",
            "print(\"Libraries successfully imported.\")"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 1. Dataset Loading & Class Distribution\n",
            "The EuroSAT dataset comprises 27,000 Sentinel-2 satellite image patches (64×64 pixels) across 10 land cover classes. During modeling, labels are strictly hidden to ensure unsupervised validity."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "metadata_df = pd.read_csv(os.path.join(OUTPUT_DIR, \"metadata.csv\"))\n",
            "print(f\"Total Patches: {len(metadata_df):,}\")\n",
            "print(\"Class Distribution:\")\n",
            "print(metadata_df['true_label'].value_counts())"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 2. Principal Component Analysis (PCA)\n",
            "We analyze the cumulative variance spectrum and project high-dimensional patches (12,288 D) into a lower-dimensional orthogonal subspace."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "exp_var = np.load(os.path.join(OUTPUT_DIR, \"pca_explained_variance.npy\"))\n",
            "cum_var = np.cumsum(exp_var) * 100\n",
            "\n",
            "plt.figure(figsize=(9, 4.5))\n",
            "plt.plot(range(1, len(cum_var) + 1), cum_var, color=\"#2563EB\", lw=2.5)\n",
            "plt.axhline(90, color=\"#EF4444\", linestyle=\"--\", label=\"90% Variance Threshold\")\n",
            "plt.title(\"PCA Cumulative Explained Variance Spectrum\", fontsize=13, fontweight='bold')\n",
            "plt.xlabel(\"Principal Component Index\")\n",
            "plt.ylabel(\"Cumulative Explained Variance (%)\")\n",
            "plt.legend()\n",
            "plt.tight_layout()\n",
            "plt.show()\n",
            "\n",
            "print(f\"Variance explained by top 2 components: {cum_var[1]:.2f}%\")\n",
            "print(f\"Total components fitted: {len(exp_var)} ({cum_var[-1]:.2f}% variance)\")"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "### 2D PCA Projection"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "pca_2d = np.load(os.path.join(OUTPUT_DIR, \"pca_2d.npy\"))\n",
            "sub_idx = np.load(os.path.join(OUTPUT_DIR, \"tsne_subset_indices.npy\"))\n",
            "\n",
            "sub_pca = pca_2d[sub_idx]\n",
            "sub_labels = metadata_df.iloc[sub_idx][\"true_label\"].values\n",
            "\n",
            "plt.figure(figsize=(10, 7))\n",
            "sns.scatterplot(x=sub_pca[:, 0], y=sub_pca[:, 1], hue=sub_labels, s=20, alpha=0.7)\n",
            "plt.title(\"PCA 2D Space Colored by True Land-Use Category (Post-Hoc)\", fontsize=13, fontweight='bold')\n",
            "plt.xlabel(\"PC1 (65.30% Var)\")\n",
            "plt.ylabel(\"PC2\")\n",
            "plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left')\n",
            "plt.tight_layout()\n",
            "plt.show()"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 3. t-SNE Non-Linear Manifold Learning & Perplexity Sensitivity\n",
            "We inspect embeddings across perplexities [5, 15, 30, 50, 100] to evaluate local vs global preservation."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "perps = [5, 15, 30, 50, 100]\n",
            "fig, axes = plt.subplots(1, 5, figsize=(22, 4.2))\n",
            "\n",
            "for i, p in enumerate(perps):\n",
            "    emb = np.load(os.path.join(OUTPUT_DIR, f\"tsne_2d_perp{p}.npy\"))\n",
            "    axes[i].scatter(emb[:, 0], emb[:, 1], c=metadata_df.iloc[sub_idx][\"true_label_id\"], cmap=\"tab10\", s=8, alpha=0.6)\n",
            "    axes[i].set_title(f\"Perplexity = {p}\", fontweight='bold')\n",
            "    axes[i].axis(\"off\")\n",
            "\n",
            "plt.suptitle(\"t-SNE Embeddings Across Perplexity Values\", fontsize=14, fontweight='bold')\n",
            "plt.tight_layout()\n",
            "plt.show()"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 4. K-Means Clustering on PCA-Reduced Features\n",
            "Evaluation across $k \\in [2, 20]$ via Inertia, Silhouette Score, and Davies-Bouldin Index."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "with open(os.path.join(OUTPUT_DIR, \"kmeans_metrics.json\")) as f:\n",
            "    km_metrics = json.load(f)\n",
            "\n",
            "k_vals = [int(k) for k in km_metrics.keys()]\n",
            "inertias = [km_metrics[str(k)][\"inertia\"] for k in k_vals]\n",
            "silhouettes = [km_metrics[str(k)][\"silhouette\"] for k in k_vals]\n",
            "\n",
            "fig, ax1 = plt.subplots(figsize=(9, 4.5))\n",
            "color = 'tab:blue'\n",
            "ax1.set_xlabel('Number of Clusters (k)')\n",
            "ax1.set_ylabel('Inertia (Elbow)', color=color)\n",
            "ax1.plot(k_vals, inertias, marker='o', color=color, lw=2)\n",
            "ax1.tick_params(axis='y', labelcolor=color)\n",
            "\n",
            "ax2 = ax1.twinx()\n",
            "color = 'tab:green'\n",
            "ax2.set_ylabel('Silhouette Score', color=color)\n",
            "ax2.plot(k_vals, silhouettes, marker='s', color=color, lw=2, linestyle='--')\n",
            "ax2.tick_params(axis='y', labelcolor=color)\n",
            "\n",
            "plt.title(\"K-Means Clustering: Elbow & Silhouette Curves\", fontsize=13, fontweight='bold')\n",
            "plt.tight_layout()\n",
            "plt.show()"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "### K=10 Contingency Matrix (Post-Hoc vs True Labels)"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "cm_df = pd.read_csv(os.path.join(OUTPUT_DIR, \"kmeans_k10_contingency.csv\"), index_col=0)\n",
            "plt.figure(figsize=(10, 7))\n",
            "sns.heatmap(cm_df, annot=True, fmt=\"d\", cmap=\"Blues\", cbar=True)\n",
            "plt.title(\"K-Means (k=10) vs Ground Truth Contingency Matrix\", fontsize=12, fontweight='bold')\n",
            "plt.xlabel(\"True Land-Cover Class\")\n",
            "plt.ylabel(\"Cluster Index\")\n",
            "plt.xticks(rotation=45, ha='right')\n",
            "plt.tight_layout()\n",
            "plt.show()\n",
            "\n",
            "k10_res = km_metrics[\"10\"]\n",
            "print(f\"Adjusted Rand Index (ARI): {k10_res['adjusted_rand_index']:.4f}\")\n",
            "print(f\"Normalized Mutual Information (NMI): {k10_res['normalized_mutual_info']:.4f}\")\n",
            "print(f\"Overall Archetype Purity: {k10_res['overall_purity']*100:.2f}%\")"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 5. Inference Engine & Nearest Neighbor Lookup\n",
            "Test classification of an unlabelled satellite patch."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "from classifier import EuroSATClassifier\n",
            "classifier = joblib.load(os.path.join(OUTPUT_DIR, \"classifier.pkl\"))\n",
            "\n",
            "# Test sample\n",
            "test_sample_path = metadata_df.iloc[100][\"filepath\"]\n",
            "prediction = classifier.predict(test_sample_path)\n",
            "\n",
            "print(\"Prediction Result:\")\n",
            "print(f\"  Target Image: {metadata_df.iloc[100]['filename']}\")\n",
            "print(f\"  Ground Truth: {metadata_df.iloc[100]['true_label']}\")\n",
            "print(f\"  Detected Cluster: Cluster {prediction['predicted_cluster']}\")\n",
            "print(f\"  Mapped Land-Type: {prediction['majority_label']} (Purity: {prediction['confidence_purity']*100:.1f}%)\")\n",
            "print(f\"  Top Nearest Neighbor: {prediction['nearest_neighbors'][0]['filename']} ({prediction['nearest_neighbors'][0]['true_label']})\")"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 6. Conclusions & Ethical Considerations\n",
            "- **Clustering Archetypes:** Land cover groups naturally into broad spectral regimes (water bodies, dense canopy, sparse vegetation/crop, and high-reflectance urban/industrial).\n",
            "- **Geographic Bias:** EuroSAT captures European geography; models may exhibit misclassification when applied to arid, tropical, or peri-urban geographies without domain adaptation.\n",
            "- **Responsible Use:** Unsupervised grouping provides automated stratification and screening for geospatial analysts, but should not replace ground truth verification in critical ecological or legal domains."
        ]
    }
]

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.13"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 5
}

with open("d:\\Projects\\EuroSAT_Project\\analysis_notebook.ipynb", "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=2)

logging.info("Created analysis_notebook.ipynb successfully.")
