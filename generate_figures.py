"""
Generate publication-quality static figures for EuroSAT project reports and presentations.
Outputs to figures/ directory (300 DPI PNGs).
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

OUTPUT_DIR = "pipeline_outputs"
FIG_DIR = "figures"
os.makedirs(FIG_DIR, exist_ok=True)

sns.set_theme(style="whitegrid", font_scale=1.1)

CLASS_COLORS = {
    "AnnualCrop": "#2ca02c",
    "Forest": "#006400",
    "HerbaceousVegetation": "#8c564b",
    "Highway": "#7f7f7f",
    "Industrial": "#e377c2",
    "Pasture": "#bcbd22",
    "PermanentCrop": "#ff7f0e",
    "Residential": "#d62728",
    "River": "#17becf",
    "SeaLake": "#1f77b4"
}

def generate_class_distribution():
    meta = pd.read_csv(os.path.join(OUTPUT_DIR, "metadata.csv"))
    counts = meta["true_label"].value_counts().sort_index()

    plt.figure(figsize=(10, 5))
    bars = plt.bar(counts.index, counts.values, color=[CLASS_COLORS[c] for c in counts.index], edgecolor='black', alpha=0.85)
    plt.title("EuroSAT Dataset: Image Patch Count per Land-Use Class (Total: 27,000)", fontsize=13, fontweight='bold', pad=12)
    plt.xlabel("Land-Use Category", fontsize=11)
    plt.ylabel("Number of Patches", fontsize=11)
    plt.xticks(rotation=40, ha='right')
    for bar in bars:
        h = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., h + 30, f"{int(h):,}", ha='center', va='bottom', fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "class_distribution.png"), dpi=300)
    plt.close()
    print("Saved class_distribution.png")

def generate_pca_variance():
    exp_var = np.load(os.path.join(OUTPUT_DIR, "pca_explained_variance.npy"))
    cum_var = np.cumsum(exp_var) * 100

    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(cum_var) + 1), cum_var, color="#2563EB", lw=2.5, label="Cumulative Variance")
    plt.axhline(90, color="#EF4444", linestyle="--", label="90% Variance Threshold")
    plt.title("PCA Explained Variance Spectrum across Principal Components", fontsize=13, fontweight='bold', pad=12)
    plt.xlabel("Principal Component Index", fontsize=11)
    plt.ylabel("Cumulative Explained Variance (%)", fontsize=11)
    plt.legend(frameon=True)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "pca_variance.png"), dpi=300)
    plt.close()
    print("Saved pca_variance.png")

def generate_pca_2d_scatter():
    pca_2d = np.load(os.path.join(OUTPUT_DIR, "pca_2d.npy"))
    meta = pd.read_csv(os.path.join(OUTPUT_DIR, "metadata.csv"))
    sub_idx = np.load(os.path.join(OUTPUT_DIR, "tsne_subset_indices.npy"))

    sub_pca = pca_2d[sub_idx]
    sub_meta = meta.iloc[sub_idx]

    plt.figure(figsize=(10, 8))
    for cls in sorted(meta["true_label"].unique()):
        mask = (sub_meta["true_label"] == cls).values
        plt.scatter(sub_pca[mask, 0], sub_pca[mask, 1], label=cls, color=CLASS_COLORS[cls], alpha=0.6, s=18)
    plt.title("PCA 2D Projection of EuroSAT Satellite Patches (Labels Hidden During Fit)", fontsize=13, fontweight='bold', pad=12)
    plt.xlabel("Principal Component 1 (65.30% Var)", fontsize=11)
    plt.ylabel("Principal Component 2", fontsize=11)
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', frameon=True, title="True Class (Post-Hoc)")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "pca_2d_scatter.png"), dpi=300)
    plt.close()
    print("Saved pca_2d_scatter.png")

def generate_tsne_comparison():
    meta = pd.read_csv(os.path.join(OUTPUT_DIR, "metadata.csv"))
    sub_idx = np.load(os.path.join(OUTPUT_DIR, "tsne_subset_indices.npy"))
    sub_meta = meta.iloc[sub_idx]

    perps = [5, 15, 30, 50, 100]
    fig, axes = plt.subplots(1, 5, figsize=(22, 4.5), sharex=False, sharey=False)

    for i, p in enumerate(perps):
        emb = np.load(os.path.join(OUTPUT_DIR, f"tsne_2d_perp{p}.npy"))
        ax = axes[i]
        for cls in sorted(meta["true_label"].unique()):
            mask = (sub_meta["true_label"] == cls).values
            ax.scatter(emb[mask, 0], emb[mask, 1], color=CLASS_COLORS[cls], s=8, alpha=0.6)
        ax.set_title(f"Perplexity = {p}", fontsize=11, fontweight='bold')
        ax.set_xticks([])
        ax.set_yticks([])

    plt.suptitle("t-SNE Parameter Sensitivity Across Perplexities (5 to 100)", fontsize=14, fontweight='bold', y=1.03)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "tsne_comparison.png"), dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved tsne_comparison.png")

def generate_clustering_curves():
    with open(os.path.join(OUTPUT_DIR, "kmeans_metrics.json"), "r") as f:
        metrics = json.load(f)

    k_list = [int(k) for k in metrics.keys()]
    inertias = [metrics[str(k)]["inertia"] for k in k_list]
    silhouettes = [metrics[str(k)]["silhouette"] for k in k_list]

    # Elbow
    plt.figure(figsize=(8, 4.5))
    plt.plot(k_list, inertias, marker='o', color="#2563EB", lw=2)
    plt.title("K-Means Elbow Method (Within-Cluster Sum of Squares)", fontsize=12, fontweight='bold')
    plt.xlabel("Number of Clusters (k)", fontsize=10)
    plt.ylabel("Inertia", fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "elbow_curve.png"), dpi=300)
    plt.close()

    # Silhouette
    plt.figure(figsize=(8, 4.5))
    plt.plot(k_list, silhouettes, marker='s', color="#10B981", lw=2)
    plt.title("Silhouette Coefficient vs Number of Clusters (k)", fontsize=12, fontweight='bold')
    plt.xlabel("Number of Clusters (k)", fontsize=10)
    plt.ylabel("Silhouette Score", fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "silhouette_scores.png"), dpi=300)
    plt.close()
    print("Saved elbow_curve.png and silhouette_scores.png")

def generate_confusion_heatmap():
    cm = pd.read_csv(os.path.join(OUTPUT_DIR, "kmeans_k10_contingency.csv"), index_col=0)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=True, linewidths=0.5)
    plt.title("K-Means (k=10) vs True Land-Cover Class Contingency Heatmap", fontsize=12, fontweight='bold', pad=12)
    plt.xlabel("Ground Truth Land Cover Class", fontsize=10)
    plt.ylabel("Unsupervised Cluster Assignment", fontsize=10)
    plt.xticks(rotation=40, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "confusion_heatmap.png"), dpi=300)
    plt.close()
    print("Saved confusion_heatmap.png")

if __name__ == "__main__":
    generate_class_distribution()
    generate_pca_variance()
    generate_pca_2d_scatter()
    generate_tsne_comparison()
    generate_clustering_curves()
    generate_confusion_heatmap()
    print("All static figures generated successfully.")
