"""
Unit Tests for Dashboard Data Dependencies and Visual Assets
"""

import os
import json
import numpy as np
import pandas as pd
import pytest

OUTPUT_DIR = "pipeline_outputs"
FIG_DIR = "figures"

def test_dashboard_arrays_loadable():
    pca_2d = np.load(os.path.join(OUTPUT_DIR, "pca_2d.npy"))
    assert pca_2d.ndim == 2
    assert pca_2d.shape[1] == 2

    sub_indices = np.load(os.path.join(OUTPUT_DIR, "tsne_subset_indices.npy"))
    assert len(sub_indices) == 5000

    labels_k10 = np.load(os.path.join(OUTPUT_DIR, "kmeans_labels_k10.npy"))
    assert len(labels_k10) == 27000
    assert len(np.unique(labels_k10)) == 10

def test_cluster_grids_exist():
    grids_dir = os.path.join(OUTPUT_DIR, "cluster_grids")
    assert os.path.exists(grids_dir)
    for c_id in range(10):
        grid_file = os.path.join(grids_dir, f"cluster_{c_id}_grid.png")
        cent_file = os.path.join(grids_dir, f"centroid_cluster_{c_id}.png")
        assert os.path.exists(grid_file), f"Missing exemplar grid: {grid_file}"
        assert os.path.exists(cent_file), f"Missing centroid image: {cent_file}"

def test_contingency_matrix_valid():
    cm_path = os.path.join(OUTPUT_DIR, "kmeans_k10_contingency.csv")
    cm = pd.read_csv(cm_path, index_col=0)
    assert cm.shape == (10, 10)
    assert cm.values.sum() == 27000

def test_static_figures_exist():
    expected_figs = [
        "class_distribution.png",
        "pca_variance.png",
        "pca_2d_scatter.png",
        "tsne_comparison.png",
        "elbow_curve.png",
        "silhouette_scores.png",
        "confusion_heatmap.png"
    ]
    for fig in expected_figs:
        path = os.path.join(FIG_DIR, fig)
        assert os.path.exists(path), f"Missing figure: {fig}"
        assert os.path.getsize(path) > 1000, f"Figure too small: {fig}"
