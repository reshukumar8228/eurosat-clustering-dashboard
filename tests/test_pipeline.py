"""
Unit Tests for EuroSAT Unsupervised Pipeline & Classifier
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from PIL import Image
import pytest
from classifier import EuroSATClassifier

OUTPUT_DIR = "pipeline_outputs"

def test_pipeline_outputs_exist():
    required_files = [
        "features_scaled.npy",
        "metadata.csv",
        "scaler.pkl",
        "pca_model.pkl",
        "pca_reduced_95.npy",
        "pca_2d.npy",
        "pca_explained_variance.npy",
        "pca_loadings_top10.json",
        "tsne_subset_indices.npy",
        "tsne_params.json",
        "kmeans_labels_k10.npy",
        "kmeans_metrics.json",
        "kmeans_k10_contingency.csv",
        "classifier.pkl",
        "cluster_label_mapping.json"
    ]
    for fname in required_files:
        path = os.path.join(OUTPUT_DIR, fname)
        assert os.path.exists(path), f"Missing required file: {fname}"
        assert os.path.getsize(path) > 0, f"File is empty: {fname}"

def test_metadata_integrity():
    meta = pd.read_csv(os.path.join(OUTPUT_DIR, "metadata.csv"))
    assert len(meta) == 27000
    assert set(["index", "filename", "filepath", "true_label", "true_label_id"]).issubset(meta.columns)
    assert meta["true_label"].nunique() == 10

def test_pca_shapes_and_variance():
    pca_2d = np.load(os.path.join(OUTPUT_DIR, "pca_2d.npy"))
    pca_reduced = np.load(os.path.join(OUTPUT_DIR, "pca_reduced_95.npy"))
    exp_var = np.load(os.path.join(OUTPUT_DIR, "pca_explained_variance.npy"))

    assert pca_2d.shape == (27000, 2)
    assert pca_reduced.shape[0] == 27000
    assert pca_reduced.shape[1] >= 50
    assert len(exp_var) == pca_reduced.shape[1]
    assert np.all(exp_var > 0)
    assert np.isclose(np.sum(exp_var), np.cumsum(exp_var)[-1])

def test_tsne_outputs():
    perps = [5, 15, 30, 50, 100]
    for p in perps:
        path = os.path.join(OUTPUT_DIR, f"tsne_2d_perp{p}.npy")
        assert os.path.exists(path)
        arr = np.load(path)
        assert arr.shape == (5000, 2)

def test_kmeans_metrics():
    with open(os.path.join(OUTPUT_DIR, "kmeans_metrics.json")) as f:
        metrics = json.load(f)
    assert "10" in metrics
    k10 = metrics["10"]
    assert "inertia" in k10
    assert "silhouette" in k10
    assert "adjusted_rand_index" in k10
    assert "normalized_mutual_info" in k10
    assert "overall_purity" in k10
    assert 0.0 <= k10["adjusted_rand_index"] <= 1.0

def test_classifier_inference():
    clf_path = os.path.join(OUTPUT_DIR, "classifier.pkl")
    clf = joblib.load(clf_path)
    assert isinstance(clf, EuroSATClassifier)

    # Test dummy 64x64 synthetic image
    synthetic_img = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
    res = clf.predict(synthetic_img)

    assert "predicted_cluster" in res
    assert 0 <= res["predicted_cluster"] < 10
    assert "majority_label" in res
    assert "confidence_purity" in res
    assert len(res["nearest_neighbors"]) == 5
    assert len(res["pca_2d"]) == 2
    assert "image_stats" in res
