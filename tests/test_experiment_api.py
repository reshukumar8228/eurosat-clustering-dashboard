"""
Unit & Integration Tests for Experiment Bench Endpoint Functionality
"""

import pytest
from pydantic import ValidationError
from backend.api import run_experiment, ExperimentParams


def test_experiment_default_params():
    params = ExperimentParams()
    data = run_experiment(params)

    assert "pca" in data
    assert "kmeans" in data
    assert "tsne" in data
    assert "comparison" in data

    assert data["pca"]["components_used"] == 220
    assert data["pca"]["axis_x"] == 1
    assert data["pca"]["axis_y"] == 2
    assert len(data["pca"]["scatter_points"]) > 0

    assert data["kmeans"]["k"] == 10
    assert data["kmeans"]["is_precomputed"] is True
    assert len(data["kmeans"]["cluster_sizes"]) == 10

    assert data["tsne"]["perplexity"] == 30
    assert len(data["tsne"]["scatter_points"]) > 0


def test_experiment_precomputed_k():
    params = ExperimentParams(kmeans_k=5)
    data = run_experiment(params)

    assert data["kmeans"]["k"] == 5
    assert data["kmeans"]["is_precomputed"] is True
    assert len(data["kmeans"]["cluster_sizes"]) == 5


def test_experiment_live_k():
    # k=13 is not in precomputed set [2..12, 15, 20]
    params = ExperimentParams(kmeans_k=13, pca_components=30)
    data = run_experiment(params)

    assert data["kmeans"]["k"] == 13
    assert data["kmeans"]["is_precomputed"] is False
    assert len(data["kmeans"]["cluster_sizes"]) == 13
    assert data["kmeans"]["silhouette"] != 0.0


def test_experiment_pca_axes():
    params = ExperimentParams(pca_axis_x=2, pca_axis_y=4)
    data = run_experiment(params)

    assert data["pca"]["axis_x"] == 2
    assert data["pca"]["axis_y"] == 4
    assert len(data["pca"]["scatter_points"]) > 0


def test_experiment_tsne_perplexity():
    params = ExperimentParams(tsne_perplexity=50)
    data = run_experiment(params)

    assert data["tsne"]["perplexity"] == 50


def test_experiment_validation_error():
    # kmeans_k must be <= 30
    with pytest.raises(ValidationError):
        ExperimentParams(kmeans_k=50)

    # pca_components must be >= 2
    with pytest.raises(ValidationError):
        ExperimentParams(pca_components=1)
