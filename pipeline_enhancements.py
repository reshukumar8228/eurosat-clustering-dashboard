"""
EuroSAT Pipeline Enhancements (Stages 5 & 6)
Modularized inference classifier engine, exemplar grid creation, UMAP & spectral analysis.
"""

import os
import glob
import time
import json
import cv2
import joblib
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from classifier import EuroSATClassifier
import yaml

try:
    import umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False

try:
    import rasterio
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False


def load_config(config_path="config.yaml"):
    """Load configuration yaml file with default fallback."""
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            return yaml.safe_load(f)
    return {
        "data": {"rgb_dir": "Data/EuroSAT", "allbands_dir": "Data/EuroSATallBands", "image_size": 64},
        "pca": {"variance_threshold": 0.95, "n_components": 220, "random_state": 42},
        "tsne": {"perplexities": [5, 15, 30, 50, 100], "max_iter": 1000, "random_state": 42, "subset_size": 5000},
        "kmeans": {"k_values": [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 15, 20], "random_state": 42, "n_init": 10, "eval_sample_size": 5000},
        "umap": {"n_neighbors": 15, "min_dist": 0.1, "random_state": 42},
        "spectral": {"sample_size": 500, "random_state": 42},
        "output_dir": "pipeline_outputs",
        "figures_dir": "figures"
    }


def _read_image(path):
    """Load, convert to RGB, resize to 64x64, and flatten image array."""
    try:
        im = cv2.imread(path)
        if im is None:
            return None
        im_rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        if im_rgb.shape[:2] != (64, 64):
            im_rgb = cv2.resize(im_rgb, (64, 64))
        return im_rgb.ravel()
    except Exception:
        return None


def run_stage_5_classifier_engine(output_dir, scaler, pca, km10, cluster_mapping, **kwargs):
    """Stage 5: Image Inference Classifier Engine generation."""
    metadata_df = kwargs["metadata_df"]
    pca_reduced_95 = kwargs["pca_reduced_95"]

    print("\n" + "="*60)
    print("STAGE 5: Image Inference Classifier Engine")
    print("="*60)

    print("Building NearestNeighbors index for sample retrieval...")
    nn_model = NearestNeighbors(n_neighbors=5, metric="euclidean", algorithm="auto")
    nn_model.fit(pca_reduced_95)

    classifier = EuroSATClassifier(
        scaler=scaler,
        pca_model=pca,
        kmeans_model=km10,
        cluster_mapping=cluster_mapping,
        nn_model=nn_model,
        metadata_df=metadata_df,
        pca_coords=pca_reduced_95[:, :2]
    )

    joblib.dump(classifier, os.path.join(output_dir, "classifier.pkl"))
    print("Saved EuroSATClassifier to pipeline_outputs/classifier.pkl")

    test_res = classifier.predict(metadata_df.iloc[0]["filepath"])
    print(f"Sample test prediction on {metadata_df.iloc[0]['filename']}:")
    print(f"  Predicted Cluster: {test_res['predicted_cluster']}")
    print(f"  Mapped Land-Type: {test_res['majority_label']} (Cluster Purity: {test_res['confidence_purity']*100:.1f}%)")
    print(f"  Top Neighbor: {test_res['nearest_neighbors'][0]['filename']} ({test_res['nearest_neighbors'][0]['true_label']})")

    return classifier


def run_stage_6_enhancements(config, output_dir, models, metadata_df, pca_reduced_95, subset_indices):
    """Stage 6: Additional Enhancements (Spectral Analysis, UMAP, Exemplar Grids)."""
    scaler, pca, km10 = models["scaler"], models["pca"], models["km10"]

    print("\n" + "="*60)
    print("STAGE 6: Additional Enhancements (Spectral Analysis, UMAP, Exemplar Grids)")
    print("="*60)

    # 1. Cluster Exemplar Image Grids and Centroid Images
    grids_dir = os.path.join(output_dir, "cluster_grids")
    os.makedirs(grids_dir, exist_ok=True)
    labels_k10 = np.load(os.path.join(output_dir, "kmeans_labels_k10.npy"))
    centroids = km10.cluster_centers_

    print("Generating representative 4x4 image grids & centroid images for each cluster...")
    for c_id in range(10):
        c_mask = np.where(labels_k10 == c_id)[0]
        if len(c_mask) == 0:
            continue
        c_feats = pca_reduced_95[c_mask]
        dists = np.linalg.norm(c_feats - centroids[c_id], axis=1)
        closest_indices = c_mask[np.argsort(dists)[:16]]

        # Plot 4x4 grid
        fig, axes = plt.subplots(4, 4, figsize=(7, 7))
        for idx, ax in zip(closest_indices, axes.ravel()):
            path = metadata_df.iloc[idx]["filepath"]
            img = cv2.imread(path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            ax.imshow(img)
            true_lbl = metadata_df.iloc[idx]["true_label"]
            ax.set_title(true_lbl, fontsize=8)
            ax.axis('off')
        plt.suptitle(f"Cluster {c_id} Exemplars (Closest to Centroid)", fontsize=12, y=0.98)
        plt.tight_layout()
        plt.savefig(os.path.join(grids_dir, f"cluster_{c_id}_grid.png"), dpi=150)
        plt.close(fig)

        # Centroid reconstructed image
        centroid_pca = np.zeros((1, pca.n_components_))
        centroid_pca[0, :centroids.shape[1]] = centroids[c_id]
        centroid_scaled = pca.inverse_transform(centroid_pca)
        centroid_raw = scaler.inverse_transform(centroid_scaled)[0].reshape((64, 64, 3))
        centroid_clipped = np.clip(centroid_raw, 0, 255).astype(np.uint8)
        Image.fromarray(centroid_clipped).save(os.path.join(grids_dir, f"centroid_cluster_{c_id}.png"))

    # 2. UMAP 2D Embedding
    if HAS_UMAP:
        print("Computing UMAP 2D embedding...")
        umap_params = config.get("umap", {})
        n_neighbors = umap_params.get("n_neighbors", 15)
        min_dist = umap_params.get("min_dist", 0.1)
        rand_s = umap_params.get("random_state", 42)

        umap_input = pca_reduced_95[subset_indices]
        t0 = time.time()
        reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, random_state=rand_s)
        umap_2d = reducer.fit_transform(umap_input)
        print(f"UMAP completed in {time.time()-t0:.2f}s")
        np.save(os.path.join(output_dir, "umap_2d.npy"), umap_2d)
    else:
        print("umap-learn not installed, skipping UMAP embedding.")

    # 3. Spectral Band Analysis (Sentinel-2 13 Bands)
    allbands_dir = config["data"]["allbands_dir"]
    if HAS_RASTERIO and os.path.exists(allbands_dir):
        print("Extracting multi-spectral statistics from EuroSATallBands TIFs...")
        tifs = sorted(glob.glob(os.path.join(allbands_dir, "*", "*.tif")))
        if tifs:
            np.random.seed(config["spectral"].get("random_state", 42))
            sample_size = min(config["spectral"].get("sample_size", 500), len(tifs))
            sample_tifs = np.random.choice(tifs, size=sample_size, replace=False)

            band_records = []
            valid_tifs_meta = []
            band_names = [f"B{i+1:02d}" for i in range(13)]
            col_names = [f"{b}_mean" for b in band_names] + [f"{b}_std" for b in band_names]

            for tif_path in sample_tifs:
                try:
                    with rasterio.open(tif_path) as src:
                        data = src.read()  # (13, 64, 64)
                        means = [float(np.mean(data[b])) for b in range(13)]
                        stds = [float(np.std(data[b])) for b in range(13)]
                        band_records.append(means + stds)
                        cls_name = os.path.basename(os.path.dirname(tif_path))
                        valid_tifs_meta.append({"filename": os.path.basename(tif_path), "class_name": cls_name})
                except Exception:
                    continue

            if band_records:
                spectral_df = pd.DataFrame(band_records, columns=col_names)
                meta_df = pd.DataFrame(valid_tifs_meta)
                spectral_df.to_csv(os.path.join(output_dir, "spectral_features.csv"), index=False)
                meta_df.to_csv(os.path.join(output_dir, "spectral_metadata.csv"), index=False)

                # Spectral PCA 2D
                spec_scaled = StandardScaler().fit_transform(spectral_df.values)
                spec_pca = PCA(n_components=2, random_state=42)
                spec_2d = spec_pca.fit_transform(spec_scaled)
                np.save(os.path.join(output_dir, "spectral_pca_2d.npy"), spec_2d)
                print(f"Extracted {len(band_records)} multi-spectral profiles with 2D PCA representation.")
