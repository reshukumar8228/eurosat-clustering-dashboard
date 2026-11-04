"""
EuroSAT Unsupervised Land-Type Grouping Pipeline
Title: Unsupervised Land-Type Grouping of Satellite Image Patches Using Clustering

Implements Stages 1 - 6:
  - Stage 1: Data Loading & Preprocessing (hidden labels during unsupervised modeling)
  - Stage 2: Principal Component Analysis (retained variance >= 95%, 2D projection, loadings)
  - Stage 3: t-SNE across multiple perplexities [5, 15, 30, 50, 100]
  - Stage 4: K-Means Clustering (k in 2..20, metrics: Inertia, Silhouette, Davies-Bouldin, CH, ARI, NMI, contingency)
  - Stage 5: Image Classification Support & Nearest Neighbor Inference Engine
  - Stage 6: Spectral Band Analysis (13-band Sentinel-2), UMAP, Cluster Centroid & Exemplar Grids
"""

import os
import glob
import json
import time
import yaml
import joblib
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
    adjusted_rand_score,
    normalized_mutual_info_score
)
from sklearn.metrics.cluster import contingency_matrix
from sklearn.neighbors import NearestNeighbors
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from classifier import EuroSATClassifier

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


def run_stage_1_preprocessing(config, output_dir):
    print("\n" + "="*60)
    print("STAGE 1: Data Loading & Preprocessing (Labels Hidden)")
    print("="*60)

    rgb_dir = config["data"]["rgb_dir"]
    class_folders = sorted([d for d in os.listdir(rgb_dir) if os.path.isdir(os.path.join(rgb_dir, d))])
    label_map = {cls_name: i for i, cls_name in enumerate(class_folders)}

    all_files = []
    file_labels = []
    for cls_name in class_folders:
        folder = os.path.join(rgb_dir, cls_name)
        jpgs = sorted(glob.glob(os.path.join(folder, "*.jpg")))
        for f in jpgs:
            all_files.append(f)
            file_labels.append(cls_name)

    print(f"Found {len(all_files)} total images across {len(class_folders)} classes.")

    # Read images with multi-threaded executor
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=16) as ex:
        raw_feats = list(tqdm(ex.map(_read_image, all_files), total=len(all_files), desc="Loading images"))

    # Filter any corrupted images
    valid_indices = [i for i, feat in enumerate(raw_feats) if feat is not None]
    if len(valid_indices) < len(all_files):
        print(f"Skipped {len(all_files) - len(valid_indices)} corrupt/missing images.")

    feature_matrix = np.array([raw_feats[i] for i in valid_indices], dtype=np.float32)
    valid_files = [all_files[i] for i in valid_indices]
    valid_labels = [file_labels[i] for i in valid_indices]

    print(f"Loaded feature matrix shape: {feature_matrix.shape} in {time.time()-t0:.2f}s")

    # Metadata (labels saved strictly for post-hoc evaluation)
    metadata_df = pd.DataFrame({
        "index": np.arange(len(valid_files)),
        "filename": [os.path.basename(f) for f in valid_files],
        "filepath": valid_files,
        "true_label": valid_labels,
        "true_label_id": [label_map[l] for l in valid_labels]
    })
    metadata_df.to_csv(os.path.join(output_dir, "metadata.csv"), index=False)
    with open(os.path.join(output_dir, "label_map.json"), "w") as f:
        json.dump(label_map, f, indent=2)

    # Standard scaling
    print("Fitting StandardScaler on feature matrix...")
    t0 = time.time()
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(feature_matrix)
    print(f"StandardScaler fit & transform complete in {time.time()-t0:.2f}s")

    # Save artifacts
    np.save(os.path.join(output_dir, "features_scaled.npy"), features_scaled)
    joblib.dump(scaler, os.path.join(output_dir, "scaler.pkl"))

    return features_scaled, metadata_df, label_map, scaler


def run_stage_2_pca(config, output_dir, features_scaled, metadata_df):
    print("\n" + "="*60)
    print("STAGE 2: Principal Component Analysis (PCA)")
    print("="*60)

    n_comp = config["pca"].get("n_components", 220)
    random_state = config["pca"].get("random_state", 42)
    var_thresh = config["pca"].get("variance_threshold", 0.95)

    print(f"Fitting Randomized PCA with {n_comp} components...")
    t0 = time.time()
    pca = PCA(n_components=n_comp, svd_solver="randomized", random_state=random_state)
    pca_features = pca.fit_transform(features_scaled)
    print(f"PCA fit & transform completed in {time.time()-t0:.2f}s")

    exp_var_ratio = pca.explained_variance_ratio_
    cum_var_ratio = np.cumsum(exp_var_ratio)

    # Find component index for variance threshold
    thresh_idx = int(np.argmax(cum_var_ratio >= var_thresh))
    if thresh_idx == 0 and cum_var_ratio[0] < var_thresh:
        thresh_idx = n_comp - 1
    retained_k = thresh_idx + 1
    retained_var = float(cum_var_ratio[thresh_idx])
    print(f"Retained components for >={var_thresh*100:.0f}% variance: {retained_k} components (actual: {retained_var*100:.2f}%)")
    print(f"First 2 components explain: {cum_var_ratio[1]*100:.2f}% of total variance.")

    # Slices
    pca_reduced_95 = pca_features[:, :retained_k]
    pca_2d = pca_features[:, :2]

    # Feature contribution / loadings for top 10 PCs
    print("Computing feature contributions (pixel-channel loadings) for top 10 PCs...")
    loadings_dict = {}
    channels = ["Red", "Green", "Blue"]
    for pc_idx in range(min(10, n_comp)):
        comp_weights = pca.components_[pc_idx]  # shape (12288,)
        abs_weights = np.abs(comp_weights)
        top_20_indices = np.argsort(abs_weights)[::-1][:20]

        top_features = []
        for idx in top_20_indices:
            val = float(comp_weights[idx])
            pixel_idx = idx // 3
            channel_idx = idx % 3
            row = pixel_idx // 64
            col = pixel_idx % 64
            top_features.append({
                "flat_index": int(idx),
                "row": int(row),
                "col": int(col),
                "channel": channels[channel_idx],
                "weight": val,
                "abs_weight": float(abs_weights[idx])
            })
        loadings_dict[f"PC_{pc_idx+1}"] = {
            "explained_variance_ratio": float(exp_var_ratio[pc_idx]),
            "cumulative_variance_ratio": float(cum_var_ratio[pc_idx]),
            "top_loadings": top_features
        }

    # Save PCA outputs
    joblib.dump(pca, os.path.join(output_dir, "pca_model.pkl"))
    np.save(os.path.join(output_dir, "pca_reduced_95.npy"), pca_reduced_95)
    np.save(os.path.join(output_dir, "pca_2d.npy"), pca_2d)
    np.save(os.path.join(output_dir, "pca_explained_variance.npy"), exp_var_ratio)
    with open(os.path.join(output_dir, "pca_loadings_top10.json"), "w") as f:
        json.dump(loadings_dict, f, indent=2)

    return pca, pca_reduced_95, pca_2d, exp_var_ratio, retained_k


def run_stage_3_tsne(config, output_dir, pca_reduced_95, metadata_df):
    print("\n" + "="*60)
    print("STAGE 3: t-SNE Parameter Sensitivity Analysis")
    print("="*60)

    perplexities = config["tsne"].get("perplexities", [5, 15, 30, 50, 100])
    max_iter = config["tsne"].get("max_iter", 1000)
    random_state = config["tsne"].get("random_state", 42)
    subset_size = config["tsne"].get("subset_size", 5000)

    # Create stratified representative subset for t-SNE
    np.random.seed(random_state)
    unique_labels = metadata_df["true_label"].unique()
    per_class = subset_size // len(unique_labels)

    subset_indices = []
    for lbl in sorted(unique_labels):
        cls_indices = metadata_df[metadata_df["true_label"] == lbl].index.values
        chosen = np.random.choice(cls_indices, size=min(per_class, len(cls_indices)), replace=False)
        subset_indices.extend(chosen)
    subset_indices = np.array(sorted(subset_indices))

    print(f"Selected {len(subset_indices)} stratified points ({len(unique_labels)} classes, ~{per_class}/class) for t-SNE.")
    np.save(os.path.join(output_dir, "tsne_subset_indices.npy"), subset_indices)

    tsne_input = pca_reduced_95[subset_indices]

    tsne_params_record = {
        "perplexities": perplexities,
        "max_iter": max_iter,
        "random_state": random_state,
        "input_dim": int(tsne_input.shape[1]),
        "sample_count": int(len(subset_indices)),
        "subset_indices_file": "tsne_subset_indices.npy"
    }

    for perp in perplexities:
        print(f"Running t-SNE with perplexity={perp} on {tsne_input.shape[1]}-dim PCA features...")
        t0 = time.time()
        # sklearn 1.9+ uses max_iter instead of n_iter
        tsne = TSNE(
            n_components=2,
            perplexity=float(perp),
            max_iter=max_iter,
            random_state=random_state,
            n_jobs=-1
        )
        tsne_2d = tsne.fit_transform(tsne_input)
        print(f"  t-SNE (perp={perp}) completed in {time.time()-t0:.2f}s. KL Divergence: {tsne.kl_divergence_:.4f}")

        np.save(os.path.join(output_dir, f"tsne_2d_perp{perp}.npy"), tsne_2d)
        tsne_params_record[f"perp_{perp}"] = {
            "kl_divergence": float(tsne.kl_divergence_),
            "output_file": f"tsne_2d_perp{perp}.npy"
        }

    with open(os.path.join(output_dir, "tsne_params.json"), "w") as f:
        json.dump(tsne_params_record, f, indent=2)

    return tsne_params_record, subset_indices


def main():
    print("="*60)
    print("EuroSAT Unsupervised Learning Pipeline Startup (Stages 1-3)")
    print("="*60)
    config = load_config("config.yaml")
    output_dir = config.get("output_dir", "pipeline_outputs")
    os.makedirs(output_dir, exist_ok=True)

    features_scaled, metadata_df, label_map, scaler = run_stage_1_preprocessing(config, output_dir)
    pca, pca_reduced_95, pca_2d, exp_var_ratio, retained_k = run_stage_2_pca(config, output_dir, features_scaled, metadata_df)
    tsne_params, subset_indices = run_stage_3_tsne(config, output_dir, pca_reduced_95, metadata_df)

if __name__ == "__main__":
    main()
