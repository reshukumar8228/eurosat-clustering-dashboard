"""
Classifier and Inference Engine for EuroSAT Unsupervised Land-Type Grouping.
"""

import numpy as np
from PIL import Image

class EuroSATClassifier:
    """Inference engine to classify newly uploaded satellite patches based on unsupervised cluster structure."""
    def __init__(self, scaler, pca_model, kmeans_model, cluster_mapping, nn_model, **kwargs):
        self.scaler = scaler
        self.pca_model = pca_model
        self.kmeans_model = kmeans_model
        self.cluster_mapping = cluster_mapping
        self.nn_model = nn_model
        self.metadata_df = kwargs.get("metadata_df")
        self.pca_coords = kwargs.get("pca_coords")

    def predict(self, image_input):
        """
        Accepts: file path, PIL.Image, or numpy ndarray (H, W, 3).
        Returns: detailed dictionary with predictions, nearest neighbors, metrics and coords.
        """
        if isinstance(image_input, str):
            img = Image.open(image_input).convert('RGB')
        elif isinstance(image_input, Image.Image):
            img = image_input.convert('RGB')
        elif isinstance(image_input, np.ndarray):
            if image_input.ndim == 2:
                img = Image.fromarray(image_input).convert('RGB')
            else:
                img = Image.fromarray(image_input[:, :, :3].astype(np.uint8))
        else:
            raise ValueError(f"Unsupported image input type: {type(image_input)}")

        # Resize to 64x64
        img_resized = img.resize((64, 64), Image.Resampling.BILINEAR)
        img_arr = np.array(img_resized, dtype=np.float32)

        # Image characteristics
        mean_rgb = img_arr.mean(axis=(0, 1)).tolist()
        brightness = float(np.mean(img_arr))
        roughness = float(np.std(img_arr))

        # Flatten & scale
        flat_feat = img_arr.reshape(1, -1)
        scaled_feat = self.scaler.transform(flat_feat)

        # PCA projection
        pca_feat_95 = self.pca_model.transform(scaled_feat)
        pca_2d = pca_feat_95[0, :2].tolist()

        # Cluster prediction
        cluster_id = int(self.kmeans_model.predict(pca_feat_95)[0])

        # Distances to all centroids
        centroids = self.kmeans_model.cluster_centers_
        dists = np.linalg.norm(centroids - pca_feat_95, axis=1)
        centroid_distances = {f"Cluster {i}": float(dists[i]) for i in range(len(centroids))}

        # Cluster metadata
        cluster_info = self.cluster_mapping.get(str(cluster_id), {
            "majority_true_label": "Unknown",
            "purity": 0.0,
            "count": 0
        })

        # Nearest neighbors in training dataset
        dists_nn, indices_nn = self.nn_model.kneighbors(pca_feat_95, n_neighbors=5)
        neighbors = []
        for dist, idx in zip(dists_nn[0], indices_nn[0]):
            row = self.metadata_df.iloc[idx]
            neighbors.append({
                "index": int(row["index"]),
                "filename": str(row["filename"]),
                "filepath": str(row["filepath"]),
                "true_label": str(row["true_label"]),
                "distance": float(dist)
            })

        return {
            "predicted_cluster": cluster_id,
            "majority_label": cluster_info["majority_true_label"],
            "confidence_purity": float(cluster_info["purity"]),
            "cluster_size": int(cluster_info["count"]),
            "centroid_distances": centroid_distances,
            "nearest_neighbors": neighbors,
            "pca_2d": pca_2d,
            "image_stats": {
                "mean_brightness": brightness,
                "texture_roughness": roughness,
                "mean_red": mean_rgb[0],
                "mean_green": mean_rgb[1],
                "mean_blue": mean_rgb[2]
            }
        }
