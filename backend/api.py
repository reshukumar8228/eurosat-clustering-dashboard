"""
FastAPI Backend for EuroSAT Vision UI Dashboard
"""

import os
import sys
import json
import base64
import logging
import joblib
import numpy as np
import pandas as pd
from PIL import Image
import io

# Setup structured logging
logger = logging.getLogger("eurosat.api")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# Ensure project root is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fastapi import FastAPI, UploadFile, File, Form, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel, Field
from sklearn.cluster import KMeans as KMeansModel
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

from classifier import EuroSATClassifier
from features import extract_rich_features, align_image_illumination, detect_non_satellite

app = FastAPI(title="EuroSAT Vision API", version="2.0")

# Security: Configured origins instead of unrestricted wildcard with credentials
ALLOWED_ORIGINS = [
    origin.strip()
    for origin in os.getenv(
        "ALLOWED_ORIGINS",
        "http://localhost:8000,http://127.0.0.1:8000,http://localhost:5173,http://127.0.0.1:5173"
    ).split(",")
    if origin.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

OUTPUT_DIR = "pipeline_outputs"
DATA_DIR = "Data/EuroSAT"
MAX_FILE_SIZE = 25 * 1024 * 1024  # 25 MB maximum upload size
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp", ".bmp", ".jfif", ".avif"}

# Mount static directories
os.makedirs("uploads", exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
app.mount("/static/outputs", StaticFiles(directory=OUTPUT_DIR), name="outputs")


@app.get("/static/data/{category}/{filename}")
def get_satellite_patch(category: str, filename: str):
    """Serve satellite image patch with path traversal defense and fallback."""
    # Sanitize path segments to prevent directory traversal
    clean_category = os.path.basename(category.strip())
    clean_filename = os.path.basename(filename.strip())

    base_dir = os.path.realpath(DATA_DIR)
    target = os.path.realpath(os.path.join(base_dir, clean_category, clean_filename))

    # Path traversal check: target must be inside DATA_DIR
    if not (target.startswith(base_dir + os.sep) or target == base_dir):
        return JSONResponse(status_code=403, content={"error": "Access denied: Path traversal detected"})

    if os.path.exists(target) and os.path.isfile(target):
        return FileResponse(target, media_type="image/jpeg")

    # Fallback to any representative sample patch in this category
    cat_dir = os.path.realpath(os.path.join(base_dir, clean_category))
    if cat_dir.startswith(base_dir + os.sep) and os.path.isdir(cat_dir):
        available = [f for f in os.listdir(cat_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        if available:
            return FileResponse(os.path.join(cat_dir, available[0]), media_type="image/jpeg")

    # Fallback to centroid cluster thumbnail if available
    centroid = os.path.join(OUTPUT_DIR, "cluster_grids", "centroid_cluster_0.png")
    if os.path.exists(centroid):
        return FileResponse(centroid, media_type="image/png")
    return JSONResponse(status_code=404, content={"error": "Image not found"})

# In-memory cached dataset state
print("Loading precomputed pipeline outputs into memory...")
metadata_df = pd.read_csv(os.path.join(OUTPUT_DIR, "metadata.csv"))
pca_2d = np.load(os.path.join(OUTPUT_DIR, "pca_2d.npy"))
exp_var = np.load(os.path.join(OUTPUT_DIR, "pca_explained_variance.npy"))
with open(os.path.join(OUTPUT_DIR, "pca_loadings_top10.json")) as f:
    pca_loadings = json.load(f)

subset_indices = np.load(os.path.join(OUTPUT_DIR, "tsne_subset_indices.npy"))
with open(os.path.join(OUTPUT_DIR, "tsne_params.json")) as f:
    tsne_params = json.load(f)

tsne_embeddings = {}
for p in tsne_params.get("perplexities", [5, 15, 30, 50, 100]):
    path = os.path.join(OUTPUT_DIR, f"tsne_2d_perp{p}.npy")
    if os.path.exists(path):
        tsne_embeddings[str(p)] = np.load(path)

with open(os.path.join(OUTPUT_DIR, "kmeans_metrics.json")) as f:
    kmeans_metrics = json.load(f)
with open(os.path.join(OUTPUT_DIR, "cluster_label_mapping.json")) as f:
    cluster_mapping = json.load(f)
contingency_df = pd.read_csv(os.path.join(OUTPUT_DIR, "kmeans_k10_contingency.csv"), index_col=0)
labels_k10 = np.load(os.path.join(OUTPUT_DIR, "kmeans_labels_k10.npy"))

pca_reduced_full = np.load(os.path.join(OUTPUT_DIR, "pca_reduced_95.npy"))

kmeans_labels_cache = {}
kmeans_models_cache = {}
models_dir = os.path.join(OUTPUT_DIR, "kmeans_models")
if os.path.exists(models_dir):
    for k_file in os.listdir(models_dir):
        if k_file.endswith(".pkl") and k_file.startswith("kmeans_k"):
            try:
                k_val = int(k_file.replace("kmeans_k", "").replace(".pkl", ""))
                kmeans_models_cache[k_val] = joblib.load(os.path.join(models_dir, k_file))
                label_path = os.path.join(OUTPUT_DIR, f"kmeans_labels_k{k_val}.npy")
                if os.path.exists(label_path):
                    kmeans_labels_cache[k_val] = np.load(label_path)
            except Exception as e:
                logger.warning("Failed loading kmeans model %s: %s", k_file, e)

umap_2d = None
if os.path.exists(os.path.join(OUTPUT_DIR, "umap_2d.npy")):
    umap_2d = np.load(os.path.join(OUTPUT_DIR, "umap_2d.npy"))

classifier: EuroSATClassifier = joblib.load(os.path.join(OUTPUT_DIR, "classifier.pkl"))

supervised_clf = None
sup_path = os.path.join(OUTPUT_DIR, "supervised_classifier.pkl")
if os.path.exists(sup_path):
    supervised_clf = joblib.load(sup_path)

label_names = sorted(metadata_df["true_label"].unique())

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


@app.get("/api/health")
def health():
    return {"status": "ok", "samples": len(metadata_df)}


@app.get("/api/overview")
def get_overview():
    counts = metadata_df["true_label"].value_counts().to_dict()
    breakdown = [
        {"name": k, "count": int(v), "percentage": round(v / len(metadata_df) * 100, 2), "color": CLASS_COLORS.get(k, "#3B82F6")}
        for k, v in sorted(counts.items())
    ]

    # Pre-seed 1 sample image per class and 12 samples for instant zero-latency gallery rendering
    sample_images = []
    category_samples = {}
    for cls in label_names:
        cat_df = metadata_df[metadata_df["true_label"] == cls]
        first_row = cat_df.iloc[0]
        sample_images.append({
            "class": cls,
            "filename": first_row["filename"],
            "url": f"/static/data/{cls}/{first_row['filename']}"
        })

        # Pre-seed 12 samples per class for instant zero-latency gallery rendering
        top12 = cat_df.head(12)
        samples_list = []
        for _, row in top12.iterrows():
            idx_val = int(row["index"])
            samples_list.append({
                "index": idx_val,
                "filename": row["filename"],
                "category": cls,
                "url": f"/static/data/{cls}/{row['filename']}",
                "cluster_k10": int(labels_k10[idx_val])
            })
        category_samples[cls] = samples_list

    return {
        "total_images": len(metadata_df),
        "total_classes": len(label_names),
        "dimensions": "64 × 64 × 3",
        "features_per_image": 12288,
        "spectral_bands": 13,
        "class_breakdown": breakdown,
        "sample_images": sample_images,
        "category_samples": category_samples
    }


@app.get("/api/category-samples")
def get_category_samples(category: str = Query("Highway"), count: int = Query(12, ge=1, le=48)):
    cat_df = metadata_df[metadata_df["true_label"] == category]
    if len(cat_df) == 0:
        return JSONResponse(status_code=404, content={"error": f"Category '{category}' not found"})

    total_available = len(cat_df)
    n_samples = min(count, total_available)
    sampled_rows = cat_df.iloc[:n_samples]

    samples = []
    for _, row in sampled_rows.iterrows():
        idx_val = int(row["index"])
        samples.append({
            "index": idx_val,
            "filename": row["filename"],
            "category": category,
            "url": f"/static/data/{category}/{row['filename']}",
            "cluster_k10": int(labels_k10[idx_val])
        })

    return {
        "category": category,
        "total_available": total_available,
        "count": len(samples),
        "color": CLASS_COLORS.get(category, "#3B82F6"),
        "samples": samples
    }


@app.get("/api/pca")
def get_pca(sample_size: int = Query(3000, le=27000)):
    cum_var = np.cumsum(exp_var).tolist()
    scree_data = [
        {"component": i + 1, "variance": float(exp_var[i] * 100), "cumulative": float(cum_var[i] * 100)}
        for i in range(min(50, len(exp_var)))
    ]

    # Sample points for scatter plot
    step = max(1, len(subset_indices) // sample_size)
    sampled_idx = subset_indices[::step]

    points = []
    for idx in sampled_idx:
        row = metadata_df.iloc[idx]
        points.append({
            "x": round(float(pca_2d[idx, 0]), 2),
            "y": round(float(pca_2d[idx, 1]), 2),
            "label": row["true_label"],
            "cluster": int(labels_k10[idx]),
            "filename": row["filename"]
        })

    return {
        "variance_pc1": round(float(cum_var[0] * 100), 2),
        "variance_pc2": round(float((cum_var[1] - cum_var[0]) * 100), 2),
        "variance_top2": round(float(cum_var[1] * 100), 2),
        "variance_total": round(float(cum_var[-1] * 100), 2),
        "total_components": len(exp_var),
        "scree": scree_data,
        "scatter_points": points,
        "loadings": pca_loadings
    }


@app.get("/api/tsne")
def get_tsne(perplexity: int = Query(30)):
    p_str = str(perplexity)
    if p_str not in tsne_embeddings:
        p_str = "30"

    emb = tsne_embeddings[p_str]
    step = 2  # 2500 points for silky smooth 60fps web rendering
    sampled = list(range(0, len(subset_indices), step))

    points = []
    for i in sampled:
        idx = subset_indices[i]
        row = metadata_df.iloc[idx]
        points.append({
            "x": round(float(emb[i, 0]), 2),
            "y": round(float(emb[i, 1]), 2),
            "label": row["true_label"],
            "cluster": int(labels_k10[idx]),
            "filename": row["filename"]
        })

    params_info = tsne_params.get(f"perp_{p_str}", {})

    return {
        "perplexity": int(p_str),
        "available_perplexities": [5, 15, 30, 50, 100],
        "kl_divergence": round(float(params_info.get("kl_divergence", 0.0)), 4),
        "points": points
    }


@app.get("/api/clustering")
def get_clustering():
    k_vals = sorted([int(k) for k in kmeans_metrics.keys()])
    eval_curves = [
        {
            "k": k,
            "inertia": round(float(kmeans_metrics[str(k)]["inertia"]), 1),
            "silhouette": round(float(kmeans_metrics[str(k)]["silhouette"]), 4),
            "davies_bouldin": round(float(kmeans_metrics[str(k)]["davies_bouldin"]), 4),
            "calinski_harabasz": round(float(kmeans_metrics[str(k)]["calinski_harabasz"]), 1)
        }
        for k in k_vals
    ]

    k10_info = kmeans_metrics.get("10", {})
    mapping_list = [
        {
            "cluster_id": int(cid),
            "majority_label": info["majority_true_label"],
            "purity": round(float(info["purity"]) * 100, 1),
            "count": int(info["count"]),
            "grid_url": f"/static/outputs/cluster_grids/cluster_{cid}_grid.png",
            "centroid_url": f"/static/outputs/cluster_grids/centroid_cluster_{cid}.png"
        }
        for cid, info in cluster_mapping.items()
    ]

    contingency_matrix_list = []
    for r_idx, row in contingency_df.iterrows():
        contingency_matrix_list.append({
            "cluster": r_idx,
            "values": {col: int(row[col]) for col in contingency_df.columns}
        })

    return {
        "k_curves": eval_curves,
        "ari": round(float(k10_info.get("adjusted_rand_index", 0.0)), 4),
        "nmi": round(float(k10_info.get("normalized_mutual_info", 0.0)), 4),
        "overall_purity": round(float(k10_info.get("overall_purity", 0.0)) * 100, 1),
        "clusters": mapping_list,
        "contingency": contingency_matrix_list,
        "classes": list(contingency_df.columns)
    }


@app.post("/api/predict")
async def predict_image(file: UploadFile = File(None), preset_path: str = Form(None)):
    try:
        is_upload = (
            file is not None 
            and not isinstance(file, type(File(None))) 
            and hasattr(file, "filename") 
            and bool(file.filename)
        )
        has_preset = (
            preset_path is not None 
            and not isinstance(preset_path, type(Form(None))) 
            and bool(preset_path.strip())
        )

        if is_upload:
            filename = file.filename or "uploaded.jpg"
            ext = os.path.splitext(filename)[1].lower()
            if ext and ext not in ALLOWED_EXTENSIONS:
                return JSONResponse(
                    status_code=400,
                    content={"error": f"Unsupported format '{ext}'. Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}"}
                )

            # Read contents safely without deadlocks in anyio/starlette threads
            if hasattr(file, "file") and hasattr(file.file, "read"):
                contents = file.file.read()
            elif hasattr(file, "read"):
                contents = await file.read() if asyncio.iscoroutinefunction(file.read) else file.read()
            else:
                contents = b""
            if len(contents) > MAX_FILE_SIZE:
                return JSONResponse(status_code=413, content={"error": "Uploaded image exceeds 25MB size limit"})

            # Prevent decompression bomb attacks
            Image.MAX_IMAGE_PIXELS = 50_000_000

            # Robust multi-format image decoding
            img = None
            try:
                raw_img = Image.open(io.BytesIO(contents))
                img = raw_img.convert("RGB")
            except Exception as pil_err:
                # Fallback for multi-band TIF (e.g., Sentinel-2 13-band TIFF)
                try:
                    import tifffile
                    arr = tifffile.imread(io.BytesIO(contents))
                    if arr.ndim == 3 and arr.shape[2] >= 3:
                        if arr.shape[2] == 13:
                            # Sentinel-2 B4(Red)=3, B3(Green)=2, B2(Blue)=1
                            rgb_arr = arr[:, :, [3, 2, 1]].astype(np.float32)
                        else:
                            rgb_arr = arr[:, :, :3].astype(np.float32)
                        min_v, max_v = float(rgb_arr.min()), float(rgb_arr.max())
                        if max_v > min_v:
                            rgb_arr = (rgb_arr - min_v) / (max_v - min_v) * 255.0
                        img = Image.fromarray(rgb_arr.astype(np.uint8))
                    elif arr.ndim == 2:
                        min_v, max_v = float(arr.min()), float(arr.max())
                        norm = (arr - min_v) / (max_v - min_v + 1e-5) * 255.0
                        img = Image.fromarray(norm.astype(np.uint8)).convert("RGB")
                    else:
                        raise pil_err
                except Exception:
                    # Final fallback to OpenCV
                    try:
                        import cv2
                        nparr = np.frombuffer(contents, np.uint8)
                        cv_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                        if cv_img is not None:
                            img = Image.fromarray(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB))
                        else:
                            raise pil_err
                    except Exception:
                        raise pil_err

            source_name = os.path.basename(filename)
        elif has_preset:
            base_dir = os.path.realpath(DATA_DIR)
            resolved_preset = os.path.realpath(preset_path)
            # Strictly ensure preset_path stays inside DATA_DIR
            if not (resolved_preset.startswith(base_dir + os.sep) and os.path.isfile(resolved_preset)):
                return JSONResponse(status_code=400, content={"error": "Invalid or unauthorized preset path"})

            img = Image.open(resolved_preset).convert("RGB")
            source_name = os.path.basename(resolved_preset)
        else:
            return JSONResponse(status_code=400, content={"error": "No image provided"})

        if img is None:
            return JSONResponse(status_code=400, content={"error": "Could not decode uploaded image"})

        # Align illumination for external images whose camera/sensor exposure differs
        img_aligned = align_image_illumination(img)

        # Run inference via EuroSATClassifier
        res = classifier.predict(img_aligned)

        # Supervised classification prediction with multi-scale feature support
        supervised_prediction = None
        supervised_probs = None
        supervised_confidence = None
        if supervised_clf is not None:
            img_resized = img_aligned.resize((64, 64), Image.Resampling.BILINEAR)
            flat = np.array(img_resized, dtype=np.float32).reshape(1, -1)
            scaled = classifier.scaler.transform(flat)
            pca_feat = classifier.pca_model.transform(scaled)

            n_in = getattr(supervised_clf, "n_features_in_", 220)
            if n_in == 244:
                rich_feat = np.array([extract_rich_features(img_aligned)], dtype=np.float32)
                eval_feat = np.hstack([pca_feat, rich_feat])
            else:
                eval_feat = pca_feat

            pred_id = int(supervised_clf.predict(eval_feat)[0])
            supervised_prediction = label_names[pred_id]
            if hasattr(supervised_clf, "predict_proba"):
                probs = supervised_clf.predict_proba(eval_feat)[0]
                supervised_probs = {label_names[i]: round(float(probs[i]) * 100, 1) for i in range(len(label_names))}
                supervised_confidence = round(float(probs[pred_id]) * 100, 1)

        # Format nearest neighbors with web-accessible URLs
        neighbors_formatted = []
        min_neighbor_dist = float("inf")
        for n in res.get("nearest_neighbors", []):
            dist_val = round(float(n["distance"]), 2)
            if dist_val < min_neighbor_dist:
                min_neighbor_dist = dist_val
            rel_path = n["filepath"].replace("\\", "/").split("Data/EuroSAT/")[-1]
            clean_rel = "/".join(os.path.basename(seg) for seg in rel_path.split("/") if seg and seg != "..")
            neighbors_formatted.append({
                "filename": n["filename"],
                "true_label": n["true_label"],
                "distance": dist_val,
                "url": f"/static/data/{clean_rel}"
            })

        # Base64 thumbnail for instant preview
        buffered = io.BytesIO()
        img.resize((128, 128)).save(buffered, format="JPEG")
        img_b64 = "data:image/jpeg;base64," + base64.b64encode(buffered.getvalue()).decode()

        # Outlier / Domain-shift & Non-Satellite (OOD) assessment
        is_ood, ood_score, ood_min_dist, ood_reasons = detect_non_satellite(
            img, classifier.scaler, classifier.pca_model, classifier.nn_model
        )

        if is_ood:
            supervised_prediction = "Unrecognized (Non-Satellite Image)"
            supervised_confidence = 0.0
            supervised_probs = {"Unrecognized / Non-Satellite": 100.0}
            reasons_msg = "; ".join(ood_reasons) if ood_reasons else f"Euclidean distance ({min_neighbor_dist:.1f}) exceeds valid terrain manifold"
            domain_note = f"Non-Satellite Image Detected: This image does not match any recognized EuroSAT Earth observation land-cover categories ({reasons_msg}). The model cannot reliably categorize non-satellite imagery."
            cluster_mapped_label = "No Satellite Match (Out of Distribution)"
            cluster_purity = 0.0
        else:
            cluster_mapped_label = res["majority_label"]
            cluster_purity = round(res["confidence_purity"] * 100, 1)
            is_outlier = min_neighbor_dist > 30.0
            domain_note = None
            if is_outlier:
                domain_note = "External / Cross-Sensor Patch: Resolution and spectral response diverge from EuroSAT Sentinel-2 baseline. Classification leverages spatial structural and texture heuristics."

        return {
            "source_name": source_name,
            "thumbnail_b64": img_b64,
            "is_out_of_distribution": is_ood,
            "unsupervised": {
                "predicted_cluster": res["predicted_cluster"],
                "mapped_label": cluster_mapped_label,
                "purity_confidence": cluster_purity,
                "cluster_size": res["cluster_size"],
                "centroid_distances": [
                    {"cluster": k, "distance": round(v, 2)}
                    for k, v in sorted(res["centroid_distances"].items(), key=lambda x: x[1])
                ]
            },
            "supervised_semantic": {
                "predicted_label": supervised_prediction or cluster_mapped_label,
                "confidence": supervised_confidence if supervised_confidence is not None else cluster_purity,
                "probabilities": supervised_probs,
                "domain_note": domain_note,
                "is_out_of_distribution": is_ood
            },
            "nearest_neighbors": neighbors_formatted,
            "pca_coords": [round(res["pca_2d"][0], 2), round(res["pca_2d"][1], 2)],
            "physical_stats": {
                "brightness": round(float(res["image_stats"]["mean_brightness"]), 1),
                "roughness": round(float(res["image_stats"]["texture_roughness"]), 1),
                "red": round(float(res["image_stats"]["mean_red"]), 1),
                "green": round(float(res["image_stats"]["mean_green"]), 1),
                "blue": round(float(res["image_stats"]["mean_blue"]), 1)
            }
        }
    except Exception as e:
        logger.error("Error during image inference: %s", str(e), exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"error": f"Failed to process image inference: {str(e)}"}
        )


class ExperimentParams(BaseModel):
    pca_components: int = Field(default=220, ge=2, le=220)
    pca_axis_x: int = Field(default=1, ge=1, le=220)
    pca_axis_y: int = Field(default=2, ge=1, le=220)
    kmeans_k: int = Field(default=10, ge=2, le=30)
    tsne_perplexity: int = Field(default=30)
    sample_size: int = Field(default=2000, ge=500, le=5000)


@app.post("/api/experiment")
def run_experiment(params: ExperimentParams):
    import time as _time
    t_start = _time.time()

    n_comp = params.pca_components
    ax_x = params.pca_axis_x - 1
    ax_y = params.pca_axis_y - 1

    if ax_x >= pca_reduced_full.shape[1]:
        ax_x = 0
    if ax_y >= pca_reduced_full.shape[1]:
        ax_y = 1

    # --- PCA Section ---
    pca_subset = pca_reduced_full[:, :n_comp]
    cum_var = np.cumsum(exp_var[:n_comp])
    variance_retained = float(cum_var[-1] * 100) if len(cum_var) > 0 else 0.0

    scree_data = [
        {
            "component": i + 1,
            "variance": round(float(exp_var[i] * 100), 2),
            "cumulative": round(float(np.cumsum(exp_var)[i] * 100), 2)
        }
        for i in range(min(50, len(exp_var)))
    ]

    # --- K-Means Section ---
    k = params.kmeans_k
    is_precomputed = (n_comp == 220) and (k in kmeans_labels_cache)
    km_time_start = _time.time()

    if is_precomputed:
        cluster_labels = kmeans_labels_cache[k]
        km_model = kmeans_models_cache[k]
        cached_metric = kmeans_metrics.get(str(k), {})
        inertia_val = float(cached_metric.get("inertia", getattr(km_model, "inertia_", 0.0)))
        sil_val = float(cached_metric.get("silhouette", 0.0))
        db_val = float(cached_metric.get("davies_bouldin", 0.0))
        ch_val = float(cached_metric.get("calinski_harabasz", 0.0))
    else:
        # Live recomputation on the retained PCA components
        km_model = KMeansModel(n_clusters=k, random_state=42, n_init=5 if n_comp < 50 else 10)
        cluster_labels = km_model.fit_predict(pca_subset)
        inertia_val = float(km_model.inertia_)

        eval_idx = np.random.RandomState(42).choice(
            len(pca_subset), size=min(3000, len(pca_subset)), replace=False
        )
        eval_X = pca_subset[eval_idx]
        eval_labels = cluster_labels[eval_idx]

        sil_val = float(silhouette_score(eval_X, eval_labels))
        db_val = float(davies_bouldin_score(eval_X, eval_labels))
        ch_val = float(calinski_harabasz_score(eval_X, eval_labels))

    km_time_ms = round((_time.time() - km_time_start) * 1000, 1)

    # Cluster sizes
    unique, counts = np.unique(cluster_labels, return_counts=True)
    cluster_sizes = [{"cluster": int(u), "size": int(c)} for u, c in zip(unique, counts)]

    # --- PCA Scatter Points ---
    step = max(1, len(subset_indices) // params.sample_size)
    sampled_idx = subset_indices[::step]

    pca_points = []
    for idx in sampled_idx:
        row = metadata_df.iloc[idx]
        pca_points.append({
            "x": round(float(pca_reduced_full[idx, ax_x]), 2),
            "y": round(float(pca_reduced_full[idx, ax_y]), 2),
            "label": row["true_label"],
            "cluster": int(cluster_labels[idx]),
            "filename": row["filename"]
        })

    # --- t-SNE Section ---
    p_str = str(params.tsne_perplexity)
    if p_str not in tsne_embeddings:
        p_str = "30"
    emb = tsne_embeddings[p_str]
    step_t = 2
    sampled_t = list(range(0, len(subset_indices), step_t))

    tsne_points = []
    for i in sampled_t:
        idx = subset_indices[i]
        row = metadata_df.iloc[idx]
        tsne_points.append({
            "x": round(float(emb[i, 0]), 2),
            "y": round(float(emb[i, 1]), 2),
            "label": row["true_label"],
            "cluster": int(cluster_labels[idx]),
            "filename": row["filename"]
        })

    tsne_info = tsne_params.get(f"perp_{p_str}", {})

    total_ms = round((_time.time() - t_start) * 1000, 1)

    return {
        "pca": {
            "variance_retained": round(variance_retained, 2),
            "components_used": n_comp,
            "axis_x": params.pca_axis_x,
            "axis_y": params.pca_axis_y,
            "scatter_points": pca_points,
            "scree": scree_data
        },
        "kmeans": {
            "k": k,
            "inertia": round(inertia_val, 1),
            "silhouette": round(sil_val, 4),
            "davies_bouldin": round(db_val, 4),
            "calinski_harabasz": round(ch_val, 1),
            "is_precomputed": is_precomputed,
            "cluster_sizes": cluster_sizes,
            "computation_time_ms": km_time_ms
        },
        "tsne": {
            "perplexity": int(p_str),
            "kl_divergence": round(float(tsne_info.get("kl_divergence", 0.0)), 4),
            "scatter_points": tsne_points
        },
        "comparison": {
            "total_samples": len(metadata_df),
            "subset_size": len(subset_indices),
            "computation_time_ms": total_ms
        }
    }


# Mount built React frontend distribution (Vision UI)
if os.path.exists("frontend/dist"):
    app.mount("/", StaticFiles(directory="frontend/dist", html=True), name="frontend")


def _launch_browser_when_ready(host="127.0.0.1", port=8000, timeout=30):
    import time
    import urllib.request
    import webbrowser

    target_url = f"http://localhost:{port}"
    check_url = f"http://{host}:{port}/api/health"
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            with urllib.request.urlopen(check_url, timeout=0.8) as resp:
                if resp.status == 200:
                    time.sleep(0.3)
                    print(f"\n[EuroSAT Studio] Backend ready! Opening browser at {target_url}\n")
                    webbrowser.open(target_url)
                    return
        except Exception:
            time.sleep(0.3)
    webbrowser.open(target_url)


if __name__ == "__main__":
    import threading
    import uvicorn

    host = os.getenv("HOST", "127.0.0.1")
    port = int(os.getenv("PORT", "8000"))

    # Launch browser automatically only once the server is alive and responding
    threading.Thread(target=_launch_browser_when_ready, args=(host, port), daemon=True).start()
    uvicorn.run(app, host=host, port=port)
