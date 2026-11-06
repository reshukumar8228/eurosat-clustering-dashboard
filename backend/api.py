"""
FastAPI Backend for EuroSAT Vision UI Dashboard
"""

import os
import sys
import json
import base64
import joblib
import numpy as np
import pandas as pd
from PIL import Image
import io

# Ensure project root is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fastapi import FastAPI, UploadFile, File, Form, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse

from classifier import EuroSATClassifier

app = FastAPI(title="EuroSAT Vision API", version="2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

OUTPUT_DIR = "pipeline_outputs"
DATA_DIR = "Data/EuroSAT"

# Mount static directories
os.makedirs("uploads", exist_ok=True)
app.mount("/static/data", StaticFiles(directory=DATA_DIR), name="data")
app.mount("/static/outputs", StaticFiles(directory=OUTPUT_DIR), name="outputs")

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

    # 1 sample image per class
    sample_images = []
    for cls in label_names:
        row = metadata_df[metadata_df["true_label"] == cls].iloc[0]
        sample_images.append({
            "class": cls,
            "filename": row["filename"],
            "url": f"/static/data/{cls}/{row['filename']}"
        })

    return {
        "total_images": len(metadata_df),
        "total_classes": len(label_names),
        "dimensions": "64 × 64 × 3",
        "features_per_image": 12288,
        "spectral_bands": 13,
        "class_breakdown": breakdown,
        "sample_images": sample_images
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
        if file is not None:
            contents = await file.read()
            img = Image.open(io.BytesIO(contents)).convert("RGB")
            source_name = file.filename
        elif preset_path:
            img = Image.open(preset_path).convert("RGB")
            source_name = os.path.basename(preset_path)
        else:
            return JSONResponse(status_code=400, content={"error": "No image provided"})

        # Run inference via EuroSATClassifier
        res = classifier.predict(img)

        # Supervised classification prediction (if trained)
        supervised_prediction = None
        supervised_probs = None
        if supervised_clf is not None:
            img_resized = img.resize((64, 64), Image.Resampling.BILINEAR)
            flat = np.array(img_resized, dtype=np.float32).reshape(1, -1)
            scaled = classifier.scaler.transform(flat)
            pca_feat = classifier.pca_model.transform(scaled)
            pred_id = int(supervised_clf.predict(pca_feat)[0])
            supervised_prediction = label_names[pred_id]
            if hasattr(supervised_clf, "predict_proba"):
                probs = supervised_clf.predict_proba(pca_feat)[0]
                supervised_probs = {label_names[i]: round(float(probs[i]) * 100, 1) for i in range(len(label_names))}

        # Format nearest neighbors with web-accessible URLs
        neighbors_formatted = []
        for n in res["nearest_neighbors"]:
            rel_path = n["filepath"].replace("\\", "/").split("Data/EuroSAT/")[-1]
            neighbors_formatted.append({
                "filename": n["filename"],
                "true_label": n["true_label"],
                "distance": round(float(n["distance"]), 2),
                "url": f"/static/data/{rel_path}"
            })

        # Base64 thumbnail for instant preview
        buffered = io.BytesIO()
        img.resize((128, 128)).save(buffered, format="JPEG")
        img_b64 = "data:image/jpeg;base64," + base64.b64encode(buffered.getvalue()).decode()

        return {
            "source_name": source_name,
            "thumbnail_b64": img_b64,
            "unsupervised": {
                "predicted_cluster": res["predicted_cluster"],
                "mapped_label": res["majority_label"],
                "purity_confidence": round(res["confidence_purity"] * 100, 1),
                "cluster_size": res["cluster_size"],
                "centroid_distances": [
                    {"cluster": k, "distance": round(v, 2)}
                    for k, v in sorted(res["centroid_distances"].items(), key=lambda x: x[1])
                ]
            },
            "supervised_semantic": {
                "predicted_label": supervised_prediction or res["majority_label"],
                "probabilities": supervised_probs
            },
            "nearest_neighbors": neighbors_formatted,
            "pca_coords": [round(res["pca_2d"][0], 2), round(res["pca_2d"][1], 2)],
            "physical_stats": {
                "brightness": round(res["image_stats"]["mean_brightness"], 1),
                "roughness": round(res["image_stats"]["texture_roughness"], 1),
                "red": round(res["image_stats"]["mean_red"], 1),
                "green": round(res["image_stats"]["mean_green"], 1),
                "blue": round(res["image_stats"]["mean_blue"], 1)
            }
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


# Mount built React frontend distribution (Vision UI)
if os.path.exists("frontend/dist"):
    app.mount("/", StaticFiles(directory="frontend/dist", html=True), name="frontend")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
