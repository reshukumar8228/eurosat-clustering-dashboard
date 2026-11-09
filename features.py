"""
Feature Extraction and Image Alignment Utilities for EuroSAT Vision Engine.
Provides multi-scale structural, edge, and color-space representations to enhance
classification accuracy beyond linear pixel PCA.
"""

import cv2
import numpy as np
from PIL import Image


def align_image_illumination(img_pil: Image.Image, target_mean: float = 96.0) -> Image.Image:
    """
    Gently aligns external/aerial imagery whose exposure differs from Sentinel-2
    satellite surface reflectance (mean ~96). Prevents pixel-level PCA from
    misclassifying dark patches as deep forest canopy or water.
    """
    arr = np.array(img_pil, dtype=np.float32)
    current_mean = float(arr.mean())
    if current_mean < 1.0 or current_mean > 250.0:
        return img_pil

    # If the image is substantially darker or brighter than EuroSAT normal range (75 - 135)
    if current_mean < 75.0 or current_mean > 135.0:
        factor = target_mean / current_mean
        # Clamp factor to avoid over-amplification
        factor = max(0.6, min(factor, 2.0))
        arr_aligned = np.clip(arr * factor, 0, 255).astype(np.uint8)
        return Image.fromarray(arr_aligned)

    return img_pil


def extract_rich_features(path_or_img) -> list:
    """
    Extract 24 discriminative structural, texture, edge, and color features from a 64x64 patch.
    """
    if isinstance(path_or_img, str):
        img_bgr = cv2.imread(path_or_img)
        if img_bgr is None:
            return [0.0] * 24
    elif isinstance(path_or_img, Image.Image):
        img_rgb = np.array(path_or_img.convert("RGB"))
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    elif isinstance(path_or_img, np.ndarray):
        if path_or_img.ndim == 2:
            img_bgr = cv2.cvtColor(path_or_img, cv2.COLOR_GRAY2BGR)
        elif path_or_img.shape[2] == 4:
            img_bgr = cv2.cvtColor(path_or_img, cv2.COLOR_RGBA2BGR)
        else:
            img_bgr = cv2.cvtColor(path_or_img[:, :, :3], cv2.COLOR_RGB2BGR)
    else:
        return [0.0] * 24

    if img_bgr.shape[:2] != (64, 64):
        img_bgr = cv2.resize(img_bgr, (64, 64), interpolation=cv2.INTER_LINEAR)

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    # 1. Edge & structure features
    edges = cv2.Canny(gray, 50, 150)
    edge_dens = float(np.mean(edges > 0))

    sobelx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(sobelx**2 + sobely**2)
    sobel_mean = float(np.mean(mag))
    sobel_std = float(np.std(mag))
    sobel_p90 = float(np.percentile(mag, 90))

    corners = cv2.goodFeaturesToTrack(gray, maxCorners=80, qualityLevel=0.01, minDistance=4)
    corner_count = float(len(corners)) if corners is not None else 0.0

    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=25, minLineLength=12, maxLineGap=4)
    line_count = float(len(lines)) if lines is not None else 0.0

    lap_var = float(cv2.Laplacian(gray, cv2.CV_32F).var())

    # 2. Color & vegetation metrics
    b = img_bgr[:, :, 0].astype(np.float32)
    g = img_bgr[:, :, 1].astype(np.float32)
    r = img_bgr[:, :, 2].astype(np.float32)
    mr, mg, mb = float(np.mean(r)), float(np.mean(g)), float(np.mean(b))
    sr, sg, sb = float(np.std(r)), float(np.std(g)), float(np.std(b))

    denom_gr = mg + mr + 1e-4
    gr_diff = float((mg - mr) / denom_gr)
    denom_gb = mg + mb + 1e-4
    gb_diff = float((mg - mb) / denom_gb)

    # 3. HSV representations
    h_m = float(np.mean(hsv[:, :, 0]))
    s_m = float(np.mean(hsv[:, :, 1]))
    v_m = float(np.mean(hsv[:, :, 2]))
    s_s = float(np.std(hsv[:, :, 1]))
    v_s = float(np.std(hsv[:, :, 2]))

    # 4. Spatial quadrant layout
    q1 = float(np.mean(gray[:32, :32]))
    q2 = float(np.mean(gray[:32, 32:]))
    q3 = float(np.mean(gray[32:, :32]))
    q4 = float(np.mean(gray[32:, 32:]))

    return [
        edge_dens, sobel_mean, sobel_std, sobel_p90, corner_count, line_count, lap_var,
        mr, mg, mb, sr, sg, sb, gr_diff, gb_diff,
        h_m, s_m, v_m, s_s, v_s,
        q1, q2, q3, q4
    ]


def detect_non_satellite(img_pil: Image.Image, scaler, pca, nn_model) -> tuple[bool, float, float, list[str]]:
    """
    Detects whether an uploaded image is Out-Of-Distribution (OOD), such as a non-satellite
    photo, smartphone screenshot, document, code editor, graphic, or random object.
    
    Returns:
        (is_ood: bool, score: float, min_dist: float, reasons: list[str])
    """
    arr_rgb = np.array(img_pil.convert("RGB"))
    h, w = arr_rgb.shape[:2]
    im_resized = img_pil.resize((64, 64), Image.Resampling.BILINEAR)
    flat = np.array(im_resized, dtype=np.float32).reshape(1, -1)
    scaled = scaler.transform(flat)
    pca_feat = pca.transform(scaled)

    # 1. Nearest neighbor Euclidean distance in PCA space (dimension safe)
    nn_in = getattr(nn_model, "n_features_in_", pca_feat.shape[1])
    pca_feat_nn = pca_feat[:, :nn_in]
    dists, _ = nn_model.kneighbors(pca_feat_nn, n_neighbors=3)
    min_dist = float(dists[0][0])

    # 2. PCA reconstruction residual
    recon = pca.inverse_transform(pca_feat)
    residual = float(np.linalg.norm(scaled - recon) / np.sqrt(scaled.shape[1]))

    # 3. Artificial border/bezel check (phone framing, letterboxing, app window)
    gray = cv2.cvtColor(arr_rgb, cv2.COLOR_RGB2GRAY)
    bw = max(2, int(min(h, w) * 0.08))
    border_mask = np.zeros_like(gray, dtype=bool)
    border_mask[:bw, :] = True
    border_mask[-bw:, :] = True
    border_mask[:, :bw] = True
    border_mask[:, -bw:] = True

    border_mean = float(np.mean(gray[border_mask]))
    center_mean = float(np.mean(gray[~border_mask]))
    is_framed = (border_mean < 25.0 and center_mean > 45.0) or (border_mean > 240.0 and center_mean < 210.0)

    # 4. Text/code screenshot line rhythm
    row_means = np.mean(gray, axis=1)
    row_diffs = float(np.std(np.diff(row_means))) if len(row_means) > 1 else 0.0

    score = 0
    reasons = []

    if min_dist >= 62.0:
        score += 50
        reasons.append(f"Nearest satellite patch distance ({min_dist:.1f}) is far outside dataset bounds (>=62.0)")
    elif min_dist >= 55.0:
        score += 25
        reasons.append(f"Distance to EuroSAT manifold ({min_dist:.1f}) is elevated")

    if residual >= 0.44:
        score += 25
        reasons.append(f"High reconstruction residual ({residual:.3f})")

    if is_framed:
        score += 35
        reasons.append("Device screen bezel or artificial frame detected")

    if row_diffs > 18.0 and min_dist > 40.0:
        score += 20
        reasons.append("Periodic horizontal line patterns characteristic of computer text/code")

    is_ood = score >= 50
    return is_ood, float(score), min_dist, reasons
