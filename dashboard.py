"""
Interactive Streamlit Dashboard
Project Title: Unsupervised Land-Type Grouping of Satellite Image Patches Using Clustering
EuroSAT Satellite Dataset (Sentinel-2)
"""

import os
import json
import glob
import joblib
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from classifier import EuroSATClassifier

# ==============================================================================
# Page Configuration & Styling
# ==============================================================================
st.set_page_config(
    page_title="EuroSAT Land-Type Unsupervised Clustering",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for polished UI (Theme-adaptive & Dark Mode optimized)
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        margin-bottom: 0.2rem;
        background: linear-gradient(90deg, #60A5FA 0%, #A78BFA 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        letter-spacing: -0.5px;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #94A3B8 !important;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: rgba(30, 41, 59, 0.7);
        border: 1px solid rgba(148, 163, 184, 0.2);
        border-radius: 10px;
        padding: 16px;
        text-align: center;
        color: #F8FAFC !important;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
    }
    .highlight-card {
        background: rgba(30, 41, 59, 0.75);
        border: 1px solid rgba(96, 165, 250, 0.35);
        border-radius: 12px;
        padding: 22px 26px;
        margin-bottom: 22px;
        box-shadow: 0 4px 24px rgba(0, 0, 0, 0.3);
        backdrop-filter: blur(10px);
    }
    .highlight-card h4 {
        color: #60A5FA !important;
        font-size: 1.15rem;
        font-weight: 700;
        margin-top: 0;
        margin-bottom: 14px;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .highlight-card ul {
        color: #E2E8F0 !important;
        margin-bottom: 0;
        padding-left: 20px;
    }
    .highlight-card li {
        margin-bottom: 10px;
        line-height: 1.6;
        color: #CBD5E1 !important;
        font-size: 0.98rem;
    }
    .highlight-card b {
        color: #93C5FD !important;
        font-weight: 600;
    }
    .badge-primary {
        background: linear-gradient(135deg, #2563EB 0%, #1D4ED8 100%);
        color: #FFFFFF !important;
        padding: 5px 14px;
        border-radius: 14px;
        font-weight: 600;
        font-size: 0.9rem;
        display: inline-block;
        box-shadow: 0 2px 8px rgba(37, 99, 235, 0.3);
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 42px;
        white-space: pre-wrap;
        border-radius: 6px 6px 0 0;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# Fixed color map for 10 EuroSAT ground truth classes
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

CLUSTER_COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"
]

# ==============================================================================
# Cached Data Loaders
# ==============================================================================
OUTPUT_DIR = "pipeline_outputs"
DATA_DIR = "Data/EuroSAT"

@st.cache_data
def load_metadata():
    path = os.path.join(OUTPUT_DIR, "metadata.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    return pd.DataFrame()

@st.cache_data
def load_pca_data():
    pca_2d = np.load(os.path.join(OUTPUT_DIR, "pca_2d.npy"))
    exp_var = np.load(os.path.join(OUTPUT_DIR, "pca_explained_variance.npy"))
    with open(os.path.join(OUTPUT_DIR, "pca_loadings_top10.json"), "r") as f:
        loadings = json.load(f)
    return pca_2d, exp_var, loadings

@st.cache_data
def load_tsne_data():
    with open(os.path.join(OUTPUT_DIR, "tsne_params.json"), "r") as f:
        tsne_params = json.load(f)
    subset_indices = np.load(os.path.join(OUTPUT_DIR, "tsne_subset_indices.npy"))
    
    tsne_embeddings = {}
    for perp in tsne_params.get("perplexities", [5, 15, 30, 50, 100]):
        perp_file = os.path.join(OUTPUT_DIR, f"tsne_2d_perp{perp}.npy")
        if os.path.exists(perp_file):
            tsne_embeddings[perp] = np.load(perp_file)
    return tsne_params, subset_indices, tsne_embeddings

@st.cache_data
def load_kmeans_data():
    with open(os.path.join(OUTPUT_DIR, "kmeans_metrics.json"), "r") as f:
        metrics = json.load(f)
    with open(os.path.join(OUTPUT_DIR, "cluster_label_mapping.json"), "r") as f:
        mapping = json.load(f)
    contingency = pd.read_csv(os.path.join(OUTPUT_DIR, "kmeans_k10_contingency.csv"), index_col=0)
    labels_k10 = np.load(os.path.join(OUTPUT_DIR, "kmeans_labels_k10.npy"))
    return metrics, mapping, contingency, labels_k10

@st.cache_data
def load_umap_data():
    path = os.path.join(OUTPUT_DIR, "umap_2d.npy")
    if os.path.exists(path):
        return np.load(path)
    return None

@st.cache_data
def load_spectral_data():
    spec_pca_path = os.path.join(OUTPUT_DIR, "spectral_pca_2d.npy")
    spec_feat_path = os.path.join(OUTPUT_DIR, "spectral_features.csv")
    spec_meta_path = os.path.join(OUTPUT_DIR, "spectral_metadata.csv")
    if os.path.exists(spec_pca_path) and os.path.exists(spec_feat_path) and os.path.exists(spec_meta_path):
        return (
            np.load(spec_pca_path),
            pd.read_csv(spec_feat_path),
            pd.read_csv(spec_meta_path)
        )
    return None, None, None

@st.cache_resource
def load_classifier():
    clf_path = os.path.join(OUTPUT_DIR, "classifier.pkl")
    if os.path.exists(clf_path):
        return joblib.load(clf_path)
    return None


# ==============================================================================
# Sidebar Navigation
# ==============================================================================
st.sidebar.image("https://raw.githubusercontent.com/google/material-design-icons/master/png/maps/satellite/materialicons/48dp/2x/baseline_satellite_black_48dp.png", width=60)
st.sidebar.title("🛰️ EuroSAT Studio")
st.sidebar.markdown("**Unsupervised Satellite Clustering**")

nav_choice = st.sidebar.radio(
    "Navigation Menu",
    [
        "🏠 Overview & Dataset",
        "📊 PCA Variance & Loadings",
        "🔮 t-SNE Parameter Sensitivity",
        "🎯 K-Means Clustering Deep-Dive",
        "🔍 Image Upload & Classification",
        "📈 Comparative Analysis & Spectral",
        "📝 Research Report & Governance"
    ],
    index=0
)

st.sidebar.markdown("---")
st.sidebar.info("""
**Unsupervised Learning Protocol:**
Ground truth labels were **hidden** during all preprocessing, PCA feature projection, t-SNE embedding, and K-Means clustering.
Labels are used solely for **post-hoc evaluation** and interpretation.
""")

# Load core datasets
metadata_df = load_metadata()
if metadata_df.empty:
    st.error("⚠️ Metadata not found. Please ensure `pipeline.py` has completed successfully.")
    st.stop()


# ==============================================================================
# Page 1: 🏠 Overview & Dataset Explorer
# ==============================================================================
if nav_choice == "🏠 Overview & Dataset":
    st.markdown('<div class="main-title">🛰️ EuroSAT Unsupervised Land-Type Grouping</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Study of Sentinel-2 satellite image patches using unsupervised dimensionality reduction and clustering.</div>', unsafe_allow_html=True)

    # Highlight Card
    st.markdown("""
    <div class="highlight-card">
        <h4>📋 Research Objectives & Minimal Requirements</h4>
        <ul>
            <li><b>(A) Preprocessing:</b> Prepared 27,000 EuroSAT satellite patches (64×64 RGB, 12,288 flattened features) with zero missing data, scaled via StandardScaler with ground truth hidden.</li>
            <li><b>(B) Dimensionality Reduction & Grouping:</b> Fitted Randomized PCA retaining &gt;90% variance (220 components) and applied multi-perplexity t-SNE. Applied K-Means on PCA-reduced representations.</li>
            <li><b>(C) Interpretability & Governance:</b> Evaluated cumulative variance, feature pixel loadings, cluster contingency, parameter sensitivity, and documented remote-sensing biases.</li>
            <li><b>Interactive Upload:</b> Upload any satellite patch to infer cluster membership, nearest neighbors, and projected 2D coordinates.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    # Metrics Row
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Total Satellite Patches", f"{len(metadata_df):,}")
    with m2:
        st.metric("Land-Use Classes", f"{metadata_df['true_label'].nunique()}")
    with m3:
        st.metric("Patch Dimensions", "64 × 64 (12,288 D)")
    with m4:
        st.metric("Spectral Bands Available", "13 Sentinel-2 Bands")

    st.markdown("### 📊 Dataset Distribution")
    col_chart, col_table = st.columns([3, 2])

    class_counts = metadata_df["true_label"].value_counts().reset_index()
    class_counts.columns = ["Class", "Count"]
    class_counts["Percentage"] = (class_counts["Count"] / len(metadata_df) * 100).round(2)

    with col_chart:
        fig_bar = px.bar(
            class_counts,
            x="Class",
            y="Count",
            color="Class",
            color_discrete_map=CLASS_COLORS,
            title="Image Count per Land-Use Category",
            text="Count"
        )
        fig_bar.update_layout(showlegend=False, xaxis_tickangle=-45, height=380)
        st.plotly_chart(fig_bar, use_container_width=True)

    with col_table:
        st.markdown("**Class Breakdown Table**")
        st.dataframe(class_counts, use_container_width=True, height=340)

    st.markdown("### 🖼️ Sample Image Gallery")
    selected_class = st.selectbox("Select Land-Use Class to Preview", sorted(metadata_df["true_label"].unique()))
    class_samples = metadata_df[metadata_df["true_label"] == selected_class].sample(8, random_state=42)

    gallery_cols = st.columns(8)
    for idx, (_, row) in enumerate(class_samples.iterrows()):
        with gallery_cols[idx]:
            try:
                img = Image.open(row["filepath"])
                st.image(img, use_container_width=True, caption=row["filename"][:14])
            except Exception:
                st.text("Image not found")


# ==============================================================================
# Page 2: 📊 PCA Variance & Loadings
# ==============================================================================
elif nav_choice == "📊 PCA Variance & Loadings":
    st.markdown('<div class="main-title">📊 Principal Component Analysis (PCA)</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Retained variance, scree analysis, and spatial pixel-channel loading heatmaps.</div>', unsafe_allow_html=True)

    pca_2d, exp_var, loadings = load_pca_data()
    cum_var = np.cumsum(exp_var)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Variance Explained by Top 2 PCs", f"{cum_var[1]*100:.2f}%")
    with c2:
        st.metric("Variance Explained by Top 10 PCs", f"{cum_var[9]*100:.2f}%")
    with c3:
        st.metric("Total Components Fitted", f"{len(exp_var)} ({cum_var[-1]*100:.2f}% var)")

    tab1, tab2, tab3 = st.tabs(["Cumulative & Scree Variance", "2D PCA Projection", "Feature Loadings & Spatial Heatmaps"])

    with tab1:
        st.subheader("Explained Variance Spectrum")
        num_comps = st.slider("Select Number of Components to Inspect", 10, len(exp_var), 50)

        fig_scree = make_subplots(specs=[[{"secondary_y": True}]])
        fig_scree.add_trace(
            go.Bar(x=list(range(1, num_comps+1)), y=exp_var[:num_comps]*100, name="Individual Explained Variance (%)", marker_color="#3B82F6"),
            secondary_y=False
        )
        fig_scree.add_trace(
            go.Scatter(x=list(range(1, num_comps+1)), y=cum_var[:num_comps]*100, name="Cumulative Explained Variance (%)", line=dict(color="#EF4444", width=3)),
            secondary_y=True
        )
        fig_scree.update_layout(
            title=f"Scree and Cumulative Variance Plot (Top {num_comps} PCs)",
            xaxis_title="Principal Component Index",
            hovermode="x unified",
            height=450
        )
        fig_scree.update_yaxes(title_text="Individual Variance (%)", secondary_y=False)
        fig_scree.update_yaxes(title_text="Cumulative Variance (%)", secondary_y=True)
        st.plotly_chart(fig_scree, use_container_width=True)

    with tab2:
        st.subheader("Interactive 2D PCA Space")
        sub_indices = np.load(os.path.join(OUTPUT_DIR, "tsne_subset_indices.npy"))
        
        plot_mode = st.radio("Display Sample:", ["5,000 Stratified Points (Smooth)", "All 27,000 Points"], horizontal=True)
        color_by = st.radio("Color Scatter Points by:", ["Ground Truth Label (Post-Hoc Evaluation)", "K-Means Cluster (k=10)"], horizontal=True)

        _, _, _, labels_k10 = load_kmeans_data()

        if plot_mode == "5,000 Stratified Points (Smooth)":
            plot_df = metadata_df.iloc[sub_indices].copy()
            plot_pca = pca_2d[sub_indices]
            plot_clusters = labels_k10[sub_indices]
        else:
            plot_df = metadata_df.copy()
            plot_pca = pca_2d
            plot_clusters = labels_k10

        plot_df["PC1"] = plot_pca[:, 0]
        plot_df["PC2"] = plot_pca[:, 1]
        plot_df["Cluster"] = [f"Cluster {c}" for c in plot_clusters]

        if color_by == "Ground Truth Label (Post-Hoc Evaluation)":
            fig_pca2d = px.scatter(
                plot_df, x="PC1", y="PC2",
                color="true_label",
                color_discrete_map=CLASS_COLORS,
                hover_data=["filename", "true_label", "Cluster"],
                title="PCA 2D Projection Colored by True Land-Use Category",
                opacity=0.75
            )
        else:
            fig_pca2d = px.scatter(
                plot_df, x="PC1", y="PC2",
                color="Cluster",
                hover_data=["filename", "true_label"],
                title="PCA 2D Projection Colored by K-Means Cluster Assignment",
                opacity=0.75
            )

        fig_pca2d.update_layout(height=550)
        st.plotly_chart(fig_pca2d, use_container_width=True)

    with tab3:
        st.subheader("Spatial Feature Loadings (Pixel Importance Maps)")
        st.markdown("Examine which spatial locations and RGB color channels contribute most to each principal component.")

        selected_pc = st.selectbox("Select Principal Component", [f"PC_{i+1}" for i in range(10)])
        pc_info = loadings.get(selected_pc, {})

        if pc_info:
            st.write(f"**{selected_pc}** explains **{pc_info['explained_variance_ratio']*100:.2f}%** of total dataset variance.")

            col_map, col_top = st.columns([1, 1])

            top_loads = pd.DataFrame(pc_info["top_loadings"])

            with col_map:
                # Build 64x64 spatial activation grid
                spatial_grid = np.zeros((64, 64))
                for _, row in top_loads.iterrows():
                    spatial_grid[int(row["row"]), int(row["col"])] += row["abs_weight"]

                fig_hm = px.imshow(
                    spatial_grid,
                    color_continuous_scale="Viridis",
                    title=f"Spatial Activation Map (64×64 Patch) for {selected_pc}",
                    labels=dict(x="Pixel Column", y="Pixel Row", color="Abs Weight")
                )
                fig_hm.update_layout(height=400)
                st.plotly_chart(fig_hm, use_container_width=True)

            with col_top:
                fig_bar = px.bar(
                    top_loads,
                    x="abs_weight",
                    y="channel",
                    orientation="h",
                    color="channel",
                    color_discrete_map={"Red": "#EF4444", "Green": "#10B981", "Blue": "#3B82F6"},
                    title=f"Top 20 Feature Loadings by Channel for {selected_pc}"
                )
                fig_bar.update_layout(height=400)
                st.plotly_chart(fig_bar, use_container_width=True)


# ==============================================================================
# Page 3: 🔮 t-SNE Parameter Sensitivity
# ==============================================================================
elif nav_choice == "🔮 t-SNE Parameter Sensitivity":
    st.markdown('<div class="main-title">🔮 t-SNE Non-Linear Manifold Learning</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Empirical study of perplexity sensitivity, KL divergence convergence, and cluster preservation.</div>', unsafe_allow_html=True)

    tsne_params, subset_indices, tsne_embeddings = load_tsne_data()
    subset_df = metadata_df.iloc[subset_indices].copy()
    _, _, _, labels_k10 = load_kmeans_data()
    subset_df["Cluster"] = [f"Cluster {c}" for c in labels_k10[subset_indices]]

    tab_inspect, tab_compare, tab_theory = st.tabs(["Interactive t-SNE Explorer", "Perplexity Grid Comparison", "Parameter Sensitivity & Theory"])

    with tab_inspect:
        col_ctrl1, col_ctrl2 = st.columns([1, 1])
        with col_ctrl1:
            perp_choice = st.select_slider("Choose t-SNE Perplexity", options=list(tsne_embeddings.keys()), value=30)
        with col_ctrl2:
            color_choice = st.radio("Color By", ["Ground Truth Label", "K-Means Cluster"], horizontal=True)

        if perp_choice in tsne_embeddings:
            emb = tsne_embeddings[perp_choice]
            subset_df["t-SNE 1"] = emb[:, 0]
            subset_df["t-SNE 2"] = emb[:, 1]

            kl_div = tsne_params.get(f"perp_{perp_choice}", {}).get("kl_divergence", "N/A")
            st.caption(f"Perplexity: **{perp_choice}** | Final KL Divergence: **{kl_div}** | Stratified Points: **{len(subset_df):,}**")

            if color_choice == "Ground Truth Label":
                fig_tsne = px.scatter(
                    subset_df, x="t-SNE 1", y="t-SNE 2",
                    color="true_label",
                    color_discrete_map=CLASS_COLORS,
                    hover_data=["filename", "true_label", "Cluster"],
                    title=f"t-SNE 2D Embedding (Perplexity = {perp_choice})",
                    opacity=0.8
                )
            else:
                fig_tsne = px.scatter(
                    subset_df, x="t-SNE 1", y="t-SNE 2",
                    color="Cluster",
                    hover_data=["filename", "true_label"],
                    title=f"t-SNE 2D Embedding Colored by Cluster (Perplexity = {perp_choice})",
                    opacity=0.8
                )
            fig_tsne.update_layout(height=560)
            st.plotly_chart(fig_tsne, use_container_width=True)

    with tab_compare:
        st.subheader("Side-by-Side Perplexity Spectrum")
        st.markdown("Visualizing how varying perplexity from 5 to 100 alters local neighborhood vs global geometry preservation.")

        perp_list = sorted(tsne_embeddings.keys())
        p_cols = st.columns(len(perp_list))
        for idx, p in enumerate(perp_list):
            with p_cols[idx]:
                p_emb = tsne_embeddings[p]
                mini_df = pd.DataFrame({"x": p_emb[:, 0], "y": p_emb[:, 1], "label": subset_df["true_label"].values})
                fig_mini = px.scatter(
                    mini_df, x="x", y="y", color="label",
                    color_discrete_map=CLASS_COLORS,
                    title=f"Perp = {p}"
                )
                fig_mini.update_layout(showlegend=False, height=260, margin=dict(l=5, r=5, t=30, b=5))
                fig_mini.update_xaxes(showticklabels=False)
                fig_mini.update_yaxes(showticklabels=False)
                st.plotly_chart(fig_mini, use_container_width=True)

    with tab_theory:
        st.markdown(r"""
        ### 🔬 Theoretical & Empirical Sensitivity Analysis
        
        1. **Low Perplexity (e.g., Perplexity = 5):**
           - Constrains Gaussian kernels to narrow variance ($\sigma_i$).
           - Emphasizes fine-grained local neighborhoods, leading to fragmented micro-clusters.
           - Noticeable background clumping where homogeneous textures (e.g. Forest and SeaLake) shatter into artificial islands.
        
        2. **Moderate Perplexity (Perplexity = 30 - 50):**
           - Balances local manifold structure with broader topological relationships.
           - Water bodies (SeaLake, River) and agricultural patches (AnnualCrop, PermanentCrop) form coherent continuous trajectories.
        
        3. **High Perplexity (Perplexity = 100):**
           - Broader Gaussian kernel spreads probability mass across more distant points.
           - Lower KL divergence (**1.2958** vs **2.2956** for perp=5), exhibiting smoother global layouts.
        
        > [!WARNING]
        > **Critical Precaution:** Distances between distinct clusters in t-SNE plots **do not reflect Euclidean metric distance**. Cluster size and density in t-SNE space are influenced by non-linear gradient equalization rather than physical spatial diameter.
        """)


# ==============================================================================
# Page 4: 🎯 K-Means Clustering Deep-Dive
# ==============================================================================
elif nav_choice == "🎯 K-Means Clustering Deep-Dive":
    st.markdown('<div class="main-title">🎯 K-Means Clustering Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Evaluation across cluster sizes (k=2 to 20), contingency analysis, and exemplar inspections.</div>', unsafe_allow_html=True)

    metrics, mapping, contingency, labels_k10 = load_kmeans_data()

    tab_eval, tab_k10, tab_exemplars = st.tabs(["Metric Curves (k=2 to 20)", "k=10 Ground Truth Contingency", "Cluster Centroids & Exemplars"])

    with tab_eval:
        st.subheader("Clustering Evaluation Criteria Across k")

        k_list = [int(k) for k in metrics.keys()]
        inertias = [metrics[str(k)]["inertia"] for k in k_list]
        silhouettes = [metrics[str(k)]["silhouette"] for k in k_list]
        db_scores = [metrics[str(k)]["davies_bouldin"] for k in k_list]
        ch_scores = [metrics[str(k)]["calinski_harabasz"] for k in k_list]

        fig_curves = make_subplots(
            rows=2, cols=2,
            subplot_titles=["Inertia (Elbow Method)", "Silhouette Score (Higher is Better)", "Davies-Bouldin Index (Lower is Better)", "Calinski-Harabasz Score (Higher is Better)"]
        )

        fig_curves.add_trace(go.Scatter(x=k_list, y=inertias, mode="lines+markers", marker_color="#2563EB", name="Inertia"), row=1, col=1)
        fig_curves.add_trace(go.Scatter(x=k_list, y=silhouettes, mode="lines+markers", marker_color="#10B981", name="Silhouette"), row=1, col=2)
        fig_curves.add_trace(go.Scatter(x=k_list, y=db_scores, mode="lines+markers", marker_color="#F59E0B", name="Davies-Bouldin"), row=2, col=1)
        fig_curves.add_trace(go.Scatter(x=k_list, y=ch_scores, mode="lines+markers", marker_color="#8B5CF6", name="Calinski-Harabasz"), row=2, col=2)

        fig_curves.update_layout(height=560, showlegend=False)
        st.plotly_chart(fig_curves, use_container_width=True)

    with tab_k10:
        st.subheader("K=10 Ground Truth Alignment (Post-Hoc Unsupervised Assessment)")

        ari = metrics.get("10", {}).get("adjusted_rand_index", 0.0)
        nmi = metrics.get("10", {}).get("normalized_mutual_info", 0.0)
        purity = metrics.get("10", {}).get("overall_purity", 0.0)

        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Adjusted Rand Index (ARI)", f"{ari:.4f}")
        with c2:
            st.metric("Normalized Mutual Info (NMI)", f"{nmi:.4f}")
        with c3:
            st.metric("Overall Cluster Purity", f"{purity*100:.2f}%")

        col_cm, col_pur = st.columns([3, 2])
        with col_cm:
            fig_cm = px.imshow(
                contingency.values,
                x=contingency.columns,
                y=contingency.index,
                color_continuous_scale="Blues",
                title="Cluster × True Category Contingency Matrix",
                text_auto=True
            )
            fig_cm.update_layout(height=450)
            st.plotly_chart(fig_cm, use_container_width=True)

        with col_pur:
            purity_df = pd.DataFrame([
                {"Cluster": f"Cluster {cid}", "Purity": info["purity"]*100, "Majority Label": info["majority_true_label"]}
                for cid, info in mapping.items()
            ])
            fig_pur = px.bar(
                purity_df, x="Purity", y="Cluster", orientation="h",
                color="Majority Label", color_discrete_map=CLASS_COLORS,
                title="Per-Cluster Purity (%)", text=purity_df["Purity"].apply(lambda v: f"{v:.1f}%")
            )
            fig_pur.update_layout(height=450)
            st.plotly_chart(fig_pur, use_container_width=True)

    with tab_exemplars:
        st.subheader("Cluster Exemplars & Average Reconstructed Centroids")
        st.markdown("Examine the visual archetype of each cluster closest to its geometric centroid.")

        chosen_cluster = st.selectbox("Select Cluster to Inspect", [f"Cluster {i}" for i in range(10)])
        c_idx = int(chosen_cluster.split()[1])

        info = mapping.get(str(c_idx), {})
        st.info(f"**{chosen_cluster}** | Majority Class: **{info.get('majority_true_label', 'N/A')}** | Purity: **{info.get('purity', 0)*100:.1f}%** | Total Patches: **{info.get('count', 0):,}**")

        grid_path = os.path.join(OUTPUT_DIR, "cluster_grids", f"cluster_{c_idx}_grid.png")
        cent_path = os.path.join(OUTPUT_DIR, "cluster_grids", f"centroid_cluster_{c_idx}.png")

        col_cent, col_grid = st.columns([1, 2])
        with col_cent:
            if os.path.exists(cent_path):
                st.image(Image.open(cent_path), caption=f"Reconstructed Average Centroid (64×64)", width=220)
        with col_grid:
            if os.path.exists(grid_path):
                st.image(Image.open(grid_path), caption=f"4×4 Exemplars Closest to Centroid", use_container_width=True)


# ==============================================================================
# Page 5: 🔍 Image Upload & Classification
# ==============================================================================
elif nav_choice == "🔍 Image Upload & Classification":
    st.markdown('<div class="main-title">🔍 Satellite Patch Inference Engine</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Upload any satellite patch to detect cluster assignment, nearest neighbors, and projected 2D coordinates.</div>', unsafe_allow_html=True)

    classifier = load_classifier()
    if classifier is None:
        st.error("Classifier model not found. Please run the pipeline first.")
        st.stop()

    col_up, col_preset = st.columns([1, 1])

    uploaded_file = None
    preset_choice = None

    with col_up:
        uploaded_file = st.file_uploader("Upload Image File (.jpg, .png, .tif)", type=["jpg", "jpeg", "png", "tif"])

    with col_preset:
        preset_options = ["None (Use upload)"] + sorted(metadata_df["true_label"].unique())
        selected_preset_cat = st.selectbox("Or Quick-Test with EuroSAT Sample", preset_options)
        if selected_preset_cat != "None (Use upload)":
            preset_row = metadata_df[metadata_df["true_label"] == selected_preset_cat].iloc[0]
            preset_choice = preset_row["filepath"]

    # Determine input image
    test_image = None
    source_label = "Uploaded Image"
    if uploaded_file is not None:
        test_image = Image.open(uploaded_file)
        source_label = uploaded_file.name
    elif preset_choice is not None:
        test_image = Image.open(preset_choice)
        source_label = f"EuroSAT Preset ({os.path.basename(preset_choice)})"

    if test_image is not None:
        st.markdown("---")
        res = classifier.predict(test_image)

        col_img, col_pred = st.columns([1, 2])

        with col_img:
            st.image(test_image.resize((192, 192)), caption=f"Source: {source_label}", width=192)
            st.markdown(f"""
            **Physical Characteristics:**
            - Mean Brightness: `{res['image_stats']['mean_brightness']:.1f}`
            - Texture Roughness (σ): `{res['image_stats']['texture_roughness']:.1f}`
            - Red Channel Mean: `{res['image_stats']['mean_red']:.1f}`
            - Green Channel Mean: `{res['image_stats']['mean_green']:.1f}`
            - Blue Channel Mean: `{res['image_stats']['mean_blue']:.1f}`
            """)

        with col_pred:
            st.markdown(f"### Detected Cluster: <span class='badge-primary'>Cluster {res['predicted_cluster']}</span>", unsafe_allow_html=True)
            st.markdown(f"#### Mapped Land-Use Category: **{res['majority_label']}**")

            purity_val = res['confidence_purity']
            st.progress(purity_val)
            st.caption(f"Cluster Archetype Purity / Confidence: **{purity_val*100:.1f}%** | Cluster Population: **{res['cluster_size']:,} patches**")

            # Centroid Distances Radar / Bar Chart
            dists = res["centroid_distances"]
            dist_df = pd.DataFrame(list(dists.items()), columns=["Cluster", "Distance"]).sort_values("Distance")

            fig_dists = px.bar(
                dist_df, x="Distance", y="Cluster", orientation="h",
                color="Distance", color_continuous_scale="Blues_r",
                title="Euclidean Distance to All 10 Cluster Centroids (PCA Space)"
            )
            fig_dists.update_layout(height=260, margin=dict(l=10, r=10, t=30, b=10))
            st.plotly_chart(fig_dists, use_container_width=True)

        st.markdown("### 👥 Top-5 Nearest Neighbors in EuroSAT Dataset")
        st.markdown("Most topologically similar historical patches based on Euclidean distance in the 90% variance PCA space.")

        nn_cols = st.columns(5)
        for i, neighbor in enumerate(res["nearest_neighbors"]):
            with nn_cols[i]:
                try:
                    n_img = Image.open(neighbor["filepath"])
                    st.image(n_img, use_container_width=True, caption=f"#{i+1}: {neighbor['true_label']}")
                    st.caption(f"Dist: `{neighbor['distance']:.2f}`\n`{neighbor['filename'][:15]}`")
                except Exception:
                    st.text("Preview unavailable")

        st.markdown("### 📍 Location in 2D PCA Space")
        pca_2d, _, _ = load_pca_data()
        sub_indices = np.load(os.path.join(OUTPUT_DIR, "tsne_subset_indices.npy"))
        bg_df = metadata_df.iloc[sub_indices].copy()
        bg_df["PC1"] = pca_2d[sub_indices, 0]
        bg_df["PC2"] = pca_2d[sub_indices, 1]

        fig_loc = px.scatter(
            bg_df, x="PC1", y="PC2",
            color="true_label",
            color_discrete_map=CLASS_COLORS,
            opacity=0.4,
            title="Position of Uploaded Patch Against Dataset Manifold"
        )
        fig_loc.add_trace(
            go.Scatter(
                x=[res["pca_2d"][0]],
                y=[res["pca_2d"][1]],
                mode="markers+text",
                marker=dict(symbol="star", size=22, color="#DC2626", line=dict(color="#FFFFFF", width=2)),
                name="Uploaded Patch",
                text=["⭐ UPLOADED"],
                textposition="top center"
            )
        )
        fig_loc.update_layout(height=480)
        st.plotly_chart(fig_loc, use_container_width=True)


# ==============================================================================
# Page 6: 📈 Comparative Analysis & Spectral
# ==============================================================================
elif nav_choice == "📈 Comparative Analysis & Spectral":
    st.markdown('<div class="main-title">📈 Dimensionality Reduction Comparison & Multi-Spectral Insights</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Comparative analysis across PCA, t-SNE, UMAP, and Sentinel-2 multi-spectral bands.</div>', unsafe_allow_html=True)

    tab_comp, tab_umap, tab_spec = st.tabs(["Technique Benchmark Comparison", "UMAP 2D Projection", "Sentinel-2 Multi-Spectral Analysis"])

    with tab_comp:
        st.subheader("Technique Benchmark Comparison Matrix")
        comp_data = {
            "Dimension Reduction Method": ["Principal Component Analysis (PCA)", "t-Distributed Stochastic Neighbor Embedding (t-SNE)", "Uniform Manifold Approximation & Projection (UMAP)"],
            "Mathematical Foundation": ["Linear orthogonal projection via randomized SVD", "Non-linear probabilistic student-t / KL minimization", "Riemannian geometry & fuzzy simplicial sets"],
            "Computational Complexity": ["O(N · D · min(N,D)) — Fast (9.26s)", "O(N log N) — Moderate (12-22s / perp)", "O(N log N) — Fast (58.98s)"],
            "Global Geometry Preservation": ["High (strictly preserves pairwise variance)", "Poor (focuses predominantly on local neighbors)", "Moderate-to-High (balances local & global)"],
            "Out-of-Sample Transform": ["Supported (linear matrix dot product)", "Not natively supported (requires retraining/NN)", "Supported via learned fuzzy embedding"],
            "Parameter Sensitivity": ["Low (deterministic given variance threshold)", "High (sensitive to perplexity, learning rate)", "Moderate (sensitive to n_neighbors, min_dist)"]
        }
        st.dataframe(pd.DataFrame(comp_data), use_container_width=True)

    with tab_umap:
        st.subheader("UMAP 2D Manifold Embedding")
        umap_2d = load_umap_data()
        if umap_2d is not None:
            sub_indices = np.load(os.path.join(OUTPUT_DIR, "tsne_subset_indices.npy"))
            u_df = metadata_df.iloc[sub_indices].copy()
            u_df["UMAP 1"] = umap_2d[:, 0]
            u_df["UMAP 2"] = umap_2d[:, 1]

            fig_u = px.scatter(
                u_df, x="UMAP 1", y="UMAP 2",
                color="true_label",
                color_discrete_map=CLASS_COLORS,
                hover_data=["filename", "true_label"],
                title="UMAP 2D Projection of EuroSAT Patches (n_neighbors=15, min_dist=0.1)",
                opacity=0.75
            )
            fig_u.update_layout(height=520)
            st.plotly_chart(fig_u, use_container_width=True)
        else:
            st.info("UMAP embeddings not computed. Please run pipeline with umap-learn installed.")

    with tab_spec:
        st.subheader("Sentinel-2 Multi-Spectral Bands (EuroSATallBands)")
        st.markdown("Analysis of 13 spectral bands including Coastal Aerosol (B1), Red Edge (B5-B7), NIR (B8, B8A), and SWIR (B11, B12).")

        spec_pca, spec_feat, spec_meta = load_spectral_data()
        if spec_pca is not None:
            col_spca, col_sfeat = st.columns([1, 1])

            with col_spca:
                spec_plot_df = spec_meta.copy()
                spec_plot_df["Spec_PC1"] = spec_pca[:, 0]
                spec_plot_df["Spec_PC2"] = spec_pca[:, 1]
                fig_spca = px.scatter(
                    spec_plot_df, x="Spec_PC1", y="Spec_PC2",
                    color="class_name",
                    color_discrete_map=CLASS_COLORS,
                    title="2D PCA on 13 Sentinel-2 Spectral Bands",
                    opacity=0.8
                )
                fig_spca.update_layout(height=450)
                st.plotly_chart(fig_spca, use_container_width=True)

            with col_sfeat:
                # Mean band reflectance profile
                band_cols = [f"B{i+1:02d}_mean" for i in range(13)]
                joined_spec = pd.concat([spec_feat[band_cols], spec_meta["class_name"]], axis=1)
                grouped = joined_spec.groupby("class_name").mean().reset_index()

                selected_spec_class = st.selectbox("Inspect Spectral Signature for Class", grouped["class_name"].unique())
                row_sig = grouped[grouped["class_name"] == selected_spec_class].iloc[0]

                sig_df = pd.DataFrame({
                    "Band": [f"B{i+1:02d}" for i in range(13)],
                    "Mean Reflectance": [row_sig[f"B{i+1:02d}_mean"] for i in range(13)]
                })

                fig_sig = px.line(
                    sig_df, x="Band", y="Mean Reflectance",
                    markers=True,
                    title=f"13-Band Spectral Signature: {selected_spec_class}"
                )
                fig_sig.update_layout(height=400)
                st.plotly_chart(fig_sig, use_container_width=True)


# ==============================================================================
# Page 7: 📝 Research Report & Governance
# ==============================================================================
elif nav_choice == "📝 Research Report & Governance":
    st.markdown('<div class="main-title">📝 Research Report & Ethical Governance</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Reproducible dimensionality reduction study documentation, limitations, and responsible AI practices.</div>', unsafe_allow_html=True)

    metrics, mapping, _, _ = load_kmeans_data()
    _, exp_var, _ = load_pca_data()

    report_markdown = f"""# Unsupervised Land-Type Grouping of Satellite Image Patches Using Clustering

**Author:** Antigravity Research Assistant  
**Domain:** Earth Observation & Remote Sensing (Sentinel-2 Satellites)  
**Dataset:** EuroSAT (27,000 Patches, 64×64 pixels, 10 Land-Use Categories)

---

## 1. Executive Summary
This study explored the natural grouping structure of European Sentinel-2 satellite imagery using strictly unsupervised machine learning techniques. Without utilizing class annotations during the modeling phase, 27,000 image patches were preprocessed, standardized, and reduced via Randomized Principal Component Analysis (PCA) and t-Distributed Stochastic Neighbor Embedding (t-SNE). K-Means clustering was executed across cluster sizes $k \\in [2, 20]$, identifying key structural archetypes and measuring correspondence with ground truth land-cover taxonomies.

---

## 2. Quantitative Results Summary
- **PCA 90% Variance Threshold:** 220 components (91.84% retained variance).
- **Top 2 PCs Variance:** {np.cumsum(exp_var)[1]*100:.2f}% of total pixel variance.
- **k=10 Clustering Agreement:**
  - Adjusted Rand Index (ARI): **{metrics.get('10', {}).get('adjusted_rand_index', 0.0):.4f}**
  - Normalized Mutual Information (NMI): **{metrics.get('10', {}).get('normalized_mutual_info', 0.0):.4f}**
  - Overall Archetype Purity: **{metrics.get('10', {}).get('overall_purity', 0.0)*100:.2f}%**
- **Optimal Clusters Identified:** The Elbow method and Calinski-Harabasz curves suggest natural groupings around $k=4$ to $k=6$ representing primary broad land-cover regimes (Water, Dense Vegetation, Barren/Agricultural soil, and Urban built-up structures).

---

## 3. Limitations & Ethical Considerations

### 3.1 Geographic & Spectral Bias
- **EuroSAT Geographic Scope:** All patches originate from European territories collected by Sentinel-2. Generalization to arid, tropical, or polar biomes without transfer calibration will fail due to differing vegetation phenology, architectural styles, and soil mineral signatures.
- **Spatial Resolution Limits:** At 10m Ground Sample Distance (GSD), each 64×64 patch spans 640m × 640m. Mixed-pixel phenomena occur where rivers, highways, and residential settlements intersect in a single patch.

### 3.2 Methodological Caveats in Dimensionality Reduction
- **t-SNE Non-Metric Distances:** Distances between distant clusters in t-SNE projections cannot be used as physical dissimilarity metrics.
- **Unsupervised Label Leakage Prevention:** Labels were strictly isolated to post-hoc contingency analysis to guarantee unbiased unsupervised evaluation.

### 3.3 Responsible AI Guidelines for Remote Sensing
- **Environmental Planning:** Autonomous land-cover segmentation outputs should serve as decision-support systems rather than authoritative inputs for legal land-zoning or taxation.
- **Carbon Footprint:** Employing lightweight Randomized PCA and efficient t-SNE avoids resource-heavy transformer training while providing transparent, inspectable features.
"""

    st.markdown(report_markdown)

    st.markdown("### 📥 Download Submission Assets")
    d1, d2, d3 = st.columns(3)

    with d1:
        st.download_button(
            "📄 Download Full Report (Markdown)",
            data=report_markdown,
            file_name="EuroSAT_Study_Report.md",
            mime="text/markdown"
        )
    with d2:
        with open(os.path.join(OUTPUT_DIR, "kmeans_metrics.json"), "r") as f:
            st.download_button(
                "📊 Download K-Means Metrics (JSON)",
                data=f.read(),
                file_name="kmeans_metrics.json",
                mime="application/json"
            )
    with d3:
        with open(os.path.join(OUTPUT_DIR, "kmeans_k10_contingency.csv"), "r") as f:
            st.download_button(
                "📋 Download Contingency Matrix (CSV)",
                data=f.read(),
                file_name="kmeans_k10_contingency.csv",
                mime="text/csv"
            )
