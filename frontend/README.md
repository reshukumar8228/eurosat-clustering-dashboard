# EuroSAT Studio — Frontend

Interactive web dashboard for unsupervised land-cover clustering and analysis of Sentinel-2 satellite patches. Built with React 19, Vite, and Tailwind CSS.

## Features

- **Overview & Dataset Explorer:** Dataset statistics, class distribution, and interactive patch lightbox.
- **PCA Variance & Loadings:** Cumulative scree curve, 2D PCA scatter manifold, and spatial pixel loading heatmaps.
- **t-SNE Manifold:** Multi-perplexity manifold learning comparisons ($p \in [5, 100]$) with live KL divergence metrics.
- **K-Means Clustering:** Inertia elbow analysis, silhouette scores, per-cluster purity, and centroid exemplar grids.
- **Experiment Bench:** Interactive parameter playground with precomputed and on-demand model fitting.
- **Patch Inference Engine:** Drag-and-drop satellite patch classification with nearest-neighbor retrieval.
- **Spectral Band Analysis:** 13-band Sentinel-2 signature profiles and UMAP manifold visualization.
- **Research Report:** Formal methodology, ethical governance, and reproducibility data exports.

## Development

```bash
npm install
npm run dev
```

## Production Build

```bash
npm run build
```

Built assets are bundled into `dist/` and served automatically by the FastAPI backend.
