# EuroSAT Unsupervised Land-Type Grouping
### Comprehensive Walkthrough for New & Non-Technical Users, Visual Guide & Future Roadmap

---

## 1. The 30-Second Plain-English Pitch

> *"Imagine an astronaut takes 27,000 satellite photos of Europe from space, but nobody has the time to manually label every single river, farm, and city. Can an AI look at all these photos and sort them into neat piles all by itself without being told what’s in them?"*

This project answers that question with a resounding **Yes**.

We built an end-to-end Machine Learning pipeline and an interactive web dashboard (**EuroSAT Studio**) that ingests satellite imagery from the European Space Agency's **Sentinel-2** satellite constellation, strips away all labels, and uses mathematics and unsupervised clustering to automatically group land into categories like **forests, rivers, farmland, industrial zones, and residential neighborhoods**.

![EuroSAT Studio Vision UI Dashboard](figures/dashboard_preview.png)

---

## 2. The Non-Technical Analogy: How It Works

If you have never studied Data Science or Machine Learning, here is the mental model to understand the entire project:

```
+-------------------------------------------------------------------------------------------------+
|                                    THE "SORTING GAME" ANALOGY                                   |
+-------------------------------------------------------------------------------------------------+
| 1. RAW PIXELS (The Heavy Suitcase):                                                             |
|    Each 64×64 color image has 12,288 numbers describing colors. Packing 12,288 numbers for      |
|    27,000 photos is like carrying a 10-ton suitcase. The computer slows down to a crawl.        |
|                                                                                                 |
| 2. PCA (The Master Sketch Artist):                                                              |
|    Instead of memorizing every pixel, the AI asks: "What are the 2 or 3 biggest patterns?"      |
|    (e.g., Is it mostly blue like water? Or green like grass?). It compresses 12,288 numbers     |
|    into just 2 coordinates on a flat map without losing the essential picture.                  |
|                                                                                                 |
| 3. t-SNE & UMAP (The Magic Microscope):                                                         |
|    Like looking through a lens that pulls similar pictures together and pushes different ones   |
|    far apart, creating distinct "islands" of forests, water, and cities.                        |
|                                                                                                 |
| 4. K-MEANS (The Blindfolded Organizer):                                                         |
|    Without knowing the words "Forest" or "Highway", the algorithm places 10 pins on the map     |
|    and groups nearby photos together. When we take off the blindfold, we find the AI grouped    |
|    almost all water bodies into one pile and forests into another!                              |
+-------------------------------------------------------------------------------------------------+
```

---

## 3. Step-by-Step Project Pipeline: From Space to Screen

```mermaid
flowchart TD
    A["Sentinel-2 Satellite\n(27,000 64×64 Patches)"] --> B["Stage 1: Standardization & Preprocessing\n(12,288 Features, Labels Hidden)"]
    B --> C["Stage 2: PCA Dimensionality Reduction\n(220 PCs capture >90% variance)"]
    B --> D["Stage 3: Multi-Perplexity t-SNE & UMAP\n(Non-linear 2D Manifolds)"]
    C --> E["Stage 4: K-Means Clustering\n(k = 2 to 20 Evaluation)"]
    E --> F["Stage 5: Inference & Nearest Neighbors\n(Real-time prediction & retrieval)"]
    B --> G["Stage 6: 13-Band Multi-Spectral\n(Red-Edge, SWIR & Centroid Grids)"]
    E --> H["Modern Web Dashboard\n(React 19 + Tailwind + FastAPI)"]
    F --> H
    G --> H
```

### Stage 1: Data Preparation & Feature Extraction
* **Source:** The **EuroSAT benchmark**, featuring 27,000 Sentinel-2 satellite image patches covering 34 European countries.
* **Resolution:** $64 \times 64$ pixels with a 10-meter Ground Sample Distance (each image covers an area of $640\text{ m} \times 640\text{ m}$ on Earth).
* **Feature Dimension:** $64 \times 64 \times 3 = 12,288$ continuous pixel values standardized using zero-mean unit-variance scaling (`StandardScaler`).
* **The Rule of Unsupervised Learning:** Ground-truth class names (`Forest`, `River`, `SeaLake`, etc.) were **completely hidden** during training so the AI could not "cheat".

### Stage 2: Dimensionality Reduction via PCA
* **Randomized Singular Value Decomposition (SVD):** Decomposes the centered data matrix $\mathbf{X} = \mathbf{U} \mathbf{\Sigma} \mathbf{V}^T$.
* **Compression Ratio:** 12,288 input dimensions compressed down to **220 orthogonal principal components**, achieving a **55.8-fold parameter reduction** while preserving $>95\%$ of all empirical visual variance.
* **Dominant Variance:** The first **2 principal components** capture **65.30% of total variance** across the entire dataset.

### Stage 3: Manifold Projection (t-SNE & UMAP)
* Projects high-dimensional feature points onto 2D coordinate spaces using the Student-t kernel to capture non-linear local neighborhood similarities.
* Evaluates perplexity across a spectrum: $p \in \{5, 15, 30, 50, 100\}$.

### Stage 4: K-Means Clustering
* Tests cluster counts from $k = 2$ up to $k = 20$.
* Evaluated rigorously using intrinsic metrics (Inertia, Silhouette Score, Davies-Bouldin, Calinski-Harabasz) and post-hoc validation (Adjusted Rand Index, Normalized Mutual Information).

### Stage 5 & 6: Interactive Vision Engine & Multi-Spectral Analysis
* Serialized into an inference engine ([`classifier.py`](classifier.py)) supporting instant image uploads, nearest neighbor retrieval, and full 13-band Sentinel-2 spectral inspection.

---

## 4. Detailed Tour of Every Screen, View, and Graph

When you open the web application at `http://localhost:8000`, the sidebar provides access to **7 comprehensive modules**:

```
+------------------------------------------------------------------------------------+
| EuroSAT Studio                                                                     |
+------------------------------------------------------------------------------------+
| [1. Overview]        [2. PCA Explorer]       [3. t-SNE Manifold]                   |
| [4. K-Means Deep-Dive] [5. Live Inference]   [6. Spectral & UMAP]                  |
| [7. Study Report]                                                                  |
+------------------------------------------------------------------------------------+
```

---

### View 1: Overview & Dataset Explorer

![EuroSAT Dataset Balanced Class Distribution](figures/class_distribution.png)

```
+------------------------------------------------------------------------------------+
| [Total Patches: 27,000] [Taxonomies: 10] [Features: 12,288] [Spectral Bands: 13]  |
+------------------------------------------------------------------------------------+
| Class Distribution Bar Chart                     | Class Breakdown & Samples       |
| 3,000 |  █   █   █   █   █   █   █   █   █   █   | • Forest: 3,000 (11.1%) [IMG]   |
| 2,000 |  █   █   █   █   █   █   █   █   █   █   | • River:  2,500 (9.25%) [IMG]   |
| 1,000 |  █   █   █   █   █   █   █   █   █   █   | • SeaLake:3,000 (11.1%) [IMG]   |
|     0 +------------------------------------------+ • ...                           |
+------------------------------------------------------------------------------------+
```

#### What It Shows:
1. **Top Metric Cards:** 27,000 verified images, 10 balanced land categories, 12,288 raw dimensions, and 13 spectral bands.
2. **Class Distribution Bar Chart:** Visualizes image counts across all 10 categories (ranging from 2,000 to 3,000 per class), confirming the dataset is balanced with zero missing entries.
3. **Interactive Multi-Patch Gallery:** Select any category tab (`Highway`, `Forest`, `SeaLake`, etc.) to instantly view a responsive grid of 12 real satellite patches belonging to that land class.
4. **Patch Inspection Lightbox Modal:** Click on any patch card to open an enlarged $128 \times 128$ preview showing its filename, resolution, K-Means cluster assignment, and a direct button to test it in the Classifier.

> **Non-Technical Explanation:** *"This is the inventory page. It shows what our satellite saw and verifies that our sample collection has plenty of examples for every type of landscape."*

---

### View 2: PCA Variance & Spatial Loadings

![PCA Explained Variance Progression](figures/pca_variance.png)

![PCA 2D Projection Scatter](figures/pca_2d_scatter.png)

```
+------------------------------------------------------------------------------------+
| Scree / Variance Plot           | Interactive 2D PCA Scatter                       |
| 100% |        ,---=== Cumulative|    PC2 ^         . : * (Forests)                 |
|  90% | - - - / - - - - (90% cut)|        |       . : : :                           |
|  65% |  * (PC1 alone = 65.3%)   |        |    . : (Crops)  * (Cities)              |
|   0% +--------------------------+        +-------------------> PC1                 |
|       1   10   50   100   220   |              * * * (Water / SeaLake)             |
+------------------------------------------------------------------------------------+
| Spatial Pixel Activation Heatmap (PC1 & PC2): Highlights center vs edge focus      |
+------------------------------------------------------------------------------------+
```

#### What It Shows:
1. **Scree Curve (Cumulative Variance):** Tracks information retention across components. The first component captures **65.30% of all visual variance**. By component 220, we retain over **95%** while discarding 98% of raw data volume.
2. **2D PCA Scatter Plot:** Every dot represents a satellite image. Dots with similar colors and lighting cluster together. Deep blue water patches gather at the bottom, dense dark green forests cluster near the top, and urban settlements form a distinct rightward cluster.
3. **Spatial Pixel Loadings Heatmap:** A $64 \times 64$ grid revealing which spatial pixel regions (center vs perimeter) drive the principal components.

> **Non-Technical Explanation:** *"Think of PCA as a giant zoom-out. Instead of examining every blade of grass, it identifies the two main dials that distinguish scenes: overall brightness and greenness versus grey concrete."*

---

### View 3: t-SNE Parameter Sensitivity

![t-SNE Parameter Sensitivity Across Perplexities](figures/tsne_comparison.png)

```
+------------------------------------------------------------------------------------+
| Perplexity Slider: [ 5 | 15 | (30) | 50 | 100 ]   KL Divergence: 1.842             |
+------------------------------------------------------------------------------------+
| t-SNE 2D Manifold Scatter                                                          |
|   (Island A: Deep Water)       (Island B: Forests)                                 |
|       * * * *                      * * * * *                                       |
|       * * * *                      * * * * *                                       |
|                                                                                    |
|              (Island C: Farmland / Crops)     (Island D: Residential / Roads)      |
|                   * * * * * *                      * * * * *                       |
+------------------------------------------------------------------------------------+
```

#### What It Shows:
1. **Interactive Perplexity Slider (5 to 100):** Adjusts t-SNE's effective neighborhood scale. Low perplexities ($p=5$) create fragmented micro-islands; balanced perplexities ($p=30$) form clear continental groupings; high perplexities ($p=100$) emphasize broad global relationships.
2. **KL Divergence Indicator:** Quantifies the mismatch between high-dimensional and 2D probability distributions (lower indicates higher fidelity).
3. **Separation Islands:** Non-linear clustering cleanly isolates water bodies (`SeaLake`) and dense vegetation (`Forest`) from man-made structures (`Residential` and `Industrial`).

> **Non-Technical Explanation:** *"t-SNE acts like an automated country map maker. It arranges photos so that similar landscapes are pulled together into distinct islands, making natural groupings easy to spot."*

---

### View 4: K-Means Clustering Deep-Dive

![K-Means Elbow Inertia Curve](figures/elbow_curve.png)

![Silhouette Scores Spectrum](figures/silhouette_scores.png)

![K-Means vs True Label Contingency Heatmap](figures/confusion_heatmap.png)

![Real Satellite Exemplars Closest to Centroids](figures/cluster_exemplars_combined.png)

```
+------------------------------------------------------------------------------------+
| [ARI: 0.384]                  [NMI: 0.521]                  [Purity: 58.4%]        |
+------------------------------------------------------------------------------------+
| Elbow Inertia Curve (k=2..20)        | Silhouette Coefficient Spectrum             |
| Inertia                              | Score                                       |
|   \                                  | 0.16 |  *                                   |
|    `--.  (Elbow at k=4 to 6)         | 0.12 |    \                                 |
|        `------_                      | 0.08 |     `-------                         |
| +-----------------------------> k    | +-------------------------------------> k   |
+------------------------------------------------------------------------------------+
| Cluster vs True Label Heatmap        | 4×4 Exemplar Grid (Closest to Centroid)     |
| Cluster 0: [SeaLake: 97.4% purity]   | [IMG] [IMG] [IMG] [IMG]                     |
| Cluster 1: [Forest:  86.2% purity]   | [IMG] [IMG] [IMG] [IMG]  (Pure Blue Water)  |
| Cluster 2: [Residential/Highway]     | [IMG] [IMG] [IMG] [IMG]                     |
+------------------------------------------------------------------------------------+
```

#### What It Shows:
1. **Evaluation Metrics (ARI & NMI):** Adjusted Rand Index (0.384) and Normalized Mutual Information (0.521) quantify agreement between unsupervised clusters and ground-truth taxonomy.
2. **The Elbow Curve (Inertia vs $k$):** The bend ("elbow") occurs between **$k=4$ and $k=6$**, revealing that satellite landscapes naturally fall into 4 primary archetypes: *Water, Forest, Agriculture, and Human Infrastructure*.
3. **Silhouette Plot:** Confirms that coarse macro-groupings have the highest geometric separation.
4. **Contingency Matrix Heatmap:** Cross-tabulates cluster assignments against ground truth. Cluster 0 captures `SeaLake` with **97.4% purity**; Cluster 1 captures dense `Forest` with **86.2% purity**.
5. **4×4 Exemplar Grids & Centroids:** Displays the 16 real satellite images located closest to the mathematical center of each cluster.

> **Non-Technical Explanation:** *"This view reveals what the AI learned without human supervision. When we inspect Cluster 0's exemplar grid, every single image is open water. The AI recognized what water looks like entirely on its own."*

---

### View 5: Live Image Upload & Classification Support

![Inference Engine Architecture](figures/inference_workflow.png)

```
+------------------------------------------------------------------------------------+
| Upload Satellite Image (or Pick EuroSAT Preset)                                    |
| [ Drag & Drop File ]  or  [ Preset: Highway | Forest | River | SeaLake | ... ]     |
+------------------------------------------------------------------------------------+
| Prediction Results:                  | Historical Top-5 Nearest Neighbors:         |
| • Predicted Cluster: Cluster 1       | 1. Forest_142.jpg   (Distance: 2.14) [IMG]  |
| • Inferred Land-Type: Forest         | 2. Forest_891.jpg   (Distance: 2.45) [IMG]  |
| • Confidence / Purity: 86.2%         | 3. Forest_023.jpg   (Distance: 2.71) [IMG]  |
| • Physical Stats: Brightness 42.1    | 4. Pasture_112.jpg  (Distance: 3.10) [IMG]  |
|                   Roughness 18.4     | 5. Forest_541.jpg   (Distance: 3.22) [IMG]  |
+------------------------------------------------------------------------------------+
| Centroid Distance Bar Chart          | Manifold Marker Location                    |
| Cl 1: [██████████] (Nearest)         |       PC2 ^                                 |
| Cl 5: [████████████████]             |           |       * (Your Upload Here)      |
| Cl 0: [████████████████████████]     |           +-------------------> PC1         |
+------------------------------------------------------------------------------------+
```

#### What It Shows:
1. **Drag-and-Drop or Preset Selector:** Upload any external satellite image or pick a pre-loaded EuroSAT test patch.
2. **Real-Time Classification Card:** Instantly reports the assigned cluster, mapped land cover category, confidence score, and physical attributes (average brightness, surface roughness, RGB balance).
3. **Centroid Distances:** A bar chart comparing distances to all 10 cluster centers.
4. **Top-5 Nearest Neighbors Gallery:** Displays the 5 closest historical images from the 27,000-image dataset, including their filenames and similarity scores.
5. **PCA Manifold Locator:** Plots a target marker on the 2D map showing exactly where your uploaded image falls relative to the entire dataset.

> **Non-Technical Explanation:** *"Drop in any satellite photo, and the system instantly compares it to 27,000 European landscapes, marks its location on the map, and retrieves the 5 most visually similar historical sites."*

---

### View 6: Spectral & Multi-Manifold Analysis

![13-Band Multi-Spectral Profiles](figures/spectral_signature_plot.png)

```
+------------------------------------------------------------------------------------+
| 13 Sentinel-2 Spectral Bands Profile (Beyond Human RGB)                            |
| B1 (Aerosol) | B2-B4 (RGB) | B5-B7 (Red-Edge) | B8 (NIR) | B11-B12 (SWIR)         |
+------------------------------------------------------------------------------------+
| Reflectance                                                                        |
|   ^               .---* (NIR B8: Explosive reflection in Healthy Vegetation)       |
|   |              /                                                                 |
|   |   *---*-----* (Red-Edge transition)                                            |
|   +-------------------------------------> Band Wavelength                         |
+------------------------------------------------------------------------------------+
| UMAP 2D Manifold Scatter             | Projection Method Comparison Table          |
| (Preserves local & global geometry)  | Feature     | PCA    | t-SNE   | UMAP       |
|                                      | Speed       | Fast   | Moderate| Fast       |
|                                      | Global Str. | Strong | Weak    | Strong     |
+------------------------------------------------------------------------------------+
```

#### What It Shows:
1. **13-Band Sentinel-2 Spectral Signatures:** Explores data beyond visible red, green, and blue. Shows how vegetative landscapes reflect near-infrared (NIR) and shortwave-infrared (SWIR) light.
2. **UMAP 2D Scatter:** Uniform Manifold Approximation and Projection provides a complementary perspective to t-SNE, preserving both tight local neighbor groups and broader global relationships.
3. **Methodology Matrix:** Head-to-head comparison of PCA, t-SNE, and UMAP across execution speed, interpretability, and preservation of metric distances.

> **Non-Technical Explanation:** *"Human eyes only see Red, Green, and Blue. Satellites capture 10 additional invisible light bands (like infrared). In infrared, living plants shine brightly, making it easy to separate healthy crops from bare soil."*

---

### View 7: Study Report, Governance & Data Exports

#### What It Shows:
1. Complete academic report covering methodology, findings, and ethical considerations (such as geographic bias toward temperate Europe and mixed-pixel effects at 10-meter resolution).
2. **1-Click Export Center:** Direct download buttons for the executive report (`.md`), clustering metrics (`.json`), and the contingency matrix (`.csv`).

---

## 5. Zero-Setup Execution (How Any User Runs It)

Running this project requires no complex environment setup or manual terminal commands:

### On Windows:
Double-click [`start.bat`](start.bat) or run in terminal:
```cmd
start.bat
```

### On Linux / macOS:
```bash
chmod +x start.sh && ./start.sh
```

### What the Automated Script Does Behind the Scenes:

```mermaid
sequenceDiagram
    autonumber
    actor User
    participant Launcher as start.bat / start.sh
    participant Py as Python Runtime
    participant API as FastAPI Backend (Port 8000)
    participant Browser as Default Browser

    User->>Launcher: Double-clicks start.bat
    Launcher->>Py: Checks Python 3.10+ & Virtual Environment
    Launcher->>Py: Validates dependencies (skips re-install if cached)
    Launcher->>API: Boots FastAPI backend & static distribution
    API->>API: Verifies models in pipeline_outputs/
    Launcher->>API: Probes /api/health until status == 200
    Launcher->>Browser: Automatically opens http://localhost:8000
    Browser-->>User: Displays Vision UI Dashboard!
```

---

## 6. Recommended Next Steps: How to Elevate the Project

While the current pipeline is fully functional and rigorously validated, here are **6 high-impact extensions** to elevate it to production grade:

```
+-----------------------------------------------------------------------------------------------+
|                             RECOMMENDED ARCHITECTURAL UPGRADES                                |
+-----------------------------------------------------------------------------------------------+
| 1. Deep Foundation Embeddings (DINOv2 / RemoteCLIP)                                           |
|    Replace raw pixel PCA with self-supervised geospatial vision transformers for              |
|    near-perfect land boundary recognition.                                                    |
|                                                                                               |
| 2. Interactive Leaflet / Mapbox Geographic Map                                                |
|    Let users click anywhere on a map of Europe to pull real-time satellite tiles              |
|    and cluster them on the fly.                                                               |
|                                                                                               |
| 3. Automated Vegetation Indices (NDVI, NDWI, NDBI)                                            |
|    Calculate standard vegetation and moisture indices directly from the 13-band TIF files     |
|    to distinguish dry cropland from lush pasture.                                             |
|                                                                                               |
| 4. Multi-Temporal Phenology & Deforestation Tracking                                          |
|    Compare image patches from different seasons to identify crop cycles, urban growth,       |
|    or tree loss over time.                                                                    |
|                                                                                               |
| 5. Human-in-the-Loop Active Learning                                                          |
|    Allow domain experts to re-assign misclassified images in the UI, updating cluster         |
|    weights dynamically.                                                                       |
|                                                                                               |
| 6. Direct GIS Export (GeoTIFF, Shapefile, GeoJSON)                                            |
|    Enable urban planners to export predictions directly into QGIS and ArcGIS.                 |
+-----------------------------------------------------------------------------------------------+
```

### Detailed Breakdown of Enhancements:

#### 1. Self-Supervised Vision Transformer (DINOv2 / RemoteCLIP)
* **Current Approach:** Raw flattened pixels ($12,288$ dimensions) scaled and reduced via PCA.
* **Limitation:** Raw pixels do not account for rotational invariance or spatial textures (a rotated building looks like different numbers to basic PCA).
* **Upgrade:** Run patches through a lightweight geospatial backbone (e.g., `dinov2_vits14` or `RemoteCLIP`).
* **Expected Result:** Cluster purity should increase from **58.4% to >88%** without requiring labeled training data.

#### 2. Interactive Real-World Map Integration
* **Current Approach:** Presets and uploaded image files.
* **Upgrade:** Embed a **Leaflet.js / OpenStreetMap** widget on the Inference page. Users can pan across Europe (e.g., the Alps, Rhine river, Paris suburbs), click an area to fetch the corresponding Sentinel-2 tile via the Sentinel Hub API, and view live cluster predictions.

#### 3. Native Environmental Indices (NDVI & NDWI)
* **Current Approach:** Multi-spectral bands are presented as raw reflectance numbers.
* **Upgrade:** Add standard remote sensing formulas in [`pipeline.py`](pipeline.py):
  $$\text{NDVI} = \frac{\text{B08 (NIR)} - \text{B04 (Red)}}{\text{B08 (NIR)} + \text{B04 (Red)}}$$
  $$\text{NDWI} = \frac{\text{B03 (Green)} - \text{B08 (NIR)}}{\text{B03 (Green)} + \text{B08 (NIR)}}$$
* **Benefit:** Gives non-technical users intuitive health gauges: *Vegetation Index: 0.82 (Dense Healthy Canopy)* or *Water Index: 0.74 (Open Deep Water)*.

#### 4. Temporal Multi-Date Change Detection
* **Limitation:** A major challenge noted in the report is differentiating `AnnualCrop` from `PermanentCrop` in single static summer snapshots.
* **Upgrade:** Ingest two timestamps (e.g., May vs October). If an agricultural patch goes from green to brown, it is an annual crop; if it stays green, it is permanent orchard/vineyard.

#### 5. Active Learning & Analyst Refinement
* **Upgrade:** Add a *"Correct this classification"* button on the Image Inference page.
* **Benefit:** When an analyst marks a misclustered river patch, the system caches the correction and adjusts boundary weights, turning the project into an evolving decision-support system.

#### 6. Standard GIS Export Tools
* **Upgrade:** Add an export button producing `.geojson` and `.tif` files tagged with coordinate reference systems (CRS WGS84).
* **Benefit:** Allows government agencies and environmental researchers to import findings directly into standard GIS tools like QGIS and ArcGIS.

---

## 7. Summary Reference Table

| Feature / Question | Project Implementation | Non-Technical Takeaway |
|---|---|---|
| **What data is used?** | 27,000 Sentinel-2 satellite patches across 10 European land classes. | High-resolution satellite snapshots covering rivers, forests, cities, and farms across Europe. |
| **Was the AI given labels?** | No. Ground truth labels were isolated during model fitting. | The AI sorted images based purely on visual patterns, with zero human assistance. |
| **Why PCA?** | Captures 65.3% variance in 2 components, >95% in 220 components. | Compresses massive image files down to essential color and brightness patterns. |
| **Why t-SNE / UMAP?** | Non-linear manifold learning across perplexities 5 to 100. | Groups visually similar landscapes into distinct "islands" on a 2D map. |
| **What did K-Means find?** | Natural elbow between $k=4$ and $k=6$; 10-cluster purity reaches 97%+ on water. | The AI automatically discovered broad macro-types: Water, Forest, Farmland, and Urban Infrastructure. |
| **How do I test an image?** | Open the web UI, drag-and-drop any satellite picture, and view real-time results. | Instantly shows predicted category, similar historical matches, and placement on the map. |
