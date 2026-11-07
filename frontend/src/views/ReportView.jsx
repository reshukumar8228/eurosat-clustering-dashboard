import React from 'react';
import { Download } from 'lucide-react';

export default function ReportView() {
  const downloadFile = (url, filename) => {
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
  };

  return (
    <div className="space-y-6 max-w-5xl">
      {/* Top Header Card */}
      <div className="p-6 rounded-2xl bg-[#0F172A]/70 backdrop-blur-xl border border-slate-800/80 shadow-xl flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div>
          <span className="text-xs font-semibold text-blue-400 uppercase tracking-wider">Formal Study Submission</span>
          <h3 className="text-xl font-bold text-white tracking-tight mt-0.5">
            Unsupervised Land-Type Grouping of Satellite Image Patches
          </h3>
          <p className="text-xs text-slate-400">Reproducible remote sensing study using Sentinel-2 optical imagery</p>
        </div>

        <div className="flex items-center gap-2.5">
          <button
            onClick={() => downloadFile('/static/outputs/kmeans_metrics.json', 'kmeans_metrics.json')}
            className="flex items-center gap-2 px-3.5 py-2 rounded-xl bg-slate-800/80 hover:bg-slate-700/80 border border-slate-700 text-xs font-semibold text-white transition-all shadow-md"
          >
            <Download className="w-3.5 h-3.5 text-blue-400" />
            <span>Metrics JSON</span>
          </button>
          <button
            onClick={() => downloadFile('/static/outputs/kmeans_k10_contingency.csv', 'contingency_matrix.csv')}
            className="flex items-center gap-2 px-3.5 py-2 rounded-xl bg-blue-600 hover:bg-blue-500 text-xs font-semibold text-white transition-all shadow-lg shadow-blue-500/25"
          >
            <Download className="w-3.5 h-3.5" />
            <span>Contingency CSV</span>
          </button>
        </div>
      </div>

      {/* Main Report Document */}
      <div className="p-8 rounded-2xl bg-[#0F172A]/70 backdrop-blur-xl border border-slate-800/80 shadow-xl space-y-8 text-slate-300 text-sm leading-relaxed">
        {/* Section 1 */}
        <section className="space-y-3">
          <h4 className="text-base font-bold text-white tracking-tight pb-2 border-b border-slate-800 flex items-center gap-2">
            <span>1. Executive Summary & Objective Alignment</span>
          </h4>
          <p>
            This investigation analyzes the natural structure of 27,000 multi-spectral and optical satellite patches from the EuroSAT dataset. Ground truth class labels were strictly isolated during preprocessing, PCA variance analysis, t-SNE manifold learning, and K-Means clustering. Post-hoc evaluation quantifies the alignment between unsupervised spectral clusters and human-annotated land cover categories.
          </p>
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 pt-2">
            <div className="p-4 rounded-xl bg-slate-900/60 border border-slate-800">
              <span className="text-xs text-slate-400">PCA Retained Variance</span>
              <p className="text-lg font-bold text-white mt-1">91.84% (220 Components)</p>
            </div>
            <div className="p-4 rounded-xl bg-slate-900/60 border border-slate-800">
              <span className="text-xs text-slate-400">Post-Hoc k=10 ARI</span>
              <p className="text-lg font-bold text-white mt-1">0.1286</p>
            </div>
            <div className="p-4 rounded-xl bg-slate-900/60 border border-slate-800">
              <span className="text-xs text-slate-400">Overall Archetype Purity</span>
              <p className="text-lg font-bold text-white mt-1">30.99%</p>
            </div>
          </div>
        </section>

        {/* Section 2 */}
        <section className="space-y-3">
          <h4 className="text-base font-bold text-white tracking-tight pb-2 border-b border-slate-800 flex items-center gap-2">
            <span>2. Mathematical Methodology</span>
          </h4>
          <ul className="list-disc pl-5 space-y-2 text-xs leading-relaxed text-slate-300">
            <li>
              <strong className="text-white">Principal Component Analysis (PCA):</strong> Maximizes projection variance via randomized singular value decomposition (SVD) on the standardized centered feature matrix (27,000 samples × 12,288 dimensions).
            </li>
            <li>
              <strong className="text-white">t-Distributed Stochastic Neighbor Embedding (t-SNE):</strong> Evaluated across perplexities p in [5, 15, 30, 50, 100]. Minimizes the Kullback-Leibler divergence between high-dimensional joint probabilities and low-dimensional Student-t kernel representations.
            </li>
            <li>
              <strong className="text-white">K-Means Objective:</strong> Partitioned the 220-dimensional PCA space across cluster counts k in [2, 20] minimizing the within-cluster sum of squares (inertia).
            </li>
          </ul>
        </section>

        {/* Section 3 */}
        <section className="space-y-3">
          <h4 className="text-base font-bold text-white tracking-tight pb-2 border-b border-slate-800 flex items-center gap-2">
            <span>3. Limitations & Ethical Governance</span>
          </h4>

          <div className="space-y-3 text-xs">
            <div className="p-4 rounded-xl bg-slate-900/60 border border-slate-800 space-y-1">
              <strong className="text-white">Geographic and Seasonal Bias:</strong>
              <p className="text-slate-400">
                EuroSAT patches are gathered strictly across European territories. Applying this model directly to equatorial rainforests, desert regions, or peri-urban settlements in the Global South will yield systematic misclassification due to differing architectural materials, soil minerality, and vegetation phenology.
              </p>
            </div>

            <div className="p-4 rounded-xl bg-slate-900/60 border border-slate-800 space-y-1">
              <strong className="text-white">Spatial Resolution Limits & Mixed Pixels:</strong>
              <p className="text-slate-400">
                At 10 meters Ground Sample Distance (GSD), each 64×64 patch spans 640m × 640m. Mixed-pixel phenomena occur where roadways, water channels, and residential settlements bisect in a single patch.
              </p>
            </div>

            <div className="p-4 rounded-xl bg-slate-900/60 border border-slate-800 space-y-1">
              <strong className="text-white">Responsible Environmental AI:</strong>
              <p className="text-slate-400">
                Automated land grouping outputs should serve as a screening and decision-support mechanism for qualified remote sensing analysts, not as unilateral inputs for land taxation, agricultural subsidies, or punitive zoning.
              </p>
            </div>
          </div>
        </section>
      </div>
    </div>
  );
}
