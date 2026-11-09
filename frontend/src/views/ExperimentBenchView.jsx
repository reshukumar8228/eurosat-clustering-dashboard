import { useState, useEffect, useCallback, useMemo } from 'react';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  CartesianGrid,
  Cell
} from 'recharts';
import {
  FlaskConical,
  Zap,
  Play,
  RotateCcw,
  Sliders,
  Layers,
  Sparkles,
  Info
} from 'lucide-react';
import ParameterSlider from '../components/ParameterSlider';

const CLASS_COLORS = {
  AnnualCrop: '#2ca02c',
  Forest: '#006400',
  HerbaceousVegetation: '#8c564b',
  Highway: '#7f7f7f',
  Industrial: '#e377c2',
  Pasture: '#bcbd22',
  PermanentCrop: '#ff7f0e',
  Residential: '#d62728',
  River: '#17becf',
  SeaLake: '#1f77b4'
};

const CLUSTER_COLORS = [
  '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
  '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
  '#6366F1', '#A855F7', '#D946EF', '#E11D48', '#FB923C',
  '#FACC15', '#4ADE80', '#2DD4BF', '#38BDF8', '#818CF8',
  '#C084FC', '#F472B6', '#FB7185', '#94A3B8', '#64748B',
  '#0EA5E9', '#22C55E', '#EAB308', '#F43F5E', '#A855F7'
];

const PRECOMPUTED_K = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 15, 20];
const TSNE_PERPLEXITIES = [5, 15, 30, 50, 100];

const DEFAULT_PARAMS = {
  pca_components: 220,
  pca_axis_x: 1,
  pca_axis_y: 2,
  kmeans_k: 10,
  tsne_perplexity: 30,
  sample_size: 2000
};

export default function ExperimentBenchView() {
  const [params, setParams] = useState({ ...DEFAULT_PARAMS });
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [hasChanges, setHasChanges] = useState(false);
  const [colorMode, setColorMode] = useState('cluster');
  const [hoveredPoint, setHoveredPoint] = useState(null);

  const updateParam = (key, value) => {
    setParams(prev => ({ ...prev, [key]: value }));
    setHasChanges(true);
  };

  const runExperiment = useCallback(async (currentParams = params) => {
    setLoading(true);
    try {
      const res = await fetch('/api/experiment', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(currentParams)
      });
      if (!res.ok) throw new Error(`HTTP error ${res.status}`);
      const json = await res.json();
      setData(json);
      setHasChanges(false);
    } catch (err) {
      console.error('Failed to run experiment:', err);
    } finally {
      setLoading(false);
    }
  }, [params]);

  // Initial load on mount
  useEffect(() => {
    let ignore = false;
    fetch('/api/experiment', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(DEFAULT_PARAMS)
    })
      .then(res => res.json())
      .then(json => {
        if (!ignore) {
          setData(json);
        }
      })
      .catch(err => {
        if (!ignore) {
          console.error('Failed initial experiment load:', err);
        }
      });
    return () => {
      ignore = true;
    };
  }, []);

  const resetDefaults = () => {
    setParams({ ...DEFAULT_PARAMS });
    runExperiment(DEFAULT_PARAMS);
  };

  const isPrecomputedK = useMemo(() => {
    return params.pca_components === 220 && PRECOMPUTED_K.includes(params.kmeans_k);
  }, [params.pca_components, params.kmeans_k]);

  // PCA scatter bounds calculation
  const pcaBounds = useMemo(() => {
    if (!data?.pca?.scatter_points?.length) return { minX: -10, maxX: 10, minY: -10, maxY: 10 };
    const xs = data.pca.scatter_points.map(p => p.x);
    const ys = data.pca.scatter_points.map(p => p.y);
    return {
      minX: Math.min(...xs),
      maxX: Math.max(...xs),
      minY: Math.min(...ys),
      maxY: Math.max(...ys)
    };
  }, [data]);

  // t-SNE scatter bounds calculation
  const tsneBounds = useMemo(() => {
    if (!data?.tsne?.scatter_points?.length) return { minX: -10, maxX: 10, minY: -10, maxY: 10 };
    const xs = data.tsne.scatter_points.map(p => p.x);
    const ys = data.tsne.scatter_points.map(p => p.y);
    return {
      minX: Math.min(...xs),
      maxX: Math.max(...xs),
      minY: Math.min(...ys),
      maxY: Math.max(...ys)
    };
  }, [data]);

  return (
    <div className="space-y-6">
      {/* Top Banner / Explanation */}
      <div className="p-5 rounded-2xl bg-gradient-to-r from-blue-950/40 via-indigo-950/30 to-slate-900/60 border border-blue-500/20 backdrop-blur-xl shadow-xl flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div className="flex items-center gap-3.5">
          <div className="w-10 h-10 rounded-xl bg-blue-600/20 border border-blue-500/30 flex items-center justify-center text-blue-400">
            <FlaskConical className="w-5 h-5" />
          </div>
          <div>
            <h3 className="text-base font-bold text-white tracking-tight flex items-center gap-2">
              Experiment Bench
              <span className="text-[11px] font-semibold px-2 py-0.5 rounded-full bg-blue-500/20 border border-blue-500/30 text-blue-400">
                Interactive Playground
              </span>
            </h3>
            <p className="text-xs text-slate-400 mt-0.5">
              Tune PCA dimensionality, K-Means clustering partitions, and t-SNE projection manifolds in real-time.
            </p>
          </div>
        </div>

        {/* Quick Presets */}
        <div className="flex items-center gap-2 flex-wrap">
          <span className="text-[11px] text-slate-400 font-medium mr-1">Presets:</span>
          <button
            onClick={() => {
              const p = { ...DEFAULT_PARAMS, pca_components: 220, kmeans_k: 10, tsne_perplexity: 30 };
              setParams(p);
              runExperiment(p);
            }}
            className="px-2.5 py-1 text-xs bg-slate-900/80 border border-slate-800 rounded-lg text-slate-300 hover:text-white hover:border-slate-700 transition-all"
          >
            Baseline (K=10)
          </button>
          <button
            onClick={() => {
              const p = { ...DEFAULT_PARAMS, pca_components: 15, kmeans_k: 6, tsne_perplexity: 15 };
              setParams(p);
              runExperiment(p);
            }}
            className="px-2.5 py-1 text-xs bg-slate-900/80 border border-slate-800 rounded-lg text-slate-300 hover:text-white hover:border-slate-700 transition-all"
          >
            Compact (15 PCs, K=6)
          </button>
          <button
            onClick={() => {
              const p = { ...DEFAULT_PARAMS, pca_components: 100, kmeans_k: 15, tsne_perplexity: 50 };
              setParams(p);
              runExperiment(p);
            }}
            className="px-2.5 py-1 text-xs bg-slate-900/80 border border-slate-800 rounded-lg text-slate-300 hover:text-white hover:border-slate-700 transition-all"
          >
            Granular (K=15)
          </button>
        </div>
      </div>

      {/* Main Grid: Controls on Left, Visualizations on Right */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">
        {/* Left Column: Parameter Control Panel */}
        <div className="lg:col-span-4 space-y-5">
          <div className="p-5 rounded-2xl bg-[#0F172A]/70 backdrop-blur-xl border border-slate-800/80 shadow-xl space-y-5">
            <div className="flex items-center justify-between pb-3 border-b border-slate-800/80">
              <div className="flex items-center gap-2">
                <Sliders className="w-4 h-4 text-blue-400" />
                <h4 className="text-sm font-bold text-white tracking-tight">Hyperparameter Controls</h4>
              </div>
              <button
                onClick={resetDefaults}
                className="text-[11px] text-slate-400 hover:text-white flex items-center gap-1 transition-colors"
                title="Reset to defaults"
              >
                <RotateCcw className="w-3 h-3" />
                Reset
              </button>
            </div>

            {/* Run Button (Highlighted when parameters have changed) */}
            <button
              onClick={() => runExperiment()}
              disabled={loading}
              className={`w-full py-2.5 px-4 rounded-xl font-semibold text-xs flex items-center justify-center gap-2 transition-all shadow-lg ${
                hasChanges
                  ? 'bg-gradient-to-r from-blue-600 to-indigo-600 text-white shadow-blue-500/30 ring-2 ring-blue-500/50 animate-pulse'
                  : 'bg-blue-600 text-white hover:bg-blue-500 shadow-blue-600/20'
              } disabled:opacity-50 disabled:cursor-not-allowed`}
            >
              {loading ? (
                <>
                  <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin" />
                  <span>Computing Models...</span>
                </>
              ) : (
                <>
                  <Play className="w-4 h-4 fill-current" />
                  <span>{hasChanges ? 'Apply & Run Experiment' : 'Re-run Experiment'}</span>
                </>
              )}
            </button>

            {/* Parameter Sliders */}
            <div className="space-y-4 pt-1">
              {/* PCA Components Slider */}
              <div className="p-3 rounded-xl bg-slate-900/60 border border-slate-800/60 space-y-2">
                <ParameterSlider
                  label="Retained PCA Components"
                  value={params.pca_components}
                  min={2}
                  max={220}
                  step={1}
                  onChange={(v) => updateParam('pca_components', v)}
                  description="Number of principal components passed downstream to K-Means."
                  highlights={[
                    { value: 10, label: '10' },
                    { value: 50, label: '50' },
                    { value: 100, label: '100' },
                    { value: 220, label: '220 (Full)' }
                  ]}
                />
              </div>

              {/* PCA Projection Axes */}
              <div className="p-3 rounded-xl bg-slate-900/60 border border-slate-800/60 space-y-3">
                <div className="flex items-center justify-between">
                  <span className="text-xs font-semibold text-slate-300 uppercase tracking-wider">
                    PCA Scatter Axes
                  </span>
                  <span className="text-[11px] text-blue-400 font-medium">
                    PC{params.pca_axis_x} vs PC{params.pca_axis_y}
                  </span>
                </div>
                <div className="grid grid-cols-2 gap-3">
                  <div>
                    <label className="text-[11px] text-slate-400 block mb-1">X-Axis</label>
                    <select
                      value={params.pca_axis_x}
                      onChange={(e) => updateParam('pca_axis_x', Number(e.target.value))}
                      className="w-full bg-slate-950 border border-slate-800 rounded-lg px-2 py-1.5 text-xs text-white focus:outline-none focus:border-blue-500"
                    >
                      {Array.from({ length: 10 }, (_, i) => i + 1).map(n => (
                        <option key={n} value={n}>PC {n}</option>
                      ))}
                    </select>
                  </div>
                  <div>
                    <label className="text-[11px] text-slate-400 block mb-1">Y-Axis</label>
                    <select
                      value={params.pca_axis_y}
                      onChange={(e) => updateParam('pca_axis_y', Number(e.target.value))}
                      className="w-full bg-slate-950 border border-slate-800 rounded-lg px-2 py-1.5 text-xs text-white focus:outline-none focus:border-blue-500"
                    >
                      {Array.from({ length: 10 }, (_, i) => i + 1).map(n => (
                        <option key={n} value={n}>PC {n}</option>
                      ))}
                    </select>
                  </div>
                </div>
              </div>

              {/* K-Means Clusters Slider */}
              <div className="p-3 rounded-xl bg-slate-900/60 border border-slate-800/60 space-y-2">
                <div className="flex items-center justify-between">
                  <span className="text-xs font-semibold text-slate-300 uppercase tracking-wider">
                    K-Means Clusters (k)
                  </span>
                  <div className="flex items-center gap-1.5">
                    {isPrecomputedK ? (
                      <span className="inline-flex items-center gap-1 text-[10px] font-semibold text-emerald-400 bg-emerald-500/10 px-2 py-0.5 rounded border border-emerald-500/20">
                        <Zap className="w-2.5 h-2.5" /> Instant
                      </span>
                    ) : (
                      <span className="inline-flex items-center gap-1 text-[10px] font-semibold text-amber-400 bg-amber-500/10 px-2 py-0.5 rounded border border-amber-500/20">
                        <Sparkles className="w-2.5 h-2.5" /> Live Fit
                      </span>
                    )}
                    <span className="text-xs font-bold text-blue-400 bg-blue-500/10 px-2 py-0.5 rounded border border-blue-500/20 tabular-nums">
                      k = {params.kmeans_k}
                    </span>
                  </div>
                </div>
                <input
                  type="range"
                  min={2}
                  max={30}
                  step={1}
                  value={params.kmeans_k}
                  onChange={(e) => updateParam('kmeans_k', Number(e.target.value))}
                  className="w-full h-1.5 bg-slate-800 rounded-full appearance-none cursor-pointer accent-blue-500"
                />
                <div className="flex justify-between items-center text-[10px] text-slate-500 pt-0.5">
                  <span>k=2</span>
                  <span>k=10 (Natural)</span>
                  <span>k=20</span>
                  <span>k=30</span>
                </div>
              </div>

              {/* t-SNE Perplexity Presets */}
              <div className="p-3 rounded-xl bg-slate-900/60 border border-slate-800/60 space-y-2">
                <div className="flex items-center justify-between">
                  <span className="text-xs font-semibold text-slate-300 uppercase tracking-wider">
                    t-SNE Perplexity
                  </span>
                  <span className="text-xs font-bold text-blue-400 bg-blue-500/10 px-2 py-0.5 rounded border border-blue-500/20">
                    p = {params.tsne_perplexity}
                  </span>
                </div>
                <p className="text-[11px] text-slate-400">
                  Pre-computed manifold embeddings across neighborhood radii.
                </p>
                <div className="grid grid-cols-5 gap-1.5 pt-1">
                  {TSNE_PERPLEXITIES.map((perp) => (
                    <button
                      key={perp}
                      type="button"
                      onClick={() => updateParam('tsne_perplexity', perp)}
                      className={`py-1.5 text-xs font-semibold rounded-lg transition-all ${
                        params.tsne_perplexity === perp
                          ? 'bg-blue-600 text-white shadow-md shadow-blue-500/20'
                          : 'bg-slate-950 border border-slate-800 text-slate-400 hover:text-white hover:border-slate-700'
                      }`}
                    >
                      {perp}
                    </button>
                  ))}
                </div>
              </div>

              {/* Sample Size Slider */}
              <div className="p-3 rounded-xl bg-slate-900/60 border border-slate-800/60 space-y-2">
                <ParameterSlider
                  label="Scatter Sample Density"
                  value={params.sample_size}
                  min={500}
                  max={4000}
                  step={500}
                  onChange={(v) => updateParam('sample_size', v)}
                  description="Number of stratified sample points rendered in scatter plots."
                />
              </div>
            </div>
          </div>
        </div>

        {/* Right Column: Metrics & Visualizations */}
        <div className="lg:col-span-8 space-y-6">
          {/* Live Metrics Cards */}
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
            <div className="p-4 rounded-2xl bg-[#0F172A]/70 backdrop-blur-xl border border-slate-800/80 shadow-xl">
              <p className="text-[11px] font-semibold text-slate-400 uppercase tracking-wider">PCA Variance</p>
              <h3 className="text-xl font-bold text-white mt-0.5 tabular-nums">
                {data ? `${data.pca.variance_retained}%` : '—'}
              </h3>
              <p className="text-[10px] text-blue-400 mt-0.5">Top {params.pca_components} components</p>
            </div>

            <div className="p-4 rounded-2xl bg-[#0F172A]/70 backdrop-blur-xl border border-slate-800/80 shadow-xl">
              <p className="text-[11px] font-semibold text-slate-400 uppercase tracking-wider">Silhouette</p>
              <h3 className="text-xl font-bold text-white mt-0.5 tabular-nums">
                {data ? data.kmeans.silhouette : '—'}
              </h3>
              <p className="text-[10px] text-emerald-400 mt-0.5">Cluster cohesion</p>
            </div>

            <div className="p-4 rounded-2xl bg-[#0F172A]/70 backdrop-blur-xl border border-slate-800/80 shadow-xl">
              <p className="text-[11px] font-semibold text-slate-400 uppercase tracking-wider">Davies-Bouldin</p>
              <h3 className="text-xl font-bold text-white mt-0.5 tabular-nums">
                {data ? data.kmeans.davies_bouldin : '—'}
              </h3>
              <p className="text-[10px] text-amber-400 mt-0.5">Separation index</p>
            </div>

            <div className="p-4 rounded-2xl bg-[#0F172A]/70 backdrop-blur-xl border border-slate-800/80 shadow-xl">
              <p className="text-[11px] font-semibold text-slate-400 uppercase tracking-wider">Computation</p>
              <div className="flex items-center gap-1.5 mt-0.5">
                {data?.kmeans?.is_precomputed ? (
                  <span className="text-base font-bold text-emerald-400 flex items-center gap-1">
                    <Zap className="w-3.5 h-3.5" /> Instant
                  </span>
                ) : (
                  <span className="text-base font-bold text-amber-400 tabular-nums">
                    {data?.comparison?.computation_time_ms ? `${data.comparison.computation_time_ms} ms` : '—'}
                  </span>
                )}
              </div>
              <p className="text-[10px] text-slate-400 mt-0.5">
                {data?.kmeans?.is_precomputed ? 'Cached model' : 'Live fit'}
              </p>
            </div>
          </div>

          {/* Dual Scatter Plots Section */}
          <div className="p-6 rounded-2xl bg-[#0F172A]/70 backdrop-blur-xl border border-slate-800/80 shadow-xl space-y-4">
            <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
              <div>
                <h4 className="text-sm font-bold text-white tracking-tight flex items-center gap-2">
                  <Layers className="w-4 h-4 text-blue-400" />
                  Dual Manifold Projection Space
                </h4>
                <p className="text-xs text-slate-400">
                  Compare linear PCA projection with non-linear t-SNE manifold using unified cluster assignments.
                </p>
              </div>

              {/* Color Mode Toggle */}
              <div className="flex items-center gap-1 bg-slate-900/90 p-1 rounded-xl border border-slate-800">
                <button
                  type="button"
                  onClick={() => setColorMode('cluster')}
                  className={`px-3 py-1 rounded-lg text-xs font-semibold transition-all ${
                    colorMode === 'cluster'
                      ? 'bg-blue-600 text-white shadow-md'
                      : 'text-slate-400 hover:text-white'
                  }`}
                >
                  K-Means (k={params.kmeans_k})
                </button>
                <button
                  type="button"
                  onClick={() => setColorMode('label')}
                  className={`px-3 py-1 rounded-lg text-xs font-semibold transition-all ${
                    colorMode === 'label'
                      ? 'bg-blue-600 text-white shadow-md'
                      : 'text-slate-400 hover:text-white'
                  }`}
                >
                  True Class
                </button>
              </div>
            </div>

            {/* Scatter Plots Side-by-Side */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {/* PCA Plot */}
              <div className="relative bg-[#070B14] rounded-xl border border-slate-800/80 p-3 h-80 flex flex-col justify-between">
                <div className="flex justify-between items-center text-xs font-medium text-slate-400 px-2 pb-1 border-b border-slate-800/60">
                  <span>PCA (PC {params.pca_axis_x} vs PC {params.pca_axis_y})</span>
                  <span className="text-[11px] text-blue-400">{data?.pca?.scatter_points?.length || 0} pts</span>
                </div>
                <div className="relative flex-1 w-full overflow-hidden flex items-center justify-center">
                  {loading && (
                    <div className="absolute inset-0 bg-slate-950/60 backdrop-blur-xs flex items-center justify-center z-10">
                      <div className="w-6 h-6 border-2 border-blue-500 border-t-transparent rounded-full animate-spin" />
                    </div>
                  )}
                  {data?.pca?.scatter_points && (
                    <svg className="w-full h-full" viewBox="0 0 400 300">
                      <line x1="20" y1="150" x2="380" y2="150" stroke="#1E293B" strokeWidth="1" strokeDasharray="3 3" />
                      <line x1="200" y1="20" x2="200" y2="280" stroke="#1E293B" strokeWidth="1" strokeDasharray="3 3" />
                      {data.pca.scatter_points.map((p, idx) => {
                        const cx = 20 + ((p.x - pcaBounds.minX) / (pcaBounds.maxX - pcaBounds.minX || 1)) * 360;
                        const cy = 280 - ((p.y - pcaBounds.minY) / (pcaBounds.maxY - pcaBounds.minY || 1)) * 260;
                        const fill = colorMode === 'label'
                          ? (CLASS_COLORS[p.label] || '#3B82F6')
                          : CLUSTER_COLORS[p.cluster % CLUSTER_COLORS.length];
                        return (
                          <circle
                            key={idx}
                            cx={cx}
                            cy={cy}
                            r="3"
                            fill={fill}
                            opacity={hoveredPoint ? (hoveredPoint.filename === p.filename && hoveredPoint.source === 'PCA' ? 1 : 0.45) : 0.8}
                            className="cursor-pointer"
                            onMouseEnter={() => setHoveredPoint({ ...p, source: 'PCA' })}
                            onMouseLeave={() => setHoveredPoint(prev => (prev?.filename === p.filename && prev?.source === 'PCA' ? null : prev))}
                          />
                        );
                      })}
                      {hoveredPoint?.source === 'PCA' && (
                        <circle
                          cx={20 + ((hoveredPoint.x - pcaBounds.minX) / (pcaBounds.maxX - pcaBounds.minX || 1)) * 360}
                          cy={280 - ((hoveredPoint.y - pcaBounds.minY) / (pcaBounds.maxY - pcaBounds.minY || 1)) * 260}
                          r="6"
                          fill="none"
                          stroke="#FFFFFF"
                          strokeWidth="2"
                          className="pointer-events-none"
                        />
                      )}
                    </svg>
                  )}
                </div>
              </div>

              {/* t-SNE Plot */}
              <div className="relative bg-[#070B14] rounded-xl border border-slate-800/80 p-3 h-80 flex flex-col justify-between">
                <div className="flex justify-between items-center text-xs font-medium text-slate-400 px-2 pb-1 border-b border-slate-800/60">
                  <span>t-SNE (Perplexity = {params.tsne_perplexity})</span>
                  <span className="text-[11px] text-emerald-400">KL: {data?.tsne?.kl_divergence || '—'}</span>
                </div>
                <div className="relative flex-1 w-full overflow-hidden flex items-center justify-center">
                  {loading && (
                    <div className="absolute inset-0 bg-slate-950/60 backdrop-blur-xs flex items-center justify-center z-10">
                      <div className="w-6 h-6 border-2 border-emerald-500 border-t-transparent rounded-full animate-spin" />
                    </div>
                  )}
                  {data?.tsne?.scatter_points && (
                    <svg className="w-full h-full" viewBox="0 0 400 300">
                      <line x1="20" y1="150" x2="380" y2="150" stroke="#1E293B" strokeWidth="1" strokeDasharray="3 3" />
                      <line x1="200" y1="20" x2="200" y2="280" stroke="#1E293B" strokeWidth="1" strokeDasharray="3 3" />
                      {data.tsne.scatter_points.map((p, idx) => {
                        const cx = 20 + ((p.x - tsneBounds.minX) / (tsneBounds.maxX - tsneBounds.minX || 1)) * 360;
                        const cy = 280 - ((p.y - tsneBounds.minY) / (tsneBounds.maxY - tsneBounds.minY || 1)) * 260;
                        const fill = colorMode === 'label'
                          ? (CLASS_COLORS[p.label] || '#3B82F6')
                          : CLUSTER_COLORS[p.cluster % CLUSTER_COLORS.length];
                        return (
                          <circle
                            key={idx}
                            cx={cx}
                            cy={cy}
                            r="3"
                            fill={fill}
                            opacity={hoveredPoint ? (hoveredPoint.filename === p.filename && hoveredPoint.source === 't-SNE' ? 1 : 0.45) : 0.8}
                            className="cursor-pointer"
                            onMouseEnter={() => setHoveredPoint({ ...p, source: 't-SNE' })}
                            onMouseLeave={() => setHoveredPoint(prev => (prev?.filename === p.filename && prev?.source === 't-SNE' ? null : prev))}
                          />
                        );
                      })}
                      {hoveredPoint?.source === 't-SNE' && (
                        <circle
                          cx={20 + ((hoveredPoint.x - tsneBounds.minX) / (tsneBounds.maxX - tsneBounds.minX || 1)) * 360}
                          cy={280 - ((hoveredPoint.y - tsneBounds.minY) / (tsneBounds.maxY - tsneBounds.minY || 1)) * 260}
                          r="6"
                          fill="none"
                          stroke="#FFFFFF"
                          strokeWidth="2"
                          className="pointer-events-none"
                        />
                      )}
                    </svg>
                  )}
                </div>
              </div>
            </div>

            {/* Hover Tooltip Card */}
            {hoveredPoint && (
              <div className="p-3 bg-slate-900/90 border border-slate-700/80 rounded-xl text-xs text-white shadow-2xl backdrop-blur-md flex items-center justify-between pointer-events-none">
                <div>
                  <span className="font-semibold text-blue-400">{hoveredPoint.filename}</span>
                  <span className="text-slate-400 mx-2">|</span>
                  <span className="text-slate-300">True Class: </span>
                  <span className="font-semibold" style={{ color: CLASS_COLORS[hoveredPoint.label] || '#fff' }}>
                    {hoveredPoint.label}
                  </span>
                  <span className="text-slate-400 mx-2">|</span>
                  <span className="text-slate-300">Cluster: </span>
                  <span className="font-bold text-white">Cluster {hoveredPoint.cluster}</span>
                </div>
                <span className="text-[11px] text-slate-500">
                  {hoveredPoint.source}: ({hoveredPoint.x}, {hoveredPoint.y})
                </span>
              </div>
            )}

            {/* Legend */}
            <div className="flex flex-wrap items-center justify-center gap-3 pt-3 border-t border-slate-800/60 text-xs">
              {colorMode === 'label' ? (
                Object.entries(CLASS_COLORS).map(([name, color]) => (
                  <div key={name} className="flex items-center gap-1.5 text-slate-300">
                    <span className="w-2.5 h-2.5 rounded-full" style={{ backgroundColor: color }} />
                    <span className="text-[11px]">{name}</span>
                  </div>
                ))
              ) : (
                Array.from({ length: params.kmeans_k }, (_, i) => (
                  <div key={i} className="flex items-center gap-1.5 text-slate-300">
                    <span
                      className="w-2.5 h-2.5 rounded-full"
                      style={{ backgroundColor: CLUSTER_COLORS[i % CLUSTER_COLORS.length] }}
                    />
                    <span className="text-[11px]">C-{i}</span>
                  </div>
                ))
              )}
            </div>
          </div>

          {/* Cluster Size Distribution Bar Chart */}
          {data?.kmeans?.cluster_sizes && (
            <div className="p-6 rounded-2xl bg-[#0F172A]/70 backdrop-blur-xl border border-slate-800/80 shadow-xl space-y-3">
              <div className="flex items-center justify-between">
                <div>
                  <h4 className="text-sm font-bold text-white tracking-tight">Cluster Population Distribution</h4>
                  <p className="text-xs text-slate-400">Total samples allocated across the {params.kmeans_k} clusters</p>
                </div>
                <span className="text-xs text-slate-400 font-medium">
                  Total: {data.comparison?.total_samples?.toLocaleString() || 27000} patches
                </span>
              </div>

              <div className="h-48 w-full">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={data.kmeans.cluster_sizes}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#1E293B" />
                    <XAxis
                      dataKey="cluster"
                      tickFormatter={(c) => `C-${c}`}
                      stroke="#64748B"
                      fontSize={11}
                    />
                    <YAxis stroke="#64748B" fontSize={11} />
                    <Tooltip
                      contentStyle={{
                        backgroundColor: '#0F172A',
                        borderColor: '#334155',
                        borderRadius: '8px',
                        fontSize: '12px'
                      }}
                      formatter={(val) => [`${val.toLocaleString()} patches`, 'Population']}
                      labelFormatter={(c) => `Cluster ${c}`}
                    />
                    <Bar dataKey="size" radius={[4, 4, 0, 0]}>
                      {data.kmeans.cluster_sizes.map((entry, index) => (
                        <Cell
                          key={`cell-${index}`}
                          fill={CLUSTER_COLORS[entry.cluster % CLUSTER_COLORS.length]}
                        />
                      ))}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </div>
          )}

          {/* Methodology Guidance Card */}
          <div className="p-4 rounded-xl bg-blue-950/20 border border-blue-500/20 flex items-start gap-3 text-xs text-blue-200">
            <Info className="w-4 h-4 text-blue-400 shrink-0 mt-0.5" />
            <div className="space-y-1">
              <span className="font-bold text-blue-300">How to interpret this bench: </span>
              <p className="text-slate-300 leading-relaxed">
                Decreasing PCA components eliminates noise while preserving dominant luminance variance. Increasing K reveals finer intra-class sub-types (e.g. deciduous vs evergreen forest) at the expense of cluster purity. Observe how cluster boundaries align between linear (PCA) and manifold (t-SNE) spaces.
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
