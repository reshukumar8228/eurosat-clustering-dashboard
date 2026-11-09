import { useState, useEffect, useMemo } from 'react';
import { AlertTriangle } from 'lucide-react';

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
  '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
];

export default function TsneView() {
  const [data, setData] = useState(null);
  const [perplexity, setPerplexity] = useState(30);
  const [colorMode, setColorMode] = useState('label');
  const [hoveredPoint, setHoveredPoint] = useState(null);

  useEffect(() => {
    fetch(`/api/tsne?perplexity=${perplexity}`)
      .then(res => res.json())
      .then(d => setData(d))
      .catch(err => console.error(err));
  }, [perplexity]);

  const { projectedPoints } = useMemo(() => {
    if (!data?.points?.length) return { projectedPoints: [] };
    const xVals = data.points.map(p => p.x);
    const yVals = data.points.map(p => p.y);
    const minX = Math.min(...xVals);
    const maxX = Math.max(...xVals);
    const minY = Math.min(...yVals);
    const maxY = Math.max(...yVals);
    const dx = maxX - minX || 1;
    const dy = maxY - minY || 1;

    const projected = data.points.map(p => ({
      ...p,
      cx: 50 + ((p.x - minX) / dx) * 700,
      cy: 420 - ((p.y - minY) / dy) * 390
    }));

    return { projectedPoints: projected };
  }, [data]);

  if (!data) {
    return (
      <div className="flex items-center justify-center h-96">
        <div className="w-8 h-8 border-2 border-blue-500 border-t-transparent rounded-full animate-spin"></div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Top Parameter & Metrics Row */}
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
        <div className="p-5 rounded-2xl bg-[#0F172A]/70 backdrop-blur-xl border border-slate-800/80 shadow-xl">
          <p className="text-xs font-semibold text-slate-400 mb-1 uppercase tracking-wider">Selected Perplexity</p>
          <div className="flex items-baseline gap-2">
            <h3 className="text-2xl font-bold text-white tracking-tight">{data.perplexity}</h3>
            <span className="text-xs text-blue-400">Effective Neighbors</span>
          </div>
        </div>

        <div className="p-5 rounded-2xl bg-[#0F172A]/70 backdrop-blur-xl border border-slate-800/80 shadow-xl">
          <p className="text-xs font-semibold text-slate-400 mb-1 uppercase tracking-wider">Final KL Divergence</p>
          <div className="flex items-baseline gap-2">
            <h3 className="text-2xl font-bold text-white tracking-tight">{data.kl_divergence}</h3>
            <span className="text-xs text-emerald-400">Converged (1,000 iter)</span>
          </div>
        </div>

        <div className="p-5 rounded-2xl bg-[#0F172A]/70 backdrop-blur-xl border border-slate-800/80 shadow-xl">
          <p className="text-xs font-semibold text-slate-400 mb-1 uppercase tracking-wider">Manifold Points</p>
          <div className="flex items-baseline gap-2">
            <h3 className="text-2xl font-bold text-white tracking-tight">{data.points.length.toLocaleString()}</h3>
            <span className="text-xs text-slate-400">Stratified Subsample</span>
          </div>
        </div>
      </div>

      {/* Interactive Perplexity Selector & Controls */}
      <div className="p-6 rounded-2xl bg-[#0F172A]/70 backdrop-blur-xl border border-slate-800/80 shadow-xl">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 mb-5">
          <div>
            <h4 className="text-sm font-semibold text-white tracking-tight">t-SNE Non-Linear Manifold Projection</h4>
            <p className="text-xs text-slate-400">Inspect neighborhood structure as perplexity changes from 5 to 100</p>
          </div>

          <div className="flex flex-wrap items-center gap-4">
            {/* Perplexity Buttons */}
            <div className="flex items-center gap-1.5 bg-slate-900/80 p-1 rounded-xl border border-slate-800">
              {data.available_perplexities.map((p) => (
                <button
                  key={p}
                  onClick={() => setPerplexity(p)}
                  className={`px-3 py-1 rounded-lg text-xs font-semibold transition-all ${
                    perplexity === p ? 'bg-blue-600 text-white shadow-md shadow-blue-500/25' : 'text-slate-400 hover:text-white'
                  }`}
                >
                  p = {p}
                </button>
              ))}
            </div>

            {/* Color Mode */}
            <div className="flex items-center gap-1 bg-slate-900/80 p-1 rounded-xl border border-slate-800">
              <button
                onClick={() => setColorMode('label')}
                className={`px-2.5 py-1 rounded-lg text-xs font-medium transition-all ${
                  colorMode === 'label' ? 'bg-blue-600 text-white' : 'text-slate-400'
                }`}
              >
                True Class
              </button>
              <button
                onClick={() => setColorMode('cluster')}
                className={`px-2.5 py-1 rounded-lg text-xs font-medium transition-all ${
                  colorMode === 'cluster' ? 'bg-blue-600 text-white' : 'text-slate-400'
                }`}
              >
                Cluster
              </button>
            </div>
          </div>
        </div>

        {/* 2D t-SNE Canvas */}
        <div className="relative h-96 w-full bg-[#070B14] rounded-xl border border-slate-800/80 overflow-hidden flex items-center justify-center p-4">
          <svg className="w-full h-full" viewBox="0 0 800 450">
            {projectedPoints.map((p, idx) => {
              const fill = colorMode === 'label' ? CLASS_COLORS[p.label] || '#3B82F6' : CLUSTER_COLORS[p.cluster % 10];
              const isHovered = hoveredPoint?.filename === p.filename;

              return (
                <circle
                  key={idx}
                  cx={p.cx}
                  cy={p.cy}
                  r="3.5"
                  fill={fill}
                  opacity={hoveredPoint ? (isHovered ? 1 : 0.45) : 0.8}
                  className="cursor-pointer"
                  onMouseEnter={() => setHoveredPoint({ ...p, fill })}
                  onMouseLeave={() => setHoveredPoint(prev => (prev?.filename === p.filename ? null : prev))}
                />
              );
            })}

            {/* Active Hover Highlight Ring (rendered on top, non-blocking) */}
            {hoveredPoint && (
              <g className="pointer-events-none">
                <circle
                  cx={hoveredPoint.cx}
                  cy={hoveredPoint.cy}
                  r="7.5"
                  fill="none"
                  stroke="#FFFFFF"
                  strokeWidth="2.5"
                />
                <circle
                  cx={hoveredPoint.cx}
                  cy={hoveredPoint.cy}
                  r="4.5"
                  fill={hoveredPoint.fill}
                />
              </g>
            )}
          </svg>

          {hoveredPoint && (
            <div className="absolute top-4 right-4 bg-slate-900/90 border border-slate-700/80 p-3 rounded-xl text-xs text-white shadow-2xl backdrop-blur-md pointer-events-none">
              <p className="font-semibold text-blue-400">{hoveredPoint.filename}</p>
              <p className="text-slate-300">Class: <span className="text-white font-medium">{hoveredPoint.label}</span></p>
              <p className="text-slate-300">Cluster: <span className="text-white font-medium">Cluster {hoveredPoint.cluster}</span></p>
              <p className="text-slate-400 mt-1">
                t-SNE 1: {typeof hoveredPoint.x === 'number' ? hoveredPoint.x.toFixed(2) : hoveredPoint.x} | t-SNE 2: {typeof hoveredPoint.y === 'number' ? hoveredPoint.y.toFixed(2) : hoveredPoint.y}
              </p>
            </div>
          )}
        </div>

        {/* Legend */}
        <div className="flex flex-wrap items-center justify-center gap-4 mt-4 pt-4 border-t border-slate-800/60">
          {colorMode === 'label' ? (
            Object.entries(CLASS_COLORS).map(([name, color]) => (
              <div key={name} className="flex items-center gap-1.5 text-xs text-slate-300">
                <span className="w-2.5 h-2.5 rounded-full" style={{ backgroundColor: color }}></span>
                <span>{name}</span>
              </div>
            ))
          ) : (
            CLUSTER_COLORS.map((color, idx) => (
              <div key={idx} className="flex items-center gap-1.5 text-xs text-slate-300">
                <span className="w-2.5 h-2.5 rounded-full" style={{ backgroundColor: color }}></span>
                <span>Cluster {idx}</span>
              </div>
            ))
          )}
        </div>
      </div>

      {/* Critical Caution Card */}
      <div className="p-5 rounded-2xl bg-amber-950/20 border border-amber-500/30 flex items-start gap-3.5 text-xs text-amber-200">
        <AlertTriangle className="w-5 h-5 text-amber-400 shrink-0 mt-0.5" />
        <div>
          <span className="font-bold text-amber-300">Non-Metric Projection Caution: </span>
          Distances between isolated clusters in t-SNE projections do not correlate linearly with physical Euclidean distance. The scale of clusters is influenced by probability density equalization rather than physical patch diameter.
        </div>
      </div>
    </div>
  );
}
