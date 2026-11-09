import { useState, useEffect, useMemo, memo } from 'react';
import { 
  AreaChart, 
  Area, 
  XAxis, 
  YAxis, 
  Tooltip, 
  ResponsiveContainer, 
  CartesianGrid 
} from 'recharts';
import fullScreeData from '../data/pca_scree_full.json';

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

// Memoized Spatial Map to avoid re-rendering 4,096 SVG rects on hover
const SpatialLoadingMap = memo(function SpatialLoadingMap({ matrix }) {
  return (
    <svg viewBox="0 0 64 64" className="w-full h-full rounded">
      {matrix.map((row, r) =>
        row.map((val, c) => (
          <rect
            key={`${r}-${c}`}
            x={c}
            y={r}
            width="1"
            height="1"
            fill={val > 0 ? `rgba(59, 130, 246, ${Math.min(1, val * 40)})` : '#070B14'}
          />
        ))
      )}
    </svg>
  );
});

// Memoized Scree Plot to avoid re-rendering Recharts on hover
const ScreePlot = memo(function ScreePlot({ screeData, screeMode }) {
  const xTicks = screeMode === '30' ? [1, 5, 10, 15, 20, 25, 30] : [1, 50, 100, 150, 200, 220];
  return (
    <div className="h-56">
      <ResponsiveContainer width="100%" height="100%">
        <AreaChart data={screeData}>
          <CartesianGrid strokeDasharray="3 3" stroke="#1E293B" />
          <XAxis dataKey="component" stroke="#64748B" fontSize={10} ticks={xTicks} interval={0} />
          <YAxis stroke="#64748B" fontSize={10} />
          <Tooltip
            contentStyle={{ backgroundColor: '#0F172A', borderColor: '#334155', borderRadius: '8px', fontSize: '12px' }}
            formatter={(val, name, item) => [
              `${typeof val === 'number' ? val.toFixed(2) : val}% (Cumulative: ${item?.payload?.cumulative ? item.payload.cumulative.toFixed(2) : ''}%)`,
              `PC ${item?.payload?.component || ''}`
            ]}
          />
          <Area type="monotone" dataKey="variance" stroke="#10B981" strokeWidth={2} fill="#10B981" fillOpacity={0.2} isAnimationActive={false} />
        </AreaChart>
      </ResponsiveContainer>
    </div>
  );
});

export default function PcaView() {
  const [data, setData] = useState(null);
  const [colorMode, setColorMode] = useState('label'); // 'label' or 'cluster'
  const [selectedPc, setSelectedPc] = useState('PC_1');
  const [hoveredPoint, setHoveredPoint] = useState(null);
  const [screeRange, setScreeRange] = useState('all');

  useEffect(() => {
    fetch('/api/pca?sample_size=2000')
      .then(res => res.json())
      .then(d => setData(d))
      .catch(err => console.error(err));
  }, []);

  const pcInfo = data?.loadings?.[selectedPc];

  // Pre-calculate spatial activation map for selected PC (memoized)
  const spatialMatrix = useMemo(() => {
    const matrix = Array.from({ length: 64 }, () => Array(64).fill(0));
    if (pcInfo?.top_loadings) {
      pcInfo.top_loadings.forEach(item => {
        matrix[item.row][item.col] = Math.max(matrix[item.row][item.col], item.abs_weight);
      });
    }
    return matrix;
  }, [pcInfo]);

  // Find min/max and pre-calculate canvas projection coordinates (memoized)
  const { projectedPoints } = useMemo(() => {
    if (!data?.scatter_points?.length) {
      return { projectedPoints: [] };
    }
    const xVals = data.scatter_points.map(p => p.x);
    const yVals = data.scatter_points.map(p => p.y);
    const minX = Math.min(...xVals);
    const maxX = Math.max(...xVals);
    const minY = Math.min(...yVals);
    const maxY = Math.max(...yVals);
    const dx = maxX - minX || 1;
    const dy = maxY - minY || 1;

    const projected = data.scatter_points.map(p => ({
      ...p,
      cx: 50 + ((p.x - minX) / dx) * 700,
      cy: 420 - ((p.y - minY) / dy) * 390
    }));

    return { projectedPoints: projected };
  }, [data]);

  const screeSlice = useMemo(() => {
    const fullScree = (data?.scree && data.scree.length >= 220) ? data.scree : fullScreeData;
    return screeRange === '30' ? fullScree.slice(0, 30) : fullScree;
  }, [data?.scree, screeRange]);

  if (!data) {
    return (
      <div className="flex items-center justify-center h-96">
        <div className="w-8 h-8 border-2 border-blue-500 border-t-transparent rounded-full animate-spin"></div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Top Variance Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        <div className="p-5 rounded-2xl bg-[#0F172A]/70 backdrop-blur-xl border border-slate-800/80 shadow-xl">
          <p className="text-xs font-semibold text-slate-400 mb-1 uppercase tracking-wider">PC1 Explained</p>
          <h3 className="text-2xl font-bold text-white tracking-tight">{data.variance_pc1}%</h3>
          <p className="text-xs text-blue-400 mt-1">Primary luminance vector</p>
        </div>

        <div className="p-5 rounded-2xl bg-[#0F172A]/70 backdrop-blur-xl border border-slate-800/80 shadow-xl">
          <p className="text-xs font-semibold text-slate-400 mb-1 uppercase tracking-wider">Top 2 PCs Combined</p>
          <h3 className="text-2xl font-bold text-white tracking-tight">{data.variance_top2}%</h3>
          <p className="text-xs text-emerald-400 mt-1">Majority scene variance</p>
        </div>

        <div className="p-5 rounded-2xl bg-[#0F172A]/70 backdrop-blur-xl border border-slate-800/80 shadow-xl">
          <p className="text-xs font-semibold text-slate-400 mb-1 uppercase tracking-wider">Total Fitted Components</p>
          <h3 className="text-2xl font-bold text-white tracking-tight">{data.total_components}</h3>
          <p className="text-xs text-slate-400 mt-1">Dimensionality reduced</p>
        </div>

        <div className="p-5 rounded-2xl bg-[#0F172A]/70 backdrop-blur-xl border border-slate-800/80 shadow-xl">
          <p className="text-xs font-semibold text-slate-400 mb-1 uppercase tracking-wider">Total Variance Retained</p>
          <h3 className="text-2xl font-bold text-white tracking-tight">{data.variance_total}%</h3>
          <p className="text-xs text-cyan-400 mt-1">Preserves dataset geometry</p>
        </div>
      </div>

      {/* 2D PCA Interactive Scatter Canvas */}
      <div className="p-6 rounded-2xl bg-[#0F172A]/70 backdrop-blur-xl border border-slate-800/80 shadow-xl">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 mb-4">
          <div>
            <h4 className="text-sm font-semibold text-white tracking-tight">Interactive 2D Principal Component Space</h4>
            <p className="text-xs text-slate-400">Projection of EuroSAT patches onto PC1 vs PC2 (2,000 stratified points)</p>
          </div>

          <div className="flex items-center gap-3">
            <span className="text-xs text-slate-400 font-medium">Color by:</span>
            <div className="flex bg-slate-900/80 p-1 rounded-xl border border-slate-800">
              <button
                onClick={() => setColorMode('label')}
                className={`px-3 py-1 rounded-lg text-xs font-medium transition-all ${
                  colorMode === 'label' ? 'bg-blue-600 text-white shadow-md' : 'text-slate-400 hover:text-white'
                }`}
              >
                True Class (Post-Hoc)
              </button>
              <button
                onClick={() => setColorMode('cluster')}
                className={`px-3 py-1 rounded-lg text-xs font-medium transition-all ${
                  colorMode === 'cluster' ? 'bg-blue-600 text-white shadow-md' : 'text-slate-400 hover:text-white'
                }`}
              >
                K-Means Cluster
              </button>
            </div>
          </div>
        </div>

        {/* 2D Scatter Canvas Container */}
        <div className="relative h-96 w-full bg-[#070B14] rounded-xl border border-slate-800/80 overflow-hidden flex items-center justify-center p-4">
          <svg className="w-full h-full" viewBox="0 0 800 450">
            {/* Grid lines */}
            <line x1="50" y1="225" x2="750" y2="225" stroke="#1E293B" strokeWidth="1" strokeDasharray="4 4" />
            <line x1="400" y1="30" x2="400" y2="420" stroke="#1E293B" strokeWidth="1" strokeDasharray="4 4" />

            {/* Scatter points */}
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
                  opacity={hoveredPoint ? (isHovered ? 1 : 0.45) : 0.75}
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

          {/* Hover Card */}
          {hoveredPoint && (
            <div className="absolute top-4 right-4 bg-slate-900/90 border border-slate-700/80 p-3 rounded-xl text-xs text-white shadow-2xl backdrop-blur-md pointer-events-none">
              <p className="font-semibold text-blue-400">{hoveredPoint.filename}</p>
              <p className="text-slate-300">Class: <span className="text-white font-medium">{hoveredPoint.label}</span></p>
              <p className="text-slate-300">Cluster: <span className="text-white font-medium">Cluster {hoveredPoint.cluster}</span></p>
              <p className="text-slate-400 mt-1">
                PC1: {typeof hoveredPoint.x === 'number' ? hoveredPoint.x.toFixed(2) : hoveredPoint.x} | PC2: {typeof hoveredPoint.y === 'number' ? hoveredPoint.y.toFixed(2) : hoveredPoint.y}
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

      {/* Spatial Feature Loadings & Scree Section */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Spatial Feature Loadings */}
        <div className="p-6 rounded-2xl bg-[#0F172A]/70 backdrop-blur-xl border border-slate-800/80 shadow-xl">
          <div className="flex items-center justify-between mb-4">
            <div>
              <h4 className="text-sm font-semibold text-white tracking-tight">Spatial Pixel Loading Map (64×64)</h4>
              <p className="text-xs text-slate-400">Pixel regions contributing highest absolute weight</p>
            </div>

            <select
              value={selectedPc}
              onChange={(e) => setSelectedPc(e.target.value)}
              className="bg-slate-900 border border-slate-800 rounded-lg px-2.5 py-1 text-xs text-white focus:outline-none"
            >
              {Array.from({ length: 10 }).map((_, i) => (
                <option key={i} value={`PC_${i + 1}`}>{`PC ${i + 1}`}</option>
              ))}
            </select>
          </div>

          <div className="flex flex-col sm:flex-row items-center gap-6">
            <div className="w-48 h-48 bg-slate-950 rounded-xl border border-slate-800 p-2 flex items-center justify-center">
              <SpatialLoadingMap matrix={spatialMatrix} />
            </div>

            <div className="flex-1 space-y-2">
              <div className="text-xs text-slate-300">
                <p className="font-semibold text-white mb-1">{selectedPc} Weight Distribution</p>
                <p className="text-slate-400 leading-relaxed">
                  Individual Variance: <span className="text-blue-400 font-medium">{(pcInfo?.explained_variance_ratio * 100).toFixed(2)}%</span>
                </p>
                <p className="text-slate-400 leading-relaxed">
                  Cumulative Variance: <span className="text-emerald-400 font-medium">{(pcInfo?.cumulative_variance_ratio * 100).toFixed(2)}%</span>
                </p>
              </div>

              <div className="pt-2 border-t border-slate-800/80">
                <span className="text-[11px] font-semibold text-slate-400 uppercase tracking-wider">Channel Drivers:</span>
                <div className="flex items-center gap-3 mt-1.5 text-xs">
                  <span className="flex items-center gap-1 text-red-400"><span className="w-2 h-2 rounded-full bg-red-500"></span>Red</span>
                  <span className="flex items-center gap-1 text-emerald-400"><span className="w-2 h-2 rounded-full bg-emerald-500"></span>Green</span>
                  <span className="flex items-center gap-1 text-blue-400"><span className="w-2 h-2 rounded-full bg-blue-500"></span>Blue</span>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Scree Plot */}
        <div className="p-6 rounded-2xl bg-[#0F172A]/70 backdrop-blur-xl border border-slate-800/80 shadow-xl">
          <div className="flex items-start justify-between gap-3 mb-4">
            <div className="min-w-0 flex-1">
              <h4 className="text-sm font-semibold text-white tracking-tight">Scree Variance Decomposition</h4>
              <p className="text-xs text-slate-400 truncate">
                {screeRange === 'all' 
                  ? 'All 220 components (cumulative 91.84%)' 
                  : 'Individual variance of top 30 principal components'}
              </p>
            </div>
            <div className="flex items-center gap-2 shrink-0">
              <div className="flex items-center bg-slate-900/90 p-0.5 rounded-lg border border-slate-800 text-xs">
                <button
                  type="button"
                  onClick={() => setScreeRange('30')}
                  className={`px-2 py-0.5 rounded font-medium transition-all ${
                    screeRange === '30' 
                      ? 'bg-blue-600 text-white shadow-sm' 
                      : 'text-slate-400 hover:text-slate-200'
                  }`}
                >
                  Top 30
                </button>
                <button
                  type="button"
                  onClick={() => setScreeRange('all')}
                  className={`px-2 py-0.5 rounded font-medium transition-all ${
                    screeRange === 'all' 
                      ? 'bg-blue-600 text-white shadow-sm' 
                      : 'text-slate-400 hover:text-slate-200'
                  }`}
                >
                  All 220
                </button>
              </div>

              <span className="text-xs font-semibold px-2.5 py-1 rounded-lg bg-emerald-500/10 text-emerald-400 border border-emerald-500/20 whitespace-nowrap">
                {screeRange === 'all' ? `Total Variance: ${data?.variance_total ?? 91.84}%` : `PC1: ${data?.variance_pc1}%`}
              </span>
            </div>
          </div>

          <ScreePlot screeData={screeSlice} screeMode={screeRange} />
        </div>
      </div>
    </div>
  );
}
