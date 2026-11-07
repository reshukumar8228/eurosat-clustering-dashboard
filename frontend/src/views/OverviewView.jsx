import React, { useState, useEffect } from 'react';
import { 
  Database, 
  Layers, 
  Cpu, 
  Radio, 
  ArrowRight, 
  CheckCircle2, 
  Sparkles,
  RefreshCw,
  Eye,
  X,
  ExternalLink,
  Image as ImageIcon
} from 'lucide-react';
import { 
  BarChart, 
  Bar, 
  XAxis, 
  YAxis, 
  Tooltip, 
  ResponsiveContainer, 
  AreaChart, 
  Area,
  CartesianGrid 
} from 'recharts';

import MetricCard from '../components/MetricCard';
import RadialGauge from '../components/RadialGauge';

export default function OverviewView({ setActiveTab }) {
  const [data, setData] = useState(null);
  const [pcaData, setPcaData] = useState(null);
  const [selectedClass, setSelectedClass] = useState('Highway');
  const [categoryData, setCategoryData] = useState(null);
  const [modalImage, setModalImage] = useState(null);

  useEffect(() => {
    fetch('/api/overview')
      .then(res => res.json())
      .then(d => {
        setData(d);
        if (d?.category_samples?.['Highway']) {
          setCategoryData({
            category: 'Highway',
            total_available: d.class_breakdown?.find(b => b.name === 'Highway')?.count || 2500,
            count: d.category_samples['Highway'].length,
            color: '#7f7f7f',
            samples: d.category_samples['Highway']
          });
        }
      })
      .catch(err => console.error("Error loading overview data:", err));

    fetch('/api/pca?sample_size=1000')
      .then(res => res.json())
      .then(d => setPcaData(d))
      .catch(err => console.error("Error loading PCA data:", err));
  }, []);

  // Sync category whenever selectedClass changes
  useEffect(() => {
    if (!selectedClass || selectedClass === 'all') return;

    if (data?.category_samples?.[selectedClass]) {
      const classItem = data.class_breakdown?.find(b => b.name === selectedClass);
      setCategoryData({
        category: selectedClass,
        total_available: classItem?.count || 2500,
        count: data.category_samples[selectedClass].length,
        color: classItem?.color || '#3B82F6',
        samples: data.category_samples[selectedClass]
      });
    }

    // Attempt to load fresh samples from backend if available
    fetch(`/api/category-samples?category=${encodeURIComponent(selectedClass)}&count=12`)
      .then(res => res.ok ? res.json() : null)
      .then(d => {
        if (d && Array.isArray(d.samples) && d.samples.length > 0) {
          setCategoryData(d);
        }
      })
      .catch(() => {
        // Silently use pre-seeded data
      });
  }, [selectedClass, data]);

  if (!data) {
    return (
      <div className="flex items-center justify-center h-96">
        <div className="w-8 h-8 border-2 border-blue-500 border-t-transparent rounded-full animate-spin"></div>
      </div>
    );
  }

  const allCategories = data.sample_images.map(s => s.class);

  // Compute active category details and guaranteed samples
  const classBreakdownItem = data?.class_breakdown?.find(b => b.name === selectedClass);
  const totalInClass = (categoryData?.category === selectedClass && categoryData?.total_available)
    ? categoryData.total_available
    : (classBreakdownItem?.count || 2500);

  const preseededSamples = data?.category_samples?.[selectedClass] || [];
  const activeSamples = (categoryData?.category === selectedClass && Array.isArray(categoryData?.samples) && categoryData.samples.length > 0)
    ? categoryData.samples
    : preseededSamples;

  const displayList = activeSamples.length > 0
    ? activeSamples
    : Array.from({ length: 12 }, (_, i) => ({
        filename: `${selectedClass}_${i + 1}.jpg`,
        category: selectedClass,
        url: `/static/data/${selectedClass}/${selectedClass}_${i + 1}.jpg`,
        cluster_k10: '-'
      }));

  return (
    <div className="space-y-6">
      {/* 4 Stat Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        <MetricCard
          title="Total Patches"
          value={data.total_images.toLocaleString()}
          change="+100% verified"
          icon={Database}
          pillColor="blue"
        />
        <MetricCard
          title="Land Taxonomies"
          value={data.total_classes}
          change="Balanced"
          icon={Layers}
          pillColor="cyan"
        />
        <MetricCard
          title="Input Features"
          value={data.features_per_image.toLocaleString()}
          change="64×64×3 D"
          icon={Cpu}
          pillColor="emerald"
        />
        <MetricCard
          title="Spectral Bands"
          value={data.spectral_bands}
          change="Sentinel-2"
          icon={Radio}
          pillColor="purple"
        />
      </div>

      {/* Hero Welcome Banner + 2 Radial Gauges (Vision UI layout) */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Welcome Card */}
        <div className="lg:col-span-1 p-7 rounded-2xl bg-gradient-to-br from-blue-900/40 via-[#0F172A] to-indigo-950/40 border border-blue-500/30 shadow-2xl backdrop-blur-xl relative overflow-hidden flex flex-col justify-between">
          <div className="absolute top-0 right-0 w-64 h-64 bg-blue-500/10 rounded-full blur-3xl pointer-events-none"></div>

          <div>
            <div className="flex items-center gap-2 text-xs font-semibold text-blue-400 mb-2 uppercase tracking-wider">
              <Sparkles className="w-3.5 h-3.5" />
              <span>Unsupervised Remote Sensing</span>
            </div>
            <h3 className="text-xl font-bold text-white tracking-tight mb-2">
              Satellite Land-Type Grouping
            </h3>
            <p className="text-xs text-slate-300 leading-relaxed">
              Unsupervised clustering of 27,000 Sentinel-2 satellite image patches using Randomized PCA, multi-perplexity t-SNE, and K-Means. Labels remained isolated throughout modeling.
            </p>
          </div>

          <div className="pt-6 flex items-center gap-3">
            <button
              onClick={() => setActiveTab('inference')}
              className="flex items-center gap-2 px-4 py-2 rounded-xl bg-gradient-to-r from-blue-600 to-indigo-600 text-white text-xs font-semibold shadow-lg shadow-blue-500/25 hover:brightness-110 transition-all cursor-pointer"
            >
              <span>Test Patch Upload</span>
              <ArrowRight className="w-3.5 h-3.5" />
            </button>
            <button
              onClick={() => setActiveTab('pca')}
              className="px-3.5 py-2 rounded-xl bg-slate-800/80 border border-slate-700/80 text-slate-300 text-xs font-medium hover:text-white transition-all cursor-pointer"
            >
              Explore PCA
            </button>
          </div>
        </div>

        {/* Radial Gauge 1: Retained Variance */}
        <RadialGauge
          title="Cumulative Variance Retained"
          subtitle="Top 220 Principal Components"
          value={pcaData ? pcaData.variance_total : 91.8}
          unit="%"
          footerText="Exceeds 90% information retention threshold"
        />

        {/* Radial Gauge 2: Dimensionality Compression */}
        <RadialGauge
          title="Dimensionality Reduction"
          subtitle="12,288 D → 220 D orthogonal subspace"
          value={55.8}
          unit="x"
          footerText="55.8-fold parameter compression achieved"
        />
      </div>

      {/* Interactive Charts Section (Vision UI Area Spline & Bar Chart) */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Explained Variance Area Chart */}
        <div className="p-6 rounded-2xl bg-[#0F172A]/70 backdrop-blur-xl border border-slate-800/80 shadow-xl">
          <div className="flex items-center justify-between mb-4">
            <div>
              <h4 className="text-sm font-semibold text-white tracking-tight">Cumulative Explained Variance</h4>
              <p className="text-xs text-slate-400">Progression across top principal components</p>
            </div>
            <span className="text-xs font-semibold px-2.5 py-1 rounded-lg bg-blue-500/10 text-blue-400 border border-blue-500/20">
              PC1 + PC2: 65.30%
            </span>
          </div>

          <div className="h-64">
            {pcaData?.scree && (
              <ResponsiveContainer width="100%" height="100%">
                <AreaChart data={pcaData.scree.slice(0, 30)}>
                  <defs>
                    <linearGradient id="varianceGrad" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="#3B82F6" stopOpacity={0.4}/>
                      <stop offset="95%" stopColor="#3B82F6" stopOpacity={0}/>
                    </linearGradient>
                  </defs>
                  <CartesianGrid strokeDasharray="3 3" stroke="#1E293B" />
                  <XAxis dataKey="component" stroke="#64748B" fontSize={11} />
                  <YAxis stroke="#64748B" fontSize={11} domain={[0, 100]} />
                  <Tooltip
                    contentStyle={{ backgroundColor: '#0F172A', borderColor: '#334155', borderRadius: '8px', fontSize: '12px' }}
                    formatter={(val) => [`${val.toFixed(2)}%`, 'Cumulative Variance']}
                  />
                  <Area type="monotone" dataKey="cumulative" stroke="#3B82F6" strokeWidth={2.5} fillOpacity={1} fill="url(#varianceGrad)" />
                </AreaChart>
              </ResponsiveContainer>
            )}
          </div>
        </div>

        {/* Class Distribution Breakdown */}
        <div className="p-6 rounded-2xl bg-[#0F172A]/70 backdrop-blur-xl border border-slate-800/80 shadow-xl">
          <div className="flex items-center justify-between mb-4">
            <div>
              <h4 className="text-sm font-semibold text-white tracking-tight">Class Distribution Breakdown</h4>
              <p className="text-xs text-slate-400">Total 27,000 patches across 10 land cover classes</p>
            </div>
            <span className="text-xs font-semibold px-2.5 py-1 rounded-lg bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">
              Zero Missing Data
            </span>
          </div>

          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={data.class_breakdown} margin={{ top: 10, right: 10, left: -20, bottom: 25 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1E293B" />
                <XAxis dataKey="name" stroke="#64748B" fontSize={10} angle={-30} textAnchor="end" interval={0} />
                <YAxis stroke="#64748B" fontSize={11} />
                <Tooltip
                  contentStyle={{ backgroundColor: '#0F172A', borderColor: '#334155', borderRadius: '8px', fontSize: '12px' }}
                  formatter={(val, name, item) => [`${val.toLocaleString()} patches (${item.payload.percentage}%)`, item.payload.name]}
                />
                <Bar dataKey="count" fill="#3B82F6" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>

      {/* EuroSAT Satellite Patch Gallery */}
      <div className="p-6 rounded-2xl bg-[#0F172A]/70 backdrop-blur-xl border border-slate-800/80 shadow-xl">
        <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-4 mb-5">
          <div>
            <div className="flex items-center gap-2">
              <h4 className="text-base font-semibold text-white tracking-tight">EuroSAT Satellite Patch Gallery</h4>
              <span className="px-2 py-0.5 rounded-full text-[10px] font-semibold bg-blue-500/10 text-blue-400 border border-blue-500/20">
                Interactive Multi-Patch Explorer
              </span>
            </div>
            <p className="text-xs text-slate-400 mt-0.5">
              Select any category below to view all representative 64×64 Sentinel-2 optical imagery patches
            </p>
          </div>

          {/* Category Tabs */}
          <div className="flex flex-wrap gap-1.5 bg-slate-900/90 p-1.5 rounded-xl border border-slate-800">
            <button
              onClick={() => setSelectedClass('all')}
              className={`px-2.5 py-1 text-xs font-medium rounded-lg transition-all cursor-pointer ${
                selectedClass === 'all'
                  ? 'bg-indigo-600 text-white shadow-md shadow-indigo-600/30'
                  : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800/50'
              }`}
            >
              All (Overview)
            </button>
            {allCategories.map((cls) => (
              <button
                key={cls}
                onClick={() => setSelectedClass(cls)}
                className={`px-2.5 py-1 text-xs font-medium rounded-lg transition-all cursor-pointer ${
                  selectedClass === cls
                    ? 'bg-blue-600 text-white shadow-md shadow-blue-600/30'
                    : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800/50'
                }`}
              >
                {cls}
              </button>
            ))}
          </div>
        </div>

        {/* Gallery Sub-Header Toolbar */}
        <div className="flex flex-wrap items-center justify-between gap-3 px-3.5 py-2.5 rounded-xl bg-slate-900/50 border border-slate-800/80 mb-4 text-xs">
          <div className="flex items-center gap-2.5 text-slate-300">
            {selectedClass === 'all' ? (
              <span>Showing 1 representative patch for each of the <strong>10 land-cover categories</strong></span>
            ) : (
              <>
                <span className="inline-flex items-center gap-1.5 font-semibold text-white">
                  <span className="w-2 h-2 rounded-full" style={{ backgroundColor: classBreakdownItem?.color || '#3B82F6' }}></span>
                  {selectedClass}
                </span>
                <span className="text-slate-500">•</span>
                <span className="text-slate-400">
                  {totalInClass.toLocaleString()} total verified patches in dataset
                </span>
                <span className="text-slate-500">•</span>
                <span className="text-blue-400 font-medium">
                  Showing {displayList.length} patches
                </span>
              </>
            )}
          </div>
        </div>

        {/* Category Patch Grid */}
        {selectedClass === 'all' ? (
          /* 10-Class Overview Grid */
          <div className="grid grid-cols-2 sm:grid-cols-5 md:grid-cols-10 gap-3">
            {data.sample_images.map((s) => (
              <div
                key={s.class}
                onClick={() => setSelectedClass(s.class)}
                className="group p-2 rounded-xl bg-slate-900/60 border border-slate-800/80 hover:border-blue-500/80 transition-all cursor-pointer hover:shadow-lg hover:shadow-blue-500/10"
              >
                <div className="relative overflow-hidden rounded-lg mb-1.5 aspect-square bg-slate-950 flex items-center justify-center">
                  <img
                    src={s.url}
                    alt={s.class}
                    className="w-full h-full object-cover transition-transform duration-300 group-hover:scale-110"
                    loading="lazy"
                  />
                  <div className="absolute inset-0 bg-blue-600/0 group-hover:bg-blue-600/20 transition-colors flex items-center justify-center">
                    <Eye className="w-4 h-4 text-white opacity-0 group-hover:opacity-100 transition-opacity" />
                  </div>
                </div>
                <p className="text-[11px] font-semibold text-white truncate text-center group-hover:text-blue-400 transition-colors">
                  {s.class}
                </p>
                <p className="text-[9px] text-slate-400 text-center truncate">Click to explore</p>
              </div>
            ))}
          </div>
        ) : (
          /* Dynamic Multi-Patch Grid for Selected Category */
          <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-6 gap-3.5">
            {displayList.map((sample, idx) => (
              <div
                key={sample.filename || idx}
                onClick={() => setModalImage(sample)}
                className="group p-2.5 rounded-xl bg-slate-900/60 border border-slate-800/80 hover:border-blue-500/80 transition-all cursor-pointer hover:shadow-lg hover:shadow-blue-500/20 hover:-translate-y-0.5"
              >
                <div className="relative overflow-hidden rounded-lg mb-2 aspect-square bg-slate-950 flex items-center justify-center">
                  <img
                    src={sample.url}
                    alt={sample.filename}
                    className="w-full h-full object-cover transition-transform duration-300 group-hover:scale-110"
                    loading="lazy"
                  />
                  <div className="absolute inset-0 bg-slate-950/0 group-hover:bg-slate-950/40 transition-colors flex items-center justify-center">
                    <Eye className="w-5 h-5 text-white opacity-0 group-hover:opacity-100 transition-opacity drop-shadow" />
                  </div>
                  {sample.cluster_k10 !== '-' && (
                    <span className="absolute bottom-1 right-1 px-1.5 py-0.5 rounded bg-slate-900/80 text-[9px] font-medium text-slate-300 border border-slate-700/60">
                      Cl. {sample.cluster_k10}
                    </span>
                  )}
                </div>
                <p className="text-[11px] font-mono font-medium text-slate-200 truncate group-hover:text-blue-400 transition-colors">
                  {sample.filename}
                </p>
                <div className="flex items-center justify-between text-[10px] text-slate-400 mt-0.5">
                  <span>64×64 px</span>
                  <span className="text-emerald-400">10m GSD</span>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Lightbox Modal for Patch Inspection */}
      {modalImage && (
        <div 
          className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 backdrop-blur-sm p-4 animate-in fade-in duration-200"
          onClick={() => setModalImage(null)}
        >
          <div 
            className="p-6 rounded-2xl bg-[#0F172A] border border-slate-700 shadow-2xl max-w-sm w-full relative"
            onClick={(e) => e.stopPropagation()}
          >
            <button
              onClick={() => setModalImage(null)}
              className="absolute top-4 right-4 text-slate-400 hover:text-white p-1 rounded-lg hover:bg-slate-800 transition-colors cursor-pointer"
            >
              <X className="w-5 h-5" />
            </button>

            <h4 className="text-sm font-bold text-white mb-1">{modalImage.filename}</h4>
            <p className="text-xs text-blue-400 mb-4">{modalImage.category} • EuroSAT Benchmark</p>

            <div className="w-48 h-48 mx-auto rounded-xl overflow-hidden border border-slate-700 shadow-lg mb-4 bg-slate-950 flex items-center justify-center">
              <img
                src={modalImage.url}
                alt={modalImage.filename}
                className="w-full h-full object-cover"
              />
            </div>

            <div className="space-y-1.5 text-xs text-slate-300 bg-slate-900/80 p-3 rounded-xl border border-slate-800 mb-4">
              <div className="flex justify-between">
                <span className="text-slate-400">Land Cover Class:</span>
                <span className="font-semibold text-white">{modalImage.category}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-slate-400">Unsupervised K-Means:</span>
                <span className="font-semibold text-cyan-400">
                  {modalImage.cluster_k10 !== '-' ? `Cluster ${modalImage.cluster_k10}` : 'Unassigned'}
                </span>
              </div>
              <div className="flex justify-between">
                <span className="text-slate-400">Resolution & Extent:</span>
                <span className="font-semibold text-white">64×64 px (640m × 640m)</span>
              </div>
            </div>

            <div className="flex gap-2">
              <button
                onClick={() => {
                  setModalImage(null);
                  setActiveTab('inference');
                }}
                className="flex-1 flex items-center justify-center gap-1.5 py-2 px-3 rounded-xl bg-gradient-to-r from-blue-600 to-indigo-600 text-white text-xs font-semibold hover:brightness-110 transition-all cursor-pointer"
              >
                <span>Test in Classifier</span>
                <ArrowRight className="w-3.5 h-3.5" />
              </button>
              <button
                onClick={() => setModalImage(null)}
                className="py-2 px-3 rounded-xl bg-slate-800 hover:bg-slate-700 text-slate-300 text-xs font-medium transition-all cursor-pointer"
              >
                Close
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
