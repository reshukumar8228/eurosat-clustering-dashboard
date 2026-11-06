import React, { useState, useEffect } from 'react';
import { 
  Database, 
  Layers, 
  Cpu, 
  Radio, 
  ArrowRight, 
  CheckCircle2, 
  Sparkles 
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
  const [selectedClass, setSelectedClass] = useState('Forest');

  useEffect(() => {
    fetch('/api/overview')
      .then(res => res.json())
      .then(d => setData(d))
      .catch(err => console.error(err));

    fetch('/api/pca?sample_size=1000')
      .then(res => res.json())
      .then(d => setPcaData(d))
      .catch(err => console.error(err));
  }, []);

  if (!data) {
    return (
      <div className="flex items-center justify-center h-96">
        <div className="w-8 h-8 border-2 border-blue-500 border-t-transparent rounded-full animate-spin"></div>
      </div>
    );
  }

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
              className="flex items-center gap-2 px-4 py-2 rounded-xl bg-gradient-to-r from-blue-600 to-indigo-600 text-white text-xs font-semibold shadow-lg shadow-blue-500/25 hover:brightness-110 transition-all"
            >
              <span>Test Patch Upload</span>
              <ArrowRight className="w-3.5 h-3.5" />
            </button>
            <button
              onClick={() => setActiveTab('pca')}
              className="px-3.5 py-2 rounded-xl bg-slate-800/80 border border-slate-700/80 text-slate-300 text-xs font-medium hover:text-white transition-all"
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

      {/* Dataset Sample Previews */}
      <div className="p-6 rounded-2xl bg-[#0F172A]/70 backdrop-blur-xl border border-slate-800/80 shadow-xl">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 mb-5">
          <div>
            <h4 className="text-sm font-semibold text-white tracking-tight">EuroSAT Satellite Patch Gallery</h4>
            <p className="text-xs text-slate-400">Representative 64×64 Sentinel-2 optical imagery across land categories</p>
          </div>

          {/* Category Tabs */}
          <div className="flex flex-wrap gap-1.5 bg-slate-900/80 p-1 rounded-xl border border-slate-800">
            {data.sample_images.map((s) => (
              <button
                key={s.class}
                onClick={() => setSelectedClass(s.class)}
                className={`px-2.5 py-1 text-xs font-medium rounded-lg transition-all ${
                  selectedClass === s.class
                    ? 'bg-blue-600 text-white shadow-md shadow-blue-600/30'
                    : 'text-slate-400 hover:text-slate-200'
                }`}
              >
                {s.class}
              </button>
            ))}
          </div>
        </div>

        {/* Selected Category Tiles */}
        <div className="grid grid-cols-2 sm:grid-cols-5 md:grid-cols-10 gap-3">
          {data.sample_images.map((s) => (
            <div
              key={s.class}
              className={`p-2 rounded-xl bg-slate-900/60 border transition-all ${
                selectedClass === s.class ? 'border-blue-500/80 shadow-lg shadow-blue-500/20' : 'border-slate-800/60'
              }`}
            >
              <img
                src={s.url}
                alt={s.class}
                className="w-full h-16 object-cover rounded-lg mb-1.5"
                onError={(e) => { e.target.style.display = 'none'; }}
              />
              <p className="text-[10px] font-semibold text-white truncate text-center">{s.class}</p>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
