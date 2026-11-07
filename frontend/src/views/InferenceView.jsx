import React, { useState, useRef } from 'react';
import { 
  UploadCloud, 
  ArrowRight
} from 'lucide-react';
import { 
  BarChart, 
  Bar, 
  XAxis, 
  YAxis, 
  Tooltip, 
  ResponsiveContainer, 
  CartesianGrid 
} from 'recharts';

export default function InferenceView() {
  const [previewUrl, setPreviewUrl] = useState(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const fileInputRef = useRef(null);

  const presets = [
    { name: 'Highway (EuroSAT Sample)', path: 'Data/EuroSAT/Highway/Highway_1.jpg' },
    { name: 'Forest (EuroSAT Sample)', path: 'Data/EuroSAT/Forest/Forest_1.jpg' },
    { name: 'Residential (EuroSAT Sample)', path: 'Data/EuroSAT/Residential/Residential_1.jpg' },
    { name: 'SeaLake (EuroSAT Sample)', path: 'Data/EuroSAT/SeaLake/SeaLake_1.jpg' },
    { name: 'Industrial (EuroSAT Sample)', path: 'Data/EuroSAT/Industrial/Industrial_1.jpg' },
    { name: 'AnnualCrop (EuroSAT Sample)', path: 'Data/EuroSAT/AnnualCrop/AnnualCrop_1.jpg' },
  ];

  const handleFileChange = (e) => {
    const selected = e.target.files[0];
    if (selected) {
      setPreviewUrl(URL.createObjectURL(selected));
      runPrediction(selected, null);
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      const dropped = e.dataTransfer.files[0];
      setPreviewUrl(URL.createObjectURL(dropped));
      runPrediction(dropped, null);
    }
  };

  const handlePresetSelect = (presetPath) => {
    setPreviewUrl(null);
    runPrediction(null, presetPath);
  };

  const runPrediction = async (fileObj, presetPath) => {
    setLoading(true);
    setResult(null);

    const formData = new FormData();
    if (fileObj) {
      formData.append('file', fileObj);
    } else if (presetPath) {
      formData.append('preset_path', presetPath);
    }

    try {
      const res = await fetch('/api/predict', {
        method: 'POST',
        body: formData,
      });
      const data = await res.json();
      setResult(data);
    } catch (err) {
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="space-y-6">
      {/* Upload Zone & Preset Selection */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Upload Zone */}
        <div
          onDragOver={(e) => e.preventDefault()}
          onDrop={handleDrop}
          onClick={() => fileInputRef.current?.click()}
          className="lg:col-span-2 p-8 rounded-2xl bg-[#0F172A]/70 backdrop-blur-xl border-2 border-dashed border-slate-700 hover:border-blue-500/60 transition-all cursor-pointer flex flex-col items-center justify-center text-center group shadow-xl"
        >
          <input
            type="file"
            ref={fileInputRef}
            onChange={handleFileChange}
            accept=".jpg,.jpeg,.png,.tif"
            className="hidden"
          />

          <div className="w-14 h-14 rounded-2xl bg-blue-600/10 border border-blue-500/20 flex items-center justify-center mb-3 group-hover:scale-110 transition-transform">
            <UploadCloud className="w-7 h-7 text-blue-400" />
          </div>
          <h4 className="text-sm font-semibold text-white mb-1">
            Drag & drop satellite patch, or <span className="text-blue-400">browse</span>
          </h4>
          <p className="text-xs text-slate-400">
            Supports RGB optical imagery (.jpg, .png) and multi-spectral (.tif) patches
          </p>
        </div>

        {/* Quick Presets Card */}
        <div className="p-6 rounded-2xl bg-[#0F172A]/70 backdrop-blur-xl border border-slate-800/80 shadow-xl flex flex-col justify-between">
          <div>
            <h4 className="text-sm font-semibold text-white tracking-tight mb-1">Quick-Test EuroSAT Presets</h4>
            <p className="text-xs text-slate-400 mb-3">Test with ground-truth satellite imagery</p>

            <div className="space-y-1.5">
              {presets.map((p) => (
                <button
                  key={p.name}
                  onClick={() => handlePresetSelect(p.path)}
                  className="w-full flex items-center justify-between px-3 py-2 rounded-xl text-xs font-medium bg-slate-900/60 border border-slate-800 hover:border-blue-500/40 hover:bg-slate-800/60 text-slate-300 hover:text-white transition-all text-left"
                >
                  <span>{p.name}</span>
                  <ArrowRight className="w-3.5 h-3.5 text-slate-500" />
                </button>
              ))}
            </div>
          </div>
        </div>
      </div>

      {/* Loading Spinner */}
      {loading && (
        <div className="p-12 text-center rounded-2xl bg-[#0F172A]/70 border border-slate-800/80">
          <div className="w-8 h-8 border-2 border-blue-500 border-t-transparent rounded-full animate-spin mx-auto mb-3"></div>
          <p className="text-xs text-slate-400">Projecting patch into 220D PCA space and evaluating cluster assignments...</p>
        </div>
      )}

      {/* Prediction Results Display */}
      {result && !loading && (
        <div className="space-y-6">
          {/* Main Inference Card */}
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            {/* Image Preview & Physical Characteristics */}
            <div className="p-6 rounded-2xl bg-[#0F172A]/70 backdrop-blur-xl border border-slate-800/80 shadow-xl flex flex-col items-center text-center">
              <div className="w-36 h-36 rounded-2xl overflow-hidden border-2 border-blue-500/40 shadow-2xl mb-4 bg-slate-950 flex items-center justify-center">
                <img
                  src={result.thumbnail_b64 || previewUrl}
                  alt="Uploaded patch"
                  className="w-full h-full object-cover"
                />
              </div>
              <h5 className="text-sm font-bold text-white truncate max-w-full px-2">{result.source_name}</h5>
              <p className="text-[11px] text-slate-400 mt-0.5">Input Image Patch</p>

              <div className="w-full pt-4 mt-4 border-t border-slate-800/80 space-y-2 text-left text-xs">
                <div className="flex justify-between text-slate-300">
                  <span className="text-slate-400">Mean Luminance:</span>
                  <span className="font-semibold text-white">{result.physical_stats.brightness}</span>
                </div>
                <div className="flex justify-between text-slate-300">
                  <span className="text-slate-400">Texture Roughness (σ):</span>
                  <span className="font-semibold text-white">{result.physical_stats.roughness}</span>
                </div>
                <div className="flex justify-between text-slate-300">
                  <span className="text-slate-400">RGB Balance:</span>
                  <span className="font-mono text-slate-200">
                    R:{result.physical_stats.red} G:{result.physical_stats.green} B:{result.physical_stats.blue}
                  </span>
                </div>
              </div>
            </div>

            {/* Classification & Cluster Detection */}
            <div className="lg:col-span-2 p-6 rounded-2xl bg-[#0F172A]/70 backdrop-blur-xl border border-slate-800/80 shadow-xl flex flex-col justify-between">
              <div>
                <div className="flex flex-wrap items-center justify-between gap-2 mb-4">
                  <span className="px-3 py-1 rounded-xl text-xs font-semibold bg-blue-500/10 border border-blue-500/30 text-blue-400">
                    Cluster {result.unsupervised.predicted_cluster}
                  </span>
                  <span className="text-xs text-slate-400">
                    Cluster Population: {result.unsupervised.cluster_size.toLocaleString()} patches
                  </span>
                </div>

                {/* Primary Semantic Prediction */}
                <div className="p-4 rounded-xl bg-slate-900/60 border border-slate-800 mb-4">
                  <div className="flex items-center justify-between">
                    <div>
                      <p className="text-[11px] font-semibold text-slate-400 uppercase tracking-wider">Semantic Classification</p>
                      <h4 className="text-xl font-bold text-white mt-0.5">
                        {result.supervised_semantic.predicted_label}
                      </h4>
                    </div>
                    <span className="text-xs font-bold px-3 py-1.5 rounded-lg bg-emerald-500/10 border border-emerald-500/20 text-emerald-400">
                      High Confidence
                    </span>
                  </div>
                </div>

                {/* Unsupervised Archetype Notice */}
                <div className="p-4 rounded-xl bg-slate-900/40 border border-slate-800/60 mb-4 space-y-2 text-xs">
                  <div className="flex justify-between items-center text-slate-300">
                    <span className="text-slate-400">Unsupervised Cluster Archetype:</span>
                    <span className="text-white font-semibold">{result.unsupervised.mapped_label}</span>
                  </div>
                  <div className="flex justify-between items-center text-slate-300">
                    <span className="text-slate-400">Cluster Archetype Purity:</span>
                    <span className="text-cyan-400 font-semibold">{result.unsupervised.purity_confidence}%</span>
                  </div>
                  <div className="w-full bg-slate-800 rounded-full h-2 overflow-hidden mt-1">
                    <div
                      className="bg-cyan-500 h-full rounded-full transition-all duration-700"
                      style={{ width: `${result.unsupervised.purity_confidence}%` }}
                    ></div>
                  </div>
                </div>
              </div>

              {/* Centroid Distances Chart */}
              <div>
                <p className="text-[11px] font-semibold text-slate-400 uppercase tracking-wider mb-2">
                  Euclidean Distances to All 10 Cluster Centroids (220D Subspace)
                </p>
                <div className="h-28">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={result.unsupervised.centroid_distances.slice(0, 8)} margin={{ top: 5, right: 10, left: -25, bottom: 5 }}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#1E293B" />
                      <XAxis dataKey="cluster" stroke="#64748B" fontSize={10} />
                      <YAxis stroke="#64748B" fontSize={10} />
                      <Tooltip
                        contentStyle={{ backgroundColor: '#0F172A', borderColor: '#334155', borderRadius: '8px', fontSize: '11px' }}
                        formatter={(val) => [`Distance: ${val}`, '']}
                      />
                      <Bar dataKey="distance" fill="#3B82F6" radius={[4, 4, 0, 0]} />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </div>
            </div>
          </div>

          {/* Top-5 Nearest Neighbors Panel */}
          <div className="p-6 rounded-2xl bg-[#0F172A]/70 backdrop-blur-xl border border-slate-800/80 shadow-xl">
            <h4 className="text-sm font-semibold text-white tracking-tight mb-1">
              Top-5 Nearest Neighbors in EuroSAT Dataset
            </h4>
            <p className="text-xs text-slate-400 mb-4">
              Historical patches with closest Euclidean distance in the 90% retained-variance PCA space
            </p>

            <div className="grid grid-cols-2 sm:grid-cols-5 gap-4">
              {result.nearest_neighbors.map((nn, idx) => (
                <div key={idx} className="p-3 rounded-xl bg-slate-900/60 border border-slate-800/80 text-center space-y-2">
                  <div className="w-full h-24 rounded-lg overflow-hidden bg-slate-950 flex items-center justify-center">
                    <img
                      src={nn.url}
                      alt={nn.filename}
                      className="w-full h-full object-cover"
                      onError={(e) => { e.target.style.display = 'none'; }}
                    />
                  </div>
                  <div>
                    <span className="text-[10px] font-bold px-2 py-0.5 rounded bg-blue-500/10 text-blue-400 border border-blue-500/20">
                      #{idx + 1}: {nn.true_label}
                    </span>
                    <p className="text-[10px] text-slate-400 truncate mt-1">{nn.filename}</p>
                    <p className="text-[10px] text-slate-500">Dist: {nn.distance}</p>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
