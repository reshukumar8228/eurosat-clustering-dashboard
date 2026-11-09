import { useState, useRef } from 'react';
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
  const [error, setError] = useState(null);
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
    setError(null);

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

      if (!res.ok) {
        const errData = await res.json().catch(() => ({}));
        setError(errData.error || `Server returned error status (${res.status}). Please try another valid image.`);
        setResult(null);
        return;
      }

      const data = await res.json();
      if (data.error) {
        setError(data.error);
        setResult(null);
      } else {
        setResult(data);
        setError(null);
      }
    } catch (err) {
      console.error("Prediction error:", err);
      setError('Unable to reach prediction server. Please verify backend is running on port 8000.');
      setResult(null);
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
            accept=".jpg,.jpeg,.png,.tif,.tiff,.webp,.bmp,.jfif,.avif"
            className="hidden"
          />

          <h4 className="text-sm font-semibold text-white mb-1">
            Drag & drop satellite patch, or <span className="text-blue-400">browse</span>
          </h4>
          <p className="text-xs text-slate-400">
            Supports RGB optical imagery (.jpg, .png, .webp, .bmp), Sentinel-2 multi-spectral (.tif), and aerial patches
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
                  <span className="text-slate-500 text-[10px]">Select</span>
                </button>
              ))}
            </div>
          </div>
        </div>
      </div>

      {/* Error Alert Display */}
      {error && (
        <div className="p-4 rounded-xl bg-rose-500/10 border border-rose-500/30 text-rose-300 flex items-start justify-between gap-3 text-xs animate-in fade-in">
          <div className="flex items-start gap-2.5">
            <span className="w-2 h-2 rounded-full bg-rose-400 shrink-0 mt-1.5" />
            <div>
              <p className="font-semibold text-white">Inference Message</p>
              <p className="mt-0.5 text-rose-300/90 leading-relaxed">{error}</p>
            </div>
          </div>
          <button
            onClick={() => setError(null)}
            className="text-rose-400 hover:text-white px-2 py-1 rounded bg-rose-500/10 hover:bg-rose-500/20 text-xs font-semibold"
          >
            Dismiss
          </button>
        </div>
      )}

      {/* Loading Spinner */}
      {loading && (
        <div className="p-12 text-center rounded-2xl bg-[#0F172A]/70 border border-slate-800/80">
          <div className="w-8 h-8 border-2 border-blue-500 border-t-transparent rounded-full animate-spin mx-auto mb-3"></div>
          <p className="text-xs text-slate-400">Projecting patch into 220D PCA space and evaluating cluster assignments...</p>
        </div>
      )}

      {/* Prediction Results Display */}
      {result && !loading && !error && result.physical_stats && (
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
              <h5 className="text-sm font-bold text-white truncate max-w-full px-2">{result.source_name || "Custom Patch"}</h5>
              <p className="text-[11px] text-slate-400 mt-0.5">Input Image Patch</p>

              <div className="w-full pt-4 mt-4 border-t border-slate-800/80 space-y-2 text-left text-xs">
                <div className="flex justify-between text-slate-300">
                  <span className="text-slate-400">Mean Luminance:</span>
                  <span className="font-semibold text-white">{result.physical_stats?.brightness ?? '—'}</span>
                </div>
                <div className="flex justify-between text-slate-300">
                  <span className="text-slate-400">Texture Roughness (σ):</span>
                  <span className="font-semibold text-white">{result.physical_stats?.roughness ?? '—'}</span>
                </div>
                <div className="flex justify-between text-slate-300">
                  <span className="text-slate-400">RGB Balance:</span>
                  <span className="font-mono text-slate-200">
                    R:{result.physical_stats?.red ?? '—'} G:{result.physical_stats?.green ?? '—'} B:{result.physical_stats?.blue ?? '—'}
                  </span>
                </div>
              </div>
            </div>

            {/* Classification & Cluster Detection */}
            <div className="lg:col-span-2 p-6 rounded-2xl bg-[#0F172A]/70 backdrop-blur-xl border border-slate-800/80 shadow-xl flex flex-col justify-between">
              <div>
                <div className="flex flex-wrap items-center justify-between gap-2 mb-4">
                  <span className="px-3 py-1 rounded-xl text-xs font-semibold bg-blue-500/10 border border-blue-500/30 text-blue-400">
                    Cluster {result.unsupervised?.predicted_cluster ?? '—'}
                  </span>
                  <span className="text-xs text-slate-400">
                    Cluster Population: {(result.unsupervised?.cluster_size ?? 0).toLocaleString()} patches
                  </span>
                </div>

                {/* Primary Semantic Prediction */}
                <div className="p-4 rounded-xl bg-slate-900/60 border border-slate-800 mb-4">
                  <div className="flex items-center justify-between">
                    <div>
                      <p className="text-[11px] font-semibold text-slate-400 uppercase tracking-wider">Semantic Classification</p>
                      <h4 className="text-xl font-bold text-white mt-0.5">
                        {result.supervised_semantic?.predicted_label ?? result.unsupervised?.mapped_label ?? 'Unknown'}
                      </h4>
                    </div>
                    {(() => {
                      if (result.is_out_of_distribution || result.supervised_semantic?.is_out_of_distribution) {
                        return (
                          <span className="text-xs font-bold px-3 py-1.5 rounded-lg border bg-rose-500/10 border-rose-500/30 text-rose-400">
                            Non-Satellite / Out of Distribution
                          </span>
                        );
                      }
                      const conf = result.supervised_semantic?.confidence ?? 0;
                      let badgeStyle = "bg-emerald-500/10 border-emerald-500/20 text-emerald-400";
                      let badgeText = `High Confidence (${conf}%)`;
                      if (conf < 50) {
                        badgeStyle = "bg-amber-500/10 border-amber-500/20 text-amber-400";
                        badgeText = `Low Confidence (${conf}%)`;
                      } else if (conf < 75) {
                        badgeStyle = "bg-blue-500/10 border-blue-500/20 text-blue-400";
                        badgeText = `Moderate Confidence (${conf}%)`;
                      }
                      return (
                        <span className={`text-xs font-bold px-3 py-1.5 rounded-lg border ${badgeStyle}`}>
                          {badgeText}
                        </span>
                      );
                    })()}
                  </div>

                  {/* Top Probabilities Breakdown */}
                  {result.supervised_semantic?.probabilities && (
                    <div className="mt-3 pt-3 border-t border-slate-800/60">
                      <p className="text-[10px] font-semibold text-slate-400 uppercase tracking-wider mb-1.5">
                        {result.is_out_of_distribution ? "Distribution Analysis:" : "Top Class Probabilities:"}
                      </p>
                      <div className="flex flex-wrap gap-2 text-xs">
                        {result.is_out_of_distribution ? (
                          <div className="flex items-center gap-1.5 px-2.5 py-1 rounded-lg bg-rose-500/10 border border-rose-500/30">
                            <span className="text-rose-300 font-medium">Out-of-Distribution Input:</span>
                            <span className="font-bold text-rose-400">100% (Not a Satellite Patch)</span>
                          </div>
                        ) : (
                          Object.entries(result.supervised_semantic.probabilities)
                            .sort((a, b) => b[1] - a[1])
                            .slice(0, 4)
                            .map(([cls, prob]) => (
                              <div key={cls} className="flex items-center gap-1.5 px-2.5 py-1 rounded-lg bg-slate-800/60 border border-slate-700/60">
                                <span className="text-slate-300 font-medium">{cls}:</span>
                                <span className="font-bold text-cyan-400">{prob}%</span>
                              </div>
                            ))
                        )}
                      </div>
                    </div>
                  )}
                </div>

                {/* Domain Shift / Outlier Notice if applicable */}
                {result.supervised_semantic?.domain_note && (
                  <div className={`p-3.5 rounded-xl border text-xs mb-4 flex items-start gap-2.5 ${
                    result.is_out_of_distribution
                      ? "bg-rose-500/10 border-rose-500/30 text-rose-300"
                      : "bg-cyan-500/10 border-cyan-500/30 text-cyan-300"
                  }`}>
                    <span className={`w-2 h-2 rounded-full shrink-0 mt-1.5 ${
                      result.is_out_of_distribution ? "bg-rose-400" : "bg-cyan-400"
                    }`} />
                    <span className="leading-relaxed">{result.supervised_semantic.domain_note}</span>
                  </div>
                )}

                {/* Unsupervised Archetype Notice */}
                <div className="p-4 rounded-xl bg-slate-900/40 border border-slate-800/60 mb-4 space-y-2 text-xs">
                  <div className="flex justify-between items-center text-slate-300">
                    <span className="text-slate-400">Unsupervised Cluster Archetype:</span>
                    <span className="text-white font-semibold">{result.unsupervised?.mapped_label ?? '—'}</span>
                  </div>
                  <div className="flex justify-between items-center text-slate-300">
                    <span className="text-slate-400">Cluster Archetype Purity:</span>
                    <span className="text-cyan-400 font-semibold">{result.unsupervised?.purity_confidence ?? 0}%</span>
                  </div>
                  <div className="w-full bg-slate-800 rounded-full h-2 overflow-hidden mt-1">
                    <div
                      className="bg-cyan-500 h-full rounded-full transition-all duration-700"
                      style={{ width: `${result.unsupervised?.purity_confidence ?? 0}%` }}
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
                    <BarChart
                      data={(result.unsupervised?.centroid_distances || []).slice(0, 8)}
                      margin={{ top: 5, right: 10, left: -25, bottom: 5 }}
                    >
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
              {(result.nearest_neighbors || []).map((nn, idx) => (
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
