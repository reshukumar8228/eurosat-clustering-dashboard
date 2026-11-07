import { useState, useEffect } from 'react';
import { 
  LineChart, 
  Line, 
  XAxis, 
  YAxis, 
  Tooltip, 
  ResponsiveContainer, 
  CartesianGrid
} from 'recharts';

export default function ClusteringView() {
  const [data, setData] = useState(null);
  const [selectedCluster, setSelectedCluster] = useState(0);

  useEffect(() => {
    fetch('/api/clustering')
      .then(res => res.json())
      .then(d => setData(d))
      .catch(err => console.error(err));
  }, []);

  if (!data) {
    return (
      <div className="flex items-center justify-center h-96">
        <div className="w-8 h-8 border-2 border-blue-500 border-t-transparent rounded-full animate-spin"></div>
      </div>
    );
  }

  const activeClusterInfo = data.clusters.find(c => c.cluster_id === selectedCluster) || data.clusters[0];

  return (
    <div className="space-y-6">
      {/* k=10 Evaluation Metric Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
        <div className="p-5 rounded-2xl bg-[#0F172A]/70 backdrop-blur-xl border border-slate-800/80 shadow-xl">
          <p className="text-xs font-semibold text-slate-400 mb-1 uppercase tracking-wider">Adjusted Rand Index (ARI)</p>
          <div className="flex items-baseline gap-2">
            <h3 className="text-2xl font-bold text-white tracking-tight">{data.ari}</h3>
            <span className="text-xs text-blue-400">vs 10 True Classes</span>
          </div>
          <p className="text-[11px] text-slate-400 mt-1">Measures cluster partition agreement</p>
        </div>

        <div className="p-5 rounded-2xl bg-[#0F172A]/70 backdrop-blur-xl border border-slate-800/80 shadow-xl">
          <p className="text-xs font-semibold text-slate-400 mb-1 uppercase tracking-wider">Normalized Mutual Info (NMI)</p>
          <div className="flex items-baseline gap-2">
            <h3 className="text-2xl font-bold text-white tracking-tight">{data.nmi}</h3>
            <span className="text-xs text-emerald-400">Entropy Shared</span>
          </div>
          <p className="text-[11px] text-slate-400 mt-1">Information-theoretic correspondence</p>
        </div>

        <div className="p-5 rounded-2xl bg-[#0F172A]/70 backdrop-blur-xl border border-slate-800/80 shadow-xl">
          <p className="text-xs font-semibold text-slate-400 mb-1 uppercase tracking-wider">Overall Cluster Purity</p>
          <div className="flex items-baseline gap-2">
            <h3 className="text-2xl font-bold text-white tracking-tight">{data.overall_purity}%</h3>
            <span className="text-xs text-cyan-400">Majority Match</span>
          </div>
          <p className="text-[11px] text-slate-400 mt-1">Mean percentage of dominant labels</p>
        </div>
      </div>

      {/* Metric Curves Section: Elbow & Silhouette */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Elbow Curve */}
        <div className="p-6 rounded-2xl bg-[#0F172A]/70 backdrop-blur-xl border border-slate-800/80 shadow-xl">
          <div className="flex items-center justify-between mb-4">
            <div>
              <h4 className="text-sm font-semibold text-white tracking-tight">K-Means Inertia (Elbow Method)</h4>
              <p className="text-xs text-slate-400">Within-cluster sum of squares across k ∈ [2, 20]</p>
            </div>
          </div>
          <div className="h-60">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={data.k_curves}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1E293B" />
                <XAxis dataKey="k" stroke="#64748B" fontSize={11} />
                <YAxis stroke="#64748B" fontSize={11} />
                <Tooltip
                  contentStyle={{ backgroundColor: '#0F172A', borderColor: '#334155', borderRadius: '8px', fontSize: '12px' }}
                  formatter={(val) => [val.toLocaleString(), 'Inertia']}
                />
                <Line type="monotone" dataKey="inertia" stroke="#3B82F6" strokeWidth={2.5} dot={{ r: 3, fill: '#3B82F6' }} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Silhouette Score */}
        <div className="p-6 rounded-2xl bg-[#0F172A]/70 backdrop-blur-xl border border-slate-800/80 shadow-xl">
          <div className="flex items-center justify-between mb-4">
            <div>
              <h4 className="text-sm font-semibold text-white tracking-tight">Silhouette Coefficient Spectrum</h4>
              <p className="text-xs text-slate-400">Cluster separation quality (higher is superior)</p>
            </div>
          </div>
          <div className="h-60">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={data.k_curves}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1E293B" />
                <XAxis dataKey="k" stroke="#64748B" fontSize={11} />
                <YAxis stroke="#64748B" fontSize={11} domain={[0, 'auto']} />
                <Tooltip
                  contentStyle={{ backgroundColor: '#0F172A', borderColor: '#334155', borderRadius: '8px', fontSize: '12px' }}
                  formatter={(val) => [val, 'Silhouette Score']}
                />
                <Line type="monotone" dataKey="silhouette" stroke="#10B981" strokeWidth={2.5} dot={{ r: 3, fill: '#10B981' }} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>

      {/* Cluster Archetypes & Exemplars */}
      <div className="p-6 rounded-2xl bg-[#0F172A]/70 backdrop-blur-xl border border-slate-800/80 shadow-xl">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 mb-5">
          <div>
            <h4 className="text-sm font-semibold text-white tracking-tight">Cluster Archetypes & Exemplar Inspection</h4>
            <p className="text-xs text-slate-400">Centroid reconstructions and 4×4 representative images closest to geometric cluster centers</p>
          </div>

          {/* Cluster Selector Tabs */}
          <div className="flex flex-wrap gap-1.5 bg-slate-900/80 p-1 rounded-xl border border-slate-800">
            {data.clusters.map((c) => (
              <button
                key={c.cluster_id}
                onClick={() => setSelectedCluster(c.cluster_id)}
                className={`px-2.5 py-1 text-xs font-semibold rounded-lg transition-all ${
                  selectedCluster === c.cluster_id
                    ? 'bg-blue-600 text-white shadow-md shadow-blue-500/30'
                    : 'text-slate-400 hover:text-white'
                }`}
              >
                C-{c.cluster_id}
              </button>
            ))}
          </div>
        </div>

        {/* Selected Cluster Details & Images */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6 items-start">
          <div className="p-5 rounded-xl bg-slate-900/60 border border-slate-800/80 space-y-4">
            <div>
              <span className="text-[11px] font-semibold text-blue-400 uppercase tracking-wider">Cluster Archetype</span>
              <h5 className="text-lg font-bold text-white mt-0.5">Cluster {activeClusterInfo.cluster_id}</h5>
            </div>

            <div className="space-y-2 text-xs">
              <div className="flex justify-between py-1 border-b border-slate-800">
                <span className="text-slate-400">Dominant Class:</span>
                <span className="text-white font-semibold">{activeClusterInfo.majority_label}</span>
              </div>
              <div className="flex justify-between py-1 border-b border-slate-800">
                <span className="text-slate-400">Archetype Purity:</span>
                <span className="text-emerald-400 font-semibold">{activeClusterInfo.purity}%</span>
              </div>
              <div className="flex justify-between py-1 border-b border-slate-800">
                <span className="text-slate-400">Cluster Population:</span>
                <span className="text-white font-semibold">{activeClusterInfo.count.toLocaleString()} patches</span>
              </div>
            </div>

            <div>
              <p className="text-[11px] text-slate-400 uppercase font-semibold mb-2">Centroid Reconstructed Patch (64×64)</p>
              <img
                src={activeClusterInfo.centroid_url}
                alt={`Centroid ${activeClusterInfo.cluster_id}`}
                className="w-32 h-32 rounded-xl object-cover border border-slate-700/80 shadow-lg"
              />
            </div>
          </div>

          <div className="md:col-span-2 p-5 rounded-xl bg-slate-900/60 border border-slate-800/80">
            <p className="text-[11px] text-slate-400 uppercase font-semibold mb-3">4×4 Exemplar Patches Closest to Centroid</p>
            <img
              src={activeClusterInfo.grid_url}
              alt={`Exemplars ${activeClusterInfo.cluster_id}`}
              className="w-full rounded-xl object-cover border border-slate-700/80 shadow-lg"
            />
          </div>
        </div>
      </div>

      {/* Contingency Matrix Table */}
      <div className="p-6 rounded-2xl bg-[#0F172A]/70 backdrop-blur-xl border border-slate-800/80 shadow-xl overflow-x-auto">
        <h4 className="text-sm font-semibold text-white tracking-tight mb-1">K=10 Contingency Matrix (Cluster × True Class)</h4>
        <p className="text-xs text-slate-400 mb-4">Post-hoc evaluation count distribution</p>

        <table className="w-full text-xs text-left text-slate-300">
          <thead className="text-[11px] text-slate-400 uppercase bg-slate-900/80 border-b border-slate-800">
            <tr>
              <th className="py-2.5 px-3 font-semibold text-white">Cluster</th>
              {data.classes.map(c => (
                <th key={c} className="py-2.5 px-3 font-semibold">{c}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {data.contingency.map((row, idx) => (
              <tr key={idx} className="border-b border-slate-800/60 hover:bg-slate-800/30 transition-all">
                <td className="py-2 px-3 font-bold text-white">{row.cluster}</td>
                {data.classes.map(col => {
                  const val = row.values[col] || 0;
                  return (
                    <td key={col} className={`py-2 px-3 ${val > 400 ? 'text-blue-400 font-bold bg-blue-500/10' : ''}`}>
                      {val.toLocaleString()}
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
