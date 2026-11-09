import { 
  LayoutDashboard, 
  Layers, 
  GitGraph, 
  Hexagon, 
  FlaskConical,
  UploadCloud, 
  Radio, 
  FileText,
  Satellite
} from 'lucide-react';

export default function Sidebar({ activeTab, setActiveTab }) {
  const menuItems = [
    { id: 'overview', label: 'Dashboard', icon: LayoutDashboard },
    { id: 'pca', label: 'PCA Variance', icon: Layers },
    { id: 'tsne', label: 't-SNE Manifold', icon: GitGraph },
    { id: 'clustering', label: 'Clustering', icon: Hexagon },
    { id: 'experiment', label: 'Experiment Bench', icon: FlaskConical },
    { id: 'inference', label: 'Patch Inference', icon: UploadCloud },
    { id: 'spectral', label: 'Spectral Bands', icon: Radio },
    { id: 'report', label: 'Research Report', icon: FileText },
  ];

  return (
    <aside className="w-64 min-h-screen bg-[#070B14]/95 border-r border-slate-800/80 flex flex-col justify-between p-5 select-none shrink-0">
      <div>
        {/* Brand Header */}
        <div className="flex items-center gap-3 px-2 py-3 mb-6">
          <div className="w-9 h-9 rounded-xl bg-blue-600/20 border border-blue-500/30 flex items-center justify-center text-blue-400 shadow-md shadow-blue-500/10">
            <Satellite className="w-5 h-5" />
          </div>
          <div>
            <h1 className="text-sm font-bold tracking-wider text-white uppercase">EuroSAT Studio</h1>
            <p className="text-[11px] text-slate-400 font-medium">Unsupervised Clustering</p>
          </div>
        </div>

        {/* Navigation Menu */}
        <div className="text-[11px] font-semibold tracking-wider text-slate-500 uppercase px-3 mb-3">
          Core Workflows
        </div>

        <nav className="space-y-2">
          {menuItems.map((item) => {
            const Icon = item.icon;
            const isActive = activeTab === item.id;
            return (
              <button
                key={item.id}
                onClick={() => setActiveTab(item.id)}
                className={`w-full flex items-center gap-3.5 px-3 py-2 rounded-2xl text-sm font-medium transition-all duration-200 ${
                  isActive
                    ? 'bg-blue-600 text-white shadow-lg shadow-blue-600/30'
                    : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800/40'
                }`}
              >
                <div
                  className={`w-9 h-9 rounded-xl flex items-center justify-center shrink-0 transition-colors ${
                    isActive
                      ? 'bg-white/20 text-white'
                      : 'bg-slate-900/90 border border-slate-800/80 text-blue-400'
                  }`}
                >
                  <Icon className="w-4 h-4" />
                </div>
                <span className={isActive ? 'font-semibold text-white' : ''}>{item.label}</span>
              </button>
            );
          })}
        </nav>
      </div>

      {/* Dataset Specs Footer */}
      <div className="p-4 rounded-xl bg-slate-900/60 border border-slate-800/80">
        <div className="text-xs font-semibold text-white mb-1.5">
          Sentinel-2 Registry
        </div>
        <p className="text-[11px] text-slate-400 leading-relaxed mb-3">
          27,000 multi-spectral and RGB patches loaded with ground-truth labels isolated for post-hoc analysis.
        </p>
        <div className="flex items-center justify-between text-[11px] font-medium text-slate-300 bg-slate-800/70 px-2.5 py-1.5 rounded-lg border border-slate-700/60">
          <span>Engine Status</span>
          <span className="text-emerald-400 font-semibold">Online</span>
        </div>
      </div>
    </aside>
  );
}
