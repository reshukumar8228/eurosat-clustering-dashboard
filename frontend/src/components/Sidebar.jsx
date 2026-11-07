import { 
  LayoutDashboard, 
  Layers, 
  GitGraph, 
  Hexagon, 
  UploadCloud, 
  Radio, 
  FileText,
  Database,
  Cpu
} from 'lucide-react';

export default function Sidebar({ activeTab, setActiveTab }) {
  const menuItems = [
    { id: 'overview', label: 'Dashboard', icon: LayoutDashboard },
    { id: 'pca', label: 'PCA Variance', icon: Layers },
    { id: 'tsne', label: 't-SNE Manifold', icon: GitGraph },
    { id: 'clustering', label: 'Clustering', icon: Hexagon },
    { id: 'inference', label: 'Patch Inference', icon: UploadCloud },
    { id: 'spectral', label: 'Spectral Bands', icon: Radio },
    { id: 'report', label: 'Research Report', icon: FileText },
  ];

  return (
    <aside className="w-64 min-h-screen bg-[#070B14]/95 border-r border-slate-800/80 flex flex-col justify-between p-5 select-none shrink-0">
      <div>
        {/* Brand Header - Clean typography without logo/emojis */}
        <div className="flex items-center gap-3 px-2 py-3 mb-6">
          <div className="w-9 h-9 rounded-xl bg-gradient-to-tr from-blue-600 to-indigo-500 flex items-center justify-center shadow-lg shadow-blue-500/30">
            <Cpu className="w-5 h-5 text-white" />
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

        <nav className="space-y-1.5">
          {menuItems.map((item) => {
            const Icon = item.icon;
            const isActive = activeTab === item.id;
            return (
              <button
                key={item.id}
                onClick={() => setActiveTab(item.id)}
                className={`w-full flex items-center gap-3.5 px-3.5 py-2.5 rounded-xl text-sm font-medium transition-all duration-200 ${
                  isActive
                    ? 'bg-gradient-to-r from-blue-600 to-blue-500 text-white shadow-lg shadow-blue-600/30'
                    : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800/50'
                }`}
              >
                <div className={`p-1.5 rounded-lg ${isActive ? 'bg-white/20' : 'bg-slate-800/80 text-blue-400'}`}>
                  <Icon className="w-4 h-4" />
                </div>
                <span>{item.label}</span>
              </button>
            );
          })}
        </nav>
      </div>

      {/* Dataset Specs Floating Card (Vision UI style) */}
      <div className="p-4 rounded-2xl bg-gradient-to-br from-blue-950/40 via-slate-900/60 to-indigo-950/40 border border-blue-500/20 shadow-xl backdrop-blur-md">
        <div className="flex items-center gap-2 mb-2">
          <Database className="w-4 h-4 text-blue-400" />
          <span className="text-xs font-semibold text-white">Sentinel-2 Registry</span>
        </div>
        <p className="text-[11px] text-slate-400 leading-relaxed mb-3">
          27,000 multi-spectral and RGB patches loaded with ground-truth labels isolated for post-hoc analysis.
        </p>
        <div className="flex items-center justify-between text-[11px] font-medium text-blue-400 bg-blue-500/10 px-2.5 py-1.5 rounded-lg border border-blue-500/20">
          <span>Engine Status</span>
          <span className="text-emerald-400">Online</span>
        </div>
      </div>
    </aside>
  );
}
