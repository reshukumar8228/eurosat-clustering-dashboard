export default function Header({ activeTab }) {
  const titles = {
    overview: 'Overview & Dataset',
    pca: 'Principal Component Analysis (PCA)',
    tsne: 't-SNE Manifold Learning',
    clustering: 'K-Means Clustering Analysis',
    experiment: 'Experiment Bench — Parameter Playground',
    inference: 'Satellite Patch Inference Engine',
    spectral: 'Multi-Spectral Band Analysis',
    report: 'Study Report & Ethical Governance'
  };

  return (
    <header className="flex flex-col md:flex-row md:items-center justify-between gap-4 pb-6 border-b border-slate-800/60 mb-6">
      <div>
        <div className="flex items-center gap-2 text-xs text-slate-400 mb-1">
          <span>Pages</span>
          <span>/</span>
          <span className="text-slate-200 capitalize">{activeTab}</span>
        </div>
        <h2 className="text-xl font-bold text-white tracking-tight">
          {titles[activeTab] || 'Dashboard'}
        </h2>
      </div>

      <div className="flex items-center gap-3">
        {/* Search Bar */}
        <div className="relative">
          <input
            type="text"
            placeholder="Search parameters, classes..."
            className="w-56 px-3.5 py-2 bg-slate-900/80 border border-slate-800/80 rounded-xl text-xs text-slate-200 placeholder:text-slate-500 focus:outline-none focus:border-blue-500/50 transition-all"
          />
        </div>

        {/* Verification Pill */}
        <div className="px-3 py-1.5 rounded-xl bg-emerald-500/10 border border-emerald-500/20 text-emerald-400 text-xs font-semibold">
          Labels Isolated
        </div>
      </div>
    </header>
  );
}
