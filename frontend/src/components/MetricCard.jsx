export default function MetricCard({ title, value, change, icon: Icon, pillColor = 'blue' }) {
  const colorMap = {
    blue: 'bg-blue-600 text-white shadow-blue-500/30',
    cyan: 'bg-cyan-600 text-white shadow-cyan-500/30',
    emerald: 'bg-emerald-600 text-white shadow-emerald-500/30',
    purple: 'bg-purple-600 text-white shadow-purple-500/30',
  };

  return (
    <div className="p-5 rounded-2xl bg-[#0F172A]/70 backdrop-blur-xl border border-slate-800/80 shadow-xl flex items-center justify-between">
      <div>
        <p className="text-xs font-semibold text-slate-400 mb-1 tracking-wider uppercase">{title}</p>
        <div className="flex items-baseline gap-2">
          <h3 className="text-2xl font-bold text-white tracking-tight">{value}</h3>
          {change && (
            <span className="text-xs font-semibold text-emerald-400">
              {change}
            </span>
          )}
        </div>
      </div>
      <div className={`w-11 h-11 rounded-xl flex items-center justify-center shadow-lg ${colorMap[pillColor] || colorMap.blue}`}>
        <Icon className="w-5 h-5" />
      </div>
    </div>
  );
}
