export default function MetricCard({ title, value, change }) {
  return (
    <div className="p-5 rounded-2xl bg-[#0F172A]/70 backdrop-blur-xl border border-slate-800/80 shadow-xl flex flex-col justify-between">
      <p className="text-xs font-semibold text-slate-400 mb-2 tracking-wider uppercase">{title}</p>
      <div className="flex items-baseline justify-between gap-2">
        <h3 className="text-2xl font-bold text-white tracking-tight">{value}</h3>
        {change && (
          <span className="text-xs font-semibold text-emerald-400 bg-emerald-500/10 px-2.5 py-0.5 rounded-md border border-emerald-500/20">
            {change}
          </span>
        )}
      </div>
    </div>
  );
}
