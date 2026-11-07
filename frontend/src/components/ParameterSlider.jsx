export default function ParameterSlider({
  label,
  value,
  min,
  max,
  step = 1,
  onChange,
  unit = '',
  description = '',
  highlights = []
}) {
  return (
    <div className="space-y-1.5">
      <div className="flex items-center justify-between">
        <label className="text-xs font-semibold text-slate-300 uppercase tracking-wider">
          {label}
        </label>
        <span className="text-xs font-bold text-blue-400 bg-blue-500/10 px-2 py-0.5 rounded border border-blue-500/20 tabular-nums">
          {value}{unit}
        </span>
      </div>
      {description && (
        <p className="text-[11px] text-slate-400 leading-snug">{description}</p>
      )}
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
        className="w-full h-1.5 bg-slate-800 rounded-full appearance-none cursor-pointer accent-blue-500
                   focus:outline-none transition-all"
      />
      {highlights && highlights.length > 0 && (
        <div className="flex justify-between items-center px-0.5 pt-0.5">
          {highlights.map((h) => (
            <button
              key={h.value}
              type="button"
              onClick={() => onChange(h.value)}
              className={`text-[10px] font-medium transition-colors ${
                value === h.value
                  ? 'text-blue-400 font-bold'
                  : 'text-slate-500 hover:text-slate-300'
              }`}
            >
              {h.label}
            </button>
          ))}
        </div>
      )}
    </div>
  );
}
