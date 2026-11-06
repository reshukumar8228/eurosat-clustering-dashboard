import React from 'react';

export default function RadialGauge({ title, subtitle, value, max = 100, unit = '%', footerText }) {
  const percentage = Math.min(100, Math.max(0, (value / max) * 100));
  const strokeDashoffset = 251.2 - (251.2 * percentage) / 100;

  return (
    <div className="p-6 rounded-2xl bg-[#0F172A]/70 backdrop-blur-xl border border-slate-800/80 shadow-xl flex flex-col justify-between">
      <div>
        <h4 className="text-sm font-semibold text-white tracking-tight">{title}</h4>
        {subtitle && <p className="text-xs text-slate-400 mt-0.5">{subtitle}</p>}
      </div>

      <div className="flex flex-col items-center justify-center my-4 relative">
        <svg className="w-32 h-32 transform -rotate-90">
          <circle
            cx="64"
            cy="64"
            r="40"
            className="stroke-slate-800"
            strokeWidth="9"
            fill="transparent"
          />
          <circle
            cx="64"
            cy="64"
            r="40"
            className="stroke-blue-500 transition-all duration-1000 ease-out"
            strokeWidth="9"
            strokeDasharray="251.2"
            strokeDashoffset={strokeDashoffset}
            strokeLinecap="round"
            fill="transparent"
          />
        </svg>

        <div className="absolute flex flex-col items-center justify-center text-center">
          <span className="text-2xl font-extrabold text-white tracking-tight">
            {value}{unit}
          </span>
          <span className="text-[10px] text-slate-400 uppercase font-medium tracking-wider">Score</span>
        </div>
      </div>

      {footerText && (
        <div className="text-center pt-2 border-t border-slate-800/60 text-xs text-slate-400 font-medium">
          {footerText}
        </div>
      )}
    </div>
  );
}
