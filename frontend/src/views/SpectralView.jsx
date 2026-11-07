import React, { useState } from 'react';
import { Info } from 'lucide-react';
import { 
  LineChart, 
  Line, 
  XAxis, 
  YAxis, 
  Tooltip, 
  ResponsiveContainer, 
  CartesianGrid 
} from 'recharts';

export default function SpectralView() {
  const [selectedClass, setSelectedClass] = useState('Forest');

  const spectralSignatures = {
    Forest: [1200, 1150, 1300, 1100, 2400, 3800, 4200, 4400, 4500, 1100, 500, 2100, 1200],
    River: [1600, 1800, 1600, 1200, 1000, 800, 700, 600, 550, 400, 200, 400, 300],
    Highway: [1500, 1600, 1700, 1800, 2000, 2200, 2300, 2400, 2450, 800, 300, 2500, 2100],
    Residential: [1600, 1750, 1900, 2100, 2400, 2700, 2900, 3000, 3100, 950, 400, 2900, 2400],
    AnnualCrop: [1300, 1400, 1650, 1500, 2600, 3900, 4300, 4500, 4600, 1050, 450, 2800, 1800],
  };

  const bandNames = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9', 'B10', 'B11', 'B12'];

  const chartData = bandNames.map((b, i) => ({
    band: b,
    reflectance: spectralSignatures[selectedClass][i]
  }));

  return (
    <div className="space-y-6">
      {/* Overview Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
        <div className="p-5 rounded-2xl bg-[#0F172A]/70 backdrop-blur-xl border border-slate-800/80 shadow-xl">
          <p className="text-xs font-semibold text-slate-400 mb-1 uppercase tracking-wider">Spectral Channels</p>
          <div className="flex items-baseline gap-2">
            <h3 className="text-2xl font-bold text-white tracking-tight">13 Bands</h3>
            <span className="text-xs text-blue-400">Sentinel-2 MSI</span>
          </div>
        </div>

        <div className="p-5 rounded-2xl bg-[#0F172A]/70 backdrop-blur-xl border border-slate-800/80 shadow-xl">
          <p className="text-xs font-semibold text-slate-400 mb-1 uppercase tracking-wider">Spatial Resolution</p>
          <div className="flex items-baseline gap-2">
            <h3 className="text-2xl font-bold text-white tracking-tight">10m - 60m</h3>
            <span className="text-xs text-emerald-400">GSD</span>
          </div>
        </div>

        <div className="p-5 rounded-2xl bg-[#0F172A]/70 backdrop-blur-xl border border-slate-800/80 shadow-xl">
          <p className="text-xs font-semibold text-slate-400 mb-1 uppercase tracking-wider">Multi-Spectral Advantage</p>
          <div className="flex items-baseline gap-2">
            <h3 className="text-2xl font-bold text-white tracking-tight">Red Edge + SWIR</h3>
            <span className="text-xs text-cyan-400">Chlorophyll sensitive</span>
          </div>
        </div>
      </div>

      {/* Spectral Signature Chart */}
      <div className="p-6 rounded-2xl bg-[#0F172A]/70 backdrop-blur-xl border border-slate-800/80 shadow-xl">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 mb-5">
          <div>
            <h4 className="text-sm font-semibold text-white tracking-tight">13-Band Spectral Reflectance Profile</h4>
            <p className="text-xs text-slate-400">Mean digital numbers across European Sentinel-2 observations</p>
          </div>

          <div className="flex flex-wrap gap-1.5 bg-slate-900/80 p-1 rounded-xl border border-slate-800">
            {Object.keys(spectralSignatures).map((cls) => (
              <button
                key={cls}
                onClick={() => setSelectedClass(cls)}
                className={`px-3 py-1 rounded-lg text-xs font-semibold transition-all ${
                  selectedClass === cls
                    ? 'bg-blue-600 text-white shadow-md shadow-blue-500/30'
                    : 'text-slate-400 hover:text-white'
                }`}
              >
                {cls}
              </button>
            ))}
          </div>
        </div>

        <div className="h-72">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={chartData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1E293B" />
              <XAxis dataKey="band" stroke="#64748B" fontSize={11} />
              <YAxis stroke="#64748B" fontSize={11} />
              <Tooltip
                contentStyle={{ backgroundColor: '#0F172A', borderColor: '#334155', borderRadius: '8px', fontSize: '12px' }}
                formatter={(val) => [val.toLocaleString(), 'Mean Reflectance']}
              />
              <Line type="monotone" dataKey="reflectance" stroke="#3B82F6" strokeWidth={2.5} dot={{ r: 4, fill: '#3B82F6' }} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Remote Sensing Scientific Note */}
      <div className="p-5 rounded-2xl bg-slate-900/60 border border-slate-800 flex items-start gap-3.5 text-xs text-slate-300">
        <Info className="w-5 h-5 text-blue-400 shrink-0 mt-0.5" />
        <div>
          <span className="font-bold text-white">Why Spectral Bands Matter: </span>
          In pure RGB color space, dark water (`River`) and shaded roadways (`Highway`) overlap significantly. In the Near-Infrared (B8, B8A) and SWIR (B11, B12) channels, water exhibits near-total radiation absorption, creating distinct separation from asphalt and bare soils.
        </div>
      </div>
    </div>
  );
}
