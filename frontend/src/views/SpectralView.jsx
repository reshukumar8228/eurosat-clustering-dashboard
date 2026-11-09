import { useState, useEffect } from 'react';
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

const DEFAULT_SIGNATURES = {
  AnnualCrop: [1383.4, 1227.0, 1255.0, 1394.4, 1679.9, 2660.5, 3220.1, 3120.8, 836.7, 14.0, 2598.5, 1586.8, 3592.9],
  Forest: [1130.5, 824.5, 690.6, 420.9, 752.2, 2083.6, 2658.2, 2608.0, 840.7, 10.0, 1320.7, 541.1, 2975.3],
  HerbaceousVegetation: [1324.0, 1113.5, 1064.5, 1064.8, 1399.0, 2132.4, 2451.2, 2462.4, 1078.3, 15.9, 2411.4, 1475.5, 2763.2],
  Highway: [1342.7, 1096.2, 1017.4, 910.3, 1184.7, 2078.3, 2463.7, 2388.4, 768.5, 12.4, 1824.2, 1138.8, 2693.3],
  Industrial: [1626.7, 1437.2, 1359.6, 1368.4, 1516.5, 1997.9, 2232.4, 2119.3, 642.2, 12.5, 2139.4, 1581.6, 2383.1],
  Pasture: [1305.2, 1012.7, 951.6, 665.4, 1179.1, 2813.0, 3482.1, 3421.4, 1080.7, 12.4, 2040.3, 977.4, 3895.1],
  PermanentCrop: [1371.3, 1166.4, 1165.9, 1195.1, 1489.3, 2386.3, 2808.2, 2716.7, 749.1, 12.8, 2561.0, 1618.3, 3096.1],
  Residential: [1453.0, 1225.0, 1145.6, 1128.4, 1352.5, 2055.0, 2363.2, 2266.3, 711.7, 12.7, 2070.4, 1457.3, 2546.5],
  River: [1310.3, 1048.3, 939.9, 720.2, 901.3, 1618.1, 1940.9, 1852.7, 529.9, 9.8, 1199.4, 664.5, 2085.5],
  SeaLake: [1240.5, 951.5, 732.7, 498.3, 485.0, 390.9, 391.0, 326.8, 133.0, 16.9, 75.9, 80.6, 283.4]
};

const BAND_NAMES = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9', 'B10', 'B11', 'B12'];

export default function SpectralView() {
  const [selectedClass, setSelectedClass] = useState('Forest');
  const [signatures, setSignatures] = useState(DEFAULT_SIGNATURES);
  const [availableClasses, setAvailableClasses] = useState(Object.keys(DEFAULT_SIGNATURES));

  useEffect(() => {
    fetch('/api/spectral')
      .then(res => res.ok ? res.json() : null)
      .then(data => {
        if (data?.profiles && Object.keys(data.profiles).length > 0) {
          setSignatures(data.profiles);
          setAvailableClasses(data.classes || Object.keys(data.profiles));
        }
      })
      .catch(() => {});
  }, []);

  const profile = signatures[selectedClass] || DEFAULT_SIGNATURES[selectedClass] || [];
  const chartData = BAND_NAMES.map((b, i) => ({
    band: b,
    reflectance: profile[i] !== undefined ? profile[i] : 0
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
            {availableClasses.map((cls) => (
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
