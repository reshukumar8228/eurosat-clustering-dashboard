import { useState } from 'react';
import Sidebar from './components/Sidebar';
import Header from './components/Header';
import OverviewView from './views/OverviewView';
import PcaView from './views/PcaView';
import TsneView from './views/TsneView';
import ClusteringView from './views/ClusteringView';
import InferenceView from './views/InferenceView';
import SpectralView from './views/SpectralView';
import ReportView from './views/ReportView';

export default function App() {
  const [activeTab, setActiveTab] = useState('overview');

  const renderView = () => {
    switch (activeTab) {
      case 'overview':
        return <OverviewView setActiveTab={setActiveTab} />;
      case 'pca':
        return <PcaView />;
      case 'tsne':
        return <TsneView />;
      case 'clustering':
        return <ClusteringView />;
      case 'inference':
        return <InferenceView />;
      case 'spectral':
        return <SpectralView />;
      case 'report':
        return <ReportView />;
      default:
        return <OverviewView setActiveTab={setActiveTab} />;
    }
  };

  return (
    <div className="flex min-h-screen bg-[#070B14] text-slate-100 font-sans antialiased">
      {/* Vision UI Left Sidebar */}
      <Sidebar activeTab={activeTab} setActiveTab={setActiveTab} />

      {/* Main Content Area */}
      <main className="flex-1 p-6 lg:p-8 max-w-7xl mx-auto overflow-y-auto">
        <Header activeTab={activeTab} />
        {renderView()}
      </main>
    </div>
  );
}
