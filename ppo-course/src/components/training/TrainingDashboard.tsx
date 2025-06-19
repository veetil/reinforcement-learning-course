'use client';

import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  Play, Pause, RotateCcw, Download, Settings, 
  Activity, TrendingUp, Zap, AlertCircle 
} from 'lucide-react';
import { useTraining } from '@/hooks/useTraining';
import { TrainingConfig } from '@/lib/api';
import dynamic from 'next/dynamic';

const LineChart = dynamic(
  () => import('recharts').then(mod => mod.LineChart),
  { ssr: false }
);
const Line = dynamic(
  () => import('recharts').then(mod => mod.Line),
  { ssr: false }
);
const XAxis = dynamic(
  () => import('recharts').then(mod => mod.XAxis),
  { ssr: false }
);
const YAxis = dynamic(
  () => import('recharts').then(mod => mod.YAxis),
  { ssr: false }
);
const Tooltip = dynamic(
  () => import('recharts').then(mod => mod.Tooltip),
  { ssr: false }
);
const ResponsiveContainer = dynamic(
  () => import('recharts').then(mod => mod.ResponsiveContainer),
  { ssr: false }
);

interface TrainingDashboardProps {
  environment?: string;
  algorithm?: string;
}

export const TrainingDashboard: React.FC<TrainingDashboardProps> = ({
  environment = 'CartPole-v1',
  algorithm = 'PPO'
}) => {
  const { trainingId, status, isConnected, error, startTraining, stopTraining } = useTraining();
  const [showSettings, setShowSettings] = useState(false);
  const [config, setConfig] = useState<TrainingConfig>({
    environment,
    algorithm,
    hyperparameters: {
      learningRate: 0.0003,
      clipRange: 0.2,
      gamma: 0.99,
      gaeBalance: 0.95,
      nEpochs: 10,
      batchSize: 64,
      entropyCoef: 0.01,
      valueCoef: 0.5
    }
  });

  const [metricsHistory, setMetricsHistory] = useState<any[]>([]);

  // Update metrics history when status changes
  React.useEffect(() => {
    if (status?.metrics) {
      setMetricsHistory(prev => [...prev, {
        episode: status.currentEpisode,
        reward: status.metrics.meanReward,
        policyLoss: status.metrics.policyLoss,
        valueLoss: status.metrics.valueLoss,
        klDivergence: status.metrics.klDivergence
      }].slice(-100)); // Keep last 100 points
    }
  }, [status]);

  const handleStart = async () => {
    try {
      await startTraining(config);
      setMetricsHistory([]);
    } catch (err) {
      console.error('Failed to start training:', err);
    }
  };

  const handleUpdateConfig = (field: string, value: any) => {
    setConfig(prev => ({
      ...prev,
      hyperparameters: {
        ...prev.hyperparameters,
        [field]: value
      }
    }));
  };

  const isTraining = status?.status === 'running';

  return (
    <div className="w-full space-y-6">
      {/* Header Controls */}
      <div className="bg-white rounded-lg shadow p-4">
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center gap-4">
            <h2 className="text-2xl font-bold">Training Dashboard</h2>
            {isConnected && (
              <div className="flex items-center gap-2">
                <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse"></div>
                <span className="text-sm text-gray-600">Connected</span>
              </div>
            )}
          </div>
          
          <div className="flex items-center gap-2">
            <button
              onClick={() => setShowSettings(!showSettings)}
              className="p-2 text-gray-600 hover:bg-gray-100 rounded-lg"
            >
              <Settings className="w-5 h-5" />
            </button>
            
            {!isTraining ? (
              <button
                onClick={handleStart}
                className="px-4 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700 flex items-center gap-2"
              >
                <Play className="w-4 h-4" />
                Start Training
              </button>
            ) : (
              <button
                onClick={stopTraining}
                className="px-4 py-2 bg-red-600 text-white rounded-lg hover:bg-red-700 flex items-center gap-2"
              >
                <Pause className="w-4 h-4" />
                Stop Training
              </button>
            )}
          </div>
        </div>

        {/* Progress Bar */}
        {status && (
          <div className="space-y-2">
            <div className="flex justify-between text-sm">
              <span>Episode {status.currentEpisode} / {status.totalEpisodes}</span>
              <span>{Math.round(status.progress)}%</span>
            </div>
            <div className="w-full bg-gray-200 rounded-full h-2">
              <div 
                className="bg-blue-600 h-2 rounded-full transition-all duration-300"
                style={{ width: `${status.progress}%` }}
              />
            </div>
          </div>
        )}
      </div>

      {/* Settings Panel */}
      <AnimatePresence>
        {showSettings && (
          <motion.div
            initial={{ opacity: 0, y: -20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
            className="bg-white rounded-lg shadow p-6"
          >
            <h3 className="text-lg font-bold mb-4">Hyperparameters</h3>
            <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-4">
              <div>
                <label className="block text-sm font-medium mb-1">Learning Rate</label>
                <input
                  type="number"
                  value={config.hyperparameters.learningRate}
                  onChange={(e) => handleUpdateConfig('learningRate', parseFloat(e.target.value))}
                  className="w-full px-3 py-1 border rounded"
                  step="0.0001"
                  disabled={isTraining}
                />
              </div>
              
              <div>
                <label className="block text-sm font-medium mb-1">Clip Range</label>
                <input
                  type="number"
                  value={config.hyperparameters.clipRange}
                  onChange={(e) => handleUpdateConfig('clipRange', parseFloat(e.target.value))}
                  className="w-full px-3 py-1 border rounded"
                  step="0.05"
                  disabled={isTraining}
                />
              </div>
              
              <div>
                <label className="block text-sm font-medium mb-1">Gamma</label>
                <input
                  type="number"
                  value={config.hyperparameters.gamma}
                  onChange={(e) => handleUpdateConfig('gamma', parseFloat(e.target.value))}
                  className="w-full px-3 py-1 border rounded"
                  step="0.01"
                  disabled={isTraining}
                />
              </div>
              
              <div>
                <label className="block text-sm font-medium mb-1">GAE Lambda</label>
                <input
                  type="number"
                  value={config.hyperparameters.gaeBalance}
                  onChange={(e) => handleUpdateConfig('gaeBalance', parseFloat(e.target.value))}
                  className="w-full px-3 py-1 border rounded"
                  step="0.01"
                  disabled={isTraining}
                />
              </div>
              
              <div>
                <label className="block text-sm font-medium mb-1">Epochs</label>
                <input
                  type="number"
                  value={config.hyperparameters.nEpochs}
                  onChange={(e) => handleUpdateConfig('nEpochs', parseInt(e.target.value))}
                  className="w-full px-3 py-1 border rounded"
                  disabled={isTraining}
                />
              </div>
              
              <div>
                <label className="block text-sm font-medium mb-1">Batch Size</label>
                <input
                  type="number"
                  value={config.hyperparameters.batchSize}
                  onChange={(e) => handleUpdateConfig('batchSize', parseInt(e.target.value))}
                  className="w-full px-3 py-1 border rounded"
                  disabled={isTraining}
                />
              </div>
              
              <div>
                <label className="block text-sm font-medium mb-1">Entropy Coef</label>
                <input
                  type="number"
                  value={config.hyperparameters.entropyCoef}
                  onChange={(e) => handleUpdateConfig('entropyCoef', parseFloat(e.target.value))}
                  className="w-full px-3 py-1 border rounded"
                  step="0.001"
                  disabled={isTraining}
                />
              </div>
              
              <div>
                <label className="block text-sm font-medium mb-1">Value Coef</label>
                <input
                  type="number"
                  value={config.hyperparameters.valueCoef}
                  onChange={(e) => handleUpdateConfig('valueCoef', parseFloat(e.target.value))}
                  className="w-full px-3 py-1 border rounded"
                  step="0.1"
                  disabled={isTraining}
                />
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Metrics Display */}
      {status && (
        <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-4">
          <motion.div
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            className="bg-white rounded-lg shadow p-4"
          >
            <div className="flex items-center justify-between mb-2">
              <h4 className="text-sm font-medium text-gray-600">Mean Reward</h4>
              <TrendingUp className="w-4 h-4 text-green-500" />
            </div>
            <p className="text-2xl font-bold">{status.metrics.meanReward.toFixed(2)}</p>
          </motion.div>
          
          <motion.div
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ delay: 0.1 }}
            className="bg-white rounded-lg shadow p-4"
          >
            <div className="flex items-center justify-between mb-2">
              <h4 className="text-sm font-medium text-gray-600">Episode Length</h4>
              <Activity className="w-4 h-4 text-blue-500" />
            </div>
            <p className="text-2xl font-bold">{status.metrics.meanEpisodeLength.toFixed(0)}</p>
          </motion.div>
          
          <motion.div
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ delay: 0.2 }}
            className="bg-white rounded-lg shadow p-4"
          >
            <div className="flex items-center justify-between mb-2">
              <h4 className="text-sm font-medium text-gray-600">Clip Fraction</h4>
              <Zap className="w-4 h-4 text-purple-500" />
            </div>
            <p className="text-2xl font-bold">{(status.metrics.clipFraction * 100).toFixed(1)}%</p>
          </motion.div>
          
          <motion.div
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ delay: 0.3 }}
            className="bg-white rounded-lg shadow p-4"
          >
            <div className="flex items-center justify-between mb-2">
              <h4 className="text-sm font-medium text-gray-600">KL Divergence</h4>
              <AlertCircle className="w-4 h-4 text-orange-500" />
            </div>
            <p className="text-2xl font-bold">{status.metrics.klDivergence.toFixed(4)}</p>
          </motion.div>
        </div>
      )}

      {/* Charts */}
      {metricsHistory.length > 0 && (
        <div className="grid md:grid-cols-2 gap-6">
          <div className="bg-white rounded-lg shadow p-6">
            <h3 className="text-lg font-bold mb-4">Reward Progress</h3>
            <ResponsiveContainer width="100%" height={200}>
              <LineChart data={metricsHistory}>
                <XAxis dataKey="episode" />
                <YAxis />
                <Tooltip />
                <Line 
                  type="monotone" 
                  dataKey="reward" 
                  stroke="#10B981" 
                  strokeWidth={2}
                  dot={false}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
          
          <div className="bg-white rounded-lg shadow p-6">
            <h3 className="text-lg font-bold mb-4">Training Losses</h3>
            <ResponsiveContainer width="100%" height={200}>
              <LineChart data={metricsHistory}>
                <XAxis dataKey="episode" />
                <YAxis />
                <Tooltip />
                <Line 
                  type="monotone" 
                  dataKey="policyLoss" 
                  stroke="#3B82F6" 
                  strokeWidth={2}
                  dot={false}
                  name="Policy Loss"
                />
                <Line 
                  type="monotone" 
                  dataKey="valueLoss" 
                  stroke="#8B5CF6" 
                  strokeWidth={2}
                  dot={false}
                  name="Value Loss"
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>
      )}

      {/* Error Display */}
      {error && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          className="bg-red-50 border border-red-200 rounded-lg p-4"
        >
          <div className="flex items-center gap-2">
            <AlertCircle className="w-5 h-5 text-red-600" />
            <p className="text-red-800">{error}</p>
          </div>
        </motion.div>
      )}
    </div>
  );
};