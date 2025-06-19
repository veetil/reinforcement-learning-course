'use client';

import React, { useState, useCallback, useEffect, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  Play, Pause, RotateCcw, Download, TrendingUp, 
  AlertTriangle, BarChart3, Zap 
} from 'lucide-react';
import { PPOCalculator, PolicyParams, UpdateStats } from './PPOCalculator';
import { Line } from 'recharts';
import dynamic from 'next/dynamic';

const ResponsiveContainer = dynamic(
  () => import('recharts').then(mod => mod.ResponsiveContainer),
  { ssr: false }
);
const LineChart = dynamic(
  () => import('recharts').then(mod => mod.LineChart),
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
const Legend = dynamic(
  () => import('recharts').then(mod => mod.Legend),
  { ssr: false }
);

interface PolicyUpdateSimulatorProps {
  initialEpsilon?: number;
  initialLearningRate?: number;
  onUpdate?: (stats: UpdateStats) => void;
}

export const PolicyUpdateSimulator: React.FC<PolicyUpdateSimulatorProps> = ({
  initialEpsilon = 0.2,
  initialLearningRate = 0.0003,
  onUpdate
}) => {
  const [calculator] = useState(() => new PPOCalculator());
  const [epsilon, setEpsilon] = useState(initialEpsilon);
  const [learningRate, setLearningRate] = useState(initialLearningRate);
  const [isTraining, setIsTraining] = useState(false);
  const [episode, setEpisode] = useState(0);
  const [showKL, setShowKL] = useState(false);
  const [earlyStop, setEarlyStop] = useState(false);
  const [compareMode, setCompareMode] = useState(false);
  const [exportMessage, setExportMessage] = useState('');
  
  const [oldPolicy, setOldPolicy] = useState<PolicyParams>({ mean: 0, std: 1 });
  const [newPolicy, setNewPolicy] = useState<PolicyParams>({ mean: 0, std: 1 });
  const [trajectory, setTrajectory] = useState<ReturnType<typeof calculator.generateTrajectory>>();
  const [updateHistory, setUpdateHistory] = useState<UpdateStats[]>([]);
  const [lossHistory, setLossHistory] = useState<any[]>([]);
  
  const trainingRef = useRef<NodeJS.Timeout>();

  useEffect(() => {
    return () => {
      if (trainingRef.current) {
        clearTimeout(trainingRef.current);
      }
    };
  }, []);

  const performUpdate = useCallback(() => {
    // Generate trajectory
    const traj = calculator.generateTrajectory(oldPolicy, 32);
    setTrajectory(traj);

    // Calculate advantages
    const advantages = calculator.computeAdvantages(traj.rewards, traj.values);

    // Calculate old log probs
    const oldLogProbs = traj.actions.map(action => 
      calculator.calculateLogProb(action, oldPolicy)
    );

    // Update policy
    const gradient = {
      mean: advantages.reduce((sum, adv, i) => 
        sum + adv * (traj.actions[i] - oldPolicy.mean) / (oldPolicy.std * oldPolicy.std), 0
      ) / advantages.length,
      std: advantages.reduce((sum, adv, i) => {
        const diff = traj.actions[i] - oldPolicy.mean;
        return sum + adv * ((diff * diff) / (oldPolicy.std * oldPolicy.std * oldPolicy.std) - 1 / oldPolicy.std);
      }, 0) / advantages.length
    };

    const updated = calculator.updatePolicy(oldPolicy, gradient, learningRate);
    setNewPolicy(updated);

    // Calculate new log probs and ratios
    const newLogProbs = traj.actions.map(action => 
      calculator.calculateLogProb(action, updated)
    );
    const ratios = newLogProbs.map((newLP, i) => 
      calculator.calculateRatio(newLP, oldLogProbs[i])
    );

    // Calculate losses
    const policyLosses = ratios.map((ratio, i) => 
      -calculator.clipObjective(ratio, advantages[i], epsilon)
    );
    const policyLoss = policyLosses.reduce((sum, loss) => sum + loss, 0) / policyLosses.length;

    // Value loss (simplified)
    const valueLoss = traj.values.reduce((sum, v, i) => 
      sum + Math.pow(traj.rewards[i] - v, 2), 0
    ) / traj.values.length;

    // Entropy
    const probs = traj.actions.map(() => 1 / 4); // Simplified
    const entropy = calculator.calculateEntropy(probs);

    // KL divergence
    const klDivergence = Math.abs(updated.mean - oldPolicy.mean) + 
                        Math.abs(Math.log(updated.std / oldPolicy.std));

    // Clip fraction
    const clipFraction = calculator.calculateClipFraction(ratios, advantages, epsilon);

    const stats: UpdateStats = {
      policyLoss,
      valueLoss,
      entropy,
      klDivergence,
      clipFraction
    };

    setUpdateHistory(prev => [...prev, stats]);
    setLossHistory(prev => [...prev, {
      episode: episode + 1,
      policyLoss,
      valueLoss,
      entropy,
      klDivergence,
      totalLoss: policyLoss + 0.5 * valueLoss - 0.01 * entropy
    }]);

    // Check early stopping
    if (earlyStop && calculator.checkEarlyStopping(klDivergence)) {
      setIsTraining(false);
    }

    setEpisode(prev => prev + 1);
    onUpdate?.(stats);

    // Update old policy for next iteration
    setOldPolicy(updated);
  }, [calculator, oldPolicy, learningRate, epsilon, episode, earlyStop, onUpdate]);

  const handleStep = useCallback(() => {
    performUpdate();
  }, [performUpdate]);

  const handleTraining = useCallback(() => {
    if (isTraining) {
      setIsTraining(false);
      if (trainingRef.current) {
        clearTimeout(trainingRef.current);
      }
    } else {
      setIsTraining(true);
      const train = () => {
        performUpdate();
        trainingRef.current = setTimeout(train, 100);
      };
      train();
    }
  }, [isTraining, performUpdate]);

  const handleReset = useCallback(() => {
    setIsTraining(false);
    if (trainingRef.current) {
      clearTimeout(trainingRef.current);
    }
    setEpisode(0);
    setOldPolicy({ mean: 0, std: 1 });
    setNewPolicy({ mean: 0, std: 1 });
    setUpdateHistory([]);
    setLossHistory([]);
    setTrajectory(undefined);
  }, []);

  const handleExport = useCallback(() => {
    const data = {
      settings: { epsilon, learningRate },
      episode,
      oldPolicy,
      newPolicy,
      updateHistory,
      lossHistory
    };

    if (typeof window !== 'undefined' && window.URL && window.URL.createObjectURL) {
      const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = 'ppo-training-data.json';
      a.click();
      URL.revokeObjectURL(url);
    }

    setExportMessage('Data exported!');
    setTimeout(() => setExportMessage(''), 2000);
  }, [epsilon, learningRate, episode, oldPolicy, newPolicy, updateHistory, lossHistory]);

  const renderPolicyDistribution = () => {
    const x = Array.from({ length: 100 }, (_, i) => (i - 50) / 10);
    const oldY = x.map(val => 
      Math.exp(calculator.calculateLogProb(val, oldPolicy))
    );
    const newY = x.map(val => 
      Math.exp(calculator.calculateLogProb(val, newPolicy))
    );

    const maxY = Math.max(...oldY, ...newY);

    return (
      <div data-testid="policy-distribution" className="h-48 relative">
        <svg className="w-full h-full">
          <path
            d={`M ${x.map((val, i) => 
              `${(val + 5) * 30},${180 - oldY[i] / maxY * 160}`
            ).join(' L ')}`}
            fill="none"
            stroke="blue"
            strokeWidth="2"
            opacity="0.5"
          />
          <path
            d={`M ${x.map((val, i) => 
              `${(val + 5) * 30},${180 - newY[i] / maxY * 160}`
            ).join(' L ')}`}
            fill="none"
            stroke="green"
            strokeWidth="2"
          />
        </svg>
        <div className="absolute top-0 right-0 text-xs">
          <div className="flex items-center gap-2">
            <div className="w-4 h-0.5 bg-blue-500 opacity-50"></div>
            <span>Old Policy</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-4 h-0.5 bg-green-500"></div>
            <span>New Policy</span>
          </div>
        </div>
      </div>
    );
  };

  const renderClippingVisualization = () => {
    const advantages = [-1, -0.5, 0, 0.5, 1];
    const ratios = Array.from({ length: 50 }, (_, i) => i / 25);

    return (
      <div data-testid="clipping-visualization" className="h-48 bg-gray-50 rounded p-4">
        <svg className="w-full h-full">
          {advantages.map((adv, idx) => {
            const color = adv > 0 ? 'green' : adv < 0 ? 'red' : 'gray';
            const opacity = Math.abs(adv);
            
            return (
              <g key={idx}>
                <path
                  d={`M ${ratios.map((r, i) => {
                    const obj = calculator.clipObjective(r, adv, epsilon);
                    return `${i * 6},${80 - obj * 40}`;
                  }).join(' L ')}`}
                  fill="none"
                  stroke={color}
                  strokeWidth="2"
                  opacity={opacity}
                />
              </g>
            );
          })}
          <line x1="60" y1="0" x2="60" y2="160" stroke="black" strokeDasharray="2,2" />
          <text x="65" y="10" fontSize="10">r=1</text>
        </svg>
        <p className="text-xs text-center mt-2">PPO Clipping (ε = {epsilon.toFixed(2)})</p>
      </div>
    );
  };

  return (
    <div className="w-full space-y-4">
      <h2 className="text-2xl font-bold">PPO Policy Update Simulator</h2>
      
      {/* Controls */}
      <div className="flex flex-wrap gap-4">
        <div className="flex items-center gap-2">
          <label htmlFor="epsilon-slider" className="text-sm font-medium">
            Clip Range (ε)
          </label>
          <input
            id="epsilon-slider"
            type="range"
            min="0"
            max="0.5"
            step="0.05"
            value={epsilon}
            onChange={(e) => setEpsilon(parseFloat(e.target.value))}
            className="w-32"
            aria-label="Clip Range (ε)"
          />
          <span className="text-sm font-mono">ε = {epsilon.toFixed(2)}</span>
        </div>

        <div className="flex items-center gap-2">
          <label htmlFor="lr-slider" className="text-sm font-medium">
            Learning Rate
          </label>
          <input
            id="lr-slider"
            type="range"
            min="0.0001"
            max="0.001"
            step="0.0001"
            value={learningRate}
            onChange={(e) => setLearningRate(parseFloat(e.target.value))}
            className="w-32"
            aria-label="Learning Rate"
          />
          <span className="text-sm font-mono">{learningRate.toFixed(4)}</span>
        </div>

        <button
          onClick={handleStep}
          disabled={isTraining}
          className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:bg-gray-400"
        >
          Step Update
        </button>

        <button
          onClick={handleTraining}
          className="px-4 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700 flex items-center gap-2"
        >
          {isTraining ? (
            <>
              <Pause className="w-4 h-4" />
              Pause Training
            </>
          ) : (
            <>
              <Play className="w-4 h-4" />
              Start Training
            </>
          )}
        </button>

        <button
          onClick={handleReset}
          className="px-4 py-2 bg-gray-600 text-white rounded-lg hover:bg-gray-700 flex items-center gap-2"
        >
          <RotateCcw className="w-4 h-4" />
          Reset
        </button>

        <button
          onClick={() => setShowKL(!showKL)}
          className="px-4 py-2 bg-purple-600 text-white rounded-lg hover:bg-purple-700"
        >
          {showKL ? 'Hide' : 'Show'} KL
        </button>

        <button
          onClick={() => setEarlyStop(!earlyStop)}
          className={`px-4 py-2 rounded-lg ${
            earlyStop ? 'bg-orange-600 text-white' : 'bg-gray-200 text-gray-700'
          }`}
        >
          Early Stopping
        </button>

        <button
          onClick={() => setCompareMode(!compareMode)}
          className="px-4 py-2 bg-indigo-600 text-white rounded-lg hover:bg-indigo-700"
        >
          Compare Clipped/Unclipped
        </button>

        <button
          onClick={handleExport}
          className="px-4 py-2 bg-teal-600 text-white rounded-lg hover:bg-teal-700 flex items-center gap-2"
        >
          <Download className="w-4 h-4" />
          Export Data
        </button>
      </div>

      {/* Export message */}
      <AnimatePresence>
        {exportMessage && (
          <motion.div
            initial={{ opacity: 0, y: -10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0 }}
            className="text-green-600 font-medium"
          >
            {exportMessage}
          </motion.div>
        )}
      </AnimatePresence>

      {/* Status */}
      <div className="bg-white rounded-lg shadow p-4">
        <div className="flex items-center justify-between">
          <h3 className="font-bold">Training Status</h3>
          <span className="text-sm text-gray-600">Episode: {episode}</span>
        </div>
        
        {updateHistory.length > 0 && (
          <div className="mt-2 grid grid-cols-3 gap-4 text-sm">
            <div>
              <span className="text-gray-600">Policy Loss:</span>
              <span className="ml-2 font-mono">
                {updateHistory[updateHistory.length - 1].policyLoss.toFixed(4)}
              </span>
            </div>
            <div>
              <span className="text-gray-600">Value Loss:</span>
              <span className="ml-2 font-mono">
                {updateHistory[updateHistory.length - 1].valueLoss.toFixed(4)}
              </span>
            </div>
            <div>
              <span className="text-gray-600">Entropy:</span>
              <span className="ml-2 font-mono">
                {updateHistory[updateHistory.length - 1].entropy.toFixed(4)}
              </span>
            </div>
          </div>
        )}
      </div>

      {/* Main Visualizations */}
      <div className="grid md:grid-cols-2 gap-6">
        {/* Policy Distribution */}
        <div className="bg-white rounded-lg shadow p-4">
          <h3 className="font-bold mb-2">Policy Distribution</h3>
          {renderPolicyDistribution()}
          <div className="mt-2 text-xs text-gray-600">
            <p>μ_old = {oldPolicy.mean.toFixed(3)}, σ_old = {oldPolicy.std.toFixed(3)}</p>
            <p>μ_new = {newPolicy.mean.toFixed(3)}, σ_new = {newPolicy.std.toFixed(3)}</p>
          </div>
        </div>

        {/* Clipping Visualization */}
        <div className="bg-white rounded-lg shadow p-4">
          <h3 className="font-bold mb-2">PPO Clipping Function</h3>
          {renderClippingVisualization()}
        </div>
      </div>

      {/* Loss History Chart */}
      {lossHistory.length > 0 && (
        <div className="bg-white rounded-lg shadow p-4">
          <h3 className="font-bold mb-2">Loss History</h3>
          <ResponsiveContainer width="100%" height={200}>
            <LineChart data={lossHistory}>
              <XAxis dataKey="episode" />
              <YAxis />
              <Tooltip />
              <Legend />
              <Line type="monotone" dataKey="policyLoss" stroke="#3B82F6" strokeWidth={2} />
              <Line type="monotone" dataKey="valueLoss" stroke="#10B981" strokeWidth={2} />
              <Line type="monotone" dataKey="totalLoss" stroke="#8B5CF6" strokeWidth={2} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      )}

      {/* KL Divergence */}
      {showKL && updateHistory.length > 0 && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          data-testid="kl-divergence"
          className="bg-white rounded-lg shadow p-4"
        >
          <h3 className="font-bold mb-2 flex items-center gap-2">
            <AlertTriangle className="w-4 h-4 text-orange-500" />
            KL Divergence
          </h3>
          <div className="h-32">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={updateHistory.map((s, i) => ({ episode: i, kl: s.klDivergence }))}>
                <XAxis dataKey="episode" />
                <YAxis />
                <Tooltip />
                <Line type="monotone" dataKey="kl" stroke="#F97316" strokeWidth={2} />
                {earlyStop && (
                  <Line
                    type="monotone"
                    dataKey={() => 0.01}
                    stroke="#EF4444"
                    strokeDasharray="5 5"
                  />
                )}
              </LineChart>
            </ResponsiveContainer>
          </div>
          {earlyStop && (
            <p className="text-xs text-gray-600 mt-2">
              KL Threshold: 0.01 (training stops if exceeded)
            </p>
          )}
        </motion.div>
      )}

      {/* Comparison Chart */}
      {compareMode && trajectory && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          data-testid="comparison-chart"
          className="bg-white rounded-lg shadow p-4"
        >
          <h3 className="font-bold mb-2">Clipped vs Unclipped Objective</h3>
          <div className="text-xs text-gray-600">
            Shows how clipping prevents large policy updates
          </div>
        </motion.div>
      )}
    </div>
  );
};