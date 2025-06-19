'use client';

import React, { useState, useEffect, useMemo } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  Play, Pause, RotateCcw, Download, Settings, 
  Cpu, HardDrive, Wifi, Zap, AlertTriangle,
  BarChart3, TrendingUp, Network, Activity,
  Users, Server, Cloud, GitBranch
} from 'lucide-react';

interface ClusterNode {
  id: string;
  name: string;
  gpus: number;
  position: { x: number; y: number };
  status: 'active' | 'failed' | 'idle';
  load: number;
  memoryUsage: number;
  bandwidth: number;
}

interface TrainingConfig {
  nodes: number;
  gpusPerNode: number;
  strategy: 'data_parallel' | 'model_parallel' | 'pipeline_parallel';
  batchSize: number;
  learningRate: number;
}

interface TrainingMetrics {
  throughput: number;
  efficiency: number;
  communicationOverhead: number;
  scalingFactor: number;
}

const trainingStrategies = [
  { value: 'data_parallel', label: 'Data Parallel', icon: Users },
  { value: 'model_parallel', label: 'Model Parallel', icon: GitBranch },
  { value: 'pipeline_parallel', label: 'Pipeline Parallel', icon: BarChart3 },
];

export const DistributedTrainingVisualizer: React.FC = () => {
  const [isTraining, setIsTraining] = useState(false);
  const [selectedStrategy, setSelectedStrategy] = useState<'data_parallel' | 'model_parallel' | 'pipeline_parallel'>('data_parallel');
  const [showFault, setShowFault] = useState(false);
  const [showExport, setShowExport] = useState(false);
  const [optimizing, setOptimizing] = useState(false);
  
  const [config, setConfig] = useState<TrainingConfig>({
    nodes: 4,
    gpusPerNode: 8,
    strategy: 'data_parallel',
    batchSize: 512,
    learningRate: 0.001,
  });

  const [nodes, setNodes] = useState<ClusterNode[]>([]);

  // Generate cluster nodes based on configuration
  useEffect(() => {
    const newNodes: ClusterNode[] = [];
    const nodePositions = [
      { x: 200, y: 150 }, { x: 600, y: 150 },
      { x: 200, y: 350 }, { x: 600, y: 350 },
      { x: 400, y: 100 }, { x: 400, y: 400 },
      { x: 100, y: 250 }, { x: 700, y: 250 },
    ];

    for (let i = 0; i < config.nodes; i++) {
      newNodes.push({
        id: `node-${i}`,
        name: `Node ${i + 1}`,
        gpus: config.gpusPerNode,
        position: nodePositions[i] || { x: 400 + (i % 3 - 1) * 200, y: 250 + Math.floor(i / 3) * 150 },
        status: showFault && i === 1 ? 'failed' : isTraining ? 'active' : 'idle',
        load: isTraining ? 70 + Math.random() * 25 : 0,
        memoryUsage: isTraining ? 60 + Math.random() * 30 : 10,
        bandwidth: isTraining ? 80 + Math.random() * 15 : 5,
      });
    }
    setNodes(newNodes);
  }, [config, isTraining, showFault]);

  const trainingMetrics: TrainingMetrics = useMemo(() => {
    const activeNodes = nodes.filter(n => n.status === 'active').length;
    const totalNodes = nodes.length;
    
    return {
      throughput: activeNodes * config.gpusPerNode * 1250, // samples/sec
      efficiency: activeNodes / totalNodes * 100,
      communicationOverhead: selectedStrategy === 'data_parallel' ? 15 : 
                            selectedStrategy === 'model_parallel' ? 25 : 35,
      scalingFactor: Math.min(activeNodes / totalNodes, 0.95),
    };
  }, [nodes, config, selectedStrategy]);

  const startTraining = () => {
    setIsTraining(true);
  };

  const pauseTraining = () => {
    setIsTraining(false);
  };

  const resetTraining = () => {
    setIsTraining(false);
    setShowFault(false);
    setOptimizing(false);
  };

  const simulateFailure = () => {
    setShowFault(true);
  };

  const optimizeTopology = () => {
    setOptimizing(true);
    setTimeout(() => setOptimizing(false), 3000);
  };

  const exportConfig = () => {
    const configData = {
      config,
      metrics: trainingMetrics,
      timestamp: new Date().toISOString(),
    };
    
    const blob = new Blob([JSON.stringify(configData, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'distributed-training-config.json';
    a.click();
    URL.revokeObjectURL(url);
  };

  const renderNode = (node: ClusterNode) => {
    const statusColor = node.status === 'active' ? 'bg-green-500' : 
                       node.status === 'failed' ? 'bg-red-500' : 'bg-gray-400';

    return (
      <motion.div
        key={node.id}
        data-testid={node.id}
        className="absolute transform -translate-x-1/2 -translate-y-1/2"
        style={{
          left: node.position.x,
          top: node.position.y,
        }}
        whileHover={{ scale: 1.05 }}
        animate={optimizing ? { scale: [1, 1.1, 1] } : {}}
        transition={{ duration: 0.5, repeat: optimizing ? Infinity : 0 }}
      >
        <div className={`${statusColor} rounded-lg p-4 shadow-lg min-w-[120px] text-center text-white`}>
          <Server className="w-6 h-6 mx-auto mb-2" />
          <h3 className="font-bold text-sm">{node.name}</h3>
          <div className="text-xs mt-1">
            <div>{node.gpus}x GPUs</div>
            <div>{node.load.toFixed(0)}% Load</div>
            <div>{node.memoryUsage.toFixed(0)}% Mem</div>
          </div>
          
          {node.status === 'failed' && (
            <AlertTriangle className="w-4 h-4 mx-auto mt-1 text-yellow-300" />
          )}
        </div>

        {/* GPU visualization */}
        <div className="absolute -bottom-8 left-1/2 transform -translate-x-1/2">
          <div className="flex gap-1">
            {Array.from({ length: Math.min(node.gpus, 8) }).map((_, i) => (
              <div
                key={i}
                className={`w-2 h-3 rounded-sm ${
                  node.status === 'active' ? 'bg-blue-400' : 'bg-gray-300'
                }`}
              />
            ))}
          </div>
        </div>
      </motion.div>
    );
  };

  const renderCommunicationLines = () => {
    if (!isTraining) return null;

    return (
      <g data-testid="communication-patterns">
        {nodes.map((fromNode, i) => 
          nodes.slice(i + 1).map((toNode, j) => {
            if (fromNode.status === 'failed' || toNode.status === 'failed') return null;
            
            return (
              <g key={`${fromNode.id}-${toNode.id}`}>
                <motion.line
                  x1={fromNode.position.x}
                  y1={fromNode.position.y}
                  x2={toNode.position.x}
                  y2={toNode.position.y}
                  stroke="#3B82F6"
                  strokeWidth="2"
                  strokeOpacity="0.6"
                  animate={{
                    strokeDasharray: ['0,10', '10,0'],
                  }}
                  transition={{
                    duration: 1,
                    repeat: Infinity,
                    ease: 'linear'
                  }}
                />
                
                {/* Data flow animation */}
                <motion.circle
                  cx={fromNode.position.x}
                  cy={fromNode.position.y}
                  r="3"
                  fill="#EF4444"
                  animate={{
                    cx: [fromNode.position.x, toNode.position.x],
                    cy: [fromNode.position.y, toNode.position.y],
                  }}
                  transition={{
                    duration: 2,
                    repeat: Infinity,
                    ease: 'easeInOut'
                  }}
                />
              </g>
            );
          })
        )}
      </g>
    );
  };

  return (
    <div className="bg-white rounded-lg border shadow-lg p-6">
      {/* Header */}
      <div className="flex justify-between items-center mb-6">
        <div>
          <h2 className="text-2xl font-bold">Distributed Training Visualizer</h2>
          <p className="text-gray-600">Visualize distributed machine learning across clusters</p>
        </div>
        
        <div className="flex items-center gap-2">
          <button
            onClick={isTraining ? pauseTraining : startTraining}
            className={`px-4 py-2 rounded-lg flex items-center gap-2 text-white ${
              isTraining ? 'bg-red-500 hover:bg-red-600' : 'bg-green-500 hover:bg-green-600'
            }`}
          >
            {isTraining ? <Pause size={16} /> : <Play size={16} />}
            {isTraining ? 'Pause Training' : 'Start Training'}
          </button>
          
          <button
            onClick={resetTraining}
            className="px-4 py-2 bg-gray-500 text-white rounded-lg hover:bg-gray-600 flex items-center gap-2"
          >
            <RotateCcw size={16} />
            Reset
          </button>
          
          <button
            onClick={simulateFailure}
            disabled={showFault}
            className="px-4 py-2 bg-yellow-500 text-white rounded-lg hover:bg-yellow-600 disabled:opacity-50 flex items-center gap-2"
          >
            <AlertTriangle size={16} />
            Simulate Failure
          </button>
          
          <button
            onClick={() => setShowExport(true)}
            className="px-4 py-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600 flex items-center gap-2"
          >
            <Download size={16} />
            Export Config
          </button>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
        {/* Configuration Panel */}
        <div className="lg:col-span-1">
          <div className="bg-gray-50 rounded-lg p-4 mb-4">
            <h3 className="font-semibold mb-4">Cluster Configuration</h3>
            
            <div className="space-y-4">
              <div>
                <label htmlFor="nodes" className="block text-sm font-medium mb-1">Nodes</label>
                <input
                  id="nodes"
                  type="number"
                  value={config.nodes}
                  onChange={(e) => setConfig({
                    ...config,
                    nodes: Math.min(8, Math.max(1, parseInt(e.target.value) || 1))
                  })}
                  className="w-full p-2 border rounded"
                  min="1"
                  max="8"
                />
              </div>
              
              <div>
                <label htmlFor="gpus-per-node" className="block text-sm font-medium mb-1">GPUs per Node</label>
                <input
                  id="gpus-per-node"
                  type="number"
                  value={config.gpusPerNode}
                  onChange={(e) => setConfig({
                    ...config,
                    gpusPerNode: Math.min(16, Math.max(1, parseInt(e.target.value) || 1))
                  })}
                  className="w-full p-2 border rounded"
                  min="1"
                  max="16"
                />
              </div>
              
              <div>
                <label htmlFor="training-strategy" className="block text-sm font-medium mb-1">Training Strategy</label>
                <select
                  id="training-strategy"
                  value={selectedStrategy}
                  onChange={(e) => {
                    const strategy = e.target.value as typeof selectedStrategy;
                    setSelectedStrategy(strategy);
                    setConfig({ ...config, strategy });
                  }}
                  className="w-full p-2 border rounded"
                >
                  {trainingStrategies.map(strategy => (
                    <option key={strategy.value} value={strategy.value}>
                      {strategy.label}
                    </option>
                  ))}
                </select>
              </div>
              
              <div>
                <label htmlFor="batch-size" className="block text-sm font-medium mb-1">Global Batch Size</label>
                <input
                  id="batch-size"
                  type="number"
                  value={config.batchSize}
                  onChange={(e) => setConfig({
                    ...config,
                    batchSize: parseInt(e.target.value) || 512
                  })}
                  className="w-full p-2 border rounded"
                  step="64"
                />
              </div>
            </div>

            <button
              onClick={optimizeTopology}
              disabled={optimizing}
              className="w-full mt-4 px-4 py-2 bg-purple-500 text-white rounded-lg hover:bg-purple-600 disabled:opacity-50 flex items-center justify-center gap-2"
            >
              <Network size={16} />
              Optimize Topology
            </button>
          </div>

          {/* Parallelism Strategy Buttons */}
          <div className="space-y-2">
            {trainingStrategies.map(strategy => {
              const Icon = strategy.icon;
              return (
                <button
                  key={strategy.value}
                  onClick={() => {
                    setSelectedStrategy(strategy.value as typeof selectedStrategy);
                    setConfig({ ...config, strategy: strategy.value as typeof selectedStrategy });
                  }}
                  className={`w-full p-3 rounded-lg border-2 flex items-center gap-2 ${
                    selectedStrategy === strategy.value
                      ? 'border-blue-500 bg-blue-50 text-blue-700'
                      : 'border-gray-200 hover:border-gray-300'
                  }`}
                >
                  <Icon size={16} />
                  {strategy.label}
                </button>
              );
            })}
          </div>
        </div>

        {/* Cluster Visualization */}
        <div className="lg:col-span-3">
          <div className="relative bg-gray-50 rounded-lg p-8 min-h-[500px] overflow-hidden">
            {isTraining && (
              <div 
                data-testid="gradient-sync-animation"
                className="absolute top-4 left-4 bg-blue-500 text-white px-3 py-1 rounded-lg z-10"
              >
                Gradient Synchronization Active
              </div>
            )}
            
            {showFault && (
              <div 
                data-testid="fault-simulation"
                className="absolute top-4 right-4 bg-red-500 text-white px-3 py-1 rounded-lg z-10"
              >
                Node Failure Detected
              </div>
            )}
            
            {optimizing && (
              <div 
                data-testid="topology-optimization"
                className="absolute top-4 left-1/2 transform -translate-x-1/2 bg-purple-500 text-white px-3 py-1 rounded-lg z-10"
              >
                Optimizing Network Topology...
              </div>
            )}

            <svg 
              data-testid="cluster-topology"
              className="absolute inset-0 w-full h-full pointer-events-none"
              style={{ zIndex: 5 }}
            >
              {renderCommunicationLines()}
            </svg>

            {/* Master Node */}
            <div className="absolute top-4 left-1/2 transform -translate-x-1/2">
              <div className="bg-blue-600 rounded-lg p-3 text-white text-center">
                <Cloud className="w-6 h-6 mx-auto mb-1" />
                <div className="text-sm font-bold">Master Node</div>
                <div className="text-xs">Parameter Server</div>
              </div>
            </div>

            {/* Worker Nodes */}
            {nodes.map(renderNode)}
          </div>
        </div>
      </div>

      {/* Training Metrics */}
      <div className="mt-6 grid grid-cols-1 md:grid-cols-4 gap-4">
        <div className="bg-green-50 p-4 rounded-lg">
          <h3 className="font-semibold text-green-800 mb-2">Throughput</h3>
          <div className="text-2xl font-bold text-green-600">
            {trainingMetrics.throughput.toLocaleString()}
          </div>
          <div className="text-sm text-green-600">samples/sec</div>
        </div>
        
        <div className="bg-blue-50 p-4 rounded-lg">
          <h3 className="font-semibold text-blue-800 mb-2">Efficiency</h3>
          <div className="text-2xl font-bold text-blue-600">
            {trainingMetrics.efficiency.toFixed(1)}%
          </div>
          <div className="text-sm text-blue-600">cluster utilization</div>
        </div>
        
        <div className="bg-orange-50 p-4 rounded-lg">
          <h3 className="font-semibold text-orange-800 mb-2">Communication Overhead</h3>
          <div className="text-2xl font-bold text-orange-600">
            {trainingMetrics.communicationOverhead}%
          </div>
          <div className="text-sm text-orange-600">network usage</div>
        </div>
        
        <div className="bg-purple-50 p-4 rounded-lg">
          <h3 className="font-semibold text-purple-800 mb-2">Scaling Efficiency</h3>
          <div className="text-2xl font-bold text-purple-600">
            {(trainingMetrics.scalingFactor * 100).toFixed(1)}%
          </div>
          <div className="text-sm text-purple-600">linear scaling</div>
        </div>
      </div>

      {/* Resource Utilization */}
      <div className="mt-6 grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* Load Distribution */}
        <div className="bg-gray-50 p-4 rounded-lg">
          <h3 className="font-semibold mb-3">Load Distribution</h3>
          <div className="space-y-2">
            {nodes.map((node, index) => (
              <div key={node.id} className="flex items-center gap-3">
                <div className="w-16 text-sm">{node.name}</div>
                <div 
                  data-testid={`load-bar-${index}`}
                  className="flex-1 bg-gray-200 rounded-full h-4 relative"
                >
                  <div
                    className={`h-4 rounded-full transition-all ${
                      node.status === 'active' ? 'bg-green-500' :
                      node.status === 'failed' ? 'bg-red-500' : 'bg-gray-400'
                    }`}
                    style={{ width: `${node.load}%` }}
                  />
                  <div className="absolute inset-0 flex items-center justify-center text-xs font-medium text-white">
                    {node.load.toFixed(0)}%
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Memory Usage */}
        <div className="bg-gray-50 p-4 rounded-lg">
          <h3 className="font-semibold mb-3">Memory Usage</h3>
          <div className="space-y-2">
            {nodes.map((node, index) => (
              <div key={node.id} className="flex items-center gap-3">
                <div className="w-16 text-sm">{node.name}</div>
                <div 
                  data-testid={`memory-usage-${index}`}
                  className="flex-1 bg-gray-200 rounded-full h-4 relative"
                >
                  <div
                    className={`h-4 rounded-full transition-all ${
                      node.memoryUsage > 80 ? 'bg-red-500' :
                      node.memoryUsage > 60 ? 'bg-yellow-500' : 'bg-blue-500'
                    }`}
                    style={{ width: `${node.memoryUsage}%` }}
                  />
                  <div className="absolute inset-0 flex items-center justify-center text-xs font-medium text-white">
                    {node.memoryUsage.toFixed(0)}%
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Network and GPU Utilization */}
      <div className="mt-6 grid grid-cols-1 md:grid-cols-3 gap-4">
        <div className="bg-white border rounded-lg p-4">
          <div className="flex items-center gap-2 mb-2">
            <Wifi className="w-5 h-5 text-blue-600" />
            <span className="font-semibold">Network Bandwidth</span>
          </div>
          <div className="text-2xl font-bold text-blue-600">
            {nodes.reduce((sum, node) => sum + node.bandwidth, 0).toFixed(0)} Gbps
          </div>
        </div>
        
        <div className="bg-white border rounded-lg p-4">
          <div className="flex items-center gap-2 mb-2">
            <Zap className="w-5 h-5 text-green-600" />
            <span className="font-semibold">GPU Utilization</span>
          </div>
          <div className="text-2xl font-bold text-green-600">
            {(nodes.reduce((sum, node) => sum + node.load, 0) / nodes.length).toFixed(0)}%
          </div>
        </div>
        
        <div className="bg-white border rounded-lg p-4">
          <div className="flex items-center gap-2 mb-2">
            <Activity className="w-5 h-5 text-purple-600" />
            <span className="font-semibold">Active Nodes</span>
          </div>
          <div className="text-2xl font-bold text-purple-600">
            {nodes.filter(n => n.status === 'active').length}/{nodes.length}
          </div>
        </div>
      </div>

      {/* Scaling Efficiency Chart */}
      <div className="mt-6 bg-gray-50 p-4 rounded-lg">
        <h3 className="font-semibold mb-3">Scaling Efficiency</h3>
        <div 
          data-testid="efficiency-chart"
          className="h-32 bg-white rounded border flex items-end justify-around p-4"
        >
          {[1, 2, 4, 8].map(nodeCount => {
            const efficiency = Math.max(20, 100 - (nodeCount - 1) * 12); // Diminishing returns
            return (
              <div key={nodeCount} className="flex flex-col items-center">
                <div
                  className="bg-blue-500 w-8 rounded-t"
                  style={{ height: `${efficiency}%` }}
                />
                <div className="text-xs mt-1">{nodeCount} nodes</div>
              </div>
            );
          })}
        </div>
      </div>

      {/* Export Modal */}
      {showExport && (
        <div 
          data-testid="export-modal"
          className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50"
          onClick={() => setShowExport(false)}
        >
          <div 
            className="bg-white rounded-lg p-6 max-w-md w-full mx-4"
            onClick={(e) => e.stopPropagation()}
          >
            <h3 className="text-lg font-bold mb-4">Export Configuration</h3>
            <p className="text-gray-600 mb-4">
              Download your distributed training configuration and metrics.
            </p>
            <div className="flex justify-end gap-2">
              <button
                onClick={() => setShowExport(false)}
                className="px-4 py-2 text-gray-600 hover:text-gray-800"
              >
                Cancel
              </button>
              <button
                onClick={() => {
                  exportConfig();
                  setShowExport(false);
                }}
                className="px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600 flex items-center gap-2"
              >
                <Download size={16} />
                Download
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};