'use client';

import React, { useState, useCallback, useMemo } from 'react';
import {
  ReactFlow,
  Node,
  Edge,
  Background,
  Controls,
  MiniMap,
  useNodesState,
  useEdgesState,
  MarkerType,
  Position,
  Handle,
  NodeProps,
  getBezierPath,
  EdgeProps,
} from '@xyflow/react';
import { motion, AnimatePresence } from 'framer-motion';
import '@xyflow/react/dist/style.css';

// Custom node types for VERL components
interface WorkerNodeData {
  label: string;
  workerType: 'actor' | 'critic' | 'rollout' | 'reward' | 'controller';
  gpuCount?: number;
  status: 'idle' | 'busy' | 'syncing';
  metrics?: {
    throughput?: number;
    gpuUtilization?: number;
    memoryUsage?: number;
  };
}

interface ResourcePoolData {
  label: string;
  totalGPUs: number;
  allocatedGPUs: number;
  poolId: string;
}

// Custom Worker Node Component
const WorkerNode: React.FC<NodeProps<WorkerNodeData>> = ({ data, selected }) => {
  const statusColors = {
    idle: '#94a3b8',
    busy: '#22c55e',
    syncing: '#f59e0b',
  };

  const workerColors = {
    actor: '#8b5cf6',
    critic: '#3b82f6',
    rollout: '#10b981',
    reward: '#f59e0b',
    controller: '#ef4444',
  };

  return (
    <motion.div
      initial={{ scale: 0 }}
      animate={{ scale: selected ? 1.1 : 1 }}
      className="relative bg-white rounded-lg shadow-lg p-4 min-w-[200px]"
      style={{ borderColor: workerColors[data.workerType], borderWidth: 2 }}
    >
      <Handle type="target" position={Position.Top} className="w-3 h-3" />
      <Handle type="source" position={Position.Bottom} className="w-3 h-3" />
      
      <div className="flex items-center justify-between mb-2">
        <h3 className="font-bold text-sm">{data.label}</h3>
        <div
          className="w-3 h-3 rounded-full"
          style={{ backgroundColor: statusColors[data.status] }}
        />
      </div>
      
      {data.gpuCount && (
        <div className="text-xs text-gray-600 mb-1">
          GPUs: {data.gpuCount}
        </div>
      )}
      
      {data.metrics && (
        <div className="space-y-1 mt-2 pt-2 border-t">
          {data.metrics.throughput && (
            <div className="text-xs">
              Throughput: {data.metrics.throughput.toFixed(0)} tok/s
            </div>
          )}
          {data.metrics.gpuUtilization && (
            <div className="text-xs">
              GPU: {(data.metrics.gpuUtilization * 100).toFixed(0)}%
            </div>
          )}
          {data.metrics.memoryUsage && (
            <div className="text-xs">
              Memory: {(data.metrics.memoryUsage * 100).toFixed(0)}%
            </div>
          )}
        </div>
      )}
    </motion.div>
  );
};

// Custom Resource Pool Node
const ResourcePoolNode: React.FC<NodeProps<ResourcePoolData>> = ({ data }) => {
  const utilizationPercent = (data.allocatedGPUs / data.totalGPUs) * 100;
  
  return (
    <div className="bg-gray-100 rounded-lg p-4 min-w-[250px] border-2 border-gray-300">
      <h3 className="font-bold text-sm mb-2">{data.label}</h3>
      <div className="mb-2">
        <div className="text-xs text-gray-600 mb-1">
          {data.allocatedGPUs} / {data.totalGPUs} GPUs
        </div>
        <div className="w-full bg-gray-300 rounded-full h-2">
          <motion.div
            className="bg-blue-500 h-2 rounded-full"
            initial={{ width: 0 }}
            animate={{ width: `${utilizationPercent}%` }}
            transition={{ duration: 0.5 }}
          />
        </div>
      </div>
    </div>
  );
};

// Animated Edge Component
const AnimatedEdge: React.FC<EdgeProps> = ({
  id,
  sourceX,
  sourceY,
  targetX,
  targetY,
  sourcePosition,
  targetPosition,
  style = {},
  markerEnd,
}) => {
  const [edgePath] = getBezierPath({
    sourceX,
    sourceY,
    sourcePosition,
    targetX,
    targetY,
    targetPosition,
  });

  return (
    <>
      <path
        id={id}
        style={style}
        className="react-flow__edge-path"
        d={edgePath}
        markerEnd={markerEnd}
      />
      <motion.circle
        r="4"
        fill="#3b82f6"
        initial={{ offsetDistance: '0%' }}
        animate={{ offsetDistance: '100%' }}
        transition={{
          duration: 2,
          repeat: Infinity,
          ease: 'linear',
        }}
      >
        <animateMotion dur="2s" repeatCount="indefinite">
          <mpath href={`#${id}`} />
        </animateMotion>
      </motion.circle>
    </>
  );
};

const nodeTypes = {
  worker: WorkerNode,
  resourcePool: ResourcePoolNode,
};

const edgeTypes = {
  animated: AnimatedEdge,
};

interface VERLSystemVisualizerProps {
  className?: string;
}

export const VERLSystemVisualizer: React.FC<VERLSystemVisualizerProps> = ({
  className = '',
}) => {
  const [selectedConfig, setSelectedConfig] = useState<'colocated' | 'split' | 'hybrid'>('split');
  const [showDataFlow, setShowDataFlow] = useState(true);
  const [simulationRunning, setSimulationRunning] = useState(false);

  // Define nodes based on configuration
  const getNodesForConfig = useCallback((config: string) => {
    const baseNodes: Node[] = [
      {
        id: 'controller',
        type: 'worker',
        position: { x: 400, y: 50 },
        data: {
          label: 'Controller (Driver)',
          workerType: 'controller',
          status: simulationRunning ? 'busy' : 'idle',
        },
      },
    ];

    if (config === 'colocated') {
      return [
        ...baseNodes,
        {
          id: 'pool1',
          type: 'resourcePool',
          position: { x: 100, y: 200 },
          data: {
            label: 'Global Resource Pool',
            totalGPUs: 32,
            allocatedGPUs: 32,
            poolId: 'global_pool',
          },
        },
        {
          id: 'actor-rollout-ref',
          type: 'worker',
          position: { x: 100, y: 350 },
          data: {
            label: 'ActorRolloutRef',
            workerType: 'actor',
            gpuCount: 16,
            status: simulationRunning ? 'busy' : 'idle',
            metrics: simulationRunning ? {
              throughput: 2500,
              gpuUtilization: 0.85,
              memoryUsage: 0.7,
            } : undefined,
          },
        },
        {
          id: 'critic',
          type: 'worker',
          position: { x: 400, y: 350 },
          data: {
            label: 'Critic',
            workerType: 'critic',
            gpuCount: 8,
            status: simulationRunning ? 'busy' : 'idle',
            metrics: simulationRunning ? {
              throughput: 1800,
              gpuUtilization: 0.75,
              memoryUsage: 0.6,
            } : undefined,
          },
        },
        {
          id: 'reward',
          type: 'worker',
          position: { x: 700, y: 350 },
          data: {
            label: 'Reward Model',
            workerType: 'reward',
            gpuCount: 8,
            status: simulationRunning ? 'syncing' : 'idle',
            metrics: simulationRunning ? {
              throughput: 1200,
              gpuUtilization: 0.65,
              memoryUsage: 0.5,
            } : undefined,
          },
        },
      ];
    } else if (config === 'split') {
      return [
        ...baseNodes,
        {
          id: 'pool1',
          type: 'resourcePool',
          position: { x: 50, y: 200 },
          data: {
            label: 'Actor Pool',
            totalGPUs: 16,
            allocatedGPUs: 16,
            poolId: 'actor_pool',
          },
        },
        {
          id: 'pool2',
          type: 'resourcePool',
          position: { x: 350, y: 200 },
          data: {
            label: 'Critic Pool',
            totalGPUs: 8,
            allocatedGPUs: 8,
            poolId: 'critic_pool',
          },
        },
        {
          id: 'pool3',
          type: 'resourcePool',
          position: { x: 650, y: 200 },
          data: {
            label: 'Reward Pool',
            totalGPUs: 8,
            allocatedGPUs: 8,
            poolId: 'reward_pool',
          },
        },
        {
          id: 'actor',
          type: 'worker',
          position: { x: 50, y: 350 },
          data: {
            label: 'Actor',
            workerType: 'actor',
            gpuCount: 8,
            status: simulationRunning ? 'busy' : 'idle',
            metrics: simulationRunning ? {
              throughput: 1500,
              gpuUtilization: 0.9,
              memoryUsage: 0.75,
            } : undefined,
          },
        },
        {
          id: 'rollout',
          type: 'worker',
          position: { x: 50, y: 450 },
          data: {
            label: 'Rollout (vLLM)',
            workerType: 'rollout',
            gpuCount: 8,
            status: simulationRunning ? 'busy' : 'idle',
            metrics: simulationRunning ? {
              throughput: 3000,
              gpuUtilization: 0.95,
              memoryUsage: 0.8,
            } : undefined,
          },
        },
        {
          id: 'critic',
          type: 'worker',
          position: { x: 350, y: 350 },
          data: {
            label: 'Critic',
            workerType: 'critic',
            gpuCount: 8,
            status: simulationRunning ? 'busy' : 'idle',
            metrics: simulationRunning ? {
              throughput: 1800,
              gpuUtilization: 0.75,
              memoryUsage: 0.6,
            } : undefined,
          },
        },
        {
          id: 'reward',
          type: 'worker',
          position: { x: 650, y: 350 },
          data: {
            label: 'Reward Model',
            workerType: 'reward',
            gpuCount: 8,
            status: simulationRunning ? 'syncing' : 'idle',
            metrics: simulationRunning ? {
              throughput: 1200,
              gpuUtilization: 0.65,
              memoryUsage: 0.5,
            } : undefined,
          },
        },
      ];
    } else {
      // Hybrid configuration
      return [
        ...baseNodes,
        {
          id: 'pool1',
          type: 'resourcePool',
          position: { x: 150, y: 200 },
          data: {
            label: 'Actor-Ref Pool',
            totalGPUs: 16,
            allocatedGPUs: 16,
            poolId: 'actor_ref_pool',
          },
        },
        {
          id: 'pool2',
          type: 'resourcePool',
          position: { x: 550, y: 200 },
          data: {
            label: 'Critic-Reward Pool',
            totalGPUs: 16,
            allocatedGPUs: 16,
            poolId: 'critic_reward_pool',
          },
        },
        {
          id: 'actor-ref',
          type: 'worker',
          position: { x: 150, y: 350 },
          data: {
            label: 'Actor + Reference',
            workerType: 'actor',
            gpuCount: 8,
            status: simulationRunning ? 'busy' : 'idle',
            metrics: simulationRunning ? {
              throughput: 2000,
              gpuUtilization: 0.88,
              memoryUsage: 0.78,
            } : undefined,
          },
        },
        {
          id: 'rollout',
          type: 'worker',
          position: { x: 150, y: 450 },
          data: {
            label: 'Rollout (vLLM)',
            workerType: 'rollout',
            gpuCount: 8,
            status: simulationRunning ? 'busy' : 'idle',
            metrics: simulationRunning ? {
              throughput: 3000,
              gpuUtilization: 0.95,
              memoryUsage: 0.8,
            } : undefined,
          },
        },
        {
          id: 'critic',
          type: 'worker',
          position: { x: 450, y: 350 },
          data: {
            label: 'Critic',
            workerType: 'critic',
            gpuCount: 8,
            status: simulationRunning ? 'busy' : 'idle',
            metrics: simulationRunning ? {
              throughput: 1800,
              gpuUtilization: 0.75,
              memoryUsage: 0.6,
            } : undefined,
          },
        },
        {
          id: 'reward',
          type: 'worker',
          position: { x: 650, y: 350 },
          data: {
            label: 'Reward Model',
            workerType: 'reward',
            gpuCount: 8,
            status: simulationRunning ? 'syncing' : 'idle',
            metrics: simulationRunning ? {
              throughput: 1200,
              gpuUtilization: 0.65,
              memoryUsage: 0.5,
            } : undefined,
          },
        },
      ];
    }
  }, [simulationRunning]);

  // Define edges based on configuration
  const getEdgesForConfig = useCallback((config: string) => {
    if (!showDataFlow) return [];

    const baseEdges: Edge[] = [];

    if (config === 'colocated') {
      return [
        {
          id: 'controller-actor',
          source: 'controller',
          target: 'actor-rollout-ref',
          type: 'animated',
          animated: simulationRunning,
          style: { stroke: '#8b5cf6' },
        },
        {
          id: 'controller-critic',
          source: 'controller',
          target: 'critic',
          type: 'animated',
          animated: simulationRunning,
          style: { stroke: '#3b82f6' },
        },
        {
          id: 'controller-reward',
          source: 'controller',
          target: 'reward',
          type: 'animated',
          animated: simulationRunning,
          style: { stroke: '#f59e0b' },
        },
        {
          id: 'actor-controller',
          source: 'actor-rollout-ref',
          target: 'controller',
          type: 'animated',
          animated: simulationRunning,
          style: { stroke: '#8b5cf6' },
        },
      ];
    } else if (config === 'split') {
      return [
        {
          id: 'controller-actor',
          source: 'controller',
          target: 'actor',
          type: 'animated',
          animated: simulationRunning,
          style: { stroke: '#8b5cf6' },
        },
        {
          id: 'controller-rollout',
          source: 'controller',
          target: 'rollout',
          type: 'animated',
          animated: simulationRunning,
          style: { stroke: '#10b981' },
        },
        {
          id: 'controller-critic',
          source: 'controller',
          target: 'critic',
          type: 'animated',
          animated: simulationRunning,
          style: { stroke: '#3b82f6' },
        },
        {
          id: 'controller-reward',
          source: 'controller',
          target: 'reward',
          type: 'animated',
          animated: simulationRunning,
          style: { stroke: '#f59e0b' },
        },
        {
          id: 'rollout-actor',
          source: 'rollout',
          target: 'actor',
          type: 'animated',
          animated: simulationRunning,
          style: { stroke: '#22c55e', strokeDasharray: '5 5' },
        },
      ];
    } else {
      return [
        {
          id: 'controller-actor',
          source: 'controller',
          target: 'actor-ref',
          type: 'animated',
          animated: simulationRunning,
          style: { stroke: '#8b5cf6' },
        },
        {
          id: 'controller-rollout',
          source: 'controller',
          target: 'rollout',
          type: 'animated',
          animated: simulationRunning,
          style: { stroke: '#10b981' },
        },
        {
          id: 'controller-critic',
          source: 'controller',
          target: 'critic',
          type: 'animated',
          animated: simulationRunning,
          style: { stroke: '#3b82f6' },
        },
        {
          id: 'controller-reward',
          source: 'controller',
          target: 'reward',
          type: 'animated',
          animated: simulationRunning,
          style: { stroke: '#f59e0b' },
        },
        {
          id: 'actor-rollout',
          source: 'actor-ref',
          target: 'rollout',
          type: 'animated',
          animated: simulationRunning,
          style: { stroke: '#22c55e', strokeDasharray: '5 5' },
        },
      ];
    }
  }, [showDataFlow, simulationRunning]);

  const [nodes, setNodes, onNodesChange] = useNodesState(getNodesForConfig(selectedConfig));
  const [edges, setEdges, onEdgesChange] = useEdgesState(getEdgesForConfig(selectedConfig));

  // Update nodes and edges when configuration changes
  React.useEffect(() => {
    setNodes(getNodesForConfig(selectedConfig));
    setEdges(getEdgesForConfig(selectedConfig));
  }, [selectedConfig, getNodesForConfig, getEdgesForConfig, setNodes, setEdges]);

  return (
    <div className={`w-full h-full ${className}`}>
      <div className="absolute top-4 left-4 z-10 bg-white rounded-lg shadow-lg p-4">
        <h3 className="font-bold mb-3">VERL System Configuration</h3>
        
        <div className="space-y-3">
          <div>
            <label className="text-sm font-medium mb-1 block">Placement Strategy</label>
            <select
              value={selectedConfig}
              onChange={(e) => setSelectedConfig(e.target.value as any)}
              className="w-full px-3 py-2 border rounded-md"
            >
              <option value="colocated">Colocated (All on same GPUs)</option>
              <option value="split">Split (Different GPU pools)</option>
              <option value="hybrid">Hybrid (Actor+Ref colocated)</option>
            </select>
          </div>
          
          <div className="flex items-center space-x-2">
            <input
              type="checkbox"
              id="showDataFlow"
              checked={showDataFlow}
              onChange={(e) => setShowDataFlow(e.target.checked)}
              className="rounded"
            />
            <label htmlFor="showDataFlow" className="text-sm">Show Data Flow</label>
          </div>
          
          <button
            onClick={() => setSimulationRunning(!simulationRunning)}
            className={`w-full px-4 py-2 rounded-md font-medium transition-colors ${
              simulationRunning
                ? 'bg-red-500 hover:bg-red-600 text-white'
                : 'bg-green-500 hover:bg-green-600 text-white'
            }`}
          >
            {simulationRunning ? 'Stop Simulation' : 'Start Simulation'}
          </button>
        </div>
        
        {simulationRunning && (
          <div className="mt-4 pt-4 border-t">
            <h4 className="font-medium mb-2 text-sm">System Metrics</h4>
            <div className="space-y-1 text-xs">
              <div>Total Throughput: 8,500 tok/s</div>
              <div>Avg GPU Utilization: 82%</div>
              <div>Communication Overhead: 12%</div>
            </div>
          </div>
        )}
      </div>
      
      <ReactFlow
        nodes={nodes}
        edges={edges}
        onNodesChange={onNodesChange}
        onEdgesChange={onEdgesChange}
        nodeTypes={nodeTypes}
        edgeTypes={edgeTypes}
        fitView
        className="bg-gray-50"
      >
        <Background />
        <Controls />
        <MiniMap />
      </ReactFlow>
    </div>
  );
};