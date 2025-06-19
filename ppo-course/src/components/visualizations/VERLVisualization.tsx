'use client';

import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { 
  Cpu, HardDrive, Zap, Network, 
  ArrowRight, ArrowDown, Play, Pause,
  Monitor, Activity, BarChart3, TrendingUp
} from 'lucide-react';

interface VERLComponent {
  id: string;
  name: string;
  description: string;
  gpuCount: number;
  tensorParallel: number;
  pipelineParallel: number;
  position: { x: number; y: number };
  color: string;
  icon: React.ElementType;
}

interface DataFlow {
  from: string;
  to: string;
  label: string;
  type: 'gradient' | 'action' | 'reward' | 'state';
}

const verlComponents: VERLComponent[] = [
  {
    id: 'actor',
    name: 'Actor',
    description: 'Generates actions and updates policy parameters using PPO',
    gpuCount: 4,
    tensorParallel: 4,
    pipelineParallel: 1,
    position: { x: 100, y: 150 },
    color: 'bg-blue-500',
    icon: Cpu
  },
  {
    id: 'critic',
    name: 'Critic',
    description: 'Estimates value functions for advantage computation',
    gpuCount: 2,
    tensorParallel: 2,
    pipelineParallel: 1,
    position: { x: 100, y: 300 },
    color: 'bg-green-500',
    icon: BarChart3
  },
  {
    id: 'rollout',
    name: 'Rollout',
    description: 'Collects experience from environment interactions',
    gpuCount: 8,
    tensorParallel: 4,
    pipelineParallel: 2,
    position: { x: 400, y: 150 },
    color: 'bg-purple-500',
    icon: Activity
  },
  {
    id: 'reference',
    name: 'Reference Policy',
    description: 'Provides KL divergence baseline for policy constraints',
    gpuCount: 2,
    tensorParallel: 2,
    pipelineParallel: 1,
    position: { x: 400, y: 300 },
    color: 'bg-orange-500',
    icon: Monitor
  },
  {
    id: 'reward',
    name: 'Reward Model',
    description: 'Scores responses using learned human preferences',
    gpuCount: 4,
    tensorParallel: 4,
    pipelineParallel: 1,
    position: { x: 700, y: 225 },
    color: 'bg-red-500',
    icon: TrendingUp
  }
];

const dataFlows: DataFlow[] = [
  { from: 'actor', to: 'rollout', label: 'Actions', type: 'action' },
  { from: 'rollout', to: 'reward', label: 'Responses', type: 'state' },
  { from: 'reward', to: 'critic', label: 'Rewards', type: 'reward' },
  { from: 'critic', to: 'actor', label: 'Advantages', type: 'gradient' },
  { from: 'reference', to: 'actor', label: 'KL Penalty', type: 'gradient' },
  { from: 'rollout', to: 'reference', label: 'Log Probs', type: 'state' }
];

export const VERLVisualization: React.FC = () => {
  const [isAnimated, setIsAnimated] = useState(false);
  const [selectedComponent, setSelectedComponent] = useState<string | null>(null);
  const [hoveredComponent, setHoveredComponent] = useState<string | null>(null);
  const [currentStep, setCurrentStep] = useState(0);

  const workflowSteps = [
    'Rollout workers collect experience',
    'Reward model scores responses',
    'Critic computes value estimates',
    'Actor updates policy with PPO'
  ];

  useEffect(() => {
    if (isAnimated) {
      const interval = setInterval(() => {
        setCurrentStep(prev => (prev + 1) % workflowSteps.length);
      }, 2000);
      return () => clearInterval(interval);
    }
  }, [isAnimated, workflowSteps.length]);

  const getComponentById = (id: string) => verlComponents.find(c => c.id === id);

  const renderComponent = (component: VERLComponent) => {
    const Icon = component.icon;
    const isSelected = selectedComponent === component.id;
    const isHovered = hoveredComponent === component.id;

    return (
      <motion.div
        key={component.id}
        data-testid={`verl-${component.id}`}
        className={`absolute cursor-pointer transform -translate-x-1/2 -translate-y-1/2 ${
          isSelected ? 'z-20' : 'z-10'
        }`}
        style={{
          left: component.position.x,
          top: component.position.y
        }}
        whileHover={{ scale: 1.05 }}
        whileTap={{ scale: 0.95 }}
        onClick={() => setSelectedComponent(
          selectedComponent === component.id ? null : component.id
        )}
        onMouseEnter={() => setHoveredComponent(component.id)}
        onMouseLeave={() => setHoveredComponent(null)}
      >
        <div className={`${component.color} rounded-lg p-4 shadow-lg border-2 ${
          isSelected ? 'border-yellow-400' : 'border-transparent'
        } min-w-[120px] text-center`}>
          <Icon className="w-8 h-8 text-white mx-auto mb-2" />
          <h3 className="text-white font-bold text-sm">{component.name}</h3>
          <div className="text-white text-xs mt-1">
            <div>{component.gpuCount}x GPUs</div>
            <div>TP: {component.tensorParallel}x</div>
            {component.pipelineParallel > 1 && (
              <div>PP: {component.pipelineParallel}x</div>
            )}
          </div>
        </div>

        {(isHovered || isSelected) && (
          <motion.div
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            className="absolute top-full left-1/2 transform -translate-x-1/2 mt-2 bg-black text-white p-3 rounded-lg shadow-lg max-w-xs z-30"
          >
            <p className="text-sm">{component.description}</p>
            <div className="mt-2 text-xs">
              <div>Tensor Parallel: {component.tensorParallel}x</div>
              <div>Pipeline Parallel: {component.pipelineParallel}x</div>
              <div>Total GPUs: {component.gpuCount}</div>
            </div>
          </motion.div>
        )}
      </motion.div>
    );
  };

  const renderDataFlow = (flow: DataFlow, index: number) => {
    const fromComponent = getComponentById(flow.from);
    const toComponent = getComponentById(flow.to);
    
    if (!fromComponent || !toComponent) return null;

    const isActive = isAnimated && (
      (flow.type === 'action' && currentStep === 0) ||
      (flow.type === 'reward' && currentStep === 1) ||
      (flow.type === 'gradient' && (currentStep === 2 || currentStep === 3))
    );

    return (
      <g key={`${flow.from}-${flow.to}`} data-testid={`data-flow-${index}`}>
        <defs>
          <marker
            id={`arrowhead-${index}`}
            markerWidth="10"
            markerHeight="7"
            refX="9"
            refY="3.5"
            orient="auto"
          >
            <polygon
              points="0 0, 10 3.5, 0 7"
              fill={isActive ? '#EF4444' : '#64748B'}
            />
          </marker>
        </defs>
        
        <motion.line
          x1={fromComponent.position.x}
          y1={fromComponent.position.y}
          x2={toComponent.position.x}
          y2={toComponent.position.y}
          stroke={isActive ? '#EF4444' : '#64748B'}
          strokeWidth={isActive ? '3' : '2'}
          strokeDasharray={isActive ? '0' : '5,5'}
          markerEnd={`url(#arrowhead-${index})`}
          animate={{
            strokeDashoffset: isActive ? [0, -10] : 0
          }}
          transition={{
            duration: 1,
            repeat: isActive ? Infinity : 0,
            ease: 'linear'
          }}
        />
        
        {/* Label */}
        <text
          x={(fromComponent.position.x + toComponent.position.x) / 2}
          y={(fromComponent.position.y + toComponent.position.y) / 2 - 10}
          textAnchor="middle"
          className="text-xs fill-gray-600"
        >
          {flow.label}
        </text>
      </g>
    );
  };

  return (
    <div className="bg-white rounded-lg border shadow-lg p-6">
      {/* Header */}
      <div className="flex justify-between items-center mb-6">
        <div>
          <h2 className="text-2xl font-bold">VERL Distributed Architecture</h2>
          <p className="text-gray-600">Visualization of distributed RL training components</p>
        </div>
        
        <div className="flex items-center gap-4">
          <button
            onClick={() => setIsAnimated(!isAnimated)}
            className={`px-4 py-2 rounded-lg flex items-center gap-2 ${
              isAnimated ? 'bg-red-500 text-white' : 'bg-blue-500 text-white'
            }`}
          >
            {isAnimated ? <Pause size={16} /> : <Play size={16} />}
            {isAnimated ? 'Pause' : 'Animate'}
          </button>
          
          {isAnimated && (
            <div data-testid="animation-indicator" className="text-sm text-gray-600">
              Step {currentStep + 1}: {workflowSteps[currentStep]}
            </div>
          )}
        </div>
      </div>

      {/* Main Visualization */}
      <div className="relative bg-gray-50 rounded-lg p-8 min-h-[500px]">
        {/* SVG for data flows */}
        <svg 
          className="absolute inset-0 w-full h-full pointer-events-none" 
          style={{ zIndex: 5 }}
        >
          {dataFlows.map((flow, index) => renderDataFlow(flow, index))}
        </svg>

        {/* Components */}
        {verlComponents.map(renderComponent)}

        {/* Interaction highlight */}
        {selectedComponent && (
          <div data-testid="interaction-highlight" className="absolute inset-0 pointer-events-none">
            {dataFlows
              .filter(flow => flow.from === selectedComponent || flow.to === selectedComponent)
              .map((flow, index) => {
                const fromComp = getComponentById(flow.from);
                const toComp = getComponentById(flow.to);
                if (!fromComp || !toComp) return null;
                
                return (
                  <div
                    key={index}
                    className="absolute bg-yellow-300 opacity-30 rounded-full"
                    style={{
                      left: Math.min(fromComp.position.x, toComp.position.x) - 10,
                      top: Math.min(fromComp.position.y, toComp.position.y) - 10,
                      width: Math.abs(toComp.position.x - fromComp.position.x) + 20,
                      height: Math.abs(toComp.position.y - fromComp.position.y) + 20
                    }}
                  />
                );
              })}
          </div>
        )}
      </div>

      {/* Performance Metrics */}
      <div className="mt-6 grid grid-cols-1 md:grid-cols-3 gap-4">
        <div className="bg-blue-50 p-4 rounded-lg">
          <h3 className="font-semibold text-blue-800 mb-2">Throughput</h3>
          <div className="text-2xl font-bold text-blue-600">1,250</div>
          <div className="text-sm text-blue-600">tokens/sec</div>
        </div>
        
        <div className="bg-green-50 p-4 rounded-lg">
          <h3 className="font-semibold text-green-800 mb-2">Memory Usage</h3>
          <div className="text-2xl font-bold text-green-600">72%</div>
          <div className="text-sm text-green-600">GPU memory utilized</div>
        </div>
        
        <div className="bg-purple-50 p-4 rounded-lg">
          <h3 className="font-semibold text-purple-800 mb-2">GPU Allocation</h3>
          <div className="text-sm text-purple-600 space-y-1">
            <div>Actor: 4x GPUs</div>
            <div>Rollout: 8x GPUs</div>
            <div>Reward: 4x GPUs</div>
          </div>
        </div>
      </div>

      {/* Resource Utilization */}
      <div className="mt-4 bg-gray-50 p-4 rounded-lg">
        <h3 className="font-semibold mb-3">Resource Utilization</h3>
        <div className="grid grid-cols-3 gap-4 text-sm">
          <div>
            <div className="flex justify-between">
              <span>CPU: 40%</span>
              <Cpu size={16} />
            </div>
            <div className="w-full bg-gray-200 rounded-full h-2 mt-1">
              <div className="bg-blue-600 h-2 rounded-full" style={{ width: '40%' }} />
            </div>
          </div>
          
          <div>
            <div className="flex justify-between">
              <span>GPU: 85%</span>
              <Zap size={16} />
            </div>
            <div className="w-full bg-gray-200 rounded-full h-2 mt-1">
              <div className="bg-green-600 h-2 rounded-full" style={{ width: '85%' }} />
            </div>
          </div>
          
          <div>
            <div className="flex justify-between">
              <span>Memory: 72%</span>
              <HardDrive size={16} />
            </div>
            <div className="w-full bg-gray-200 rounded-full h-2 mt-1">
              <div className="bg-purple-600 h-2 rounded-full" style={{ width: '72%' }} />
            </div>
          </div>
        </div>
      </div>

      {/* Communication Patterns */}
      <div className="mt-4 grid grid-cols-1 md:grid-cols-2 gap-4">
        <div data-testid="communication-actor-critic" className="bg-white border rounded-lg p-4">
          <h4 className="font-semibold mb-2">Actor ↔ Critic Communication</h4>
          <div className="text-sm text-gray-600 space-y-1">
            <div>• Advantage estimates</div>
            <div>• Value function gradients</div>
            <div>• State representations</div>
          </div>
        </div>
        
        <div data-testid="communication-rollout-reward" className="bg-white border rounded-lg p-4">
          <h4 className="font-semibold mb-2">Rollout → Reward Communication</h4>
          <div className="text-sm text-gray-600 space-y-1">
            <div>• Generated responses</div>
            <div>• Context prompts</div>
            <div>• Attention masks</div>
          </div>
        </div>
      </div>

      {/* Workflow Steps */}
      <div className="mt-4">
        <h3 className="font-semibold mb-3">Training Workflow</h3>
        <div className="flex flex-wrap gap-2">
          {workflowSteps.map((step, index) => (
            <div
              key={index}
              data-testid={`workflow-step-${index}`}
              className={`px-3 py-2 rounded-lg text-sm border ${
                isAnimated && currentStep === index
                  ? 'bg-blue-500 text-white border-blue-500'
                  : 'bg-gray-100 text-gray-700 border-gray-200'
              }`}
            >
              {index + 1}. {step}
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};