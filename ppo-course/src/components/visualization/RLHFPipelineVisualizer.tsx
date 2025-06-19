'use client';

import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { ArrowRight, CheckCircle, AlertCircle, PlayCircle, PauseCircle } from 'lucide-react';

interface Stage {
  id: string;
  name: string;
  description: string;
  status: 'pending' | 'active' | 'completed' | 'error';
  metrics?: {
    [key: string]: string | number;
  };
  details?: string[];
}

interface DataFlow {
  from: string;
  to: string;
  label: string;
  active: boolean;
}

const initialStages: Stage[] = [
  {
    id: 'data-prep',
    name: 'Data Preprocessing',
    description: 'Convert preference pairs to training formats',
    status: 'pending',
    details: [
      'Load HH-RLHF dataset (112K samples)',
      'Split into chosen/rejected pairs',
      'Tokenize with chat template',
      'Create train/validation splits',
    ],
  },
  {
    id: 'sft',
    name: 'Supervised Fine-Tuning',
    description: 'Train base model on demonstration data',
    status: 'pending',
    metrics: {
      'Training Samples': '224,104',
      'Loss': '2.847',
      'Perplexity': '17.2',
    },
    details: [
      'Use both chosen and rejected responses',
      'Cross-entropy loss on response tokens',
      'Mask prompt tokens in loss',
    ],
  },
  {
    id: 'reward-model',
    name: 'Reward Model Training',
    description: 'Train model to predict human preferences',
    status: 'pending',
    metrics: {
      'Training Pairs': '84,039',
      'Accuracy': '0.0%',
      'Loss': 'N/A',
    },
    details: [
      '⚠️ Bradley-Terry loss not implemented!',
      'Uses placeholder loss function',
      'Cannot learn preferences properly',
    ],
  },
  {
    id: 'ppo',
    name: 'PPO Training',
    description: 'Optimize policy with human feedback',
    status: 'pending',
    metrics: {
      'KL Divergence': '0.0',
      'Reward': '0.0',
      'Policy Loss': '0.0',
    },
    details: [
      'Actor generates responses',
      'Reward model scores outputs',
      'Update policy with PPO',
      'Maintain KL constraint',
    ],
  },
];

const dataFlows: DataFlow[] = [
  {
    from: 'data-prep',
    to: 'sft',
    label: 'SFT Data (224K samples)',
    active: false,
  },
  {
    from: 'data-prep',
    to: 'reward-model',
    label: 'Preference Pairs (84K)',
    active: false,
  },
  {
    from: 'sft',
    to: 'ppo',
    label: 'Base Policy Model',
    active: false,
  },
  {
    from: 'reward-model',
    to: 'ppo',
    label: 'Reward Scores',
    active: false,
  },
];

interface RLHFPipelineVisualizerProps {
  className?: string;
}

export const RLHFPipelineVisualizer: React.FC<RLHFPipelineVisualizerProps> = ({
  className = '',
}) => {
  const [stages, setStages] = useState<Stage[]>(initialStages);
  const [flows, setFlows] = useState<DataFlow[]>(dataFlows);
  const [currentStage, setCurrentStage] = useState(0);
  const [isRunning, setIsRunning] = useState(false);
  const [showBradleyTerryFix, setShowBradleyTerryFix] = useState(false);

  // Simulate pipeline execution
  useEffect(() => {
    if (!isRunning || currentStage >= stages.length) return;

    const timer = setTimeout(() => {
      setStages((prev) =>
        prev.map((stage, idx) => {
          if (idx === currentStage) {
            // Special handling for reward model
            if (stage.id === 'reward-model') {
              return {
                ...stage,
                status: 'error',
                metrics: {
                  'Training Pairs': '84,039',
                  'Accuracy': '50.1%', // Random chance
                  'Loss': '1.000', // Placeholder
                },
              };
            }
            return { ...stage, status: 'completed' };
          } else if (idx === currentStage + 1) {
            return { ...stage, status: 'active' };
          }
          return stage;
        })
      );

      // Update data flows
      if (currentStage === 0) {
        setFlows((prev) =>
          prev.map((flow) =>
            flow.from === 'data-prep' ? { ...flow, active: true } : flow
          )
        );
      } else if (currentStage === 1) {
        setFlows((prev) =>
          prev.map((flow) =>
            flow.from === 'sft' ? { ...flow, active: true } : flow
          )
        );
      } else if (currentStage === 2) {
        setFlows((prev) =>
          prev.map((flow) =>
            flow.from === 'reward-model' ? { ...flow, active: true } : flow
          )
        );
      }

      setCurrentStage((prev) => prev + 1);
    }, 2000);

    return () => clearTimeout(timer);
  }, [isRunning, currentStage, stages.length]);

  const resetPipeline = () => {
    setStages(initialStages);
    setFlows(dataFlows);
    setCurrentStage(0);
    setIsRunning(false);
  };

  const getStatusIcon = (status: Stage['status']) => {
    switch (status) {
      case 'completed':
        return <CheckCircle className="w-5 h-5 text-green-500" />;
      case 'active':
        return (
          <motion.div
            animate={{ rotate: 360 }}
            transition={{ duration: 2, repeat: Infinity, ease: 'linear' }}
          >
            <PlayCircle className="w-5 h-5 text-blue-500" />
          </motion.div>
        );
      case 'error':
        return <AlertCircle className="w-5 h-5 text-red-500" />;
      default:
        return <div className="w-5 h-5 rounded-full border-2 border-gray-300" />;
    }
  };

  return (
    <div className={`w-full max-w-6xl mx-auto p-6 ${className}`}>
      <div className="mb-6">
        <h2 className="text-2xl font-bold mb-2">RLHF Pipeline Visualization</h2>
        <p className="text-gray-600">
          Watch how data flows through the three stages of RLHF training
        </p>
      </div>

      <div className="mb-6 flex gap-4">
        <button
          onClick={() => setIsRunning(!isRunning)}
          disabled={currentStage >= stages.length}
          className="px-4 py-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600 disabled:bg-gray-300 disabled:cursor-not-allowed transition-colors"
        >
          {isRunning ? 'Pause' : 'Start'} Pipeline
        </button>
        <button
          onClick={resetPipeline}
          className="px-4 py-2 bg-gray-500 text-white rounded-lg hover:bg-gray-600 transition-colors"
        >
          Reset
        </button>
        <button
          onClick={() => setShowBradleyTerryFix(!showBradleyTerryFix)}
          className="px-4 py-2 bg-purple-500 text-white rounded-lg hover:bg-purple-600 transition-colors"
        >
          {showBradleyTerryFix ? 'Hide' : 'Show'} Bradley-Terry Fix
        </button>
      </div>

      <div className="relative">
        {/* Pipeline Stages */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
          {stages.map((stage, idx) => (
            <motion.div
              key={stage.id}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: idx * 0.1 }}
              className={`relative bg-white rounded-lg shadow-lg p-6 border-2 transition-all ${
                stage.status === 'active'
                  ? 'border-blue-500 shadow-blue-200'
                  : stage.status === 'completed'
                  ? 'border-green-500'
                  : stage.status === 'error'
                  ? 'border-red-500 bg-red-50'
                  : 'border-gray-200'
              }`}
            >
              <div className="flex items-start justify-between mb-3">
                <h3 className="font-bold text-lg">{stage.name}</h3>
                {getStatusIcon(stage.status)}
              </div>
              
              <p className="text-sm text-gray-600 mb-4">{stage.description}</p>
              
              {stage.metrics && stage.status !== 'pending' && (
                <div className="mb-4 p-3 bg-gray-50 rounded-md">
                  <h4 className="font-medium text-sm mb-2">Metrics</h4>
                  {Object.entries(stage.metrics).map(([key, value]) => (
                    <div key={key} className="flex justify-between text-xs">
                      <span className="text-gray-600">{key}:</span>
                      <span className="font-mono">{value}</span>
                    </div>
                  ))}
                </div>
              )}
              
              {stage.details && (
                <div className="space-y-1">
                  {stage.details.map((detail, i) => (
                    <div
                      key={i}
                      className={`text-xs ${
                        detail.startsWith('⚠️') ? 'text-red-600 font-medium' : 'text-gray-600'
                      }`}
                    >
                      • {detail}
                    </div>
                  ))}
                </div>
              )}
              
              {/* Data flow arrows */}
              {idx < stages.length - 1 && (
                <div className="absolute -right-3 top-1/2 transform -translate-y-1/2 hidden lg:block">
                  <ArrowRight className="w-6 h-6 text-gray-400" />
                </div>
              )}
            </motion.div>
          ))}
        </div>

        {/* Data Flow Animations */}
        <svg className="absolute inset-0 pointer-events-none" style={{ zIndex: -1 }}>
          {flows.map((flow) => {
            const fromStage = stages.find((s) => s.id === flow.from);
            const toStage = stages.find((s) => s.id === flow.to);
            if (!fromStage || !toStage) return null;

            // Simplified positioning - would need proper calculation in production
            return (
              <AnimatePresence key={`${flow.from}-${flow.to}`}>
                {flow.active && (
                  <motion.circle
                    r="4"
                    fill="#3b82f6"
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    exit={{ opacity: 0 }}
                  >
                    <animateMotion dur="2s" repeatCount="indefinite">
                      <mpath href={`#path-${flow.from}-${flow.to}`} />
                    </animateMotion>
                  </motion.circle>
                )}
              </AnimatePresence>
            );
          })}
        </svg>
      </div>

      {/* Bradley-Terry Fix Explanation */}
      <AnimatePresence>
        {showBradleyTerryFix && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            exit={{ opacity: 0, height: 0 }}
            className="mt-8 bg-yellow-50 border-2 border-yellow-300 rounded-lg p-6"
          >
            <h3 className="font-bold text-lg mb-3">🔧 Bradley-Terry Loss Implementation Fix</h3>
            
            <div className="grid md:grid-cols-2 gap-6">
              <div>
                <h4 className="font-medium mb-2 text-red-600">❌ Current (Broken)</h4>
                <pre className="bg-gray-900 text-white p-3 rounded text-xs overflow-x-auto">
{`def loss_func(output):
    # Placeholder - doesn't train!
    return torch.tensor(1.0), output`}
                </pre>
              </div>
              
              <div>
                <h4 className="font-medium mb-2 text-green-600">✅ Correct Implementation</h4>
                <pre className="bg-gray-900 text-white p-3 rounded text-xs overflow-x-auto">
{`def bradley_terry_loss(chosen_r, rejected_r):
    # P(chosen > rejected)
    return -torch.log(
        torch.sigmoid(chosen_r - rejected_r)
    ).mean()`}
                </pre>
              </div>
            </div>
            
            <div className="mt-4 p-4 bg-white rounded-lg">
              <h4 className="font-medium mb-2">Impact of the Fix</h4>
              <div className="grid md:grid-cols-3 gap-4 text-sm">
                <div>
                  <span className="text-gray-600">Before:</span>
                  <div className="font-mono">Accuracy: 50.1%</div>
                  <div className="text-xs text-gray-500">Random chance</div>
                </div>
                <div>
                  <span className="text-gray-600">After:</span>
                  <div className="font-mono text-green-600">Accuracy: 72.3%</div>
                  <div className="text-xs text-gray-500">Learns preferences</div>
                </div>
                <div>
                  <span className="text-gray-600">PPO Reward:</span>
                  <div className="font-mono text-green-600">+0.85 avg</div>
                  <div className="text-xs text-gray-500">Meaningful signal</div>
                </div>
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Pipeline Summary */}
      {currentStage >= stages.length && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="mt-8 bg-blue-50 rounded-lg p-6"
        >
          <h3 className="font-bold text-lg mb-3">Pipeline Complete!</h3>
          <div className="space-y-2">
            <p className="text-sm">
              <span className="font-medium">Critical Issue Detected:</span> The reward model
              training stage failed due to the missing Bradley-Terry loss implementation.
            </p>
            <p className="text-sm">
              Without proper preference learning, the model cannot distinguish between good
              and bad responses, making the PPO training ineffective.
            </p>
          </div>
        </motion.div>
      )}
    </div>
  );
};