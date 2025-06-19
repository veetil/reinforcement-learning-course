'use client';

import React from 'react';
import { motion } from 'framer-motion';
import { Play, Pause, RotateCcw } from 'lucide-react';
import { VERLVisualization, NeuralNetworkDesigner, DistributedTrainingVisualizer } from '../visualizations';

interface InteractiveDemoProps {
  demoType: string;
  className?: string;
}

export const InteractiveDemo: React.FC<InteractiveDemoProps> = ({ demoType, className = '' }) => {
  const [isPlaying, setIsPlaying] = React.useState(false);

  const renderDemo = () => {
    switch (demoType) {
      case 'neural-network':
        return <NeuralNetworkDemo isPlaying={isPlaying} />;
      case 'backpropagation':
        return <BackpropagationDemo isPlaying={isPlaying} />;
      case 'gradient-descent':
        return <GradientDescentDemo isPlaying={isPlaying} />;
      case 'mdp-visualization':
        return <MDPVisualizationDemo isPlaying={isPlaying} />;
      case 'verl-architecture':
        return <VERLVisualization />;
      case 'neural-network-designer':
        return <NeuralNetworkDesigner />;
      case 'distributed-training':
        return <DistributedTrainingVisualizer />;
      default:
        return <div>Demo type not found: {demoType}</div>;
    }
  };

  // For complex visualizations, don't show the outer container since they have their own controls
  if (demoType === 'verl-architecture' || demoType === 'neural-network-designer' || demoType === 'distributed-training') {
    return (
      <div className={className}>
        {renderDemo()}
      </div>
    );
  }

  return (
    <div className={`bg-gray-50 rounded-lg p-6 ${className}`}>
      <div className="flex justify-between items-center mb-4">
        <h3 className="text-lg font-semibold">Interactive Demo</h3>
        <div className="flex gap-2">
          <button
            onClick={() => setIsPlaying(!isPlaying)}
            className="px-3 py-1 bg-blue-500 text-white rounded hover:bg-blue-600 flex items-center gap-1"
          >
            {isPlaying ? <Pause size={16} /> : <Play size={16} />}
            {isPlaying ? 'Pause' : 'Play'}
          </button>
          <button
            onClick={() => {
              setIsPlaying(false);
              // Reset logic would go here
            }}
            className="px-3 py-1 bg-gray-500 text-white rounded hover:bg-gray-600 flex items-center gap-1"
          >
            <RotateCcw size={16} />
            Reset
          </button>
        </div>
      </div>
      
      <div className="bg-white rounded border p-4 min-h-[300px] flex items-center justify-center">
        {renderDemo()}
      </div>
    </div>
  );
};

// Simple placeholder demos
const NeuralNetworkDemo: React.FC<{ isPlaying: boolean }> = ({ isPlaying }) => {
  return (
    <div className="text-center">
      <svg width="400" height="250" viewBox="0 0 400 250">
        {/* Input layer */}
        <g>
          <circle cx="50" cy="50" r="20" fill="#3B82F6" />
          <circle cx="50" cy="125" r="20" fill="#3B82F6" />
          <circle cx="50" cy="200" r="20" fill="#3B82F6" />
          <text x="10" y="55" fill="white" fontSize="12">x₁</text>
          <text x="10" y="130" fill="white" fontSize="12">x₂</text>
          <text x="10" y="205" fill="white" fontSize="12">x₃</text>
        </g>
        
        {/* Hidden layer */}
        <g>
          <circle cx="200" cy="75" r="20" fill="#10B981" />
          <circle cx="200" cy="175" r="20" fill="#10B981" />
          <text x="190" y="80" fill="white" fontSize="12">h₁</text>
          <text x="190" y="180" fill="white" fontSize="12">h₂</text>
        </g>
        
        {/* Output layer */}
        <g>
          <circle cx="350" cy="125" r="20" fill="#F59E0B" />
          <text x="345" y="130" fill="white" fontSize="12">y</text>
        </g>
        
        {/* Connections */}
        <g stroke="#CBD5E1" strokeWidth="2">
          <line x1="70" y1="50" x2="180" y2="75" />
          <line x1="70" y1="50" x2="180" y2="175" />
          <line x1="70" y1="125" x2="180" y2="75" />
          <line x1="70" y1="125" x2="180" y2="175" />
          <line x1="70" y1="200" x2="180" y2="75" />
          <line x1="70" y1="200" x2="180" y2="175" />
          
          <line x1="220" y1="75" x2="330" y2="125" />
          <line x1="220" y1="175" x2="330" y2="125" />
        </g>
        
        {isPlaying && (
          <motion.circle
            cx="50"
            cy="50"
            r="5"
            fill="#EF4444"
            animate={{
              cx: [50, 200, 350],
              cy: [50, 75, 125],
            }}
            transition={{
              duration: 2,
              repeat: Infinity,
              ease: "linear"
            }}
          />
        )}
      </svg>
      
      <p className="mt-4 text-sm text-gray-600">
        {isPlaying ? "Signal propagating through the network..." : "Click Play to see forward propagation"}
      </p>
    </div>
  );
};

const BackpropagationDemo: React.FC<{ isPlaying: boolean }> = ({ isPlaying }) => {
  return (
    <div className="text-center">
      <p className="text-gray-600">Backpropagation visualization</p>
      {isPlaying && <p className="text-blue-600 mt-2">Gradients flowing backward...</p>}
    </div>
  );
};

const GradientDescentDemo: React.FC<{ isPlaying: boolean }> = ({ isPlaying }) => {
  return (
    <div className="text-center">
      <p className="text-gray-600">Gradient descent optimization</p>
      {isPlaying && <p className="text-green-600 mt-2">Descending towards minimum...</p>}
    </div>
  );
};

const MDPVisualizationDemo: React.FC<{ isPlaying: boolean }> = ({ isPlaying }) => {
  const [currentState, setCurrentState] = React.useState(0);
  const [action, setAction] = React.useState<number | null>(null);
  
  React.useEffect(() => {
    if (isPlaying) {
      const interval = setInterval(() => {
        // Simulate MDP transitions
        const newAction = Math.floor(Math.random() * 2);
        setAction(newAction);
        
        setTimeout(() => {
          // Transition to new state based on action
          setCurrentState(prev => {
            if (newAction === 0) return Math.max(0, prev - 1);
            else return Math.min(3, prev + 1);
          });
          setAction(null);
        }, 500);
      }, 2000);
      
      return () => clearInterval(interval);
    }
  }, [isPlaying]);
  
  const states = ['S0', 'S1', 'S2', 'S3'];
  const rewards = [-1, 0, 0, 10];
  
  return (
    <div className="text-center">
      <div className="flex justify-center items-center gap-4 mb-4">
        {states.map((state, idx) => (
          <div key={idx} className="relative">
            <div className={`w-20 h-20 rounded-full flex items-center justify-center font-bold text-white transition-all ${
              currentState === idx ? 'bg-blue-600 scale-110' : 'bg-gray-400'
            }`}>
              {state}
            </div>
            <div className="text-xs mt-1">r={rewards[idx]}</div>
          </div>
        ))}
      </div>
      
      {action !== null && (
        <motion.div
          initial={{ opacity: 0, y: -10 }}
          animate={{ opacity: 1, y: 0 }}
          className="text-lg font-semibold text-purple-600"
        >
          Action: {action === 0 ? '← Left' : 'Right →'}
        </motion.div>
      )}
      
      <div className="mt-4 text-sm text-gray-600">
        {isPlaying ? (
          <>
            <p>Agent at state {states[currentState]}</p>
            <p>Reward: {rewards[currentState]}</p>
          </>
        ) : (
          <p>Click Play to see MDP transitions in action</p>
        )}
      </div>
    </div>
  );
};