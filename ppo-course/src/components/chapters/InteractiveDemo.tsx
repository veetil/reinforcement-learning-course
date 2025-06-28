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
      case 'gae-visualization':
        return <GAEVisualizationDemo isPlaying={isPlaying} />;
      case 'value-function-grid':
        return <ValueFunctionGridDemo isPlaying={isPlaying} />;
      case 'policy-gradient-intuition':
        return <PolicyGradientIntuitionDemo isPlaying={isPlaying} />;
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

// Enhanced Neural Network Demo with formulas and multiple paths
const NeuralNetworkDemo: React.FC<{ isPlaying: boolean }> = ({ isPlaying }) => {
  const [activeNeuron, setActiveNeuron] = React.useState<string | null>(null);
  const [step, setStep] = React.useState(0);
  
  React.useEffect(() => {
    if (isPlaying) {
      const interval = setInterval(() => {
        setStep((prev) => (prev + 1) % 4);
      }, 1500);
      return () => clearInterval(interval);
    } else {
      setStep(0);
      setActiveNeuron(null);
    }
  }, [isPlaying]);
  
  React.useEffect(() => {
    // Update active neuron based on step
    switch(step) {
      case 0: setActiveNeuron(null); break;
      case 1: setActiveNeuron('input'); break;
      case 2: setActiveNeuron('hidden'); break;
      case 3: setActiveNeuron('output'); break;
    }
  }, [step]);
  
  return (
    <div className="space-y-4">
      {/* Mathematical formulas */}
      <div className="bg-gray-100 rounded-lg p-4 text-left">
        <h4 className="font-semibold mb-2">Neural Network Equations:</h4>
        <div className="space-y-2 font-mono text-sm">
          <div className="flex items-center gap-2">
            <span className="text-blue-600">Input Layer:</span>
            <span>x = [x₁, x₂, x₃]</span>
          </div>
          <div className="flex items-center gap-2">
            <span className="text-green-600">Hidden Layer:</span>
            <span>h = σ(W₁x + b₁)</span>
            <span className="text-gray-500 text-xs ml-2">where σ is activation function</span>
          </div>
          <div className="flex items-center gap-2">
            <span className="text-orange-600">Output Layer:</span>
            <span>y = W₂h + b₂</span>
          </div>
        </div>
      </div>
      
      {/* Neural network visualization */}
      <div className="bg-white rounded-lg p-4 border">
        <svg width="500" height="300" viewBox="0 0 500 300" className="mx-auto">
          {/* Weight labels */}
          <g className="text-xs text-gray-500">
            {/* W1 weights */}
            <text x="110" y="50" textAnchor="middle">w₁₁</text>
            <text x="110" y="95" textAnchor="middle">w₁₂</text>
            <text x="110" y="140" textAnchor="middle">w₂₁</text>
            <text x="110" y="185" textAnchor="middle">w₂₂</text>
            <text x="110" y="230" textAnchor="middle">w₃₁</text>
            <text x="110" y="250" textAnchor="middle">w₃₂</text>
            
            {/* W2 weights */}
            <text x="285" y="110" textAnchor="middle">v₁</text>
            <text x="285" y="190" textAnchor="middle">v₂</text>
          </g>
          
          {/* Input layer */}
          <g>
            <circle cx="50" cy="60" r="25" fill="#3B82F6" 
              opacity={activeNeuron === 'input' ? 1 : 0.7}
              stroke={activeNeuron === 'input' ? '#1D4ED8' : 'none'}
              strokeWidth="3"
            />
            <circle cx="50" cy="150" r="25" fill="#3B82F6" 
              opacity={activeNeuron === 'input' ? 1 : 0.7}
              stroke={activeNeuron === 'input' ? '#1D4ED8' : 'none'}
              strokeWidth="3"
            />
            <circle cx="50" cy="240" r="25" fill="#3B82F6" 
              opacity={activeNeuron === 'input' ? 1 : 0.7}
              stroke={activeNeuron === 'input' ? '#1D4ED8' : 'none'}
              strokeWidth="3"
            />
            <text x="50" y="65" fill="white" fontSize="14" textAnchor="middle">x₁</text>
            <text x="50" y="155" fill="white" fontSize="14" textAnchor="middle">x₂</text>
            <text x="50" y="245" fill="white" fontSize="14" textAnchor="middle">x₃</text>
          </g>
          
          {/* Hidden layer */}
          <g>
            <circle cx="250" cy="100" r="25" fill="#10B981" 
              opacity={activeNeuron === 'hidden' ? 1 : 0.7}
              stroke={activeNeuron === 'hidden' ? '#059669' : 'none'}
              strokeWidth="3"
            />
            <circle cx="250" cy="200" r="25" fill="#10B981" 
              opacity={activeNeuron === 'hidden' ? 1 : 0.7}
              stroke={activeNeuron === 'hidden' ? '#059669' : 'none'}
              strokeWidth="3"
            />
            <text x="250" y="105" fill="white" fontSize="14" textAnchor="middle">h₁</text>
            <text x="250" y="205" fill="white" fontSize="14" textAnchor="middle">h₂</text>
          </g>
          
          {/* Output layer */}
          <g>
            <circle cx="450" cy="150" r="25" fill="#F59E0B" 
              opacity={activeNeuron === 'output' ? 1 : 0.7}
              stroke={activeNeuron === 'output' ? '#D97706' : 'none'}
              strokeWidth="3"
            />
            <text x="450" y="155" fill="white" fontSize="14" textAnchor="middle">y</text>
          </g>
          
          {/* Connections */}
          <g stroke="#CBD5E1" strokeWidth="2" opacity="0.5">
            {/* Input to hidden */}
            <line x1="75" y1="60" x2="225" y2="100" />
            <line x1="75" y1="60" x2="225" y2="200" />
            <line x1="75" y1="150" x2="225" y2="100" />
            <line x1="75" y1="150" x2="225" y2="200" />
            <line x1="75" y1="240" x2="225" y2="100" />
            <line x1="75" y1="240" x2="225" y2="200" />
            
            {/* Hidden to output */}
            <line x1="275" y1="100" x2="425" y2="150" />
            <line x1="275" y1="200" x2="425" y2="150" />
          </g>
          
          {/* Animated signals for all paths */}
          {isPlaying && step >= 1 && (
            <>
              {/* Signals from all inputs */}
              <motion.circle
                cx="50"
                cy="60"
                r="4"
                fill="#EF4444"
                initial={{ cx: 50, cy: 60 }}
                animate={step >= 2 ? { cx: 250, cy: 100 } : {}}
                transition={{ duration: 1, ease: "easeInOut" }}
              />
              <motion.circle
                cx="50"
                cy="60"
                r="4"
                fill="#EF4444"
                initial={{ cx: 50, cy: 60 }}
                animate={step >= 2 ? { cx: 250, cy: 200 } : {}}
                transition={{ duration: 1, ease: "easeInOut" }}
              />
              <motion.circle
                cx="50"
                cy="150"
                r="4"
                fill="#EF4444"
                initial={{ cx: 50, cy: 150 }}
                animate={step >= 2 ? { cx: 250, cy: 100 } : {}}
                transition={{ duration: 1, ease: "easeInOut" }}
              />
              <motion.circle
                cx="50"
                cy="150"
                r="4"
                fill="#EF4444"
                initial={{ cx: 50, cy: 150 }}
                animate={step >= 2 ? { cx: 250, cy: 200 } : {}}
                transition={{ duration: 1, ease: "easeInOut" }}
              />
              <motion.circle
                cx="50"
                cy="240"
                r="4"
                fill="#EF4444"
                initial={{ cx: 50, cy: 240 }}
                animate={step >= 2 ? { cx: 250, cy: 100 } : {}}
                transition={{ duration: 1, ease: "easeInOut" }}
              />
              <motion.circle
                cx="50"
                cy="240"
                r="4"
                fill="#EF4444"
                initial={{ cx: 50, cy: 240 }}
                animate={step >= 2 ? { cx: 250, cy: 200 } : {}}
                transition={{ duration: 1, ease: "easeInOut" }}
              />
              
              {/* Signals from hidden to output */}
              {step >= 3 && (
                <>
                  <motion.circle
                    cx="250"
                    cy="100"
                    r="4"
                    fill="#10B981"
                    initial={{ cx: 250, cy: 100 }}
                    animate={{ cx: 450, cy: 150 }}
                    transition={{ duration: 1, ease: "easeInOut" }}
                  />
                  <motion.circle
                    cx="250"
                    cy="200"
                    r="4"
                    fill="#10B981"
                    initial={{ cx: 250, cy: 200 }}
                    animate={{ cx: 450, cy: 150 }}
                    transition={{ duration: 1, ease: "easeInOut" }}
                  />
                </>
              )}
            </>
          )}
        </svg>
      </div>
      
      {/* Step explanation */}
      <div className="text-center space-y-2">
        <p className="text-sm font-semibold text-gray-700">
          {step === 0 && "Ready to start forward propagation"}
          {step === 1 && "Step 1: Input values x₁, x₂, x₃ are received"}
          {step === 2 && "Step 2: Computing hidden layer: h = σ(W₁x + b₁)"}
          {step === 3 && "Step 3: Computing output: y = W₂h + b₂"}
        </p>
        <p className="text-xs text-gray-500">
          {isPlaying ? "Watch how signals flow through all connections" : "Click Play to see forward propagation through all paths"}
        </p>
      </div>
    </div>
  );
};

const BackpropagationDemo: React.FC<{ isPlaying: boolean }> = ({ isPlaying }) => {
  const [step, setStep] = React.useState(0);
  
  React.useEffect(() => {
    if (isPlaying) {
      const interval = setInterval(() => {
        setStep((prev) => (prev + 1) % 5);
      }, 2000);
      return () => clearInterval(interval);
    } else {
      setStep(0);
    }
  }, [isPlaying]);
  
  const getLossGradientPath = () => {
    switch(step) {
      case 1: return "M450,150 L450,100"; // Loss gradient
      case 2: return "M425,150 L275,100 M425,150 L275,200"; // Output to hidden
      case 3: return "M225,100 L75,60 M225,100 L75,150 M225,100 L75,240 M225,200 L75,60 M225,200 L75,150 M225,200 L75,240"; // Hidden to input
      case 4: return "M50,60 L50,30 M50,150 L50,120 M50,240 L50,210"; // Input gradients
      default: return "";
    }
  };
  
  return (
    <div className="space-y-4">
      {/* Backpropagation equations */}
      <div className="bg-gray-100 rounded-lg p-4 text-left">
        <h4 className="font-semibold mb-2">Backpropagation Steps:</h4>
        <div className="space-y-2 font-mono text-sm">
          <div className={`flex items-center gap-2 ${step === 1 ? 'text-red-600 font-bold' : ''}`}>
            <span>1. Compute loss gradient:</span>
            <span>∂L/∂y</span>
          </div>
          <div className={`flex items-center gap-2 ${step === 2 ? 'text-orange-600 font-bold' : ''}`}>
            <span>2. Hidden layer gradients:</span>
            <span>∂L/∂h = W₂ᵀ × ∂L/∂y</span>
          </div>
          <div className={`flex items-center gap-2 ${step === 3 ? 'text-green-600 font-bold' : ''}`}>
            <span>3. Weight gradients:</span>
            <span>∂L/∂W₁ = ∂L/∂h × xᵀ</span>
          </div>
          <div className={`flex items-center gap-2 ${step === 4 ? 'text-blue-600 font-bold' : ''}`}>
            <span>4. Input gradients:</span>
            <span>∂L/∂x = W₁ᵀ × ∂L/∂h</span>
          </div>
        </div>
      </div>
      
      {/* Neural network with backward flow */}
      <div className="bg-white rounded-lg p-4 border">
        <svg width="500" height="300" viewBox="0 0 500 300" className="mx-auto">
          {/* Neural network structure (same as forward) */}
          <g opacity="0.3">
            {/* Input layer */}
            <circle cx="50" cy="60" r="25" fill="#3B82F6" />
            <circle cx="50" cy="150" r="25" fill="#3B82F6" />
            <circle cx="50" cy="240" r="25" fill="#3B82F6" />
            <text x="50" y="65" fill="white" fontSize="14" textAnchor="middle">x₁</text>
            <text x="50" y="155" fill="white" fontSize="14" textAnchor="middle">x₂</text>
            <text x="50" y="245" fill="white" fontSize="14" textAnchor="middle">x₃</text>
            
            {/* Hidden layer */}
            <circle cx="250" cy="100" r="25" fill="#10B981" />
            <circle cx="250" cy="200" r="25" fill="#10B981" />
            <text x="250" y="105" fill="white" fontSize="14" textAnchor="middle">h₁</text>
            <text x="250" y="205" fill="white" fontSize="14" textAnchor="middle">h₂</text>
            
            {/* Output layer */}
            <circle cx="450" cy="150" r="25" fill="#F59E0B" />
            <text x="450" y="155" fill="white" fontSize="14" textAnchor="middle">y</text>
            
            {/* Forward connections */}
            <line x1="75" y1="60" x2="225" y2="100" stroke="#CBD5E1" strokeWidth="1" />
            <line x1="75" y1="60" x2="225" y2="200" stroke="#CBD5E1" strokeWidth="1" />
            <line x1="75" y1="150" x2="225" y2="100" stroke="#CBD5E1" strokeWidth="1" />
            <line x1="75" y1="150" x2="225" y2="200" stroke="#CBD5E1" strokeWidth="1" />
            <line x1="75" y1="240" x2="225" y2="100" stroke="#CBD5E1" strokeWidth="1" />
            <line x1="75" y1="240" x2="225" y2="200" stroke="#CBD5E1" strokeWidth="1" />
            <line x1="275" y1="100" x2="425" y2="150" stroke="#CBD5E1" strokeWidth="1" />
            <line x1="275" y1="200" x2="425" y2="150" stroke="#CBD5E1" strokeWidth="1" />
          </g>
          
          {/* Loss indicator */}
          {step >= 1 && (
            <g>
              <rect x="430" y="50" width="40" height="30" fill="#DC2626" rx="5" />
              <text x="450" y="70" fill="white" fontSize="12" textAnchor="middle">Loss</text>
            </g>
          )}
          
          {/* Gradient flow arrows */}
          {isPlaying && step >= 1 && (
            <g>
              {/* Arrow markers */}
              <defs>
                <marker id="arrowhead" markerWidth="10" markerHeight="7" 
                  refX="0" refY="3.5" orient="auto">
                  <polygon points="0 0, 10 3.5, 0 7" fill="#DC2626" />
                </marker>
              </defs>
              
              {/* Animated gradient paths */}
              <motion.path
                d={getLossGradientPath()}
                stroke="#DC2626"
                strokeWidth="3"
                fill="none"
                markerEnd="url(#arrowhead)"
                initial={{ pathLength: 0 }}
                animate={{ pathLength: 1 }}
                transition={{ duration: 1.5, ease: "easeInOut" }}
              />
              
              {/* Gradient labels */}
              {step === 1 && <text x="460" y="120" fill="#DC2626" fontSize="12">∂L/∂y</text>}
              {step === 2 && (
                <>
                  <text x="350" y="120" fill="#DC2626" fontSize="12">∂L/∂h₁</text>
                  <text x="350" y="180" fill="#DC2626" fontSize="12">∂L/∂h₂</text>
                </>
              )}
              {step === 3 && (
                <>
                  <text x="150" y="80" fill="#DC2626" fontSize="10">∂L/∂W₁₁</text>
                  <text x="150" y="170" fill="#DC2626" fontSize="10">∂L/∂W₂₁</text>
                </>  
              )}
              {step === 4 && (
                <>
                  <text x="20" y="40" fill="#DC2626" fontSize="12">∂L/∂x₁</text>
                  <text x="20" y="130" fill="#DC2626" fontSize="12">∂L/∂x₂</text>
                  <text x="20" y="220" fill="#DC2626" fontSize="12">∂L/∂x₃</text>
                </>
              )}
            </g>
          )}
        </svg>
      </div>
      
      {/* Step explanation */}
      <div className="text-center space-y-2">
        <p className="text-sm font-semibold text-gray-700">
          {step === 0 && "Ready to start backpropagation"}
          {step === 1 && "Step 1: Computing loss gradient at output"}
          {step === 2 && "Step 2: Propagating gradients to hidden layer"}
          {step === 3 && "Step 3: Computing weight gradients for update"}
          {step === 4 && "Step 4: Gradients reach input layer"}
        </p>
        <p className="text-xs text-gray-500">
          {isPlaying ? "Gradients flow backward through the network" : "Click Play to see how gradients flow backward"}
        </p>
      </div>
    </div>
  );
};

const GradientDescentDemo: React.FC<{ isPlaying: boolean }> = ({ isPlaying }) => {
  const [position, setPosition] = React.useState({ x: -1.5, y: 0 });
  const [learningRate, setLearningRate] = React.useState(0.1);
  const [trajectory, setTrajectory] = React.useState<Array<{x: number, y: number}>>([]);
  const [step, setStep] = React.useState(0);
  
  // Simple quadratic loss function: f(x) = x^2 + 1
  const lossFunction = (x: number) => x * x + 1;
  const gradient = (x: number) => 2 * x;
  
  React.useEffect(() => {
    if (isPlaying) {
      const interval = setInterval(() => {
        setPosition(prev => {
          const grad = gradient(prev.x);
          const newX = prev.x - learningRate * grad;
          const newY = lossFunction(newX);
          
          setTrajectory(t => [...t, { x: newX, y: newY }]);
          setStep(s => s + 1);
          
          // Stop if converged
          if (Math.abs(grad) < 0.01) {
            return prev;
          }
          
          return { x: newX, y: newY };
        });
      }, 500);
      
      return () => clearInterval(interval);
    } else {
      // Reset on stop
      setPosition({ x: -1.5, y: lossFunction(-1.5) });
      setTrajectory([]);
      setStep(0);
    }
  }, [isPlaying, learningRate]);
  
  // Generate loss landscape points
  const landscapePoints = [];
  for (let x = -2; x <= 2; x += 0.1) {
    landscapePoints.push({ x, y: lossFunction(x) });
  }
  
  return (
    <div className="space-y-4">
      {/* Learning rate control */}
      <div className="bg-gray-100 rounded-lg p-4">
        <div className="flex items-center justify-between mb-2">
          <label className="text-sm font-semibold">Learning Rate: {learningRate.toFixed(2)}</label>
          <div className="text-xs text-gray-600">
            {learningRate < 0.05 ? 'Slow convergence' : learningRate > 0.5 ? 'May overshoot!' : 'Good rate'}
          </div>
        </div>
        <input
          type="range"
          min="0.01"
          max="1.0"
          step="0.01"
          value={learningRate}
          onChange={(e) => setLearningRate(parseFloat(e.target.value))}
          className="w-full"
          disabled={isPlaying}
        />
        
        <div className="mt-4 grid grid-cols-2 gap-4 text-sm">
          <div>
            <span className="font-semibold">Current position:</span> x = {position.x.toFixed(3)}
          </div>
          <div>
            <span className="font-semibold">Loss:</span> f(x) = {lossFunction(position.x).toFixed(3)}
          </div>
          <div>
            <span className="font-semibold">Gradient:</span> f'(x) = {gradient(position.x).toFixed(3)}
          </div>
          <div>
            <span className="font-semibold">Step:</span> {step}
          </div>
        </div>
      </div>
      
      {/* Loss landscape visualization */}
      <div className="bg-white rounded-lg p-4 border">
        <svg width="500" height="300" viewBox="-2.5 -0.5 5 6" className="mx-auto">
          {/* Axes */}
          <line x1="-2.5" y1="5" x2="2.5" y2="5" stroke="#000" strokeWidth="0.02" />
          <line x1="0" y1="0" x2="0" y2="5.5" stroke="#000" strokeWidth="0.02" />
          
          {/* Axis labels */}
          <text x="2.3" y="5.3" fontSize="0.2" textAnchor="end">x</text>
          <text x="-0.2" y="0.3" fontSize="0.2">Loss</text>
          
          {/* Grid lines */}
          {[-2, -1, 1, 2].map(x => (
            <g key={x}>
              <line x1={x} y1="4.95" x2={x} y2="5.05" stroke="#000" strokeWidth="0.02" />
              <text x={x} y="5.3" fontSize="0.15" textAnchor="middle">{x}</text>
            </g>
          ))}
          
          {/* Loss function curve */}
          <path
            d={`M ${landscapePoints.map(p => `${p.x},${5 - p.y}`).join(' L ')}`}
            fill="none"
            stroke="#3B82F6"
            strokeWidth="0.04"
          />
          
          {/* Trajectory */}
          {trajectory.length > 0 && (
            <path
              d={`M ${position.x},${5 - lossFunction(position.x)} ${trajectory.map(p => `L ${p.x},${5 - p.y}`).join(' ')}`}
              fill="none"
              stroke="#DC2626"
              strokeWidth="0.03"
              strokeDasharray="0.1"
            />
          )}
          
          {/* Trajectory points */}
          {trajectory.map((point, idx) => (
            <circle
              key={idx}
              cx={point.x}
              cy={5 - point.y}
              r="0.05"
              fill="#DC2626"
              opacity={0.5 + (idx / trajectory.length) * 0.5}
            />
          ))}
          
          {/* Current position */}
          <motion.circle
            cx={position.x}
            cy={5 - lossFunction(position.x)}
            r="0.1"
            fill="#DC2626"
            animate={{
              scale: [1, 1.2, 1],
            }}
            transition={{
              duration: 0.5,
              repeat: Infinity,
            }}
          />
          
          {/* Gradient arrow */}
          {!isPlaying && Math.abs(gradient(position.x)) > 0.01 && (
            <g>
              <defs>
                <marker id="gradArrow" markerWidth="10" markerHeight="7" 
                  refX="9" refY="3.5" orient="auto">
                  <polygon points="0 0, 10 3.5, 0 7" fill="#059669" />
                </marker>
              </defs>
              <line
                x1={position.x}
                y1={5 - lossFunction(position.x)}
                x2={position.x - Math.sign(gradient(position.x)) * 0.5}
                y2={5 - lossFunction(position.x)}
                stroke="#059669"
                strokeWidth="0.04"
                markerEnd="url(#gradArrow)"
              />
              <text
                x={position.x - Math.sign(gradient(position.x)) * 0.3}
                y={5 - lossFunction(position.x) - 0.2}
                fontSize="0.15"
                fill="#059669"
                textAnchor="middle"
              >
                -lr × ∇f
              </text>
            </g>
          )}
          
          {/* Minimum indicator */}
          <g>
            <line x1="0" y1="4" x2="0" y2="4.2" stroke="#059669" strokeWidth="0.02" strokeDasharray="0.05" />
            <text x="0" y="3.8" fontSize="0.15" fill="#059669" textAnchor="middle">minimum</text>
          </g>
        </svg>
      </div>
      
      {/* Explanation */}
      <div className="text-center space-y-2">
        <p className="text-sm font-semibold text-gray-700">
          {!isPlaying && "Gradient descent minimizes the loss function"}
          {isPlaying && Math.abs(gradient(position.x)) > 0.01 && `Step ${step}: Moving in direction of negative gradient`}
          {isPlaying && Math.abs(gradient(position.x)) <= 0.01 && "Converged to minimum!"}
        </p>
        <p className="text-xs text-gray-500">
          Loss function: f(x) = x² + 1 | Update rule: xₙ₊₁ = xₙ - α × f'(xₙ)
        </p>
      </div>
    </div>
  );
};

const MDPVisualizationDemo: React.FC<{ isPlaying: boolean }> = ({ isPlaying }) => {
  const [currentState, setCurrentState] = React.useState(0);
  const [action, setAction] = React.useState<number | null>(null);
  const [totalReward, setTotalReward] = React.useState(0);
  const [step, setStep] = React.useState(0);
  const [trajectory, setTrajectory] = React.useState<Array<{state: number, action: number | null, reward: number}>>([]);
  const [explorationStrategy, setExplorationStrategy] = React.useState<'random' | 'greedy' | 'epsilon-greedy'>('random');
  
  React.useEffect(() => {
    if (isPlaying) {
      const interval = setInterval(() => {
        const qValues = [0, 1, 2, 10]; // Simple Q-values favoring rightward movement
        let newAction: number;
        
        // Choose action based on exploration strategy
        switch(explorationStrategy) {
          case 'greedy':
            // Always go right towards goal (S3)
            newAction = currentState < 3 ? 1 : 0;
            break;
          case 'epsilon-greedy':
            // 80% greedy, 20% random
            if (Math.random() < 0.8) {
              newAction = currentState < 3 ? 1 : 0; // Greedy choice
            } else {
              newAction = Math.floor(Math.random() * 2); // Random choice
            }
            break;
          default: // random
            newAction = Math.floor(Math.random() * 2);
        }
        
        setAction(newAction);
        
        setTimeout(() => {
          setCurrentState(prev => {
            const nextState = newAction === 0 ? Math.max(0, prev - 1) : Math.min(3, prev + 1);
            const reward = getReward(nextState);
            
            setTotalReward(total => total + reward);
            setTrajectory(traj => [...traj, { state: nextState, action: newAction, reward }]);
            setStep(s => s + 1);
            
            return nextState;
          });
          setAction(null);
        }, 800);
      }, 2000);
      
      return () => clearInterval(interval);
    } else {
      // Reset everything
      setCurrentState(0);
      setAction(null);
      setTotalReward(0);
      setStep(0);
      setTrajectory([]);
    }
  }, [isPlaying, currentState, explorationStrategy]);
  
  const states = ['S0', 'S1', 'S2', 'S3'];
  const rewards = [-1, 0, 0, 10];
  const stateDescriptions = [
    'Start (Penalty)', 
    'Neutral', 
    'Almost There', 
    'Goal! (Reward)'
  ];
  
  const getReward = (state: number) => rewards[state];
  
  return (
    <div className="space-y-4">
      {/* MDP Explanation */}
      <div className="bg-gray-100 rounded-lg p-4 text-left">
        <h4 className="font-semibold mb-2">Markov Decision Process (MDP) Components:</h4>
        <div className="grid grid-cols-2 gap-4 text-sm">
          <div>
            <span className="font-semibold text-blue-600">States (S):</span> S0, S1, S2, S3 - Different positions
          </div>
          <div>
            <span className="font-semibold text-purple-600">Actions (A):</span> Left ←, Right → - Movement choices
          </div>
          <div>
            <span className="font-semibold text-green-600">Rewards (R):</span> [-1, 0, 0, +10] - Immediate feedback
          </div>
          <div>
            <span className="font-semibold text-orange-600">Transitions:</span> P(s'|s,a) - State changes
          </div>
        </div>
        <div className="mt-2 text-xs text-gray-600">
          <strong>Goal:</strong> Learn to reach S3 (highest reward) while minimizing time in S0 (penalty)
        </div>
      </div>
      
      {/* Exploration Strategy Selector */}
      <div className="bg-blue-50 rounded-lg p-3">
        <div className="flex items-center justify-between mb-2">
          <label className="text-sm font-semibold">Exploration Strategy:</label>
          <select 
            value={explorationStrategy} 
            onChange={(e) => setExplorationStrategy(e.target.value as any)}
            className="text-sm border rounded px-2 py-1"
            disabled={isPlaying}
          >
            <option value="random">Random (Pure Exploration)</option>
            <option value="greedy">Greedy (Pure Exploitation)</option>
            <option value="epsilon-greedy">ε-Greedy (80% Greedy, 20% Random)</option>
          </select>
        </div>
        <div className="text-xs text-gray-600">
          {explorationStrategy === 'random' && "Agent chooses actions randomly - explores but never learns optimal path"}
          {explorationStrategy === 'greedy' && "Agent always chooses best known action - fast goal reaching but no exploration"}
          {explorationStrategy === 'epsilon-greedy' && "Agent mostly chooses best action but sometimes explores - balances learning and performance"}
        </div>
      </div>
      
      {/* MDP Visualization */}
      <div className="bg-white rounded-lg p-4 border">
        {/* States */}
        <div className="flex justify-center items-center gap-6 mb-6">
          {states.map((state, idx) => (
            <div key={idx} className="relative text-center">
              {/* Transition arrows */}
              {idx < states.length - 1 && (
                <div className="absolute top-1/2 -right-3 transform -translate-y-1/2 z-10">
                  <svg width="24" height="24" viewBox="0 0 24 24">
                    <path d="M8 5v14l11-7z" fill={action === 1 && currentState === idx ? '#8B5CF6' : '#CBD5E1'} />
                  </svg>
                </div>
              )}
              {idx > 0 && (
                <div className="absolute top-1/2 -left-3 transform -translate-y-1/2 rotate-180 z-10">
                  <svg width="24" height="24" viewBox="0 0 24 24">
                    <path d="M8 5v14l11-7z" fill={action === 0 && currentState === idx ? '#8B5CF6' : '#CBD5E1'} />
                  </svg>
                </div>
              )}
              
              {/* State circle */}
              <div className={`w-24 h-24 rounded-full flex flex-col items-center justify-center font-bold text-white transition-all border-4 ${
                currentState === idx ? 'bg-blue-600 scale-110 border-blue-800 shadow-lg' : 'bg-gray-400 border-gray-500'
              }`}>
                <div className="text-lg">{state}</div>
                <div className="text-xs opacity-80">r={rewards[idx]}</div>
              </div>
              
              {/* State description */}
              <div className="text-xs mt-2 font-medium text-gray-700">
                {stateDescriptions[idx]}
              </div>
              
              {/* Agent indicator */}
              {currentState === idx && (
                <motion.div
                  initial={{ scale: 0 }}
                  animate={{ scale: 1 }}
                  className="absolute -top-2 -right-2 w-6 h-6 bg-red-500 rounded-full flex items-center justify-center text-white text-xs font-bold"
                >
                  🤖
                </motion.div>
              )}
            </div>
          ))}
        </div>
        
        {/* Current Action Display */}
        {action !== null && (
          <motion.div
            initial={{ opacity: 0, scale: 0.8 }}
            animate={{ opacity: 1, scale: 1 }}
            className="text-center mb-4"
          >
            <div className="inline-block bg-purple-100 border-2 border-purple-500 rounded-lg px-4 py-2">
              <div className="text-lg font-semibold text-purple-700">
                Taking Action: {action === 0 ? '← Left' : 'Right →'}
              </div>
              <div className="text-sm text-purple-600">
                Strategy: {explorationStrategy}
              </div>
            </div>
          </motion.div>
        )}
        
        {/* Statistics */}
        <div className="grid grid-cols-3 gap-4 text-center bg-gray-50 rounded p-3">
          <div>
            <div className="text-2xl font-bold text-blue-600">{step}</div>
            <div className="text-xs text-gray-600">Steps Taken</div>
          </div>
          <div>
            <div className="text-2xl font-bold text-green-600">{totalReward}</div>
            <div className="text-xs text-gray-600">Total Reward</div>
          </div>
          <div>
            <div className="text-2xl font-bold text-orange-600">
              {trajectory.filter(t => t.state === 3).length}
            </div>
            <div className="text-xs text-gray-600">Goal Reached</div>
          </div>
        </div>
      </div>
      
      {/* Learning Insights */}
      <div className="text-center space-y-2">
        <p className="text-sm font-semibold text-gray-700">
          {!isPlaying && "This MDP teaches how agents learn to make sequential decisions"}
          {isPlaying && step === 0 && "Agent starting exploration..."}
          {isPlaying && step > 0 && step < 5 && "Learning state transitions and rewards..."}
          {isPlaying && step >= 5 && totalReward > 20 && "Agent is finding the goal efficiently!"}
          {isPlaying && step >= 5 && totalReward <= 0 && "Agent struggling - needs better strategy"}
        </p>
        <p className="text-xs text-gray-500">
          The Markov Property: Next state depends only on current state + action (not history)
        </p>
      </div>
    </div>
  );
};

// GAE Visualization Demo
const GAEVisualizationDemo: React.FC<{ isPlaying: boolean }> = ({ isPlaying }) => {
  const [lambda, setLambda] = React.useState(0.95);
  const [step, setStep] = React.useState(0);
  
  React.useEffect(() => {
    if (isPlaying) {
      const interval = setInterval(() => {
        setStep(prev => (prev + 1) % 5);
      }, 1000);
      return () => clearInterval(interval);
    }
  }, [isPlaying]);
  
  const advantages = [2.5, 1.8, -0.5, 1.2, 0.8];
  const tdErrors = [1.0, 0.5, -0.8, 0.6, 0.3];
  
  return (
    <div className="w-full">
      <div className="mb-4">
        <label className="block text-sm font-medium text-gray-700 mb-1">
          Lambda (λ) = {lambda.toFixed(2)}
        </label>
        <input
          type="range"
          min="0"
          max="1"
          step="0.05"
          value={lambda}
          onChange={(e) => setLambda(parseFloat(e.target.value))}
          className="w-full"
        />
      </div>
      
      <div className="grid grid-cols-5 gap-2 mb-4">
        {advantages.map((adv, idx) => (
          <div
            key={idx}
            className={`p-3 rounded text-center transition-all ${
              idx <= step ? 'bg-blue-100 border-2 border-blue-500' : 'bg-gray-100'
            }`}
          >
            <div className="text-xs font-semibold mb-1">t = {idx}</div>
            <div className="text-sm">δ = {tdErrors[idx].toFixed(1)}</div>
            <div className="text-xs text-gray-600 mt-1">
              weight: {Math.pow(lambda, idx).toFixed(2)}
            </div>
          </div>
        ))}
      </div>
      
      <div className="bg-gray-50 p-4 rounded">
        <div className="text-sm font-semibold mb-2">GAE Calculation:</div>
        <div className="text-xs font-mono">
          A_t = Σ(λ^l * δ_(t+l))
        </div>
        {isPlaying && (
          <div className="mt-2 text-sm">
            Current advantage estimate: {
              advantages.slice(0, step + 1)
                .reduce((sum, _, idx) => sum + tdErrors[idx] * Math.pow(lambda, idx), 0)
                .toFixed(2)
            }
          </div>
        )}
      </div>
      
      <p className="text-xs text-gray-600 mt-3">
        {isPlaying 
          ? "Calculating advantage using exponentially weighted TD errors..." 
          : "Click Play to see GAE calculation step by step"
        }
      </p>
    </div>
  );
};

// Value Function Grid Demo
const ValueFunctionGridDemo: React.FC<{ isPlaying: boolean }> = ({ isPlaying }) => {
  const [iteration, setIteration] = React.useState(0);
  const [selectedCell, setSelectedCell] = React.useState<{row: number, col: number} | null>(null);
  
  // Initial grid values (5x5 grid world)
  const [gridValues, setGridValues] = React.useState([
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 10] // Goal state with reward
  ]);
  
  // Value iteration update
  const updateValues = () => {
    const gamma = 0.9; // discount factor
    const newValues = gridValues.map((row, i) => 
      row.map((_, j) => {
        if (i === 4 && j === 4) return 10; // Goal state
        
        // Calculate max value from possible next states
        const neighbors = [
          [i-1, j], [i+1, j], [i, j-1], [i, j+1]
        ];
        
        let maxValue = 0;
        neighbors.forEach(([ni, nj]) => {
          if (ni >= 0 && ni < 5 && nj >= 0 && nj < 5) {
            const immediateReward = (ni === 4 && nj === 4) ? 10 : 0;
            const futureValue = gridValues[ni][nj];
            maxValue = Math.max(maxValue, immediateReward + gamma * futureValue);
          }
        });
        
        return maxValue;
      })
    );
    
    setGridValues(newValues);
    setIteration(prev => prev + 1);
  };
  
  React.useEffect(() => {
    if (isPlaying) {
      const interval = setInterval(updateValues, 1000);
      return () => clearInterval(interval);
    } else {
      // Reset
      setGridValues([
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 10]
      ]);
      setIteration(0);
    }
  }, [isPlaying]);
  
  // Color based on value
  const getColor = (value: number) => {
    const intensity = Math.min(value / 10, 1);
    return `rgba(34, 197, 94, ${intensity})`; // Green with varying intensity
  };
  
  return (
    <div className="space-y-4">
      {/* Info panel */}
      <div className="bg-gray-100 rounded-lg p-4">
        <h4 className="font-semibold mb-2">Value Iteration in Grid World</h4>
        <div className="grid grid-cols-2 gap-4 text-sm">
          <div>
            <span className="font-semibold">Iteration:</span> {iteration}
          </div>
          <div>
            <span className="font-semibold">Discount factor (γ):</span> 0.9
          </div>
          <div className="col-span-2">
            <span className="font-semibold">Update rule:</span> V(s) = max_a[R(s,a) + γ Σ P(s'|s,a)V(s')]
          </div>
        </div>
      </div>
      
      {/* Grid visualization */}
      <div className="bg-white rounded-lg p-4 border">
        <div className="grid grid-cols-5 gap-2 max-w-md mx-auto">
          {gridValues.map((row, i) => 
            row.map((value, j) => (
              <div
                key={`${i}-${j}`}
                className="relative aspect-square border-2 border-gray-300 rounded cursor-pointer transition-all hover:border-blue-500"
                style={{ backgroundColor: getColor(value) }}
                onMouseEnter={() => setSelectedCell({row: i, col: j})}
                onMouseLeave={() => setSelectedCell(null)}
              >
                <div className="absolute inset-0 flex flex-col items-center justify-center">
                  <div className="text-xs font-semibold">
                    {value.toFixed(1)}
                  </div>
                  {i === 4 && j === 4 && (
                    <div className="text-xs text-green-700 font-bold">GOAL</div>
                  )}
                </div>
                
                {/* Arrows showing policy */}
                {value > 0 && !(i === 4 && j === 4) && isPlaying && iteration > 2 && (
                  <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
                    {/* Find best action */}
                    {(() => {
                      const neighbors = [
                        {di: -1, dj: 0, arrow: '↑'},
                        {di: 1, dj: 0, arrow: '↓'},
                        {di: 0, dj: -1, arrow: '←'},
                        {di: 0, dj: 1, arrow: '→'}
                      ];
                      
                      let bestAction = '';
                      let bestValue = -Infinity;
                      
                      neighbors.forEach(({di, dj, arrow}) => {
                        const ni = i + di;
                        const nj = j + dj;
                        if (ni >= 0 && ni < 5 && nj >= 0 && nj < 5) {
                          if (gridValues[ni][nj] > bestValue) {
                            bestValue = gridValues[ni][nj];
                            bestAction = arrow;
                          }
                        }
                      });
                      
                      return (
                        <div className="text-2xl text-gray-700 opacity-50">
                          {bestAction}
                        </div>
                      );
                    })()}
                  </div>
                )}
              </div>
            ))
          )}
        </div>
        
        {/* Cell details */}
        {selectedCell && (
          <div className="mt-4 text-sm text-gray-600 text-center">
            Cell ({selectedCell.row}, {selectedCell.col}): V = {gridValues[selectedCell.row][selectedCell.col].toFixed(2)}
          </div>
        )}
      </div>
      
      {/* Explanation */}
      <div className="text-center space-y-2">
        <p className="text-sm font-semibold text-gray-700">
          {!isPlaying && "Value iteration computes optimal state values"}
          {isPlaying && iteration < 3 && "Values propagating from goal state..."}
          {isPlaying && iteration >= 3 && "Arrows show optimal policy based on values"}
        </p>
        <p className="text-xs text-gray-500">
          Each cell shows V(s) - the expected return from that state following the optimal policy
        </p>
      </div>
    </div>
  );
};

// Policy Gradient Intuition Demo
const PolicyGradientIntuitionDemo: React.FC<{ isPlaying: boolean }> = ({ isPlaying }) => {
  const [step, setStep] = React.useState(0);
  const [policyParams, setPolicyParams] = React.useState({ mean: 0, std: 1 });
  const [samples, setSamples] = React.useState<Array<{action: number, reward: number, gradient: number}>>([]);
  
  React.useEffect(() => {
    if (isPlaying) {
      const interval = setInterval(() => {
        setStep(prev => {
          const nextStep = (prev + 1) % 5;
          
          if (nextStep === 1) {
            // Generate samples
            const newSamples = [];
            for (let i = 0; i < 5; i++) {
              const action = policyParams.mean + (Math.random() - 0.5) * 2 * policyParams.std;
              // Reward is higher for actions closer to 2
              const reward = -Math.pow(action - 2, 2) + 4;
              const gradient = (action - policyParams.mean) * reward / (policyParams.std * policyParams.std);
              newSamples.push({ action, reward, gradient });
            }
            setSamples(newSamples);
          } else if (nextStep === 3) {
            // Update policy
            const avgGradient = samples.reduce((sum, s) => sum + s.gradient, 0) / samples.length;
            setPolicyParams(prev => ({
              mean: prev.mean + 0.1 * avgGradient,
              std: prev.std
            }));
          }
          
          return nextStep;
        });
      }, 1500);
      
      return () => clearInterval(interval);
    } else {
      setStep(0);
      setPolicyParams({ mean: 0, std: 1 });
      setSamples([]);
    }
  }, [isPlaying, samples]);
  
  // Generate policy distribution points
  const policyPoints = [];
  for (let x = -4; x <= 6; x += 0.1) {
    const prob = Math.exp(-0.5 * Math.pow((x - policyParams.mean) / policyParams.std, 2)) / 
                  (policyParams.std * Math.sqrt(2 * Math.PI));
    policyPoints.push({ x, y: prob });
  }
  
  // Generate reward function points
  const rewardPoints = [];
  for (let x = -4; x <= 6; x += 0.1) {
    const reward = -Math.pow(x - 2, 2) + 4;
    rewardPoints.push({ x, y: reward / 8 + 0.5 }); // Scaled for visualization
  }
  
  return (
    <div className="space-y-4">
      {/* Algorithm steps */}
      <div className="bg-gray-100 rounded-lg p-4">
        <h4 className="font-semibold mb-2">Policy Gradient Algorithm</h4>
        <div className="space-y-2 text-sm">
          <div className={`flex items-center gap-2 ${step === 1 ? 'text-blue-600 font-bold' : ''}`}>
            <div className="w-6 h-6 rounded-full bg-blue-500 text-white flex items-center justify-center text-xs">1</div>
            <span>Sample actions from policy π(a|θ)</span>
          </div>
          <div className={`flex items-center gap-2 ${step === 2 ? 'text-green-600 font-bold' : ''}`}>
            <div className="w-6 h-6 rounded-full bg-green-500 text-white flex items-center justify-center text-xs">2</div>
            <span>Observe rewards R(a)</span>
          </div>
          <div className={`flex items-center gap-2 ${step === 3 ? 'text-purple-600 font-bold' : ''}`}>
            <div className="w-6 h-6 rounded-full bg-purple-500 text-white flex items-center justify-center text-xs">3</div>
            <span>Calculate gradients: ∇log π(a|θ) × R(a)</span>
          </div>
          <div className={`flex items-center gap-2 ${step === 4 ? 'text-orange-600 font-bold' : ''}`}>
            <div className="w-6 h-6 rounded-full bg-orange-500 text-white flex items-center justify-center text-xs">4</div>
            <span>Update parameters: θ ← θ + α∇J(θ)</span>
          </div>
        </div>
      </div>
      
      {/* Visualization */}
      <div className="bg-white rounded-lg p-4 border">
        <svg width="500" height="300" viewBox="-4.5 -0.5 11 3" className="mx-auto">
          {/* Axes */}
          <line x1="-4.5" y1="2" x2="6.5" y2="2" stroke="#000" strokeWidth="0.02" />
          <line x1="-4" y1="0" x2="-4" y2="2.5" stroke="#000" strokeWidth="0.02" />
          
          {/* Axis labels */}
          <text x="6.3" y="2.3" fontSize="0.2" textAnchor="end">action</text>
          <text x="-4.3" y="0.3" fontSize="0.2" textAnchor="end">π(a)</text>
          
          {/* Reward function (background) */}
          <path
            d={`M ${rewardPoints.map(p => `${p.x},${2 - p.y}`).join(' L ')}`}
            fill="none"
            stroke="#FEF3C7"
            strokeWidth="0.03"
            opacity="0.8"
          />
          <text x="2" y="0.8" fontSize="0.15" fill="#F59E0B" textAnchor="middle">reward function</text>
          
          {/* Policy distribution */}
          <path
            d={`M ${policyPoints.map(p => `${p.x},${2 - p.y * 2}`).join(' L ')}`}
            fill="none"
            stroke="#3B82F6"
            strokeWidth="0.04"
          />
          
          {/* Policy parameters */}
          <line x1={policyParams.mean} y1="1.8" x2={policyParams.mean} y2="2.2" 
            stroke="#3B82F6" strokeWidth="0.03" strokeDasharray="0.05" />
          <text x={policyParams.mean} y="2.4" fontSize="0.15" fill="#3B82F6" textAnchor="middle">
            μ={policyParams.mean.toFixed(2)}
          </text>
          
          {/* Samples and gradients */}
          {samples.map((sample, idx) => (
            <g key={idx}>
              {/* Action sample */}
              <circle cx={sample.action} cy="2" r="0.08" 
                fill={sample.reward > 0 ? '#10B981' : '#EF4444'} 
                opacity="0.8" 
              />
              
              {/* Gradient arrow */}
              {step >= 3 && (
                <g>
                  <defs>
                    <marker id={`arrow-${idx}`} markerWidth="10" markerHeight="7" 
                      refX="9" refY="3.5" orient="auto">
                      <polygon points="0 0, 10 3.5, 0 7" fill="#8B5CF6" />
                    </marker>
                  </defs>
                  <line
                    x1={sample.action}
                    y1="1.7"
                    x2={sample.action + Math.sign(sample.gradient) * 0.3}
                    y2="1.7"
                    stroke="#8B5CF6"
                    strokeWidth="0.03"
                    markerEnd={`url(#arrow-${idx})`}
                    opacity="0.7"
                  />
                </g>
              )}
            </g>
          ))}
          
          {/* Target indicator */}
          <g>
            <line x1="2" y1="1.5" x2="2" y2="2" stroke="#059669" strokeWidth="0.02" strokeDasharray="0.05" />
            <text x="2" y="1.4" fontSize="0.15" fill="#059669" textAnchor="middle">target</text>
          </g>
        </svg>
      </div>
      
      {/* Current stats */}
      {samples.length > 0 && (
        <div className="bg-gray-50 rounded p-3 text-sm grid grid-cols-2 gap-2">
          <div><span className="font-semibold">Avg reward:</span> {(samples.reduce((s, x) => s + x.reward, 0) / samples.length).toFixed(2)}</div>
          <div><span className="font-semibold">Avg gradient:</span> {(samples.reduce((s, x) => s + x.gradient, 0) / samples.length).toFixed(3)}</div>
        </div>
      )}
      
      {/* Explanation */}
      <div className="text-center space-y-2">
        <p className="text-sm font-semibold text-gray-700">
          {!isPlaying && "Policy gradients directly optimize the policy parameters"}
          {isPlaying && step === 1 && "Sampling actions from current policy..."}
          {isPlaying && step === 2 && "Observing rewards for each action..."}
          {isPlaying && step === 3 && "Computing gradients weighted by rewards..."}
          {isPlaying && step === 4 && "Policy shifts towards high-reward actions!"}
        </p>
        <p className="text-xs text-gray-500">
          Good actions (high reward) increase their probability, bad actions decrease
        </p>
      </div>
    </div>
  );
};