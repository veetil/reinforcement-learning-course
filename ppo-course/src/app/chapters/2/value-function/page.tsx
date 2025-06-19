'use client';

import React, { useState } from 'react';
import { ValueFunctionVisualizer } from '@/components/interactive/ValueFunction';
import { motion } from 'framer-motion';
import { Info, BookOpen, Lightbulb } from 'lucide-react';

export default function ValueFunctionDemo() {
  const [showValues, setShowValues] = useState(true);
  const [showPolicy, setShowPolicy] = useState(false);
  const [showConvergence, setShowConvergence] = useState(false);

  return (
    <div className="min-h-screen bg-gray-50 py-8">
      <div className="max-w-7xl mx-auto px-4">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="mb-8"
        >
          <h1 className="text-4xl font-bold mb-4">Chapter 2: Value Function Visualization</h1>
          <p className="text-xl text-gray-600">
            Understand how value functions propagate through state space
          </p>
        </motion.div>

        <div className="grid lg:grid-cols-3 gap-8">
          {/* Main Visualizer */}
          <div className="lg:col-span-2">
            <motion.div
              initial={{ opacity: 0, scale: 0.95 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ delay: 0.1 }}
              className="bg-white rounded-lg shadow-lg p-6"
            >
              <h2 className="text-2xl font-bold mb-4">Interactive Value Function</h2>
              
              {/* Visualization Options */}
              <div className="mb-4 flex flex-wrap gap-2">
                <button
                  onClick={() => setShowValues(!showValues)}
                  className={`px-3 py-1 rounded-lg transition-colors ${
                    showValues 
                      ? 'bg-blue-600 text-white' 
                      : 'bg-gray-200 text-gray-700 hover:bg-gray-300'
                  }`}
                >
                  Show Values
                </button>
                
                <button
                  onClick={() => setShowPolicy(!showPolicy)}
                  className={`px-3 py-1 rounded-lg transition-colors ${
                    showPolicy 
                      ? 'bg-blue-600 text-white' 
                      : 'bg-gray-200 text-gray-700 hover:bg-gray-300'
                  }`}
                >
                  Show Policy
                </button>
                
                <button
                  onClick={() => setShowConvergence(!showConvergence)}
                  className={`px-3 py-1 rounded-lg transition-colors ${
                    showConvergence 
                      ? 'bg-blue-600 text-white' 
                      : 'bg-gray-200 text-gray-700 hover:bg-gray-300'
                  }`}
                >
                  Convergence Graph
                </button>
              </div>
              
              <ValueFunctionVisualizer
                width={8}
                height={8}
                showValues={showValues}
                showPolicy={showPolicy}
                animated={true}
                showConvergence={showConvergence}
              />
            </motion.div>
          </div>

          {/* Learning Panel */}
          <div className="space-y-6">
            {/* Key Concepts */}
            <motion.div
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: 0.2 }}
              className="bg-white rounded-lg shadow-lg p-6"
            >
              <h3 className="text-xl font-bold mb-4 flex items-center gap-2">
                <BookOpen className="w-5 h-5" />
                Value Function Concepts
              </h3>
              
              <div className="space-y-4">
                <div>
                  <h4 className="font-semibold text-blue-600">V(s) - State Value</h4>
                  <p className="text-sm text-gray-600">
                    Expected cumulative reward starting from state s, following policy π
                  </p>
                </div>
                
                <div>
                  <h4 className="font-semibold text-green-600">Bellman Equation</h4>
                  <p className="text-sm text-gray-600">
                    V(s) = R(s) + γ × max[V(s')]
                  </p>
                  <p className="text-xs text-gray-500 mt-1">
                    Value = immediate reward + discounted future value
                  </p>
                </div>
                
                <div>
                  <h4 className="font-semibold text-purple-600">Discount Factor (γ)</h4>
                  <p className="text-sm text-gray-600">
                    Controls importance of future rewards (0-1)
                  </p>
                  <ul className="text-xs text-gray-500 mt-1 space-y-1">
                    <li>• γ = 0: Only immediate rewards matter</li>
                    <li>• γ = 1: Future rewards equally important</li>
                    <li>• γ = 0.9: Balanced (common choice)</li>
                  </ul>
                </div>
                
                <div>
                  <h4 className="font-semibold text-orange-600">Value Iteration</h4>
                  <p className="text-sm text-gray-600">
                    Iteratively update values until convergence
                  </p>
                </div>
              </div>
            </motion.div>

            {/* Insights */}
            <motion.div
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: 0.3 }}
              className="bg-gradient-to-br from-yellow-50 to-orange-50 rounded-lg shadow-lg p-6"
            >
              <h3 className="text-xl font-bold mb-4 flex items-center gap-2">
                <Lightbulb className="w-5 h-5" />
                Key Insights
              </h3>
              
              <div className="space-y-3 text-sm">
                <div className="flex items-start gap-2">
                  <span className="text-yellow-600">💡</span>
                  <p>
                    Value propagates backward from goal to start
                  </p>
                </div>
                
                <div className="flex items-start gap-2">
                  <span className="text-yellow-600">💡</span>
                  <p>
                    Higher discount factor = longer planning horizon
                  </p>
                </div>
                
                <div className="flex items-start gap-2">
                  <span className="text-yellow-600">💡</span>
                  <p>
                    Obstacles create "value shadows" behind them
                  </p>
                </div>
                
                <div className="flex items-start gap-2">
                  <span className="text-yellow-600">💡</span>
                  <p>
                    Optimal policy follows value gradient upward
                  </p>
                </div>
              </div>
            </motion.div>

            {/* PPO Connection */}
            <motion.div
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: 0.4 }}
              className="bg-blue-50 rounded-lg shadow-lg p-6"
            >
              <h3 className="text-xl font-bold mb-4 flex items-center gap-2">
                <Info className="w-5 h-5" />
                Connection to PPO
              </h3>
              
              <p className="text-sm text-gray-700 mb-3">
                In PPO, we use value functions for:
              </p>
              
              <ol className="text-sm space-y-2">
                <li className="flex items-start gap-2">
                  <span className="font-bold text-blue-600">1.</span>
                  <div>
                    <strong>Advantage Estimation:</strong>
                    <p className="text-xs text-gray-600 mt-1">
                      A(s,a) = Q(s,a) - V(s)
                    </p>
                  </div>
                </li>
                
                <li className="flex items-start gap-2">
                  <span className="font-bold text-blue-600">2.</span>
                  <div>
                    <strong>Critic Network:</strong>
                    <p className="text-xs text-gray-600 mt-1">
                      Learns V(s) to reduce variance
                    </p>
                  </div>
                </li>
                
                <li className="flex items-start gap-2">
                  <span className="font-bold text-blue-600">3.</span>
                  <div>
                    <strong>GAE (Generalized Advantage Estimation):</strong>
                    <p className="text-xs text-gray-600 mt-1">
                      Uses V(s) for bias-variance tradeoff
                    </p>
                  </div>
                </li>
              </ol>
            </motion.div>
          </div>
        </div>

        {/* Interactive Exercise */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.5 }}
          className="mt-8 bg-purple-50 rounded-lg p-6 border-2 border-purple-200"
        >
          <h3 className="text-xl font-bold mb-4">🧪 Try This Experiment</h3>
          
          <div className="grid md:grid-cols-2 gap-6">
            <div>
              <h4 className="font-semibold mb-2">Experiment 1: Discount Factor</h4>
              <ol className="text-sm space-y-1">
                <li>1. Set a goal in the corner</li>
                <li>2. Try γ = 0.5 (low discount)</li>
                <li>3. Try γ = 0.99 (high discount)</li>
                <li>4. Notice how value spread changes</li>
              </ol>
            </div>
            
            <div>
              <h4 className="font-semibold mb-2">Experiment 2: Obstacle Impact</h4>
              <ol className="text-sm space-y-1">
                <li>1. Place goal in center</li>
                <li>2. Add obstacles around it</li>
                <li>3. Watch value flow around obstacles</li>
                <li>4. Enable policy arrows to see paths</li>
              </ol>
            </div>
            
            <div>
              <h4 className="font-semibold mb-2">Experiment 3: Convergence</h4>
              <ol className="text-sm space-y-1">
                <li>1. Enable convergence graph</li>
                <li>2. Click "Animate" button</li>
                <li>3. Watch values propagate</li>
                <li>4. See convergence rate in graph</li>
              </ol>
            </div>
            
            <div>
              <h4 className="font-semibold mb-2">Challenge Question</h4>
              <p className="text-sm text-gray-700">
                Why do values decrease exponentially with distance from goal? 
                How does this relate to the discount factor?
              </p>
            </div>
          </div>
        </motion.div>
      </div>
    </div>
  );
}