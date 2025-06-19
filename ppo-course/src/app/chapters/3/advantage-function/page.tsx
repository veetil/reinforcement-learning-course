'use client';

import React, { useState } from 'react';
import { AdvantageFunctionVisualizer } from '@/components/interactive/AdvantageFunction';
import { motion } from 'framer-motion';
import { TrendingUp, Brain, Calculator, AlertCircle } from 'lucide-react';

export default function AdvantageFunctionDemo() {
  const [showAdvantage, setShowAdvantage] = useState(true);
  const [showGAE, setShowGAE] = useState(false);

  return (
    <div className="min-h-screen bg-gray-50 py-8">
      <div className="max-w-7xl mx-auto px-4">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="mb-8"
        >
          <h1 className="text-4xl font-bold mb-4">Chapter 3: Advantage Functions</h1>
          <p className="text-xl text-gray-600">
            Understanding the core of PPO: How much better is an action than average?
          </p>
        </motion.div>

        <div className="grid lg:grid-cols-4 gap-8">
          {/* Main Visualizer */}
          <div className="lg:col-span-3">
            <motion.div
              initial={{ opacity: 0, scale: 0.95 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ delay: 0.1 }}
              className="bg-white rounded-lg shadow-lg p-6"
            >
              <h2 className="text-2xl font-bold mb-4">Interactive Advantage Explorer</h2>
              
              <AdvantageFunctionVisualizer
                gridSize={8}
                showAdvantage={showAdvantage}
                showGAE={showGAE}
              />
            </motion.div>
          </div>

          {/* Learning Panel */}
          <div className="space-y-6">
            {/* Core Concepts */}
            <motion.div
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: 0.2 }}
              className="bg-white rounded-lg shadow-lg p-6"
            >
              <h3 className="text-xl font-bold mb-4 flex items-center gap-2">
                <Brain className="w-5 h-5" />
                Core Concepts
              </h3>
              
              <div className="space-y-4">
                <div>
                  <h4 className="font-semibold text-blue-600">Advantage A(s,a)</h4>
                  <p className="text-sm text-gray-600">
                    A(s,a) = Q(s,a) - V(s)
                  </p>
                  <p className="text-xs text-gray-500 mt-1">
                    How much better is action a than the average action?
                  </p>
                </div>
                
                <div>
                  <h4 className="font-semibold text-green-600">Q(s,a) - Action Value</h4>
                  <p className="text-sm text-gray-600">
                    Expected return when taking action a in state s
                  </p>
                </div>
                
                <div>
                  <h4 className="font-semibold text-purple-600">V(s) - State Value</h4>
                  <p className="text-sm text-gray-600">
                    Average expected return from state s
                  </p>
                </div>
                
                <div>
                  <h4 className="font-semibold text-orange-600">GAE (λ)</h4>
                  <p className="text-sm text-gray-600">
                    Generalized Advantage Estimation
                  </p>
                  <p className="text-xs text-gray-500 mt-1">
                    λ = 0: High bias, low variance<br/>
                    λ = 1: Low bias, high variance
                  </p>
                </div>
              </div>
            </motion.div>

            {/* Why Advantages? */}
            <motion.div
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: 0.3 }}
              className="bg-gradient-to-br from-blue-50 to-purple-50 rounded-lg shadow-lg p-6"
            >
              <h3 className="text-xl font-bold mb-4 flex items-center gap-2">
                <TrendingUp className="w-5 h-5" />
                Why Use Advantages?
              </h3>
              
              <div className="space-y-3 text-sm">
                <div>
                  <strong className="text-blue-700">1. Reduced Variance</strong>
                  <p className="text-gray-600">
                    Subtracting baseline V(s) reduces gradient variance
                  </p>
                </div>
                
                <div>
                  <strong className="text-blue-700">2. Better Credit Assignment</strong>
                  <p className="text-gray-600">
                    Identifies which actions are actually good vs lucky
                  </p>
                </div>
                
                <div>
                  <strong className="text-blue-700">3. Stable Learning</strong>
                  <p className="text-gray-600">
                    Prevents large policy updates from high-return trajectories
                  </p>
                </div>
                
                <div>
                  <strong className="text-blue-700">4. PPO's Secret Sauce</strong>
                  <p className="text-gray-600">
                    Clipped objective works best with normalized advantages
                  </p>
                </div>
              </div>
            </motion.div>

            {/* GAE Explained */}
            <motion.div
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: 0.4 }}
              className="bg-yellow-50 rounded-lg shadow-lg p-6"
            >
              <h3 className="text-xl font-bold mb-4 flex items-center gap-2">
                <Calculator className="w-5 h-5" />
                GAE Formula
              </h3>
              
              <div className="space-y-2 text-sm">
                <p className="font-mono bg-white p-2 rounded text-xs">
                  Â<sub>t</sub> = Σ(γλ)<sup>l</sup>δ<sub>t+l</sub>
                </p>
                <p className="text-gray-600">
                  where δ<sub>t</sub> = r<sub>t</sub> + γV(s<sub>t+1</sub>) - V(s<sub>t</sub>)
                </p>
                <div className="mt-3 text-xs space-y-1">
                  <p>• δ is the TD error</p>
                  <p>• λ controls bias-variance tradeoff</p>
                  <p>• Exponentially weighted sum of TD errors</p>
                </div>
              </div>
            </motion.div>

            {/* Common Pitfalls */}
            <motion.div
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: 0.5 }}
              className="bg-red-50 rounded-lg shadow-lg p-6"
            >
              <h3 className="text-xl font-bold mb-4 flex items-center gap-2">
                <AlertCircle className="w-5 h-5 text-red-600" />
                Common Pitfalls
              </h3>
              
              <ul className="space-y-2 text-sm text-gray-700">
                <li className="flex items-start gap-2">
                  <span className="text-red-600">•</span>
                  Not normalizing advantages before policy update
                </li>
                <li className="flex items-start gap-2">
                  <span className="text-red-600">•</span>
                  Using raw returns instead of advantages
                </li>
                <li className="flex items-start gap-2">
                  <span className="text-red-600">•</span>
                  Setting λ too high (unstable) or too low (biased)
                </li>
                <li className="flex items-start gap-2">
                  <span className="text-red-600">•</span>
                  Forgetting to handle terminal states in GAE
                </li>
              </ul>
            </motion.div>
          </div>
        </div>

        {/* Interactive Exercises */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.6 }}
          className="mt-8 bg-indigo-50 rounded-lg p-6 border-2 border-indigo-200"
        >
          <h3 className="text-xl font-bold mb-4">🎯 Interactive Exercises</h3>
          
          <div className="grid md:grid-cols-3 gap-6">
            <div>
              <h4 className="font-semibold mb-2">Exercise 1: Advantage Signs</h4>
              <ol className="text-sm space-y-1">
                <li>1. Set a goal in the corner</li>
                <li>2. Click cells near the goal</li>
                <li>3. Which actions have positive advantages?</li>
                <li>4. Why are some actions negative?</li>
              </ol>
            </div>
            
            <div>
              <h4 className="font-semibold mb-2">Exercise 2: GAE Analysis</h4>
              <ol className="text-sm space-y-1">
                <li>1. Enable trajectory mode</li>
                <li>2. Create a path to the goal</li>
                <li>3. Toggle GAE visualization</li>
                <li>4. How does λ affect the values?</li>
              </ol>
            </div>
            
            <div>
              <h4 className="font-semibold mb-2">Exercise 3: Policy Comparison</h4>
              <ol className="text-sm space-y-1">
                <li>1. Note advantages for optimal path</li>
                <li>2. Try a suboptimal path</li>
                <li>3. Compare advantage magnitudes</li>
                <li>4. What does PPO learn from this?</li>
              </ol>
            </div>
          </div>

          <div className="mt-6 p-4 bg-white rounded-lg">
            <h4 className="font-semibold mb-2">💡 Key Insight</h4>
            <p className="text-sm text-gray-700">
              PPO uses advantages to determine policy updates. Positive advantages → increase action probability. 
              Negative advantages → decrease action probability. The clipping prevents too large changes, 
              maintaining stability while improving the policy.
            </p>
          </div>
        </motion.div>

        {/* Code Example */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.7 }}
          className="mt-8 bg-gray-900 rounded-lg p-6 text-white"
        >
          <h3 className="text-xl font-bold mb-4">📝 Implementation Snippet</h3>
          <pre className="text-sm overflow-x-auto">
{`# PPO Advantage Calculation
def compute_advantages(rewards, values, gamma=0.99, lam=0.95):
    advantages = []
    gae = 0
    
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * values[t + 1] - values[t]
        gae = delta + gamma * lam * gae
        advantages.insert(0, gae)
    
    # Normalize advantages
    advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-8)
    return advantages`}
          </pre>
        </motion.div>
      </div>
    </div>
  );
}