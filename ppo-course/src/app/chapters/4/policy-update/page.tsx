'use client';

import React from 'react';
import { PolicyUpdateSimulator } from '@/components/interactive/PolicyUpdate';
import { motion } from 'framer-motion';
import { Shield, Zap, TrendingUp, AlertCircle, Code } from 'lucide-react';

export default function PolicyUpdateDemo() {
  return (
    <div className="min-h-screen bg-gray-50 py-8">
      <div className="max-w-7xl mx-auto px-4">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="mb-8"
        >
          <h1 className="text-4xl font-bold mb-4">Chapter 4: PPO Policy Updates</h1>
          <p className="text-xl text-gray-600">
            The heart of PPO: Clipped objective function for stable policy improvement
          </p>
        </motion.div>

        <div className="grid lg:grid-cols-4 gap-8">
          {/* Main Simulator */}
          <div className="lg:col-span-3">
            <motion.div
              initial={{ opacity: 0, scale: 0.95 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ delay: 0.1 }}
              className="bg-white rounded-lg shadow-lg p-6"
            >
              <PolicyUpdateSimulator />
            </motion.div>
          </div>

          {/* Learning Panel */}
          <div className="space-y-6">
            {/* PPO Objective */}
            <motion.div
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: 0.2 }}
              className="bg-white rounded-lg shadow-lg p-6"
            >
              <h3 className="text-xl font-bold mb-4 flex items-center gap-2">
                <Shield className="w-5 h-5" />
                PPO Objective
              </h3>
              
              <div className="space-y-3 text-sm">
                <div className="bg-gray-50 p-3 rounded font-mono text-xs">
                  L<sup>CLIP</sup>(θ) = E[min(r<sub>t</sub>(θ)Â<sub>t</sub>, clip(r<sub>t</sub>(θ), 1-ε, 1+ε)Â<sub>t</sub>)]
                </div>
                
                <div className="space-y-2">
                  <p><strong>r<sub>t</sub>(θ)</strong> = π<sub>θ</sub>(a|s) / π<sub>θ_old</sub>(a|s)</p>
                  <p className="text-xs text-gray-600">Probability ratio</p>
                </div>
                
                <div className="space-y-2">
                  <p><strong>Â<sub>t</sub></strong> = Normalized advantages</p>
                  <p className="text-xs text-gray-600">How good was the action?</p>
                </div>
                
                <div className="space-y-2">
                  <p><strong>ε</strong> = Clip range (typically 0.2)</p>
                  <p className="text-xs text-gray-600">Limits policy change</p>
                </div>
              </div>
            </motion.div>

            {/* Why Clipping? */}
            <motion.div
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: 0.3 }}
              className="bg-gradient-to-br from-green-50 to-blue-50 rounded-lg shadow-lg p-6"
            >
              <h3 className="text-xl font-bold mb-4 flex items-center gap-2">
                <Zap className="w-5 h-5" />
                Why Clipping Works
              </h3>
              
              <div className="space-y-3 text-sm">
                <div>
                  <strong className="text-blue-700">Prevents Catastrophic Updates</strong>
                  <p className="text-gray-600">
                    Limits how much policy can change in one step
                  </p>
                </div>
                
                <div>
                  <strong className="text-blue-700">Conservative When Uncertain</strong>
                  <p className="text-gray-600">
                    Large advantages don't cause huge updates
                  </p>
                </div>
                
                <div>
                  <strong className="text-blue-700">Automatic Trust Region</strong>
                  <p className="text-gray-600">
                    No need for complex KL constraints
                  </p>
                </div>
                
                <div>
                  <strong className="text-blue-700">Symmetric Clipping</strong>
                  <p className="text-gray-600">
                    Works for both positive and negative advantages
                  </p>
                </div>
              </div>
            </motion.div>

            {/* Key Insights */}
            <motion.div
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: 0.4 }}
              className="bg-purple-50 rounded-lg shadow-lg p-6"
            >
              <h3 className="text-xl font-bold mb-4 flex items-center gap-2">
                <TrendingUp className="w-5 h-5" />
                Key Insights
              </h3>
              
              <ul className="space-y-2 text-sm">
                <li className="flex items-start gap-2">
                  <span className="text-purple-600">•</span>
                  Clip fraction ~0.1-0.3 is healthy
                </li>
                <li className="flex items-start gap-2">
                  <span className="text-purple-600">•</span>
                  Too much clipping = learning too fast
                </li>
                <li className="flex items-start gap-2">
                  <span className="text-purple-600">•</span>
                  No clipping = learning too slow
                </li>
                <li className="flex items-start gap-2">
                  <span className="text-purple-600">•</span>
                  KL divergence tracks policy drift
                </li>
                <li className="flex items-start gap-2">
                  <span className="text-purple-600">•</span>
                  Early stopping prevents instability
                </li>
              </ul>
            </motion.div>

            {/* Common Issues */}
            <motion.div
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: 0.5 }}
              className="bg-red-50 rounded-lg shadow-lg p-6"
            >
              <h3 className="text-xl font-bold mb-4 flex items-center gap-2">
                <AlertCircle className="w-5 h-5 text-red-600" />
                Watch Out For
              </h3>
              
              <ul className="space-y-2 text-sm text-gray-700">
                <li className="flex items-start gap-2">
                  <span className="text-red-600">⚠</span>
                  ε too small → slow learning
                </li>
                <li className="flex items-start gap-2">
                  <span className="text-red-600">⚠</span>
                  ε too large → unstable
                </li>
                <li className="flex items-start gap-2">
                  <span className="text-red-600">⚠</span>
                  High KL → policy collapse
                </li>
                <li className="flex items-start gap-2">
                  <span className="text-red-600">⚠</span>
                  Low entropy → exploration dies
                </li>
              </ul>
            </motion.div>
          </div>
        </div>

        {/* Interactive Experiments */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.6 }}
          className="mt-8 bg-yellow-50 rounded-lg p-6 border-2 border-yellow-200"
        >
          <h3 className="text-xl font-bold mb-4">🧪 Experiments to Try</h3>
          
          <div className="grid md:grid-cols-2 gap-6">
            <div>
              <h4 className="font-semibold mb-2">Experiment 1: Clip Range Impact</h4>
              <ol className="text-sm space-y-1">
                <li>1. Set ε = 0.05 (very conservative)</li>
                <li>2. Run training and note convergence speed</li>
                <li>3. Set ε = 0.4 (very aggressive)</li>
                <li>4. Compare stability and clip fraction</li>
              </ol>
            </div>
            
            <div>
              <h4 className="font-semibold mb-2">Experiment 2: Learning Rate Effects</h4>
              <ol className="text-sm space-y-1">
                <li>1. Use default ε = 0.2</li>
                <li>2. Try very low learning rate (0.0001)</li>
                <li>3. Try high learning rate (0.001)</li>
                <li>4. Watch policy distribution changes</li>
              </ol>
            </div>
            
            <div>
              <h4 className="font-semibold mb-2">Experiment 3: Early Stopping</h4>
              <ol className="text-sm space-y-1">
                <li>1. Enable KL divergence display</li>
                <li>2. Turn on early stopping</li>
                <li>3. Use aggressive settings</li>
                <li>4. See when training auto-stops</li>
              </ol>
            </div>
            
            <div>
              <h4 className="font-semibold mb-2">Challenge Question</h4>
              <p className="text-sm text-gray-700">
                Why does PPO clip the objective instead of the gradient or the policy parameters directly?
                Think about what happens during backpropagation.
              </p>
            </div>
          </div>
        </motion.div>

        {/* Implementation Notes */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.7 }}
          className="mt-8 bg-gray-900 rounded-lg p-6 text-white"
        >
          <h3 className="text-xl font-bold mb-4 flex items-center gap-2">
            <Code className="w-5 h-5" />
            Implementation Tips
          </h3>
          
          <div className="grid md:grid-cols-2 gap-6">
            <div>
              <h4 className="font-semibold mb-2">Hyperparameters</h4>
              <pre className="text-xs bg-gray-800 p-3 rounded overflow-x-auto">
{`# Common PPO hyperparameters
clip_range = 0.2
learning_rate = 3e-4
n_epochs = 10
batch_size = 64
gae_lambda = 0.95
value_coef = 0.5
entropy_coef = 0.01`}
              </pre>
            </div>
            
            <div>
              <h4 className="font-semibold mb-2">Multiple Epochs</h4>
              <pre className="text-xs bg-gray-800 p-3 rounded overflow-x-auto">
{`# PPO reuses data multiple times
for epoch in range(n_epochs):
    for batch in minibatches:
        # Calculate ratio and advantages
        ratio = new_prob / old_prob
        
        # Clip and optimize
        loss = clip_loss + value_loss + entropy
        optimizer.step()`}
              </pre>
            </div>
          </div>
        </motion.div>
      </div>
    </div>
  );
}