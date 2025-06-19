'use client';

import React, { useState } from 'react';
import { GridWorld } from '@/components/interactive/GridWorld';
import { GridState } from '@/components/interactive/GridWorld';
import { motion } from 'framer-motion';
import { Settings, Eye, Route, Gift } from 'lucide-react';

export default function GridWorldDemo() {
  const [showPolicy, setShowPolicy] = useState(false);
  const [showTrajectory, setShowTrajectory] = useState(true);
  const [showRewards, setShowRewards] = useState(false);
  const [lastState, setLastState] = useState<GridState | null>(null);

  return (
    <div className="min-h-screen bg-gray-50 py-8">
      <div className="max-w-7xl mx-auto px-4">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
        >
          <div className="mb-8">
            <h1 className="text-4xl font-bold mb-4">Chapter 1: Grid World Explorer</h1>
            <p className="text-xl text-gray-600">
              Learn reinforcement learning basics through interactive exploration
            </p>
          </div>
        </motion.div>

        <div className="grid lg:grid-cols-3 gap-8">
          {/* Main Grid World */}
          <div className="lg:col-span-2">
            <motion.div
              initial={{ opacity: 0, scale: 0.95 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ delay: 0.1 }}
            >
              <div className="bg-white rounded-lg shadow-lg p-6">
              <h2 className="text-2xl font-bold mb-4">Interactive Environment</h2>
              
              {/* Visualization Options */}
              <div className="mb-4 flex flex-wrap gap-2">
                <button
                  onClick={() => setShowPolicy(!showPolicy)}
                  className={`px-3 py-1 rounded-lg flex items-center gap-2 transition-colors ${
                    showPolicy 
                      ? 'bg-blue-600 text-white' 
                      : 'bg-gray-200 text-gray-700 hover:bg-gray-300'
                  }`}
                >
                  <Eye className="w-4 h-4" />
                  Policy
                </button>
                
                <button
                  onClick={() => setShowTrajectory(!showTrajectory)}
                  className={`px-3 py-1 rounded-lg flex items-center gap-2 transition-colors ${
                    showTrajectory 
                      ? 'bg-blue-600 text-white' 
                      : 'bg-gray-200 text-gray-700 hover:bg-gray-300'
                  }`}
                >
                  <Route className="w-4 h-4" />
                  Trajectory
                </button>
                
                <button
                  onClick={() => setShowRewards(!showRewards)}
                  className={`px-3 py-1 rounded-lg flex items-center gap-2 transition-colors ${
                    showRewards 
                      ? 'bg-blue-600 text-white' 
                      : 'bg-gray-200 text-gray-700 hover:bg-gray-300'
                  }`}
                >
                  <Gift className="w-4 h-4" />
                  Rewards
                </button>
              </div>
              
              <GridWorld
                width={6}
                height={6}
                showPolicy={showPolicy}
                showTrajectory={showTrajectory}
                showRewards={showRewards}
                onStateChange={setLastState}
              />
              </div>
            </motion.div>
          </div>

          {/* Learning Panel */}
          <div className="space-y-6">
            {/* Key Concepts */}
            <motion.div
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: 0.2 }}
            >
              <div className="bg-white rounded-lg shadow-lg p-6">
              <h3 className="text-xl font-bold mb-4">Key Concepts</h3>
              
              <div className="space-y-4">
                <div>
                  <h4 className="font-semibold text-blue-600">State</h4>
                  <p className="text-sm text-gray-600">
                    Your current position in the grid. In RL, the state contains all 
                    information needed to make decisions.
                  </p>
                </div>
                
                <div>
                  <h4 className="font-semibold text-green-600">Action</h4>
                  <p className="text-sm text-gray-600">
                    The moves you can make (up, down, left, right). Actions transition 
                    you between states.
                  </p>
                </div>
                
                <div>
                  <h4 className="font-semibold text-purple-600">Reward</h4>
                  <p className="text-sm text-gray-600">
                    Feedback from the environment. Negative for each step, positive for 
                    reaching the goal, penalty for obstacles.
                  </p>
                </div>
                
                <div>
                  <h4 className="font-semibold text-orange-600">Policy</h4>
                  <p className="text-sm text-gray-600">
                    A strategy that maps states to actions. The arrows show a simple 
                    policy that moves toward the goal.
                  </p>
                </div>
              </div>
              </div>
            </motion.div>

            {/* Challenge */}
            <motion.div
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: 0.3 }}
            >
              <div className="bg-gradient-to-br from-purple-50 to-blue-50 rounded-lg shadow-lg p-6">
              <h3 className="text-xl font-bold mb-4">🎯 Challenge</h3>
              
              <div className="space-y-3">
                <div className="flex items-center gap-2">
                  <input type="checkbox" id="challenge1" className="rounded" />
                  <label htmlFor="challenge1" className="text-sm">
                    Reach the goal in under 10 steps
                  </label>
                </div>
                
                <div className="flex items-center gap-2">
                  <input type="checkbox" id="challenge2" className="rounded" />
                  <label htmlFor="challenge2" className="text-sm">
                    Find the optimal path with obstacles
                  </label>
                </div>
                
                <div className="flex items-center gap-2">
                  <input type="checkbox" id="challenge3" className="rounded" />
                  <label htmlFor="challenge3" className="text-sm">
                    Achieve a total reward greater than 0
                  </label>
                </div>
              </div>
              </div>
            </motion.div>

            {/* State Info */}
            {lastState && (
              <motion.div
                initial={{ opacity: 0, x: 20 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: 0.4 }}
              >
                <div className="bg-white rounded-lg shadow-lg p-6">
                <h3 className="text-xl font-bold mb-4">Current State Info</h3>
                
                <pre className="text-xs bg-gray-100 p-3 rounded overflow-auto">
                  {JSON.stringify({
                    position: lastState.agentPosition,
                    reward: lastState.totalReward,
                    steps: lastState.stepCount,
                    done: lastState.isDone,
                  }, null, 2)}
                </pre>
                </div>
              </motion.div>
            )}
          </div>
        </div>

        {/* Learning Questions */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.5 }}
        >
          <div className="mt-8 bg-yellow-50 rounded-lg p-6 border-2 border-yellow-200">
          <h3 className="text-xl font-bold mb-4">🤔 Think About It</h3>
          
          <div className="grid md:grid-cols-2 gap-4">
            <div>
              <p className="text-sm font-medium mb-2">
                1. How does the reward structure influence the agent's behavior?
              </p>
              <p className="text-xs text-gray-600">
                Try changing the step penalty or goal reward and observe the difference.
              </p>
            </div>
            
            <div>
              <p className="text-sm font-medium mb-2">
                2. What makes a policy "optimal"?
              </p>
              <p className="text-xs text-gray-600">
                Is the shortest path always the best when considering rewards?
              </p>
            </div>
            
            <div>
              <p className="text-sm font-medium mb-2">
                3. How would you handle partially observable states?
              </p>
              <p className="text-xs text-gray-600">
                What if the agent couldn't see the entire grid?
              </p>
            </div>
            
            <div>
              <p className="text-sm font-medium mb-2">
                4. How does this relate to real-world problems?
              </p>
              <p className="text-xs text-gray-600">
                Think of examples where similar state-action-reward patterns exist.
              </p>
            </div>
          </div>
          </div>
        </motion.div>
      </div>
    </div>
  );
}