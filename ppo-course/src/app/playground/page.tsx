'use client';

import React from 'react';
import { CodePlayground } from '@/components/playground/CodePlayground';
import { motion } from 'framer-motion';
import { Code2, BookOpen, Lightbulb, Zap } from 'lucide-react';

export default function PlaygroundPage() {
  return (
    <div className="min-h-screen bg-gray-50">
      <div className="max-w-7xl mx-auto px-4 py-8">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="mb-8"
        >
          <h1 className="text-4xl font-bold mb-4">Code Playground</h1>
          <p className="text-xl text-gray-600">
            Experiment with PPO implementations and visualizations
          </p>
        </motion.div>

        {/* Feature Cards */}
        <div className="grid md:grid-cols-4 gap-4 mb-8">
          <motion.div
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ delay: 0.1 }}
            className="bg-white rounded-lg shadow p-4"
          >
            <div className="flex items-center gap-3 mb-2">
              <Code2 className="w-6 h-6 text-blue-500" />
              <h3 className="font-semibold">Live Coding</h3>
            </div>
            <p className="text-sm text-gray-600">
              Edit and run Python code directly in your browser
            </p>
          </motion.div>

          <motion.div
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ delay: 0.2 }}
            className="bg-white rounded-lg shadow p-4"
          >
            <div className="flex items-center gap-3 mb-2">
              <BookOpen className="w-6 h-6 text-green-500" />
              <h3 className="font-semibold">Templates</h3>
            </div>
            <p className="text-sm text-gray-600">
              Pre-built examples covering key PPO concepts
            </p>
          </motion.div>

          <motion.div
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ delay: 0.3 }}
            className="bg-white rounded-lg shadow p-4"
          >
            <div className="flex items-center gap-3 mb-2">
              <Lightbulb className="w-6 h-6 text-yellow-500" />
              <h3 className="font-semibold">Learn by Doing</h3>
            </div>
            <p className="text-sm text-gray-600">
              Modify code and see immediate results
            </p>
          </motion.div>

          <motion.div
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ delay: 0.4 }}
            className="bg-white rounded-lg shadow p-4"
          >
            <div className="flex items-center gap-3 mb-2">
              <Zap className="w-6 h-6 text-purple-500" />
              <h3 className="font-semibold">Safe Execution</h3>
            </div>
            <p className="text-sm text-gray-600">
              Sandboxed environment for safe experimentation
            </p>
          </motion.div>
        </div>

        {/* Playground */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.5 }}
          className="bg-white rounded-lg shadow-lg overflow-hidden"
          style={{ height: '600px' }}
        >
          <CodePlayground />
        </motion.div>

        {/* Learning Exercises */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.6 }}
          className="mt-8 bg-indigo-50 rounded-lg p-6 border-2 border-indigo-200"
        >
          <h3 className="text-xl font-bold mb-4">📚 Learning Exercises</h3>
          
          <div className="grid md:grid-cols-2 gap-6">
            <div>
              <h4 className="font-semibold mb-2">Exercise 1: Modify Hyperparameters</h4>
              <ol className="text-sm space-y-1 text-gray-700">
                <li>1. Load the "PPO Basic Implementation" template</li>
                <li>2. Change the learning rate to 1e-3</li>
                <li>3. Modify epsilon (clip range) to 0.3</li>
                <li>4. Add print statements to track the changes</li>
              </ol>
            </div>
            
            <div>
              <h4 className="font-semibold mb-2">Exercise 2: Implement Custom Network</h4>
              <ol className="text-sm space-y-1 text-gray-700">
                <li>1. Start with "Actor-Critic Network" template</li>
                <li>2. Add a third hidden layer</li>
                <li>3. Implement layer normalization</li>
                <li>4. Test with different state dimensions</li>
              </ol>
            </div>
            
            <div>
              <h4 className="font-semibold mb-2">Exercise 3: GAE Analysis</h4>
              <ol className="text-sm space-y-1 text-gray-700">
                <li>1. Use the "GAE Calculation" template</li>
                <li>2. Try different lambda values (0, 0.5, 1)</li>
                <li>3. Compare the resulting advantages</li>
                <li>4. Plot advantages vs timesteps</li>
              </ol>
            </div>
            
            <div>
              <h4 className="font-semibold mb-2">Exercise 4: Visualize Clipping</h4>
              <ol className="text-sm space-y-1 text-gray-700">
                <li>1. Run the "PPO Clipping Visualization"</li>
                <li>2. Experiment with different epsilon values</li>
                <li>3. Add more advantage values to test</li>
                <li>4. Create a 3D visualization if possible</li>
              </ol>
            </div>
          </div>
        </motion.div>

        {/* Tips */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.7 }}
          className="mt-8 bg-yellow-50 rounded-lg p-6 border-2 border-yellow-200"
        >
          <h3 className="text-xl font-bold mb-4">💡 Pro Tips</h3>
          
          <ul className="space-y-2 text-sm text-gray-700">
            <li className="flex items-start gap-2">
              <span className="text-yellow-600">•</span>
              <span>Use print statements liberally to understand the flow</span>
            </li>
            <li className="flex items-start gap-2">
              <span className="text-yellow-600">•</span>
              <span>Start with small modifications before major changes</span>
            </li>
            <li className="flex items-start gap-2">
              <span className="text-yellow-600">•</span>
              <span>Compare outputs with different hyperparameters</span>
            </li>
            <li className="flex items-start gap-2">
              <span className="text-yellow-600">•</span>
              <span>Save interesting code snippets for future reference</span>
            </li>
            <li className="flex items-start gap-2">
              <span className="text-yellow-600">•</span>
              <span>Combine concepts from different templates</span>
            </li>
          </ul>
        </motion.div>
      </div>
    </div>
  );
}