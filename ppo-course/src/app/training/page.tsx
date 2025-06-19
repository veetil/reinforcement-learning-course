'use client';

import React from 'react';
import { TrainingDashboard } from '@/components/training/TrainingDashboard';
import { motion } from 'framer-motion';
import { Rocket, Code, Monitor, Server } from 'lucide-react';

export default function TrainingPage() {
  return (
    <div className="min-h-screen bg-gray-50 py-8">
      <div className="max-w-7xl mx-auto px-4">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="mb-8"
        >
          <h1 className="text-4xl font-bold mb-4">Live PPO Training</h1>
          <p className="text-xl text-gray-600">
            Train and monitor PPO agents in real-time with interactive controls
          </p>
        </motion.div>

        {/* Info Cards */}
        <div className="grid md:grid-cols-4 gap-4 mb-8">
          <motion.div
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ delay: 0.1 }}
            className="bg-white rounded-lg shadow p-4"
          >
            <div className="flex items-center gap-3 mb-2">
              <Rocket className="w-8 h-8 text-blue-500" />
              <h3 className="font-semibold">Real-time Training</h3>
            </div>
            <p className="text-sm text-gray-600">
              Watch your agent learn with live metric updates
            </p>
          </motion.div>

          <motion.div
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ delay: 0.2 }}
            className="bg-white rounded-lg shadow p-4"
          >
            <div className="flex items-center gap-3 mb-2">
              <Monitor className="w-8 h-8 text-green-500" />
              <h3 className="font-semibold">Interactive Charts</h3>
            </div>
            <p className="text-sm text-gray-600">
              Visualize rewards, losses, and key metrics
            </p>
          </motion.div>

          <motion.div
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ delay: 0.3 }}
            className="bg-white rounded-lg shadow p-4"
          >
            <div className="flex items-center gap-3 mb-2">
              <Code className="w-8 h-8 text-purple-500" />
              <h3 className="font-semibold">Customizable</h3>
            </div>
            <p className="text-sm text-gray-600">
              Adjust hyperparameters on the fly
            </p>
          </motion.div>

          <motion.div
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ delay: 0.4 }}
            className="bg-white rounded-lg shadow p-4"
          >
            <div className="flex items-center gap-3 mb-2">
              <Server className="w-8 h-8 text-orange-500" />
              <h3 className="font-semibold">WebSocket Updates</h3>
            </div>
            <p className="text-sm text-gray-600">
              Low-latency streaming of training data
            </p>
          </motion.div>
        </div>

        {/* Training Dashboard */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.5 }}
        >
          <TrainingDashboard />
        </motion.div>

        {/* Tips Section */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.6 }}
          className="mt-8 bg-blue-50 rounded-lg p-6 border-2 border-blue-200"
        >
          <h3 className="text-xl font-bold mb-4">💡 Training Tips</h3>
          
          <div className="grid md:grid-cols-2 gap-6">
            <div>
              <h4 className="font-semibold mb-2">Getting Started</h4>
              <ul className="text-sm space-y-1 text-gray-700">
                <li>• Start with default hyperparameters</li>
                <li>• Monitor reward progress closely</li>
                <li>• Watch for stable clip fraction (10-30%)</li>
                <li>• Stop if KL divergence spikes</li>
              </ul>
            </div>
            
            <div>
              <h4 className="font-semibold mb-2">Troubleshooting</h4>
              <ul className="text-sm space-y-1 text-gray-700">
                <li>• Flat rewards? Try increasing learning rate</li>
                <li>• Unstable training? Reduce clip range</li>
                <li>• High KL? Lower learning rate or epochs</li>
                <li>• Low entropy? Increase entropy coefficient</li>
              </ul>
            </div>
          </div>
        </motion.div>

        {/* Backend Status */}
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.7 }}
          className="mt-4 text-center text-sm text-gray-500"
        >
          <p>
            Backend API: {process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'}
          </p>
          <p className="mt-1">
            Make sure the backend is running: <code className="bg-gray-100 px-2 py-1 rounded">cd backend && uvicorn main:app --reload</code>
          </p>
        </motion.div>
      </div>
    </div>
  );
}