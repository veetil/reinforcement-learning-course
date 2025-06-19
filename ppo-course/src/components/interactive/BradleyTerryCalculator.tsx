'use client';

import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { AlertCircle, CheckCircle, TrendingUp, TrendingDown } from 'lucide-react';

interface PreferencePair {
  id: string;
  prompt: string;
  chosenResponse: string;
  rejectedResponse: string;
  chosenReward?: number;
  rejectedReward?: number;
  probability?: number;
  loss?: number;
}

const samplePairs: PreferencePair[] = [
  {
    id: '1',
    prompt: 'How can I improve my productivity?',
    chosenResponse: 'To improve productivity: 1) Use time-blocking for focused work, 2) Eliminate distractions, 3) Take regular breaks, 4) Prioritize important tasks.',
    rejectedResponse: 'Just work harder.',
    chosenReward: 0.8,
    rejectedReward: 0.2,
  },
  {
    id: '2',
    prompt: 'Explain quantum computing',
    chosenResponse: 'Quantum computing uses quantum bits (qubits) that can exist in superposition, allowing them to process multiple states simultaneously, unlike classical bits that are either 0 or 1.',
    rejectedResponse: 'Quantum computing is when computers use quantum stuff to compute things quantumly.',
    chosenReward: 0.9,
    rejectedReward: 0.3,
  },
  {
    id: '3',
    prompt: 'What is the meaning of life?',
    chosenResponse: 'The meaning of life is subjective and varies by individual. Many find meaning through relationships, personal growth, contributing to society, and pursuing passions.',
    rejectedResponse: '42',
    chosenReward: 0.7,
    rejectedReward: 0.4,
  },
];

export const BradleyTerryCalculator: React.FC = () => {
  const [pairs, setPairs] = useState<PreferencePair[]>(samplePairs);
  const [selectedPair, setSelectedPair] = useState<PreferencePair>(samplePairs[0]);
  const [chosenReward, setChosenReward] = useState(0.8);
  const [rejectedReward, setRejectedReward] = useState(0.2);
  const [showMath, setShowMath] = useState(false);
  const [trainingMode, setTrainingMode] = useState(false);
  const [epoch, setEpoch] = useState(0);

  // Calculate Bradley-Terry probability
  const calculateProbability = (rChosen: number, rRejected: number) => {
    const diff = rChosen - rRejected;
    return 1 / (1 + Math.exp(-diff));
  };

  // Calculate loss
  const calculateLoss = (probability: number) => {
    return -Math.log(Math.max(probability, 1e-7));
  };

  // Update calculations when rewards change
  useEffect(() => {
    const probability = calculateProbability(chosenReward, rejectedReward);
    const loss = calculateLoss(probability);
    
    setSelectedPair(prev => ({
      ...prev,
      chosenReward,
      rejectedReward,
      probability,
      loss,
    }));
  }, [chosenReward, rejectedReward]);

  // Simulate training
  useEffect(() => {
    if (!trainingMode) return;

    const timer = setInterval(() => {
      setEpoch(prev => prev + 1);
      
      // Simulate gradient descent
      setPairs(prevPairs => 
        prevPairs.map(pair => {
          const currentProb = calculateProbability(
            pair.chosenReward || 0.5,
            pair.rejectedReward || 0.5
          );
          
          // Simple gradient update simulation
          const learningRate = 0.1;
          const gradient = 1 - currentProb;
          
          return {
            ...pair,
            chosenReward: (pair.chosenReward || 0.5) + learningRate * gradient,
            rejectedReward: (pair.rejectedReward || 0.5) - learningRate * gradient * 0.5,
            probability: calculateProbability(
              (pair.chosenReward || 0.5) + learningRate * gradient,
              (pair.rejectedReward || 0.5) - learningRate * gradient * 0.5
            ),
          };
        })
      );
    }, 1000);

    return () => clearInterval(timer);
  }, [trainingMode]);

  const resetTraining = () => {
    setTrainingMode(false);
    setEpoch(0);
    setPairs(samplePairs);
  };

  return (
    <div className="w-full max-w-4xl mx-auto p-6 space-y-6">
      <div className="bg-white rounded-lg shadow-lg p-6">
        <h2 className="text-2xl font-bold mb-4">Bradley-Terry Model Calculator</h2>
        
        {/* Preference Pair Selection */}
        <div className="mb-6">
          <h3 className="font-medium mb-3">Select Preference Pair</h3>
          <div className="space-y-2">
            {pairs.map((pair, idx) => (
              <button
                key={pair.id}
                onClick={() => {
                  setSelectedPair(pair);
                  setChosenReward(pair.chosenReward || 0.5);
                  setRejectedReward(pair.rejectedReward || 0.5);
                }}
                className={`w-full text-left p-3 rounded-lg border transition-all ${
                  selectedPair.id === pair.id
                    ? 'border-blue-500 bg-blue-50'
                    : 'border-gray-200 hover:border-gray-300'
                }`}
              >
                <div className="font-medium">Pair {idx + 1}: {pair.prompt}</div>
                {trainingMode && pair.probability && (
                  <div className="text-sm text-gray-600 mt-1">
                    P(chosen > rejected) = {pair.probability.toFixed(3)}
                  </div>
                )}
              </button>
            ))}
          </div>
        </div>

        {/* Reward Adjustment */}
        <div className="grid md:grid-cols-2 gap-6 mb-6">
          <div>
            <label className="block font-medium mb-2">
              Chosen Response Reward: {chosenReward.toFixed(2)}
            </label>
            <input
              type="range"
              min="-2"
              max="2"
              step="0.1"
              value={chosenReward}
              onChange={(e) => setChosenReward(parseFloat(e.target.value))}
              className="w-full"
              disabled={trainingMode}
            />
            <div className="mt-2 p-3 bg-green-50 rounded text-sm">
              <div className="font-medium text-green-700">Chosen Response:</div>
              <div className="text-gray-700 line-clamp-2">{selectedPair.chosenResponse}</div>
            </div>
          </div>
          
          <div>
            <label className="block font-medium mb-2">
              Rejected Response Reward: {rejectedReward.toFixed(2)}
            </label>
            <input
              type="range"
              min="-2"
              max="2"
              step="0.1"
              value={rejectedReward}
              onChange={(e) => setRejectedReward(parseFloat(e.target.value))}
              className="w-full"
              disabled={trainingMode}
            />
            <div className="mt-2 p-3 bg-red-50 rounded text-sm">
              <div className="font-medium text-red-700">Rejected Response:</div>
              <div className="text-gray-700 line-clamp-2">{selectedPair.rejectedResponse}</div>
            </div>
          </div>
        </div>

        {/* Probability Visualization */}
        <div className="bg-gray-50 rounded-lg p-6">
          <h3 className="font-medium mb-4">Bradley-Terry Probability</h3>
          
          <div className="mb-4">
            <div className="text-3xl font-bold text-center">
              P(chosen > rejected) = {(selectedPair.probability || 0).toFixed(3)}
            </div>
            
            {/* Probability Bar */}
            <div className="mt-4 relative h-8 bg-gray-200 rounded-full overflow-hidden">
              <motion.div
                className="absolute left-0 top-0 h-full bg-gradient-to-r from-red-500 to-green-500"
                initial={{ width: '50%' }}
                animate={{ width: `${(selectedPair.probability || 0.5) * 100}%` }}
                transition={{ duration: 0.5 }}
              />
              <div className="absolute inset-0 flex items-center justify-center text-white font-medium">
                {((selectedPair.probability || 0.5) * 100).toFixed(1)}%
              </div>
            </div>
          </div>
          
          {/* Mathematical Explanation */}
          <button
            onClick={() => setShowMath(!showMath)}
            className="text-blue-600 hover:text-blue-700 text-sm font-medium"
          >
            {showMath ? 'Hide' : 'Show'} Mathematical Details
          </button>
          
          {showMath && (
            <motion.div
              initial={{ opacity: 0, height: 0 }}
              animate={{ opacity: 1, height: 'auto' }}
              className="mt-4 space-y-3 text-sm"
            >
              <div className="p-3 bg-white rounded border">
                <div className="font-mono">
                  P(A > B) = exp(r_A) / (exp(r_A) + exp(r_B))
                </div>
                <div className="font-mono mt-2">
                  = 1 / (1 + exp(r_B - r_A))
                </div>
                <div className="font-mono mt-2">
                  = sigmoid(r_A - r_B)
                </div>
              </div>
              
              <div className="p-3 bg-white rounded border">
                <div className="font-medium mb-1">With current values:</div>
                <div className="font-mono text-xs">
                  r_chosen = {chosenReward.toFixed(2)}
                </div>
                <div className="font-mono text-xs">
                  r_rejected = {rejectedReward.toFixed(2)}
                </div>
                <div className="font-mono text-xs">
                  r_chosen - r_rejected = {(chosenReward - rejectedReward).toFixed(2)}
                </div>
                <div className="font-mono text-xs">
                  P = sigmoid({(chosenReward - rejectedReward).toFixed(2)}) = {(selectedPair.probability || 0).toFixed(3)}
                </div>
              </div>
              
              <div className="p-3 bg-white rounded border">
                <div className="font-medium mb-1">Bradley-Terry Loss:</div>
                <div className="font-mono text-xs">
                  L = -log(P(chosen > rejected))
                </div>
                <div className="font-mono text-xs">
                  = -log({(selectedPair.probability || 0).toFixed(3)})
                </div>
                <div className="font-mono text-xs">
                  = {(selectedPair.loss || 0).toFixed(3)}
                </div>
              </div>
            </motion.div>
          )}
        </div>

        {/* Training Simulation */}
        <div className="mt-6">
          <div className="flex items-center justify-between mb-4">
            <h3 className="font-medium">Training Simulation</h3>
            {trainingMode && (
              <div className="text-sm text-gray-600">
                Epoch: {epoch}
              </div>
            )}
          </div>
          
          <div className="flex gap-3">
            <button
              onClick={() => setTrainingMode(!trainingMode)}
              className={`px-4 py-2 rounded-lg font-medium transition-colors ${
                trainingMode
                  ? 'bg-red-500 hover:bg-red-600 text-white'
                  : 'bg-green-500 hover:bg-green-600 text-white'
              }`}
            >
              {trainingMode ? 'Stop Training' : 'Start Training'}
            </button>
            
            <button
              onClick={resetTraining}
              className="px-4 py-2 bg-gray-500 hover:bg-gray-600 text-white rounded-lg font-medium transition-colors"
            >
              Reset
            </button>
          </div>
          
          {trainingMode && (
            <div className="mt-4 p-4 bg-blue-50 rounded-lg">
              <div className="text-sm text-blue-700">
                <AlertCircle className="inline w-4 h-4 mr-1" />
                The reward model is learning to assign higher rewards to chosen responses
                and lower rewards to rejected responses.
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Key Insights */}
      <div className="bg-yellow-50 rounded-lg p-6 border-2 border-yellow-200">
        <h3 className="font-bold mb-3 flex items-center">
          <AlertCircle className="w-5 h-5 mr-2" />
          Key Insights
        </h3>
        <ul className="space-y-2 text-sm">
          <li>
            • The Bradley-Terry model converts reward differences into probabilities
          </li>
          <li>
            • Larger reward differences lead to more confident predictions
          </li>
          <li>
            • The loss function encourages the model to predict P(chosen > rejected) ≈ 1
          </li>
          <li>
            • Without this loss, VERL's reward model cannot learn preferences!
          </li>
        </ul>
      </div>
    </div>
  );
};