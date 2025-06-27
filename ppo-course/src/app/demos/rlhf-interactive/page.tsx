'use client';

import { RLHFInteractiveDemo } from '@/components/interactive/RLHFInteractiveDemo';

export default function RLHFInteractivePage() {
  return (
    <div className="min-h-screen bg-gray-50 py-12">
      <div className="container mx-auto px-4">
        <div className="text-center mb-8">
          <h1 className="text-4xl font-bold mb-4">Interactive RLHF Tutorial</h1>
          <p className="text-xl text-gray-600 max-w-3xl mx-auto">
            Learn how Reinforcement Learning from Human Feedback works by building a reward model step by step. 
            Fill in the matrices, validate your understanding, and see the math come to life!
          </p>
        </div>
        
        <RLHFInteractiveDemo />
        
        <div className="mt-12 max-w-4xl mx-auto">
          <div className="bg-white rounded-lg shadow-lg p-6">
            <h2 className="text-2xl font-bold mb-4">What you'll learn:</h2>
            <ul className="space-y-3">
              <li className="flex items-start">
                <span className="text-green-500 mr-2">✓</span>
                <span>How human preferences are collected and represented mathematically</span>
              </li>
              <li className="flex items-start">
                <span className="text-green-500 mr-2">✓</span>
                <span>The Bradley-Terry model for converting comparisons into reward values</span>
              </li>
              <li className="flex items-start">
                <span className="text-green-500 mr-2">✓</span>
                <span>How to build and validate a preference matrix</span>
              </li>
              <li className="flex items-start">
                <span className="text-green-500 mr-2">✓</span>
                <span>The mathematical foundation behind RLHF in modern AI systems</span>
              </li>
            </ul>
          </div>
          
          <div className="bg-blue-50 rounded-lg p-6 mt-6">
            <h3 className="text-xl font-semibold mb-3">🎯 Learning Tips:</h3>
            <ul className="space-y-2 text-gray-700">
              <li>• Try to solve each step yourself before using hints</li>
              <li>• If you get stuck after 2 attempts, the solution button will appear</li>
              <li>• Pay attention to the insights - they explain the "why" behind each calculation</li>
              <li>• The green checkmarks show correct answers, red alerts show mistakes</li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  );
}