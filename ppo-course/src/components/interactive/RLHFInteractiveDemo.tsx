'use client';

import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { ChevronRight, ChevronLeft, HelpCircle, CheckCircle, AlertCircle, RefreshCw, Eye } from 'lucide-react';

interface MatrixCell {
  value: number | null;
  isCorrect?: boolean;
  isHighlighted?: boolean;
}

interface Step {
  id: number;
  title: string;
  description: string;
  insight: string;
  matrix?: MatrixCell[][];
  expectedValues?: number[][];
  inputPositions?: [number, number][];
  calculation?: string;
  showComparison?: boolean;
  comparisonPairs?: { a: string; b: string; winner: 'a' | 'b' }[];
}

const steps: Step[] = [
  {
    id: 1,
    title: "Collecting Preference Data",
    description: "We have 3 response options (A, B, C) and collected human preferences from 5 pairwise comparisons. Each comparison shows which option humans preferred.",
    insight: "Humans compare responses in pairs. We aggregate multiple comparisons to understand overall preferences. Here we see 5 total comparisons between our 3 options.",
    showComparison: true,
    comparisonPairs: [
      // A vs B: 5 comparisons (A wins 3, B wins 2)
      { a: "Option A", b: "Option B", winner: 'a' },
      { a: "Option A", b: "Option B", winner: 'a' },
      { a: "Option A", b: "Option B", winner: 'a' },
      { a: "Option A", b: "Option B", winner: 'b' },
      { a: "Option A", b: "Option B", winner: 'b' },
      // A vs C: 5 comparisons (A wins 4, C wins 1)
      { a: "Option A", b: "Option C", winner: 'a' },
      { a: "Option A", b: "Option C", winner: 'a' },
      { a: "Option A", b: "Option C", winner: 'a' },
      { a: "Option A", b: "Option C", winner: 'a' },
      { a: "Option A", b: "Option C", winner: 'c' },
      // B vs C: 5 comparisons (B wins 3, C wins 2)
      { a: "Option B", b: "Option C", winner: 'b' },
      { a: "Option B", b: "Option C", winner: 'b' },
      { a: "Option B", b: "Option C", winner: 'b' },
      { a: "Option B", b: "Option C", winner: 'c' },
      { a: "Option B", b: "Option C", winner: 'c' }
    ]
  },
  {
    id: 2,
    title: "Building the Preference Matrix",
    description: "Based on the comparisons from Step 1, fill in the preference matrix. Each cell (i,j) shows how many times row option was preferred over column option.",
    insight: "The diagonal is always 0 (can't compare to itself). The matrix is antisymmetric: if A beat B 3 times out of 5, then B beat A 2 times, so matrix[A][B] = 3 and matrix[B][A] = 2.",
    matrix: Array(3).fill(null).map(() => Array(3).fill(null).map(() => ({ value: null }))),
    expectedValues: [
      [0, 3, 4],  // A wins: 3 vs B, 4 vs C
      [2, 0, 3],  // B wins: 2 vs A, 3 vs C
      [1, 2, 0]   // C wins: 1 vs A, 2 vs B
    ],
    inputPositions: [[0, 1], [0, 2], [1, 0], [1, 2], [2, 0], [2, 1]]
  },
  {
    id: 3,
    title: "Calculating Win Rates",
    description: "Calculate the total wins for each option by summing each row. Enter the win counts for each option.",
    insight: "The Bradley-Terry model assumes that the probability of option i beating option j depends on their relative 'strengths' or rewards.",
    matrix: Array(3).fill(null).map(() => Array(4).fill(null).map(() => ({ value: null }))),
    expectedValues: [
      [0, 3, 4, 7],
      [2, 0, 3, 5],
      [1, 2, 0, 3]
    ],
    inputPositions: [[0, 3], [1, 3], [2, 3]],
    calculation: "Row sum = Total wins for that option"
  },
  {
    id: 4,
    title: "Normalizing to Probabilities",
    description: "Convert win counts to probabilities by dividing by the total number of comparisons. Fill in the probability values.",
    insight: "Normalization ensures our reward values are on a consistent scale, making them easier to interpret and use in training.",
    matrix: Array(3).fill(null).map(() => Array(2).fill(null).map(() => ({ value: null }))),
    expectedValues: [
      [7, 0.47],
      [5, 0.33],
      [3, 0.20]
    ],
    inputPositions: [[0, 1], [1, 1], [2, 1]],
    calculation: "P(option) = wins / total_comparisons"
  },
  {
    id: 5,
    title: "Computing Reward Values",
    description: "Apply the log transformation to get reward values. The reward represents how 'good' each option is according to human preferences.",
    insight: "The log transformation in the Bradley-Terry model ensures that reward differences correspond to probability ratios in pairwise comparisons.",
    matrix: Array(3).fill(null).map(() => Array(3).fill(null).map(() => ({ value: null }))),
    expectedValues: [
      [0.47, -0.76, -0.76],
      [0.33, -1.11, -1.11],
      [0.20, -1.61, -1.61]
    ],
    inputPositions: [[0, 1], [1, 1], [2, 1]],
    calculation: "r(option) = log(P(option))"
  },
  {
    id: 6,
    title: "Verifying the Model",
    description: "Let's verify our reward model by checking if it correctly predicts the original preferences. Calculate P(i > j) using the rewards.",
    insight: "A good reward model should accurately reconstruct the original preference probabilities. This validates our training process.",
    matrix: Array(3).fill(null).map(() => Array(3).fill(null).map(() => ({ value: null }))),
    expectedValues: [
      [0.5, 0.60, 0.71],
      [0.40, 0.5, 0.62],
      [0.29, 0.38, 0.5]
    ],
    inputPositions: [[0, 1], [0, 2], [1, 0], [1, 2], [2, 0], [2, 1]],
    calculation: "P(i > j) = exp(r_i) / (exp(r_i) + exp(r_j))"
  }
];

export function RLHFInteractiveDemo() {
  const [currentStep, setCurrentStep] = useState(0);
  const [userInputs, setUserInputs] = useState<Record<string, Record<string, number>>>({});
  const [showHint, setShowHint] = useState(false);
  const [showSolution, setShowSolution] = useState(false);
  const [validationResults, setValidationResults] = useState<Record<string, boolean>>({});
  const [attemptCount, setAttemptCount] = useState(0);

  const step = steps[currentStep];

  const handleInputChange = (row: number, col: number, value: string) => {
    const numValue = parseFloat(value);
    if (!isNaN(numValue) || value === '') {
      setUserInputs(prev => ({
        ...prev,
        [currentStep]: {
          ...prev[currentStep],
          [`${row}-${col}`]: value === '' ? 0 : numValue
        }
      }));
    }
  };

  const validateStep = () => {
    if (!step.expectedValues || !step.inputPositions) return;

    const results: Record<string, boolean> = {};
    let allCorrect = true;

    step.inputPositions.forEach(([row, col]) => {
      const key = `${row}-${col}`;
      const userValue = userInputs[currentStep]?.[key] ?? 0;
      const expectedValue = step.expectedValues![row][col];
      
      const isCorrect = Math.abs(userValue - expectedValue) < 0.01;
      results[key] = isCorrect;
      if (!isCorrect) allCorrect = false;
    });

    setValidationResults(results);
    setAttemptCount(prev => prev + 1);

    if (allCorrect) {
      setTimeout(() => {
        if (currentStep < steps.length - 1) {
          handleNextStep();
        }
      }, 1500);
    }
  };

  const handleNextStep = () => {
    if (currentStep < steps.length - 1) {
      setCurrentStep(prev => prev + 1);
      setShowHint(false);
      setShowSolution(false);
      setValidationResults({});
      setAttemptCount(0);
    }
  };

  const handlePrevStep = () => {
    if (currentStep > 0) {
      setCurrentStep(prev => prev - 1);
      setShowHint(false);
      setShowSolution(false);
      setValidationResults({});
      setAttemptCount(0);
    }
  };

  const fillSolution = () => {
    if (!step.expectedValues || !step.inputPositions) return;

    const newInputs: Record<string, number> = {};
    step.inputPositions.forEach(([row, col]) => {
      const key = `${row}-${col}`;
      newInputs[key] = step.expectedValues![row][col];
    });

    setUserInputs(prev => ({
      ...prev,
      [currentStep]: newInputs
    }));

    setShowSolution(true);
    setTimeout(() => validateStep(), 100);
  };

  const renderMatrix = () => {
    if (!step.matrix) return null;

    const labels = currentStep === 1 ? ['A', 'B', 'C'] : 
                   currentStep === 2 ? ['A wins', 'B wins', 'C wins'] :
                   currentStep === 3 ? ['Total', 'P(win)'] :
                   currentStep === 4 ? ['P(win)', 'log(P)', 'Reward'] :
                   ['A', 'B', 'C'];

    const showRowLabels = currentStep <= 2 || currentStep === 5;
    const showColLabels = currentStep === 1 || currentStep === 5;

    return (
      <div className="bg-gray-50 p-6 rounded-lg">
        {currentStep === 1 && (
          <div className="text-center mb-4 text-sm text-gray-600">
            Row = Winner, Column = Loser
          </div>
        )}
        
        <div className="flex">
          {showRowLabels && (
            <div className="flex flex-col justify-center mr-3">
              <div className="h-10"></div>
              {labels.slice(0, step.matrix.length).map((label, idx) => (
                <div key={idx} className="h-12 flex items-center justify-end pr-2 font-semibold text-gray-700">
                  {label}
                </div>
              ))}
            </div>
          )}
          
          <div>
            {showColLabels && (
              <div className="flex mb-3">
                <div className="w-10"></div>
                {labels.slice(0, step.matrix[0].length).map((label, idx) => (
                  <div key={idx} className="w-12 text-center font-semibold text-gray-700">
                    {label}
                  </div>
                ))}
              </div>
            )}
            
            <div className="grid gap-2" style={{ gridTemplateColumns: `repeat(${step.matrix[0].length}, 1fr)` }}>
              {step.matrix.map((row, rowIdx) => 
                row.map((cell, colIdx) => {
                  const key = `${rowIdx}-${colIdx}`;
                  const isInput = step.inputPositions?.some(([r, c]) => r === rowIdx && c === colIdx);
                  const expectedValue = step.expectedValues?.[rowIdx][colIdx];
                  const userValue = userInputs[currentStep]?.[key];
                  const isValidated = validationResults[key] !== undefined;
                  const isCorrect = validationResults[key];
                  const isDiagonal = rowIdx === colIdx && currentStep === 1;

                  return (
                    <motion.div
                      key={key}
                      initial={{ scale: 0.8, opacity: 0 }}
                      animate={{ scale: 1, opacity: 1 }}
                      transition={{ delay: rowIdx * 0.1 + colIdx * 0.05 }}
                      className={`
                        relative w-12 h-12 rounded-lg border-2 transition-all flex items-center justify-center
                        ${isInput ? 'bg-white' : isDiagonal ? 'bg-gray-200' : 'bg-gray-100'}
                        ${isValidated ? (isCorrect ? 'border-green-500' : 'border-red-500') : 'border-gray-300'}
                        ${isInput && !isValidated ? 'hover:border-blue-400' : ''}
                      `}
                    >
                      {isInput ? (
                        <input
                          type="number"
                          step="0.01"
                          className="w-full text-center font-mono text-sm bg-transparent outline-none"
                          value={userValue ?? ''}
                          onChange={(e) => handleInputChange(rowIdx, colIdx, e.target.value)}
                          placeholder="?"
                        />
                      ) : (
                        <div className="text-center font-mono text-sm">
                          {expectedValue !== undefined ? expectedValue : isDiagonal ? '0' : '-'}
                        </div>
                      )}
                      
                      {isValidated && (
                        <motion.div
                          initial={{ scale: 0 }}
                          animate={{ scale: 1 }}
                          className="absolute -top-2 -right-2"
                        >
                          {isCorrect ? (
                            <CheckCircle className="w-5 h-5 text-green-500" />
                          ) : (
                            <AlertCircle className="w-5 h-5 text-red-500" />
                          )}
                        </motion.div>
                      )}
                    </motion.div>
                  );
                })
              )}
            </div>
          </div>
        </div>
      </div>
    );
  };

  const renderComparison = () => {
    if (!step.showComparison || !step.comparisonPairs) return null;

    // Calculate comparison summary
    const comparisonCounts: Record<string, number> = {};
    step.comparisonPairs.forEach(pair => {
      const key = `${pair.a} > ${pair.b}`;
      if (pair.winner === 'a') {
        comparisonCounts[key] = (comparisonCounts[key] || 0) + 1;
      } else {
        const reverseKey = `${pair.b} > ${pair.a}`;
        comparisonCounts[reverseKey] = (comparisonCounts[reverseKey] || 0) + 1;
      }
    });

    // Group comparisons
    const groupedComparisons: Record<string, { total: number; wins: { a: number; b: number } }> = {};
    step.comparisonPairs.forEach(pair => {
      const key = [pair.a, pair.b].sort().join(' vs ');
      if (!groupedComparisons[key]) {
        groupedComparisons[key] = { total: 0, wins: { a: 0, b: 0 } };
      }
      groupedComparisons[key].total++;
      // Determine which option in the sorted key won
      const [optionA, optionB] = [pair.a, pair.b].sort();
      if (pair.winner === 'a') {
        if (pair.a === optionA) {
          groupedComparisons[key].wins.a++;
        } else {
          groupedComparisons[key].wins.b++;
        }
      } else if (pair.winner === 'b') {
        if (pair.b === optionA) {
          groupedComparisons[key].wins.a++;
        } else {
          groupedComparisons[key].wins.b++;
        }
      } else if (pair.winner === 'c') {
        // For Option C wins
        if (pair.a.includes('C')) {
          groupedComparisons[key].wins.b++;
        } else {
          groupedComparisons[key].wins.b++;
        }
      }
    });

    return (
      <div className="space-y-6">
        <div className="bg-blue-50 p-4 rounded-lg">
          <h4 className="font-semibold mb-3">Comparison Summary:</h4>
          <div className="grid gap-3">
            {Object.entries(groupedComparisons).map(([comparison, data]) => {
              const [optionA, optionB] = comparison.split(' vs ');
              return (
                <div key={comparison} className="bg-white p-3 rounded border border-blue-200">
                  <div className="flex items-center justify-between mb-2">
                    <span className="font-medium">{comparison}</span>
                    <span className="text-sm text-gray-600">{data.total} comparisons</span>
                  </div>
                  <div className="flex items-center space-x-2 text-sm">
                    <div className="flex-1">
                      <div className="flex justify-between mb-1">
                        <span>{optionA}</span>
                        <span className="font-semibold">{data.wins.a} wins</span>
                      </div>
                      <div className="w-full bg-gray-200 rounded-full h-2">
                        <div 
                          className="bg-green-500 h-2 rounded-full transition-all"
                          style={{ width: `${(data.wins.a / data.total) * 100}%` }}
                        />
                      </div>
                    </div>
                    <div className="text-gray-400">vs</div>
                    <div className="flex-1">
                      <div className="flex justify-between mb-1">
                        <span>{optionB}</span>
                        <span className="font-semibold">{data.wins.b} wins</span>
                      </div>
                      <div className="w-full bg-gray-200 rounded-full h-2">
                        <div 
                          className="bg-blue-500 h-2 rounded-full transition-all"
                          style={{ width: `${(data.wins.b / data.total) * 100}%` }}
                        />
                      </div>
                    </div>
                  </div>
                </div>
              );
            })}
          </div>
        </div>
        
        <div className="text-center text-sm text-gray-600">
          <p>These comparisons will form our preference matrix in the next step.</p>
          <p className="mt-1">For example: A won against B in 3 out of 5 comparisons, so matrix[A][B] = 3</p>
        </div>
      </div>
    );
  };

  return (
    <div className="max-w-4xl mx-auto p-6">
      <div className="mb-8">
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-3xl font-bold">RLHF Interactive Demo</h2>
          <div className="flex items-center space-x-2">
            <span className="text-sm text-gray-600">Step {currentStep + 1} of {steps.length}</span>
          </div>
        </div>
        
        <div className="w-full bg-gray-200 rounded-full h-2">
          <motion.div
            className="bg-blue-600 h-2 rounded-full"
            initial={{ width: 0 }}
            animate={{ width: `${((currentStep + 1) / steps.length) * 100}%` }}
            transition={{ duration: 0.3 }}
          />
        </div>
      </div>

      <AnimatePresence mode="wait">
        <motion.div
          key={currentStep}
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0, y: -20 }}
          transition={{ duration: 0.3 }}
          className="space-y-6"
        >
          <div>
            <h3 className="text-2xl font-semibold mb-3">{step.title}</h3>
            <p className="text-gray-700 mb-4">{step.description}</p>
            
            <div className="bg-blue-50 border-l-4 border-blue-400 p-4 mb-6">
              <p className="text-sm font-medium text-blue-900">💡 Insight</p>
              <p className="text-sm text-blue-800 mt-1">{step.insight}</p>
            </div>
          </div>

          {step.calculation && (
            <div className="bg-gray-100 p-3 rounded font-mono text-sm text-center">
              {step.calculation}
            </div>
          )}

          {renderComparison()}
          {renderMatrix()}

          <div className="flex items-center justify-between mt-8">
            <button
              onClick={handlePrevStep}
              disabled={currentStep === 0}
              className={`flex items-center space-x-2 px-4 py-2 rounded-lg transition-all ${
                currentStep === 0 
                  ? 'bg-gray-200 text-gray-400 cursor-not-allowed' 
                  : 'bg-gray-600 text-white hover:bg-gray-700'
              }`}
            >
              <ChevronLeft className="w-4 h-4" />
              <span>Previous</span>
            </button>

            <div className="flex items-center space-x-3">
              {step.inputPositions && (
                <>
                  <button
                    onClick={() => setShowHint(!showHint)}
                    className="p-2 rounded-lg bg-yellow-100 text-yellow-700 hover:bg-yellow-200 transition-all"
                  >
                    <HelpCircle className="w-5 h-5" />
                  </button>
                  
                  {attemptCount >= 2 && (
                    <button
                      onClick={fillSolution}
                      className="flex items-center space-x-2 px-3 py-2 rounded-lg bg-purple-100 text-purple-700 hover:bg-purple-200 transition-all"
                    >
                      <Eye className="w-4 h-4" />
                      <span>Show Solution</span>
                    </button>
                  )}
                  
                  <button
                    onClick={validateStep}
                    className="flex items-center space-x-2 px-4 py-2 rounded-lg bg-green-600 text-white hover:bg-green-700 transition-all"
                  >
                    <CheckCircle className="w-4 h-4" />
                    <span>Check Answer</span>
                  </button>
                </>
              )}
              
              {(!step.inputPositions || Object.keys(validationResults).length > 0) && (
                <button
                  onClick={handleNextStep}
                  disabled={currentStep === steps.length - 1}
                  className={`flex items-center space-x-2 px-4 py-2 rounded-lg transition-all ${
                    currentStep === steps.length - 1
                      ? 'bg-gray-200 text-gray-400 cursor-not-allowed'
                      : 'bg-blue-600 text-white hover:bg-blue-700'
                  }`}
                >
                  <span>Next</span>
                  <ChevronRight className="w-4 h-4" />
                </button>
              )}
            </div>
          </div>

          {showHint && step.inputPositions && (
            <motion.div
              initial={{ opacity: 0, y: -10 }}
              animate={{ opacity: 1, y: 0 }}
              className="bg-yellow-50 border border-yellow-200 rounded-lg p-4 mt-4"
            >
              <p className="text-sm text-yellow-800">
                <strong>Hint:</strong> 
                {currentStep === 1 && " Look at the comparison summary above. For example, if A won against B in 3 out of 5 comparisons, then matrix[A][B] = 3 and matrix[B][A] = 2."}
                {currentStep === 2 && " Sum each row to get the total wins for that option. For example, row A: 0 + 3 + 4 = 7 total wins."}
                {currentStep === 3 && " Divide each win count by the total number of comparisons (15) to get probabilities. Round to 2 decimal places."}
                {currentStep === 4 && " Apply the natural logarithm to each probability: reward = log(probability). Use negative values."}
                {currentStep === 5 && " Use the formula P(i > j) = exp(r_i) / (exp(r_i) + exp(r_j)) where r_i and r_j are the rewards."}
              </p>
            </motion.div>
          )}
        </motion.div>
      </AnimatePresence>
    </div>
  );
}