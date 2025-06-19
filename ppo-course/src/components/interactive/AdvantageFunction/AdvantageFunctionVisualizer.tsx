'use client';

import React, { useState, useCallback, useMemo } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  RotateCcw, Download, Info, ArrowUp, ArrowDown, 
  ArrowLeft, ArrowRight, Target, BarChart3, GitCompare 
} from 'lucide-react';
import { AdvantageCalculator, State, ActionValues, Action } from './AdvantageCalculator';
import { ValueFunctionCalculator } from '../ValueFunction/ValueFunctionCalculator';

interface AdvantageFunctionVisualizerProps {
  gridSize?: number;
  showAdvantage?: boolean;
  showGAE?: boolean;
  onAdvantageChange?: (advantages: { [key: string]: ActionValues }) => void;
}

interface CellData {
  state: State;
  value: number;
  qValues?: ActionValues;
  advantages?: ActionValues;
  isGoal: boolean;
  isSelected: boolean;
  isInTrajectory: boolean;
}

export const AdvantageFunctionVisualizer: React.FC<AdvantageFunctionVisualizerProps> = ({
  gridSize = 6,
  showAdvantage = true,
  showGAE = false,
  onAdvantageChange
}) => {
  const [calculator] = useState(() => new AdvantageCalculator());
  const [valueCalculator] = useState(() => new ValueFunctionCalculator(gridSize, gridSize));
  const [selectedState, setSelectedState] = useState<State | null>(null);
  const [goalState, setGoalState] = useState<State | null>(null);
  const [trajectory, setTrajectory] = useState<State[]>([]);
  const [showGAEPanel, setShowGAEPanel] = useState(showGAE);
  const [lambda, setLambda] = useState(0.95);
  const [gamma, setGamma] = useState(0.9);
  const [trajectoryMode, setTrajectoryMode] = useState(false);
  const [compareMode, setCompareMode] = useState(false);
  const [exportMessage, setExportMessage] = useState('');

  // Calculate values when goal changes
  useMemo(() => {
    if (goalState) {
      valueCalculator.setGoal(goalState.x, goalState.y);
      valueCalculator.setDiscountFactor(gamma);
      valueCalculator.calculateValues();
    }
  }, [goalState, gamma, valueCalculator]);

  const getCellData = useCallback((x: number, y: number): CellData => {
    const state = { x, y };
    const value = valueCalculator.getValue(x, y);
    const isGoal = goalState?.x === x && goalState?.y === y;
    const isSelected = selectedState?.x === x && selectedState?.y === y;
    const isInTrajectory = trajectory.some(s => s.x === x && s.y === y);

    let qValues: ActionValues | undefined;
    let advantages: ActionValues | undefined;

    if (isSelected && !isGoal) {
      // Calculate Q-values for each action
      const nextStateValues: ActionValues = {
        up: y > 0 ? valueCalculator.getValue(x, y - 1) : 0,
        down: y < gridSize - 1 ? valueCalculator.getValue(x, y + 1) : 0,
        left: x > 0 ? valueCalculator.getValue(x - 1, y) : 0,
        right: x < gridSize - 1 ? valueCalculator.getValue(x + 1, y) : 0
      };

      const reward = -0.1; // Step penalty
      qValues = calculator.calculateQValues(state, nextStateValues, reward, gamma);
      
      // Calculate advantages
      advantages = {
        up: calculator.calculateAdvantage(qValues.up, value),
        down: calculator.calculateAdvantage(qValues.down, value),
        left: calculator.calculateAdvantage(qValues.left, value),
        right: calculator.calculateAdvantage(qValues.right, value)
      };
    }

    return { state, value, qValues, advantages, isGoal, isSelected, isInTrajectory };
  }, [valueCalculator, selectedState, goalState, trajectory, calculator, gamma, gridSize]);

  const handleCellClick = useCallback((x: number, y: number) => {
    const state = { x, y };

    if (trajectoryMode) {
      setTrajectory(prev => [...prev, state]);
    } else if (!goalState || (goalState.x === x && goalState.y === y)) {
      setGoalState(state);
      setSelectedState(null);
      setTrajectory([]);
    } else {
      setSelectedState(state);
    }
  }, [goalState, trajectoryMode]);

  const handleReset = useCallback(() => {
    setSelectedState(null);
    setGoalState(null);
    setTrajectory([]);
    valueCalculator.reset();
    setExportMessage('');
  }, [valueCalculator]);

  const handleExport = useCallback(() => {
    const data = {
      gridSize,
      goalState,
      gamma,
      lambda,
      values: Array.from({ length: gridSize }, (_, y) =>
        Array.from({ length: gridSize }, (_, x) => valueCalculator.getValue(x, y))
      ),
      trajectory: trajectory.map(s => ({
        state: s,
        value: valueCalculator.getValue(s.x, s.y)
      }))
    };

    if (typeof window !== 'undefined' && window.URL && window.URL.createObjectURL) {
      const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = 'advantage-function.json';
      a.click();
      URL.revokeObjectURL(url);
    }

    setExportMessage('Data exported!');
    setTimeout(() => setExportMessage(''), 2000);
  }, [gridSize, goalState, gamma, lambda, trajectory, valueCalculator]);

  const getAdvantageColor = (advantage: number): string => {
    if (advantage > 0) {
      const intensity = Math.min(255, Math.floor(advantage * 500));
      return `rgb(0, ${intensity}, 0)`;
    } else {
      const intensity = Math.min(255, Math.floor(-advantage * 500));
      return `rgb(${intensity}, 0, 0)`;
    }
  };

  const renderActionArrows = (cellData: CellData) => {
    if (!cellData.advantages || !showAdvantage) return null;

    const arrows = [
      { action: 'up' as Action, icon: ArrowUp, className: 'top-1' },
      { action: 'down' as Action, icon: ArrowDown, className: 'bottom-1' },
      { action: 'left' as Action, icon: ArrowLeft, className: 'left-1' },
      { action: 'right' as Action, icon: ArrowRight, className: 'right-1' }
    ];

    return arrows.map(({ action, icon: Icon, className }) => {
      const advantage = cellData.advantages![action];
      const opacity = Math.abs(advantage) * 2;
      
      return (
        <div
          key={action}
          data-testid={`advantage-arrow-${action}`}
          className={`absolute ${className} flex items-center justify-center`}
          style={{ opacity: Math.min(1, opacity) }}
        >
          <Icon 
            className="w-4 h-4" 
            style={{ color: getAdvantageColor(advantage) }}
          />
          {Math.abs(advantage) > 0.1 && (
            <span className="text-xs font-mono ml-1">
              {advantage.toFixed(2)}
            </span>
          )}
        </div>
      );
    });
  };

  const renderGAEPanel = () => {
    if (!showGAEPanel || trajectory.length < 2) return null;

    const rewards = trajectory.slice(0, -1).map(() => -0.1);
    const values = trajectory.map(s => valueCalculator.getValue(s.x, s.y));
    const gae = calculator.calculateGAE(rewards, values, gamma, lambda);

    return (
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        data-testid="gae-panel"
        className="mt-4 bg-white rounded-lg shadow-lg p-4"
      >
        <h3 className="font-bold mb-2">GAE Values Along Trajectory</h3>
        <div className="space-y-2">
          {gae.map((advantage, idx) => (
            <div key={idx} className="flex items-center gap-2">
              <span className="text-sm font-mono">
                Step {idx}: ({trajectory[idx].x}, {trajectory[idx].y})
              </span>
              <div 
                className="h-4 rounded"
                style={{
                  width: `${Math.abs(advantage) * 50}px`,
                  backgroundColor: getAdvantageColor(advantage)
                }}
              />
              <span className="text-sm font-mono">
                {advantage.toFixed(3)}
              </span>
            </div>
          ))}
        </div>
      </motion.div>
    );
  };

  return (
    <div className="w-full space-y-4">
      {/* Controls */}
      <div className="flex flex-wrap gap-4">
        <div className="flex items-center gap-2">
          <label htmlFor="gamma-slider" className="text-sm font-medium">
            Discount (γ)
          </label>
          <input
            id="gamma-slider"
            type="range"
            min="0"
            max="1"
            step="0.05"
            value={gamma}
            onChange={(e) => setGamma(parseFloat(e.target.value))}
            className="w-32"
          />
          <span className="text-sm font-mono">γ = {gamma.toFixed(2)}</span>
        </div>

        <div className="flex items-center gap-2">
          <label htmlFor="lambda-slider" className="text-sm font-medium">
            Lambda (λ)
          </label>
          <input
            id="lambda-slider"
            type="range"
            min="0"
            max="1"
            step="0.05"
            value={lambda}
            onChange={(e) => setLambda(parseFloat(e.target.value))}
            className="w-32"
            aria-label="Lambda (λ)"
          />
          <span className="text-sm font-mono">λ = {lambda.toFixed(2)}</span>
        </div>

        <button
          onClick={handleReset}
          className="px-4 py-2 bg-gray-600 text-white rounded-lg hover:bg-gray-700 flex items-center gap-2"
        >
          <RotateCcw className="w-4 h-4" />
          Reset
        </button>

        <button
          onClick={() => setTrajectoryMode(!trajectoryMode)}
          className={`px-4 py-2 rounded-lg flex items-center gap-2 ${
            trajectoryMode 
              ? 'bg-purple-600 text-white' 
              : 'bg-gray-200 text-gray-700 hover:bg-gray-300'
          }`}
        >
          <BarChart3 className="w-4 h-4" />
          Trajectory Mode
        </button>

        <button
          onClick={() => setShowGAEPanel(!showGAEPanel)}
          className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700"
        >
          {showGAEPanel ? 'Hide' : 'Show'} GAE
        </button>

        <button
          onClick={() => setCompareMode(!compareMode)}
          className="px-4 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700 flex items-center gap-2"
        >
          <GitCompare className="w-4 h-4" />
          Compare Policies
        </button>

        <button
          onClick={handleExport}
          className="px-4 py-2 bg-indigo-600 text-white rounded-lg hover:bg-indigo-700 flex items-center gap-2"
        >
          <Download className="w-4 h-4" />
          Export Data
        </button>
      </div>

      {/* Export message */}
      <AnimatePresence>
        {exportMessage && (
          <motion.div
            initial={{ opacity: 0, y: -10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0 }}
            className="text-green-600 font-medium"
          >
            {exportMessage}
          </motion.div>
        )}
      </AnimatePresence>

      {/* Main Grid */}
      <div className="flex gap-6">
        <div className="flex-1">
          <div className="bg-gray-100 p-4 rounded-lg shadow-lg">
            <div
              className="grid gap-1 mx-auto"
              style={{
                gridTemplateColumns: `repeat(${gridSize}, minmax(0, 1fr))`,
                width: 'fit-content'
              }}
            >
              {Array.from({ length: gridSize }, (_, y) =>
                Array.from({ length: gridSize }, (_, x) => {
                  const cellData = getCellData(x, y);
                  
                  return (
                    <motion.div
                      key={`${x}-${y}`}
                      data-testid={`advantage-cell-${x}-${y}`}
                      className={`
                        relative border-2 flex items-center justify-center cursor-pointer
                        transition-all w-20 h-20
                        ${cellData.isGoal ? 'bg-green-400 border-green-600' : ''}
                        ${cellData.isSelected ? 'bg-blue-200 border-blue-600 selected' : ''}
                        ${cellData.isInTrajectory ? 'bg-purple-100 border-purple-400' : ''}
                        ${!cellData.isGoal && !cellData.isSelected && !cellData.isInTrajectory 
                          ? 'bg-white border-gray-300 hover:border-gray-400' : ''}
                      `}
                      onClick={() => handleCellClick(x, y)}
                      whileHover={{ scale: 1.05 }}
                      whileTap={{ scale: 0.95 }}
                    >
                      {cellData.isGoal && <Target className="w-6 h-6 text-white" />}
                      {!cellData.isGoal && (
                        <span className="text-xs font-mono text-gray-500">
                          {cellData.value.toFixed(2)}
                        </span>
                      )}
                      {renderActionArrows(cellData)}
                    </motion.div>
                  );
                })
              )}
            </div>
          </div>

          {/* Instructions */}
          <div className="mt-4 text-sm text-gray-600 space-y-1">
            <p>• Click to set goal (green), then click other cells to see advantages</p>
            <p>• Green arrows = positive advantage, Red = negative advantage</p>
            <p>• Enable trajectory mode to analyze sequences</p>
          </div>
        </div>

        {/* Side Panel */}
        {selectedState && !(goalState?.x === selectedState.x && goalState?.y === selectedState.y) && (
          <motion.div
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            className="w-80 bg-white rounded-lg shadow-lg p-4"
          >
            <h3 className="font-bold mb-3">State Analysis</h3>
            <div className="space-y-3">
              <div>
                <p className="text-sm text-gray-600">Position: ({selectedState.x}, {selectedState.y})</p>
                <p className="text-sm text-gray-600">
                  V(s) = {getCellData(selectedState.x, selectedState.y).value.toFixed(3)}
                </p>
              </div>

              {getCellData(selectedState.x, selectedState.y).qValues && (
                <div>
                  <h4 className="font-semibold text-sm mb-1">Q-Values</h4>
                  <div className="space-y-1 text-xs">
                    {Object.entries(getCellData(selectedState.x, selectedState.y).qValues!).map(
                      ([action, value]) => (
                        <div key={action} className="flex justify-between">
                          <span className="capitalize">{action}:</span>
                          <span className="font-mono">{value.toFixed(3)}</span>
                        </div>
                      )
                    )}
                  </div>
                </div>
              )}

              {showAdvantage && getCellData(selectedState.x, selectedState.y).advantages && (
                <div data-testid="advantage-display">
                  <h4 className="font-semibold text-sm mb-1">Advantages A(s,a)</h4>
                  <div className="space-y-1">
                    {Object.entries(getCellData(selectedState.x, selectedState.y).advantages!).map(
                      ([action, advantage]) => (
                        <div 
                          key={action} 
                          className="flex items-center justify-between p-1 rounded"
                          style={{ 
                            backgroundColor: advantage > 0 
                              ? 'rgba(0, 255, 0, 0.1)' 
                              : 'rgba(255, 0, 0, 0.1)' 
                          }}
                        >
                          <span className="capitalize text-sm">{action}:</span>
                          <span 
                            className="font-mono text-sm font-bold"
                            style={{ color: getAdvantageColor(advantage) }}
                          >
                            {advantage.toFixed(3)}
                          </span>
                        </div>
                      )
                    )}
                  </div>
                </div>
              )}
            </div>
          </motion.div>
        )}
      </div>

      {/* GAE Panel */}
      {renderGAEPanel()}

      {/* Policy Comparison */}
      {compareMode && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="bg-white rounded-lg shadow-lg p-4"
        >
          <h3 className="font-bold mb-2">Policy Comparison</h3>
          <p className="text-sm text-gray-600">
            Compare how different policies would act in each state
          </p>
          {/* Implementation would go here */}
        </motion.div>
      )}

      {/* Trajectory Advantages */}
      {trajectoryMode && trajectory.length > 0 && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="bg-white rounded-lg shadow-lg p-4"
        >
          <h3 className="font-bold mb-2">Trajectory Advantages</h3>
          <div className="flex items-center gap-4">
            {trajectory.map((state, idx) => (
              <div key={idx} className="text-center">
                <div className="text-xs text-gray-600">Step {idx}</div>
                <div className="font-mono text-sm">({state.x}, {state.y})</div>
              </div>
            ))}
          </div>
        </motion.div>
      )}
    </div>
  );
};