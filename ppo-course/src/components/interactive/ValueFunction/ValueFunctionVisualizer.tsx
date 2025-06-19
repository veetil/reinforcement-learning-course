'use client';

import React, { useState, useEffect, useCallback, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { RotateCcw, Download, Play, Pause, Target, X } from 'lucide-react';
import { ValueFunctionCalculator } from './ValueFunctionCalculator';

interface ValueFunctionVisualizerProps {
  width?: number;
  height?: number;
  showValues?: boolean;
  showPolicy?: boolean;
  animated?: boolean;
  showConvergence?: boolean;
  onValueChange?: (values: number[][]) => void;
}

export const ValueFunctionVisualizer: React.FC<ValueFunctionVisualizerProps> = ({
  width = 6,
  height = 6,
  showValues = false,
  showPolicy = false,
  animated = false,
  showConvergence = false,
  onValueChange
}) => {
  const [calculator] = useState(() => new ValueFunctionCalculator(width, height));
  const [values, setValues] = useState<number[][]>(calculator.values);
  const [discountFactor, setDiscountFactor] = useState(0.9);
  const [isAnimating, setIsAnimating] = useState(false);
  const [currentIteration, setCurrentIteration] = useState(0);
  const [convergenceData, setConvergenceData] = useState<number[]>([]);
  const [showIterations, setShowIterations] = useState(false);
  const [selectedCell, setSelectedCell] = useState<{ x: number; y: number } | null>(null);
  const [exportMessage, setExportMessage] = useState('');
  const animationRef = useRef<NodeJS.Timeout>();

  useEffect(() => {
    return () => {
      if (animationRef.current) {
        clearTimeout(animationRef.current);
      }
    };
  }, []);

  const updateValues = useCallback(() => {
    const iterations = calculator.calculateValues();
    setValues([...calculator.values]);
    onValueChange?.(calculator.values);
    return iterations;
  }, [calculator, onValueChange]);

  const handleCellClick = useCallback((x: number, y: number) => {
    if (calculator.isObstacle(x, y)) {
      calculator.removeObstacle(x, y);
    } else if (calculator.isGoal(x, y)) {
      // Do nothing if clicking on goal
      return;
    } else if (selectedCell && selectedCell.x === x && selectedCell.y === y) {
      // Toggle obstacle
      calculator.addObstacle(x, y);
      setSelectedCell(null);
    } else {
      // Set as goal
      calculator.setGoal(x, y);
      setSelectedCell({ x, y });
    }
    
    updateValues();
  }, [calculator, selectedCell, updateValues]);

  const handleReset = useCallback(() => {
    calculator.reset();
    setValues([...calculator.values]);
    setSelectedCell(null);
    setCurrentIteration(0);
    setConvergenceData([]);
    setIsAnimating(false);
    if (animationRef.current) {
      clearTimeout(animationRef.current);
    }
  }, [calculator]);

  const handleDiscountChange = useCallback((value: number) => {
    setDiscountFactor(value);
    calculator.setDiscountFactor(value);
    if (calculator.goalX !== -1) {
      updateValues();
    }
  }, [calculator, updateValues]);

  const animateValueIteration = useCallback(() => {
    setIsAnimating(true);
    setCurrentIteration(0);
    const tempConvergence: number[] = [];

    const runIteration = (iter: number) => {
      if (iter >= 50 || !calculator.goalX || calculator.goalX === -1) {
        setIsAnimating(false);
        return;
      }

      // Run one iteration manually
      const oldValues = calculator.values.map(row => [...row]);
      calculator.calculateValues();
      
      // Calculate max change
      let maxChange = 0;
      for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
          maxChange = Math.max(maxChange, Math.abs(calculator.values[y][x] - oldValues[y][x]));
        }
      }
      
      tempConvergence.push(maxChange);
      setConvergenceData([...tempConvergence]);
      setValues([...calculator.values]);
      setCurrentIteration(iter + 1);

      if (maxChange > 0.001) {
        animationRef.current = setTimeout(() => runIteration(iter + 1), 100);
      } else {
        setIsAnimating(false);
      }
    };

    calculator.reset();
    if (selectedCell) {
      calculator.setGoal(selectedCell.x, selectedCell.y);
    }
    runIteration(0);
  }, [calculator, selectedCell, width, height]);

  const handleExport = useCallback(() => {
    const data = calculator.exportData();
    
    // In test environment, just set the message
    if (typeof window === 'undefined' || !window.URL || !window.URL.createObjectURL) {
      setExportMessage('Values exported!');
      setTimeout(() => setExportMessage(''), 2000);
      return;
    }
    
    const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'value-function.json';
    a.click();
    URL.revokeObjectURL(url);
    
    setExportMessage('Values exported!');
    setTimeout(() => setExportMessage(''), 2000);
  }, [calculator]);

  const getValueColor = (value: number): string => {
    const intensity = Math.floor(value * 255);
    return `rgb(${255 - intensity}, ${255 - intensity * 0.5}, ${255})`;
  };

  const getCellClasses = (x: number, y: number): string => {
    const classes = ['relative border-2 flex flex-col items-center justify-center cursor-pointer transition-all'];
    
    if (calculator.isGoal(x, y)) {
      classes.push('goal', 'bg-green-400 border-green-600');
    } else if (calculator.isObstacle(x, y)) {
      classes.push('obstacle', 'bg-gray-800 border-gray-900');
    } else {
      classes.push('border-gray-300 hover:border-blue-400');
    }
    
    if (animated && isAnimating) {
      classes.push('animating');
    }
    
    return classes.join(' ');
  };

  const renderPolicyArrow = (x: number, y: number): React.ReactNode => {
    if (!showPolicy || calculator.isGoal(x, y) || calculator.isObstacle(x, y)) {
      return null;
    }

    const policy = calculator.getOptimalPolicy();
    const action = policy[`${x},${y}`];

    const arrowMap: { [key: string]: string } = {
      up: '↑',
      down: '↓',
      left: '←',
      right: '→'
    };

    return arrowMap[action] ? (
      <div className="absolute text-2xl font-bold text-gray-700 opacity-50">
        {arrowMap[action]}
      </div>
    ) : null;
  };

  return (
    <div className="w-full space-y-4">
      {/* Controls */}
      <div className="flex flex-wrap gap-4">
        <div className="flex items-center gap-2">
          <label htmlFor="discount" className="text-sm font-medium">
            Discount Factor
          </label>
          <input
            id="discount"
            type="range"
            min="0"
            max="1"
            step="0.05"
            value={discountFactor}
            onChange={(e) => handleDiscountChange(parseFloat(e.target.value))}
            className="w-32"
            aria-label="Discount Factor"
          />
          <span className="text-sm font-mono">γ = {discountFactor.toFixed(2)}</span>
        </div>

        <button
          onClick={handleReset}
          className="px-4 py-2 bg-gray-600 text-white rounded-lg hover:bg-gray-700 flex items-center gap-2"
        >
          <RotateCcw className="w-4 h-4" />
          Reset
        </button>

        <button
          onClick={() => setShowIterations(!showIterations)}
          className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700"
        >
          {showIterations ? 'Hide' : 'Show'} Iterations
        </button>

        {animated && (
          <button
            onClick={animateValueIteration}
            disabled={isAnimating || !selectedCell}
            className="px-4 py-2 bg-purple-600 text-white rounded-lg hover:bg-purple-700 disabled:bg-gray-400 flex items-center gap-2"
          >
            {isAnimating ? (
              <>
                <Pause className="w-4 h-4" />
                Animating...
              </>
            ) : (
              <>
                <Play className="w-4 h-4" />
                Animate
              </>
            )}
          </button>
        )}

        <button
          onClick={handleExport}
          className="px-4 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700 flex items-center gap-2"
        >
          <Download className="w-4 h-4" />
          Export Values
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

      {/* Iteration info */}
      {showIterations && isAnimating && (
        <div className="bg-blue-50 p-3 rounded-lg">
          <p className="text-sm">
            Iteration: <span className="font-bold">{currentIteration}</span>
          </p>
        </div>
      )}

      {/* Grid */}
      <div className="bg-gray-100 p-4 rounded-lg shadow-lg">
        <div
          className="grid gap-1 mx-auto"
          style={{
            gridTemplateColumns: `repeat(${width}, minmax(0, 1fr))`,
            width: 'fit-content'
          }}
        >
          {Array.from({ length: height }, (_, y) =>
            Array.from({ length: width }, (_, x) => {
              const value = values[y][x];
              const isGoal = calculator.isGoal(x, y);
              const isObstacle = calculator.isObstacle(x, y);

              return (
                <motion.div
                  key={`${x}-${y}`}
                  data-testid={`value-cell-${x}-${y}`}
                  className={getCellClasses(x, y)}
                  style={{
                    backgroundColor: !isGoal && !isObstacle ? getValueColor(value) : undefined,
                    width: '80px',
                    height: '80px'
                  }}
                  onClick={() => handleCellClick(x, y)}
                  whileHover={{ scale: 1.05 }}
                  whileTap={{ scale: 0.95 }}
                  animate={animated && isAnimating ? {
                    opacity: [0.5, 1, 0.5],
                    transition: { duration: 1, repeat: Infinity }
                  } : {}}
                >
                  {isGoal && (
                    <Target className="w-8 h-8 text-white" />
                  )}
                  {isObstacle && (
                    <X className="w-8 h-8 text-white" />
                  )}
                  {showValues && !isGoal && !isObstacle && (
                    <span
                      data-testid={`value-label-${x}-${y}`}
                      className="text-xs font-mono font-bold"
                      style={{
                        color: value > 0.5 ? 'white' : 'black'
                      }}
                    >
                      {value.toFixed(2)}
                    </span>
                  )}
                  {renderPolicyArrow(x, y)}
                </motion.div>
              );
            })
          )}
        </div>
      </div>

      {/* Instructions */}
      <div className="text-sm text-gray-600 space-y-1">
        <p>• Click a cell to set it as the goal (green)</p>
        <p>• Click the goal cell again to place an obstacle</p>
        <p>• Brighter cells have higher values (closer to goal)</p>
        <p>• Value = γ × max(neighbor values)</p>
      </div>

      {/* Convergence Graph */}
      {showConvergence && convergenceData.length > 0 && (
        <div data-testid="convergence-graph" className="bg-white p-4 rounded-lg shadow">
          <h3 className="font-bold mb-2">Convergence</h3>
          <div className="h-32 flex items-end gap-1">
            {convergenceData.map((value, idx) => (
              <div
                key={idx}
                className="bg-blue-500 flex-1"
                style={{
                  height: `${Math.min(100, value * 1000)}%`
                }}
              />
            ))}
          </div>
          <p className="text-xs text-gray-500 mt-2">Max value change per iteration</p>
        </div>
      )}
    </div>
  );
};