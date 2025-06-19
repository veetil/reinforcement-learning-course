'use client';

import React, { useState, useEffect, useCallback, useMemo } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  ArrowUp, 
  ArrowDown, 
  ArrowLeft, 
  ArrowRight, 
  Target, 
  RotateCcw,
  Play,
  Pause,
  Settings
} from 'lucide-react';
import { GridEnvironment, Action, Position, GridState } from './GridEnvironment';

interface GridWorldProps {
  width?: number;
  height?: number;
  showPolicy?: boolean;
  showTrajectory?: boolean;
  showRewards?: boolean;
  interactive?: boolean;
  onStateChange?: (state: GridState) => void;
}

interface PolicyArrow {
  action: Action;
  probability: number;
}

export const GridWorld: React.FC<GridWorldProps> = ({
  width = 5,
  height = 5,
  showPolicy = false,
  showTrajectory = false,
  showRewards = false,
  interactive = true,
  onStateChange,
}) => {
  const [environment] = useState(() => new GridEnvironment(width, height));
  const [state, setState] = useState<GridState>(environment.getState());
  const [isAutoPlaying, setIsAutoPlaying] = useState(false);
  const [selectedCell, setSelectedCell] = useState<Position | null>(null);
  const [policy, setPolicy] = useState<Map<string, PolicyArrow>>(new Map());

  // Initialize with some obstacles
  useEffect(() => {
    environment.addObstacle(2, 2);
    environment.addObstacle(3, 1);
    environment.addObstacle(1, 3);
    setState(environment.getState());
  }, [environment]);

  // Update policy visualization
  useEffect(() => {
    if (showPolicy) {
      const newPolicy = new Map<string, PolicyArrow>();
      
      for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
          const key = `${x}-${y}`;
          
          // Simple policy: move towards goal
          const dx = state.goalPosition.x - x;
          const dy = state.goalPosition.y - y;
          
          let bestAction: Action;
          if (Math.abs(dx) > Math.abs(dy)) {
            bestAction = dx > 0 ? Action.RIGHT : Action.LEFT;
          } else {
            bestAction = dy > 0 ? Action.DOWN : Action.UP;
          }
          
          newPolicy.set(key, {
            action: bestAction,
            probability: 0.7, // 70% for best action
          });
        }
      }
      
      setPolicy(newPolicy);
    }
  }, [showPolicy, width, height, state.goalPosition]);

  const handleCellClick = useCallback((x: number, y: number) => {
    if (!interactive || state.isDone) return;
    
    const currentPos = state.agentPosition;
    
    // Determine action based on clicked position
    let action: Action | null = null;
    
    if (x === currentPos.x + 1 && y === currentPos.y) action = Action.RIGHT;
    else if (x === currentPos.x - 1 && y === currentPos.y) action = Action.LEFT;
    else if (x === currentPos.x && y === currentPos.y + 1) action = Action.DOWN;
    else if (x === currentPos.x && y === currentPos.y - 1) action = Action.UP;
    
    if (action) {
      const result = environment.step(action);
      setState(result.state);
      onStateChange?.(result.state);
    }
    
    setSelectedCell({ x, y });
  }, [environment, state, interactive, onStateChange]);

  const handleKeyPress = useCallback((e: KeyboardEvent) => {
    if (!interactive || state.isDone) return;
    
    let action: Action | null = null;
    
    switch (e.key) {
      case 'ArrowUp':
        action = Action.UP;
        break;
      case 'ArrowDown':
        action = Action.DOWN;
        break;
      case 'ArrowLeft':
        action = Action.LEFT;
        break;
      case 'ArrowRight':
        action = Action.RIGHT;
        break;
    }
    
    if (action) {
      e.preventDefault();
      const result = environment.step(action);
      setState(result.state);
      onStateChange?.(result.state);
    }
  }, [environment, state, interactive, onStateChange]);

  useEffect(() => {
    if (interactive) {
      window.addEventListener('keydown', handleKeyPress);
      return () => window.removeEventListener('keydown', handleKeyPress);
    }
  }, [interactive, handleKeyPress]);

  const handleReset = useCallback(() => {
    const newState = environment.reset();
    setState(newState);
    setIsAutoPlaying(false);
    onStateChange?.(newState);
  }, [environment, onStateChange]);

  const toggleObstacle = useCallback((x: number, y: number) => {
    const hasObstacle = state.obstacles.some(obs => obs.x === x && obs.y === y);
    
    if (hasObstacle) {
      environment.removeObstacle(x, y);
    } else {
      environment.addObstacle(x, y);
    }
    
    setState(environment.getState());
  }, [environment, state.obstacles]);

  // Auto-play functionality
  useEffect(() => {
    if (!isAutoPlaying || state.isDone) {
      setIsAutoPlaying(false);
      return;
    }
    
    const timer = setTimeout(() => {
      // Random action for demo
      const validActions = environment.getValidActions();
      if (validActions.length > 0) {
        const randomAction = validActions[Math.floor(Math.random() * validActions.length)];
        const result = environment.step(randomAction);
        setState(result.state);
        onStateChange?.(result.state);
      }
    }, 500);
    
    return () => clearTimeout(timer);
  }, [isAutoPlaying, state.isDone, environment, onStateChange]);

  const getActionIcon = (action: Action) => {
    switch (action) {
      case Action.UP:
        return <ArrowUp className="w-4 h-4" />;
      case Action.DOWN:
        return <ArrowDown className="w-4 h-4" />;
      case Action.LEFT:
        return <ArrowLeft className="w-4 h-4" />;
      case Action.RIGHT:
        return <ArrowRight className="w-4 h-4" />;
    }
  };

  const getCellContent = (x: number, y: number) => {
    const isAgent = state.agentPosition.x === x && state.agentPosition.y === y;
    const isGoal = state.goalPosition.x === x && state.goalPosition.y === y;
    const isObstacle = state.obstacles.some(obs => obs.x === x && obs.y === y);
    const isInTrajectory = showTrajectory && state.trajectory.some(
      pos => pos.x === x && pos.y === y
    );
    
    if (isAgent && isGoal) {
      return (
        <motion.div
          initial={{ scale: 0 }}
          animate={{ scale: 1, rotate: 360 }}
          className="text-2xl"
        >
          🎉
        </motion.div>
      );
    }
    
    if (isAgent) {
      return (
        <motion.div
          layoutId="agent"
          initial={{ scale: 0 }}
          animate={{ scale: 1 }}
          className="w-8 h-8 bg-blue-500 rounded-full flex items-center justify-center text-white font-bold"
        >
          A
        </motion.div>
      );
    }
    
    if (isGoal) {
      return (
        <motion.div
          animate={{ scale: [1, 1.1, 1] }}
          transition={{ repeat: Infinity, duration: 2 }}
        >
          <Target className="w-8 h-8 text-green-500" />
        </motion.div>
      );
    }
    
    if (isObstacle) {
      return (
        <div className="w-8 h-8 bg-gray-700 rounded flex items-center justify-center text-white">
          X
        </div>
      );
    }
    
    if (showPolicy && !isInTrajectory) {
      const policyArrow = policy.get(`${x}-${y}`);
      if (policyArrow) {
        return (
          <div 
            className="text-gray-400 opacity-50"
            data-testid={`policy-arrow-${x}-${y}`}
          >
            {getActionIcon(policyArrow.action)}
          </div>
        );
      }
    }
    
    return null;
  };

  const getCellClasses = (x: number, y: number) => {
    const isAgent = state.agentPosition.x === x && state.agentPosition.y === y;
    const isGoal = state.goalPosition.x === x && state.goalPosition.y === y;
    const isObstacle = state.obstacles.some(obs => obs.x === x && obs.y === y);
    const isInTrajectory = showTrajectory && state.trajectory.some(
      pos => pos.x === x && pos.y === y
    );
    const isSelected = selectedCell?.x === x && selectedCell?.y === y;
    const isAdjacent = Math.abs(state.agentPosition.x - x) + 
                       Math.abs(state.agentPosition.y - y) === 1;
    
    const classes = [
      'relative w-16 h-16 border-2 flex items-center justify-center cursor-pointer transition-all',
    ];
    
    if (isAgent) classes.push('agent');
    if (isGoal) classes.push('goal', 'bg-green-100 border-green-400');
    else if (isObstacle) classes.push('obstacle', 'bg-gray-200 border-gray-400');
    else if (isInTrajectory && !isAgent) classes.push('trajectory', 'bg-blue-50');
    else if (isSelected) classes.push('bg-yellow-100 border-yellow-400');
    else if (isAdjacent && interactive && !state.isDone) {
      classes.push('hover:bg-blue-100 border-blue-300');
    } else {
      classes.push('bg-white border-gray-300 hover:bg-gray-50');
    }
    
    return classes.join(' ');
  };

  return (
    <div className="w-full max-w-4xl mx-auto p-4">
      {/* Controls */}
      <div className="mb-4 flex flex-wrap gap-2">
        <button
          onClick={handleReset}
          className="px-4 py-2 bg-gray-600 text-white rounded-lg hover:bg-gray-700 flex items-center gap-2"
        >
          <RotateCcw className="w-4 h-4" />
          Reset
        </button>
        
        <button
          onClick={() => setIsAutoPlaying(!isAutoPlaying)}
          disabled={state.isDone}
          className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:bg-gray-400 flex items-center gap-2"
        >
          {isAutoPlaying ? (
            <>
              <Pause className="w-4 h-4" />
              Pause
            </>
          ) : (
            <>
              <Play className="w-4 h-4" />
              Auto Play
            </>
          )}
        </button>
      </div>
      
      {/* Status Display */}
      <div className="mb-4 grid grid-cols-2 md:grid-cols-4 gap-4">
        <div className="bg-white rounded-lg p-3 shadow">
          <div className="text-sm text-gray-600">Total Reward:</div>
          <div className="text-2xl font-bold">{state.totalReward}</div>
        </div>
        <div className="bg-white rounded-lg p-3 shadow">
          <div className="text-sm text-gray-600">Steps:</div>
          <div className="text-2xl font-bold">{state.stepCount}</div>
        </div>
        <div className="bg-white rounded-lg p-3 shadow">
          <div className="text-sm text-gray-600">Status:</div>
          <div className="text-lg font-medium">
            {state.isDone ? '🎯 Goal!' : '🏃 Active'}
          </div>
        </div>
        <div className="bg-white rounded-lg p-3 shadow">
          <div className="text-sm text-gray-600">Position:</div>
          <div className="text-lg font-mono">
            ({state.agentPosition.x}, {state.agentPosition.y})
          </div>
        </div>
      </div>
      
      {/* Grid */}
      <div className="bg-gray-100 p-4 rounded-lg shadow-lg">
        <div 
          className="grid gap-1 mx-auto"
          style={{
            gridTemplateColumns: `repeat(${width}, minmax(0, 1fr))`,
            width: 'fit-content',
          }}
        >
          <AnimatePresence>
            {Array.from({ length: height }, (_, y) =>
              Array.from({ length: width }, (_, x) => (
                <motion.div
                  key={`${x}-${y}`}
                  data-testid={`grid-cell-${x}-${y}`}
                  className={getCellClasses(x, y)}
                  onClick={() => handleCellClick(x, y)}
                  onContextMenu={(e) => {
                    e.preventDefault();
                    toggleObstacle(x, y);
                  }}
                  whileHover={{ scale: 1.05 }}
                  whileTap={{ scale: 0.95 }}
                >
                  {getCellContent(x, y)}
                  
                  {showRewards && (
                    <div className="absolute bottom-0 right-0 text-xs text-gray-500">
                      {environment.getRewardAt(x, y)}
                    </div>
                  )}
                </motion.div>
              ))
            )}
          </AnimatePresence>
        </div>
      </div>
      
      {/* Instructions */}
      <div className="mt-4 text-sm text-gray-600">
        <p>• Use arrow keys or click adjacent cells to move</p>
        <p>• Right-click to add/remove obstacles</p>
        <p>• Reach the green target to win!</p>
        {showPolicy && <p>• Arrows show the current policy</p>}
        {showTrajectory && <p>• Blue cells show your path</p>}
      </div>
      
      {/* Policy Visualization Legend */}
      {showPolicy && (
        <div className="mt-4 p-3 bg-white rounded-lg shadow">
          <h3 className="font-medium mb-2">Policy Visualization</h3>
          <p className="text-sm text-gray-600">
            Arrows indicate the recommended action at each position
          </p>
        </div>
      )}
    </div>
  );
};