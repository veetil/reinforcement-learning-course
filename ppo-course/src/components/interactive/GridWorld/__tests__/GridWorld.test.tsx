import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import '@testing-library/jest-dom';
import { GridWorld } from '../GridWorld';
import { GridEnvironment, GridState, Action } from '../GridEnvironment';

describe('GridEnvironment', () => {
  let env: GridEnvironment;

  beforeEach(() => {
    env = new GridEnvironment(5, 5);
  });

  describe('initialization', () => {
    test('should create a grid with specified dimensions', () => {
      expect(env.width).toBe(5);
      expect(env.height).toBe(5);
    });

    test('should initialize agent at starting position', () => {
      const state = env.getState();
      expect(state.agentPosition).toEqual({ x: 0, y: 0 });
    });

    test('should set goal position', () => {
      const state = env.getState();
      expect(state.goalPosition).toEqual({ x: 4, y: 4 });
    });

    test('should initialize with zero reward', () => {
      const state = env.getState();
      expect(state.totalReward).toBe(0);
    });

    test('should not be done initially', () => {
      const state = env.getState();
      expect(state.isDone).toBe(false);
    });
  });

  describe('actions', () => {
    test('should move agent up', () => {
      env.reset();
      env.step(Action.DOWN); // Move to (0, 1) first
      const result = env.step(Action.UP);
      
      expect(result.state.agentPosition).toEqual({ x: 0, y: 0 });
      expect(result.reward).toBe(-1); // Step penalty
    });

    test('should move agent down', () => {
      const result = env.step(Action.DOWN);
      
      expect(result.state.agentPosition).toEqual({ x: 0, y: 1 });
      expect(result.reward).toBe(-1);
    });

    test('should move agent left', () => {
      env.step(Action.RIGHT); // Move to (1, 0) first
      const result = env.step(Action.LEFT);
      
      expect(result.state.agentPosition).toEqual({ x: 0, y: 0 });
      expect(result.reward).toBe(-1);
    });

    test('should move agent right', () => {
      const result = env.step(Action.RIGHT);
      
      expect(result.state.agentPosition).toEqual({ x: 1, y: 0 });
      expect(result.reward).toBe(-1);
    });

    test('should not move beyond grid boundaries', () => {
      const result = env.step(Action.UP); // Already at top
      
      expect(result.state.agentPosition).toEqual({ x: 0, y: 0 });
      expect(result.reward).toBe(-1); // Still get step penalty
    });

    test('should handle obstacles', () => {
      env.addObstacle(1, 0);
      const result = env.step(Action.RIGHT);
      
      expect(result.state.agentPosition).toEqual({ x: 0, y: 0 }); // Didn't move
      expect(result.reward).toBe(-5); // Collision penalty
    });
  });

  describe('goal and rewards', () => {
    test('should give positive reward when reaching goal', () => {
      // Move to goal position (4, 4)
      for (let i = 0; i < 4; i++) {
        env.step(Action.RIGHT);
        env.step(Action.DOWN);
      }
      
      const state = env.getState();
      expect(state.agentPosition).toEqual({ x: 4, y: 4 });
      expect(state.isDone).toBe(true);
      expect(state.totalReward).toBeGreaterThan(0); // Should include goal reward
    });

    test('should accumulate rewards over episode', () => {
      env.step(Action.RIGHT);
      env.step(Action.DOWN);
      
      const state = env.getState();
      expect(state.totalReward).toBe(-2); // Two step penalties
    });
  });

  describe('reset', () => {
    test('should reset environment to initial state', () => {
      env.step(Action.RIGHT);
      env.step(Action.DOWN);
      
      const newState = env.reset();
      
      expect(newState.agentPosition).toEqual({ x: 0, y: 0 });
      expect(newState.totalReward).toBe(0);
      expect(newState.isDone).toBe(false);
      expect(newState.stepCount).toBe(0);
    });
  });
});

describe('GridWorld Component', () => {
  test('should render grid with correct dimensions', () => {
    render(<GridWorld width={5} height={5} />);
    
    const cells = screen.getAllByTestId(/grid-cell-/);
    expect(cells).toHaveLength(25);
  });

  test('should show agent at starting position', () => {
    render(<GridWorld width={5} height={5} />);
    
    const agentCell = screen.getByTestId('grid-cell-0-0');
    expect(agentCell).toHaveClass('agent');
  });

  test('should show goal position', () => {
    render(<GridWorld width={5} height={5} />);
    
    const goalCell = screen.getByTestId('grid-cell-4-4');
    expect(goalCell).toHaveClass('goal');
  });

  test('should move agent on cell click', async () => {
    render(<GridWorld width={5} height={5} />);
    
    const targetCell = screen.getByTestId('grid-cell-1-0');
    fireEvent.click(targetCell);
    
    await waitFor(() => {
      expect(targetCell).toHaveClass('agent');
    });
  });

  test('should show reward display', () => {
    render(<GridWorld width={5} height={5} />);
    
    expect(screen.getByText(/Total Reward:/)).toBeInTheDocument();
    expect(screen.getByText(/Steps:/)).toBeInTheDocument();
  });

  test('should update reward on movement', async () => {
    render(<GridWorld width={5} height={5} />);
    
    const targetCell = screen.getByTestId('grid-cell-1-0');
    fireEvent.click(targetCell);
    
    await waitFor(() => {
      expect(screen.getByText(/-1/)).toBeInTheDocument(); // Step penalty
    });
  });

  test('should show policy visualization when enabled', () => {
    render(<GridWorld width={5} height={5} showPolicy={true} />);
    
    expect(screen.getByText(/Policy Visualization/)).toBeInTheDocument();
    const arrows = screen.getAllByTestId(/policy-arrow-/);
    expect(arrows.length).toBeGreaterThan(0);
  });

  test('should reset on button click', async () => {
    render(<GridWorld width={5} height={5} />);
    
    // Move agent
    const targetCell = screen.getByTestId('grid-cell-1-0');
    fireEvent.click(targetCell);
    
    // Reset
    const resetButton = screen.getByText('Reset');
    fireEvent.click(resetButton);
    
    await waitFor(() => {
      const agentCell = screen.getByTestId('grid-cell-0-0');
      expect(agentCell).toHaveClass('agent');
      // Check that the reward value is 0 (they're in separate elements)
      const rewardElements = screen.getAllByText('0');
      expect(rewardElements.length).toBeGreaterThan(0);
    });
  });

  test('should show trajectory when enabled', async () => {
    render(<GridWorld width={5} height={5} showTrajectory={true} />);
    
    // Move agent
    const cell1 = screen.getByTestId('grid-cell-1-0');
    fireEvent.click(cell1);
    
    const cell2 = screen.getByTestId('grid-cell-1-1');
    fireEvent.click(cell2);
    
    await waitFor(() => {
      expect(cell1).toHaveClass('trajectory');
    });
  });

  test('should handle keyboard controls', async () => {
    render(<GridWorld width={5} height={5} />);
    
    fireEvent.keyDown(document, { key: 'ArrowRight' });
    
    await waitFor(() => {
      const agentCell = screen.getByTestId('grid-cell-1-0');
      expect(agentCell).toHaveClass('agent');
    });
  });
});