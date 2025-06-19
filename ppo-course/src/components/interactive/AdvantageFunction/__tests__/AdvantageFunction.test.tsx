import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import '@testing-library/jest-dom';
import { AdvantageFunctionVisualizer } from '../AdvantageFunctionVisualizer';
import { AdvantageCalculator } from '../AdvantageCalculator';

describe('AdvantageCalculator', () => {
  let calculator: AdvantageCalculator;

  beforeEach(() => {
    calculator = new AdvantageCalculator();
  });

  test('should calculate advantage correctly', () => {
    const qValue = 10;
    const vValue = 7;
    const advantage = calculator.calculateAdvantage(qValue, vValue);
    expect(advantage).toBe(3);
  });

  test('should calculate Q-values for all actions', () => {
    const state = { x: 2, y: 2 };
    const nextStateValues = {
      up: 0.9,
      down: 0.7,
      left: 0.8,
      right: 0.85
    };
    const reward = -1;
    const gamma = 0.9;

    const qValues = calculator.calculateQValues(state, nextStateValues, reward, gamma);
    
    expect(qValues.up).toBeCloseTo(reward + gamma * nextStateValues.up);
    expect(qValues.down).toBeCloseTo(reward + gamma * nextStateValues.down);
    expect(qValues.left).toBeCloseTo(reward + gamma * nextStateValues.left);
    expect(qValues.right).toBeCloseTo(reward + gamma * nextStateValues.right);
  });

  test('should identify optimal action', () => {
    const qValues = {
      up: 5,
      down: 3,
      left: 4,
      right: 6
    };
    
    const optimal = calculator.getOptimalAction(qValues);
    expect(optimal).toBe('right');
  });

  test('should calculate GAE correctly', () => {
    const rewards = [1, 2, 3];
    const values = [10, 12, 15, 20];
    const gamma = 0.9;
    const lambda = 0.95;

    const gae = calculator.calculateGAE(rewards, values, gamma, lambda);
    
    expect(gae).toHaveLength(3);
    // First step: delta = 1 + 0.9*12 - 10 = 1.8
    // GAE accumulates backwards, so check general properties
    expect(gae[0]).toBeGreaterThan(0); // Positive advantage
    expect(gae[2]).toBeGreaterThan(0); // Last step advantage
  });

  test('should handle terminal states in GAE', () => {
    const rewards = [1, 2, 100]; // Big reward at end
    const values = [10, 12, 15, 0]; // Terminal state has 0 value
    const gamma = 0.9;
    const lambda = 0.95;
    const dones = [false, false, true];

    const gae = calculator.calculateGAEWithDones(rewards, values, gamma, lambda, dones);
    
    expect(gae[2]).toBeCloseTo(85, 0); // Big positive advantage for terminal reward
  });
});

describe('AdvantageFunctionVisualizer Component', () => {
  test('should render grid with correct dimensions', () => {
    render(<AdvantageFunctionVisualizer gridSize={4} />);
    
    const cells = screen.getAllByTestId(/advantage-cell-/);
    expect(cells).toHaveLength(16);
  });

  test('should show Q-values for selected state', async () => {
    render(<AdvantageFunctionVisualizer gridSize={4} />);
    
    // First set a goal
    const goalCell = screen.getByTestId('advantage-cell-3-3');
    fireEvent.click(goalCell);
    
    // Then select another cell
    const cell = screen.getByTestId('advantage-cell-2-2');
    fireEvent.click(cell);
    
    await waitFor(() => {
      expect(screen.getByText('Q-Values')).toBeInTheDocument();
    });
  });

  test('should display advantage values', async () => {
    render(<AdvantageFunctionVisualizer gridSize={4} showAdvantage={true} />);
    
    // First set a goal
    const goalCell = screen.getByTestId('advantage-cell-3-3');
    fireEvent.click(goalCell);
    
    // Then select another cell
    const cell = screen.getByTestId('advantage-cell-1-1');
    fireEvent.click(cell);
    
    await waitFor(() => {
      expect(screen.getByTestId('advantage-display')).toBeInTheDocument();
    });
  });

  test('should highlight positive and negative advantages', async () => {
    render(<AdvantageFunctionVisualizer gridSize={4} />);
    
    // Set goal to create value gradient
    const goalCell = screen.getByTestId('advantage-cell-3-3');
    fireEvent.click(goalCell);
    
    // Select a state to see advantages
    const stateCell = screen.getByTestId('advantage-cell-1-1');
    fireEvent.click(stateCell);
    
    await waitFor(() => {
      const advantages = screen.getAllByTestId(/advantage-arrow-/);
      expect(advantages.length).toBeGreaterThan(0);
    });
  });

  test('should toggle GAE visualization', async () => {
    render(<AdvantageFunctionVisualizer gridSize={4} />);
    
    // Enable trajectory mode and create a trajectory
    const trajButton = screen.getByText('Trajectory Mode');
    fireEvent.click(trajButton);
    
    // Create trajectory
    fireEvent.click(screen.getByTestId('advantage-cell-0-0'));
    fireEvent.click(screen.getByTestId('advantage-cell-1-0'));
    
    // Now show GAE
    const gaeButton = screen.getByText('Show GAE');
    fireEvent.click(gaeButton);
    
    await waitFor(() => {
      expect(screen.getByTestId('gae-panel')).toBeInTheDocument();
    });
  });

  test('should update lambda parameter', async () => {
    render(<AdvantageFunctionVisualizer gridSize={4} />);
    
    const slider = screen.getByLabelText('Lambda (λ)');
    fireEvent.change(slider, { target: { value: '0.8' } });
    
    await waitFor(() => {
      expect(screen.getByText('λ = 0.80')).toBeInTheDocument();
    });
  });

  test('should show trajectory advantages', async () => {
    render(<AdvantageFunctionVisualizer gridSize={4} />);
    
    // Enable trajectory mode
    const trajButton = screen.getByText('Trajectory Mode');
    fireEvent.click(trajButton);
    
    // Click multiple cells to create trajectory
    fireEvent.click(screen.getByTestId('advantage-cell-0-0'));
    fireEvent.click(screen.getByTestId('advantage-cell-1-0'));
    fireEvent.click(screen.getByTestId('advantage-cell-2-0'));
    
    await waitFor(() => {
      expect(screen.getByText('Trajectory Advantages')).toBeInTheDocument();
    });
  });

  test('should compare policies', async () => {
    render(<AdvantageFunctionVisualizer gridSize={4} />);
    
    const compareButton = screen.getByText('Compare Policies');
    fireEvent.click(compareButton);
    
    await waitFor(() => {
      expect(screen.getByText('Policy Comparison')).toBeInTheDocument();
    });
  });

  test('should export advantage data', async () => {
    render(<AdvantageFunctionVisualizer gridSize={4} />);
    
    const exportButton = screen.getByText('Export Data');
    fireEvent.click(exportButton);
    
    await waitFor(() => {
      expect(screen.getByText('Data exported!')).toBeInTheDocument();
    });
  });

  test('should reset visualization', async () => {
    render(<AdvantageFunctionVisualizer gridSize={4} />);
    
    // Select some cells first
    fireEvent.click(screen.getByTestId('advantage-cell-2-2'));
    
    // Reset
    const resetButton = screen.getByText('Reset');
    fireEvent.click(resetButton);
    
    await waitFor(() => {
      const selectedCells = screen.queryAllByTestId(/selected/);
      expect(selectedCells).toHaveLength(0);
    });
  });
});