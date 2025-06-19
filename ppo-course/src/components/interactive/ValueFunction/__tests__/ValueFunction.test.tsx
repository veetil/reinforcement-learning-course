import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import '@testing-library/jest-dom';
import { ValueFunctionVisualizer } from '../ValueFunctionVisualizer';
import { ValueFunctionCalculator } from '../ValueFunctionCalculator';

describe('ValueFunctionCalculator', () => {
  let calculator: ValueFunctionCalculator;

  beforeEach(() => {
    calculator = new ValueFunctionCalculator(4, 4);
  });

  test('should initialize with correct dimensions', () => {
    expect(calculator.width).toBe(4);
    expect(calculator.height).toBe(4);
    expect(calculator.values.length).toBe(4);
    expect(calculator.values[0].length).toBe(4);
  });

  test('should set goal and calculate values', () => {
    calculator.setGoal(3, 3);
    calculator.calculateValues();
    
    // Goal should have highest value
    expect(calculator.getValue(3, 3)).toBeCloseTo(1.0);
    
    // Adjacent cells should have discounted values
    expect(calculator.getValue(2, 3)).toBeCloseTo(0.9);
    expect(calculator.getValue(3, 2)).toBeCloseTo(0.9);
  });

  test('should handle obstacles', () => {
    calculator.setGoal(3, 3);
    calculator.addObstacle(2, 2);
    calculator.calculateValues();
    
    // Obstacle should have zero value
    expect(calculator.getValue(2, 2)).toBe(0);
    
    // Path around obstacle should exist
    expect(calculator.getValue(1, 2)).toBeGreaterThan(0);
  });

  test('should use correct discount factor', () => {
    calculator.setGoal(3, 3);
    calculator.setDiscountFactor(0.95);
    calculator.calculateValues();
    
    expect(calculator.getValue(2, 3)).toBeCloseTo(0.95);
    expect(calculator.getValue(1, 3)).toBeCloseTo(0.95 * 0.95);
  });

  test('should converge after iterations', () => {
    calculator.setGoal(3, 3);
    const iterations = calculator.calculateValues();
    
    expect(iterations).toBeLessThan(100);
    expect(iterations).toBeGreaterThan(0);
  });
});

describe('ValueFunctionVisualizer Component', () => {
  test('should render grid with correct dimensions', () => {
    render(<ValueFunctionVisualizer width={4} height={4} />);
    
    const cells = screen.getAllByTestId(/value-cell-/);
    expect(cells).toHaveLength(16);
  });

  test('should show value heatmap', () => {
    render(<ValueFunctionVisualizer width={4} height={4} />);
    
    // Check that cells have background colors
    const cell = screen.getByTestId('value-cell-3-3');
    expect(cell).toHaveStyle({ backgroundColor: expect.any(String) });
  });

  test('should update values when goal changes', async () => {
    render(<ValueFunctionVisualizer width={4} height={4} />);
    
    const cell = screen.getByTestId('value-cell-2-2');
    fireEvent.click(cell);
    
    await waitFor(() => {
      expect(cell).toHaveClass('goal');
    });
  });

  test('should show value labels when enabled', async () => {
    render(<ValueFunctionVisualizer width={4} height={4} showValues={true} />);
    
    // Set a goal first to have non-zero values
    const goalCell = screen.getByTestId('value-cell-3-3');
    fireEvent.click(goalCell);
    
    await waitFor(() => {
      const values = screen.getAllByTestId(/value-label-/);
      expect(values.length).toBeGreaterThan(0);
    });
  });

  test('should animate value changes', async () => {
    render(<ValueFunctionVisualizer width={4} height={4} animated={true} />);
    
    // Set a goal first
    const goalCell = screen.getByTestId('value-cell-2-2');
    fireEvent.click(goalCell);
    
    // Start animation
    const animateButton = screen.getByText('Animate');
    fireEvent.click(animateButton);
    
    // Check that animation is running
    await waitFor(() => {
      expect(screen.getByText(/Animating.../)).toBeInTheDocument();
    });
  });

  test('should handle discount factor slider', async () => {
    render(<ValueFunctionVisualizer width={4} height={4} />);
    
    const slider = screen.getByLabelText('Discount Factor');
    fireEvent.change(slider, { target: { value: '0.8' } });
    
    await waitFor(() => {
      expect(screen.getByText('γ = 0.80')).toBeInTheDocument();
    });
  });

  test('should toggle iteration visualization', async () => {
    render(<ValueFunctionVisualizer width={4} height={4} animated={true} />);
    
    // Set a goal first
    const goalCell = screen.getByTestId('value-cell-2-2');
    fireEvent.click(goalCell);
    
    const button = screen.getByText('Show Iterations');
    fireEvent.click(button);
    
    // Start animation to see iteration info
    const animateButton = screen.getByText('Animate');
    fireEvent.click(animateButton);
    
    await waitFor(() => {
      expect(screen.getByText(/Iteration:/)).toBeInTheDocument();
    });
  });

  test('should reset values on reset button', async () => {
    render(<ValueFunctionVisualizer width={4} height={4} />);
    
    // Set a goal first
    const goalCell = screen.getByTestId('value-cell-2-2');
    fireEvent.click(goalCell);
    
    // Reset
    const resetButton = screen.getByText('Reset');
    fireEvent.click(resetButton);
    
    await waitFor(() => {
      expect(goalCell).not.toHaveClass('goal');
    });
  });

  test('should export value function data', async () => {
    render(<ValueFunctionVisualizer width={4} height={4} />);
    
    const exportButton = screen.getByText('Export Values');
    fireEvent.click(exportButton);
    
    // Check that export message appears
    await waitFor(() => {
      expect(screen.getByText('Values exported!')).toBeInTheDocument();
    });
  });

  test('should show convergence graph', async () => {
    render(<ValueFunctionVisualizer width={4} height={4} showConvergence={true} animated={true} />);
    
    // Set a goal first
    const goalCell = screen.getByTestId('value-cell-2-2');
    fireEvent.click(goalCell);
    
    // Start animation
    const animateButton = screen.getByText('Animate');
    fireEvent.click(animateButton);
    
    // Wait for convergence data
    await waitFor(() => {
      expect(screen.getByTestId('convergence-graph')).toBeInTheDocument();
    }, { timeout: 3000 });
  });
});