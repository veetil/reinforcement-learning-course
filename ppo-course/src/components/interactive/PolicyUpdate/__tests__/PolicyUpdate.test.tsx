import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import '@testing-library/jest-dom';
import { PolicyUpdateSimulator } from '../PolicyUpdateSimulator';
import { PPOCalculator } from '../PPOCalculator';

describe('PPOCalculator', () => {
  let calculator: PPOCalculator;

  beforeEach(() => {
    calculator = new PPOCalculator();
  });

  test('should calculate policy ratio correctly', () => {
    const newLogProb = -0.5;
    const oldLogProb = -1.0;
    const ratio = calculator.calculateRatio(newLogProb, oldLogProb);
    expect(ratio).toBeCloseTo(Math.exp(newLogProb - oldLogProb));
  });

  test('should apply PPO clipping correctly', () => {
    const ratio = 1.5;
    const advantage = 0.1;
    const epsilon = 0.2;
    
    const clipped = calculator.clipObjective(ratio, advantage, epsilon);
    expect(clipped).toBe(0.12); // min(1.5 * 0.1, 1.2 * 0.1)
  });

  test('should handle negative advantages in clipping', () => {
    const ratio = 0.5;
    const advantage = -0.1;
    const epsilon = 0.2;
    
    const clipped = calculator.clipObjective(ratio, advantage, epsilon);
    expect(clipped).toBeCloseTo(-0.05); // max(0.5 * -0.1, 0.8 * -0.1) = max(-0.05, -0.08) = -0.05
  });

  test('should calculate KL divergence', () => {
    const oldProbs = [0.25, 0.25, 0.25, 0.25];
    const newProbs = [0.4, 0.3, 0.2, 0.1];
    
    const kl = calculator.calculateKL(oldProbs, newProbs);
    expect(kl).toBeGreaterThan(0);
    expect(kl).toBeLessThan(1);
  });

  test('should calculate entropy correctly', () => {
    const probs = [0.25, 0.25, 0.25, 0.25]; // Maximum entropy
    const entropy = calculator.calculateEntropy(probs);
    expect(entropy).toBeCloseTo(Math.log(4));
  });

  test('should compute policy gradient', () => {
    const advantages = [0.5, -0.3, 0.2];
    const logProbs = [-1.0, -0.5, -0.8];
    
    const gradient = calculator.computePolicyGradient(advantages, logProbs);
    expect(gradient).toHaveLength(3);
  });

  test('should update policy with learning rate', () => {
    const oldPolicy = { mean: 0, std: 1 };
    const gradient = { mean: 0.1, std: -0.05 };
    const learningRate = 0.01;
    
    const newPolicy = calculator.updatePolicy(oldPolicy, gradient, learningRate);
    expect(newPolicy.mean).toBeCloseTo(0.001);
    expect(newPolicy.std).toBeCloseTo(1.0005);
  });
});

describe('PolicyUpdateSimulator Component', () => {
  test('should render with default settings', () => {
    render(<PolicyUpdateSimulator />);
    
    expect(screen.getByText(/PPO Policy Update/)).toBeInTheDocument();
    expect(screen.getByLabelText('Clip Range (ε)')).toBeInTheDocument();
    expect(screen.getByLabelText('Learning Rate')).toBeInTheDocument();
  });

  test('should update epsilon parameter', async () => {
    render(<PolicyUpdateSimulator />);
    
    const slider = screen.getByLabelText('Clip Range (ε)');
    fireEvent.change(slider, { target: { value: '0.3' } });
    
    await waitFor(() => {
      expect(screen.getByText('ε = 0.30')).toBeInTheDocument();
    });
  });

  test('should show policy distribution', () => {
    render(<PolicyUpdateSimulator />);
    
    expect(screen.getByTestId('policy-distribution')).toBeInTheDocument();
  });

  test('should visualize clipping region', () => {
    render(<PolicyUpdateSimulator />);
    
    expect(screen.getByTestId('clipping-visualization')).toBeInTheDocument();
  });

  test('should simulate single update step', async () => {
    render(<PolicyUpdateSimulator />);
    
    const updateButton = screen.getByText('Step Update');
    fireEvent.click(updateButton);
    
    await waitFor(() => {
      expect(screen.getByText(/Episode: 1/)).toBeInTheDocument();
    });
  });

  test('should run continuous training', async () => {
    render(<PolicyUpdateSimulator />);
    
    const trainButton = screen.getByText('Start Training');
    fireEvent.click(trainButton);
    
    await waitFor(() => {
      expect(screen.getByText('Pause Training')).toBeInTheDocument();
    });
  });

  test('should display loss components', async () => {
    render(<PolicyUpdateSimulator />);
    
    // Do an update first to generate loss data
    const updateButton = screen.getByText('Step Update');
    fireEvent.click(updateButton);
    
    await waitFor(() => {
      expect(screen.getByText(/Policy Loss/)).toBeInTheDocument();
      expect(screen.getByText(/Value Loss/)).toBeInTheDocument();
      expect(screen.getByText(/Entropy/)).toBeInTheDocument();
    });
  });

  test('should show KL divergence', async () => {
    render(<PolicyUpdateSimulator />);
    
    // First do an update to generate data
    const updateButton = screen.getByText('Step Update');
    fireEvent.click(updateButton);
    
    await waitFor(() => {
      expect(screen.getByText(/Episode: 1/)).toBeInTheDocument();
    });
    
    const klToggle = screen.getByText('Show KL');
    fireEvent.click(klToggle);
    
    expect(screen.getByTestId('kl-divergence')).toBeInTheDocument();
  });

  test('should handle early stopping', async () => {
    render(<PolicyUpdateSimulator />);
    
    // First show KL divergence panel
    const klToggle = screen.getByText('Show KL');
    fireEvent.click(klToggle);
    
    // Then enable early stopping
    const earlyStopToggle = screen.getByText('Early Stopping');
    fireEvent.click(earlyStopToggle);
    
    // Do an update to generate data
    const updateButton = screen.getByText('Step Update');
    fireEvent.click(updateButton);
    
    await waitFor(() => {
      expect(screen.getByText(/KL Threshold/)).toBeInTheDocument();
    });
  });

  test('should reset simulation', async () => {
    render(<PolicyUpdateSimulator />);
    
    // Run some updates first
    const updateButton = screen.getByText('Step Update');
    fireEvent.click(updateButton);
    
    // Reset
    const resetButton = screen.getByText('Reset');
    fireEvent.click(resetButton);
    
    await waitFor(() => {
      expect(screen.getByText(/Episode: 0/)).toBeInTheDocument();
    });
  });

  test('should export training data', async () => {
    render(<PolicyUpdateSimulator />);
    
    const exportButton = screen.getByText('Export Data');
    fireEvent.click(exportButton);
    
    await waitFor(() => {
      expect(screen.getByText('Data exported!')).toBeInTheDocument();
    });
  });

  test('should compare clipped vs unclipped', async () => {
    render(<PolicyUpdateSimulator />);
    
    // First do an update to generate trajectory
    const updateButton = screen.getByText('Step Update');
    fireEvent.click(updateButton);
    
    await waitFor(() => {
      expect(screen.getByText(/Episode: 1/)).toBeInTheDocument();
    });
    
    const compareToggle = screen.getByText('Compare Clipped/Unclipped');
    fireEvent.click(compareToggle);
    
    expect(screen.getByTestId('comparison-chart')).toBeInTheDocument();
  });
});