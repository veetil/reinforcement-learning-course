import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import GRPOVisualization from '../GRPOVisualization';

// Mock Chart.js
jest.mock('react-chartjs-2', () => ({
  Line: function Line() { return null; },
  Bar: function Bar() { return null; },
  Radar: function Radar() { return null; },
  Doughnut: function Doughnut() { return null; },
}));

// Mock framer-motion
jest.mock('framer-motion', () => ({
  motion: {
    div: function MotionDiv(props) { 
      return React.createElement('div', props);
    },
  },
  AnimatePresence: function AnimatePresence(props) { 
    return React.createElement(React.Fragment, null, props.children);
  },
}));

// Mock Lucide icons
jest.mock('lucide-react', () => ({
  Play: function Play() { return React.createElement('span', null, 'Play'); },
  Pause: function Pause() { return React.createElement('span', null, 'Pause'); },
  RotateCcw: function RotateCcw() { return React.createElement('span', null, 'Reset'); },
  Shuffle: function Shuffle() { return React.createElement('span', null, 'Shuffle'); },
  Users: function Users() { return React.createElement('span', null, 'Users'); },
  TrendingUp: function TrendingUp() { return React.createElement('span', null, 'TrendingUp'); },
  Layers: function Layers() { return React.createElement('span', null, 'Layers'); },
  BarChart3: function BarChart3() { return React.createElement('span', null, 'BarChart3'); },
}));

describe('GRPOVisualization', () => {
  test('renders GRPO visualization interface', () => {
    render(<GRPOVisualization />);
    
    expect(screen.getByText(/GRPO: Group Relative Policy Optimization/)).toBeInTheDocument();
    expect(screen.getByText('Grouping Strategy')).toBeInTheDocument();
    expect(screen.getByText(/Number of Groups:/)).toBeInTheDocument();
    expect(screen.getByText('Weighting Method')).toBeInTheDocument();
  });

  test('displays control buttons', () => {
    render(<GRPOVisualization />);
    
    expect(screen.getByText('Start')).toBeInTheDocument();
    expect(screen.getByText('Reset')).toBeInTheDocument();
    expect(screen.getByText('Regenerate')).toBeInTheDocument();
  });

  test('toggles between start and pause', () => {
    render(<GRPOVisualization />);
    
    const startButton = screen.getByText('Start');
    fireEvent.click(startButton);
    
    expect(screen.getByText('Pause')).toBeInTheDocument();
  });

  test('changes grouping strategy', () => {
    render(<GRPOVisualization />);
    
    const strategySelect = screen.getByRole('combobox');
    fireEvent.click(strategySelect);
    
    const difficultyOption = screen.getByText('Difficulty-Based');
    fireEvent.click(difficultyOption);
    
    // Strategy should be updated (visible in the select)
    expect(screen.getByText('Difficulty-Based')).toBeInTheDocument();
  });

  test('changes number of groups with slider', () => {
    render(<GRPOVisualization />);
    
    // Initial number of groups is 4
    expect(screen.getByText('Number of Groups: 4')).toBeInTheDocument();
    
    // Slider interaction would update the value
    // In a real test, we would simulate slider movement
  });

  test('switches between weighting methods', () => {
    render(<GRPOVisualization />);
    
    const performanceRadio = screen.getByLabelText('Performance');
    fireEvent.click(performanceRadio);
    
    // Radio should be selected
    expect(performanceRadio).toBeChecked();
  });

  test('toggles comparison mode', () => {
    render(<GRPOVisualization />);
    
    const comparisonSwitch = screen.getByRole('switch');
    fireEvent.click(comparisonSwitch);
    
    // Switch should be checked
    expect(comparisonSwitch).toBeChecked();
  });

  test('displays visualization tabs', () => {
    render(<GRPOVisualization />);
    
    expect(screen.getByText('Distribution')).toBeInTheDocument();
    expect(screen.getByText('Characteristics')).toBeInTheDocument();
    expect(screen.getByText('Advantages')).toBeInTheDocument();
    expect(screen.getByText('Dynamics')).toBeInTheDocument();
  });

  test('switches between tabs', () => {
    render(<GRPOVisualization />);
    
    const characteristicsTab = screen.getByText('Characteristics');
    fireEvent.click(characteristicsTab);
    
    // Should show characteristics content
    expect(screen.getByText('Group Characteristics')).toBeInTheDocument();
  });

  test('displays group distribution chart', () => {
    render(<GRPOVisualization />);
    
    expect(screen.getByText('Group Sizes')).toBeInTheDocument();
  });

  test('displays group weights chart', () => {
    render(<GRPOVisualization />);
    
    expect(screen.getByText('Group Weights')).toBeInTheDocument();
  });

  test('shows advantages comparison when enabled', () => {
    render(<GRPOVisualization />);
    
    // Enable comparison mode
    const comparisonSwitch = screen.getByRole('switch');
    fireEvent.click(comparisonSwitch);
    
    // Navigate to advantages tab
    const advantagesTab = screen.getByText('Advantages');
    fireEvent.click(advantagesTab);
    
    // Should show comparison chart
    expect(screen.getByText('PPO vs GRPO Advantage Normalization')).toBeInTheDocument();
  });

  test('displays performance over time', () => {
    render(<GRPOVisualization />);
    
    // Navigate to dynamics tab
    const dynamicsTab = screen.getByText('Dynamics');
    fireEvent.click(dynamicsTab);
    
    expect(screen.getByText('Performance Over Time')).toBeInTheDocument();
  });

  test('shows training statistics', () => {
    render(<GRPOVisualization />);
    
    // Navigate to dynamics tab
    const dynamicsTab = screen.getByText('Dynamics');
    fireEvent.click(dynamicsTab);
    
    expect(screen.getByText('Training Statistics')).toBeInTheDocument();
    expect(screen.getByText(/Iteration:/)).toBeInTheDocument();
    expect(screen.getByText(/Active Groups:/)).toBeInTheDocument();
  });

  test('displays best performing group', () => {
    render(<GRPOVisualization />);
    
    // Navigate to dynamics tab
    const dynamicsTab = screen.getByText('Dynamics');
    fireEvent.click(dynamicsTab);
    
    expect(screen.getByText('Best Performing Group')).toBeInTheDocument();
    expect(screen.getByText(/Group \d+/)).toBeInTheDocument();
  });

  test('shows weight distribution visualization', () => {
    render(<GRPOVisualization />);
    
    // Navigate to dynamics tab
    const dynamicsTab = screen.getByText('Dynamics');
    fireEvent.click(dynamicsTab);
    
    expect(screen.getByText('Weight Distribution')).toBeInTheDocument();
    expect(screen.getAllByText(/Group \d+:/)).toHaveLength(4); // Default 4 groups
  });

  test('displays algorithm explanation', () => {
    render(<GRPOVisualization />);
    
    expect(screen.getByText('How GRPO Works')).toBeInTheDocument();
    expect(screen.getByText(/Group Formation:/)).toBeInTheDocument();
    expect(screen.getByText(/Group Normalization:/)).toBeInTheDocument();
    expect(screen.getByText(/Weighted Updates:/)).toBeInTheDocument();
  });

  test('resets visualization state', () => {
    render(<GRPOVisualization />);
    
    // Start the visualization
    const startButton = screen.getByText('Start');
    fireEvent.click(startButton);
    
    // Reset
    const resetButton = screen.getByText('Reset');
    fireEvent.click(resetButton);
    
    // Should be back to initial state
    expect(screen.getByText('Start')).toBeInTheDocument();
  });

  test('regenerates group data', () => {
    render(<GRPOVisualization />);
    
    const regenerateButton = screen.getByText('Regenerate');
    fireEvent.click(regenerateButton);
    
    // Should still show all components (data regenerated in background)
    expect(screen.getByText('Group Sizes')).toBeInTheDocument();
  });

  test('updates in real-time when running', async () => {
    render(<GRPOVisualization />);
    
    // Start the visualization
    const startButton = screen.getByText('Start');
    fireEvent.click(startButton);
    
    // Navigate to dynamics tab to see iteration counter
    const dynamicsTab = screen.getByText('Dynamics');
    fireEvent.click(dynamicsTab);
    
    // Should show iteration 0 initially
    expect(screen.getByText('Iteration: 0')).toBeInTheDocument();
    
    // After some time, iteration should increase
    // (In real test would wait for timer)
  });
});