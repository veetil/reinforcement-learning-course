import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { DistributedTrainingVisualizer } from '../DistributedTrainingVisualizer';

// Mock framer-motion
jest.mock('framer-motion', () => ({
  motion: {
    div: ({ children, whileHover, whileTap, animate, transition, ...props }: any) => <div {...props}>{children}</div>,
    circle: ({ children, animate, transition, ...props }: any) => <circle {...props}>{children}</circle>,
    line: ({ children, animate, transition, ...props }: any) => <line {...props}>{children}</line>,
    path: ({ children, animate, transition, ...props }: any) => <path {...props}>{children}</path>,
    rect: ({ children, animate, transition, ...props }: any) => <rect {...props}>{children}</rect>,
  },
  AnimatePresence: ({ children }: any) => <>{children}</>,
}));

describe('DistributedTrainingVisualizer', () => {
  test('renders distributed training interface', () => {
    render(<DistributedTrainingVisualizer />);
    
    expect(screen.getByText('Distributed Training Visualizer')).toBeInTheDocument();
    expect(screen.getByText('Start Training')).toBeInTheDocument();
    expect(screen.getByText('Reset')).toBeInTheDocument();
  });

  test('shows cluster configuration panel', () => {
    render(<DistributedTrainingVisualizer />);
    
    expect(screen.getByText('Cluster Configuration')).toBeInTheDocument();
    expect(screen.getByLabelText(/Nodes/)).toBeInTheDocument();
    expect(screen.getByLabelText(/GPUs per Node/)).toBeInTheDocument();
    expect(screen.getByLabelText(/Training Strategy/)).toBeInTheDocument();
  });

  test('displays cluster topology', () => {
    render(<DistributedTrainingVisualizer />);
    
    expect(screen.getByTestId('cluster-topology')).toBeInTheDocument();
    expect(screen.getAllByTestId(/node-/).length).toBeGreaterThan(0);
  });

  test('allows configuring number of nodes', () => {
    render(<DistributedTrainingVisualizer />);
    
    const nodesInput = screen.getByLabelText(/Nodes/);
    fireEvent.change(nodesInput, { target: { value: '4' } });
    
    expect(nodesInput).toHaveValue(4);
  });

  test('allows configuring GPUs per node', () => {
    render(<DistributedTrainingVisualizer />);
    
    const gpusInput = screen.getByLabelText(/GPUs per Node/);
    fireEvent.change(gpusInput, { target: { value: '8' } });
    
    expect(gpusInput).toHaveValue(8);
  });

  test('allows selecting training strategy', () => {
    render(<DistributedTrainingVisualizer />);
    
    const strategySelect = screen.getByLabelText(/Training Strategy/);
    fireEvent.change(strategySelect, { target: { value: 'data_parallel' } });
    
    expect(strategySelect).toHaveValue('data_parallel');
  });

  test('shows communication patterns during training', async () => {
    render(<DistributedTrainingVisualizer />);
    
    const startButton = screen.getByText('Start Training');
    fireEvent.click(startButton);
    
    await waitFor(() => {
      expect(screen.getByTestId('communication-patterns')).toBeInTheDocument();
    });
  });

  test('displays training metrics', () => {
    render(<DistributedTrainingVisualizer />);
    
    expect(screen.getByText(/Throughput/)).toBeInTheDocument();
    const efficiencyElements = screen.getAllByText(/Efficiency/);
    expect(efficiencyElements.length).toBeGreaterThan(0);
    expect(screen.getByText(/Communication Overhead/)).toBeInTheDocument();
    expect(screen.getByText(/samples\/sec/)).toBeInTheDocument();
  });

  test('shows gradient synchronization animation', async () => {
    render(<DistributedTrainingVisualizer />);
    
    const startButton = screen.getByText('Start Training');
    fireEvent.click(startButton);
    
    await waitFor(() => {
      expect(screen.getByTestId('gradient-sync-animation')).toBeInTheDocument();
    });
  });

  test('displays bandwidth utilization', () => {
    render(<DistributedTrainingVisualizer />);
    
    expect(screen.getByText(/Network Bandwidth/)).toBeInTheDocument();
    expect(screen.getByText(/GPU Utilization/)).toBeInTheDocument();
  });

  test('allows switching between different parallelism types', () => {
    render(<DistributedTrainingVisualizer />);
    
    const buttons = ['Data Parallel', 'Model Parallel', 'Pipeline Parallel'];
    buttons.forEach(button => {
      const elements = screen.getAllByText(button);
      expect(elements.length).toBeGreaterThan(0);
    });
  });

  test('shows fault tolerance demonstration', () => {
    render(<DistributedTrainingVisualizer />);
    
    const faultButton = screen.getByText('Simulate Failure');
    fireEvent.click(faultButton);
    
    expect(screen.getByTestId('fault-simulation')).toBeInTheDocument();
  });

  test('displays scaling efficiency charts', () => {
    render(<DistributedTrainingVisualizer />);
    
    const scalingText = screen.getAllByText('Scaling Efficiency');
    expect(scalingText.length).toBeGreaterThan(0);
    expect(screen.getByTestId('efficiency-chart')).toBeInTheDocument();
  });

  test('shows load balancing visualization', () => {
    render(<DistributedTrainingVisualizer />);
    
    expect(screen.getByText('Load Distribution')).toBeInTheDocument();
    expect(screen.getAllByTestId(/load-bar-/).length).toBeGreaterThan(0);
  });

  test('allows exporting training configuration', () => {
    render(<DistributedTrainingVisualizer />);
    
    const exportButton = screen.getByText('Export Config');
    fireEvent.click(exportButton);
    
    expect(screen.getByTestId('export-modal')).toBeInTheDocument();
  });

  test('shows memory usage across nodes', () => {
    render(<DistributedTrainingVisualizer />);
    
    expect(screen.getByText(/Memory Usage/)).toBeInTheDocument();
    expect(screen.getAllByTestId(/memory-usage-/).length).toBeGreaterThan(0);
  });

  test('displays network topology optimization', () => {
    render(<DistributedTrainingVisualizer />);
    
    const optimizeButton = screen.getByText('Optimize Topology');
    fireEvent.click(optimizeButton);
    
    expect(screen.getByTestId('topology-optimization')).toBeInTheDocument();
  });
});