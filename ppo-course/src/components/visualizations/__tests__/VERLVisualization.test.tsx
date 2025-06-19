import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { VERLVisualization } from '../VERLVisualization';

// Mock framer-motion
jest.mock('framer-motion', () => ({
  motion: {
    div: ({ children, whileHover, whileTap, ...props }: any) => <div {...props}>{children}</div>,
    path: ({ children, ...props }: any) => <path {...props}>{children}</path>,
    circle: ({ children, ...props }: any) => <circle {...props}>{children}</circle>,
    line: ({ children, animate, transition, ...props }: any) => <line {...props}>{children}</line>,
  },
  AnimatePresence: ({ children }: any) => <>{children}</>,
}));

describe('VERLVisualization', () => {
  test('renders VERL architecture components', () => {
    render(<VERLVisualization />);
    
    // Check for main components
    expect(screen.getByText('Actor')).toBeInTheDocument();
    expect(screen.getByText('Critic')).toBeInTheDocument();
    expect(screen.getByText('Rollout')).toBeInTheDocument();
    expect(screen.getByText('Reference Policy')).toBeInTheDocument();
    expect(screen.getByText('Reward Model')).toBeInTheDocument();
  });

  test('shows component descriptions on hover', async () => {
    render(<VERLVisualization />);
    
    const actorComponent = screen.getByTestId('verl-actor');
    fireEvent.mouseEnter(actorComponent);
    
    await waitFor(() => {
      expect(screen.getByText(/Generates actions and updates policy/)).toBeInTheDocument();
    });
  });

  test('displays data flow between components', () => {
    render(<VERLVisualization />);
    
    // Check for data flow arrows
    const dataFlows = screen.getAllByTestId(/data-flow-/);
    expect(dataFlows.length).toBeGreaterThan(0);
  });

  test('shows GPU allocation information', () => {
    render(<VERLVisualization />);
    
    expect(screen.getByText(/GPU allocation/i)).toBeInTheDocument();
    expect(screen.getAllByText(/4x GPUs/).length).toBeGreaterThan(0);
  });

  test('toggles between static and animated views', () => {
    render(<VERLVisualization />);
    
    const toggleButton = screen.getByText(/animate/i);
    fireEvent.click(toggleButton);
    
    expect(screen.getByTestId('animation-indicator')).toBeInTheDocument();
  });

  test('shows tensor parallelism configuration', () => {
    render(<VERLVisualization />);
    
    expect(screen.getAllByText(/TP: 4x/).length).toBeGreaterThan(0);
    expect(screen.getAllByText(/PP: 2x/).length).toBeGreaterThan(0);
  });

  test('displays communication patterns', () => {
    render(<VERLVisualization />);
    
    const communicationPaths = screen.getAllByTestId(/communication-/);
    expect(communicationPaths.length).toBeGreaterThan(0);
  });

  test('shows performance metrics', () => {
    render(<VERLVisualization />);
    
    expect(screen.getByText(/Throughput/)).toBeInTheDocument();
    expect(screen.getByText(/Memory Usage/)).toBeInTheDocument();
    expect(screen.getByText(/tokens\/sec/)).toBeInTheDocument();
  });

  test('highlights component interactions', async () => {
    render(<VERLVisualization />);
    
    const actorComponent = screen.getByTestId('verl-actor');
    fireEvent.click(actorComponent);
    
    await waitFor(() => {
      expect(screen.getByTestId('interaction-highlight')).toBeInTheDocument();
    });
  });

  test('displays resource utilization', () => {
    render(<VERLVisualization />);
    
    expect(screen.getByText(/CPU: 40%/)).toBeInTheDocument();
    expect(screen.getByText(/GPU: 85%/)).toBeInTheDocument();
    expect(screen.getByText(/Memory: 72%/)).toBeInTheDocument();
  });

  test('shows distributed training workflow', () => {
    render(<VERLVisualization />);
    
    const workflowSteps = screen.getAllByTestId(/workflow-step-/);
    expect(workflowSteps.length).toBeGreaterThanOrEqual(4);
  });
});