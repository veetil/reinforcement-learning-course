import React from 'react';
import { render, screen, fireEvent } from '@testing-library/react';
import { InteractiveDemo } from '../InteractiveDemo';

// Mock framer-motion
jest.mock('framer-motion', () => ({
  motion: {
    div: ({ children, whileHover, whileTap, ...props }: any) => <div {...props}>{children}</div>,
    circle: ({ children, animate, transition, ...props }: any) => <circle {...props}>{children}</circle>,
    line: ({ children, animate, transition, ...props }: any) => <line {...props}>{children}</line>,
  },
  AnimatePresence: ({ children }: any) => <>{children}</>,
}));

// Mock the visualizations
jest.mock('../../visualizations', () => ({
  VERLVisualization: () => <div data-testid="verl-visualization">VERL Architecture Visualization</div>,
  NeuralNetworkDesigner: () => <div data-testid="neural-network-designer">Neural Network Designer</div>,
  DistributedTrainingVisualizer: () => <div data-testid="distributed-training-visualizer">Distributed Training Visualizer</div>
}));

describe('InteractiveDemo', () => {
  test('renders neural network demo', () => {
    render(<InteractiveDemo demoType="neural-network" />);
    
    expect(screen.getByText('Interactive Demo')).toBeInTheDocument();
    expect(screen.getByText('Play')).toBeInTheDocument();
    expect(screen.getByText('Reset')).toBeInTheDocument();
  });

  test('renders VERL visualization without outer controls', () => {
    render(<InteractiveDemo demoType="verl-architecture" />);
    
    expect(screen.getByTestId('verl-visualization')).toBeInTheDocument();
    expect(screen.queryByText('Interactive Demo')).not.toBeInTheDocument();
    expect(screen.queryByText('Play')).not.toBeInTheDocument();
  });

  test('renders Neural Network Designer without outer controls', () => {
    render(<InteractiveDemo demoType="neural-network-designer" />);
    
    expect(screen.getByTestId('neural-network-designer')).toBeInTheDocument();
    expect(screen.queryByText('Interactive Demo')).not.toBeInTheDocument();
    expect(screen.queryByText('Play')).not.toBeInTheDocument();
  });

  test('renders Distributed Training Visualizer without outer controls', () => {
    render(<InteractiveDemo demoType="distributed-training" />);
    
    expect(screen.getByTestId('distributed-training-visualizer')).toBeInTheDocument();
    expect(screen.queryByText('Interactive Demo')).not.toBeInTheDocument();
    expect(screen.queryByText('Play')).not.toBeInTheDocument();
  });

  test('toggles play/pause for regular demos', () => {
    render(<InteractiveDemo demoType="neural-network" />);
    
    const playButton = screen.getByText('Play');
    fireEvent.click(playButton);
    
    expect(screen.getByText('Pause')).toBeInTheDocument();
  });

  test('renders MDP visualization demo', () => {
    render(<InteractiveDemo demoType="mdp-visualization" />);
    
    expect(screen.getByText('Interactive Demo')).toBeInTheDocument();
    expect(screen.getByText(/Click Play to see MDP transitions/)).toBeInTheDocument();
  });

  test('handles unknown demo type', () => {
    render(<InteractiveDemo demoType="unknown-demo" />);
    
    expect(screen.getByText(/Demo type not found: unknown-demo/)).toBeInTheDocument();
  });

  test('applies custom className', () => {
    render(<InteractiveDemo demoType="neural-network" className="custom-class" />);
    
    const container = screen.getByText('Interactive Demo').closest('div')?.parentElement;
    expect(container).toHaveClass('custom-class');
  });
});