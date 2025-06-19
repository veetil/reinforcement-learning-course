import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { NeuralNetworkDesigner } from '../NeuralNetworkDesigner';

// Mock framer-motion
jest.mock('framer-motion', () => ({
  motion: {
    div: ({ children, whileHover, whileTap, drag, dragMomentum, onDragEnd, animate, transition, ...props }: any) => <div {...props}>{children}</div>,
    circle: ({ children, animate, transition, ...props }: any) => <circle {...props}>{children}</circle>,
    line: ({ children, animate, transition, ...props }: any) => <line {...props}>{children}</line>,
    path: ({ children, ...props }: any) => <path {...props}>{children}</path>,
  },
  AnimatePresence: ({ children }: any) => <>{children}</>,
}));

describe('NeuralNetworkDesigner', () => {
  test('renders network designer interface', () => {
    render(<NeuralNetworkDesigner />);
    
    expect(screen.getByText('Neural Network Designer')).toBeInTheDocument();
    expect(screen.getByText('Add Layer')).toBeInTheDocument();
    expect(screen.getByText('Clear All')).toBeInTheDocument();
  });

  test('shows layer configuration panel', () => {
    render(<NeuralNetworkDesigner />);
    
    expect(screen.getByText('Layer Configuration')).toBeInTheDocument();
    expect(screen.getByLabelText(/Layer Type/)).toBeInTheDocument();
    expect(screen.getByLabelText(/Neurons/)).toBeInTheDocument();
    expect(screen.getByLabelText(/Activation/)).toBeInTheDocument();
  });

  test('adds new layer when button clicked', () => {
    render(<NeuralNetworkDesigner />);
    
    const addButton = screen.getByText('Add Layer');
    fireEvent.click(addButton);
    
    expect(screen.getByTestId('layer-0')).toBeInTheDocument();
  });

  test('displays network architecture visualization', () => {
    render(<NeuralNetworkDesigner />);
    
    // Add a layer first
    fireEvent.click(screen.getByText('Add Layer'));
    
    expect(screen.getByTestId('network-svg')).toBeInTheDocument();
    expect(screen.getByTestId('layer-0')).toBeInTheDocument();
  });

  test('allows selecting different layer types', () => {
    render(<NeuralNetworkDesigner />);
    
    const layerTypeSelect = screen.getByLabelText(/Layer Type/);
    fireEvent.change(layerTypeSelect, { target: { value: 'conv2d' } });
    
    expect(layerTypeSelect).toHaveValue('conv2d');
  });

  test('allows configuring neuron count', () => {
    render(<NeuralNetworkDesigner />);
    
    const neuronInput = screen.getByLabelText(/Neurons/);
    fireEvent.change(neuronInput, { target: { value: '128' } });
    
    expect(neuronInput).toHaveValue(128);
  });

  test('allows selecting activation functions', () => {
    render(<NeuralNetworkDesigner />);
    
    const activationSelect = screen.getByLabelText(/Activation/);
    fireEvent.change(activationSelect, { target: { value: 'relu' } });
    
    expect(activationSelect).toHaveValue('relu');
  });

  test('removes layer when delete button clicked', async () => {
    render(<NeuralNetworkDesigner />);
    
    // Add a layer
    fireEvent.click(screen.getByText('Add Layer'));
    expect(screen.getByTestId('layer-0')).toBeInTheDocument();
    
    // Remove the layer
    const deleteButton = screen.getByTestId('delete-layer-0');
    fireEvent.click(deleteButton);
    
    await waitFor(() => {
      expect(screen.queryByTestId('layer-0')).not.toBeInTheDocument();
    });
  });

  test('clears all layers when clear button clicked', async () => {
    render(<NeuralNetworkDesigner />);
    
    // Add multiple layers
    fireEvent.click(screen.getByText('Add Layer'));
    fireEvent.click(screen.getByText('Add Layer'));
    
    expect(screen.getByTestId('layer-0')).toBeInTheDocument();
    expect(screen.getByTestId('layer-1')).toBeInTheDocument();
    
    // Clear all
    fireEvent.click(screen.getByText('Clear All'));
    
    await waitFor(() => {
      expect(screen.queryByTestId('layer-0')).not.toBeInTheDocument();
      expect(screen.queryByTestId('layer-1')).not.toBeInTheDocument();
    });
  });

  test('displays network summary', () => {
    render(<NeuralNetworkDesigner />);
    
    expect(screen.getByText('Network Summary')).toBeInTheDocument();
    expect(screen.getByText(/Total Layers:/)).toBeInTheDocument();
    expect(screen.getByText(/Parameters:/)).toBeInTheDocument();
  });

  test('shows forward pass animation', () => {
    render(<NeuralNetworkDesigner />);
    
    // Add a layer first
    fireEvent.click(screen.getByText('Add Layer'));
    
    const animateButton = screen.getByText('Animate Forward Pass');
    fireEvent.click(animateButton);
    
    expect(screen.getByTestId('forward-pass-animation')).toBeInTheDocument();
  });

  test('allows editing layer properties', () => {
    render(<NeuralNetworkDesigner />);
    
    // Add a layer
    fireEvent.click(screen.getByText('Add Layer'));
    
    // Click on the layer to edit
    const layer = screen.getByTestId('layer-0');
    fireEvent.click(layer);
    
    expect(screen.getByTestId('layer-editor')).toBeInTheDocument();
  });

  test('exports network configuration', () => {
    render(<NeuralNetworkDesigner />);
    
    // Add a layer
    fireEvent.click(screen.getByText('Add Layer'));
    
    const exportButton = screen.getByText('Export Config');
    fireEvent.click(exportButton);
    
    expect(screen.getByTestId('export-modal')).toBeInTheDocument();
  });

  test('imports network configuration', () => {
    render(<NeuralNetworkDesigner />);
    
    const importButton = screen.getByText('Import Config');
    fireEvent.click(importButton);
    
    expect(screen.getByTestId('import-modal')).toBeInTheDocument();
  });

  test('calculates parameter count correctly', () => {
    render(<NeuralNetworkDesigner />);
    
    // Add a dense layer with 64 neurons
    const neuronInput = screen.getByLabelText(/Neurons/);
    fireEvent.change(neuronInput, { target: { value: '64' } });
    fireEvent.click(screen.getByText('Add Layer'));
    
    // Check parameter calculation for first layer (784 input + 1 bias) * 64 neurons = 50,240
    expect(screen.getByText(/Parameters: 50,240/)).toBeInTheDocument();
  });
});