'use client';

import React, { useState, useEffect, useMemo } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  Plus, Trash2, Play, Download, Upload, 
  Brain, Layers, Zap, Eye, Settings,
  X, Check, Copy
} from 'lucide-react';

interface NetworkLayer {
  id: string;
  type: 'dense' | 'conv2d' | 'lstm' | 'attention' | 'dropout' | 'batchnorm';
  neurons: number;
  activation: 'relu' | 'sigmoid' | 'tanh' | 'leaky_relu' | 'gelu' | 'none';
  position: { x: number; y: number };
  params?: {
    kernelSize?: number;
    stride?: number;
    padding?: number;
    dropout?: number;
  };
}

interface NetworkSummary {
  totalLayers: number;
  totalParams: number;
  memoryUsage: number;
  flops: number;
}

const layerTypes = [
  { value: 'dense', label: 'Dense/Linear', icon: Brain },
  { value: 'conv2d', label: 'Conv2D', icon: Layers },
  { value: 'lstm', label: 'LSTM', icon: Zap },
  { value: 'attention', label: 'Attention', icon: Eye },
  { value: 'dropout', label: 'Dropout', icon: Settings },
  { value: 'batchnorm', label: 'BatchNorm', icon: Settings },
];

const activationFunctions = [
  { value: 'relu', label: 'ReLU' },
  { value: 'sigmoid', label: 'Sigmoid' },
  { value: 'tanh', label: 'Tanh' },
  { value: 'leaky_relu', label: 'Leaky ReLU' },
  { value: 'gelu', label: 'GELU' },
  { value: 'none', label: 'None' },
];

export const NeuralNetworkDesigner: React.FC = () => {
  const [layers, setLayers] = useState<NetworkLayer[]>([]);
  const [selectedLayer, setSelectedLayer] = useState<string | null>(null);
  const [isAnimating, setIsAnimating] = useState(false);
  const [showExport, setShowExport] = useState(false);
  const [showImport, setShowImport] = useState(false);
  
  // Layer configuration state
  const [layerConfig, setLayerConfig] = useState({
    type: 'dense' as NetworkLayer['type'],
    neurons: 64,
    activation: 'relu' as NetworkLayer['activation'],
    kernelSize: 3,
    stride: 1,
    padding: 1,
    dropout: 0.1,
  });

  const networkSummary: NetworkSummary = useMemo(() => {
    let totalParams = 0;
    let totalLayers = layers.length;
    
    layers.forEach((layer, index) => {
      const prevNeurons = index > 0 ? layers[index - 1].neurons : 784; // Default input size
      
      switch (layer.type) {
        case 'dense':
          totalParams += (prevNeurons + 1) * layer.neurons; // weights + bias
          break;
        case 'conv2d':
          const kernelSize = layer.params?.kernelSize || 3;
          totalParams += (kernelSize * kernelSize * prevNeurons + 1) * layer.neurons;
          break;
        case 'lstm':
          totalParams += 4 * (prevNeurons + layer.neurons + 1) * layer.neurons; // 4 gates
          break;
        case 'attention':
          totalParams += 3 * prevNeurons * layer.neurons; // Q, K, V matrices
          break;
        case 'dropout':
        case 'batchnorm':
          totalParams += layer.neurons * 2; // scale and shift parameters
          break;
      }
    });
    
    return {
      totalLayers,
      totalParams,
      memoryUsage: totalParams * 4, // 4 bytes per float32
      flops: totalParams * 2, // rough estimate
    };
  }, [layers]);

  const addLayer = () => {
    const newLayer: NetworkLayer = {
      id: `layer-${layers.length}`,
      type: layerConfig.type,
      neurons: layerConfig.neurons,
      activation: layerConfig.activation,
      position: { x: 100 + layers.length * 150, y: 200 },
      params: {
        kernelSize: layerConfig.kernelSize,
        stride: layerConfig.stride,
        padding: layerConfig.padding,
        dropout: layerConfig.dropout,
      },
    };
    
    setLayers([...layers, newLayer]);
  };

  const removeLayer = (layerId: string) => {
    setLayers(layers.filter(layer => layer.id !== layerId));
    if (selectedLayer === layerId) {
      setSelectedLayer(null);
    }
  };

  const clearAllLayers = () => {
    setLayers([]);
    setSelectedLayer(null);
  };

  const updateLayer = (layerId: string, updates: Partial<NetworkLayer>) => {
    setLayers(layers.map(layer => 
      layer.id === layerId ? { ...layer, ...updates } : layer
    ));
  };

  const animateForwardPass = () => {
    setIsAnimating(true);
    setTimeout(() => setIsAnimating(false), 3000);
  };

  const exportConfig = () => {
    const config = {
      layers: layers.map(({ position, ...layer }) => layer), // exclude position for cleaner export
      timestamp: new Date().toISOString(),
    };
    
    const blob = new Blob([JSON.stringify(config, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'neural-network-config.json';
    a.click();
    URL.revokeObjectURL(url);
  };

  const importConfig = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;
    
    const reader = new FileReader();
    reader.onload = (e) => {
      try {
        const config = JSON.parse(e.target?.result as string);
        const importedLayers = config.layers.map((layer: any, index: number) => ({
          ...layer,
          id: `layer-${index}`,
          position: { x: 100 + index * 150, y: 200 },
        }));
        setLayers(importedLayers);
        setShowImport(false);
      } catch (error) {
        alert('Invalid configuration file');
      }
    };
    reader.readAsText(file);
  };

  const renderLayer = (layer: NetworkLayer, index: number) => {
    const isSelected = selectedLayer === layer.id;
    const layerType = layerTypes.find(t => t.type === layer.type);
    const LayerIcon = layerType?.icon || Brain;

    return (
      <motion.div
        key={layer.id}
        data-testid={layer.id}
        className={`absolute cursor-pointer transform -translate-x-1/2 -translate-y-1/2 ${
          isSelected ? 'z-20' : 'z-10'
        }`}
        style={{
          left: layer.position.x,
          top: layer.position.y,
        }}
        whileHover={{ scale: 1.05 }}
        whileTap={{ scale: 0.95 }}
        onClick={() => setSelectedLayer(layer.id)}
        drag
        dragMomentum={false}
        onDragEnd={(_, info) => {
          updateLayer(layer.id, {
            position: {
              x: layer.position.x + info.offset.x,
              y: layer.position.y + info.offset.y,
            },
          });
        }}
      >
        <div className={`bg-blue-500 rounded-lg p-4 shadow-lg border-2 ${
          isSelected ? 'border-yellow-400' : 'border-transparent'
        } min-w-[120px] text-center`}>
          <LayerIcon className="w-6 h-6 text-white mx-auto mb-2" />
          <h3 className="text-white font-bold text-sm">{layerType?.label}</h3>
          <div className="text-white text-xs mt-1">
            <div>{layer.neurons} neurons</div>
            <div>{layer.activation}</div>
          </div>
          
          <button
            data-testid={`delete-layer-${index}`}
            onClick={(e) => {
              e.stopPropagation();
              removeLayer(layer.id);
            }}
            className="absolute -top-2 -right-2 w-6 h-6 bg-red-500 rounded-full flex items-center justify-center hover:bg-red-600"
          >
            <X size={12} className="text-white" />
          </button>
        </div>

        {/* Connection to next layer */}
        {index < layers.length - 1 && (
          <div className="absolute left-full top-1/2 transform -translate-y-1/2">
            <svg width="150" height="4">
              <line
                x1="0"
                y1="2"
                x2="150"
                y2="2"
                stroke="#64748B"
                strokeWidth="2"
                markerEnd="url(#arrowhead)"
              />
              {isAnimating && (
                <motion.circle
                  cx="0"
                  cy="2"
                  r="3"
                  fill="#EF4444"
                  animate={{ cx: [0, 150] }}
                  transition={{ duration: 0.5, delay: index * 0.5 }}
                />
              )}
            </svg>
          </div>
        )}
      </motion.div>
    );
  };

  return (
    <div className="bg-white rounded-lg border shadow-lg p-6">
      {/* Header */}
      <div className="flex justify-between items-center mb-6">
        <div>
          <h2 className="text-2xl font-bold">Neural Network Designer</h2>
          <p className="text-gray-600">Design and visualize neural network architectures</p>
        </div>
        
        <div className="flex items-center gap-2">
          <button
            onClick={animateForwardPass}
            disabled={layers.length === 0}
            className="px-4 py-2 bg-green-500 text-white rounded-lg hover:bg-green-600 disabled:opacity-50 flex items-center gap-2"
          >
            <Play size={16} />
            Animate Forward Pass
          </button>
          
          <button
            onClick={() => setShowExport(true)}
            disabled={layers.length === 0}
            className="px-4 py-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600 disabled:opacity-50 flex items-center gap-2"
          >
            <Download size={16} />
            Export Config
          </button>
          
          <button
            onClick={() => setShowImport(true)}
            className="px-4 py-2 bg-gray-500 text-white rounded-lg hover:bg-gray-600 flex items-center gap-2"
          >
            <Upload size={16} />
            Import Config
          </button>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
        {/* Layer Configuration Panel */}
        <div className="lg:col-span-1">
          <div className="bg-gray-50 rounded-lg p-4">
            <h3 className="font-semibold mb-4">Layer Configuration</h3>
            
            <div className="space-y-4">
              <div>
                <label htmlFor="layer-type" className="block text-sm font-medium mb-1">Layer Type</label>
                <select
                  id="layer-type"
                  value={layerConfig.type}
                  onChange={(e) => setLayerConfig({
                    ...layerConfig,
                    type: e.target.value as NetworkLayer['type']
                  })}
                  className="w-full p-2 border rounded"
                >
                  {layerTypes.map(type => (
                    <option key={type.value} value={type.value}>{type.label}</option>
                  ))}
                </select>
              </div>
              
              <div>
                <label htmlFor="neurons" className="block text-sm font-medium mb-1">Neurons</label>
                <input
                  id="neurons"
                  type="number"
                  value={layerConfig.neurons}
                  onChange={(e) => setLayerConfig({
                    ...layerConfig,
                    neurons: parseInt(e.target.value) || 1
                  })}
                  className="w-full p-2 border rounded"
                  min="1"
                  max="2048"
                />
              </div>
              
              <div>
                <label htmlFor="activation" className="block text-sm font-medium mb-1">Activation</label>
                <select
                  id="activation"
                  value={layerConfig.activation}
                  onChange={(e) => setLayerConfig({
                    ...layerConfig,
                    activation: e.target.value as NetworkLayer['activation']
                  })}
                  className="w-full p-2 border rounded"
                >
                  {activationFunctions.map(fn => (
                    <option key={fn.value} value={fn.value}>{fn.label}</option>
                  ))}
                </select>
              </div>
              
              {layerConfig.type === 'conv2d' && (
                <>
                  <div>
                    <label className="block text-sm font-medium mb-1">Kernel Size</label>
                    <input
                      type="number"
                      value={layerConfig.kernelSize}
                      onChange={(e) => setLayerConfig({
                        ...layerConfig,
                        kernelSize: parseInt(e.target.value) || 1
                      })}
                      className="w-full p-2 border rounded"
                      min="1"
                      max="11"
                    />
                  </div>
                  
                  <div>
                    <label className="block text-sm font-medium mb-1">Stride</label>
                    <input
                      type="number"
                      value={layerConfig.stride}
                      onChange={(e) => setLayerConfig({
                        ...layerConfig,
                        stride: parseInt(e.target.value) || 1
                      })}
                      className="w-full p-2 border rounded"
                      min="1"
                      max="5"
                    />
                  </div>
                </>
              )}
              
              {layerConfig.type === 'dropout' && (
                <div>
                  <label className="block text-sm font-medium mb-1">Dropout Rate</label>
                  <input
                    type="number"
                    value={layerConfig.dropout}
                    onChange={(e) => setLayerConfig({
                      ...layerConfig,
                      dropout: parseFloat(e.target.value) || 0
                    })}
                    className="w-full p-2 border rounded"
                    min="0"
                    max="1"
                    step="0.1"
                  />
                </div>
              )}
            </div>
            
            <div className="mt-6 space-y-2">
              <button
                onClick={addLayer}
                className="w-full px-4 py-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600 flex items-center justify-center gap-2"
              >
                <Plus size={16} />
                Add Layer
              </button>
              
              <button
                onClick={clearAllLayers}
                disabled={layers.length === 0}
                className="w-full px-4 py-2 bg-red-500 text-white rounded-lg hover:bg-red-600 disabled:opacity-50 flex items-center justify-center gap-2"
              >
                <Trash2 size={16} />
                Clear All
              </button>
            </div>
          </div>
          
          {/* Network Summary */}
          <div className="bg-blue-50 rounded-lg p-4 mt-4">
            <h3 className="font-semibold mb-3">Network Summary</h3>
            <div className="text-sm space-y-1">
              <div>Total Layers: {networkSummary.totalLayers}</div>
              <div>Parameters: {networkSummary.totalParams.toLocaleString()}</div>
              <div>Memory: {(networkSummary.memoryUsage / 1024 / 1024).toFixed(1)} MB</div>
              <div>FLOPs: {(networkSummary.flops / 1e6).toFixed(1)} M</div>
            </div>
          </div>
        </div>

        {/* Network Visualization */}
        <div className="lg:col-span-3">
          <div className="relative bg-gray-50 rounded-lg p-8 min-h-[500px] overflow-auto">
            {isAnimating && (
              <div 
                data-testid="forward-pass-animation"
                className="absolute top-4 left-4 bg-green-500 text-white px-3 py-1 rounded-lg z-30"
              >
                Forward Pass Animation
              </div>
            )}
            
            <svg 
              data-testid="network-svg"
              className="absolute inset-0 w-full h-full pointer-events-none" 
              style={{ zIndex: 5 }}
            >
              <defs>
                <marker
                  id="arrowhead"
                  markerWidth="10"
                  markerHeight="7"
                  refX="9"
                  refY="3.5"
                  orient="auto"
                >
                  <polygon points="0 0, 10 3.5, 0 7" fill="#64748B" />
                </marker>
              </defs>
            </svg>

            {layers.length === 0 ? (
              <div className="flex items-center justify-center h-full text-gray-500">
                <div className="text-center">
                  <Brain size={48} className="mx-auto mb-4 opacity-50" />
                  <p>Add layers to design your neural network</p>
                </div>
              </div>
            ) : (
              layers.map((layer, index) => renderLayer(layer, index))
            )}
          </div>
        </div>
      </div>

      {/* Layer Editor */}
      {selectedLayer && (
        <div 
          data-testid="layer-editor"
          className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50"
          onClick={() => setSelectedLayer(null)}
        >
          <div 
            className="bg-white rounded-lg p-6 max-w-md w-full mx-4"
            onClick={(e) => e.stopPropagation()}
          >
            <h3 className="text-lg font-bold mb-4">Edit Layer</h3>
            {/* Layer editing form would go here */}
            <div className="flex justify-end gap-2">
              <button
                onClick={() => setSelectedLayer(null)}
                className="px-4 py-2 text-gray-600 hover:text-gray-800"
              >
                Cancel
              </button>
              <button
                onClick={() => setSelectedLayer(null)}
                className="px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600"
              >
                Save
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Export Modal */}
      {showExport && (
        <div 
          data-testid="export-modal"
          className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50"
          onClick={() => setShowExport(false)}
        >
          <div 
            className="bg-white rounded-lg p-6 max-w-md w-full mx-4"
            onClick={(e) => e.stopPropagation()}
          >
            <h3 className="text-lg font-bold mb-4">Export Configuration</h3>
            <p className="text-gray-600 mb-4">
              Download your network configuration as a JSON file.
            </p>
            <div className="flex justify-end gap-2">
              <button
                onClick={() => setShowExport(false)}
                className="px-4 py-2 text-gray-600 hover:text-gray-800"
              >
                Cancel
              </button>
              <button
                onClick={() => {
                  exportConfig();
                  setShowExport(false);
                }}
                className="px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600 flex items-center gap-2"
              >
                <Download size={16} />
                Download
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Import Modal */}
      {showImport && (
        <div 
          data-testid="import-modal"
          className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50"
          onClick={() => setShowImport(false)}
        >
          <div 
            className="bg-white rounded-lg p-6 max-w-md w-full mx-4"
            onClick={(e) => e.stopPropagation()}
          >
            <h3 className="text-lg font-bold mb-4">Import Configuration</h3>
            <p className="text-gray-600 mb-4">
              Upload a JSON configuration file to load a network.
            </p>
            <input
              type="file"
              accept=".json"
              onChange={importConfig}
              className="w-full p-2 border rounded mb-4"
            />
            <div className="flex justify-end gap-2">
              <button
                onClick={() => setShowImport(false)}
                className="px-4 py-2 text-gray-600 hover:text-gray-800"
              >
                Cancel
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};