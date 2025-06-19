'use client'

import React, { useCallback, useState, useEffect } from 'react'
import {
  ReactFlow,
  Node,
  Edge,
  Background,
  Controls,
  MiniMap,
  useNodesState,
  useEdgesState,
  addEdge,
  Connection,
  NodeTypes,
  Position
} from '@xyflow/react'
import '@xyflow/react/dist/style.css'
import { motion } from 'framer-motion'

interface NeuronData {
  label: string
  value: number
  activation?: number
  type: 'input' | 'hidden' | 'output'
}

interface NetworkLayer {
  neurons: number
  type: 'input' | 'hidden' | 'output'
  label?: string
}

interface NeuralNetworkVisualizerProps {
  layers: NetworkLayer[]
  weights?: number[][]
  activations?: number[][]
  animated?: boolean
  onNeuronClick?: (neuronId: string) => void
  showValues?: boolean
  showWeights?: boolean
}

// Custom neuron component
const NeuronNode = ({ data, selected }: { data: NeuronData; selected: boolean }) => {
  const getColor = () => {
    if (data.type === 'input') return '#3b82f6' // blue
    if (data.type === 'output') return '#10b981' // green
    return '#8b5cf6' // purple for hidden
  }

  return (
    <motion.div
      initial={{ scale: 0 }}
      animate={{ 
        scale: selected ? 1.2 : 1,
        backgroundColor: getColor()
      }}
      transition={{ duration: 0.3 }}
      className="relative w-16 h-16 rounded-full flex items-center justify-center text-white font-semibold shadow-lg cursor-pointer"
      whileHover={{ scale: 1.1 }}
      whileTap={{ scale: 0.95 }}
    >
      <div className="text-xs text-center">
        {data.label}
        {data.value !== undefined && (
          <div className="text-[10px] mt-0.5">
            {data.value.toFixed(3)}
          </div>
        )}
      </div>
      {data.activation !== undefined && data.activation > 0.5 && (
        <motion.div
          className="absolute inset-0 rounded-full border-2 border-yellow-400"
          initial={{ scale: 1, opacity: 1 }}
          animate={{ scale: 1.5, opacity: 0 }}
          transition={{ duration: 1, repeat: Infinity }}
        />
      )}
    </motion.div>
  )
}

const nodeTypes: NodeTypes = {
  neuron: NeuronNode as any
}

export default function NeuralNetworkVisualizer({
  layers,
  weights,
  activations,
  animated = true,
  onNeuronClick,
  showValues = true,
  showWeights = false
}: NeuralNetworkVisualizerProps) {
  const [nodes, setNodes, onNodesChange] = useNodesState([])
  const [edges, setEdges, onEdgesChange] = useEdgesState([])
  const [selectedNode, setSelectedNode] = useState<string | null>(null)

  // Generate nodes and edges from layers
  useEffect(() => {
    const newNodes: Node[] = []
    const newEdges: Edge[] = []
    let nodeId = 0
    const layerPositions: number[][] = []

    // Calculate positions for each layer
    const totalWidth = 800
    const totalHeight = 500
    const layerSpacing = totalWidth / (layers.length + 1)

    layers.forEach((layer, layerIndex) => {
      const layerNodes: number[] = []
      const verticalSpacing = totalHeight / (layer.neurons + 1)
      
      for (let i = 0; i < layer.neurons; i++) {
        const x = layerSpacing * (layerIndex + 1)
        const y = verticalSpacing * (i + 1)
        
        newNodes.push({
          id: `${nodeId}`,
          type: 'neuron',
          position: { x, y },
          data: {
            label: `${layer.type[0].toUpperCase()}${layerIndex}-${i}`,
            value: activations?.[layerIndex]?.[i] || 0,
            activation: activations?.[layerIndex]?.[i],
            type: layer.type
          }
        })
        
        layerNodes.push(nodeId)
        nodeId++
      }
      
      layerPositions.push(layerNodes)
    })

    // Create edges between layers
    for (let l = 0; l < layerPositions.length - 1; l++) {
      const currentLayer = layerPositions[l]
      const nextLayer = layerPositions[l + 1]
      
      currentLayer.forEach((fromNode, i) => {
        nextLayer.forEach((toNode, j) => {
          const weight = weights?.[l]?.[i * nextLayer.length + j] || 0.5
          const opacity = Math.abs(weight)
          const strokeWidth = Math.max(1, Math.abs(weight) * 3)
          
          newEdges.push({
            id: `e${fromNode}-${toNode}`,
            source: `${fromNode}`,
            target: `${toNode}`,
            type: animated ? 'smoothstep' : 'straight',
            animated: animated && Math.abs(weight) > 0.7,
            style: {
              stroke: weight > 0 ? '#3b82f6' : '#ef4444',
              strokeWidth,
              opacity: showWeights ? opacity : 0.3
            },
            label: showWeights ? weight.toFixed(2) : undefined,
            labelStyle: { fontSize: 10 }
          })
        })
      })
    }

    setNodes(newNodes)
    setEdges(newEdges)
  }, [layers, weights, activations, animated, showWeights, setNodes, setEdges])

  const onConnect = useCallback(
    (params: Connection) => setEdges((eds) => addEdge(params, eds)),
    [setEdges]
  )

  const handleNodeClick = useCallback((event: React.MouseEvent, node: Node) => {
    setSelectedNode(node.id)
    onNeuronClick?.(node.id)
  }, [onNeuronClick])

  return (
    <div className="w-full h-full bg-gray-50 rounded-lg shadow-inner">
      <ReactFlow
        nodes={nodes}
        edges={edges}
        onNodesChange={onNodesChange}
        onEdgesChange={onEdgesChange}
        onConnect={onConnect}
        onNodeClick={handleNodeClick}
        nodeTypes={nodeTypes}
        fitView
        attributionPosition="bottom-left"
      >
        <Background variant="dots" gap={12} size={1} />
        <Controls />
        <MiniMap 
          nodeColor={(node) => {
            if (node.data.type === 'input') return '#3b82f6'
            if (node.data.type === 'output') return '#10b981'
            return '#8b5cf6'
          }}
        />
      </ReactFlow>
    </div>
  )
}