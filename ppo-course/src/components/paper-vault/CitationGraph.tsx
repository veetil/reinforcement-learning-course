'use client';

import React, { useEffect, useRef, useState } from 'react';
import { Paper } from '@/lib/paper-vault/arxiv-crawler';
import { ZoomIn, ZoomOut, Maximize2, Download } from 'lucide-react';

interface CitationNode {
  id: string;
  title: string;
  authors: string[];
  year: number;
  citations: number;
  category: string;
  x?: number;
  y?: number;
  vx?: number;
  vy?: number;
}

interface CitationLink {
  source: string;
  target: string;
  type: 'cites' | 'cited-by' | 'related';
}

interface CitationGraphProps {
  papers: Paper[];
  focusPaperId?: string;
}

export function CitationGraph({ papers, focusPaperId }: CitationGraphProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [zoom, setZoom] = useState(1);
  const [offset, setOffset] = useState({ x: 0, y: 0 });
  const [selectedNode, setSelectedNode] = useState<CitationNode | null>(null);
  const [hoveredNode, setHoveredNode] = useState<CitationNode | null>(null);
  const [isDragging, setIsDragging] = useState(false);
  const [dragStart, setDragStart] = useState({ x: 0, y: 0 });

  // Generate mock citation data for demonstration
  const generateGraphData = (): { nodes: CitationNode[], links: CitationLink[] } => {
    const nodes: CitationNode[] = [
      {
        id: 'ppo-2017',
        title: 'Proximal Policy Optimization',
        authors: ['Schulman et al.'],
        year: 2017,
        citations: 5000,
        category: 'policy-gradient',
      },
      {
        id: 'trpo-2015',
        title: 'Trust Region Policy Optimization',
        authors: ['Schulman et al.'],
        year: 2015,
        citations: 3000,
        category: 'policy-gradient',
      },
      {
        id: 'gae-2016',
        title: 'Generalized Advantage Estimation',
        authors: ['Schulman et al.'],
        year: 2016,
        citations: 2500,
        category: 'policy-gradient',
      },
      {
        id: 'a3c-2016',
        title: 'Asynchronous Actor-Critic',
        authors: ['Mnih et al.'],
        year: 2016,
        citations: 4000,
        category: 'actor-critic',
      },
      {
        id: 'sac-2018',
        title: 'Soft Actor-Critic',
        authors: ['Haarnoja et al.'],
        year: 2018,
        citations: 3500,
        category: 'actor-critic',
      },
      {
        id: 'rlhf-2023',
        title: 'RLHF for Language Models',
        authors: ['OpenAI'],
        year: 2023,
        citations: 1000,
        category: 'rlhf',
      },
      {
        id: 'grpo-2024',
        title: 'Group Relative Policy Optimization',
        authors: ['Our Work'],
        year: 2024,
        citations: 50,
        category: 'policy-gradient',
      },
    ];

    const links: CitationLink[] = [
      { source: 'ppo-2017', target: 'trpo-2015', type: 'cites' },
      { source: 'ppo-2017', target: 'gae-2016', type: 'cites' },
      { source: 'grpo-2024', target: 'ppo-2017', type: 'cites' },
      { source: 'rlhf-2023', target: 'ppo-2017', type: 'cites' },
      { source: 'sac-2018', target: 'a3c-2016', type: 'related' },
      { source: 'grpo-2024', target: 'rlhf-2023', type: 'related' },
    ];

    return { nodes, links };
  };

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Set canvas size
    const resizeCanvas = () => {
      canvas.width = canvas.offsetWidth * window.devicePixelRatio;
      canvas.height = canvas.offsetHeight * window.devicePixelRatio;
      ctx.scale(window.devicePixelRatio, window.devicePixelRatio);
    };
    resizeCanvas();
    window.addEventListener('resize', resizeCanvas);

    // Initialize graph data
    const { nodes, links } = generateGraphData();

    // Initialize node positions
    nodes.forEach((node, i) => {
      const angle = (i / nodes.length) * 2 * Math.PI;
      const radius = 200;
      node.x = canvas.offsetWidth / 2 + radius * Math.cos(angle);
      node.y = canvas.offsetHeight / 2 + radius * Math.sin(angle);
      node.vx = 0;
      node.vy = 0;
    });

    // Force-directed layout simulation
    const simulate = () => {
      // Apply forces
      nodes.forEach((node, i) => {
        // Reset forces
        node.vx = (node.vx || 0) * 0.9; // Damping
        node.vy = (node.vy || 0) * 0.9;

        // Repulsion between nodes
        nodes.forEach((other, j) => {
          if (i === j) return;
          const dx = node.x! - other.x!;
          const dy = node.y! - other.y!;
          const distance = Math.sqrt(dx * dx + dy * dy);
          if (distance < 150) {
            const force = (150 - distance) / distance * 0.5;
            node.vx! += dx * force;
            node.vy! += dy * force;
          }
        });

        // Attraction along links
        links.forEach(link => {
          let other: CitationNode | undefined;
          if (link.source === node.id) {
            other = nodes.find(n => n.id === link.target);
          } else if (link.target === node.id) {
            other = nodes.find(n => n.id === link.source);
          }
          
          if (other) {
            const dx = other.x! - node.x!;
            const dy = other.y! - node.y!;
            const distance = Math.sqrt(dx * dx + dy * dy);
            const force = (distance - 100) / distance * 0.1;
            node.vx! += dx * force;
            node.vy! += dy * force;
          }
        });

        // Center gravity
        const centerX = canvas.offsetWidth / 2;
        const centerY = canvas.offsetHeight / 2;
        node.vx! += (centerX - node.x!) * 0.01;
        node.vy! += (centerY - node.y!) * 0.01;
      });

      // Update positions
      nodes.forEach(node => {
        node.x! += node.vx!;
        node.y! += node.vy!;
        
        // Boundary constraints
        node.x = Math.max(50, Math.min(canvas.offsetWidth - 50, node.x!));
        node.y = Math.max(50, Math.min(canvas.offsetHeight - 50, node.y!));
      });
    };

    // Rendering function
    const render = () => {
      ctx.clearRect(0, 0, canvas.offsetWidth, canvas.offsetHeight);
      
      // Apply zoom and pan
      ctx.save();
      ctx.translate(offset.x, offset.y);
      ctx.scale(zoom, zoom);

      // Draw links
      ctx.strokeStyle = '#ddd';
      ctx.lineWidth = 1;
      links.forEach(link => {
        const source = nodes.find(n => n.id === link.source);
        const target = nodes.find(n => n.id === link.target);
        if (!source || !target) return;

        ctx.beginPath();
        ctx.moveTo(source.x!, source.y!);
        
        if (link.type === 'related') {
          ctx.setLineDash([5, 5]);
          ctx.strokeStyle = '#bbb';
        } else {
          ctx.setLineDash([]);
          ctx.strokeStyle = '#ddd';
        }
        
        ctx.lineTo(target.x!, target.y!);
        ctx.stroke();

        // Draw arrow
        if (link.type === 'cites') {
          const angle = Math.atan2(target.y! - source.y!, target.x! - source.x!);
          const arrowLength = 10;
          const arrowAngle = 0.5;
          
          ctx.beginPath();
          ctx.moveTo(target.x!, target.y!);
          ctx.lineTo(
            target.x! - arrowLength * Math.cos(angle - arrowAngle),
            target.y! - arrowLength * Math.sin(angle - arrowAngle)
          );
          ctx.moveTo(target.x!, target.y!);
          ctx.lineTo(
            target.x! - arrowLength * Math.cos(angle + arrowAngle),
            target.y! - arrowLength * Math.sin(angle + arrowAngle)
          );
          ctx.stroke();
        }
      });

      // Draw nodes
      nodes.forEach(node => {
        const radius = Math.sqrt(node.citations) / 10 + 10;
        
        // Node circle
        ctx.beginPath();
        ctx.arc(node.x!, node.y!, radius, 0, 2 * Math.PI);
        
        // Color by category
        const colors: Record<string, string> = {
          'policy-gradient': '#8b5cf6',
          'actor-critic': '#06b6d4',
          'rlhf': '#f59e0b',
          'default': '#6b7280',
        };
        
        ctx.fillStyle = colors[node.category] || colors.default;
        
        if (node === hoveredNode || node === selectedNode) {
          ctx.strokeStyle = '#000';
          ctx.lineWidth = 3;
          ctx.stroke();
        }
        
        ctx.fill();

        // Node label
        ctx.fillStyle = '#000';
        ctx.font = '12px sans-serif';
        ctx.textAlign = 'center';
        ctx.fillText(node.title, node.x!, node.y! + radius + 15);
        
        // Year
        ctx.fillStyle = '#666';
        ctx.font = '10px sans-serif';
        ctx.fillText(node.year.toString(), node.x!, node.y! + radius + 28);
      });

      ctx.restore();
    };

    // Animation loop
    let animationId: number;
    const animate = () => {
      simulate();
      render();
      animationId = requestAnimationFrame(animate);
    };
    animate();

    // Mouse event handlers
    const getNodeAtPosition = (x: number, y: number): CitationNode | null => {
      const rect = canvas.getBoundingClientRect();
      const canvasX = (x - rect.left - offset.x) / zoom;
      const canvasY = (y - rect.top - offset.y) / zoom;
      
      for (const node of nodes) {
        const radius = Math.sqrt(node.citations) / 10 + 10;
        const dx = canvasX - node.x!;
        const dy = canvasY - node.y!;
        if (dx * dx + dy * dy < radius * radius) {
          return node;
        }
      }
      return null;
    };

    const handleMouseMove = (e: MouseEvent) => {
      const node = getNodeAtPosition(e.clientX, e.clientY);
      setHoveredNode(node);
      
      if (isDragging) {
        const dx = e.clientX - dragStart.x;
        const dy = e.clientY - dragStart.y;
        setOffset({ x: offset.x + dx, y: offset.y + dy });
        setDragStart({ x: e.clientX, y: e.clientY });
      }
    };

    const handleMouseDown = (e: MouseEvent) => {
      const node = getNodeAtPosition(e.clientX, e.clientY);
      if (node) {
        setSelectedNode(node);
      } else {
        setIsDragging(true);
        setDragStart({ x: e.clientX, y: e.clientY });
      }
    };

    const handleMouseUp = () => {
      setIsDragging(false);
    };

    const handleWheel = (e: WheelEvent) => {
      e.preventDefault();
      const delta = e.deltaY > 0 ? 0.9 : 1.1;
      setZoom(prev => Math.max(0.1, Math.min(5, prev * delta)));
    };

    canvas.addEventListener('mousemove', handleMouseMove);
    canvas.addEventListener('mousedown', handleMouseDown);
    canvas.addEventListener('mouseup', handleMouseUp);
    canvas.addEventListener('wheel', handleWheel);

    return () => {
      cancelAnimationFrame(animationId);
      window.removeEventListener('resize', resizeCanvas);
      canvas.removeEventListener('mousemove', handleMouseMove);
      canvas.removeEventListener('mousedown', handleMouseDown);
      canvas.removeEventListener('mouseup', handleMouseUp);
      canvas.removeEventListener('wheel', handleWheel);
    };
  }, [zoom, offset, isDragging, dragStart]);

  const handleZoomIn = () => setZoom(prev => Math.min(5, prev * 1.2));
  const handleZoomOut = () => setZoom(prev => Math.max(0.1, prev / 1.2));
  const handleReset = () => {
    setZoom(1);
    setOffset({ x: 0, y: 0 });
  };

  return (
    <div className="relative w-full h-full bg-gray-50 rounded-lg overflow-hidden">
      <canvas
        ref={canvasRef}
        className="w-full h-full cursor-move"
        style={{ cursor: isDragging ? 'grabbing' : 'grab' }}
      />

      {/* Controls */}
      <div className="absolute top-4 right-4 flex flex-col gap-2">
        <button
          onClick={handleZoomIn}
          className="p-2 bg-white rounded-lg shadow hover:bg-gray-50"
          title="Zoom In"
        >
          <ZoomIn className="w-4 h-4" />
        </button>
        <button
          onClick={handleZoomOut}
          className="p-2 bg-white rounded-lg shadow hover:bg-gray-50"
          title="Zoom Out"
        >
          <ZoomOut className="w-4 h-4" />
        </button>
        <button
          onClick={handleReset}
          className="p-2 bg-white rounded-lg shadow hover:bg-gray-50"
          title="Reset View"
        >
          <Maximize2 className="w-4 h-4" />
        </button>
      </div>

      {/* Legend */}
      <div className="absolute bottom-4 left-4 bg-white p-3 rounded-lg shadow">
        <h4 className="text-sm font-semibold mb-2">Categories</h4>
        <div className="space-y-1 text-xs">
          <div className="flex items-center gap-2">
            <div className="w-3 h-3 rounded-full bg-purple-500"></div>
            <span>Policy Gradient</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-3 h-3 rounded-full bg-cyan-500"></div>
            <span>Actor-Critic</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-3 h-3 rounded-full bg-amber-500"></div>
            <span>RLHF</span>
          </div>
        </div>
      </div>

      {/* Selected Node Info */}
      {selectedNode && (
        <div className="absolute top-4 left-4 bg-white p-4 rounded-lg shadow max-w-xs">
          <h3 className="font-semibold">{selectedNode.title}</h3>
          <p className="text-sm text-gray-600">{selectedNode.authors.join(', ')}</p>
          <p className="text-sm text-gray-600">{selectedNode.year}</p>
          <p className="text-sm text-gray-600">{selectedNode.citations} citations</p>
          <button
            onClick={() => setSelectedNode(null)}
            className="text-xs text-blue-600 hover:text-blue-800 mt-2"
          >
            Close
          </button>
        </div>
      )}
    </div>
  );
}