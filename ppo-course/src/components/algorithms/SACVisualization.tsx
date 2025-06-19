'use client';

import React, { useState, useEffect, useRef } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Slider } from '@/components/ui/slider';
import { Play, Pause, RotateCcw, Info } from 'lucide-react';
import { LineChart, Line, AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend } from 'recharts';

interface SACVisualizationProps {
  className?: string;
}

export function SACVisualization({ className = '' }: SACVisualizationProps) {
  const [isRunning, setIsRunning] = useState(false);
  const [temperature, setTemperature] = useState(0.2);
  const [exploration, setExploration] = useState(50);
  const [step, setStep] = useState(0);
  const [selectedComponent, setSelectedComponent] = useState<'actor' | 'critic' | 'temperature'>('actor');
  
  const [performanceData, setPerformanceData] = useState<any[]>([]);
  const [qValueData, setQValueData] = useState<any[]>([]);
  const [entropyData, setEntropyData] = useState<any[]>([]);
  
  const animationRef = useRef<number>();

  // Simulate SAC training
  useEffect(() => {
    if (isRunning) {
      const animate = () => {
        setStep(prev => {
          const newStep = prev + 1;
          
          // Generate synthetic data
          const reward = Math.sin(newStep * 0.02) * 50 + 50 + Math.random() * 10;
          const q1 = reward + Math.random() * 20 - 10;
          const q2 = reward + Math.random() * 20 - 10;
          const entropy = temperature * 10 * (1 + Math.sin(newStep * 0.05));
          
          setPerformanceData(prev => [...prev.slice(-99), {
            step: newStep,
            reward,
            value: Math.min(q1, q2) - temperature * entropy
          }]);
          
          setQValueData(prev => [...prev.slice(-99), {
            step: newStep,
            q1,
            q2,
            minQ: Math.min(q1, q2)
          }]);
          
          setEntropyData(prev => [...prev.slice(-99), {
            step: newStep,
            entropy,
            temperature: temperature,
            objective: entropy * temperature
          }]);
          
          // Auto-adjust temperature
          if (newStep % 50 === 0) {
            setTemperature(prev => Math.max(0.01, prev * 0.95));
          }
          
          return newStep;
        });
        
        animationRef.current = requestAnimationFrame(animate);
      };
      
      animationRef.current = requestAnimationFrame(animate);
    } else {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }
    }
    
    return () => {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }
    };
  }, [isRunning, temperature]);

  const reset = () => {
    setIsRunning(false);
    setStep(0);
    setTemperature(0.2);
    setPerformanceData([]);
    setQValueData([]);
    setEntropyData([]);
  };

  // Visualization of policy distribution
  const PolicyDistribution = () => {
    const canvasRef = useRef<HTMLCanvasElement>(null);
    
    useEffect(() => {
      const canvas = canvasRef.current;
      if (!canvas) return;
      
      const ctx = canvas.getContext('2d');
      if (!ctx) return;
      
      const width = canvas.width;
      const height = canvas.height;
      
      // Clear canvas
      ctx.clearRect(0, 0, width, height);
      
      // Draw axes
      ctx.strokeStyle = '#e5e7eb';
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(40, height - 40);
      ctx.lineTo(width - 20, height - 40);
      ctx.moveTo(40, 20);
      ctx.lineTo(40, height - 40);
      ctx.stroke();
      
      // Draw Gaussian distributions for different temperatures
      const drawGaussian = (mean: number, std: number, color: string, alpha: number = 1) => {
        ctx.strokeStyle = color;
        ctx.fillStyle = color;
        ctx.globalAlpha = alpha;
        ctx.lineWidth = 2;
        
        ctx.beginPath();
        for (let x = 0; x < width - 60; x++) {
          const xVal = (x / (width - 60)) * 4 - 2; // -2 to 2
          const y = Math.exp(-Math.pow(xVal - mean, 2) / (2 * Math.pow(std, 2))) / (std * Math.sqrt(2 * Math.PI));
          const yPos = height - 40 - y * (height - 60) * 2;
          
          if (x === 0) {
            ctx.moveTo(x + 40, yPos);
          } else {
            ctx.lineTo(x + 40, yPos);
          }
        }
        ctx.stroke();
        
        // Fill area under curve
        ctx.globalAlpha = alpha * 0.2;
        ctx.lineTo(width - 20, height - 40);
        ctx.lineTo(40, height - 40);
        ctx.closePath();
        ctx.fill();
        ctx.globalAlpha = 1;
      };
      
      // Draw multiple distributions showing exploration
      const explorationLevel = exploration / 100;
      drawGaussian(0, 0.1 + temperature * explorationLevel, '#3b82f6', 0.6); // Current policy
      drawGaussian(0.3, 0.05, '#10b981', 0.4); // Optimal policy
      
      // Labels
      ctx.fillStyle = '#374151';
      ctx.font = '12px sans-serif';
      ctx.fillText('Action', width / 2 - 20, height - 10);
      ctx.save();
      ctx.translate(15, height / 2);
      ctx.rotate(-Math.PI / 2);
      ctx.fillText('Probability', -30, 0);
      ctx.restore();
      
      // Legend
      ctx.fillStyle = '#3b82f6';
      ctx.fillRect(width - 100, 30, 10, 10);
      ctx.fillStyle = '#374151';
      ctx.fillText('Current', width - 85, 39);
      
      ctx.fillStyle = '#10b981';
      ctx.fillRect(width - 100, 50, 10, 10);
      ctx.fillStyle = '#374151';
      ctx.fillText('Optimal', width - 85, 59);
    }, [temperature, exploration, step]);
    
    return <canvas ref={canvasRef} width={400} height={250} className="w-full" />;
  };

  // Q-function visualization
  const QFunctionVisual = () => {
    const canvasRef = useRef<HTMLCanvasElement>(null);
    
    useEffect(() => {
      const canvas = canvasRef.current;
      if (!canvas) return;
      
      const ctx = canvas.getContext('2d');
      if (!ctx) return;
      
      const width = canvas.width;
      const height = canvas.height;
      
      // Create gradient for Q-value heatmap
      const createHeatmap = () => {
        ctx.clearRect(0, 0, width, height);
        
        // Create Q-value surface
        for (let x = 0; x < width; x += 4) {
          for (let y = 0; y < height; y += 4) {
            const stateVal = (x / width) * 2 - 1;
            const actionVal = (y / height) * 2 - 1;
            
            // Simulate Q-value
            const q1 = Math.sin(stateVal * 3) * Math.cos(actionVal * 3) + 
                      Math.random() * 0.2 + step * 0.001;
            const q2 = Math.cos(stateVal * 3) * Math.sin(actionVal * 3) + 
                      Math.random() * 0.2 + step * 0.001;
            const qValue = Math.min(q1, q2);
            
            // Map to color
            const intensity = (qValue + 1) / 2; // Normalize to 0-1
            const r = Math.floor(255 * (1 - intensity));
            const g = Math.floor(255 * intensity);
            const b = 100;
            
            ctx.fillStyle = `rgb(${r}, ${g}, ${b})`;
            ctx.fillRect(x, y, 4, 4);
          }
        }
        
        // Draw axes
        ctx.strokeStyle = '#ffffff';
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.moveTo(0, height / 2);
        ctx.lineTo(width, height / 2);
        ctx.moveTo(width / 2, 0);
        ctx.lineTo(width / 2, height);
        ctx.stroke();
        
        // Labels
        ctx.fillStyle = '#ffffff';
        ctx.font = '12px sans-serif';
        ctx.fillText('State', width - 35, height / 2 - 5);
        ctx.fillText('Action', width / 2 + 5, 15);
      };
      
      createHeatmap();
    }, [step]);
    
    return <canvas ref={canvasRef} width={400} height={250} className="w-full" />;
  };

  return (
    <div className={`space-y-6 ${className}`}>
      {/* Control Panel */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center justify-between">
            SAC Interactive Visualization
            <div className="flex gap-2">
              <Button
                size="sm"
                variant={isRunning ? "secondary" : "default"}
                onClick={() => setIsRunning(!isRunning)}
              >
                {isRunning ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4" />}
                {isRunning ? 'Pause' : 'Start'}
              </Button>
              <Button size="sm" variant="outline" onClick={reset}>
                <RotateCcw className="w-4 h-4" />
                Reset
              </Button>
            </div>
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="text-sm font-medium">Temperature (α): {temperature.toFixed(3)}</label>
              <Slider
                value={[temperature]}
                onValueChange={([v]) => setTemperature(v)}
                min={0.01}
                max={1}
                step={0.01}
                className="mt-2"
              />
              <p className="text-xs text-muted-foreground mt-1">
                Controls exploration vs exploitation trade-off
              </p>
            </div>
            <div>
              <label className="text-sm font-medium">Exploration Level: {exploration}%</label>
              <Slider
                value={[exploration]}
                onValueChange={([v]) => setExploration(v)}
                min={0}
                max={100}
                step={1}
                className="mt-2"
              />
              <p className="text-xs text-muted-foreground mt-1">
                Policy stochasticity for exploration
              </p>
            </div>
          </div>
          
          <div className="text-sm text-muted-foreground">
            Step: {step} | Current α: {temperature.toFixed(3)}
          </div>
        </CardContent>
      </Card>

      {/* Component Selector */}
      <div className="flex gap-2">
        <Button
          variant={selectedComponent === 'actor' ? 'default' : 'outline'}
          onClick={() => setSelectedComponent('actor')}
        >
          Actor Network
        </Button>
        <Button
          variant={selectedComponent === 'critic' ? 'default' : 'outline'}
          onClick={() => setSelectedComponent('critic')}
        >
          Critic Networks
        </Button>
        <Button
          variant={selectedComponent === 'temperature' ? 'default' : 'outline'}
          onClick={() => setSelectedComponent('temperature')}
        >
          Temperature Control
        </Button>
      </div>

      {/* Visualization Area */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Policy Distribution */}
        {selectedComponent === 'actor' && (
          <Card>
            <CardHeader>
              <CardTitle className="text-lg">Policy Distribution</CardTitle>
            </CardHeader>
            <CardContent>
              <PolicyDistribution />
              <p className="text-sm text-muted-foreground mt-2">
                Shows how the policy distribution changes with temperature.
                Higher temperature = more exploration.
              </p>
            </CardContent>
          </Card>
        )}

        {/* Q-Function Heatmap */}
        {selectedComponent === 'critic' && (
          <Card>
            <CardHeader>
              <CardTitle className="text-lg">Q-Function Landscape</CardTitle>
            </CardHeader>
            <CardContent>
              <QFunctionVisual />
              <p className="text-sm text-muted-foreground mt-2">
                Heatmap showing Q-values across state-action space.
                Uses double Q-learning to prevent overestimation.
              </p>
            </CardContent>
          </Card>
        )}

        {/* Performance Metrics */}
        <Card>
          <CardHeader>
            <CardTitle className="text-lg">Performance Metrics</CardTitle>
          </CardHeader>
          <CardContent>
            <ResponsiveContainer width="100%" height={250}>
              <LineChart data={performanceData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="step" />
                <YAxis />
                <Tooltip />
                <Legend />
                <Line type="monotone" dataKey="reward" stroke="#3b82f6" strokeWidth={2} dot={false} />
                <Line type="monotone" dataKey="value" stroke="#10b981" strokeWidth={2} dot={false} />
              </LineChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>

        {/* Q-Value Tracking */}
        {selectedComponent === 'critic' && (
          <Card>
            <CardHeader>
              <CardTitle className="text-lg">Q-Value Evolution</CardTitle>
            </CardHeader>
            <CardContent>
              <ResponsiveContainer width="100%" height={250}>
                <LineChart data={qValueData}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="step" />
                  <YAxis />
                  <Tooltip />
                  <Legend />
                  <Line type="monotone" dataKey="q1" stroke="#f59e0b" strokeWidth={1} dot={false} />
                  <Line type="monotone" dataKey="q2" stroke="#ef4444" strokeWidth={1} dot={false} />
                  <Line type="monotone" dataKey="minQ" stroke="#10b981" strokeWidth={2} dot={false} />
                </LineChart>
              </ResponsiveContainer>
              <p className="text-sm text-muted-foreground mt-2">
                Twin Q-networks prevent overestimation bias
              </p>
            </CardContent>
          </Card>
        )}

        {/* Entropy Tracking */}
        {selectedComponent === 'temperature' && (
          <Card>
            <CardHeader>
              <CardTitle className="text-lg">Entropy & Temperature</CardTitle>
            </CardHeader>
            <CardContent>
              <ResponsiveContainer width="100%" height={250}>
                <AreaChart data={entropyData}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="step" />
                  <YAxis />
                  <Tooltip />
                  <Legend />
                  <Area type="monotone" dataKey="entropy" stroke="#8b5cf6" fill="#8b5cf6" fillOpacity={0.3} />
                  <Area type="monotone" dataKey="objective" stroke="#ec4899" fill="#ec4899" fillOpacity={0.3} />
                </AreaChart>
              </ResponsiveContainer>
              <p className="text-sm text-muted-foreground mt-2">
                Automatic temperature adjustment maintains target entropy
              </p>
            </CardContent>
          </Card>
        )}
      </div>

      {/* Info Card */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Info className="w-5 h-5" />
            Understanding SAC
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <div>
            <h4 className="font-medium mb-2">Key Concepts</h4>
            <ul className="list-disc pl-5 space-y-1 text-sm">
              <li><strong>Maximum Entropy RL</strong>: Maximizes both reward and policy entropy</li>
              <li><strong>Temperature Parameter (α)</strong>: Controls exploration-exploitation trade-off</li>
              <li><strong>Twin Q-Networks</strong>: Prevents Q-value overestimation</li>
              <li><strong>Stochastic Policy</strong>: Naturally explores while learning</li>
            </ul>
          </div>
          
          <div>
            <h4 className="font-medium mb-2">Advantages over PPO</h4>
            <ul className="list-disc pl-5 space-y-1 text-sm">
              <li><strong>Sample Efficiency</strong>: Off-policy learning from replay buffer</li>
              <li><strong>Stability</strong>: No need for careful hyperparameter tuning</li>
              <li><strong>Exploration</strong>: Built-in exploration through entropy maximization</li>
              <li><strong>Continuous Actions</strong>: Naturally handles continuous action spaces</li>
            </ul>
          </div>
          
          <div>
            <h4 className="font-medium mb-2">The SAC Objective</h4>
            <p className="text-sm font-mono bg-muted p-2 rounded">
              J(π) = E[Σ γ^t (r_t + α H(π(·|s_t)))]
            </p>
            <p className="text-sm mt-2">
              Where H is the entropy, encouraging diverse actions
            </p>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}