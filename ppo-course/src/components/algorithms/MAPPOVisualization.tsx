'use client';

import React, { useState, useEffect, useRef, useCallback } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Badge } from '@/components/ui/badge';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Slider } from '@/components/ui/slider';
import { Switch } from '@/components/ui/switch';
import { 
  Play, Pause, RotateCcw, Users, Brain, MessageSquare, 
  Target, Award, Zap, Settings, GitBranch, Network
} from 'lucide-react';
import { LineChart, Line, ScatterPlot, ResponsiveContainer, XAxis, YAxis, CartesianGrid, Tooltip, Legend, AreaChart, Area } from 'recharts';

interface Agent {
  id: number;
  x: number;
  y: number;
  vx: number;
  vy: number;
  color: string;
  reward: number;
  action: string;
  communication: string[];
  isActive: boolean;
}

interface Environment {
  id: string;
  name: string;
  description: string;
  nAgents: number;
  gridSize: number;
  hasTargets: boolean;
  hasCommunication: boolean;
  cooperativeReward: boolean;
}

export function MAPPOVisualization() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const animationRef = useRef<number>();
  
  const [isTraining, setIsTraining] = useState(false);
  const [episode, setEpisode] = useState(0);
  const [step, setStep] = useState(0);
  const [agents, setAgents] = useState<Agent[]>([]);
  const [targets, setTargets] = useState<{x: number, y: number, collected: boolean}[]>([]);
  const [environment, setEnvironment] = useState<string>('cooperative');
  const [centralizedCritic, setCentralizedCritic] = useState(true);
  const [parameterSharing, setParameterSharing] = useState(false);
  const [communicationEnabled, setCommunicationEnabled] = useState(true);
  const [nAgents, setNAgents] = useState([3]);
  const [learningRate, setLearningRate] = useState([0.0003]);
  
  const [metrics, setMetrics] = useState({
    totalReward: 0,
    individualRewards: [] as number[],
    cooperationScore: 0,
    communicationUsage: 0,
    convergenceSpeed: 0
  });
  
  const [trainingHistory, setTrainingHistory] = useState<any[]>([]);
  const [rewardHistory, setRewardHistory] = useState<any[]>([]);
  const [cooperationHistory, setCoopeartionHistory] = useState<any[]>([]);

  const environments: Environment[] = [
    {
      id: 'cooperative',
      name: 'Cooperative Navigation',
      description: 'Agents must reach targets while avoiding collisions',
      nAgents: 3,
      gridSize: 400,
      hasTargets: true,
      hasCommunication: true,
      cooperativeReward: true
    },
    {
      id: 'predator_prey',
      name: 'Predator-Prey',
      description: 'Predators cooperate to catch prey agents',
      nAgents: 4,
      gridSize: 400,
      hasTargets: false,
      hasCommunication: true,
      cooperativeReward: false
    },
    {
      id: 'formation',
      name: 'Formation Control',
      description: 'Agents maintain formation while moving',
      nAgents: 5,
      gridSize: 400,
      hasTargets: false,
      hasCommunication: true,
      cooperativeReward: true
    },
    {
      id: 'resource',
      name: 'Resource Collection',
      description: 'Agents compete and cooperate for limited resources',
      nAgents: 4,
      gridSize: 400,
      hasTargets: true,
      hasCommunication: false,
      cooperativeReward: false
    }
  ];

  const currentEnv = environments.find(e => e.id === environment) || environments[0];

  // Initialize environment
  useEffect(() => {
    initializeEnvironment();
  }, [environment, nAgents[0]]);

  const initializeEnvironment = useCallback(() => {
    const newAgents: Agent[] = [];
    const agentCount = Math.min(nAgents[0], 6); // Max 6 agents for visualization
    
    const colors = ['#3b82f6', '#ef4444', '#10b981', '#f59e0b', '#8b5cf6', '#06b6d4'];
    
    for (let i = 0; i < agentCount; i++) {
      newAgents.push({
        id: i,
        x: Math.random() * (currentEnv.gridSize - 40) + 20,
        y: Math.random() * (currentEnv.gridSize - 40) + 20,
        vx: 0,
        vy: 0,
        color: colors[i],
        reward: 0,
        action: 'stay',
        communication: [],
        isActive: true
      });
    }
    
    setAgents(newAgents);
    
    // Initialize targets if environment has them
    if (currentEnv.hasTargets) {
      const newTargets = [];
      for (let i = 0; i < agentCount; i++) {
        newTargets.push({
          x: Math.random() * (currentEnv.gridSize - 40) + 20,
          y: Math.random() * (currentEnv.gridSize - 40) + 20,
          collected: false
        });
      }
      setTargets(newTargets);
    } else {
      setTargets([]);
    }
    
    // Reset metrics
    setMetrics({
      totalReward: 0,
      individualRewards: new Array(agentCount).fill(0),
      cooperationScore: 0,
      communicationUsage: 0,
      convergenceSpeed: 0
    });
    
    setEpisode(0);
    setStep(0);
  }, [environment, nAgents, currentEnv]);

  // Training simulation
  useEffect(() => {
    if (!isTraining) return;

    const interval = setInterval(() => {
      setStep(prev => {
        const newStep = prev + 1;
        
        // Update agent positions and behaviors
        setAgents(prevAgents => {
          return prevAgents.map(agent => {
            const newAgent = { ...agent };
            
            // Simulate agent behavior based on environment
            switch (environment) {
              case 'cooperative':
                updateCooperativeAgent(newAgent);
                break;
              case 'predator_prey':
                updatePredatorPreyAgent(newAgent, prevAgents);
                break;
              case 'formation':
                updateFormationAgent(newAgent, prevAgents);
                break;
              case 'resource':
                updateResourceAgent(newAgent);
                break;
            }
            
            // Add some randomness for realistic movement
            newAgent.x += (Math.random() - 0.5) * 4;
            newAgent.y += (Math.random() - 0.5) * 4;
            
            // Keep agents in bounds
            newAgent.x = Math.max(20, Math.min(currentEnv.gridSize - 20, newAgent.x));
            newAgent.y = Math.max(20, Math.min(currentEnv.gridSize - 20, newAgent.y));
            
            return newAgent;
          });
        });
        
        // Update metrics every few steps
        if (newStep % 10 === 0) {
          updateMetrics();
          
          // Add to training history
          const newHistoryPoint = {
            step: newStep,
            episode: episode,
            totalReward: metrics.totalReward,
            cooperationScore: metrics.cooperationScore,
            communicationUsage: metrics.communicationUsage
          };
          
          setTrainingHistory(prev => [...prev.slice(-199), newHistoryPoint]);
        }
        
        // Reset episode every 200 steps
        if (newStep % 200 === 0) {
          setEpisode(prev => prev + 1);
          initializeEnvironment();
          return 0;
        }
        
        return newStep;
      });
    }, 100);

    return () => clearInterval(interval);
  }, [isTraining, environment, episode, metrics]);

  const updateCooperativeAgent = (agent: Agent) => {
    // Find nearest uncollected target
    const uncollectedTargets = targets.filter(t => !t.collected);
    if (uncollectedTargets.length > 0) {
      const nearest = uncollectedTargets.reduce((closest, target) => {
        const distToTarget = Math.sqrt((agent.x - target.x) ** 2 + (agent.y - target.y) ** 2);
        const distToClosest = Math.sqrt((agent.x - closest.x) ** 2 + (agent.y - closest.y) ** 2);
        return distToTarget < distToClosest ? target : closest;
      });
      
      // Move towards target
      const dx = nearest.x - agent.x;
      const dy = nearest.y - agent.y;
      const dist = Math.sqrt(dx * dx + dy * dy);
      
      if (dist > 5) {
        agent.x += (dx / dist) * 2;
        agent.y += (dy / dist) * 2;
        agent.action = 'move_to_target';
      } else {
        agent.action = 'collect';
        agent.reward += 10;
        // Mark target as collected
        nearest.collected = true;
      }
    }
    
    // Add communication if enabled
    if (communicationEnabled) {
      agent.communication = [`target_${Math.floor(Math.random() * 3)}`, 'cooperate'];
    }
  };

  const updatePredatorPreyAgent = (agent: Agent, allAgents: Agent[]) => {
    // Predators (id 0, 1) try to catch prey (id 2, 3)
    const isPredator = agent.id < 2;
    
    if (isPredator) {
      // Find nearest prey
      const prey = allAgents.filter(a => a.id >= 2);
      if (prey.length > 0) {
        const nearest = prey.reduce((closest, p) => {
          const distToPrey = Math.sqrt((agent.x - p.x) ** 2 + (agent.y - p.y) ** 2);
          const distToClosest = Math.sqrt((agent.x - closest.x) ** 2 + (agent.y - closest.y) ** 2);
          return distToPrey < distToClosest ? p : closest;
        });
        
        // Chase prey
        const dx = nearest.x - agent.x;
        const dy = nearest.y - agent.y;
        const dist = Math.sqrt(dx * dx + dy * dy);
        
        if (dist > 0) {
          agent.x += (dx / dist) * 2.5;
          agent.y += (dy / dist) * 2.5;
          agent.action = 'chase';
          
          if (dist < 30) {
            agent.reward += 1;
          }
        }
      }
    } else {
      // Prey tries to escape
      const predators = allAgents.filter(a => a.id < 2);
      let escapeX = 0, escapeY = 0;
      
      predators.forEach(pred => {
        const dx = agent.x - pred.x;
        const dy = agent.y - pred.y;
        const dist = Math.sqrt(dx * dx + dy * dy);
        
        if (dist < 100) {
          escapeX += dx / dist;
          escapeY += dy / dist;
        }
      });
      
      if (escapeX !== 0 || escapeY !== 0) {
        const escapeNorm = Math.sqrt(escapeX * escapeX + escapeY * escapeY);
        agent.x += (escapeX / escapeNorm) * 3;
        agent.y += (escapeY / escapeNorm) * 3;
        agent.action = 'escape';
        agent.reward += 0.1;
      } else {
        agent.action = 'explore';
        agent.reward += 0.05;
      }
    }
  };

  const updateFormationAgent = (agent: Agent, allAgents: Agent[]) => {
    // Maintain formation - agents try to stay in a line or circle
    const centerX = allAgents.reduce((sum, a) => sum + a.x, 0) / allAgents.length;
    const centerY = allAgents.reduce((sum, a) => sum + a.y, 0) / allAgents.length;
    
    // Desired position in formation (circle)
    const angle = (agent.id / allAgents.length) * 2 * Math.PI;
    const radius = 60;
    const targetX = centerX + Math.cos(angle) * radius;
    const targetY = centerY + Math.sin(angle) * radius;
    
    // Move towards formation position
    const dx = targetX - agent.x;
    const dy = targetY - agent.y;
    const dist = Math.sqrt(dx * dx + dy * dy);
    
    if (dist > 5) {
      agent.x += (dx / dist) * 1.5;
      agent.y += (dy / dist) * 1.5;
      agent.action = 'form';
    } else {
      agent.action = 'maintain';
      agent.reward += 0.5;
    }
    
    // Bonus for good formation
    if (dist < 10) {
      agent.reward += 1;
    }
  };

  const updateResourceAgent = (agent: Agent) => {
    // Compete for resources (targets)
    const availableTargets = targets.filter(t => !t.collected);
    if (availableTargets.length > 0) {
      const nearest = availableTargets.reduce((closest, target) => {
        const distToTarget = Math.sqrt((agent.x - target.x) ** 2 + (agent.y - target.y) ** 2);
        const distToClosest = Math.sqrt((agent.x - closest.x) ** 2 + (agent.y - closest.y) ** 2);
        return distToTarget < distToClosest ? target : closest;
      });
      
      // Move towards resource
      const dx = nearest.x - agent.x;
      const dy = nearest.y - agent.y;
      const dist = Math.sqrt(dx * dx + dy * dy);
      
      if (dist > 5) {
        agent.x += (dx / dist) * 2.2;
        agent.y += (dy / dist) * 2.2;
        agent.action = 'compete';
      } else {
        agent.action = 'collect';
        agent.reward += 15;
        nearest.collected = true;
      }
    } else {
      agent.action = 'search';
    }
  };

  const updateMetrics = () => {
    setMetrics(prev => {
      const totalReward = agents.reduce((sum, agent) => sum + agent.reward, 0);
      const individualRewards = agents.map(agent => agent.reward);
      
      // Cooperation score based on reward variance (lower variance = more cooperation)
      const avgReward = totalReward / agents.length;
      const variance = individualRewards.reduce((sum, r) => sum + (r - avgReward) ** 2, 0) / agents.length;
      const cooperationScore = Math.max(0, 100 - variance);
      
      // Communication usage
      const communicationUsage = communicationEnabled ? 
        agents.filter(a => a.communication.length > 0).length / agents.length * 100 : 0;
      
      const convergenceSpeed = Math.min(100, episode * 2);
      
      return {
        totalReward,
        individualRewards,
        cooperationScore,
        communicationUsage,
        convergenceSpeed
      };
    });
  };

  // Canvas rendering
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const render = () => {
      // Clear canvas
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      
      // Draw grid
      ctx.strokeStyle = '#e5e7eb';
      ctx.lineWidth = 1;
      for (let i = 0; i <= canvas.width; i += 40) {
        ctx.beginPath();
        ctx.moveTo(i, 0);
        ctx.lineTo(i, canvas.height);
        ctx.stroke();
      }
      for (let i = 0; i <= canvas.height; i += 40) {
        ctx.beginPath();
        ctx.moveTo(0, i);
        ctx.lineTo(canvas.width, i);
        ctx.stroke();
      }
      
      // Draw targets
      targets.forEach(target => {
        if (!target.collected) {
          ctx.fillStyle = '#fbbf24';
          ctx.strokeStyle = '#f59e0b';
          ctx.lineWidth = 2;
          ctx.beginPath();
          ctx.arc(target.x, target.y, 8, 0, 2 * Math.PI);
          ctx.fill();
          ctx.stroke();
          
          // Target symbol
          ctx.strokeStyle = '#92400e';
          ctx.lineWidth = 2;
          ctx.beginPath();
          ctx.moveTo(target.x - 4, target.y);
          ctx.lineTo(target.x + 4, target.y);
          ctx.moveTo(target.x, target.y - 4);
          ctx.lineTo(target.x, target.y + 4);
          ctx.stroke();
        }
      });
      
      // Draw communication lines if enabled
      if (communicationEnabled && environment === 'cooperative') {
        agents.forEach((agent, i) => {
          agents.forEach((otherAgent, j) => {
            if (i !== j) {
              const dist = Math.sqrt((agent.x - otherAgent.x) ** 2 + (agent.y - otherAgent.y) ** 2);
              if (dist < 100) {
                const alpha = Math.max(0.1, 1 - dist / 100);
                ctx.strokeStyle = `rgba(139, 92, 246, ${alpha * 0.5})`;
                ctx.lineWidth = 1;
                ctx.beginPath();
                ctx.moveTo(agent.x, agent.y);
                ctx.lineTo(otherAgent.x, otherAgent.y);
                ctx.stroke();
              }
            }
          });
        });
      }
      
      // Draw agents
      agents.forEach(agent => {
        // Agent body
        ctx.fillStyle = agent.color;
        ctx.strokeStyle = agent.isActive ? '#1f2937' : '#9ca3af';
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.arc(agent.x, agent.y, 12, 0, 2 * Math.PI);
        ctx.fill();
        ctx.stroke();
        
        // Agent ID
        ctx.fillStyle = '#ffffff';
        ctx.font = '12px Arial';
        ctx.textAlign = 'center';
        ctx.fillText(agent.id.toString(), agent.x, agent.y + 4);
        
        // Action indicator
        ctx.fillStyle = agent.color;
        ctx.font = '10px Arial';
        ctx.fillText(agent.action, agent.x, agent.y - 18);
        
        // Reward display
        ctx.fillStyle = agent.reward > 0 ? '#10b981' : '#ef4444';
        ctx.fillText(`${agent.reward.toFixed(1)}`, agent.x, agent.y + 25);
        
        // Communication bubble
        if (communicationEnabled && agent.communication.length > 0) {
          ctx.fillStyle = 'rgba(139, 92, 246, 0.8)';
          ctx.beginPath();
          ctx.arc(agent.x + 15, agent.y - 15, 6, 0, 2 * Math.PI);
          ctx.fill();
        }
      });
      
      animationRef.current = requestAnimationFrame(render);
    };

    render();

    return () => {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }
    };
  }, [agents, targets, communicationEnabled, environment]);

  const handleStart = () => {
    setIsTraining(true);
  };

  const handlePause = () => {
    setIsTraining(false);
  };

  const handleReset = () => {
    setIsTraining(false);
    initializeEnvironment();
    setTrainingHistory([]);
    setRewardHistory([]);
    setCoopeartionHistory([]);
  };

  return (
    <div className="space-y-6">
      {/* Control Panel */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center justify-between">
            <span className="flex items-center gap-2">
              <Users className="w-5 h-5" />
              MAPPO Visualization
            </span>
            <div className="flex items-center gap-2">
              <Badge variant="outline">
                Episode {episode}
              </Badge>
              <Badge variant="outline">
                Step {step}
              </Badge>
            </div>
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          {/* Environment Selection */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <label className="text-sm font-medium mb-2 block">Environment</label>
              <Select value={environment} onValueChange={setEnvironment}>
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {environments.map(env => (
                    <SelectItem key={env.id} value={env.id}>
                      {env.name}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
              <p className="text-xs text-muted-foreground mt-1">
                {currentEnv.description}
              </p>
            </div>
            
            <div className="flex items-end">
              <div className="flex gap-2">
                <Button 
                  size="sm" 
                  variant={isTraining ? "secondary" : "default"}
                  onClick={isTraining ? handlePause : handleStart}
                >
                  {isTraining ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4" />}
                  {isTraining ? 'Pause' : 'Start'}
                </Button>
                <Button size="sm" variant="outline" onClick={handleReset}>
                  <RotateCcw className="w-4 h-4" />
                  Reset
                </Button>
              </div>
            </div>
          </div>
          
          {/* Configuration */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div className="space-y-2">
              <label className="text-sm font-medium">Number of Agents: {nAgents[0]}</label>
              <Slider
                value={nAgents}
                onValueChange={setNAgents}
                min={2}
                max={6}
                step={1}
                disabled={isTraining}
              />
            </div>
            
            <div className="space-y-2">
              <label className="text-sm font-medium">Learning Rate: {learningRate[0]}</label>
              <Slider
                value={learningRate}
                onValueChange={setLearningRate}
                min={0.0001}
                max={0.001}
                step={0.0001}
                disabled={isTraining}
              />
            </div>
            
            <div className="flex items-center space-x-2">
              <Switch
                id="centralized"
                checked={centralizedCritic}
                onCheckedChange={setCentralizedCritic}
                disabled={isTraining}
              />
              <label htmlFor="centralized" className="text-sm font-medium">
                Centralized Critic
              </label>
            </div>
            
            <div className="flex items-center space-x-2">
              <Switch
                id="sharing"
                checked={parameterSharing}
                onCheckedChange={setParameterSharing}
                disabled={isTraining}
              />
              <label htmlFor="sharing" className="text-sm font-medium">
                Parameter Sharing
              </label>
            </div>
          </div>
          
          <div className="flex items-center space-x-2">
            <Switch
              id="communication"
              checked={communicationEnabled}
              onCheckedChange={setCommunicationEnabled}
              disabled={isTraining}
            />
            <label htmlFor="communication" className="text-sm font-medium">
              Enable Communication
            </label>
            <MessageSquare className="w-4 h-4 text-muted-foreground" />
          </div>
        </CardContent>
      </Card>

      <Tabs defaultValue="environment" className="space-y-4">
        <TabsList className="grid w-full grid-cols-4">
          <TabsTrigger value="environment">Environment</TabsTrigger>
          <TabsTrigger value="metrics">Metrics</TabsTrigger>
          <TabsTrigger value="analysis">Analysis</TabsTrigger>
          <TabsTrigger value="config">Configuration</TabsTrigger>
        </TabsList>

        {/* Environment Tab */}
        <TabsContent value="environment" className="space-y-4">
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
            <div className="lg:col-span-2">
              <Card>
                <CardHeader>
                  <CardTitle className="text-lg">Multi-Agent Environment</CardTitle>
                </CardHeader>
                <CardContent>
                  <canvas
                    ref={canvasRef}
                    width={400}
                    height={400}
                    className="border rounded-lg w-full max-w-md mx-auto"
                    style={{ imageRendering: 'pixelated' }}
                  />
                  <div className="mt-4 grid grid-cols-2 gap-4 text-sm">
                    <div>
                      <h4 className="font-medium mb-2">Legend</h4>
                      <div className="space-y-1">
                        <div className="flex items-center gap-2">
                          <div className="w-3 h-3 rounded-full bg-blue-500"></div>
                          <span>Agents</span>
                        </div>
                        {currentEnv.hasTargets && (
                          <div className="flex items-center gap-2">
                            <div className="w-3 h-3 rounded-full bg-yellow-500"></div>
                            <span>Targets</span>
                          </div>
                        )}
                        {communicationEnabled && (
                          <div className="flex items-center gap-2">
                            <div className="w-3 h-1 bg-purple-500"></div>
                            <span>Communication</span>
                          </div>
                        )}
                      </div>
                    </div>
                    <div>
                      <h4 className="font-medium mb-2">Environment Info</h4>
                      <div className="space-y-1">
                        <p>Cooperative: {currentEnv.cooperativeReward ? '✓' : '✗'}</p>
                        <p>Communication: {currentEnv.hasCommunication ? '✓' : '✗'}</p>
                        <p>Agents: {agents.length}</p>
                      </div>
                    </div>
                  </div>
                </CardContent>
              </Card>
            </div>
            
            <div className="space-y-4">
              <Card>
                <CardHeader>
                  <CardTitle className="text-lg">Agent Status</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="space-y-3">
                    {agents.map(agent => (
                      <div key={agent.id} className="space-y-1">
                        <div className="flex items-center justify-between">
                          <span className="flex items-center gap-2">
                            <div 
                              className="w-3 h-3 rounded-full" 
                              style={{ backgroundColor: agent.color }}
                            />
                            Agent {agent.id}
                          </span>
                          <Badge variant="outline" className="text-xs">
                            {agent.action}
                          </Badge>
                        </div>
                        <div className="text-xs text-muted-foreground">
                          Reward: {agent.reward.toFixed(1)}
                        </div>
                        {agent.communication.length > 0 && (
                          <div className="text-xs text-purple-600">
                            Comm: {agent.communication.join(', ')}
                          </div>
                        )}
                      </div>
                    ))}
                  </div>
                </CardContent>
              </Card>
              
              <Card>
                <CardHeader>
                  <CardTitle className="text-lg">Current Metrics</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="space-y-3">
                    <div className="flex justify-between">
                      <span className="text-sm">Total Reward</span>
                      <span className="font-medium">{metrics.totalReward.toFixed(1)}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-sm">Cooperation Score</span>
                      <span className="font-medium">{metrics.cooperationScore.toFixed(1)}%</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-sm">Communication Usage</span>
                      <span className="font-medium">{metrics.communicationUsage.toFixed(1)}%</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-sm">Convergence Speed</span>
                      <span className="font-medium">{metrics.convergenceSpeed.toFixed(1)}%</span>
                    </div>
                  </div>
                </CardContent>
              </Card>
            </div>
          </div>
        </TabsContent>

        {/* Metrics Tab */}
        <TabsContent value="metrics" className="space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <Card>
              <CardHeader>
                <CardTitle className="text-lg">Training Progress</CardTitle>
              </CardHeader>
              <CardContent>
                <ResponsiveContainer width="100%" height={250}>
                  <LineChart data={trainingHistory}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="step" />
                    <YAxis />
                    <Tooltip />
                    <Legend />
                    <Line 
                      type="monotone" 
                      dataKey="totalReward" 
                      stroke="#3b82f6" 
                      strokeWidth={2}
                      dot={false}
                      name="Total Reward"
                    />
                  </LineChart>
                </ResponsiveContainer>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle className="text-lg">Cooperation & Communication</CardTitle>
              </CardHeader>
              <CardContent>
                <ResponsiveContainer width="100%" height={250}>
                  <AreaChart data={trainingHistory}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="step" />
                    <YAxis />
                    <Tooltip />
                    <Legend />
                    <Area 
                      type="monotone" 
                      dataKey="cooperationScore" 
                      stackId="1"
                      stroke="#10b981" 
                      fill="#10b981"
                      fillOpacity={0.3}
                      name="Cooperation Score"
                    />
                    <Area 
                      type="monotone" 
                      dataKey="communicationUsage" 
                      stackId="2"
                      stroke="#8b5cf6" 
                      fill="#8b5cf6"
                      fillOpacity={0.3}
                      name="Communication Usage"
                    />
                  </AreaChart>
                </ResponsiveContainer>
              </CardContent>
            </Card>
          </div>

          <Card>
            <CardHeader>
              <CardTitle>Individual Agent Performance</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                {agents.map(agent => (
                  <div key={agent.id} className="space-y-2">
                    <h4 className="font-medium flex items-center gap-2">
                      <div 
                        className="w-3 h-3 rounded-full" 
                        style={{ backgroundColor: agent.color }}
                      />
                      Agent {agent.id}
                    </h4>
                    <div className="text-sm space-y-1">
                      <div className="flex justify-between">
                        <span>Reward:</span>
                        <span className="font-medium">{agent.reward.toFixed(1)}</span>
                      </div>
                      <div className="flex justify-between">
                        <span>Action:</span>
                        <Badge variant="outline" className="text-xs">
                          {agent.action}
                        </Badge>
                      </div>
                      <div className="flex justify-between">
                        <span>Position:</span>
                        <span className="text-xs">
                          ({agent.x.toFixed(0)}, {agent.y.toFixed(0)})
                        </span>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        {/* Analysis Tab */}
        <TabsContent value="analysis" className="space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <Card>
              <CardHeader>
                <CardTitle>MAPPO Architecture Analysis</CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="space-y-3">
                  <div>
                    <h4 className="font-medium mb-2">Actor Networks</h4>
                    <div className="text-sm space-y-1">
                      <p>Type: {parameterSharing ? 'Shared Parameters' : 'Individual Networks'}</p>
                      <p>Architecture: [State] → [64] → [64] → [Actions]</p>
                      <p>Activation: ReLU → ReLU → Softmax</p>
                    </div>
                  </div>
                  
                  <div>
                    <h4 className="font-medium mb-2">Critic Networks</h4>
                    <div className="text-sm space-y-1">
                      <p>Type: {centralizedCritic ? 'Centralized Critic' : 'Individual Critics'}</p>
                      <p>Input: {centralizedCritic ? 'Global State' : 'Local State'}</p>
                      <p>Output: Value Estimates</p>
                    </div>
                  </div>
                  
                  {communicationEnabled && (
                    <div>
                      <h4 className="font-medium mb-2">Communication</h4>
                      <div className="text-sm space-y-1">
                        <p>Protocol: Broadcast messaging</p>
                        <p>Message Size: 8 dimensions</p>
                        <p>Range: Limited by distance</p>
                      </div>
                    </div>
                  )}
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>Environment Characteristics</CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="space-y-3">
                  <div>
                    <h4 className="font-medium mb-2">Multi-Agent Dynamics</h4>
                    <div className="text-sm space-y-1">
                      <p>Partial Observability: {currentEnv.id !== 'formation' ? 'Yes' : 'No'}</p>
                      <p>Action Space: Continuous movement</p>
                      <p>Reward Structure: {currentEnv.cooperativeReward ? 'Cooperative' : 'Competitive'}</p>
                    </div>
                  </div>
                  
                  <div>
                    <h4 className="font-medium mb-2">Challenges</h4>
                    <div className="text-sm space-y-1">
                      <p>• Non-stationary environment due to other agents</p>
                      <p>• Credit assignment problem</p>
                      <p>• Exploration vs exploitation in multi-agent setting</p>
                      <p>• Communication protocol learning</p>
                    </div>
                  </div>
                  
                  <div>
                    <h4 className="font-medium mb-2">MAPPO Advantages</h4>
                    <div className="text-sm space-y-1">
                      <p>• Centralized training, decentralized execution</p>
                      <p>• Stable policy gradients</p>
                      <p>• Scalable to many agents</p>
                      <p>• Handles partial observability</p>
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>

          <Card>
            <CardHeader>
              <CardTitle>Performance Insights</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                <div className="p-4 bg-muted rounded-lg">
                  <h4 className="font-medium mb-2 flex items-center gap-2">
                    <Target className="w-4 h-4" />
                    Coordination Quality
                  </h4>
                  <p className="text-sm text-muted-foreground mb-2">
                    How well agents coordinate their actions
                  </p>
                  <div className="text-2xl font-bold">
                    {metrics.cooperationScore.toFixed(0)}%
                  </div>
                  <p className="text-xs text-muted-foreground">
                    {metrics.cooperationScore > 70 ? 'Excellent' : 
                     metrics.cooperationScore > 50 ? 'Good' : 'Needs Improvement'}
                  </p>
                </div>
                
                <div className="p-4 bg-muted rounded-lg">
                  <h4 className="font-medium mb-2 flex items-center gap-2">
                    <MessageSquare className="w-4 h-4" />
                    Communication Efficiency
                  </h4>
                  <p className="text-sm text-muted-foreground mb-2">
                    Effective use of communication channels
                  </p>
                  <div className="text-2xl font-bold">
                    {metrics.communicationUsage.toFixed(0)}%
                  </div>
                  <p className="text-xs text-muted-foreground">
                    {communicationEnabled ? 'Active' : 'Disabled'}
                  </p>
                </div>
                
                <div className="p-4 bg-muted rounded-lg">
                  <h4 className="font-medium mb-2 flex items-center gap-2">
                    <Zap className="w-4 h-4" />
                    Learning Speed
                  </h4>
                  <p className="text-sm text-muted-foreground mb-2">
                    Rate of improvement across episodes
                  </p>
                  <div className="text-2xl font-bold">
                    {metrics.convergenceSpeed.toFixed(0)}%
                  </div>
                  <p className="text-xs text-muted-foreground">
                    Episode {episode}
                  </p>
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        {/* Configuration Tab */}
        <TabsContent value="config" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>MAPPO Configuration</CardTitle>
            </CardHeader>
            <CardContent className="space-y-6">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div className="space-y-4">
                  <h3 className="text-lg font-medium">Network Architecture</h3>
                  <div className="space-y-3">
                    <div className="flex items-center justify-between">
                      <label className="text-sm font-medium">Centralized Critic</label>
                      <Switch
                        checked={centralizedCritic}
                        onCheckedChange={setCentralizedCritic}
                        disabled={isTraining}
                      />
                    </div>
                    <p className="text-xs text-muted-foreground">
                      Uses global state information for value estimation
                    </p>
                    
                    <div className="flex items-center justify-between">
                      <label className="text-sm font-medium">Parameter Sharing</label>
                      <Switch
                        checked={parameterSharing}
                        onCheckedChange={setParameterSharing}
                        disabled={isTraining}
                      />
                    </div>
                    <p className="text-xs text-muted-foreground">
                      All agents share the same actor network parameters
                    </p>
                    
                    <div className="flex items-center justify-between">
                      <label className="text-sm font-medium">Communication</label>
                      <Switch
                        checked={communicationEnabled}
                        onCheckedChange={setCommunicationEnabled}
                        disabled={isTraining}
                      />
                    </div>
                    <p className="text-xs text-muted-foreground">
                      Enables agent-to-agent communication
                    </p>
                  </div>
                </div>
                
                <div className="space-y-4">
                  <h3 className="text-lg font-medium">Training Parameters</h3>
                  <div className="space-y-4">
                    <div>
                      <label className="text-sm font-medium block mb-2">
                        Number of Agents: {nAgents[0]}
                      </label>
                      <Slider
                        value={nAgents}
                        onValueChange={setNAgents}
                        min={2}
                        max={6}
                        step={1}
                        disabled={isTraining}
                      />
                    </div>
                    
                    <div>
                      <label className="text-sm font-medium block mb-2">
                        Learning Rate: {learningRate[0].toFixed(4)}
                      </label>
                      <Slider
                        value={learningRate}
                        onValueChange={setLearningRate}
                        min={0.0001}
                        max={0.001}
                        step={0.0001}
                        disabled={isTraining}
                      />
                    </div>
                  </div>
                </div>
              </div>
              
              <div className="border-t pt-4">
                <h3 className="text-lg font-medium mb-4">Environment Settings</h3>
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                  {environments.map(env => (
                    <Card 
                      key={env.id} 
                      className={`cursor-pointer transition-all ${
                        environment === env.id ? 'ring-2 ring-primary' : ''
                      }`}
                      onClick={() => !isTraining && setEnvironment(env.id)}
                    >
                      <CardContent className="p-4">
                        <h4 className="font-medium mb-2">{env.name}</h4>
                        <p className="text-xs text-muted-foreground mb-2">
                          {env.description}
                        </p>
                        <div className="space-y-1 text-xs">
                          <div className="flex items-center gap-2">
                            <Users className="w-3 h-3" />
                            {env.nAgents} agents
                          </div>
                          <div className="flex items-center gap-2">
                            {env.hasCommunication ? (
                              <MessageSquare className="w-3 h-3 text-green-500" />
                            ) : (
                              <MessageSquare className="w-3 h-3 text-gray-400" />
                            )}
                            Communication
                          </div>
                          <div className="flex items-center gap-2">
                            {env.cooperativeReward ? (
                              <Award className="w-3 h-3 text-green-500" />
                            ) : (
                              <Award className="w-3 h-3 text-red-500" />
                            )}
                            {env.cooperativeReward ? 'Cooperative' : 'Competitive'}
                          </div>
                        </div>
                      </CardContent>
                    </Card>
                  ))}
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
}