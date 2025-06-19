'use client';

import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Checkbox } from '@/components/ui/checkbox';
import { Play, Pause, RotateCcw, Settings, BarChart3, GitBranch } from 'lucide-react';
import { LineChart, Line, RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, Radar, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';

interface AlgorithmConfig {
  id: string;
  name: string;
  color: string;
  params: {
    learningRate: number;
    batchSize: number;
    updateFreq?: number;
    epsilon?: number;
    temperature?: number;
    clipRange?: number;
  };
}

interface EnvironmentConfig {
  id: string;
  name: string;
  stateSpace: number;
  actionSpace: number;
  difficulty: 'easy' | 'medium' | 'hard';
}

export function AlgorithmComparison() {
  const [selectedAlgorithms, setSelectedAlgorithms] = useState<string[]>(['ppo', 'sac']);
  const [selectedEnvironment, setSelectedEnvironment] = useState<string>('cartpole');
  const [isRunning, setIsRunning] = useState(false);
  const [episode, setEpisode] = useState(0);
  const [performanceData, setPerformanceData] = useState<any[]>([]);
  const [metricsData, setMetricsData] = useState<any[]>([]);
  const [radarData, setRadarData] = useState<any[]>([]);

  const algorithms: AlgorithmConfig[] = [
    {
      id: 'ppo',
      name: 'PPO',
      color: '#3b82f6',
      params: { learningRate: 3e-4, batchSize: 64, clipRange: 0.2 }
    },
    {
      id: 'sac',
      name: 'SAC',
      color: '#f59e0b',
      params: { learningRate: 3e-4, batchSize: 256, temperature: 0.2 }
    },
    {
      id: 'grpo',
      name: 'GRPO',
      color: '#8b5cf6',
      params: { learningRate: 3e-4, batchSize: 64, clipRange: 0.2 }
    },
    {
      id: 'dqn',
      name: 'DQN',
      color: '#10b981',
      params: { learningRate: 1e-4, batchSize: 32, epsilon: 0.1, updateFreq: 4 }
    }
  ];

  const environments: EnvironmentConfig[] = [
    { id: 'cartpole', name: 'CartPole', stateSpace: 4, actionSpace: 2, difficulty: 'easy' },
    { id: 'lunar', name: 'LunarLander', stateSpace: 8, actionSpace: 4, difficulty: 'medium' },
    { id: 'humanoid', name: 'Humanoid', stateSpace: 376, actionSpace: 17, difficulty: 'hard' },
    { id: 'atari', name: 'Atari Games', stateSpace: 84*84, actionSpace: 18, difficulty: 'medium' }
  ];

  // Simulate training
  useEffect(() => {
    if (!isRunning) return;

    const interval = setInterval(() => {
      setEpisode(prev => prev + 1);
      
      // Generate performance data
      const newDataPoint: any = { episode: episode + 1 };
      selectedAlgorithms.forEach(algoId => {
        const algo = algorithms.find(a => a.id === algoId);
        if (algo) {
          // Simulate different learning curves
          const basePerformance = {
            ppo: Math.min(200, 50 + episode * 1.5 + Math.random() * 20),
            sac: Math.min(250, 30 + episode * 2 + Math.random() * 15),
            grpo: Math.min(220, 60 + episode * 1.6 + Math.random() * 18),
            dqn: Math.min(180, 40 + episode * 1.2 + Math.random() * 25)
          };
          
          newDataPoint[algoId] = basePerformance[algoId as keyof typeof basePerformance] || 0;
        }
      });
      
      setPerformanceData(prev => [...prev.slice(-99), newDataPoint]);
      
      // Update radar data
      if (episode % 10 === 0) {
        const radarPoint = {
          metric: `E${Math.floor(episode / 10)}`,
          ...selectedAlgorithms.reduce((acc, algoId) => ({
            ...acc,
            [algoId]: Math.random() * 100
          }), {})
        };
        setRadarData(prev => [...prev.slice(-5), radarPoint]);
      }
    }, 100);

    return () => clearInterval(interval);
  }, [isRunning, episode, selectedAlgorithms]);

  // Generate metrics comparison
  useEffect(() => {
    const metrics = [
      { name: 'Sample Efficiency', ppo: 70, sac: 90, grpo: 75, dqn: 60 },
      { name: 'Stability', ppo: 85, sac: 90, grpo: 88, dqn: 75 },
      { name: 'Exploration', ppo: 65, sac: 85, grpo: 70, dqn: 70 },
      { name: 'Convergence Speed', ppo: 80, sac: 75, grpo: 82, dqn: 70 },
      { name: 'Hyperparameter Robustness', ppo: 75, sac: 85, grpo: 78, dqn: 65 },
    ];
    
    setMetricsData(metrics);
  }, []);

  const toggleAlgorithm = (algoId: string) => {
    setSelectedAlgorithms(prev => 
      prev.includes(algoId) 
        ? prev.filter(id => id !== algoId)
        : [...prev, algoId]
    );
  };

  const reset = () => {
    setIsRunning(false);
    setEpisode(0);
    setPerformanceData([]);
    setRadarData([]);
  };

  return (
    <div className="space-y-6">
      {/* Control Panel */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center justify-between">
            Algorithm Comparison Tool
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
          {/* Algorithm Selection */}
          <div>
            <label className="text-sm font-medium mb-2 block">Select Algorithms to Compare</label>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              {algorithms.map(algo => (
                <div key={algo.id} className="flex items-center space-x-2">
                  <Checkbox
                    id={algo.id}
                    checked={selectedAlgorithms.includes(algo.id)}
                    onCheckedChange={() => toggleAlgorithm(algo.id)}
                  />
                  <label
                    htmlFor={algo.id}
                    className="text-sm font-medium leading-none peer-disabled:cursor-not-allowed peer-disabled:opacity-70 cursor-pointer"
                  >
                    <span className="flex items-center gap-2">
                      <div 
                        className="w-3 h-3 rounded-full" 
                        style={{ backgroundColor: algo.color }}
                      />
                      {algo.name}
                    </span>
                  </label>
                </div>
              ))}
            </div>
          </div>

          {/* Environment Selection */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <label className="text-sm font-medium mb-2 block">Environment</label>
              <Select value={selectedEnvironment} onValueChange={setSelectedEnvironment}>
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {environments.map(env => (
                    <SelectItem key={env.id} value={env.id}>
                      {env.name} ({env.difficulty})
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
            <div className="flex items-end">
              <div className="text-sm text-muted-foreground">
                Episode: {episode} | Algorithms: {selectedAlgorithms.length}
              </div>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Comparison Tabs */}
      <Tabs defaultValue="performance" className="space-y-4">
        <TabsList className="grid w-full grid-cols-4">
          <TabsTrigger value="performance">Performance</TabsTrigger>
          <TabsTrigger value="metrics">Metrics</TabsTrigger>
          <TabsTrigger value="efficiency">Efficiency</TabsTrigger>
          <TabsTrigger value="analysis">Analysis</TabsTrigger>
        </TabsList>

        {/* Performance Tab */}
        <TabsContent value="performance" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Learning Curves</CardTitle>
            </CardHeader>
            <CardContent>
              <ResponsiveContainer width="100%" height={400}>
                <LineChart data={performanceData}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="episode" />
                  <YAxis />
                  <Tooltip />
                  <Legend />
                  {selectedAlgorithms.map(algoId => {
                    const algo = algorithms.find(a => a.id === algoId);
                    return algo ? (
                      <Line
                        key={algoId}
                        type="monotone"
                        dataKey={algoId}
                        stroke={algo.color}
                        strokeWidth={2}
                        dot={false}
                        name={algo.name}
                      />
                    ) : null;
                  })}
                </LineChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <Card>
              <CardHeader>
                <CardTitle className="text-lg">Episode Rewards</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-2">
                  {selectedAlgorithms.map(algoId => {
                    const algo = algorithms.find(a => a.id === algoId);
                    const lastData = performanceData[performanceData.length - 1];
                    const reward = lastData?.[algoId] || 0;
                    
                    return algo ? (
                      <div key={algoId} className="flex items-center justify-between">
                        <span className="flex items-center gap-2">
                          <div 
                            className="w-3 h-3 rounded-full" 
                            style={{ backgroundColor: algo.color }}
                          />
                          {algo.name}
                        </span>
                        <span className="font-mono font-medium">
                          {reward.toFixed(1)}
                        </span>
                      </div>
                    ) : null;
                  })}
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle className="text-lg">Training Statistics</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-2 text-sm">
                  <div className="flex justify-between">
                    <span>Total Episodes:</span>
                    <span className="font-medium">{episode}</span>
                  </div>
                  <div className="flex justify-between">
                    <span>Environment:</span>
                    <span className="font-medium">
                      {environments.find(e => e.id === selectedEnvironment)?.name}
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span>State Space:</span>
                    <span className="font-medium">
                      {environments.find(e => e.id === selectedEnvironment)?.stateSpace}
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span>Action Space:</span>
                    <span className="font-medium">
                      {environments.find(e => e.id === selectedEnvironment)?.actionSpace}
                    </span>
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>
        </TabsContent>

        {/* Metrics Tab */}
        <TabsContent value="metrics" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Algorithm Comparison Metrics</CardTitle>
            </CardHeader>
            <CardContent>
              <ResponsiveContainer width="100%" height={400}>
                <RadarChart data={metricsData}>
                  <PolarGrid />
                  <PolarAngleAxis dataKey="name" />
                  <PolarRadiusAxis angle={90} domain={[0, 100]} />
                  {selectedAlgorithms.map(algoId => {
                    const algo = algorithms.find(a => a.id === algoId);
                    return algo ? (
                      <Radar
                        key={algoId}
                        name={algo.name}
                        dataKey={algoId}
                        stroke={algo.color}
                        fill={algo.color}
                        fillOpacity={0.3}
                      />
                    ) : null;
                  })}
                  <Legend />
                </RadarChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>Detailed Metrics Comparison</CardTitle>
            </CardHeader>
            <CardContent>
              <ResponsiveContainer width="100%" height={300}>
                <BarChart data={metricsData}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="name" angle={-45} textAnchor="end" height={80} />
                  <YAxis />
                  <Tooltip />
                  <Legend />
                  {selectedAlgorithms.map(algoId => {
                    const algo = algorithms.find(a => a.id === algoId);
                    return algo ? (
                      <Bar
                        key={algoId}
                        dataKey={algoId}
                        fill={algo.color}
                        name={algo.name}
                      />
                    ) : null;
                  })}
                </BarChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>
        </TabsContent>

        {/* Efficiency Tab */}
        <TabsContent value="efficiency" className="space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <Card>
              <CardHeader>
                <CardTitle className="text-lg">Sample Efficiency</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  {selectedAlgorithms.map(algoId => {
                    const algo = algorithms.find(a => a.id === algoId);
                    const efficiency = {
                      ppo: { samples: '2M', time: '2h', memory: '4GB' },
                      sac: { samples: '1M', time: '1.5h', memory: '8GB' },
                      grpo: { samples: '1.8M', time: '2.2h', memory: '5GB' },
                      dqn: { samples: '3M', time: '3h', memory: '6GB' }
                    };
                    
                    const data = efficiency[algoId as keyof typeof efficiency];
                    
                    return algo && data ? (
                      <div key={algoId} className="space-y-2">
                        <h4 className="font-medium flex items-center gap-2">
                          <div 
                            className="w-3 h-3 rounded-full" 
                            style={{ backgroundColor: algo.color }}
                          />
                          {algo.name}
                        </h4>
                        <div className="grid grid-cols-3 gap-2 text-sm">
                          <div>
                            <p className="text-muted-foreground">Samples</p>
                            <p className="font-medium">{data.samples}</p>
                          </div>
                          <div>
                            <p className="text-muted-foreground">Time</p>
                            <p className="font-medium">{data.time}</p>
                          </div>
                          <div>
                            <p className="text-muted-foreground">Memory</p>
                            <p className="font-medium">{data.memory}</p>
                          </div>
                        </div>
                      </div>
                    ) : null;
                  })}
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle className="text-lg">Computational Requirements</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-3">
                  <div className="text-sm space-y-2">
                    <h4 className="font-medium">GPU Usage</h4>
                    {selectedAlgorithms.map(algoId => {
                      const usage = {
                        ppo: 65,
                        sac: 80,
                        grpo: 70,
                        dqn: 60
                      };
                      const algo = algorithms.find(a => a.id === algoId);
                      
                      return algo ? (
                        <div key={algoId} className="flex items-center gap-2">
                          <span className="w-12 text-xs">{algo.name}</span>
                          <div className="flex-1 bg-muted rounded-full h-2">
                            <div 
                              className="h-full rounded-full transition-all"
                              style={{ 
                                width: `${usage[algoId as keyof typeof usage]}%`,
                                backgroundColor: algo.color
                              }}
                            />
                          </div>
                          <span className="text-xs w-10 text-right">
                            {usage[algoId as keyof typeof usage]}%
                          </span>
                        </div>
                      ) : null;
                    })}
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>

          <Card>
            <CardHeader>
              <CardTitle>Parallelization Capabilities</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                {algorithms.map(algo => {
                  const capabilities = {
                    ppo: { parallel: true, workers: 16, scaling: 'Linear' },
                    sac: { parallel: false, workers: 1, scaling: 'N/A' },
                    grpo: { parallel: true, workers: 32, scaling: 'Sub-linear' },
                    dqn: { parallel: true, workers: 8, scaling: 'Limited' }
                  };
                  
                  const data = capabilities[algo.id as keyof typeof capabilities];
                  const isSelected = selectedAlgorithms.includes(algo.id);
                  
                  return (
                    <div 
                      key={algo.id} 
                      className={`p-3 rounded-lg border ${isSelected ? 'border-primary' : 'border-muted'} ${isSelected ? '' : 'opacity-50'}`}
                    >
                      <h4 className="font-medium flex items-center gap-2 mb-2">
                        <div 
                          className="w-3 h-3 rounded-full" 
                          style={{ backgroundColor: algo.color }}
                        />
                        {algo.name}
                      </h4>
                      <div className="space-y-1 text-xs">
                        <p>Parallel: {data.parallel ? '✓' : '✗'}</p>
                        <p>Workers: {data.workers}</p>
                        <p>Scaling: {data.scaling}</p>
                      </div>
                    </div>
                  );
                })}
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        {/* Analysis Tab */}
        <TabsContent value="analysis" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Strengths & Weaknesses Analysis</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-6">
                {selectedAlgorithms.map(algoId => {
                  const analysis = {
                    ppo: {
                      strengths: ['Simple implementation', 'Stable training', 'Works with discrete/continuous actions'],
                      weaknesses: ['Sample inefficient', 'On-policy limitations', 'Requires tuning'],
                      bestFor: 'General purpose RL, robotics'
                    },
                    sac: {
                      strengths: ['Sample efficient', 'Automatic exploration', 'Stable convergence'],
                      weaknesses: ['Complex implementation', 'Continuous actions only', 'High memory usage'],
                      bestFor: 'Continuous control, robotics'
                    },
                    grpo: {
                      strengths: ['Group normalization', 'Multi-task capable', 'Fair optimization'],
                      weaknesses: ['Computational overhead', 'Requires grouping strategy', 'Limited testing'],
                      bestFor: 'Multi-task learning, diverse rewards'
                    },
                    dqn: {
                      strengths: ['Simple concept', 'Discrete actions', 'Well-studied'],
                      weaknesses: ['Overestimation bias', 'Discrete only', 'Unstable'],
                      bestFor: 'Discrete action spaces, games'
                    }
                  };
                  
                  const algo = algorithms.find(a => a.id === algoId);
                  const data = analysis[algoId as keyof typeof analysis];
                  
                  return algo && data ? (
                    <div key={algoId} className="space-y-3">
                      <h4 className="font-medium flex items-center gap-2">
                        <div 
                          className="w-3 h-3 rounded-full" 
                          style={{ backgroundColor: algo.color }}
                        />
                        {algo.name}
                      </h4>
                      <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-sm">
                        <div>
                          <p className="font-medium text-green-600 mb-1">Strengths</p>
                          <ul className="list-disc pl-4 space-y-1">
                            {data.strengths.map((s, i) => <li key={i}>{s}</li>)}
                          </ul>
                        </div>
                        <div>
                          <p className="font-medium text-red-600 mb-1">Weaknesses</p>
                          <ul className="list-disc pl-4 space-y-1">
                            {data.weaknesses.map((w, i) => <li key={i}>{w}</li>)}
                          </ul>
                        </div>
                        <div>
                          <p className="font-medium text-blue-600 mb-1">Best For</p>
                          <p>{data.bestFor}</p>
                        </div>
                      </div>
                    </div>
                  ) : null;
                })}
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>Recommendation Engine</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="p-4 bg-muted rounded-lg">
                <h4 className="font-medium mb-2">Based on your selected environment:</h4>
                <p className="text-sm mb-4">
                  {(() => {
                    const env = environments.find(e => e.id === selectedEnvironment);
                    if (!env) return 'Select an environment for recommendations';
                    
                    if (env.difficulty === 'easy') {
                      return 'For simple environments like CartPole, PPO provides the best balance of simplicity and performance.';
                    } else if (env.difficulty === 'medium') {
                      return 'For medium complexity tasks, SAC offers superior sample efficiency while maintaining stability.';
                    } else {
                      return 'For complex environments, consider GRPO for its advanced normalization or SAC for continuous control.';
                    }
                  })()}
                </p>
                
                <div className="grid grid-cols-2 gap-2 mt-4">
                  <Button size="sm" variant="outline">
                    <Settings className="w-4 h-4 mr-2" />
                    Tune Hyperparameters
                  </Button>
                  <Button size="sm" variant="outline">
                    <GitBranch className="w-4 h-4 mr-2" />
                    Run A/B Test
                  </Button>
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
}