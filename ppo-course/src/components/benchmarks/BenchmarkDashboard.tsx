'use client';

import React, { useState, useEffect, useRef } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Progress } from '@/components/ui/progress';
import { Checkbox } from '@/components/ui/checkbox';
import { 
  Play, Pause, RotateCcw, Download, Trophy, BarChart3, 
  Clock, Target, Zap, TrendingUp, AlertCircle, CheckCircle,
  Filter, Settings, FileText, Award, Cpu
} from 'lucide-react';
import { 
  LineChart, Line, BarChart, Bar, XAxis, YAxis, CartesianGrid, 
  Tooltip, Legend, ResponsiveContainer, ScatterPlot, Scatter,
  RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, Radar
} from 'recharts';

interface BenchmarkEnvironment {
  id: string;
  name: string;
  description: string;
  category: 'control' | 'navigation' | 'games' | 'robotics' | 'multi-agent';
  difficulty: 'easy' | 'medium' | 'hard' | 'expert';
  stateSize: number;
  actionSize: number;
  continuous: boolean;
  successThreshold: number;
  tags: string[];
}

interface BenchmarkResult {
  algorithmId: string;
  environmentId: string;
  episodeReward: number;
  episodeLength: number;
  trainingSteps: number;
  wallClockTime: number;
  sampleEfficiency: number;
  convergenceEpisode: number;
  finalSuccess: boolean;
  hyperparameters: Record<string, any>;
  metadata: {
    timestamp: Date;
    version: string;
    seed: number;
  };
}

interface BenchmarkProgress {
  algorithmId: string;
  environmentId: string;
  currentRun: number;
  totalRuns: number;
  currentEpisode: number;
  totalEpisodes: number;
  status: 'running' | 'completed' | 'failed' | 'pending';
  partialResults: BenchmarkResult[];
}

export function BenchmarkDashboard() {
  const [selectedAlgorithms, setSelectedAlgorithms] = useState<string[]>(['ppo', 'sac']);
  const [selectedEnvironments, setSelectedEnvironments] = useState<string[]>(['cartpole', 'lunar_lander']);
  const [isRunning, setIsRunning] = useState(false);
  const [currentBenchmarks, setCurrentBenchmarks] = useState<BenchmarkProgress[]>([]);
  const [completedResults, setCompletedResults] = useState<BenchmarkResult[]>([]);
  const [leaderboard, setLeaderboard] = useState<any[]>([]);
  const [analysisData, setAnalysisData] = useState<any>(null);
  const [numRuns, setNumRuns] = useState(5);
  const [maxEpisodes, setMaxEpisodes] = useState(1000);
  
  const algorithms = [
    { id: 'ppo', name: 'PPO', description: 'Proximal Policy Optimization' },
    { id: 'sac', name: 'SAC', description: 'Soft Actor-Critic' },
    { id: 'grpo', name: 'GRPO', description: 'Group Relative Policy Optimization' },
    { id: 'mappo', name: 'MAPPO', description: 'Multi-Agent PPO' },
    { id: 'dqn', name: 'DQN', description: 'Deep Q-Network' }
  ];

  const environments: BenchmarkEnvironment[] = [
    {
      id: 'cartpole',
      name: 'CartPole-v1',
      description: 'Balance a pole on a cart',
      category: 'control',
      difficulty: 'easy',
      stateSize: 4,
      actionSize: 2,
      continuous: false,
      successThreshold: 475,
      tags: ['discrete', 'classic']
    },
    {
      id: 'lunar_lander',
      name: 'LunarLander-v2',
      description: 'Land a spacecraft on the moon',
      category: 'navigation',
      difficulty: 'medium',
      stateSize: 8,
      actionSize: 4,
      continuous: false,
      successThreshold: 200,
      tags: ['discrete', 'physics']
    },
    {
      id: 'pendulum',
      name: 'Pendulum-v1',
      description: 'Swing up an inverted pendulum',
      category: 'control',
      difficulty: 'medium',
      stateSize: 3,
      actionSize: 1,
      continuous: true,
      successThreshold: -200,
      tags: ['continuous', 'classic']
    },
    {
      id: 'bipedal_walker',
      name: 'BipedalWalker-v3',
      description: 'Train a bipedal robot to walk',
      category: 'navigation',
      difficulty: 'hard',
      stateSize: 24,
      actionSize: 4,
      continuous: true,
      successThreshold: 300,
      tags: ['continuous', 'locomotion']
    },
    {
      id: 'humanoid',
      name: 'Humanoid-v3',
      description: 'Control a humanoid robot',
      category: 'robotics',
      difficulty: 'expert',
      stateSize: 376,
      actionSize: 17,
      continuous: true,
      successThreshold: 6000,
      tags: ['continuous', 'high-dimensional']
    }
  ];

  const difficultyColors = {
    easy: 'bg-green-100 text-green-800',
    medium: 'bg-blue-100 text-blue-800',
    hard: 'bg-orange-100 text-orange-800',
    expert: 'bg-red-100 text-red-800'
  };

  // Simulate benchmark execution
  useEffect(() => {
    if (!isRunning || currentBenchmarks.length === 0) return;

    const interval = setInterval(() => {
      setCurrentBenchmarks(prev => 
        prev.map(benchmark => {
          if (benchmark.status === 'completed' || benchmark.status === 'failed') {
            return benchmark;
          }

          const newBenchmark = { ...benchmark };
          
          // Simulate progress
          if (benchmark.status === 'pending') {
            newBenchmark.status = 'running';
          } else if (benchmark.status === 'running') {
            newBenchmark.currentEpisode = Math.min(
              benchmark.currentEpisode + Math.floor(Math.random() * 10) + 1,
              benchmark.totalEpisodes
            );
            
            // Check if current run is complete
            if (newBenchmark.currentEpisode >= benchmark.totalEpisodes) {
              // Add simulated result
              const result: BenchmarkResult = {
                algorithmId: benchmark.algorithmId,
                environmentId: benchmark.environmentId,
                episodeReward: generateMockReward(benchmark.algorithmId, benchmark.environmentId),
                episodeLength: Math.random() * 200 + 100,
                trainingSteps: Math.random() * 50000 + 10000,
                wallClockTime: Math.random() * 600 + 60,
                sampleEfficiency: Math.random() * 1000 + 100,
                convergenceEpisode: Math.floor(Math.random() * 500) + 100,
                finalSuccess: Math.random() > 0.3,
                hyperparameters: {},
                metadata: {
                  timestamp: new Date(),
                  version: '1.0.0',
                  seed: Math.floor(Math.random() * 10000)
                }
              };
              
              newBenchmark.partialResults.push(result);
              newBenchmark.currentRun++;
              newBenchmark.currentEpisode = 0;
              
              // Check if all runs complete
              if (newBenchmark.currentRun >= newBenchmark.totalRuns) {
                newBenchmark.status = 'completed';
                setCompletedResults(prev => [...prev, ...newBenchmark.partialResults]);
              }
            }
          }
          
          return newBenchmark;
        })
      );
    }, 200);

    return () => clearInterval(interval);
  }, [isRunning, currentBenchmarks]);

  // Update analysis when results change
  useEffect(() => {
    if (completedResults.length > 0) {
      updateAnalysis();
      updateLeaderboard();
    }
  }, [completedResults]);

  const generateMockReward = (algorithmId: string, environmentId: string): number => {
    const env = environments.find(e => e.id === environmentId)!;
    const baseReward = env.successThreshold;
    
    // Algorithm-specific modifiers
    const modifiers = {
      ppo: 0.8 + Math.random() * 0.3,
      sac: 0.85 + Math.random() * 0.25,
      grpo: 0.82 + Math.random() * 0.28,
      mappo: 0.87 + Math.random() * 0.23,
      dqn: 0.75 + Math.random() * 0.35
    };
    
    const modifier = modifiers[algorithmId as keyof typeof modifiers] || 0.8;
    return baseReward * modifier + (Math.random() - 0.5) * baseReward * 0.2;
  };

  const startBenchmarks = () => {
    const benchmarks: BenchmarkProgress[] = [];
    
    for (const algId of selectedAlgorithms) {
      for (const envId of selectedEnvironments) {
        benchmarks.push({
          algorithmId: algId,
          environmentId: envId,
          currentRun: 0,
          totalRuns: numRuns,
          currentEpisode: 0,
          totalEpisodes: maxEpisodes,
          status: 'pending',
          partialResults: []
        });
      }
    }
    
    setCurrentBenchmarks(benchmarks);
    setIsRunning(true);
  };

  const stopBenchmarks = () => {
    setIsRunning(false);
    setCurrentBenchmarks(prev => 
      prev.map(b => ({ 
        ...b, 
        status: b.status === 'running' ? 'failed' : b.status 
      }))
    );
  };

  const resetBenchmarks = () => {
    setIsRunning(false);
    setCurrentBenchmarks([]);
    setCompletedResults([]);
    setLeaderboard([]);
    setAnalysisData(null);
  };

  const updateAnalysis = () => {
    const environmentAnalysis = new Map();
    
    for (const envId of selectedEnvironments) {
      const envResults = completedResults.filter(r => r.environmentId === envId);
      const algorithmStats = new Map();
      
      for (const algId of selectedAlgorithms) {
        const algResults = envResults.filter(r => r.algorithmId === algId);
        if (algResults.length > 0) {
          const rewards = algResults.map(r => r.episodeReward);
          const mean = rewards.reduce((a, b) => a + b) / rewards.length;
          const std = Math.sqrt(
            rewards.reduce((acc, val) => acc + Math.pow(val - mean, 2), 0) / rewards.length
          );
          
          algorithmStats.set(algId, {
            mean,
            std,
            successRate: algResults.filter(r => r.finalSuccess).length / algResults.length,
            avgConvergence: algResults.reduce((acc, r) => acc + r.convergenceEpisode, 0) / algResults.length,
            avgSampleEfficiency: algResults.reduce((acc, r) => acc + r.sampleEfficiency, 0) / algResults.length
          });
        }
      }
      
      environmentAnalysis.set(envId, algorithmStats);
    }
    
    setAnalysisData(environmentAnalysis);
  };

  const updateLeaderboard = () => {
    const algorithmScores = new Map();
    
    for (const result of completedResults) {
      if (!algorithmScores.has(result.algorithmId)) {
        algorithmScores.set(result.algorithmId, {
          scores: [],
          environments: new Set(),
          totalRuns: 0,
          successCount: 0
        });
      }
      
      const entry = algorithmScores.get(result.algorithmId);
      entry.scores.push(result.episodeReward);
      entry.environments.add(result.environmentId);
      entry.totalRuns++;
      if (result.finalSuccess) entry.successCount++;
    }
    
    const leaderboardData = Array.from(algorithmScores.entries())
      .map(([algId, data]) => ({
        algorithmId: algId,
        score: data.scores.reduce((a: number, b: number) => a + b) / data.scores.length,
        environmentCount: data.environments.size,
        successRate: data.successCount / data.totalRuns,
        totalRuns: data.totalRuns
      }))
      .sort((a, b) => b.score - a.score)
      .map((item, index) => ({ ...item, rank: index + 1 }));
    
    setLeaderboard(leaderboardData);
  };

  const exportResults = () => {
    const exportData = {
      algorithms: selectedAlgorithms,
      environments: selectedEnvironments,
      results: completedResults,
      analysis: analysisData ? Object.fromEntries(analysisData) : {},
      leaderboard,
      timestamp: new Date().toISOString()
    };
    
    const blob = new Blob([JSON.stringify(exportData, null, 2)], { 
      type: 'application/json' 
    });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `benchmark_results_${Date.now()}.json`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const getRunningProgress = () => {
    if (currentBenchmarks.length === 0) return 0;
    
    const totalProgress = currentBenchmarks.reduce((acc, benchmark) => {
      const runProgress = benchmark.currentRun / benchmark.totalRuns;
      const episodeProgress = benchmark.currentEpisode / benchmark.totalEpisodes;
      return acc + (runProgress + episodeProgress / benchmark.totalRuns);
    }, 0);
    
    return (totalProgress / currentBenchmarks.length) * 100;
  };

  const toggleAlgorithm = (algId: string) => {
    setSelectedAlgorithms(prev =>
      prev.includes(algId)
        ? prev.filter(id => id !== algId)
        : [...prev, algId]
    );
  };

  const toggleEnvironment = (envId: string) => {
    setSelectedEnvironments(prev =>
      prev.includes(envId)
        ? prev.filter(id => id !== envId)
        : [...prev, envId]
    );
  };

  return (
    <div className="space-y-6">
      {/* Control Panel */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center justify-between">
            <span className="flex items-center gap-2">
              <Trophy className="w-5 h-5" />
              RL Benchmark Suite
            </span>
            <div className="flex items-center gap-2">
              {!isRunning ? (
                <Button 
                  onClick={startBenchmarks}
                  disabled={selectedAlgorithms.length === 0 || selectedEnvironments.length === 0}
                >
                  <Play className="w-4 h-4 mr-2" />
                  Start Benchmarks
                </Button>
              ) : (
                <Button variant="secondary" onClick={stopBenchmarks}>
                  <Pause className="w-4 h-4 mr-2" />
                  Stop
                </Button>
              )}
              <Button variant="outline" onClick={resetBenchmarks}>
                <RotateCcw className="w-4 h-4 mr-2" />
                Reset
              </Button>
              <Button variant="outline" onClick={exportResults}>
                <Download className="w-4 h-4 mr-2" />
                Export
              </Button>
            </div>
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-6">
          {/* Algorithm Selection */}
          <div>
            <h3 className="text-lg font-medium mb-3">Select Algorithms</h3>
            <div className="grid grid-cols-2 md:grid-cols-5 gap-3">
              {algorithms.map(alg => (
                <div key={alg.id} className="flex items-center space-x-2">
                  <Checkbox
                    id={alg.id}
                    checked={selectedAlgorithms.includes(alg.id)}
                    onCheckedChange={() => toggleAlgorithm(alg.id)}
                    disabled={isRunning}
                  />
                  <label
                    htmlFor={alg.id}
                    className="text-sm font-medium leading-none peer-disabled:cursor-not-allowed peer-disabled:opacity-70 cursor-pointer"
                  >
                    {alg.name}
                  </label>
                </div>
              ))}
            </div>
          </div>
          
          {/* Environment Selection */}
          <div>
            <h3 className="text-lg font-medium mb-3">Select Environments</h3>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
              {environments.map(env => (
                <div 
                  key={env.id} 
                  className={`p-3 border rounded-lg cursor-pointer transition-all ${
                    selectedEnvironments.includes(env.id) 
                      ? 'border-primary bg-primary/5' 
                      : 'border-border hover:border-primary/50'
                  } ${isRunning ? 'opacity-50 cursor-not-allowed' : ''}`}
                  onClick={() => !isRunning && toggleEnvironment(env.id)}
                >
                  <div className="flex items-center justify-between mb-2">
                    <h4 className="font-medium">{env.name}</h4>
                    <Badge 
                      variant="secondary" 
                      className={difficultyColors[env.difficulty]}
                    >
                      {env.difficulty}
                    </Badge>
                  </div>
                  <p className="text-sm text-muted-foreground mb-2">
                    {env.description}
                  </p>
                  <div className="flex gap-1 flex-wrap">
                    {env.tags.map(tag => (
                      <Badge key={tag} variant="outline" className="text-xs">
                        {tag}
                      </Badge>
                    ))}
                  </div>
                </div>
              ))}
            </div>
          </div>
          
          {/* Configuration */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div>
              <label className="text-sm font-medium mb-2 block">
                Runs per Algorithm: {numRuns}
              </label>
              <input
                type="range"
                min="1"
                max="10"
                value={numRuns}
                onChange={(e) => setNumRuns(parseInt(e.target.value))}
                disabled={isRunning}
                className="w-full"
              />
            </div>
            <div>
              <label className="text-sm font-medium mb-2 block">
                Max Episodes: {maxEpisodes}
              </label>
              <input
                type="range"
                min="100"
                max="2000"
                step="100"
                value={maxEpisodes}
                onChange={(e) => setMaxEpisodes(parseInt(e.target.value))}
                disabled={isRunning}
                className="w-full"
              />
            </div>
            <div className="col-span-2">
              <div className="text-sm text-muted-foreground">
                Total benchmarks: {selectedAlgorithms.length * selectedEnvironments.length} 
                ({selectedAlgorithms.length * selectedEnvironments.length * numRuns} runs)
              </div>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Progress Panel */}
      {currentBenchmarks.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Clock className="w-5 h-5" />
              Benchmark Progress
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              <div>
                <div className="flex justify-between text-sm mb-2">
                  <span>Overall Progress</span>
                  <span>{getRunningProgress().toFixed(1)}%</span>
                </div>
                <Progress value={getRunningProgress()} className="h-2" />
              </div>
              
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                {currentBenchmarks.map((benchmark, i) => (
                  <div key={i} className="p-3 border rounded-lg">
                    <div className="flex items-center justify-between mb-2">
                      <span className="text-sm font-medium">
                        {benchmark.algorithmId.toUpperCase()} on {environments.find(e => e.id === benchmark.environmentId)?.name}
                      </span>
                      <Badge 
                        variant={
                          benchmark.status === 'completed' ? 'default' :
                          benchmark.status === 'running' ? 'secondary' :
                          benchmark.status === 'failed' ? 'destructive' : 'outline'
                        }
                        className="text-xs"
                      >
                        {benchmark.status === 'completed' && <CheckCircle className="w-3 h-3 mr-1" />}
                        {benchmark.status === 'failed' && <AlertCircle className="w-3 h-3 mr-1" />}
                        {benchmark.status === 'running' && <Cpu className="w-3 h-3 mr-1" />}
                        {benchmark.status}
                      </Badge>
                    </div>
                    
                    <div className="space-y-2 text-sm">
                      <div className="flex justify-between">
                        <span>Run:</span>
                        <span>{benchmark.currentRun + 1}/{benchmark.totalRuns}</span>
                      </div>
                      <div className="flex justify-between">
                        <span>Episode:</span>
                        <span>{benchmark.currentEpisode}/{benchmark.totalEpisodes}</span>
                      </div>
                      <Progress 
                        value={(benchmark.currentEpisode / benchmark.totalEpisodes) * 100} 
                        className="h-1" 
                      />
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Results Dashboard */}
      <Tabs defaultValue="leaderboard" className="space-y-4">
        <TabsList className="grid w-full grid-cols-4">
          <TabsTrigger value="leaderboard">Leaderboard</TabsTrigger>
          <TabsTrigger value="analysis">Analysis</TabsTrigger>
          <TabsTrigger value="environments">Environments</TabsTrigger>
          <TabsTrigger value="details">Details</TabsTrigger>
        </TabsList>

        {/* Leaderboard Tab */}
        <TabsContent value="leaderboard" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Award className="w-5 h-5" />
                Algorithm Leaderboard
              </CardTitle>
            </CardHeader>
            <CardContent>
              {leaderboard.length === 0 ? (
                <div className="text-center py-8 text-muted-foreground">
                  No results yet. Run benchmarks to see rankings.
                </div>
              ) : (
                <div className="space-y-3">
                  {leaderboard.map((entry, index) => (
                    <div 
                      key={entry.algorithmId}
                      className={`flex items-center justify-between p-4 rounded-lg border ${
                        index === 0 ? 'bg-yellow-50 border-yellow-200' :
                        index === 1 ? 'bg-gray-50 border-gray-200' :
                        index === 2 ? 'bg-orange-50 border-orange-200' :
                        'bg-muted/30'
                      }`}
                    >
                      <div className="flex items-center gap-4">
                        <div className={`w-8 h-8 rounded-full flex items-center justify-center font-bold ${
                          index === 0 ? 'bg-yellow-500 text-white' :
                          index === 1 ? 'bg-gray-500 text-white' :
                          index === 2 ? 'bg-orange-500 text-white' :
                          'bg-muted text-muted-foreground'
                        }`}>
                          {entry.rank}
                        </div>
                        <div>
                          <h4 className="font-medium">{entry.algorithmId.toUpperCase()}</h4>
                          <p className="text-sm text-muted-foreground">
                            {algorithms.find(a => a.id === entry.algorithmId)?.description}
                          </p>
                        </div>
                      </div>
                      
                      <div className="text-right">
                        <div className="text-lg font-bold">{entry.score.toFixed(1)}</div>
                        <div className="text-sm text-muted-foreground">
                          {entry.environmentCount} envs • {(entry.successRate * 100).toFixed(0)}% success
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </CardContent>
          </Card>
        </TabsContent>

        {/* Analysis Tab */}
        <TabsContent value="analysis" className="space-y-4">
          {analysisData && Array.from(analysisData.entries()).map(([envId, algStats]) => {
            const env = environments.find(e => e.id === envId)!;
            const chartData = Array.from(algStats.entries()).map(([algId, stats]: [string, any]) => ({
              algorithm: algId.toUpperCase(),
              mean: stats.mean,
              std: stats.std,
              successRate: stats.successRate * 100,
              convergence: stats.avgConvergence,
              sampleEfficiency: stats.avgSampleEfficiency
            }));

            return (
              <Card key={envId}>
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <BarChart3 className="w-5 h-5" />
                    {env.name} Analysis
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                    <div>
                      <h4 className="font-medium mb-3">Performance Comparison</h4>
                      <ResponsiveContainer width="100%" height={250}>
                        <BarChart data={chartData}>
                          <CartesianGrid strokeDasharray="3 3" />
                          <XAxis dataKey="algorithm" />
                          <YAxis />
                          <Tooltip />
                          <Bar dataKey="mean" fill="#3b82f6" />
                        </BarChart>
                      </ResponsiveContainer>
                    </div>
                    
                    <div>
                      <h4 className="font-medium mb-3">Success Rates</h4>
                      <ResponsiveContainer width="100%" height={250}>
                        <BarChart data={chartData}>
                          <CartesianGrid strokeDasharray="3 3" />
                          <XAxis dataKey="algorithm" />
                          <YAxis />
                          <Tooltip formatter={(value) => [`${value}%`, 'Success Rate']} />
                          <Bar dataKey="successRate" fill="#10b981" />
                        </BarChart>
                      </ResponsiveContainer>
                    </div>
                  </div>
                  
                  <div className="mt-6">
                    <h4 className="font-medium mb-3">Detailed Statistics</h4>
                    <div className="overflow-x-auto">
                      <table className="w-full text-sm">
                        <thead>
                          <tr className="border-b">
                            <th className="text-left py-2">Algorithm</th>
                            <th className="text-left py-2">Mean Reward</th>
                            <th className="text-left py-2">Std Dev</th>
                            <th className="text-left py-2">Success Rate</th>
                            <th className="text-left py-2">Avg Convergence</th>
                            <th className="text-left py-2">Sample Efficiency</th>
                          </tr>
                        </thead>
                        <tbody>
                          {chartData.map(row => (
                            <tr key={row.algorithm} className="border-b">
                              <td className="py-2 font-medium">{row.algorithm}</td>
                              <td className="py-2">{row.mean.toFixed(1)} ± {row.std.toFixed(1)}</td>
                              <td className="py-2">{row.std.toFixed(2)}</td>
                              <td className="py-2">{row.successRate.toFixed(1)}%</td>
                              <td className="py-2">{row.convergence.toFixed(0)} eps</td>
                              <td className="py-2">{row.sampleEfficiency.toFixed(0)}</td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </div>
                </CardContent>
              </Card>
            );
          })}
        </TabsContent>

        {/* Environments Tab */}
        <TabsContent value="environments" className="space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {environments.map(env => {
              const envResults = completedResults.filter(r => r.environmentId === env.id);
              const avgReward = envResults.length > 0 
                ? envResults.reduce((acc, r) => acc + r.episodeReward, 0) / envResults.length 
                : 0;
              const successRate = envResults.length > 0
                ? envResults.filter(r => r.finalSuccess).length / envResults.length
                : 0;

              return (
                <Card key={env.id}>
                  <CardHeader className="pb-3">
                    <div className="flex items-center justify-between">
                      <CardTitle className="text-lg">{env.name}</CardTitle>
                      <Badge 
                        variant="secondary" 
                        className={difficultyColors[env.difficulty]}
                      >
                        {env.difficulty}
                      </Badge>
                    </div>
                  </CardHeader>
                  <CardContent className="space-y-3">
                    <p className="text-sm text-muted-foreground">
                      {env.description}
                    </p>
                    
                    <div className="grid grid-cols-2 gap-3 text-sm">
                      <div>
                        <span className="text-muted-foreground">State Size:</span>
                        <div className="font-medium">{env.stateSize}</div>
                      </div>
                      <div>
                        <span className="text-muted-foreground">Action Size:</span>
                        <div className="font-medium">{env.actionSize}</div>
                      </div>
                      <div>
                        <span className="text-muted-foreground">Type:</span>
                        <div className="font-medium">{env.continuous ? 'Continuous' : 'Discrete'}</div>
                      </div>
                      <div>
                        <span className="text-muted-foreground">Success Threshold:</span>
                        <div className="font-medium">{env.successThreshold}</div>
                      </div>
                    </div>
                    
                    {envResults.length > 0 && (
                      <div className="pt-3 border-t">
                        <div className="grid grid-cols-2 gap-3 text-sm">
                          <div>
                            <span className="text-muted-foreground">Avg Reward:</span>
                            <div className="font-medium">{avgReward.toFixed(1)}</div>
                          </div>
                          <div>
                            <span className="text-muted-foreground">Success Rate:</span>
                            <div className="font-medium">{(successRate * 100).toFixed(0)}%</div>
                          </div>
                        </div>
                      </div>
                    )}
                    
                    <div className="flex gap-1 flex-wrap pt-2">
                      {env.tags.map(tag => (
                        <Badge key={tag} variant="outline" className="text-xs">
                          {tag}
                        </Badge>
                      ))}
                    </div>
                  </CardContent>
                </Card>
              );
            })}
          </div>
        </TabsContent>

        {/* Details Tab */}
        <TabsContent value="details" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <FileText className="w-5 h-5" />
                Detailed Results
              </CardTitle>
            </CardHeader>
            <CardContent>
              {completedResults.length === 0 ? (
                <div className="text-center py-8 text-muted-foreground">
                  No detailed results available yet.
                </div>
              ) : (
                <div className="space-y-4">
                  <div className="text-sm text-muted-foreground mb-4">
                    Showing {completedResults.length} completed benchmark runs
                  </div>
                  
                  <div className="overflow-x-auto">
                    <table className="w-full text-sm">
                      <thead>
                        <tr className="border-b">
                          <th className="text-left py-2">Algorithm</th>
                          <th className="text-left py-2">Environment</th>
                          <th className="text-left py-2">Episode Reward</th>
                          <th className="text-left py-2">Training Steps</th>
                          <th className="text-left py-2">Wall Clock Time</th>
                          <th className="text-left py-2">Convergence</th>
                          <th className="text-left py-2">Success</th>
                        </tr>
                      </thead>
                      <tbody>
                        {completedResults.slice(0, 50).map((result, index) => ( // Show only first 50 for performance
                          <tr key={index} className="border-b">
                            <td className="py-2 font-medium">{result.algorithmId.toUpperCase()}</td>
                            <td className="py-2">
                              {environments.find(e => e.id === result.environmentId)?.name}
                            </td>
                            <td className="py-2">{result.episodeReward.toFixed(1)}</td>
                            <td className="py-2">{result.trainingSteps.toLocaleString()}</td>
                            <td className="py-2">{(result.wallClockTime / 1000).toFixed(1)}s</td>
                            <td className="py-2">{result.convergenceEpisode} eps</td>
                            <td className="py-2">
                              {result.finalSuccess ? (
                                <CheckCircle className="w-4 h-4 text-green-500" />
                              ) : (
                                <AlertCircle className="w-4 h-4 text-red-500" />
                              )}
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                  
                  {completedResults.length > 50 && (
                    <div className="text-center text-sm text-muted-foreground">
                      Showing first 50 results. Export data to see all results.
                    </div>
                  )}
                </div>
              )}
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
}