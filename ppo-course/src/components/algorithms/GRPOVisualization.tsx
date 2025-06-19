'use client';

import React, { useState, useEffect, useMemo } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  BarElement,
  RadialLinearScale,
  ArcElement,
  Filler,
} from 'chart.js';
import { Line, Bar, Radar, Doughnut } from 'react-chartjs-2';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Slider } from '@/components/ui/slider';
import { RadioGroup, RadioGroupItem } from '@/components/ui/radio-group';
import { Label } from '@/components/ui/label';
import { Switch } from '@/components/ui/switch';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Play, Pause, RotateCcw, Shuffle, Users, TrendingUp, Layers, BarChart3 } from 'lucide-react';

// Register Chart.js components
ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  BarElement,
  RadialLinearScale,
  ArcElement,
  Filler
);

interface GroupData {
  id: string;
  size: number;
  avgReward: number;
  avgLength: number;
  characteristics: {
    difficulty: number;
    exploration: number;
    stability: number;
    performance: number;
  };
  advantages: number[];
  weight: number;
}

export default function GRPOVisualization() {
  const [isRunning, setIsRunning] = useState(false);
  const [iteration, setIteration] = useState(0);
  const [groupingStrategy, setGroupingStrategy] = useState<'auto' | 'task' | 'difficulty' | 'length'>('auto');
  const [numGroups, setNumGroups] = useState(4);
  const [weightingMethod, setWeightingMethod] = useState<'balanced' | 'performance' | 'adaptive'>('balanced');
  const [comparisonMode, setComparisonMode] = useState(false);
  const [groupData, setGroupData] = useState<GroupData[]>([]);
  const [ppoAdvantages, setPpoAdvantages] = useState<number[]>([]);
  const [grpoAdvantages, setGrpoAdvantages] = useState<number[]>([]);

  // Generate initial group data
  useEffect(() => {
    generateGroupData();
  }, [numGroups, groupingStrategy]);

  const generateGroupData = () => {
    const groups: GroupData[] = [];
    
    for (let i = 0; i < numGroups; i++) {
      const size = Math.floor(Math.random() * 20) + 10;
      const baseReward = groupingStrategy === 'difficulty' ? (i + 1) * 25 : Math.random() * 100;
      const avgLength = groupingStrategy === 'length' ? (i + 1) * 30 : Math.random() * 100 + 50;
      
      // Generate advantages for this group
      const advantages = Array.from({ length: size }, () => 
        (Math.random() - 0.5) * 10 + (baseReward - 50) / 10
      );
      
      groups.push({
        id: `Group ${i + 1}`,
        size,
        avgReward: baseReward + (Math.random() - 0.5) * 20,
        avgLength,
        characteristics: {
          difficulty: baseReward / 100,
          exploration: Math.random(),
          stability: 1 - Math.abs(Math.random() - 0.5),
          performance: baseReward / 100 + Math.random() * 0.2,
        },
        advantages,
        weight: 1 / numGroups, // Initial balanced weight
      });
    }
    
    setGroupData(groups);
    
    // Generate comparison data
    const allAdvantages = groups.flatMap(g => g.advantages);
    setPpoAdvantages(normalizeGlobally(allAdvantages));
    setGrpoAdvantages(groups.flatMap(g => normalizeLocally(g.advantages)));
  };

  const normalizeGlobally = (advantages: number[]): number[] => {
    const mean = advantages.reduce((a, b) => a + b, 0) / advantages.length;
    const std = Math.sqrt(
      advantages.reduce((sum, x) => sum + Math.pow(x - mean, 2), 0) / advantages.length
    );
    return advantages.map(x => (x - mean) / (std || 1));
  };

  const normalizeLocally = (advantages: number[]): number[] => {
    const mean = advantages.reduce((a, b) => a + b, 0) / advantages.length;
    const std = Math.sqrt(
      advantages.reduce((sum, x) => sum + Math.pow(x - mean, 2), 0) / advantages.length
    );
    return advantages.map(x => (x - mean) / (std || 1));
  };

  // Update weights based on method
  useEffect(() => {
    if (!isRunning) return;
    
    const timer = setInterval(() => {
      setIteration(prev => prev + 1);
      updateGroupWeights();
    }, 1000);
    
    return () => clearInterval(timer);
  }, [isRunning, weightingMethod]);

  const updateGroupWeights = () => {
    setGroupData(prevGroups => {
      const newGroups = [...prevGroups];
      
      switch (weightingMethod) {
        case 'balanced':
          // Equal weights
          newGroups.forEach(g => g.weight = 1 / numGroups);
          break;
          
        case 'performance':
          // Weight by performance
          const totalPerf = newGroups.reduce((sum, g) => sum + g.avgReward, 0);
          newGroups.forEach(g => g.weight = g.avgReward / totalPerf);
          break;
          
        case 'adaptive':
          // Simulate adaptive weighting based on improvement
          newGroups.forEach(g => {
            const improvement = Math.random() * 0.2 - 0.1;
            g.avgReward += g.avgReward * improvement;
            g.weight = Math.max(0.1, Math.min(0.5, g.weight + improvement));
          });
          
          // Normalize weights
          const totalWeight = newGroups.reduce((sum, g) => sum + g.weight, 0);
          newGroups.forEach(g => g.weight /= totalWeight);
          break;
      }
      
      return newGroups;
    });
  };

  // Chart configurations
  const groupDistributionData = {
    labels: groupData.map(g => g.id),
    datasets: [
      {
        label: 'Group Sizes',
        data: groupData.map(g => g.size),
        backgroundColor: 'rgba(59, 130, 246, 0.8)',
      },
    ],
  };

  const groupCharacteristicsData = {
    labels: ['Difficulty', 'Exploration', 'Stability', 'Performance'],
    datasets: groupData.map((g, i) => ({
      label: g.id,
      data: [
        g.characteristics.difficulty,
        g.characteristics.exploration,
        g.characteristics.stability,
        g.characteristics.performance,
      ],
      borderColor: `hsl(${i * 360 / numGroups}, 70%, 50%)`,
      backgroundColor: `hsla(${i * 360 / numGroups}, 70%, 50%, 0.2)`,
    })),
  };

  const advantageComparisonData = {
    labels: Array.from({ length: 50 }, (_, i) => i),
    datasets: [
      {
        label: 'PPO (Global Normalization)',
        data: ppoAdvantages.slice(0, 50),
        borderColor: 'rgb(239, 68, 68)',
        backgroundColor: 'rgba(239, 68, 68, 0.1)',
        tension: 0.3,
      },
      {
        label: 'GRPO (Group Normalization)',
        data: grpoAdvantages.slice(0, 50),
        borderColor: 'rgb(34, 197, 94)',
        backgroundColor: 'rgba(34, 197, 94, 0.1)',
        tension: 0.3,
      },
    ],
  };

  const groupWeightsData = {
    labels: groupData.map(g => g.id),
    datasets: [{
      data: groupData.map(g => g.weight),
      backgroundColor: groupData.map((_, i) => `hsl(${i * 360 / numGroups}, 70%, 50%)`),
    }],
  };

  const performanceOverTimeData = {
    labels: Array.from({ length: Math.max(iteration, 1) }, (_, i) => i),
    datasets: groupData.map((g, i) => ({
      label: g.id,
      data: Array.from({ length: Math.max(iteration, 1) }, () => 
        g.avgReward + (Math.random() - 0.5) * 10
      ),
      borderColor: `hsl(${i * 360 / numGroups}, 70%, 50%)`,
      backgroundColor: `hsla(${i * 360 / numGroups}, 70%, 50%, 0.1)`,
      tension: 0.3,
    })),
  };

  return (
    <div className="w-full space-y-6">
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Users className="w-6 h-6" />
            GRPO: Group Relative Policy Optimization
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-6">
          {/* Controls */}
          <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
            <div className="space-y-2">
              <Label>Grouping Strategy</Label>
              <Select value={groupingStrategy} onValueChange={(v: any) => setGroupingStrategy(v)}>
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="auto">Auto (Clustering)</SelectItem>
                  <SelectItem value="task">Task-Based</SelectItem>
                  <SelectItem value="difficulty">Difficulty-Based</SelectItem>
                  <SelectItem value="length">Length-Based</SelectItem>
                </SelectContent>
              </Select>
            </div>
            
            <div className="space-y-2">
              <Label>Number of Groups: {numGroups}</Label>
              <Slider
                value={[numGroups]}
                onValueChange={([v]) => setNumGroups(v)}
                min={2}
                max={8}
                step={1}
              />
            </div>
            
            <div className="space-y-2">
              <Label>Weighting Method</Label>
              <RadioGroup value={weightingMethod} onValueChange={(v: any) => setWeightingMethod(v)}>
                <div className="flex items-center space-x-2">
                  <RadioGroupItem value="balanced" id="balanced" />
                  <Label htmlFor="balanced">Balanced</Label>
                </div>
                <div className="flex items-center space-x-2">
                  <RadioGroupItem value="performance" id="performance" />
                  <Label htmlFor="performance">Performance</Label>
                </div>
                <div className="flex items-center space-x-2">
                  <RadioGroupItem value="adaptive" id="adaptive" />
                  <Label htmlFor="adaptive">Adaptive</Label>
                </div>
              </RadioGroup>
            </div>
            
            <div className="space-y-2">
              <Label>Comparison Mode</Label>
              <div className="flex items-center space-x-2">
                <Switch
                  checked={comparisonMode}
                  onCheckedChange={setComparisonMode}
                />
                <Label>PPO vs GRPO</Label>
              </div>
            </div>
          </div>

          {/* Control Buttons */}
          <div className="flex gap-2">
            <Button
              onClick={() => setIsRunning(!isRunning)}
              variant={isRunning ? "secondary" : "default"}
            >
              {isRunning ? <Pause className="w-4 h-4 mr-2" /> : <Play className="w-4 h-4 mr-2" />}
              {isRunning ? 'Pause' : 'Start'}
            </Button>
            <Button
              onClick={() => {
                setIteration(0);
                generateGroupData();
              }}
              variant="outline"
            >
              <RotateCcw className="w-4 h-4 mr-2" />
              Reset
            </Button>
            <Button
              onClick={generateGroupData}
              variant="outline"
            >
              <Shuffle className="w-4 h-4 mr-2" />
              Regenerate
            </Button>
          </div>

          {/* Visualization Tabs */}
          <Tabs defaultValue="distribution" className="w-full">
            <TabsList className="grid w-full grid-cols-4">
              <TabsTrigger value="distribution">
                <BarChart3 className="w-4 h-4 mr-2" />
                Distribution
              </TabsTrigger>
              <TabsTrigger value="characteristics">
                <Layers className="w-4 h-4 mr-2" />
                Characteristics
              </TabsTrigger>
              <TabsTrigger value="advantages">
                <TrendingUp className="w-4 h-4 mr-2" />
                Advantages
              </TabsTrigger>
              <TabsTrigger value="dynamics">
                <Users className="w-4 h-4 mr-2" />
                Dynamics
              </TabsTrigger>
            </TabsList>

            <TabsContent value="distribution" className="space-y-4">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <Card>
                  <CardHeader>
                    <CardTitle className="text-sm">Group Sizes</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <Bar data={groupDistributionData} options={{
                      responsive: true,
                      plugins: {
                        legend: { display: false },
                      },
                    }} />
                  </CardContent>
                </Card>
                
                <Card>
                  <CardHeader>
                    <CardTitle className="text-sm">Group Weights</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <Doughnut data={groupWeightsData} options={{
                      responsive: true,
                      plugins: {
                        legend: { position: 'right' as const },
                      },
                    }} />
                  </CardContent>
                </Card>
              </div>
            </TabsContent>

            <TabsContent value="characteristics" className="space-y-4">
              <Card>
                <CardHeader>
                  <CardTitle className="text-sm">Group Characteristics</CardTitle>
                </CardHeader>
                <CardContent>
                  <Radar data={groupCharacteristicsData} options={{
                    responsive: true,
                    scales: {
                      r: {
                        beginAtZero: true,
                        max: 1,
                      },
                    },
                  }} />
                </CardContent>
              </Card>
            </TabsContent>

            <TabsContent value="advantages" className="space-y-4">
              <Card>
                <CardHeader>
                  <CardTitle className="text-sm">
                    {comparisonMode ? 'PPO vs GRPO Advantage Normalization' : 'Group Advantages'}
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  {comparisonMode ? (
                    <Line data={advantageComparisonData} options={{
                      responsive: true,
                      plugins: {
                        legend: { position: 'top' as const },
                      },
                      scales: {
                        y: {
                          title: {
                            display: true,
                            text: 'Normalized Advantage',
                          },
                        },
                      },
                    }} />
                  ) : (
                    <div className="grid grid-cols-2 gap-4">
                      {groupData.map((group, idx) => (
                        <div key={group.id} className="p-4 border rounded-lg">
                          <h4 className="font-semibold mb-2">{group.id}</h4>
                          <div className="space-y-1 text-sm">
                            <p>Mean: {(group.advantages.reduce((a, b) => a + b, 0) / group.advantages.length).toFixed(3)}</p>
                            <p>Std: {Math.sqrt(
                              group.advantages.reduce((sum, x) => {
                                const mean = group.advantages.reduce((a, b) => a + b, 0) / group.advantages.length;
                                return sum + Math.pow(x - mean, 2);
                              }, 0) / group.advantages.length
                            ).toFixed(3)}</p>
                            <p>Weight: {(group.weight * 100).toFixed(1)}%</p>
                          </div>
                        </div>
                      ))}
                    </div>
                  )}
                </CardContent>
              </Card>
            </TabsContent>

            <TabsContent value="dynamics" className="space-y-4">
              <Card>
                <CardHeader>
                  <CardTitle className="text-sm">Performance Over Time</CardTitle>
                </CardHeader>
                <CardContent>
                  <Line data={performanceOverTimeData} options={{
                    responsive: true,
                    plugins: {
                      legend: { position: 'top' as const },
                    },
                    scales: {
                      y: {
                        title: {
                          display: true,
                          text: 'Average Reward',
                        },
                      },
                    },
                  }} />
                </CardContent>
              </Card>
              
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                <Card>
                  <CardHeader>
                    <CardTitle className="text-sm">Training Statistics</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="space-y-2 text-sm">
                      <p>Iteration: {iteration}</p>
                      <p>Active Groups: {numGroups}</p>
                      <p>Strategy: {groupingStrategy}</p>
                      <p>Weighting: {weightingMethod}</p>
                    </div>
                  </CardContent>
                </Card>
                
                <Card>
                  <CardHeader>
                    <CardTitle className="text-sm">Best Performing Group</CardTitle>
                  </CardHeader>
                  <CardContent>
                    {groupData.length > 0 && (
                      <div className="space-y-2 text-sm">
                        <p className="font-semibold">
                          {groupData.reduce((best, g) => g.avgReward > best.avgReward ? g : best).id}
                        </p>
                        <p>Avg Reward: {groupData.reduce((best, g) => g.avgReward > best.avgReward ? g : best).avgReward.toFixed(2)}</p>
                        <p>Weight: {(groupData.reduce((best, g) => g.avgReward > best.avgReward ? g : best).weight * 100).toFixed(1)}%</p>
                      </div>
                    )}
                  </CardContent>
                </Card>
                
                <Card>
                  <CardHeader>
                    <CardTitle className="text-sm">Weight Distribution</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="space-y-1">
                      {groupData.map(group => (
                        <div key={group.id} className="flex items-center gap-2">
                          <span className="text-xs w-16">{group.id}:</span>
                          <div className="flex-1 bg-gray-200 rounded-full h-2">
                            <motion.div
                              className="bg-blue-500 h-2 rounded-full"
                              animate={{ width: `${group.weight * 100}%` }}
                              transition={{ duration: 0.5 }}
                            />
                          </div>
                          <span className="text-xs w-12 text-right">{(group.weight * 100).toFixed(1)}%</span>
                        </div>
                      ))}
                    </div>
                  </CardContent>
                </Card>
              </div>
            </TabsContent>
          </Tabs>

          {/* Algorithm Explanation */}
          <Card>
            <CardHeader>
              <CardTitle className="text-sm">How GRPO Works</CardTitle>
            </CardHeader>
            <CardContent className="space-y-2 text-sm">
              <p>
                <strong>Group Formation:</strong> Trajectories are grouped based on the selected strategy
                (clustering, task-based, difficulty, or length).
              </p>
              <p>
                <strong>Group Normalization:</strong> Advantages are normalized within each group rather
                than globally, preventing high-reward tasks from dominating gradients.
              </p>
              <p>
                <strong>Weighted Updates:</strong> Policy updates are weighted based on the selected method,
                ensuring balanced learning across all groups.
              </p>
              <p className="text-muted-foreground">
                GRPO is particularly effective in multi-task settings or when dealing with diverse
                reward scales.
              </p>
            </CardContent>
          </Card>
        </CardContent>
      </Card>
    </div>
  );
}