'use client';

import React, { useState } from 'react';
import Link from 'next/link';
import { ArrowLeft, Trophy, BarChart3, Target, Zap, Award, TrendingUp } from 'lucide-react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { BenchmarkDashboard } from '@/components/benchmarks/BenchmarkDashboard';

export default function BenchmarksPage() {
  const [activeTab, setActiveTab] = useState('dashboard');
  return (
    <div className="container mx-auto py-8 px-4 max-w-7xl">
      {/* Header */}
      <div className="mb-8">
        <Link href="/">
          <Button variant="ghost" className="mb-4">
            <ArrowLeft className="w-4 h-4 mr-2" />
            Back to Home
          </Button>
        </Link>
        
        <div className="flex items-start justify-between">
          <div>
            <div className="flex items-center gap-3 mb-4">
              <div className="p-2 bg-yellow-100 rounded-lg">
                <Trophy className="w-8 h-8 text-yellow-600" />
              </div>
              <div>
                <h1 className="text-4xl font-bold">RL Benchmark Suite</h1>
                <p className="text-xl text-muted-foreground">
                  Comprehensive evaluation platform for reinforcement learning algorithms
                </p>
              </div>
            </div>
            <div className="flex items-center gap-2 mb-4">
              <Badge variant="secondary">Standardized Evaluation</Badge>
              <Badge variant="secondary">Statistical Analysis</Badge>
              <Badge variant="secondary">Reproducible Results</Badge>
              <Badge variant="outline">Research Grade</Badge>
            </div>
          </div>
          
          <div className="text-right">
            <p className="text-sm text-muted-foreground mb-2">Benchmark Type</p>
            <Badge variant="secondary" className="bg-yellow-100 text-yellow-800">
              Comprehensive
            </Badge>
          </div>
        </div>
      </div>

      <Tabs value={activeTab} onValueChange={setActiveTab} className="space-y-6">
        <TabsList className="grid w-full grid-cols-4">
          <TabsTrigger value="dashboard">Dashboard</TabsTrigger>
          <TabsTrigger value="overview">Overview</TabsTrigger>
          <TabsTrigger value="environments">Environments</TabsTrigger>
          <TabsTrigger value="methodology">Methodology</TabsTrigger>
        </TabsList>

        {/* Dashboard Tab */}
        <TabsContent value="dashboard">
          <BenchmarkDashboard />
        </TabsContent>

        {/* Overview Tab */}
        <TabsContent value="overview" className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Target className="w-5 h-5" />
                What is the RL Benchmark Suite?
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <p className="text-lg">
                The RL Benchmark Suite is a comprehensive evaluation platform designed to provide 
                standardized, reproducible, and statistically rigorous benchmarking for reinforcement 
                learning algorithms across diverse environments.
              </p>
              
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div>
                  <h3 className="text-lg font-semibold mb-3">Key Features</h3>
                  <ul className="space-y-2">
                    <li className="flex items-start gap-2">
                      <div className="w-2 h-2 bg-blue-500 rounded-full mt-2"></div>
                      <span>Standardized evaluation protocols across environments</span>
                    </li>
                    <li className="flex items-start gap-2">
                      <div className="w-2 h-2 bg-blue-500 rounded-full mt-2"></div>
                      <span>Statistical significance testing with effect sizes</span>
                    </li>
                    <li className="flex items-start gap-2">
                      <div className="w-2 h-2 bg-blue-500 rounded-full mt-2"></div>
                      <span>Multiple evaluation metrics (reward, sample efficiency, convergence)</span>
                    </li>
                    <li className="flex items-start gap-2">
                      <div className="w-2 h-2 bg-blue-500 rounded-full mt-2"></div>
                      <span>Reproducible results with controlled randomization</span>
                    </li>
                    <li className="flex items-start gap-2">
                      <div className="w-2 h-2 bg-blue-500 rounded-full mt-2"></div>
                      <span>Comprehensive reporting and data export</span>
                    </li>
                  </ul>
                </div>
                
                <div>
                  <h3 className="text-lg font-semibold mb-3">Supported Algorithms</h3>
                  <ul className="space-y-2">
                    <li className="flex items-start gap-2">
                      <Zap className="w-4 h-4 text-green-500 mt-1" />
                      <span>PPO (Proximal Policy Optimization)</span>
                    </li>
                    <li className="flex items-start gap-2">
                      <Zap className="w-4 h-4 text-green-500 mt-1" />
                      <span>SAC (Soft Actor-Critic)</span>
                    </li>
                    <li className="flex items-start gap-2">
                      <Zap className="w-4 h-4 text-green-500 mt-1" />
                      <span>GRPO (Group Relative Policy Optimization)</span>
                    </li>
                    <li className="flex items-start gap-2">
                      <Zap className="w-4 h-4 text-green-500 mt-1" />
                      <span>MAPPO (Multi-Agent PPO)</span>
                    </li>
                    <li className="flex items-start gap-2">
                      <Zap className="w-4 h-4 text-green-500 mt-1" />
                      <span>DQN (Deep Q-Network)</span>
                    </li>
                  </ul>
                </div>
              </div>
            </CardContent>
          </Card>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2 text-lg">
                  <BarChart3 className="w-5 h-5 text-blue-500" />
                  Evaluation Metrics
                </CardTitle>
              </CardHeader>
              <CardContent>
                <p className="text-sm text-muted-foreground mb-3">
                  Multiple dimensions of algorithm performance are measured to provide 
                  comprehensive evaluation beyond just final reward.
                </p>
                <div className="space-y-2 text-sm">
                  <div className="flex justify-between">
                    <span>Episode Reward:</span>
                    <span className="font-medium">Performance</span>
                  </div>
                  <div className="flex justify-between">
                    <span>Sample Efficiency:</span>
                    <span className="font-medium">Learning Speed</span>
                  </div>
                  <div className="flex justify-between">
                    <span>Convergence Rate:</span>
                    <span className="font-medium">Stability</span>
                  </div>
                  <div className="flex justify-between">
                    <span>Wall Clock Time:</span>
                    <span className="font-medium">Computational Cost</span>
                  </div>
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2 text-lg">
                  <TrendingUp className="w-5 h-5 text-green-500" />
                  Statistical Analysis
                </CardTitle>
              </CardHeader>
              <CardContent>
                <p className="text-sm text-muted-foreground mb-3">
                  Rigorous statistical methods ensure reliable comparison between 
                  algorithms with confidence intervals and significance testing.
                </p>
                <div className="space-y-2 text-sm">
                  <div className="flex justify-between">
                    <span>Multiple Runs:</span>
                    <span className="font-medium">5-10 seeds</span>
                  </div>
                  <div className="flex justify-between">
                    <span>Significance Test:</span>
                    <span className="font-medium">Mann-Whitney U</span>
                  </div>
                  <div className="flex justify-between">
                    <span>Effect Size:</span>
                    <span className="font-medium">Cohen's d</span>
                  </div>
                  <div className="flex justify-between">
                    <span>Confidence:</span>
                    <span className="font-medium">95% intervals</span>
                  </div>
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2 text-lg">
                  <Award className="w-5 h-5 text-purple-500" />
                  Reproducibility
                </CardTitle>
              </CardHeader>
              <CardContent>
                <p className="text-sm text-muted-foreground mb-3">
                  All benchmarks are fully reproducible with controlled randomization, 
                  detailed logging, and version tracking.
                </p>
                <div className="space-y-2 text-sm">
                  <div className="flex justify-between">
                    <span>Seed Control:</span>
                    <span className="font-medium">Fixed Seeds</span>
                  </div>
                  <div className="flex justify-between">
                    <span>Environment:</span>
                    <span className="font-medium">Deterministic</span>
                  </div>
                  <div className="flex justify-between">
                    <span>Hyperparameters:</span>
                    <span className="font-medium">Logged</span>
                  </div>
                  <div className="flex justify-between">
                    <span>Export Format:</span>
                    <span className="font-medium">JSON/CSV</span>
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>

          <Card>
            <CardHeader>
              <CardTitle>Benchmark Categories</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-5 gap-4">
                <div className="text-center p-4 border rounded-lg">
                  <div className="w-12 h-12 bg-green-100 rounded-full flex items-center justify-center mx-auto mb-2">
                    <Target className="w-6 h-6 text-green-600" />
                  </div>
                  <h4 className="font-medium">Classic Control</h4>
                  <p className="text-sm text-muted-foreground">CartPole, Pendulum, MountainCar</p>
                </div>
                
                <div className="text-center p-4 border rounded-lg">
                  <div className="w-12 h-12 bg-blue-100 rounded-full flex items-center justify-center mx-auto mb-2">
                    <Zap className="w-6 h-6 text-blue-600" />
                  </div>
                  <h4 className="font-medium">Navigation</h4>
                  <p className="text-sm text-muted-foreground">LunarLander, BipedalWalker</p>
                </div>
                
                <div className="text-center p-4 border rounded-lg">
                  <div className="w-12 h-12 bg-purple-100 rounded-full flex items-center justify-center mx-auto mb-2">
                    <BarChart3 className="w-6 h-6 text-purple-600" />
                  </div>
                  <h4 className="font-medium">Robotics</h4>
                  <p className="text-sm text-muted-foreground">Reacher, Humanoid</p>
                </div>
                
                <div className="text-center p-4 border rounded-lg">
                  <div className="w-12 h-12 bg-orange-100 rounded-full flex items-center justify-center mx-auto mb-2">
                    <TrendingUp className="w-6 h-6 text-orange-600" />
                  </div>
                  <h4 className="font-medium">Multi-Agent</h4>
                  <p className="text-sm text-muted-foreground">Cooperative Navigation</p>
                </div>
                
                <div className="text-center p-4 border rounded-lg">
                  <div className="w-12 h-12 bg-red-100 rounded-full flex items-center justify-center mx-auto mb-2">
                    <Award className="w-6 h-6 text-red-600" />
                  </div>
                  <h4 className="font-medium">Custom</h4>
                  <p className="text-sm text-muted-foreground">Maze Navigation</p>
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        {/* Environments Tab */}
        <TabsContent value="environments" className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle>Benchmark Environments</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-6">
                {/* Classic Control */}
                <div>
                  <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
                    <div className="w-4 h-4 bg-green-500 rounded-full"></div>
                    Classic Control Environments
                  </h3>
                  <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                    <div className="border rounded-lg p-4">
                      <h4 className="font-medium mb-2">CartPole-v1</h4>
                      <p className="text-sm text-muted-foreground mb-3">
                        Balance a pole on a cart by moving left or right. Classic discrete control problem.
                      </p>
                      <div className="space-y-1 text-sm">
                        <div className="flex justify-between">
                          <span>State Space:</span>
                          <span>4 (continuous)</span>
                        </div>
                        <div className="flex justify-between">
                          <span>Action Space:</span>
                          <span>2 (discrete)</span>
                        </div>
                        <div className="flex justify-between">
                          <span>Success Threshold:</span>
                          <span>475</span>
                        </div>
                        <div className="flex justify-between">
                          <span>Difficulty:</span>
                          <Badge variant="secondary" className="bg-green-100 text-green-800 text-xs">Easy</Badge>
                        </div>
                      </div>
                    </div>
                    
                    <div className="border rounded-lg p-4">
                      <h4 className="font-medium mb-2">Pendulum-v1</h4>
                      <p className="text-sm text-muted-foreground mb-3">
                        Swing up and balance an inverted pendulum. Continuous control benchmark.
                      </p>
                      <div className="space-y-1 text-sm">
                        <div className="flex justify-between">
                          <span>State Space:</span>
                          <span>3 (continuous)</span>
                        </div>
                        <div className="flex justify-between">
                          <span>Action Space:</span>
                          <span>1 (continuous)</span>
                        </div>
                        <div className="flex justify-between">
                          <span>Success Threshold:</span>
                          <span>-200</span>
                        </div>
                        <div className="flex justify-between">
                          <span>Difficulty:</span>
                          <Badge variant="secondary" className="bg-blue-100 text-blue-800 text-xs">Medium</Badge>
                        </div>
                      </div>
                    </div>
                    
                    <div className="border rounded-lg p-4">
                      <h4 className="font-medium mb-2">MountainCar-v0</h4>
                      <p className="text-sm text-muted-foreground mb-3">
                        Drive an underpowered car up a hill. Sparse reward environment.
                      </p>
                      <div className="space-y-1 text-sm">
                        <div className="flex justify-between">
                          <span>State Space:</span>
                          <span>2 (continuous)</span>
                        </div>
                        <div className="flex justify-between">
                          <span>Action Space:</span>
                          <span>3 (discrete)</span>
                        </div>
                        <div className="flex justify-between">
                          <span>Success Threshold:</span>
                          <span>-110</span>
                        </div>
                        <div className="flex justify-between">
                          <span>Difficulty:</span>
                          <Badge variant="secondary" className="bg-blue-100 text-blue-800 text-xs">Medium</Badge>
                        </div>
                      </div>
                    </div>
                  </div>
                </div>

                {/* Navigation & Locomotion */}
                <div>
                  <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
                    <div className="w-4 h-4 bg-blue-500 rounded-full"></div>
                    Navigation & Locomotion
                  </h3>
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    <div className="border rounded-lg p-4">
                      <h4 className="font-medium mb-2">LunarLander-v2</h4>
                      <p className="text-sm text-muted-foreground mb-3">
                        Land a spacecraft on the moon surface using thrusters. Physics-based simulation.
                      </p>
                      <div className="space-y-1 text-sm">
                        <div className="flex justify-between">
                          <span>State Space:</span>
                          <span>8 (continuous)</span>
                        </div>
                        <div className="flex justify-between">
                          <span>Action Space:</span>
                          <span>4 (discrete)</span>
                        </div>
                        <div className="flex justify-between">
                          <span>Success Threshold:</span>
                          <span>200</span>
                        </div>
                        <div className="flex justify-between">
                          <span>Difficulty:</span>
                          <Badge variant="secondary" className="bg-blue-100 text-blue-800 text-xs">Medium</Badge>
                        </div>
                      </div>
                    </div>
                    
                    <div className="border rounded-lg p-4">
                      <h4 className="font-medium mb-2">BipedalWalker-v3</h4>
                      <p className="text-sm text-muted-foreground mb-3">
                        Train a bipedal robot to walk forward. Complex locomotion task.
                      </p>
                      <div className="space-y-1 text-sm">
                        <div className="flex justify-between">
                          <span>State Space:</span>
                          <span>24 (continuous)</span>
                        </div>
                        <div className="flex justify-between">
                          <span>Action Space:</span>
                          <span>4 (continuous)</span>
                        </div>
                        <div className="flex justify-between">
                          <span>Success Threshold:</span>
                          <span>300</span>
                        </div>
                        <div className="flex justify-between">
                          <span>Difficulty:</span>
                          <Badge variant="secondary" className="bg-orange-100 text-orange-800 text-xs">Hard</Badge>
                        </div>
                      </div>
                    </div>
                  </div>
                </div>

                {/* High-Dimensional */}
                <div>
                  <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
                    <div className="w-4 h-4 bg-red-500 rounded-full"></div>
                    High-Dimensional Challenges
                  </h3>
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    <div className="border rounded-lg p-4">
                      <h4 className="font-medium mb-2">Humanoid-v3</h4>
                      <p className="text-sm text-muted-foreground mb-3">
                        Control a high-dimensional humanoid robot to walk forward. Ultimate locomotion challenge.
                      </p>
                      <div className="space-y-1 text-sm">
                        <div className="flex justify-between">
                          <span>State Space:</span>
                          <span>376 (continuous)</span>
                        </div>
                        <div className="flex justify-between">
                          <span>Action Space:</span>
                          <span>17 (continuous)</span>
                        </div>
                        <div className="flex justify-between">
                          <span>Success Threshold:</span>
                          <span>6000</span>
                        </div>
                        <div className="flex justify-between">
                          <span>Difficulty:</span>
                          <Badge variant="secondary" className="bg-red-100 text-red-800 text-xs">Expert</Badge>
                        </div>
                      </div>
                    </div>
                    
                    <div className="border rounded-lg p-4">
                      <h4 className="font-medium mb-2">Cooperative Navigation</h4>
                      <p className="text-sm text-muted-foreground mb-3">
                        Multiple agents cooperatively navigate to targets while avoiding collisions.
                      </p>
                      <div className="space-y-1 text-sm">
                        <div className="flex justify-between">
                          <span>State Space:</span>
                          <span>18 (continuous)</span>
                        </div>
                        <div className="flex justify-between">
                          <span>Action Space:</span>
                          <span>5 (discrete)</span>
                        </div>
                        <div className="flex justify-between">
                          <span>Success Threshold:</span>
                          <span>-3</span>
                        </div>
                        <div className="flex justify-between">
                          <span>Difficulty:</span>
                          <Badge variant="secondary" className="bg-orange-100 text-orange-800 text-xs">Hard</Badge>
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        {/* Methodology Tab */}
        <TabsContent value="methodology" className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle>Benchmarking Methodology</CardTitle>
            </CardHeader>
            <CardContent className="space-y-6">
              <div>
                <h3 className="text-lg font-semibold mb-3">Evaluation Protocol</h3>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  <div>
                    <h4 className="font-medium mb-2">Standard Settings</h4>
                    <ul className="space-y-1 text-sm">
                      <li>• 5-10 independent runs per algorithm-environment pair</li>
                      <li>• Fixed random seeds for reproducibility</li>
                      <li>• Maximum 1000-2000 episodes per run</li>
                      <li>• Early stopping when performance plateaus</li>
                      <li>• Hyperparameter logging and version control</li>
                    </ul>
                  </div>
                  <div>
                    <h4 className="font-medium mb-2">Evaluation Metrics</h4>
                    <ul className="space-y-1 text-sm">
                      <li>• <strong>Episode Reward:</strong> Final performance measure</li>
                      <li>• <strong>Sample Efficiency:</strong> Steps to convergence</li>
                      <li>• <strong>Convergence Rate:</strong> Episodes to success threshold</li>
                      <li>• <strong>Success Rate:</strong> Fraction of successful runs</li>
                      <li>• <strong>Wall Clock Time:</strong> Computational efficiency</li>
                    </ul>
                  </div>
                </div>
              </div>

              <div>
                <h3 className="text-lg font-semibold mb-3">Statistical Analysis</h3>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  <div>
                    <h4 className="font-medium mb-2">Descriptive Statistics</h4>
                    <ul className="space-y-1 text-sm">
                      <li>• Mean and standard deviation</li>
                      <li>• Median and interquartile range</li>
                      <li>• Min/max values across runs</li>
                      <li>• Confidence intervals (95%)</li>
                      <li>• Distribution visualization</li>
                    </ul>
                  </div>
                  <div>
                    <h4 className="font-medium mb-2">Significance Testing</h4>
                    <ul className="space-y-1 text-sm">
                      <li>• Mann-Whitney U test (non-parametric)</li>
                      <li>• Bonferroni correction for multiple comparisons</li>
                      <li>• Cohen's d effect size measurement</li>
                      <li>• Bootstrap confidence intervals</li>
                      <li>• Power analysis for sample size validation</li>
                    </ul>
                  </div>
                </div>
              </div>

              <div>
                <h3 className="text-lg font-semibold mb-3">Reproducibility Standards</h3>
                <div className="bg-muted/30 p-4 rounded-lg">
                  <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                    <div>
                      <h4 className="font-medium mb-2">Environment Control</h4>
                      <ul className="space-y-1 text-sm">
                        <li>• Fixed environment seeds</li>
                        <li>• Deterministic physics</li>
                        <li>• Consistent initialization</li>
                        <li>• Version-locked dependencies</li>
                      </ul>
                    </div>
                    <div>
                      <h4 className="font-medium mb-2">Algorithm Control</h4>
                      <ul className="space-y-1 text-sm">
                        <li>• Fixed neural network seeds</li>
                        <li>• Documented hyperparameters</li>
                        <li>• Consistent network architectures</li>
                        <li>• Standardized preprocessing</li>
                      </ul>
                    </div>
                    <div>
                      <h4 className="font-medium mb-2">Data Management</h4>
                      <ul className="space-y-1 text-sm">
                        <li>• Complete result logging</li>
                        <li>• Metadata preservation</li>
                        <li>• Structured export formats</li>
                        <li>• Audit trail maintenance</li>
                      </ul>
                    </div>
                  </div>
                </div>
              </div>

              <div>
                <h3 className="text-lg font-semibold mb-3">Best Practices</h3>
                <div className="space-y-4">
                  <div className="border-l-4 border-green-500 pl-4">
                    <h4 className="font-medium text-green-800">Recommended Approach</h4>
                    <p className="text-sm text-muted-foreground">
                      Run multiple algorithms on the same environment simultaneously to minimize environmental 
                      variance. Use statistical significance testing before claiming performance differences.
                    </p>
                  </div>
                  
                  <div className="border-l-4 border-orange-500 pl-4">
                    <h4 className="font-medium text-orange-800">Common Pitfalls</h4>
                    <p className="text-sm text-muted-foreground">
                      Avoid cherry-picking best runs, reporting single-run results, or comparing algorithms 
                      with different hyperparameter budgets. Always report confidence intervals.
                    </p>
                  </div>
                  
                  <div className="border-l-4 border-blue-500 pl-4">
                    <h4 className="font-medium text-blue-800">Interpretation Guidelines</h4>
                    <p className="text-sm text-muted-foreground">
                      Consider both statistical and practical significance. Small but consistent improvements 
                      may be statistically significant but not practically meaningful in real applications.
                    </p>
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
}