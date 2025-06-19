'use client';

import React, { useState, useEffect, useRef } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Progress } from '@/components/ui/progress';
import { Badge } from '@/components/ui/badge';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { 
  Play, Pause, Square, RotateCcw, Download, Upload, 
  Activity, Brain, Cpu, HardDrive, Zap, TrendingUp,
  AlertTriangle, CheckCircle, Clock, BarChart3, Info
} from 'lucide-react';
import { 
  LineChart, Line, AreaChart, Area, BarChart, Bar,
  XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  PieChart, Pie, Cell, Legend
} from 'recharts';

interface TrainingMetrics {
  episode: number;
  reward: number;
  loss: number;
  learningRate: number;
  epsilon?: number;
  entropy?: number;
  valueEstimate: number;
  gradientNorm: number;
  fps: number;
}

interface SystemMetrics {
  cpuUsage: number;
  gpuUsage: number;
  memoryUsage: number;
  diskUsage: number;
  temperature: number;
}

export function EnhancedTrainingDashboard() {
  const [isTraining, setIsTraining] = useState(false);
  const [isPaused, setIsPaused] = useState(false);
  const [currentEpisode, setCurrentEpisode] = useState(0);
  const [trainingMetrics, setTrainingMetrics] = useState<TrainingMetrics[]>([]);
  const [systemMetrics, setSystemMetrics] = useState<SystemMetrics>({
    cpuUsage: 0,
    gpuUsage: 0,
    memoryUsage: 0,
    diskUsage: 0,
    temperature: 0
  });
  const [logs, setLogs] = useState<string[]>([]);
  const [alerts, setAlerts] = useState<{ type: 'warning' | 'error' | 'success', message: string }[]>([]);
  
  const logsEndRef = useRef<HTMLDivElement>(null);
  const wsRef = useRef<WebSocket | null>(null);

  // Simulate training progress
  useEffect(() => {
    if (isTraining && !isPaused) {
      const interval = setInterval(() => {
        setCurrentEpisode(prev => {
          const next = prev + 1;
          
          // Generate training metrics
          const newMetric: TrainingMetrics = {
            episode: next,
            reward: Math.sin(next * 0.01) * 100 + 100 + Math.random() * 50,
            loss: Math.max(0, 5 - next * 0.01 + Math.random() * 0.5),
            learningRate: 3e-4 * Math.pow(0.99, Math.floor(next / 100)),
            epsilon: Math.max(0.01, 1 - next * 0.001),
            entropy: Math.max(0, 2 - next * 0.002 + Math.random() * 0.2),
            valueEstimate: Math.sin(next * 0.02) * 50 + 150,
            gradientNorm: Math.random() * 5 + 2,
            fps: 1000 + Math.random() * 500
          };
          
          setTrainingMetrics(prev => [...prev.slice(-199), newMetric]);
          
          // Update system metrics
          setSystemMetrics({
            cpuUsage: 60 + Math.random() * 30,
            gpuUsage: 70 + Math.random() * 25,
            memoryUsage: 40 + Math.random() * 20,
            diskUsage: 30 + Math.random() * 10,
            temperature: 65 + Math.random() * 15
          });
          
          // Add logs
          if (next % 10 === 0) {
            addLog(`Episode ${next}: Reward = ${newMetric.reward.toFixed(2)}, Loss = ${newMetric.loss.toFixed(4)}`);
          }
          
          // Check for alerts
          if (newMetric.gradientNorm > 6) {
            addAlert('warning', 'High gradient norm detected. Consider reducing learning rate.');
          }
          if (next % 100 === 0) {
            addAlert('success', `Checkpoint saved at episode ${next}`);
          }
          
          return next;
        });
      }, 100);
      
      return () => clearInterval(interval);
    }
  }, [isTraining, isPaused]);

  // Auto-scroll logs
  useEffect(() => {
    logsEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [logs]);

  const addLog = (message: string) => {
    const timestamp = new Date().toLocaleTimeString();
    setLogs(prev => [...prev.slice(-99), `[${timestamp}] ${message}`]);
  };

  const addAlert = (type: 'warning' | 'error' | 'success', message: string) => {
    setAlerts(prev => [...prev.slice(-4), { type, message }]);
  };

  const startTraining = () => {
    setIsTraining(true);
    setIsPaused(false);
    addLog('Training started');
    addAlert('success', 'Training session initiated successfully');
  };

  const pauseTraining = () => {
    setIsPaused(!isPaused);
    addLog(isPaused ? 'Training resumed' : 'Training paused');
  };

  const stopTraining = () => {
    setIsTraining(false);
    setIsPaused(false);
    addLog('Training stopped');
    addAlert('warning', 'Training session terminated');
  };

  const resetTraining = () => {
    setIsTraining(false);
    setIsPaused(false);
    setCurrentEpisode(0);
    setTrainingMetrics([]);
    setLogs([]);
    setAlerts([]);
    addLog('Training environment reset');
  };

  const exportMetrics = () => {
    const data = JSON.stringify({ trainingMetrics, systemMetrics }, null, 2);
    const blob = new Blob([data], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `training_metrics_episode_${currentEpisode}.json`;
    a.click();
    URL.revokeObjectURL(url);
    addLog('Metrics exported');
  };

  // Get latest metrics
  const latestMetric = trainingMetrics[trainingMetrics.length - 1];

  return (
    <div className="space-y-6">
      {/* Control Panel */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center justify-between">
            <span className="flex items-center gap-2">
              <Activity className="w-5 h-5" />
              Enhanced Training Dashboard
            </span>
            <div className="flex items-center gap-2">
              {!isTraining ? (
                <Button onClick={startTraining}>
                  <Play className="w-4 h-4 mr-2" />
                  Start Training
                </Button>
              ) : (
                <>
                  <Button 
                    variant="secondary" 
                    onClick={pauseTraining}
                  >
                    {isPaused ? <Play className="w-4 h-4" /> : <Pause className="w-4 h-4" />}
                  </Button>
                  <Button 
                    variant="destructive" 
                    onClick={stopTraining}
                  >
                    <Square className="w-4 h-4" />
                  </Button>
                </>
              )}
              <Button variant="outline" onClick={resetTraining}>
                <RotateCcw className="w-4 h-4" />
              </Button>
              <Button variant="outline" onClick={exportMetrics}>
                <Download className="w-4 h-4" />
              </Button>
            </div>
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
            <div>
              <p className="text-sm text-muted-foreground">Episode</p>
              <p className="text-2xl font-bold">{currentEpisode}</p>
            </div>
            <div>
              <p className="text-sm text-muted-foreground">Average Reward</p>
              <p className="text-2xl font-bold">
                {latestMetric?.reward.toFixed(1) || '0.0'}
              </p>
            </div>
            <div>
              <p className="text-sm text-muted-foreground">Loss</p>
              <p className="text-2xl font-bold">
                {latestMetric?.loss.toFixed(4) || '0.0000'}
              </p>
            </div>
            <div>
              <p className="text-sm text-muted-foreground">FPS</p>
              <p className="text-2xl font-bold">
                {latestMetric?.fps.toFixed(0) || '0'}
              </p>
            </div>
          </div>
          
          {/* Progress Bar */}
          <div className="mt-4">
            <div className="flex justify-between text-sm mb-1">
              <span>Training Progress</span>
              <span>{(currentEpisode / 1000 * 100).toFixed(1)}%</span>
            </div>
            <Progress value={currentEpisode / 10} className="h-2" />
          </div>
        </CardContent>
      </Card>

      {/* Alerts */}
      {alerts.length > 0 && (
        <div className="space-y-2">
          {alerts.map((alert, i) => (
            <div
              key={i}
              className={`flex items-center gap-2 p-3 rounded-lg ${
                alert.type === 'error' ? 'bg-red-100 text-red-900' :
                alert.type === 'warning' ? 'bg-yellow-100 text-yellow-900' :
                'bg-green-100 text-green-900'
              }`}
            >
              {alert.type === 'error' ? <AlertTriangle className="w-4 h-4" /> :
               alert.type === 'warning' ? <AlertTriangle className="w-4 h-4" /> :
               <CheckCircle className="w-4 h-4" />}
              <span className="text-sm">{alert.message}</span>
            </div>
          ))}
        </div>
      )}

      {/* Main Dashboard Tabs */}
      <Tabs defaultValue="metrics" className="space-y-4">
        <TabsList className="grid w-full grid-cols-4">
          <TabsTrigger value="metrics">Metrics</TabsTrigger>
          <TabsTrigger value="system">System</TabsTrigger>
          <TabsTrigger value="analysis">Analysis</TabsTrigger>
          <TabsTrigger value="logs">Logs</TabsTrigger>
        </TabsList>

        {/* Metrics Tab */}
        <TabsContent value="metrics" className="space-y-4">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
            {/* Reward Chart */}
            <Card>
              <CardHeader>
                <CardTitle className="text-lg">Episode Rewards</CardTitle>
              </CardHeader>
              <CardContent>
                <ResponsiveContainer width="100%" height={250}>
                  <LineChart data={trainingMetrics}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="episode" />
                    <YAxis />
                    <Tooltip />
                    <Line 
                      type="monotone" 
                      dataKey="reward" 
                      stroke="#3b82f6" 
                      strokeWidth={2}
                      dot={false}
                    />
                  </LineChart>
                </ResponsiveContainer>
              </CardContent>
            </Card>

            {/* Loss Chart */}
            <Card>
              <CardHeader>
                <CardTitle className="text-lg">Training Loss</CardTitle>
              </CardHeader>
              <CardContent>
                <ResponsiveContainer width="100%" height={250}>
                  <AreaChart data={trainingMetrics}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="episode" />
                    <YAxis />
                    <Tooltip />
                    <Area 
                      type="monotone" 
                      dataKey="loss" 
                      stroke="#ef4444" 
                      fill="#ef4444"
                      fillOpacity={0.3}
                    />
                  </AreaChart>
                </ResponsiveContainer>
              </CardContent>
            </Card>

            {/* Learning Metrics */}
            <Card>
              <CardHeader>
                <CardTitle className="text-lg">Learning Dynamics</CardTitle>
              </CardHeader>
              <CardContent>
                <ResponsiveContainer width="100%" height={250}>
                  <LineChart data={trainingMetrics}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="episode" />
                    <YAxis />
                    <Tooltip />
                    <Legend />
                    <Line 
                      type="monotone" 
                      dataKey="entropy" 
                      stroke="#8b5cf6" 
                      strokeWidth={2}
                      dot={false}
                      name="Entropy"
                    />
                    <Line 
                      type="monotone" 
                      dataKey="gradientNorm" 
                      stroke="#f59e0b" 
                      strokeWidth={2}
                      dot={false}
                      name="Gradient Norm"
                    />
                  </LineChart>
                </ResponsiveContainer>
              </CardContent>
            </Card>

            {/* Value Estimates */}
            <Card>
              <CardHeader>
                <CardTitle className="text-lg">Value Function</CardTitle>
              </CardHeader>
              <CardContent>
                <ResponsiveContainer width="100%" height={250}>
                  <LineChart data={trainingMetrics}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="episode" />
                    <YAxis />
                    <Tooltip />
                    <Line 
                      type="monotone" 
                      dataKey="valueEstimate" 
                      stroke="#10b981" 
                      strokeWidth={2}
                      dot={false}
                    />
                  </LineChart>
                </ResponsiveContainer>
              </CardContent>
            </Card>
          </div>
        </TabsContent>

        {/* System Tab */}
        <TabsContent value="system" className="space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="text-sm flex items-center gap-2">
                  <Cpu className="w-4 h-4" />
                  CPU Usage
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold">
                  {systemMetrics.cpuUsage.toFixed(1)}%
                </div>
                <Progress value={systemMetrics.cpuUsage} className="mt-2 h-1" />
              </CardContent>
            </Card>

            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="text-sm flex items-center gap-2">
                  <Zap className="w-4 h-4" />
                  GPU Usage
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold">
                  {systemMetrics.gpuUsage.toFixed(1)}%
                </div>
                <Progress value={systemMetrics.gpuUsage} className="mt-2 h-1" />
              </CardContent>
            </Card>

            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="text-sm flex items-center gap-2">
                  <HardDrive className="w-4 h-4" />
                  Memory
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold">
                  {systemMetrics.memoryUsage.toFixed(1)}%
                </div>
                <Progress value={systemMetrics.memoryUsage} className="mt-2 h-1" />
              </CardContent>
            </Card>

            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="text-sm flex items-center gap-2">
                  <Activity className="w-4 h-4" />
                  Temperature
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold">
                  {systemMetrics.temperature.toFixed(0)}°C
                </div>
                <Progress 
                  value={systemMetrics.temperature} 
                  className="mt-2 h-1"
                />
              </CardContent>
            </Card>
          </div>

          {/* Resource Distribution */}
          <Card>
            <CardHeader>
              <CardTitle>Resource Distribution</CardTitle>
            </CardHeader>
            <CardContent>
              <ResponsiveContainer width="100%" height={300}>
                <PieChart>
                  <Pie
                    data={[
                      { name: 'Model Forward', value: 35, fill: '#3b82f6' },
                      { name: 'Backpropagation', value: 25, fill: '#8b5cf6' },
                      { name: 'Data Loading', value: 15, fill: '#f59e0b' },
                      { name: 'Environment Step', value: 20, fill: '#10b981' },
                      { name: 'Other', value: 5, fill: '#6b7280' },
                    ]}
                    cx="50%"
                    cy="50%"
                    outerRadius={100}
                    label
                  />
                  <Tooltip />
                  <Legend />
                </PieChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>
        </TabsContent>

        {/* Analysis Tab */}
        <TabsContent value="analysis" className="space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <Card>
              <CardHeader>
                <CardTitle>Training Statistics</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-3">
                  <div className="flex justify-between">
                    <span className="text-sm text-muted-foreground">Total Episodes</span>
                    <span className="font-medium">{currentEpisode}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-sm text-muted-foreground">Best Reward</span>
                    <span className="font-medium">
                      {Math.max(...trainingMetrics.map(m => m.reward), 0).toFixed(1)}
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-sm text-muted-foreground">Average FPS</span>
                    <span className="font-medium">
                      {(trainingMetrics.reduce((acc, m) => acc + m.fps, 0) / trainingMetrics.length || 0).toFixed(0)}
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-sm text-muted-foreground">Learning Rate</span>
                    <span className="font-medium">
                      {latestMetric?.learningRate.toExponential(2) || '3e-4'}
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-sm text-muted-foreground">Exploration (ε)</span>
                    <span className="font-medium">
                      {latestMetric?.epsilon?.toFixed(3) || '0.01'}
                    </span>
                  </div>
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>Convergence Analysis</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-3">
                  <div>
                    <div className="flex justify-between mb-1">
                      <span className="text-sm">Reward Stability</span>
                      <Badge variant="secondary">
                        {currentEpisode > 100 ? 'Stable' : 'Training'}
                      </Badge>
                    </div>
                    <Progress value={Math.min(currentEpisode / 100 * 100, 100)} className="h-2" />
                  </div>
                  <div>
                    <div className="flex justify-between mb-1">
                      <span className="text-sm">Loss Convergence</span>
                      <Badge variant="secondary">
                        {latestMetric?.loss < 1 ? 'Good' : 'In Progress'}
                      </Badge>
                    </div>
                    <Progress value={Math.max(0, 100 - (latestMetric?.loss || 5) * 20)} className="h-2" />
                  </div>
                  <div>
                    <div className="flex justify-between mb-1">
                      <span className="text-sm">Gradient Health</span>
                      <Badge variant="secondary">
                        {latestMetric?.gradientNorm < 5 ? 'Healthy' : 'High'}
                      </Badge>
                    </div>
                    <Progress value={Math.max(0, 100 - (latestMetric?.gradientNorm || 0) * 10)} className="h-2" />
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>

          {/* Recommendations */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Brain className="w-5 h-5" />
                Training Recommendations
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-2">
                {currentEpisode > 0 && latestMetric && (
                  <>
                    {latestMetric.gradientNorm > 5 && (
                      <div className="flex items-start gap-2 p-3 bg-yellow-50 rounded-lg">
                        <AlertTriangle className="w-4 h-4 text-yellow-600 mt-0.5" />
                        <div className="text-sm">
                          <p className="font-medium">High gradient norm detected</p>
                          <p className="text-muted-foreground">
                            Consider reducing learning rate or using gradient clipping
                          </p>
                        </div>
                      </div>
                    )}
                    {latestMetric.entropy < 0.5 && (
                      <div className="flex items-start gap-2 p-3 bg-blue-50 rounded-lg">
                        <Info className="w-4 h-4 text-blue-600 mt-0.5" />
                        <div className="text-sm">
                          <p className="font-medium">Low entropy detected</p>
                          <p className="text-muted-foreground">
                            Policy may be too deterministic. Consider increasing exploration
                          </p>
                        </div>
                      </div>
                    )}
                    {currentEpisode > 500 && Math.max(...trainingMetrics.slice(-100).map(m => m.reward)) < 150 && (
                      <div className="flex items-start gap-2 p-3 bg-orange-50 rounded-lg">
                        <AlertTriangle className="w-4 h-4 text-orange-600 mt-0.5" />
                        <div className="text-sm">
                          <p className="font-medium">Slow learning progress</p>
                          <p className="text-muted-foreground">
                            Try adjusting hyperparameters or checking environment setup
                          </p>
                        </div>
                      </div>
                    )}
                  </>
                )}
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        {/* Logs Tab */}
        <TabsContent value="logs">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center justify-between">
                <span>Training Logs</span>
                <Button 
                  size="sm" 
                  variant="outline"
                  onClick={() => setLogs([])}
                >
                  Clear Logs
                </Button>
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="bg-muted rounded-lg p-4 h-96 overflow-y-auto font-mono text-sm">
                {logs.length === 0 ? (
                  <p className="text-muted-foreground">No logs yet. Start training to see logs.</p>
                ) : (
                  <>
                    {logs.map((log, i) => (
                      <div key={i} className="mb-1">
                        {log}
                      </div>
                    ))}
                    <div ref={logsEndRef} />
                  </>
                )}
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
}