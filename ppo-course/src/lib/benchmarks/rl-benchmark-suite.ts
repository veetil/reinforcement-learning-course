import * as tf from '@tensorflow/tfjs';

interface BenchmarkEnvironment {
  id: string;
  name: string;
  description: string;
  category: 'control' | 'navigation' | 'games' | 'robotics' | 'multi-agent';
  difficulty: 'easy' | 'medium' | 'hard' | 'expert';
  stateSize: number;
  actionSize: number;
  continuous: boolean;
  episodeLength: number;
  rewardRange: [number, number];
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

interface BenchmarkMetrics {
  mean: number;
  std: number;
  min: number;
  max: number;
  median: number;
  q25: number;
  q75: number;
  iqr: number;
}

interface BenchmarkComparison {
  environmentId: string;
  results: Map<string, BenchmarkMetrics>;
  rankings: Array<{
    algorithmId: string;
    score: number;
    rank: number;
  }>;
  statisticalTests: Map<string, {
    pValue: number;
    significant: boolean;
    effectSize: number;
  }>;
}

export class RLBenchmarkSuite {
  private environments: Map<string, BenchmarkEnvironment> = new Map();
  private results: BenchmarkResult[] = [];
  private algorithms: Map<string, any> = new Map();

  constructor() {
    this.initializeEnvironments();
  }

  private initializeEnvironments(): void {
    const envs: BenchmarkEnvironment[] = [
      // Classic Control
      {
        id: 'cartpole',
        name: 'CartPole-v1',
        description: 'Balance a pole on a cart by moving left or right',
        category: 'control',
        difficulty: 'easy',
        stateSize: 4,
        actionSize: 2,
        continuous: false,
        episodeLength: 500,
        rewardRange: [0, 500],
        successThreshold: 475,
        tags: ['discrete', 'continuous-state', 'classic']
      },
      {
        id: 'pendulum',
        name: 'Pendulum-v1',
        description: 'Swing up and balance an inverted pendulum',
        category: 'control',
        difficulty: 'medium',
        stateSize: 3,
        actionSize: 1,
        continuous: true,
        episodeLength: 200,
        rewardRange: [-1600, 0],
        successThreshold: -200,
        tags: ['continuous', 'continuous-state', 'classic']
      },
      {
        id: 'mountain_car',
        name: 'MountainCar-v0',
        description: 'Drive an underpowered car up a hill',
        category: 'control',
        difficulty: 'medium',
        stateSize: 2,
        actionSize: 3,
        continuous: false,
        episodeLength: 200,
        rewardRange: [-200, -1],
        successThreshold: -110,
        tags: ['discrete', 'sparse-reward', 'classic']
      },
      
      // Navigation
      {
        id: 'lunar_lander',
        name: 'LunarLander-v2',
        description: 'Land a spacecraft on the moon',
        category: 'navigation',
        difficulty: 'medium',
        stateSize: 8,
        actionSize: 4,
        continuous: false,
        episodeLength: 1000,
        rewardRange: [-200, 300],
        successThreshold: 200,
        tags: ['discrete', 'physics', 'shaped-reward']
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
        episodeLength: 2000,
        rewardRange: [-100, 350],
        successThreshold: 300,
        tags: ['continuous', 'physics', 'locomotion']
      },
      
      // Multi-Agent
      {
        id: 'cooperative_navigation',
        name: 'Cooperative Navigation',
        description: 'Multiple agents reach targets while avoiding collisions',
        category: 'multi-agent',
        difficulty: 'hard',
        stateSize: 18, // 3 agents * 6 features
        actionSize: 5,
        continuous: false,
        episodeLength: 50,
        rewardRange: [-10, 0],
        successThreshold: -3,
        tags: ['multi-agent', 'cooperative', 'discrete']
      },
      
      // Custom Environments
      {
        id: 'maze_navigation',
        name: 'Maze Navigation',
        description: 'Navigate through a randomly generated maze',
        category: 'navigation',
        difficulty: 'medium',
        stateSize: 84, // 84x84 grid observation
        actionSize: 4,
        continuous: false,
        episodeLength: 500,
        rewardRange: [-1, 100],
        successThreshold: 80,
        tags: ['discrete', 'sparse-reward', 'visual']
      },
      
      // Continuous Control
      {
        id: 'reacher',
        name: 'Reacher-v2',
        description: 'Control a 2-DOF arm to reach target positions',
        category: 'robotics',
        difficulty: 'medium',
        stateSize: 11,
        actionSize: 2,
        continuous: true,
        episodeLength: 50,
        rewardRange: [-50, 0],
        successThreshold: -5,
        tags: ['continuous', 'robotics', 'shaped-reward']
      },
      
      // High-Dimensional
      {
        id: 'humanoid',
        name: 'Humanoid-v3',
        description: 'Control a humanoid robot to walk forward',
        category: 'robotics',
        difficulty: 'expert',
        stateSize: 376,
        actionSize: 17,
        continuous: true,
        episodeLength: 1000,
        rewardRange: [0, 8000],
        successThreshold: 6000,
        tags: ['continuous', 'high-dimensional', 'locomotion']
      }
    ];

    envs.forEach(env => this.environments.set(env.id, env));
  }

  registerAlgorithm(id: string, algorithm: any): void {
    this.algorithms.set(id, algorithm);
  }

  getEnvironments(): BenchmarkEnvironment[] {
    return Array.from(this.environments.values());
  }

  getEnvironment(id: string): BenchmarkEnvironment | undefined {
    return this.environments.get(id);
  }

  async runBenchmark(
    algorithmId: string,
    environmentId: string,
    numRuns: number = 5,
    maxEpisodes: number = 1000,
    config?: any
  ): Promise<BenchmarkResult[]> {
    const algorithm = this.algorithms.get(algorithmId);
    const environment = this.environments.get(environmentId);
    
    if (!algorithm || !environment) {
      throw new Error(`Algorithm ${algorithmId} or environment ${environmentId} not found`);
    }

    const results: BenchmarkResult[] = [];
    
    for (let run = 0; run < numRuns; run++) {
      console.log(`Running benchmark ${run + 1}/${numRuns} for ${algorithmId} on ${environmentId}`);
      
      const startTime = performance.now();
      const seed = Math.floor(Math.random() * 10000);
      
      // Reset algorithm and environment
      await this.resetAlgorithm(algorithm, config, seed);
      const env = await this.createEnvironment(environmentId, seed);
      
      let totalReward = 0;
      let episodeCount = 0;
      let convergenceEpisode = -1;
      let trainingSteps = 0;
      
      // Training loop
      for (let episode = 0; episode < maxEpisodes; episode++) {
        const episodeResult = await this.runEpisode(algorithm, env);
        totalReward += episodeResult.reward;
        trainingSteps += episodeResult.steps;
        episodeCount++;
        
        // Check for convergence
        if (convergenceEpisode === -1 && episodeResult.reward >= environment.successThreshold) {
          convergenceEpisode = episode;
        }
        
        // Early stopping if converged and stable
        if (convergenceEpisode !== -1 && episode - convergenceEpisode > 100) {
          const recentAvg = await this.getRecentAverageReward(algorithm, 100);
          if (recentAvg >= environment.successThreshold) {
            break;
          }
        }
      }
      
      const endTime = performance.now();
      const wallClockTime = endTime - startTime;
      
      // Final evaluation
      const finalEpisodeResult = await this.runEpisode(algorithm, env, true);
      
      const result: BenchmarkResult = {
        algorithmId,
        environmentId,
        episodeReward: totalReward / episodeCount,
        episodeLength: trainingSteps / episodeCount,
        trainingSteps,
        wallClockTime,
        sampleEfficiency: convergenceEpisode === -1 ? 0 : trainingSteps / convergenceEpisode,
        convergenceEpisode,
        finalSuccess: finalEpisodeResult.reward >= environment.successThreshold,
        hyperparameters: config || {},
        metadata: {
          timestamp: new Date(),
          version: '1.0.0',
          seed
        }
      };
      
      results.push(result);
      this.results.push(result);
    }
    
    return results;
  }

  private async resetAlgorithm(algorithm: any, config: any, seed: number): Promise<void> {
    // Reset algorithm state
    if (algorithm.reset) {
      await algorithm.reset(config, seed);
    }
  }

  private async createEnvironment(environmentId: string, seed: number): Promise<any> {
    // Create and return environment instance
    // This would integrate with actual environment implementations
    return {
      reset: () => this.generateRandomState(environmentId),
      step: (action: any) => this.simulateEnvironmentStep(environmentId, action),
      seed: () => {}
    };
  }

  private generateRandomState(environmentId: string): tf.Tensor {
    const env = this.environments.get(environmentId)!;
    return tf.randomNormal([1, env.stateSize]);
  }

  private simulateEnvironmentStep(environmentId: string, action: any): {
    nextState: tf.Tensor;
    reward: number;
    done: boolean;
    info: any;
  } {
    const env = this.environments.get(environmentId)!;
    
    // Simple simulation for demonstration
    const reward = Math.random() * (env.rewardRange[1] - env.rewardRange[0]) + env.rewardRange[0];
    const done = Math.random() < 0.02; // 2% chance of episode ending
    
    return {
      nextState: this.generateRandomState(environmentId),
      reward,
      done,
      info: {}
    };
  }

  private async runEpisode(algorithm: any, env: any, evaluate: boolean = false): Promise<{
    reward: number;
    steps: number;
  }> {
    let state = env.reset();
    let totalReward = 0;
    let steps = 0;
    
    while (steps < 1000) { // Max episode length
      const action = await algorithm.selectAction(state, evaluate);
      const stepResult = env.step(action);
      
      if (!evaluate) {
        await algorithm.update(state, action, stepResult.reward, stepResult.nextState, stepResult.done);
      }
      
      totalReward += stepResult.reward;
      steps++;
      
      if (stepResult.done) {
        break;
      }
      
      state = stepResult.nextState;
    }
    
    // Clean up tensors
    state.dispose();
    
    return { reward: totalReward, steps };
  }

  private async getRecentAverageReward(algorithm: any, episodes: number): Promise<number> {
    // Get recent performance from algorithm
    if (algorithm.getRecentPerformance) {
      return algorithm.getRecentPerformance(episodes);
    }
    return 0;
  }

  analyzeResults(algorithmIds: string[], environmentId: string): BenchmarkComparison {
    const envResults = this.results.filter(r => r.environmentId === environmentId);
    const algorithmResults = new Map<string, BenchmarkMetrics>();
    
    for (const algId of algorithmIds) {
      const algResults = envResults.filter(r => r.algorithmId === algId);
      if (algResults.length === 0) continue;
      
      const rewards = algResults.map(r => r.episodeReward);
      const metrics = this.computeMetrics(rewards);
      algorithmResults.set(algId, metrics);
    }
    
    // Compute rankings
    const rankings = Array.from(algorithmResults.entries())
      .map(([algId, metrics]) => ({
        algorithmId: algId,
        score: metrics.mean,
        rank: 0
      }))
      .sort((a, b) => b.score - a.score)
      .map((item, index) => ({ ...item, rank: index + 1 }));
    
    // Statistical tests (simplified)
    const statisticalTests = new Map<string, any>();
    for (let i = 0; i < algorithmIds.length; i++) {
      for (let j = i + 1; j < algorithmIds.length; j++) {
        const alg1 = algorithmIds[i];
        const alg2 = algorithmIds[j];
        const comparison = this.performStatisticalTest(alg1, alg2, environmentId);
        statisticalTests.set(`${alg1}_vs_${alg2}`, comparison);
      }
    }
    
    return {
      environmentId,
      results: algorithmResults,
      rankings,
      statisticalTests
    };
  }

  private computeMetrics(values: number[]): BenchmarkMetrics {
    if (values.length === 0) {
      return { mean: 0, std: 0, min: 0, max: 0, median: 0, q25: 0, q75: 0, iqr: 0 };
    }
    
    const sorted = [...values].sort((a, b) => a - b);
    const n = values.length;
    
    const mean = values.reduce((a, b) => a + b) / n;
    const variance = values.reduce((acc, val) => acc + Math.pow(val - mean, 2), 0) / n;
    const std = Math.sqrt(variance);
    
    const min = sorted[0];
    const max = sorted[n - 1];
    const median = n % 2 === 0 ? 
      (sorted[n / 2 - 1] + sorted[n / 2]) / 2 : 
      sorted[Math.floor(n / 2)];
    
    const q25 = sorted[Math.floor(n * 0.25)];
    const q75 = sorted[Math.floor(n * 0.75)];
    const iqr = q75 - q25;
    
    return { mean, std, min, max, median, q25, q75, iqr };
  }

  private performStatisticalTest(alg1: string, alg2: string, environmentId: string): {
    pValue: number;
    significant: boolean;
    effectSize: number;
  } {
    const results1 = this.results.filter(r => r.algorithmId === alg1 && r.environmentId === environmentId);
    const results2 = this.results.filter(r => r.algorithmId === alg2 && r.environmentId === environmentId);
    
    if (results1.length === 0 || results2.length === 0) {
      return { pValue: 1.0, significant: false, effectSize: 0 };
    }
    
    const rewards1 = results1.map(r => r.episodeReward);
    const rewards2 = results2.map(r => r.episodeReward);
    
    // Simplified Mann-Whitney U test (Wilcoxon rank-sum test)
    const { pValue, effectSize } = this.mannWhitneyU(rewards1, rewards2);
    
    return {
      pValue,
      significant: pValue < 0.05,
      effectSize
    };
  }

  private mannWhitneyU(sample1: number[], sample2: number[]): { pValue: number; effectSize: number } {
    // Simplified implementation
    const n1 = sample1.length;
    const n2 = sample2.length;
    
    // Combine and rank all values
    const combined = [...sample1.map((v, i) => ({ value: v, group: 1, index: i })),
                     ...sample2.map((v, i) => ({ value: v, group: 2, index: i }))];
    combined.sort((a, b) => a.value - b.value);
    
    // Assign ranks
    let rankSum1 = 0;
    for (let i = 0; i < combined.length; i++) {
      if (combined[i].group === 1) {
        rankSum1 += i + 1;
      }
    }
    
    const U1 = rankSum1 - (n1 * (n1 + 1)) / 2;
    const U2 = n1 * n2 - U1;
    const U = Math.min(U1, U2);
    
    // Approximate p-value using normal approximation
    const muU = (n1 * n2) / 2;
    const sigmaU = Math.sqrt((n1 * n2 * (n1 + n2 + 1)) / 12);
    const z = Math.abs(U - muU) / sigmaU;
    const pValue = 2 * (1 - this.normalCDF(z));
    
    // Cohen's d effect size
    const mean1 = sample1.reduce((a, b) => a + b) / n1;
    const mean2 = sample2.reduce((a, b) => a + b) / n2;
    const pooledStd = Math.sqrt(
      ((n1 - 1) * this.variance(sample1) + (n2 - 1) * this.variance(sample2)) / (n1 + n2 - 2)
    );
    const effectSize = Math.abs(mean1 - mean2) / pooledStd;
    
    return { pValue, effectSize };
  }

  private normalCDF(x: number): number {
    // Approximation of the cumulative distribution function of the standard normal distribution
    const t = 1 / (1 + 0.2316419 * Math.abs(x));
    const d = 0.3989423 * Math.exp(-x * x / 2);
    let prob = d * t * (0.3193815 + t * (-0.3565638 + t * (1.781478 + t * (-1.821256 + t * 1.330274))));
    
    if (x > 0) {
      prob = 1 - prob;
    }
    
    return prob;
  }

  private variance(values: number[]): number {
    const mean = values.reduce((a, b) => a + b) / values.length;
    return values.reduce((acc, val) => acc + Math.pow(val - mean, 2), 0) / values.length;
  }

  exportResults(): any {
    return {
      environments: Array.from(this.environments.values()),
      results: this.results,
      summary: this.generateSummary()
    };
  }

  private generateSummary(): any {
    const algorithmStats = new Map<string, any>();
    const environmentStats = new Map<string, any>();
    
    // Aggregate by algorithm
    for (const result of this.results) {
      if (!algorithmStats.has(result.algorithmId)) {
        algorithmStats.set(result.algorithmId, {
          totalRuns: 0,
          avgReward: 0,
          avgSampleEfficiency: 0,
          successRate: 0,
          environments: new Set()
        });
      }
      
      const stats = algorithmStats.get(result.algorithmId)!;
      stats.totalRuns++;
      stats.avgReward = (stats.avgReward * (stats.totalRuns - 1) + result.episodeReward) / stats.totalRuns;
      stats.avgSampleEfficiency = (stats.avgSampleEfficiency * (stats.totalRuns - 1) + result.sampleEfficiency) / stats.totalRuns;
      stats.successRate = (stats.successRate * (stats.totalRuns - 1) + (result.finalSuccess ? 1 : 0)) / stats.totalRuns;
      stats.environments.add(result.environmentId);
    }
    
    // Aggregate by environment
    for (const result of this.results) {
      if (!environmentStats.has(result.environmentId)) {
        environmentStats.set(result.environmentId, {
          totalRuns: 0,
          avgReward: 0,
          avgConvergenceEpisode: 0,
          successRate: 0,
          algorithms: new Set()
        });
      }
      
      const stats = environmentStats.get(result.environmentId)!;
      stats.totalRuns++;
      stats.avgReward = (stats.avgReward * (stats.totalRuns - 1) + result.episodeReward) / stats.totalRuns;
      stats.avgConvergenceEpisode = (stats.avgConvergenceEpisode * (stats.totalRuns - 1) + 
        (result.convergenceEpisode === -1 ? 1000 : result.convergenceEpisode)) / stats.totalRuns;
      stats.successRate = (stats.successRate * (stats.totalRuns - 1) + (result.finalSuccess ? 1 : 0)) / stats.totalRuns;
      stats.algorithms.add(result.algorithmId);
    }
    
    return {
      totalRuns: this.results.length,
      uniqueAlgorithms: algorithmStats.size,
      uniqueEnvironments: environmentStats.size,
      algorithmStats: Object.fromEntries(
        Array.from(algorithmStats.entries()).map(([k, v]) => [k, {
          ...v,
          environments: Array.from(v.environments)
        }])
      ),
      environmentStats: Object.fromEntries(
        Array.from(environmentStats.entries()).map(([k, v]) => [k, {
          ...v,
          algorithms: Array.from(v.algorithms)
        }])
      )
    };
  }

  clearResults(): void {
    this.results = [];
  }

  getLeaderboard(environmentId?: string): Array<{
    algorithmId: string;
    score: number;
    rank: number;
    environmentCount: number;
  }> {
    let filteredResults = this.results;
    
    if (environmentId) {
      filteredResults = this.results.filter(r => r.environmentId === environmentId);
    }
    
    const algorithmScores = new Map<string, { scores: number[]; envCount: number }>();
    
    for (const result of filteredResults) {
      if (!algorithmScores.has(result.algorithmId)) {
        algorithmScores.set(result.algorithmId, { scores: [], envCount: 0 });
      }
      
      const entry = algorithmScores.get(result.algorithmId)!;
      entry.scores.push(result.episodeReward);
      
      if (!environmentId) {
        entry.envCount = new Set(
          filteredResults
            .filter(r => r.algorithmId === result.algorithmId)
            .map(r => r.environmentId)
        ).size;
      } else {
        entry.envCount = 1;
      }
    }
    
    const leaderboard = Array.from(algorithmScores.entries())
      .map(([algId, data]) => ({
        algorithmId: algId,
        score: data.scores.reduce((a, b) => a + b) / data.scores.length,
        rank: 0,
        environmentCount: data.envCount
      }))
      .sort((a, b) => b.score - a.score)
      .map((item, index) => ({ ...item, rank: index + 1 }));
    
    return leaderboard;
  }
}