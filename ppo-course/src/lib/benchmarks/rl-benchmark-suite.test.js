// Mock TensorFlow.js for testing
const tf = {
  randomNormal: (shape) => ({ shape, dispose: () => {}, dataSync: () => [Math.random()] }),
  tensor: (data) => ({ dataSync: () => data, dispose: () => {} })
};

// Mock algorithm for testing
class MockAlgorithm {
  constructor() {
    this.performance = [];
  }
  
  async reset(config, seed) {
    this.performance = [];
  }
  
  async selectAction(state, evaluate = false) {
    return { action: Math.floor(Math.random() * 4), dispose: () => {} };
  }
  
  async update(state, action, reward, nextState, done) {
    this.performance.push(reward);
  }
  
  getRecentPerformance(episodes) {
    const recent = this.performance.slice(-episodes);
    return recent.reduce((a, b) => a + b, 0) / recent.length;
  }
}

// Mock RLBenchmarkSuite
class RLBenchmarkSuite {
  constructor() {
    this.environments = new Map();
    this.results = [];
    this.algorithms = new Map();
    this.initializeEnvironments();
  }

  initializeEnvironments() {
    const envs = [
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
      }
    ];

    envs.forEach(env => this.environments.set(env.id, env));
  }

  registerAlgorithm(id, algorithm) {
    this.algorithms.set(id, algorithm);
  }

  getEnvironments() {
    return Array.from(this.environments.values());
  }

  getEnvironment(id) {
    return this.environments.get(id);
  }

  async runBenchmark(algorithmId, environmentId, numRuns = 2, maxEpisodes = 10) {
    const algorithm = this.algorithms.get(algorithmId);
    const environment = this.environments.get(environmentId);
    
    if (!algorithm || !environment) {
      throw new Error(`Algorithm ${algorithmId} or environment ${environmentId} not found`);
    }

    const results = [];
    
    for (let run = 0; run < numRuns; run++) {
      const startTime = performance.now();
      const seed = Math.floor(Math.random() * 10000);
      
      await algorithm.reset({}, seed);
      
      let totalReward = 0;
      let trainingSteps = 0;
      let convergenceEpisode = -1;
      
      // Training loop
      for (let episode = 0; episode < maxEpisodes; episode++) {
        const episodeReward = Math.random() * (environment.rewardRange[1] - environment.rewardRange[0]) + environment.rewardRange[0];
        totalReward += episodeReward;
        trainingSteps += Math.floor(Math.random() * 100) + 50;
        
        if (convergenceEpisode === -1 && episodeReward >= environment.successThreshold) {
          convergenceEpisode = episode;
        }
      }
      
      const endTime = performance.now();
      const wallClockTime = endTime - startTime;
      
      const result = {
        algorithmId,
        environmentId,
        episodeReward: totalReward / maxEpisodes,
        episodeLength: trainingSteps / maxEpisodes,
        trainingSteps,
        wallClockTime,
        sampleEfficiency: convergenceEpisode === -1 ? 0 : trainingSteps / convergenceEpisode,
        convergenceEpisode,
        finalSuccess: totalReward / maxEpisodes >= environment.successThreshold,
        hyperparameters: {},
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

  analyzeResults(algorithmIds, environmentId) {
    const envResults = this.results.filter(r => r.environmentId === environmentId);
    const algorithmResults = new Map();
    
    for (const algId of algorithmIds) {
      const algResults = envResults.filter(r => r.algorithmId === algId);
      if (algResults.length === 0) continue;
      
      const rewards = algResults.map(r => r.episodeReward);
      const metrics = this.computeMetrics(rewards);
      algorithmResults.set(algId, metrics);
    }
    
    const rankings = Array.from(algorithmResults.entries())
      .map(([algId, metrics]) => ({
        algorithmId: algId,
        score: metrics.mean,
        rank: 0
      }))
      .sort((a, b) => b.score - a.score)
      .map((item, index) => ({ ...item, rank: index + 1 }));
    
    return {
      environmentId,
      results: algorithmResults,
      rankings,
      statisticalTests: new Map()
    };
  }

  computeMetrics(values) {
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

  getLeaderboard(environmentId) {
    let filteredResults = this.results;
    
    if (environmentId) {
      filteredResults = this.results.filter(r => r.environmentId === environmentId);
    }
    
    const algorithmScores = new Map();
    
    for (const result of filteredResults) {
      if (!algorithmScores.has(result.algorithmId)) {
        algorithmScores.set(result.algorithmId, { scores: [], envCount: 0 });
      }
      
      const entry = algorithmScores.get(result.algorithmId);
      entry.scores.push(result.episodeReward);
      entry.envCount = 1;
    }
    
    return Array.from(algorithmScores.entries())
      .map(([algId, data]) => ({
        algorithmId: algId,
        score: data.scores.reduce((a, b) => a + b) / data.scores.length,
        rank: 0,
        environmentCount: data.envCount
      }))
      .sort((a, b) => b.score - a.score)
      .map((item, index) => ({ ...item, rank: index + 1 }));
  }

  exportResults() {
    return {
      environments: Array.from(this.environments.values()),
      results: this.results,
      summary: this.generateSummary()
    };
  }

  generateSummary() {
    return {
      totalRuns: this.results.length,
      uniqueAlgorithms: new Set(this.results.map(r => r.algorithmId)).size,
      uniqueEnvironments: new Set(this.results.map(r => r.environmentId)).size
    };
  }

  clearResults() {
    this.results = [];
  }
}

describe('RL Benchmark Suite', () => {
  let benchmarkSuite;
  let mockAlgorithm;

  beforeEach(() => {
    benchmarkSuite = new RLBenchmarkSuite();
    mockAlgorithm = new MockAlgorithm();
  });

  test('should initialize with predefined environments', () => {
    const environments = benchmarkSuite.getEnvironments();
    
    expect(environments.length).toBeGreaterThan(0);
    expect(environments.some(env => env.id === 'cartpole')).toBe(true);
    expect(environments.some(env => env.id === 'lunar_lander')).toBe(true);
  });

  test('should register and retrieve algorithms', () => {
    benchmarkSuite.registerAlgorithm('test_ppo', mockAlgorithm);
    
    expect(benchmarkSuite.algorithms.has('test_ppo')).toBe(true);
    expect(benchmarkSuite.algorithms.get('test_ppo')).toBe(mockAlgorithm);
  });

  test('should retrieve environment details', () => {
    const cartpole = benchmarkSuite.getEnvironment('cartpole');
    
    expect(cartpole).toBeDefined();
    expect(cartpole.name).toBe('CartPole-v1');
    expect(cartpole.stateSize).toBe(4);
    expect(cartpole.actionSize).toBe(2);
    expect(cartpole.category).toBe('control');
    expect(cartpole.difficulty).toBe('easy');
  });

  test('should run benchmark and collect results', async () => {
    benchmarkSuite.registerAlgorithm('test_ppo', mockAlgorithm);
    
    const results = await benchmarkSuite.runBenchmark('test_ppo', 'cartpole', 2, 5);
    
    expect(results).toHaveLength(2);
    expect(results[0]).toHaveProperty('algorithmId', 'test_ppo');
    expect(results[0]).toHaveProperty('environmentId', 'cartpole');
    expect(results[0]).toHaveProperty('episodeReward');
    expect(results[0]).toHaveProperty('trainingSteps');
    expect(results[0]).toHaveProperty('wallClockTime');
    expect(results[0]).toHaveProperty('finalSuccess');
  });

  test('should handle unknown algorithm or environment', async () => {
    await expect(
      benchmarkSuite.runBenchmark('unknown_alg', 'cartpole')
    ).rejects.toThrow('Algorithm unknown_alg or environment cartpole not found');
    
    benchmarkSuite.registerAlgorithm('test_ppo', mockAlgorithm);
    await expect(
      benchmarkSuite.runBenchmark('test_ppo', 'unknown_env')
    ).rejects.toThrow('Algorithm test_ppo or environment unknown_env not found');
  });

  test('should analyze results and compute metrics', async () => {
    benchmarkSuite.registerAlgorithm('ppo', mockAlgorithm);
    benchmarkSuite.registerAlgorithm('sac', mockAlgorithm);
    
    // Run benchmarks
    await benchmarkSuite.runBenchmark('ppo', 'cartpole', 2, 3);
    await benchmarkSuite.runBenchmark('sac', 'cartpole', 2, 3);
    
    const analysis = benchmarkSuite.analyzeResults(['ppo', 'sac'], 'cartpole');
    
    expect(analysis.environmentId).toBe('cartpole');
    expect(analysis.results.has('ppo')).toBe(true);
    expect(analysis.results.has('sac')).toBe(true);
    expect(analysis.rankings).toHaveLength(2);
    expect(analysis.rankings[0].rank).toBe(1);
    expect(analysis.rankings[1].rank).toBe(2);
  });

  test('should compute statistical metrics correctly', () => {
    const values = [10, 20, 30, 40, 50];
    const metrics = benchmarkSuite.computeMetrics(values);
    
    expect(metrics.mean).toBe(30);
    expect(metrics.min).toBe(10);
    expect(metrics.max).toBe(50);
    expect(metrics.median).toBe(30);
    expect(metrics.std).toBeCloseTo(14.14, 1);
  });

  test('should generate leaderboard', async () => {
    benchmarkSuite.registerAlgorithm('ppo', mockAlgorithm);
    benchmarkSuite.registerAlgorithm('sac', mockAlgorithm);
    
    await benchmarkSuite.runBenchmark('ppo', 'cartpole', 2, 3);
    await benchmarkSuite.runBenchmark('sac', 'cartpole', 2, 3);
    
    const leaderboard = benchmarkSuite.getLeaderboard('cartpole');
    
    expect(leaderboard).toHaveLength(2);
    expect(leaderboard[0].rank).toBe(1);
    expect(leaderboard[1].rank).toBe(2);
    expect(leaderboard[0]).toHaveProperty('algorithmId');
    expect(leaderboard[0]).toHaveProperty('score');
    expect(leaderboard[0]).toHaveProperty('environmentCount');
  });

  test('should export results', async () => {
    benchmarkSuite.registerAlgorithm('ppo', mockAlgorithm);
    await benchmarkSuite.runBenchmark('ppo', 'cartpole', 1, 2);
    
    const exported = benchmarkSuite.exportResults();
    
    expect(exported).toHaveProperty('environments');
    expect(exported).toHaveProperty('results');
    expect(exported).toHaveProperty('summary');
    expect(exported.summary.totalRuns).toBe(1);
    expect(exported.summary.uniqueAlgorithms).toBe(1);
    expect(exported.summary.uniqueEnvironments).toBe(1);
  });

  test('should clear results', async () => {
    benchmarkSuite.registerAlgorithm('ppo', mockAlgorithm);
    await benchmarkSuite.runBenchmark('ppo', 'cartpole', 1, 2);
    
    expect(benchmarkSuite.results.length).toBe(1);
    
    benchmarkSuite.clearResults();
    
    expect(benchmarkSuite.results.length).toBe(0);
  });

  test('should handle empty metrics computation', () => {
    const metrics = benchmarkSuite.computeMetrics([]);
    
    expect(metrics.mean).toBe(0);
    expect(metrics.std).toBe(0);
    expect(metrics.min).toBe(0);
    expect(metrics.max).toBe(0);
  });
});