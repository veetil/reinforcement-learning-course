import * as tf from '@tensorflow/tfjs';
import { PPO } from '../ppo';
import { KMeans } from './clustering';

export interface GRPOConfig {
  groupStrategy: 'auto' | 'task' | 'difficulty' | 'length';
  nGroups: number;
  groupWeightMethod: 'balanced' | 'performance' | 'adaptive';
  // PPO base parameters
  clipEpsilon: number;
  learningRate: number;
  gamma: number;
  lambda: number;
  [key: string]: any;
}

export interface GroupedRollouts {
  [groupId: string]: Rollout[];
}

export interface Rollout {
  states: tf.Tensor[];
  actions: number[];
  rewards: number[];
  dones: boolean[];
  values: number[];
  logProbs: number[];
  totalReward: number;
  taskId?: number;
}

export interface GroupAdvantages {
  [groupId: string]: tf.Tensor;
}

export interface GroupWeights {
  [groupId: string]: number;
}

export interface UpdateInfo {
  policyLoss: number;
  valueLoss: number;
  groupInfo: {
    groupSizes: { [groupId: string]: number };
    groupWeights: GroupWeights;
    groupAdvantageStats: {
      [groupId: string]: {
        mean: number;
        std: number;
      };
    };
  };
}

export class GRPO {
  public groupingStrategy: GroupingStrategy;
  public groupNormalizer: GroupAdvantageNormalizer;
  public groupWeightCalculator: GroupWeightCalculator;
  public basePPO: PPO;
  public config: GRPOConfig;

  constructor(config: GRPOConfig) {
    this.config = config;
    this.groupingStrategy = new GroupingStrategy(config.groupStrategy, config.nGroups);
    this.groupNormalizer = new GroupAdvantageNormalizer();
    this.groupWeightCalculator = new GroupWeightCalculator(config.groupWeightMethod);
    
    // Extract PPO parameters
    const { groupStrategy, nGroups, groupWeightMethod, ...ppoConfig } = config;
    this.basePPO = new PPO(ppoConfig);
  }

  async groupTrajectories(rollouts: Rollout[]): Promise<GroupedRollouts> {
    return this.groupingStrategy.groupTrajectories(rollouts);
  }

  async computeGroupAdvantages(groupedRollouts: GroupedRollouts): Promise<GroupAdvantages> {
    const groupAdvantages: GroupAdvantages = {};
    
    for (const [groupId, groupData] of Object.entries(groupedRollouts)) {
      // Compute GAE for this group
      const advantages = await this.computeGAE(groupData);
      
      // Normalize within group
      groupAdvantages[groupId] = this.groupNormalizer.normalize(advantages, groupId);
    }
    
    return groupAdvantages;
  }

  async computeGroupWeights(
    groupedRollouts: GroupedRollouts,
    groupAdvantages: GroupAdvantages
  ): Promise<GroupWeights> {
    return this.groupWeightCalculator.computeWeights(groupedRollouts, groupAdvantages);
  }

  async update(rollouts: Rollout[]): Promise<UpdateInfo> {
    // 1. Group trajectories
    const groupedRollouts = await this.groupTrajectories(rollouts);
    
    // 2. Compute advantages per group
    const groupAdvantages = await this.computeGroupAdvantages(groupedRollouts);
    
    // 3. Compute group weights
    const groupWeights = await this.computeGroupWeights(groupedRollouts, groupAdvantages);
    
    // 4. Weighted policy update
    let totalPolicyLoss = 0;
    let totalValueLoss = 0;
    const groupAdvantageStats: any = {};
    
    for (const [groupId, weight] of Object.entries(groupWeights)) {
      const groupLoss = await this.computeGroupPolicyLoss(
        groupedRollouts[groupId],
        groupAdvantages[groupId]
      );
      
      totalPolicyLoss += weight * groupLoss.policyLoss;
      totalValueLoss += weight * groupLoss.valueLoss;
      
      // Collect statistics
      const advantages = groupAdvantages[groupId];
      const mean = tf.mean(advantages).arraySync() as number;
      const variance = tf.moments(advantages).variance;
      const std = variance.sqrt().arraySync() as number;
      
      groupAdvantageStats[groupId] = { mean, std };
    }
    
    // 5. Return update info
    const groupSizes: { [groupId: string]: number } = {};
    for (const [groupId, group] of Object.entries(groupedRollouts)) {
      groupSizes[groupId] = group.length;
    }
    
    return {
      policyLoss: totalPolicyLoss,
      valueLoss: totalValueLoss,
      groupInfo: {
        groupSizes,
        groupWeights,
        groupAdvantageStats,
      },
    };
  }

  private async computeGAE(rollouts: Rollout[]): Promise<tf.Tensor> {
    // Compute Generalized Advantage Estimation
    const allAdvantages: number[] = [];
    
    for (const rollout of rollouts) {
      const advantages = [];
      let gae = 0;
      
      // Backward pass through trajectory
      for (let t = rollout.rewards.length - 1; t >= 0; t--) {
        const delta = rollout.rewards[t] + 
          (rollout.dones[t] ? 0 : this.config.gamma * (rollout.values[t + 1] || 0)) - 
          rollout.values[t];
        
        gae = delta + (rollout.dones[t] ? 0 : this.config.gamma * this.config.lambda * gae);
        advantages.unshift(gae);
      }
      
      allAdvantages.push(...advantages);
    }
    
    return tf.tensor1d(allAdvantages);
  }

  private async computeGroupPolicyLoss(
    rollouts: Rollout[],
    advantages: tf.Tensor
  ): Promise<{ policyLoss: number; valueLoss: number }> {
    // Simplified loss computation
    // In practice, this would use the actual policy network
    
    // Mock policy loss
    const policyLoss = tf.mean(tf.square(advantages)).arraySync() as number;
    
    // Mock value loss
    const valueLoss = Math.random() * 0.1;
    
    return { policyLoss, valueLoss };
  }
}

export class GroupingStrategy {
  constructor(
    private strategyType: string,
    private nGroups: number
  ) {}

  async groupTrajectories(rollouts: Rollout[]): Promise<GroupedRollouts> {
    switch (this.strategyType) {
      case 'auto':
        return this.autoGroup(rollouts);
      case 'task':
        return this.taskBasedGroup(rollouts);
      case 'difficulty':
        return this.difficultyBasedGroup(rollouts);
      case 'length':
        return this.lengthBasedGroup(rollouts);
      default:
        return this.autoGroup(rollouts);
    }
  }

  private async autoGroup(rollouts: Rollout[]): Promise<GroupedRollouts> {
    // Extract features for clustering
    const features = this.extractTrajectoryFeatures(rollouts);
    
    // Perform K-means clustering
    const kmeans = new KMeans(this.nGroups);
    const clusters = await kmeans.fit(features);
    
    // Organize by clusters
    return this.organizeByCluster(rollouts, clusters);
  }

  private async taskBasedGroup(rollouts: Rollout[]): Promise<GroupedRollouts> {
    const groups: GroupedRollouts = {};
    
    for (const rollout of rollouts) {
      const taskId = rollout.taskId || 0;
      const groupId = taskId.toString();
      
      if (!groups[groupId]) {
        groups[groupId] = [];
      }
      groups[groupId].push(rollout);
    }
    
    return groups;
  }

  private async difficultyBasedGroup(rollouts: Rollout[]): Promise<GroupedRollouts> {
    // Sort by total reward
    const sortedRollouts = [...rollouts].sort((a, b) => a.totalReward - b.totalReward);
    
    // Create groups based on quartiles
    const groups: GroupedRollouts = {};
    const groupSize = Math.ceil(rollouts.length / this.nGroups);
    
    for (let i = 0; i < this.nGroups; i++) {
      const start = i * groupSize;
      const end = Math.min((i + 1) * groupSize, rollouts.length);
      groups[i.toString()] = sortedRollouts.slice(start, end);
    }
    
    return groups;
  }

  private async lengthBasedGroup(rollouts: Rollout[]): Promise<GroupedRollouts> {
    // Sort by trajectory length
    const sortedRollouts = [...rollouts].sort((a, b) => a.states.length - b.states.length);
    
    // Create groups based on length quartiles
    const groups: GroupedRollouts = {};
    const groupSize = Math.ceil(rollouts.length / this.nGroups);
    
    for (let i = 0; i < this.nGroups; i++) {
      const start = i * groupSize;
      const end = Math.min((i + 1) * groupSize, rollouts.length);
      groups[i.toString()] = sortedRollouts.slice(start, end);
    }
    
    return groups;
  }

  private extractTrajectoryFeatures(rollouts: Rollout[]): number[][] {
    return rollouts.map(rollout => [
      rollout.totalReward,
      rollout.states.length,
      Math.max(...rollout.rewards),
      Math.min(...rollout.rewards),
      rollout.rewards.reduce((a, b) => a + b, 0) / rollout.rewards.length,
    ]);
  }

  private organizeByCluster(rollouts: Rollout[], clusters: number[]): GroupedRollouts {
    const groups: GroupedRollouts = {};
    
    for (let i = 0; i < rollouts.length; i++) {
      const clusterId = clusters[i].toString();
      if (!groups[clusterId]) {
        groups[clusterId] = [];
      }
      groups[clusterId].push(rollouts[i]);
    }
    
    return groups;
  }
}

export class GroupAdvantageNormalizer {
  private statistics: Map<string, { mean: number; std: number }> = new Map();

  normalize(advantages: tf.Tensor, groupId: string): tf.Tensor {
    const mean = tf.mean(advantages);
    const variance = tf.moments(advantages).variance;
    const std = variance.sqrt();
    
    // Store statistics
    this.statistics.set(groupId, {
      mean: mean.arraySync() as number,
      std: std.arraySync() as number,
    });
    
    // Handle edge cases
    return tf.tidy(() => {
      const epsilon = 1e-8;
      const stdSafe = tf.maximum(std, epsilon);
      
      // Normalize: (x - mean) / std
      return advantages.sub(mean).div(stdSafe);
    });
  }

  getStatistics(): { [groupId: string]: { mean: number; std: number } } {
    const stats: any = {};
    this.statistics.forEach((value, key) => {
      stats[key] = value;
    });
    return stats;
  }
}

export class GroupWeightCalculator {
  private history: Map<string, number[]> = new Map();

  constructor(private method: string) {}

  async computeWeights(
    groupedRollouts: GroupedRollouts,
    groupAdvantages: GroupAdvantages
  ): Promise<GroupWeights> {
    switch (this.method) {
      case 'balanced':
        return this.computeBalancedWeights(groupedRollouts);
      case 'performance':
        return this.computePerformanceWeights(groupedRollouts);
      case 'adaptive':
        return this.computeAdaptiveWeights(groupedRollouts, groupAdvantages);
      default:
        return this.computeBalancedWeights(groupedRollouts);
    }
  }

  private async computeBalancedWeights(groupedRollouts: GroupedRollouts): Promise<GroupWeights> {
    const weights: GroupWeights = {};
    const numGroups = Object.keys(groupedRollouts).length;
    
    for (const groupId of Object.keys(groupedRollouts)) {
      weights[groupId] = 1.0 / numGroups;
    }
    
    return weights;
  }

  private async computePerformanceWeights(groupedRollouts: GroupedRollouts): Promise<GroupWeights> {
    const weights: GroupWeights = {};
    const groupPerformance: { [groupId: string]: number } = {};
    
    // Calculate average performance per group
    for (const [groupId, group] of Object.entries(groupedRollouts)) {
      const avgReward = group.reduce((sum, r) => sum + r.totalReward, 0) / group.length;
      groupPerformance[groupId] = avgReward;
    }
    
    // Convert to weights (softmax over performance)
    const performances = Object.values(groupPerformance);
    const maxPerf = Math.max(...performances);
    const expPerformances = performances.map(p => Math.exp((p - maxPerf) / 100));
    const sumExp = expPerformances.reduce((a, b) => a + b, 0);
    
    Object.keys(groupedRollouts).forEach((groupId, i) => {
      weights[groupId] = expPerformances[i] / sumExp;
    });
    
    return weights;
  }

  private async computeAdaptiveWeights(
    groupedRollouts: GroupedRollouts,
    groupAdvantages: GroupAdvantages
  ): Promise<GroupWeights> {
    const weights: GroupWeights = {};
    
    // Update history with current performance
    for (const [groupId, group] of Object.entries(groupedRollouts)) {
      const avgReward = group.reduce((sum, r) => sum + r.totalReward, 0) / group.length;
      
      if (!this.history.has(groupId)) {
        this.history.set(groupId, []);
      }
      
      const history = this.history.get(groupId)!;
      history.push(avgReward);
      
      // Keep only recent history
      if (history.length > 10) {
        history.shift();
      }
    }
    
    // Compute weights based on improvement rate
    const improvementRates: { [groupId: string]: number } = {};
    let totalImprovement = 0;
    
    for (const [groupId, history] of this.history.entries()) {
      if (history.length < 2) {
        improvementRates[groupId] = 1.0;
      } else {
        // Calculate improvement trend
        const recent = history.slice(-3).reduce((a, b) => a + b, 0) / Math.min(3, history.length);
        const older = history.slice(0, 3).reduce((a, b) => a + b, 0) / Math.min(3, history.length);
        improvementRates[groupId] = Math.max(0.1, recent / (older + 1e-8));
      }
      totalImprovement += improvementRates[groupId];
    }
    
    // Normalize to get weights
    for (const groupId of Object.keys(groupedRollouts)) {
      weights[groupId] = improvementRates[groupId] / totalImprovement;
    }
    
    return weights;
  }
}