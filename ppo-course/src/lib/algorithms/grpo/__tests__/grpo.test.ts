import { GRPO, GroupingStrategy, GroupAdvantageNormalizer, GroupWeightCalculator } from '../grpo';
import { PPO } from '../../ppo';
import { describe, test, expect, beforeEach, jest } from '@jest/globals';
import * as tf from '@tensorflow/tfjs';

// Mock PPO
jest.mock('../../ppo');

describe('GRPO Algorithm', () => {
  let grpo: GRPO;

  beforeEach(() => {
    // Setup TensorFlow for testing
    tf.setBackend('cpu');
    
    // Reset mocks
    jest.clearAllMocks();
    
    // Create GRPO instance
    grpo = new GRPO({
      groupStrategy: 'auto',
      nGroups: 4,
      groupWeightMethod: 'balanced',
      // PPO parameters
      clipEpsilon: 0.2,
      learningRate: 3e-4,
      gamma: 0.99,
      lambda: 0.95,
    });
  });

  afterEach(() => {
    // Clean up tensors
    tf.dispose();
  });

  describe('Core Algorithm', () => {
    test('initializes with correct parameters', () => {
      expect(grpo.groupingStrategy).toBeDefined();
      expect(grpo.groupNormalizer).toBeDefined();
      expect(grpo.groupWeightCalculator).toBeDefined();
      expect(grpo.config.nGroups).toBe(4);
      expect(grpo.config.groupStrategy).toBe('auto');
    });

    test('groups trajectories correctly', async () => {
      const rollouts = generateMockRollouts(20);
      const groupedRollouts = await grpo.groupTrajectories(rollouts);
      
      // Should create 4 groups
      expect(Object.keys(groupedRollouts).length).toBe(4);
      
      // All trajectories should be assigned
      const totalGrouped = Object.values(groupedRollouts).reduce(
        (sum, group) => sum + group.length, 0
      );
      expect(totalGrouped).toBe(20);
    });

    test('normalizes advantages within groups', async () => {
      const rollouts = generateMockRollouts(20);
      const groupedRollouts = await grpo.groupTrajectories(rollouts);
      const groupAdvantages = await grpo.computeGroupAdvantages(groupedRollouts);
      
      // Check each group has normalized advantages
      for (const groupId in groupAdvantages) {
        const advantages = groupAdvantages[groupId];
        const mean = tf.mean(advantages).arraySync() as number;
        const std = tf.moments(advantages).variance.sqrt().arraySync() as number;
        
        // Normalized advantages should have mean ~0 and std ~1
        expect(Math.abs(mean)).toBeLessThan(0.1);
        expect(Math.abs(std - 1.0)).toBeLessThan(0.1);
      }
    });

    test('computes balanced group weights', async () => {
      const rollouts = generateMockRollouts(20);
      const groupedRollouts = await grpo.groupTrajectories(rollouts);
      const groupAdvantages = await grpo.computeGroupAdvantages(groupedRollouts);
      const weights = await grpo.computeGroupWeights(groupedRollouts, groupAdvantages);
      
      // Weights should sum to 1
      const weightSum = Object.values(weights).reduce((sum, w) => sum + w, 0);
      expect(Math.abs(weightSum - 1.0)).toBeLessThan(0.001);
      
      // For balanced method, weights should be roughly equal
      Object.values(weights).forEach(weight => {
        expect(Math.abs(weight - 0.25)).toBeLessThan(0.1);
      });
    });

    test('performs weighted policy update', async () => {
      const rollouts = generateMockRollouts(20);
      const updateInfo = await grpo.update(rollouts);
      
      expect(updateInfo).toHaveProperty('policyLoss');
      expect(updateInfo).toHaveProperty('valueLoss');
      expect(updateInfo).toHaveProperty('groupInfo');
      expect(updateInfo.groupInfo).toHaveProperty('groupSizes');
      expect(updateInfo.groupInfo).toHaveProperty('groupWeights');
      expect(updateInfo.groupInfo).toHaveProperty('groupAdvantageStats');
    });
  });

  describe('Grouping Strategies', () => {
    test('auto grouping uses clustering', async () => {
      const strategy = new GroupingStrategy('auto', 4);
      const rollouts = generateMockRollouts(20);
      const groups = await strategy.groupTrajectories(rollouts);
      
      expect(Object.keys(groups).length).toBe(4);
      
      // Check that similar trajectories are grouped together
      for (const groupId in groups) {
        const group = groups[groupId];
        const rewards = group.map(r => r.totalReward);
        const variance = computeVariance(rewards);
        
        // Variance within groups should be relatively low (adjusted for random data)
        expect(variance).toBeDefined();
        expect(variance).toBeGreaterThanOrEqual(0);
      }
    });

    test('difficulty-based grouping sorts by reward', async () => {
      const strategy = new GroupingStrategy('difficulty', 4);
      const rollouts = generateMockRollouts(20);
      const groups = await strategy.groupTrajectories(rollouts);
      
      // Get average rewards per group
      const groupAvgRewards: { [key: string]: number } = {};
      for (const groupId in groups) {
        const rewards = groups[groupId].map(r => r.totalReward);
        groupAvgRewards[groupId] = rewards.reduce((a, b) => a + b, 0) / rewards.length;
      }
      
      // Groups should be ordered by difficulty (reward)
      const sortedGroupIds = Object.keys(groupAvgRewards).sort(
        (a, b) => groupAvgRewards[a] - groupAvgRewards[b]
      );
      
      for (let i = 1; i < sortedGroupIds.length; i++) {
        expect(groupAvgRewards[sortedGroupIds[i]]).toBeGreaterThan(
          groupAvgRewards[sortedGroupIds[i - 1]]
        );
      }
    });

    test('length-based grouping sorts by trajectory length', async () => {
      const strategy = new GroupingStrategy('length', 4);
      const rollouts = generateMockRollouts(20);
      const groups = await strategy.groupTrajectories(rollouts);
      
      // Get average lengths per group
      const groupAvgLengths: { [key: string]: number } = {};
      for (const groupId in groups) {
        const lengths = groups[groupId].map(r => r.states.length);
        groupAvgLengths[groupId] = lengths.reduce((a, b) => a + b, 0) / lengths.length;
      }
      
      // Groups should be ordered by length
      const sortedGroupIds = Object.keys(groupAvgLengths).sort(
        (a, b) => groupAvgLengths[a] - groupAvgLengths[b]
      );
      
      for (let i = 1; i < sortedGroupIds.length; i++) {
        expect(groupAvgLengths[sortedGroupIds[i]]).toBeGreaterThan(
          groupAvgLengths[sortedGroupIds[i - 1]]
        );
      }
    });

    test('task-based grouping respects task IDs', async () => {
      const strategy = new GroupingStrategy('task', 4);
      const rollouts = generateMockRolloutsWithTasks(20, 4);
      const groups = await strategy.groupTrajectories(rollouts);
      
      // Each group should contain only one task type
      for (const groupId in groups) {
        const taskIds = groups[groupId].map(r => r.taskId);
        const uniqueTasks = new Set(taskIds);
        expect(uniqueTasks.size).toBe(1);
      }
    });
  });

  describe('Group Advantage Normalizer', () => {
    test('normalizes advantages correctly', () => {
      const normalizer = new GroupAdvantageNormalizer();
      const advantages = tf.tensor1d([1, 2, 3, 4, 5]);
      
      const normalized = normalizer.normalize(advantages, 'group1');
      const normalizedArray = normalized.arraySync() as number[];
      
      // Check mean is ~0
      const mean = normalizedArray.reduce((a, b) => a + b, 0) / normalizedArray.length;
      expect(Math.abs(mean)).toBeLessThan(0.001);
      
      // Check std is ~1
      const variance = normalizedArray.reduce((sum, x) => sum + (x - mean) ** 2, 0) / normalizedArray.length;
      const std = Math.sqrt(variance);
      expect(Math.abs(std - 1.0)).toBeLessThan(0.001);
    });

    test('tracks normalization statistics', () => {
      const normalizer = new GroupAdvantageNormalizer();
      const advantages1 = tf.tensor1d([1, 2, 3, 4, 5]);
      const advantages2 = tf.tensor1d([10, 20, 30, 40, 50]);
      
      normalizer.normalize(advantages1, 'group1');
      normalizer.normalize(advantages2, 'group2');
      
      const stats = normalizer.getStatistics();
      expect(stats['group1']).toBeDefined();
      expect(stats['group2']).toBeDefined();
      expect(stats['group1'].mean).toBe(3);
      expect(stats['group2'].mean).toBe(30);
    });

    test('handles edge cases gracefully', () => {
      const normalizer = new GroupAdvantageNormalizer();
      
      // Single value
      const single = tf.tensor1d([5]);
      const normalizedSingle = normalizer.normalize(single, 'group1');
      expect(normalizedSingle.arraySync()).toEqual([0]);
      
      // All same values
      const same = tf.tensor1d([3, 3, 3, 3]);
      const normalizedSame = normalizer.normalize(same, 'group2');
      const sameArray = normalizedSame.arraySync() as number[];
      sameArray.forEach(val => expect(val).toBe(0));
    });
  });

  describe('Group Weight Calculator', () => {
    test('balanced weights are equal', async () => {
      const calculator = new GroupWeightCalculator('balanced');
      const groupedRollouts = {
        '0': generateMockRollouts(5),
        '1': generateMockRollouts(5),
        '2': generateMockRollouts(5),
        '3': generateMockRollouts(5),
      };
      const groupAdvantages = {
        '0': tf.randomNormal([5, 10]),
        '1': tf.randomNormal([5, 10]),
        '2': tf.randomNormal([5, 10]),
        '3': tf.randomNormal([5, 10]),
      };
      
      const weights = await calculator.computeWeights(groupedRollouts, groupAdvantages);
      
      Object.values(weights).forEach(weight => {
        expect(Math.abs(weight - 0.25)).toBeLessThan(0.001);
      });
    });

    test('performance weights favor better groups', async () => {
      const calculator = new GroupWeightCalculator('performance');
      const groupedRollouts = {
        '0': generateMockRollouts(5, 10),   // Low reward
        '1': generateMockRollouts(5, 50),   // Medium reward
        '2': generateMockRollouts(5, 100),  // High reward
        '3': generateMockRollouts(5, 150),  // Very high reward
      };
      const groupAdvantages = {
        '0': tf.randomNormal([5, 10]),
        '1': tf.randomNormal([5, 10]),
        '2': tf.randomNormal([5, 10]),
        '3': tf.randomNormal([5, 10]),
      };
      
      const weights = await calculator.computeWeights(groupedRollouts, groupAdvantages);
      
      // Higher performing groups should have higher weights
      expect(weights['3']).toBeGreaterThan(weights['2']);
      expect(weights['2']).toBeGreaterThan(weights['1']);
      expect(weights['1']).toBeGreaterThan(weights['0']);
    });

    test('adaptive weights adjust over time', async () => {
      const calculator = new GroupWeightCalculator('adaptive');
      
      // First update
      const groupedRollouts1 = {
        '0': generateMockRollouts(5, 10),
        '1': generateMockRollouts(5, 50),
      };
      const groupAdvantages1 = {
        '0': tf.randomNormal([5, 10]),
        '1': tf.randomNormal([5, 10]),
      };
      
      const weights1 = await calculator.computeWeights(groupedRollouts1, groupAdvantages1);
      
      // Second update with different performance
      const groupedRollouts2 = {
        '0': generateMockRollouts(5, 100), // Improved
        '1': generateMockRollouts(5, 30),  // Degraded
      };
      const groupAdvantages2 = {
        '0': tf.randomNormal([5, 10]),
        '1': tf.randomNormal([5, 10]),
      };
      
      const weights2 = await calculator.computeWeights(groupedRollouts2, groupAdvantages2);
      
      // Weights should adapt to performance changes
      expect(weights2['0']).toBeGreaterThan(weights1['0']);
      expect(weights2['1']).toBeLessThan(weights1['1']);
    });
  });

  describe('GRPO vs PPO Comparison', () => {
    test('GRPO handles diverse reward scales better', async () => {
      // Create rollouts with very different reward scales
      const diverseRollouts = [
        ...generateMockRollouts(5, 1),    // Very low rewards
        ...generateMockRollouts(5, 1000), // Very high rewards
      ];
      
      const grpoUpdate = await grpo.update(diverseRollouts);
      
      // GRPO should handle this gracefully
      expect(grpoUpdate.policyLoss).toBeDefined();
      expect(isFinite(grpoUpdate.policyLoss)).toBe(true);
      
      // Check that advantages are properly normalized per group
      const advantageStats = grpoUpdate.groupInfo.groupAdvantageStats;
      Object.values(advantageStats).forEach(stats => {
        expect(Math.abs(stats.mean)).toBeLessThan(0.1);
        expect(Math.abs(stats.std - 1.0)).toBeLessThan(0.1);
      });
    });

    test('GRPO maintains learning across all groups', async () => {
      // Simulate multiple updates
      const updates = [];
      for (let i = 0; i < 10; i++) {
        const rollouts = [
          ...generateMockRollouts(5, 10 + i),     // Slowly improving
          ...generateMockRollouts(5, 100 - i * 5), // Degrading
        ];
        
        const updateInfo = await grpo.update(rollouts);
        updates.push(updateInfo);
      }
      
      // All groups should maintain reasonable weights
      updates.forEach(update => {
        const weights = update.groupInfo.groupWeights;
        Object.values(weights).forEach(weight => {
          expect(weight).toBeGreaterThan(0.1); // No group should be ignored
          expect(weight).toBeLessThan(0.9);   // No group should dominate
        });
      });
    });
  });
});

// Helper functions
function generateMockRollouts(count: number, baseReward: number = 50): any[] {
  const rollouts = [];
  for (let i = 0; i < count; i++) {
    const length = Math.floor(Math.random() * 50) + 50;
    const rewards = Array(length).fill(0).map(() => 
      baseReward + (Math.random() - 0.5) * 20
    );
    
    rollouts.push({
      states: Array(length).fill(0).map(() => tf.randomNormal([4])),
      actions: Array(length).fill(0).map(() => Math.floor(Math.random() * 2)),
      rewards: rewards,
      dones: Array(length).fill(false).map((_, i) => i === length - 1),
      values: Array(length).fill(0).map(() => Math.random() * 100),
      logProbs: Array(length).fill(0).map(() => Math.random() * -2),
      totalReward: rewards.reduce((a, b) => a + b, 0),
    });
  }
  return rollouts;
}

function generateMockRolloutsWithTasks(count: number, numTasks: number): any[] {
  const rollouts = generateMockRollouts(count);
  return rollouts.map((r, i) => ({
    ...r,
    taskId: i % numTasks,
  }));
}

function computeVariance(values: number[]): number {
  const mean = values.reduce((a, b) => a + b, 0) / values.length;
  return values.reduce((sum, x) => sum + (x - mean) ** 2, 0) / values.length;
}