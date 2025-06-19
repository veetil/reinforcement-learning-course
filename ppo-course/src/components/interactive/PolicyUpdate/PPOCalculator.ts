export interface PolicyParams {
  mean: number;
  std: number;
}

export interface UpdateStats {
  policyLoss: number;
  valueLoss: number;
  entropy: number;
  klDivergence: number;
  clipFraction: number;
}

export class PPOCalculator {
  calculateRatio(newLogProb: number, oldLogProb: number): number {
    return Math.exp(newLogProb - oldLogProb);
  }

  clipObjective(ratio: number, advantage: number, epsilon: number): number {
    const clippedRatio = advantage > 0 
      ? Math.min(ratio, 1 + epsilon)
      : Math.max(ratio, 1 - epsilon);
    
    if (advantage > 0) {
      return Math.min(ratio * advantage, clippedRatio * advantage);
    } else {
      return Math.max(ratio * advantage, clippedRatio * advantage);
    }
  }

  calculateKL(oldProbs: number[], newProbs: number[]): number {
    let kl = 0;
    for (let i = 0; i < oldProbs.length; i++) {
      if (oldProbs[i] > 0 && newProbs[i] > 0) {
        kl += oldProbs[i] * Math.log(oldProbs[i] / newProbs[i]);
      }
    }
    return kl;
  }

  calculateEntropy(probs: number[]): number {
    let entropy = 0;
    for (const p of probs) {
      if (p > 0) {
        entropy -= p * Math.log(p);
      }
    }
    return entropy;
  }

  computePolicyGradient(advantages: number[], logProbs: number[]): number[] {
    return advantages.map((adv, i) => adv * Math.exp(logProbs[i]));
  }

  updatePolicy(
    oldPolicy: PolicyParams,
    gradient: PolicyParams,
    learningRate: number
  ): PolicyParams {
    return {
      mean: oldPolicy.mean + learningRate * gradient.mean,
      std: oldPolicy.std + learningRate * gradient.std
    };
  }

  calculatePPOLoss(
    ratio: number,
    advantage: number,
    epsilon: number,
    valueError: number,
    entropy: number,
    c1: number = 0.5, // Value loss coefficient
    c2: number = 0.01  // Entropy coefficient
  ): number {
    const policyLoss = -this.clipObjective(ratio, advantage, epsilon);
    const valueLoss = c1 * valueError * valueError;
    const entropyBonus = -c2 * entropy;
    
    return policyLoss + valueLoss + entropyBonus;
  }

  sampleFromPolicy(policy: PolicyParams): number {
    // Box-Muller transform for normal distribution
    const u1 = Math.random();
    const u2 = Math.random();
    const z0 = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
    
    return policy.mean + policy.std * z0;
  }

  calculateLogProb(value: number, policy: PolicyParams): number {
    const diff = value - policy.mean;
    return -0.5 * Math.log(2 * Math.PI * policy.std * policy.std) 
           -0.5 * (diff * diff) / (policy.std * policy.std);
  }

  computeAdvantages(
    rewards: number[],
    values: number[],
    gamma: number = 0.99,
    lambda: number = 0.95
  ): number[] {
    const advantages: number[] = [];
    let gae = 0;

    for (let t = rewards.length - 1; t >= 0; t--) {
      const nextValue = t < rewards.length - 1 ? values[t + 1] : 0;
      const delta = rewards[t] + gamma * nextValue - values[t];
      gae = delta + gamma * lambda * gae;
      advantages.unshift(gae);
    }

    return this.normalizeAdvantages(advantages);
  }

  normalizeAdvantages(advantages: number[]): number[] {
    const mean = advantages.reduce((sum, a) => sum + a, 0) / advantages.length;
    const variance = advantages.reduce((sum, a) => sum + Math.pow(a - mean, 2), 0) / advantages.length;
    const std = Math.sqrt(variance + 1e-8);

    return advantages.map(a => (a - mean) / std);
  }

  checkEarlyStopping(klDivergence: number, klThreshold: number = 0.01): boolean {
    return klDivergence > klThreshold;
  }

  calculateClipFraction(
    ratios: number[],
    advantages: number[],
    epsilon: number
  ): number {
    let clippedCount = 0;
    
    for (let i = 0; i < ratios.length; i++) {
      const ratio = ratios[i];
      const advantage = advantages[i];
      
      if (advantage > 0 && ratio > 1 + epsilon) {
        clippedCount++;
      } else if (advantage < 0 && ratio < 1 - epsilon) {
        clippedCount++;
      }
    }
    
    return clippedCount / ratios.length;
  }

  generateTrajectory(
    policy: PolicyParams,
    steps: number
  ): { actions: number[]; rewards: number[]; values: number[] } {
    const actions: number[] = [];
    const rewards: number[] = [];
    const values: number[] = [];

    for (let i = 0; i < steps; i++) {
      const action = this.sampleFromPolicy(policy);
      actions.push(action);
      
      // Simple reward function: closer to 0 is better
      const reward = -Math.abs(action);
      rewards.push(reward);
      
      // Simple value estimate
      const value = -Math.abs(policy.mean);
      values.push(value);
    }

    return { actions, rewards, values };
  }
}