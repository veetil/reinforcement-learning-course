export interface State {
  x: number;
  y: number;
}

export interface ActionValues {
  up: number;
  down: number;
  left: number;
  right: number;
}

export type Action = keyof ActionValues;

export class AdvantageCalculator {
  calculateAdvantage(qValue: number, vValue: number): number {
    return qValue - vValue;
  }

  calculateQValues(
    state: State,
    nextStateValues: ActionValues,
    reward: number,
    gamma: number
  ): ActionValues {
    return {
      up: reward + gamma * nextStateValues.up,
      down: reward + gamma * nextStateValues.down,
      left: reward + gamma * nextStateValues.left,
      right: reward + gamma * nextStateValues.right
    };
  }

  getOptimalAction(qValues: ActionValues): Action {
    let bestAction: Action = 'up';
    let bestValue = qValues.up;

    for (const [action, value] of Object.entries(qValues) as [Action, number][]) {
      if (value > bestValue) {
        bestValue = value;
        bestAction = action;
      }
    }

    return bestAction;
  }

  calculateGAE(
    rewards: number[],
    values: number[],
    gamma: number,
    lambda: number
  ): number[] {
    const advantages: number[] = [];
    let gae = 0;

    // Work backwards through the trajectory
    for (let t = rewards.length - 1; t >= 0; t--) {
      const delta = rewards[t] + gamma * values[t + 1] - values[t];
      gae = delta + gamma * lambda * gae;
      advantages.unshift(gae);
    }

    return advantages;
  }

  calculateGAEWithDones(
    rewards: number[],
    values: number[],
    gamma: number,
    lambda: number,
    dones: boolean[]
  ): number[] {
    const advantages: number[] = [];
    let gae = 0;

    for (let t = rewards.length - 1; t >= 0; t--) {
      const nextValue = dones[t] ? 0 : values[t + 1];
      const delta = rewards[t] + gamma * nextValue - values[t];
      gae = delta + gamma * lambda * (dones[t] ? 0 : gae);
      advantages.unshift(gae);
    }

    return advantages;
  }

  calculateTDError(
    reward: number,
    currentValue: number,
    nextValue: number,
    gamma: number,
    done: boolean
  ): number {
    const nextV = done ? 0 : nextValue;
    return reward + gamma * nextV - currentValue;
  }

  normalizeAdvantages(advantages: number[]): number[] {
    const mean = advantages.reduce((sum, a) => sum + a, 0) / advantages.length;
    const variance = advantages.reduce((sum, a) => sum + Math.pow(a - mean, 2), 0) / advantages.length;
    const std = Math.sqrt(variance + 1e-8);

    return advantages.map(a => (a - mean) / std);
  }

  calculateActionProbabilities(
    qValues: ActionValues,
    temperature: number = 1.0
  ): ActionValues {
    const values = Object.values(qValues);
    const maxQ = Math.max(...values);
    
    // Compute exp(Q/T) for numerical stability
    const expValues: ActionValues = {
      up: Math.exp((qValues.up - maxQ) / temperature),
      down: Math.exp((qValues.down - maxQ) / temperature),
      left: Math.exp((qValues.left - maxQ) / temperature),
      right: Math.exp((qValues.right - maxQ) / temperature)
    };

    const sum = Object.values(expValues).reduce((a, b) => a + b, 0);

    return {
      up: expValues.up / sum,
      down: expValues.down / sum,
      left: expValues.left / sum,
      right: expValues.right / sum
    };
  }

  compareAdvantages(
    advantages1: ActionValues,
    advantages2: ActionValues
  ): ActionValues {
    return {
      up: advantages2.up - advantages1.up,
      down: advantages2.down - advantages1.down,
      left: advantages2.left - advantages1.left,
      right: advantages2.right - advantages1.right
    };
  }
}