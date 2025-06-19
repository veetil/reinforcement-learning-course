import * as tf from '@tensorflow/tfjs';

export interface SACConfig {
  // Network architecture
  actorHiddenSizes: number[];
  criticHiddenSizes: number[];
  
  // Learning rates
  actorLearningRate: number;
  criticLearningRate: number;
  alphaLearningRate: number;
  
  // SAC specific
  gamma: number;
  tau: number; // Soft update coefficient
  initialAlpha: number; // Temperature parameter
  targetEntropy?: number; // If not provided, will be -dim(A)
  rewardScale: number;
  
  // Training
  batchSize: number;
  bufferSize: number;
  updateAfter: number;
  updateEvery: number;
  
  // Exploration
  explorationNoise: number;
}

export interface SACNetworks {
  actor: tf.LayersModel;
  critic1: tf.LayersModel;
  critic2: tf.LayersModel;
  targetCritic1: tf.LayersModel;
  targetCritic2: tf.LayersModel;
  logAlpha: tf.Variable;
}

export interface Experience {
  state: tf.Tensor;
  action: tf.Tensor;
  reward: number;
  nextState: tf.Tensor;
  done: boolean;
}

export class ReplayBuffer {
  private buffer: Experience[] = [];
  private maxSize: number;
  private pointer: number = 0;
  
  constructor(maxSize: number) {
    this.maxSize = maxSize;
  }
  
  add(experience: Experience): void {
    if (this.buffer.length < this.maxSize) {
      this.buffer.push(experience);
    } else {
      this.buffer[this.pointer] = experience;
    }
    this.pointer = (this.pointer + 1) % this.maxSize;
  }
  
  sample(batchSize: number): Experience[] {
    const indices = tf.randomUniform([batchSize], 0, this.buffer.length, 'int32').arraySync() as number[];
    return indices.map(i => this.buffer[Math.floor(i)]);
  }
  
  size(): number {
    return this.buffer.length;
  }
}

export class SAC {
  private config: SACConfig;
  private networks: SACNetworks;
  private replayBuffer: ReplayBuffer;
  private targetEntropy: number;
  private actorOptimizer: tf.Optimizer;
  private critic1Optimizer: tf.Optimizer;
  private critic2Optimizer: tf.Optimizer;
  private alphaOptimizer: tf.Optimizer;
  private updateCount: number = 0;
  
  constructor(
    stateDim: number,
    actionDim: number,
    config: Partial<SACConfig> = {}
  ) {
    this.config = {
      actorHiddenSizes: [256, 256],
      criticHiddenSizes: [256, 256],
      actorLearningRate: 3e-4,
      criticLearningRate: 3e-4,
      alphaLearningRate: 3e-4,
      gamma: 0.99,
      tau: 0.005,
      initialAlpha: 0.2,
      rewardScale: 1.0,
      batchSize: 256,
      bufferSize: 1000000,
      updateAfter: 1000,
      updateEvery: 1,
      explorationNoise: 0.1,
      ...config
    };
    
    this.targetEntropy = config.targetEntropy ?? -actionDim;
    this.replayBuffer = new ReplayBuffer(this.config.bufferSize);
    
    // Initialize networks
    this.networks = this.buildNetworks(stateDim, actionDim);
    
    // Initialize optimizers
    this.actorOptimizer = tf.train.adam(this.config.actorLearningRate);
    this.critic1Optimizer = tf.train.adam(this.config.criticLearningRate);
    this.critic2Optimizer = tf.train.adam(this.config.criticLearningRate);
    this.alphaOptimizer = tf.train.adam(this.config.alphaLearningRate);
    
    // Initialize target networks
    this.updateTargetNetworks(1.0); // Hard copy
  }
  
  private buildNetworks(stateDim: number, actionDim: number): SACNetworks {
    // Actor network (outputs mean and log_std for Gaussian policy)
    const actor = tf.sequential({
      layers: [
        tf.layers.dense({ units: this.config.actorHiddenSizes[0], activation: 'relu', inputShape: [stateDim] }),
        ...this.config.actorHiddenSizes.slice(1).map(units => 
          tf.layers.dense({ units, activation: 'relu' })
        ),
        tf.layers.dense({ units: actionDim * 2 }) // mean and log_std
      ]
    });
    
    // Critic networks (Q-functions)
    const buildCritic = () => tf.sequential({
      layers: [
        tf.layers.dense({ 
          units: this.config.criticHiddenSizes[0], 
          activation: 'relu', 
          inputShape: [stateDim + actionDim] 
        }),
        ...this.config.criticHiddenSizes.slice(1).map(units => 
          tf.layers.dense({ units, activation: 'relu' })
        ),
        tf.layers.dense({ units: 1 })
      ]
    });
    
    const critic1 = buildCritic();
    const critic2 = buildCritic();
    const targetCritic1 = buildCritic();
    const targetCritic2 = buildCritic();
    
    // Temperature parameter
    const logAlpha = tf.variable(tf.scalar(Math.log(this.config.initialAlpha)));
    
    return {
      actor,
      critic1,
      critic2,
      targetCritic1,
      targetCritic2,
      logAlpha
    };
  }
  
  private updateTargetNetworks(tau: number = this.config.tau): void {
    // Soft update: target = tau * current + (1 - tau) * target
    const updateWeights = (target: tf.LayersModel, source: tf.LayersModel) => {
      const targetWeights = target.getWeights();
      const sourceWeights = source.getWeights();
      
      const newWeights = targetWeights.map((targetWeight, i) => {
        return targetWeight.mul(1 - tau).add(sourceWeights[i].mul(tau));
      });
      
      target.setWeights(newWeights);
    };
    
    updateWeights(this.networks.targetCritic1, this.networks.critic1);
    updateWeights(this.networks.targetCritic2, this.networks.critic2);
  }
  
  async selectAction(state: tf.Tensor, evaluate: boolean = false): Promise<tf.Tensor> {
    return tf.tidy(() => {
      const output = this.networks.actor.predict(state) as tf.Tensor;
      const actionDim = output.shape[1]! / 2;
      
      // Split mean and log_std
      const mean = output.slice([0, 0], [-1, actionDim]);
      const logStd = output.slice([0, actionDim], [-1, actionDim]);
      const std = logStd.exp();
      
      if (evaluate) {
        // Deterministic action (mean)
        return mean.tanh();
      } else {
        // Sample from Gaussian
        const normal = tf.randomNormal(mean.shape);
        const action = mean.add(std.mul(normal));
        return action.tanh();
      }
    });
  }
  
  private computeTargetQ(rewards: tf.Tensor, nextStates: tf.Tensor, dones: tf.Tensor): tf.Tensor {
    return tf.tidy(() => {
      // Sample next actions from current policy
      const nextOutput = this.networks.actor.predict(nextStates) as tf.Tensor;
      const actionDim = nextOutput.shape[1]! / 2;
      
      const nextMean = nextOutput.slice([0, 0], [-1, actionDim]);
      const nextLogStd = nextOutput.slice([0, actionDim], [-1, actionDim]);
      const nextStd = nextLogStd.exp();
      
      // Sample actions
      const normal = tf.randomNormal(nextMean.shape);
      const nextActions = nextMean.add(nextStd.mul(normal)).tanh();
      
      // Compute log probability for entropy term
      const logProb = this.computeLogProb(nextActions, nextMean, nextLogStd);
      
      // Get Q-values from both target critics
      const nextStateActions = tf.concat([nextStates, nextActions], 1);
      const targetQ1 = this.networks.targetCritic1.predict(nextStateActions) as tf.Tensor;
      const targetQ2 = this.networks.targetCritic2.predict(nextStateActions) as tf.Tensor;
      
      // Take minimum Q-value (clipped double Q-learning)
      const targetQ = tf.minimum(targetQ1, targetQ2);
      
      // Add entropy term
      const alpha = this.networks.logAlpha.exp();
      const targetValue = targetQ.sub(alpha.mul(logProb));
      
      // Compute TD target
      const notDones = tf.scalar(1).sub(dones);
      return rewards.add(notDones.mul(tf.scalar(this.config.gamma)).mul(targetValue.squeeze()));
    });
  }
  
  private computeLogProb(actions: tf.Tensor, mean: tf.Tensor, logStd: tf.Tensor): tf.Tensor {
    return tf.tidy(() => {
      const std = logStd.exp();
      
      // Compute Gaussian log probability
      const gaussianLogProb = tf.scalar(-0.5 * Math.log(2 * Math.PI))
        .sub(logStd)
        .sub(actions.sub(mean).square().div(std.square().mul(2)));
      
      // Correction for tanh squashing
      const logProbCorrection = tf.scalar(2)
        .mul(tf.scalar(Math.log(2))
        .sub(actions)
        .sub(tf.softplus(tf.scalar(-2).mul(actions))));
      
      return gaussianLogProb.sum(1).sub(logProbCorrection.sum(1));
    });
  }
  
  async update(): Promise<{
    actorLoss: number;
    critic1Loss: number;
    critic2Loss: number;
    alpha: number;
  }> {
    if (this.replayBuffer.size() < this.config.updateAfter) {
      return { actorLoss: 0, critic1Loss: 0, critic2Loss: 0, alpha: 0 };
    }
    
    if (this.updateCount % this.config.updateEvery !== 0) {
      this.updateCount++;
      return { actorLoss: 0, critic1Loss: 0, critic2Loss: 0, alpha: 0 };
    }
    
    const batch = this.replayBuffer.sample(this.config.batchSize);
    
    // Convert batch to tensors
    const states = tf.stack(batch.map(e => e.state));
    const actions = tf.stack(batch.map(e => e.action));
    const rewards = tf.tensor1d(batch.map(e => e.reward * this.config.rewardScale));
    const nextStates = tf.stack(batch.map(e => e.nextState));
    const dones = tf.tensor1d(batch.map(e => e.done ? 1 : 0));
    
    // Update critics
    const targetQ = this.computeTargetQ(rewards, nextStates, dones);
    const stateActions = tf.concat([states, actions], 1);
    
    const critic1Loss = await this.critic1Optimizer.minimize(() => {
      const q1 = (this.networks.critic1.predict(stateActions) as tf.Tensor).squeeze();
      return tf.losses.meanSquaredError(targetQ, q1);
    });
    
    const critic2Loss = await this.critic2Optimizer.minimize(() => {
      const q2 = (this.networks.critic2.predict(stateActions) as tf.Tensor).squeeze();
      return tf.losses.meanSquaredError(targetQ, q2);
    });
    
    // Update actor
    const actorLoss = await this.actorOptimizer.minimize(() => {
      const output = this.networks.actor.predict(states) as tf.Tensor;
      const actionDim = output.shape[1]! / 2;
      
      const mean = output.slice([0, 0], [-1, actionDim]);
      const logStd = output.slice([0, actionDim], [-1, actionDim]);
      const std = logStd.exp();
      
      // Sample actions
      const normal = tf.randomNormal(mean.shape);
      const sampledActions = mean.add(std.mul(normal)).tanh();
      
      // Compute log probability
      const logProb = this.computeLogProb(sampledActions, mean, logStd);
      
      // Compute Q-values
      const stateActions = tf.concat([states, sampledActions], 1);
      const q1 = this.networks.critic1.predict(stateActions) as tf.Tensor;
      const q2 = this.networks.critic2.predict(stateActions) as tf.Tensor;
      const minQ = tf.minimum(q1, q2).squeeze();
      
      // Actor loss: maximize Q - α * log π
      const alpha = this.networks.logAlpha.exp();
      return alpha.mul(logProb).sub(minQ).mean().neg();
    });
    
    // Update temperature
    const alphaLoss = await this.alphaOptimizer.minimize(() => {
      const output = this.networks.actor.predict(states) as tf.Tensor;
      const actionDim = output.shape[1]! / 2;
      
      const mean = output.slice([0, 0], [-1, actionDim]);
      const logStd = output.slice([0, actionDim], [-1, actionDim]);
      const std = logStd.exp();
      
      // Sample actions
      const normal = tf.randomNormal(mean.shape);
      const sampledActions = mean.add(std.mul(normal)).tanh();
      
      // Compute log probability
      const logProb = this.computeLogProb(sampledActions, mean, logStd);
      
      // Temperature loss: α * (−log π − H̄)
      return this.networks.logAlpha.mul(logProb.add(this.targetEntropy).detach()).mean().neg();
    });
    
    // Update target networks
    this.updateTargetNetworks();
    
    this.updateCount++;
    
    // Clean up tensors
    states.dispose();
    actions.dispose();
    rewards.dispose();
    nextStates.dispose();
    dones.dispose();
    targetQ.dispose();
    stateActions.dispose();
    
    return {
      actorLoss: await actorLoss.data() as unknown as number,
      critic1Loss: await critic1Loss.data() as unknown as number,
      critic2Loss: await critic2Loss.data() as unknown as number,
      alpha: await this.networks.logAlpha.exp().data() as unknown as number
    };
  }
  
  storeTransition(
    state: tf.Tensor,
    action: tf.Tensor,
    reward: number,
    nextState: tf.Tensor,
    done: boolean
  ): void {
    this.replayBuffer.add({
      state: state.clone(),
      action: action.clone(),
      reward,
      nextState: nextState.clone(),
      done
    });
  }
  
  getAlpha(): number {
    return this.networks.logAlpha.exp().dataSync()[0];
  }
  
  save(path: string): Promise<void> {
    // Save all networks
    const saves = [
      this.networks.actor.save(`${path}/actor`),
      this.networks.critic1.save(`${path}/critic1`),
      this.networks.critic2.save(`${path}/critic2`),
      this.networks.targetCritic1.save(`${path}/target_critic1`),
      this.networks.targetCritic2.save(`${path}/target_critic2`)
    ];
    
    return Promise.all(saves).then(() => {
      // Save alpha
      const alpha = this.networks.logAlpha.dataSync()[0];
      const config = { ...this.config, logAlpha: alpha };
      // In a real implementation, save config to file
    });
  }
}