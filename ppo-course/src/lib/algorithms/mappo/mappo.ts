import * as tf from '@tensorflow/tfjs';

interface MAPPOConfig {
  nAgents: number;
  stateSize: number;
  actionSize: number;
  hiddenSize: number;
  learningRate: number;
  clipRange: number;
  gamma: number;
  gaeBalance: number;
  nEpochs: number;
  batchSize: number;
  entropyCoef: number;
  valueCoef: number;
  centralizedCritic: boolean;
  parameterSharing: boolean;
  communicationSize?: number;
}

interface AgentExperience {
  agentId: number;
  state: tf.Tensor;
  action: tf.Tensor;
  reward: number;
  nextState: tf.Tensor;
  done: boolean;
  logProb: tf.Tensor;
  value: tf.Tensor;
}

interface MultiAgentRollout {
  experiences: AgentExperience[];
  globalState?: tf.Tensor;
  jointActions?: tf.Tensor;
}

export class MAPPO {
  private config: MAPPOConfig;
  private actors: tf.LayersModel[];
  private critics: tf.LayersModel[];
  private sharedActor?: tf.LayersModel;
  private centralizedCritic?: tf.LayersModel;
  private actorOptimizers: tf.Optimizer[];
  private criticOptimizers: tf.Optimizer[];
  private communicationModule?: tf.LayersModel;

  constructor(config: MAPPOConfig) {
    this.config = config;
    this.actors = [];
    this.critics = [];
    this.actorOptimizers = [];
    this.criticOptimizers = [];
    
    this.buildNetworks();
  }

  private buildNetworks(): void {
    if (this.config.parameterSharing) {
      // Build shared actor network
      this.sharedActor = this.buildActorNetwork();
      this.actors = Array(this.config.nAgents).fill(this.sharedActor);
      
      // Single optimizer for shared parameters
      this.actorOptimizers = [tf.train.adam(this.config.learningRate)];
    } else {
      // Build individual actor networks
      for (let i = 0; i < this.config.nAgents; i++) {
        const actor = this.buildActorNetwork();
        this.actors.push(actor);
        this.actorOptimizers.push(tf.train.adam(this.config.learningRate));
      }
    }

    if (this.config.centralizedCritic) {
      // Build centralized critic that sees all agent states
      this.centralizedCritic = this.buildCentralizedCriticNetwork();
      this.criticOptimizers = [tf.train.adam(this.config.learningRate)];
    } else {
      // Build individual critics
      for (let i = 0; i < this.config.nAgents; i++) {
        const critic = this.buildCriticNetwork();
        this.critics.push(critic);
        this.criticOptimizers.push(tf.train.adam(this.config.learningRate));
      }
    }

    // Optional communication module
    if (this.config.communicationSize) {
      this.communicationModule = this.buildCommunicationModule();
    }
  }

  private buildActorNetwork(): tf.LayersModel {
    const input = tf.input({ shape: [this.config.stateSize] });
    
    let x = tf.layers.dense({
      units: this.config.hiddenSize,
      activation: 'relu',
      kernelInitializer: 'glorotUniform'
    }).apply(input) as tf.SymbolicTensor;
    
    x = tf.layers.dense({
      units: this.config.hiddenSize,
      activation: 'relu'
    }).apply(x) as tf.SymbolicTensor;
    
    // Optional communication integration
    if (this.config.communicationSize) {
      const commInput = tf.input({ shape: [this.config.communicationSize] });
      const concat = tf.layers.concatenate().apply([x, commInput]) as tf.SymbolicTensor;
      x = tf.layers.dense({
        units: this.config.hiddenSize,
        activation: 'relu'
      }).apply(concat) as tf.SymbolicTensor;
    }
    
    const output = tf.layers.dense({
      units: this.config.actionSize,
      activation: 'softmax',
      kernelInitializer: 'glorotUniform'
    }).apply(x) as tf.SymbolicTensor;
    
    return tf.model({ inputs: input, outputs: output });
  }

  private buildCriticNetwork(): tf.LayersModel {
    const input = tf.input({ shape: [this.config.stateSize] });
    
    let x = tf.layers.dense({
      units: this.config.hiddenSize,
      activation: 'relu',
      kernelInitializer: 'glorotUniform'
    }).apply(input) as tf.SymbolicTensor;
    
    x = tf.layers.dense({
      units: this.config.hiddenSize,
      activation: 'relu'
    }).apply(x) as tf.SymbolicTensor;
    
    const output = tf.layers.dense({
      units: 1,
      kernelInitializer: 'glorotUniform'
    }).apply(x) as tf.SymbolicTensor;
    
    return tf.model({ inputs: input, outputs: output });
  }

  private buildCentralizedCriticNetwork(): tf.LayersModel {
    // Takes concatenated states of all agents
    const input = tf.input({ shape: [this.config.stateSize * this.config.nAgents] });
    
    let x = tf.layers.dense({
      units: this.config.hiddenSize * 2,
      activation: 'relu',
      kernelInitializer: 'glorotUniform'
    }).apply(input) as tf.SymbolicTensor;
    
    x = tf.layers.dense({
      units: this.config.hiddenSize,
      activation: 'relu'
    }).apply(x) as tf.SymbolicTensor;
    
    x = tf.layers.dense({
      units: this.config.hiddenSize,
      activation: 'relu'
    }).apply(x) as tf.SymbolicTensor;
    
    // Output value for each agent
    const output = tf.layers.dense({
      units: this.config.nAgents,
      kernelInitializer: 'glorotUniform'
    }).apply(x) as tf.SymbolicTensor;
    
    return tf.model({ inputs: input, outputs: output });
  }

  private buildCommunicationModule(): tf.LayersModel {
    const input = tf.input({ shape: [this.config.stateSize] });
    
    let x = tf.layers.dense({
      units: this.config.hiddenSize,
      activation: 'relu'
    }).apply(input) as tf.SymbolicTensor;
    
    const message = tf.layers.dense({
      units: this.config.communicationSize!,
      activation: 'tanh'
    }).apply(x) as tf.SymbolicTensor;
    
    return tf.model({ inputs: input, outputs: message });
  }

  async selectActions(states: tf.Tensor[]): Promise<{
    actions: tf.Tensor[];
    logProbs: tf.Tensor[];
    values: tf.Tensor[];
  }> {
    return tf.tidy(() => {
      const actions: tf.Tensor[] = [];
      const logProbs: tf.Tensor[] = [];
      const values: tf.Tensor[] = [];
      
      // Generate communication messages if enabled
      let messages: tf.Tensor[] | undefined;
      if (this.communicationModule) {
        messages = states.map(state => 
          this.communicationModule!.predict(state) as tf.Tensor
        );
      }
      
      // Select actions for each agent
      for (let i = 0; i < this.config.nAgents; i++) {
        // Get actor network
        const actor = this.config.parameterSharing ? this.sharedActor! : this.actors[i];
        
        // Apply actor
        let actionProbs: tf.Tensor;
        if (messages) {
          // Include communication in decision
          const avgMessage = tf.stack(messages).mean(0);
          const inputWithComm = tf.concat([states[i], avgMessage], 1);
          actionProbs = actor.predict(inputWithComm) as tf.Tensor;
        } else {
          actionProbs = actor.predict(states[i]) as tf.Tensor;
        }
        
        // Sample action
        const action = tf.multinomial(actionProbs, 1);
        const logProb = tf.log(tf.add(actionProbs.gather(action, 1), 1e-8));
        
        actions.push(action);
        logProbs.push(logProb);
      }
      
      // Compute values
      if (this.config.centralizedCritic) {
        // Concatenate all states for centralized critic
        const globalState = tf.concat(states, 1);
        const allValues = this.centralizedCritic!.predict(globalState) as tf.Tensor;
        
        // Split values for each agent
        for (let i = 0; i < this.config.nAgents; i++) {
          values.push(allValues.slice([0, i], [-1, 1]));
        }
      } else {
        // Individual critics
        for (let i = 0; i < this.config.nAgents; i++) {
          const value = this.critics[i].predict(states[i]) as tf.Tensor;
          values.push(value);
        }
      }
      
      return { actions, logProbs, values };
    });
  }

  async update(rollouts: MultiAgentRollout[]): Promise<{
    actorLoss: number[];
    criticLoss: number[];
    entropy: number[];
  }> {
    // Compute advantages and returns
    const processedData = await this.computeAdvantagesAndReturns(rollouts);
    
    // Prepare batch data for each agent
    const agentBatches = this.prepareAgentBatches(processedData);
    
    const actorLosses: number[] = [];
    const criticLosses: number[] = [];
    const entropies: number[] = [];
    
    // Update each agent (or shared parameters)
    for (let epoch = 0; epoch < this.config.nEpochs; epoch++) {
      if (this.config.parameterSharing) {
        // Update shared actor
        const { actorLoss, entropy } = await this.updateSharedActor(agentBatches);
        actorLosses.push(actorLoss);
        entropies.push(entropy);
      } else {
        // Update individual actors
        for (let i = 0; i < this.config.nAgents; i++) {
          const { actorLoss, entropy } = await this.updateActor(i, agentBatches[i]);
          actorLosses.push(actorLoss);
          entropies.push(entropy);
        }
      }
      
      // Update critics
      if (this.config.centralizedCritic) {
        const criticLoss = await this.updateCentralizedCritic(agentBatches);
        criticLosses.push(criticLoss);
      } else {
        for (let i = 0; i < this.config.nAgents; i++) {
          const criticLoss = await this.updateCritic(i, agentBatches[i]);
          criticLosses.push(criticLoss);
        }
      }
    }
    
    return {
      actorLoss: actorLosses,
      criticLoss: criticLosses,
      entropy: entropies
    };
  }

  private async computeAdvantagesAndReturns(rollouts: MultiAgentRollout[]): Promise<any> {
    // GAE computation for each agent
    const processedData: any[] = [];
    
    for (const rollout of rollouts) {
      const agentData: Map<number, any> = new Map();
      
      // Group experiences by agent
      for (const exp of rollout.experiences) {
        if (!agentData.has(exp.agentId)) {
          agentData.set(exp.agentId, {
            states: [],
            actions: [],
            rewards: [],
            values: [],
            logProbs: [],
            advantages: [],
            returns: []
          });
        }
        
        const data = agentData.get(exp.agentId)!;
        data.states.push(exp.state);
        data.actions.push(exp.action);
        data.rewards.push(exp.reward);
        data.values.push(exp.value);
        data.logProbs.push(exp.logProb);
      }
      
      // Compute GAE for each agent
      for (const [agentId, data] of agentData) {
        const advantages = await this.computeGAE(
          data.rewards,
          data.values,
          rollout.experiences.filter(e => e.agentId === agentId).map(e => e.done)
        );
        
        data.advantages = advantages;
        data.returns = advantages.map((adv: number, i: number) => 
          adv + data.values[i].dataSync()[0]
        );
      }
      
      processedData.push(agentData);
    }
    
    return processedData;
  }

  private async computeGAE(
    rewards: number[],
    values: tf.Tensor[],
    dones: boolean[]
  ): Promise<number[]> {
    const advantages: number[] = [];
    let lastGaeAdvantage = 0;
    
    for (let t = rewards.length - 1; t >= 0; t--) {
      const nextValue = t === rewards.length - 1 ? 0 : values[t + 1].dataSync()[0];
      const currentValue = values[t].dataSync()[0];
      const done = dones[t] ? 1 : 0;
      
      const delta = rewards[t] + this.config.gamma * nextValue * (1 - done) - currentValue;
      lastGaeAdvantage = delta + this.config.gamma * this.config.gaeBalance * (1 - done) * lastGaeAdvantage;
      advantages.unshift(lastGaeAdvantage);
    }
    
    // Normalize advantages
    const mean = advantages.reduce((a, b) => a + b) / advantages.length;
    const std = Math.sqrt(
      advantages.reduce((a, b) => a + Math.pow(b - mean, 2), 0) / advantages.length
    );
    
    return advantages.map(adv => (adv - mean) / (std + 1e-8));
  }

  private prepareAgentBatches(processedData: any[]): any[] {
    const agentBatches: any[] = Array(this.config.nAgents).fill(null).map(() => ({
      states: [],
      actions: [],
      oldLogProbs: [],
      advantages: [],
      returns: []
    }));
    
    for (const rolloutData of processedData) {
      for (const [agentId, data] of rolloutData) {
        agentBatches[agentId].states.push(...data.states);
        agentBatches[agentId].actions.push(...data.actions);
        agentBatches[agentId].oldLogProbs.push(...data.logProbs);
        agentBatches[agentId].advantages.push(...data.advantages);
        agentBatches[agentId].returns.push(...data.returns);
      }
    }
    
    return agentBatches;
  }

  private async updateActor(
    agentId: number,
    batch: any
  ): Promise<{ actorLoss: number; entropy: number }> {
    const actor = this.actors[agentId];
    const optimizer = this.actorOptimizers[agentId];
    
    let totalLoss = 0;
    let totalEntropy = 0;
    
    const grads = tf.variableGrads(() => {
      const states = tf.stack(batch.states);
      const actions = tf.stack(batch.actions);
      const oldLogProbs = tf.stack(batch.oldLogProbs);
      const advantages = tf.tensor(batch.advantages);
      
      const actionProbs = actor.predict(states) as tf.Tensor;
      const logProbs = tf.log(tf.add(
        actionProbs.gather(actions.reshape([-1]), 1),
        1e-8
      ));
      
      // PPO clipped objective
      const ratio = tf.exp(tf.sub(logProbs, oldLogProbs));
      const clippedRatio = tf.clipByValue(
        ratio,
        1 - this.config.clipRange,
        1 + this.config.clipRange
      );
      
      const surrogate1 = tf.mul(ratio, advantages);
      const surrogate2 = tf.mul(clippedRatio, advantages);
      const policyLoss = tf.neg(tf.mean(tf.minimum(surrogate1, surrogate2)));
      
      // Entropy bonus
      const entropy = tf.neg(tf.mean(
        tf.sum(tf.mul(actionProbs, tf.log(tf.add(actionProbs, 1e-8))), 1)
      ));
      
      const loss = tf.add(
        policyLoss,
        tf.mul(tf.scalar(-this.config.entropyCoef), entropy)
      );
      
      totalLoss = policyLoss.dataSync()[0];
      totalEntropy = entropy.dataSync()[0];
      
      return loss;
    });
    
    optimizer.applyGradients(grads.grads);
    
    return { actorLoss: totalLoss, entropy: totalEntropy };
  }

  private async updateSharedActor(
    agentBatches: any[]
  ): Promise<{ actorLoss: number; entropy: number }> {
    const actor = this.sharedActor!;
    const optimizer = this.actorOptimizers[0];
    
    let totalLoss = 0;
    let totalEntropy = 0;
    let nSamples = 0;
    
    const grads = tf.variableGrads(() => {
      let loss = tf.scalar(0);
      
      // Accumulate gradients from all agents
      for (const batch of agentBatches) {
        const states = tf.stack(batch.states);
        const actions = tf.stack(batch.actions);
        const oldLogProbs = tf.stack(batch.oldLogProbs);
        const advantages = tf.tensor(batch.advantages);
        
        const actionProbs = actor.predict(states) as tf.Tensor;
        const logProbs = tf.log(tf.add(
          actionProbs.gather(actions.reshape([-1]), 1),
          1e-8
        ));
        
        // PPO clipped objective
        const ratio = tf.exp(tf.sub(logProbs, oldLogProbs));
        const clippedRatio = tf.clipByValue(
          ratio,
          1 - this.config.clipRange,
          1 + this.config.clipRange
        );
        
        const surrogate1 = tf.mul(ratio, advantages);
        const surrogate2 = tf.mul(clippedRatio, advantages);
        const policyLoss = tf.neg(tf.mean(tf.minimum(surrogate1, surrogate2)));
        
        // Entropy bonus
        const entropy = tf.neg(tf.mean(
          tf.sum(tf.mul(actionProbs, tf.log(tf.add(actionProbs, 1e-8))), 1)
        ));
        
        const agentLoss = tf.add(
          policyLoss,
          tf.mul(tf.scalar(-this.config.entropyCoef), entropy)
        );
        
        loss = tf.add(loss, agentLoss);
        
        totalLoss += policyLoss.dataSync()[0];
        totalEntropy += entropy.dataSync()[0];
        nSamples += batch.states.length;
      }
      
      return tf.div(loss, tf.scalar(this.config.nAgents));
    });
    
    optimizer.applyGradients(grads.grads);
    
    return {
      actorLoss: totalLoss / this.config.nAgents,
      entropy: totalEntropy / this.config.nAgents
    };
  }

  private async updateCritic(
    agentId: number,
    batch: any
  ): Promise<number> {
    const critic = this.critics[agentId];
    const optimizer = this.criticOptimizers[agentId];
    
    let totalLoss = 0;
    
    const grads = tf.variableGrads(() => {
      const states = tf.stack(batch.states);
      const returns = tf.tensor(batch.returns).reshape([-1, 1]);
      
      const values = critic.predict(states) as tf.Tensor;
      const loss = tf.losses.meanSquaredError(returns, values);
      
      totalLoss = loss.dataSync()[0];
      
      return tf.mul(loss, tf.scalar(this.config.valueCoef));
    });
    
    optimizer.applyGradients(grads.grads);
    
    return totalLoss;
  }

  private async updateCentralizedCritic(
    agentBatches: any[]
  ): Promise<number> {
    const critic = this.centralizedCritic!;
    const optimizer = this.criticOptimizers[0];
    
    let totalLoss = 0;
    
    // Prepare global states and returns
    const globalStates: tf.Tensor[] = [];
    const allReturns: number[][] = [];
    
    // Align data across agents
    const minLength = Math.min(...agentBatches.map(b => b.states.length));
    
    for (let i = 0; i < minLength; i++) {
      const statesAtTime: tf.Tensor[] = [];
      const returnsAtTime: number[] = [];
      
      for (let j = 0; j < this.config.nAgents; j++) {
        statesAtTime.push(agentBatches[j].states[i]);
        returnsAtTime.push(agentBatches[j].returns[i]);
      }
      
      globalStates.push(tf.concat(statesAtTime, 1));
      allReturns.push(returnsAtTime);
    }
    
    const grads = tf.variableGrads(() => {
      const states = tf.stack(globalStates).squeeze([1]);
      const returns = tf.tensor(allReturns);
      
      const values = critic.predict(states) as tf.Tensor;
      const loss = tf.losses.meanSquaredError(returns, values);
      
      totalLoss = loss.dataSync()[0];
      
      return tf.mul(loss, tf.scalar(this.config.valueCoef));
    });
    
    optimizer.applyGradients(grads.grads);
    
    return totalLoss;
  }

  // Credit assignment mechanism
  async computeCreditAssignment(
    rollout: MultiAgentRollout,
    method: 'difference' | 'shapley' | 'counterfactual' = 'difference'
  ): Promise<Map<number, number[]>> {
    const credits = new Map<number, number[]>();
    
    if (method === 'difference') {
      // Difference rewards: individual contribution
      for (const exp of rollout.experiences) {
        const agentCredits = credits.get(exp.agentId) || [];
        
        // Compute baseline without this agent
        const baseline = await this.computeBaseline(rollout, exp.agentId);
        const credit = exp.reward - baseline;
        
        agentCredits.push(credit);
        credits.set(exp.agentId, agentCredits);
      }
    } else if (method === 'counterfactual') {
      // Counterfactual reasoning
      for (let i = 0; i < this.config.nAgents; i++) {
        const agentCredits = await this.computeCounterfactualCredits(rollout, i);
        credits.set(i, agentCredits);
      }
    }
    
    return credits;
  }

  private async computeBaseline(
    rollout: MultiAgentRollout,
    excludeAgent: number
  ): Promise<number> {
    // Simplified baseline computation
    const otherAgentRewards = rollout.experiences
      .filter(exp => exp.agentId !== excludeAgent)
      .map(exp => exp.reward);
    
    return otherAgentRewards.reduce((a, b) => a + b, 0) / otherAgentRewards.length;
  }

  private async computeCounterfactualCredits(
    rollout: MultiAgentRollout,
    agentId: number
  ): Promise<number[]> {
    // Placeholder for counterfactual credit assignment
    return rollout.experiences
      .filter(exp => exp.agentId === agentId)
      .map(exp => exp.reward);
  }

  // Save and load methods
  async save(path: string): Promise<void> {
    if (this.config.parameterSharing) {
      await this.sharedActor!.save(`${path}/shared_actor`);
    } else {
      for (let i = 0; i < this.config.nAgents; i++) {
        await this.actors[i].save(`${path}/actor_${i}`);
      }
    }
    
    if (this.config.centralizedCritic) {
      await this.centralizedCritic!.save(`${path}/centralized_critic`);
    } else {
      for (let i = 0; i < this.config.nAgents; i++) {
        await this.critics[i].save(`${path}/critic_${i}`);
      }
    }
    
    if (this.communicationModule) {
      await this.communicationModule.save(`${path}/communication`);
    }
  }

  async load(path: string): Promise<void> {
    if (this.config.parameterSharing) {
      this.sharedActor = await tf.loadLayersModel(`${path}/shared_actor/model.json`);
      this.actors = Array(this.config.nAgents).fill(this.sharedActor);
    } else {
      for (let i = 0; i < this.config.nAgents; i++) {
        this.actors[i] = await tf.loadLayersModel(`${path}/actor_${i}/model.json`);
      }
    }
    
    if (this.config.centralizedCritic) {
      this.centralizedCritic = await tf.loadLayersModel(`${path}/centralized_critic/model.json`);
    } else {
      for (let i = 0; i < this.config.nAgents; i++) {
        this.critics[i] = await tf.loadLayersModel(`${path}/critic_${i}/model.json`);
      }
    }
    
    if (this.config.communicationSize) {
      this.communicationModule = await tf.loadLayersModel(`${path}/communication/model.json`);
    }
  }
}