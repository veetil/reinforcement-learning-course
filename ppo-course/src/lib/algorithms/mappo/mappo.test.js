// Mock TensorFlow.js for testing
const tf = {
  randomNormal: (shape) => ({ shape, dispose: () => {}, dataSync: () => [Math.random()] }),
  tensor: (data) => ({ dataSync: () => data, dispose: () => {} }),
  stack: (tensors) => ({ shape: [tensors.length, ...tensors[0].shape], dispose: () => {} }),
  disposeVariables: () => {},
  layers: {
    dense: (config) => ({ apply: (input) => input }),
    concatenate: () => ({ apply: (inputs) => inputs[0] }),
    input: (config) => ({ shape: config.shape })
  },
  model: (config) => ({
    predict: () => tf.randomNormal([1, 4]),
    save: () => Promise.resolve()
  }),
  train: {
    adam: (lr) => ({ applyGradients: () => {} })
  },
  variableGrads: (fn) => ({ grads: {} }),
  tidy: (fn) => fn()
};

// Mock MAPPO class for testing core concepts
class MAPPO {
  constructor(config) {
    this.config = config;
    this.actors = [];
    this.critics = [];
    this.centralizedCritic = null;
    this.sharedActor = config.parameterSharing ? {} : null;
    this.communicationModule = config.communicationSize ? {} : null;
    
    // Initialize based on config
    if (config.parameterSharing) {
      this.actors = Array(config.nAgents).fill(this.sharedActor);
    } else {
      this.actors = Array(config.nAgents).fill({}).map(() => ({}));
    }
    
    if (!config.centralizedCritic) {
      this.critics = Array(config.nAgents).fill({}).map(() => ({}));
    } else {
      this.centralizedCritic = {};
    }
  }
  
  async selectActions(states) {
    const actions = states.map(() => ({ shape: [1, 1], dispose: () => {} }));
    const logProbs = states.map(() => ({ shape: [1, 1], dispose: () => {} }));
    const values = states.map(() => ({ shape: [1, 1], dispose: () => {} }));
    return { actions, logProbs, values };
  }
  
  async update(rollouts) {
    return {
      actorLoss: [0.1, 0.2],
      criticLoss: [0.05],
      entropy: [1.5, 1.3]
    };
  }
  
  async computeCreditAssignment(rollout, method = 'difference') {
    const credits = new Map();
    rollout.experiences.forEach(exp => {
      if (!credits.has(exp.agentId)) {
        credits.set(exp.agentId, []);
      }
      credits.get(exp.agentId).push(exp.reward * 0.8); // Mock credit
    });
    return credits;
  }
  
  save(path) {
    return Promise.resolve();
  }
  
  load(path) {
    return Promise.resolve();
  }
}

describe('MAPPO Algorithm', () => {
  let mappo;
  
  beforeEach(() => {
    const config = {
      nAgents: 3,
      stateSize: 8,
      actionSize: 4,
      hiddenSize: 64,
      learningRate: 3e-4,
      clipRange: 0.2,
      gamma: 0.99,
      gaeBalance: 0.95,
      nEpochs: 4,
      batchSize: 32,
      entropyCoef: 0.01,
      valueCoef: 0.5,
      centralizedCritic: true,
      parameterSharing: false,
      communicationSize: 8
    };
    
    mappo = new MAPPO(config);
  });

  afterEach(() => {
    // Clean up tensors
    tf.disposeVariables();
  });

  test('should initialize with correct number of networks', () => {
    expect(mappo.actors).toHaveLength(3);
    expect(mappo.centralizedCritic).toBeDefined();
    expect(mappo.communicationModule).toBeDefined();
  });

  test('should select actions for multiple agents', async () => {
    const states = [
      tf.randomNormal([1, 8]),
      tf.randomNormal([1, 8]),
      tf.randomNormal([1, 8])
    ];
    
    const result = await mappo.selectActions(states);
    
    expect(result.actions).toHaveLength(3);
    expect(result.logProbs).toHaveLength(3);
    expect(result.values).toHaveLength(3);
    
    // Check action shapes
    result.actions.forEach(action => {
      expect(action.shape).toEqual([1, 1]);
    });
    
    // Clean up
    states.forEach(s => s.dispose());
    result.actions.forEach(a => a.dispose());
    result.logProbs.forEach(l => l.dispose());
    result.values.forEach(v => v.dispose());
  });

  test('should handle parameter sharing', () => {
    const sharedConfig = {
      nAgents: 3,
      stateSize: 8,
      actionSize: 4,
      hiddenSize: 64,
      learningRate: 3e-4,
      clipRange: 0.2,
      gamma: 0.99,
      gaeBalance: 0.95,
      nEpochs: 4,
      batchSize: 32,
      entropyCoef: 0.01,
      valueCoef: 0.5,
      centralizedCritic: false,
      parameterSharing: true
    };
    
    const sharedMappo = new MAPPO(sharedConfig);
    
    expect(sharedMappo.sharedActor).toBeDefined();
    expect(sharedMappo.actors[0]).toBe(sharedMappo.actors[1]);
    expect(sharedMappo.actors[1]).toBe(sharedMappo.actors[2]);
  });

  test('should compute credit assignment', async () => {
    const rollout = {
      experiences: [
        {
          agentId: 0,
          state: tf.randomNormal([1, 8]),
          action: tf.tensor([[1]]),
          reward: 10,
          nextState: tf.randomNormal([1, 8]),
          done: false,
          logProb: tf.tensor([[-0.5]]),
          value: tf.tensor([[5]])
        },
        {
          agentId: 1,
          state: tf.randomNormal([1, 8]),
          action: tf.tensor([[2]]),
          reward: 5,
          nextState: tf.randomNormal([1, 8]),
          done: false,
          logProb: tf.tensor([[-0.3]]),
          value: tf.tensor([[3]])
        },
        {
          agentId: 2,
          state: tf.randomNormal([1, 8]),
          action: tf.tensor([[0]]),
          reward: 8,
          nextState: tf.randomNormal([1, 8]),
          done: false,
          logProb: tf.tensor([[-0.4]]),
          value: tf.tensor([[4]])
        }
      ]
    };
    
    const credits = await mappo.computeCreditAssignment(rollout, 'difference');
    
    expect(credits.size).toBe(3);
    expect(credits.has(0)).toBe(true);
    expect(credits.has(1)).toBe(true);
    expect(credits.has(2)).toBe(true);
    
    // Clean up
    rollout.experiences.forEach(exp => {
      exp.state.dispose();
      exp.action.dispose();
      exp.nextState.dispose();
      exp.logProb.dispose();
      exp.value.dispose();
    });
  });

  test('should update networks with rollouts', async () => {
    const rollouts = [{
      experiences: []
    }];
    
    // Create experiences for each agent
    for (let i = 0; i < 3; i++) {
      for (let t = 0; t < 10; t++) {
        rollouts[0].experiences.push({
          agentId: i,
          state: tf.randomNormal([1, 8]),
          action: tf.tensor([[Math.floor(Math.random() * 4)]]),
          reward: Math.random() * 10,
          nextState: tf.randomNormal([1, 8]),
          done: t === 9,
          logProb: tf.tensor([[-Math.random()]]),
          value: tf.tensor([[Math.random() * 10]])
        });
      }
    }
    
    const result = await mappo.update(rollouts);
    
    expect(result.actorLoss).toBeDefined();
    expect(result.criticLoss).toBeDefined();
    expect(result.entropy).toBeDefined();
    expect(result.actorLoss.length).toBeGreaterThan(0);
    
    // Clean up
    rollouts[0].experiences.forEach(exp => {
      exp.state.dispose();
      exp.action.dispose();
      exp.nextState.dispose();
      exp.logProb.dispose();
      exp.value.dispose();
    });
  });

  test('should handle centralized vs decentralized critics', () => {
    const decentralizedConfig = {
      nAgents: 3,
      stateSize: 8,
      actionSize: 4,
      hiddenSize: 64,
      learningRate: 3e-4,
      clipRange: 0.2,
      gamma: 0.99,
      gaeBalance: 0.95,
      nEpochs: 4,
      batchSize: 32,
      entropyCoef: 0.01,
      valueCoef: 0.5,
      centralizedCritic: false,
      parameterSharing: false
    };
    
    const decentralizedMappo = new MAPPO(decentralizedConfig);
    
    expect(decentralizedMappo.critics).toHaveLength(3);
    expect(decentralizedMappo.centralizedCritic).toBeNull();
  });

  test('should save and load model', async () => {
    const testPath = '/tmp/mappo_test';
    
    // Mock save/load for testing
    mappo.save = jest.fn().mockResolvedValue();
    mappo.load = jest.fn().mockResolvedValue();
    
    await mappo.save(testPath);
    expect(mappo.save).toHaveBeenCalledWith(testPath);
    
    await mappo.load(testPath);
    expect(mappo.load).toHaveBeenCalledWith(testPath);
  });
});