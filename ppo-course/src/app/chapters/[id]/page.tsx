'use client';

import React, { useEffect, useState, useRef } from 'react';
import { useParams, useRouter } from 'next/navigation';
import { motion } from 'framer-motion';
import Link from 'next/link';
import { 
  ArrowLeft, ArrowRight, CheckCircle, Circle, Brain, 
  Target, TrendingUp, Code, AlertCircle 
} from 'lucide-react';
import { InteractiveDemo } from '@/components/chapters/InteractiveDemo';
import { QuizSection, QuizQuestion } from '@/components/chapters/QuizSection';
import { CodeExample } from '@/components/chapters/CodeExample';
import { ConfusionClarifier } from '@/components/chapters/ConfusionClarifier';

// Chapter content data
const chaptersContent: Record<string, any> = {
  '1': {
    title: 'Chapter 1: Foundations',
    description: 'Neural networks, optimization, and mathematical prerequisites',
    objectives: [
      'Understand neural network fundamentals',
      'Master backpropagation algorithm',
      'Learn gradient descent optimization',
      'Get comfortable with PyTorch basics'
    ],
    sections: [
      {
        id: 'neural-networks',
        title: '1.1 Neural Network Fundamentals',
        content: `Artificial neural networks are computational models inspired by biological neural networks. They consist of interconnected nodes (neurons) organized in layers that process information through weighted connections.

A neural network typically has:
- **Input Layer**: Receives the input features
- **Hidden Layers**: Process and transform the data
- **Output Layer**: Produces the final predictions

The power of neural networks comes from their ability to learn complex non-linear relationships through training.`
      },
      {
        id: 'backpropagation',
        title: '1.2 Backpropagation',
        content: `Backpropagation is the cornerstone algorithm for training neural networks. It uses the chain rule of calculus to compute gradients of the loss function with respect to each parameter in the network.

The algorithm works in two phases:
1. **Forward Pass**: Compute predictions and loss
2. **Backward Pass**: Compute gradients and update weights

This process allows the network to learn from its mistakes and improve over time.`
      },
      {
        id: 'gradient-descent',
        title: '1.3 Gradient Descent Optimization',
        content: `Gradient descent is an iterative optimization algorithm used to minimize the loss function. It updates parameters in the direction opposite to the gradient.

Key variants include:
- **Batch Gradient Descent**: Uses entire dataset
- **Stochastic Gradient Descent (SGD)**: Uses single sample
- **Mini-batch Gradient Descent**: Uses small batches

Modern optimizers like Adam and RMSprop build upon these foundations with adaptive learning rates.`
      },
      {
        id: 'pytorch-basics',
        title: '1.4 PyTorch Basics',
        content: `PyTorch is a powerful deep learning framework that provides automatic differentiation and GPU acceleration. It's the foundation we'll use for implementing PPO.`
      }
    ],
    quiz: [
      {
        id: 'q1',
        question: 'What is the primary purpose of backpropagation?',
        options: [
          'To make predictions on new data',
          'To compute gradients for parameter updates',
          'To initialize network weights',
          'To evaluate model performance'
        ],
        correctAnswer: 1,
        explanation: 'Backpropagation computes gradients of the loss function with respect to network parameters, enabling gradient-based optimization.'
      },
      {
        id: 'q2',
        question: 'Which component of a neural network applies non-linearity?',
        options: [
          'Weights',
          'Biases',
          'Activation functions',
          'Loss function'
        ],
        correctAnswer: 2,
        explanation: 'Activation functions like ReLU, sigmoid, and tanh introduce non-linearity, allowing networks to learn complex patterns.'
      },
      {
        id: 'q3',
        question: 'What is the main advantage of mini-batch gradient descent over batch gradient descent?',
        options: [
          'Higher accuracy',
          'Faster convergence and better generalization',
          'Simpler implementation',
          'Deterministic updates'
        ],
        correctAnswer: 1,
        explanation: 'Mini-batch gradient descent offers a balance between computational efficiency and gradient noise, often leading to faster convergence and better generalization.'
      }
    ] as QuizQuestion[],
    exercises: [
      'Implement a 2-layer neural network from scratch using only NumPy. Compare your results with the PyTorch implementation.',
      'Visualize how different activation functions (ReLU, Sigmoid, Tanh) affect gradient flow during backpropagation.',
      'Experiment with different learning rates (0.1, 0.01, 0.001, 0.0001) and observe the training dynamics.'
    ],
    keyTakeaways: [
      'Neural networks learn through iterative weight updates using gradients',
      'Backpropagation efficiently computes gradients using the chain rule',
      'Learning rate is a critical hyperparameter that affects convergence',
      'PyTorch provides automatic differentiation, making implementation easier'
    ]
  },
  '2': {
    title: 'Chapter 2: RL Fundamentals',
    description: 'MDP framework, policies, rewards, and exploration strategies',
    objectives: [
      'Understand Markov Decision Processes (MDPs)',
      'Learn about states, actions, and rewards',
      'Master the concept of policies',
      'Explore different exploration strategies'
    ],
    sections: [
      {
        id: 'mdp-framework',
        title: '2.1 Markov Decision Process',
        content: `A Markov Decision Process (MDP) is the mathematical framework for modeling sequential decision-making problems in RL.

An MDP consists of:
- **State Space (S)**: All possible states the agent can be in
- **Action Space (A)**: All possible actions the agent can take
- **Transition Function P(s'|s,a)**: Probability of reaching state s' from state s by taking action a
- **Reward Function R(s,a,s')**: Immediate reward for transitions
- **Discount Factor γ**: Balances immediate vs. future rewards

The Markov property states that the future depends only on the current state, not the history.`
      },
      {
        id: 'policies',
        title: '2.2 Policies in RL',
        content: `A policy π defines the agent's behavior by mapping states to actions.

Types of policies:
- **Deterministic Policy**: π(s) → a (one action per state)
- **Stochastic Policy**: π(a|s) → [0,1] (probability distribution over actions)

The goal in RL is to find the optimal policy π* that maximizes expected cumulative reward.

Policy representation can be:
- **Tabular**: Lookup table for small state spaces
- **Function Approximation**: Neural networks for large/continuous spaces`
      },
      {
        id: 'rewards',
        title: '2.3 Reward Design',
        content: `Rewards are the feedback signals that guide learning. Good reward design is crucial for successful RL.

Key concepts:
- **Immediate Reward**: r_t received at time t
- **Return (G_t)**: Sum of discounted future rewards
- **Sparse vs. Dense Rewards**: Frequency of non-zero rewards
- **Reward Shaping**: Engineering rewards to guide learning

Common pitfalls:
- Reward hacking: Agent finds unintended shortcuts
- Sparse rewards: Makes learning difficult
- Conflicting objectives: Multiple reward signals that contradict`
      },
      {
        id: 'exploration',
        title: '2.4 Exploration vs. Exploitation',
        content: `The exploration-exploitation dilemma is fundamental in RL: should the agent try new actions (explore) or stick with known good actions (exploit)?

Common strategies:
- **ε-greedy**: Random action with probability ε
- **Boltzmann/Softmax**: Sample based on action values
- **Upper Confidence Bound (UCB)**: Optimism in face of uncertainty
- **Thompson Sampling**: Bayesian approach

In deep RL, exploration is often achieved through:
- Entropy regularization
- Intrinsic motivation
- Curiosity-driven exploration`
      }
    ],
    quiz: [
      {
        id: 'q1',
        question: 'What does the Markov property state?',
        options: [
          'All states must be observable',
          'The future depends only on the current state',
          'Rewards must be deterministic',
          'Actions must be discrete'
        ],
        correctAnswer: 1,
        explanation: 'The Markov property states that the future state depends only on the current state and action, not on the sequence of events that preceded it.'
      },
      {
        id: 'q2',
        question: 'What is the difference between a deterministic and stochastic policy?',
        options: [
          'Deterministic policies are faster to compute',
          'Stochastic policies can only be used with discrete actions',
          'Deterministic policies output one action, stochastic output probabilities',
          'There is no practical difference'
        ],
        correctAnswer: 2,
        explanation: 'A deterministic policy maps each state to exactly one action, while a stochastic policy outputs a probability distribution over actions.'
      },
      {
        id: 'q3',
        question: 'Why is exploration important in reinforcement learning?',
        options: [
          'To make training faster',
          'To discover potentially better actions and states',
          'To reduce memory usage',
          'To ensure deterministic behavior'
        ],
        correctAnswer: 1,
        explanation: 'Exploration allows the agent to discover new states and actions that might lead to better rewards, preventing it from getting stuck in suboptimal solutions.'
      }
    ] as QuizQuestion[],
    exercises: [
      'Implement a simple grid world MDP and visualize the state transitions and rewards.',
      'Create both deterministic and stochastic policies for a simple environment and compare their performance.',
      'Implement and compare different exploration strategies (ε-greedy, Boltzmann, UCB) on a multi-armed bandit problem.'
    ],
    keyTakeaways: [
      'MDPs provide the mathematical framework for sequential decision-making in RL',
      'Policies define agent behavior and can be deterministic or stochastic',
      'Reward design critically impacts what the agent learns to do',
      'Balancing exploration and exploitation is essential for effective learning'
    ]
  },
  '3': {
    title: 'Chapter 3: Value Functions',
    description: 'Understanding state and action value functions, Bellman equations, and value estimation methods',
    objectives: [
      'Understand state value functions V(s) and their role in RL',
      'Master action value functions Q(s,a) and their relationship to V(s)',
      'Learn the Bellman equations and their recursive nature',
      'Compare TD Learning and Monte Carlo methods for value estimation'
    ],
    sections: [
      {
        id: 'state-value-functions',
        title: '3.1 State Value Functions V(s)',
        content: `The state value function V(s) represents the expected return (cumulative reward) when starting from state s and following a particular policy π.

Mathematically:
**V^π(s) = E[G_t | S_t = s]**

Where G_t is the return from time t:
**G_t = R_(t+1) + γR_(t+2) + γ²R_(t+3) + ... = Σ_(k=0)^∞ γ^k R_(t+k+1)**

Key insights:
- V(s) tells us "how good" it is to be in state s
- Higher V(s) means we expect more reward from that state
- V(s) depends on the policy being followed
- The optimal value function V*(s) corresponds to the optimal policy

The value function helps agents make decisions by comparing the values of different states.`
      },
      {
        id: 'action-value-functions',
        title: '3.2 Action Value Functions Q(s,a)',
        content: `The action value function Q(s,a) represents the expected return when taking action a in state s and then following policy π.

Mathematically:
**Q^π(s,a) = E[G_t | S_t = s, A_t = a]**

Relationship between Q and V:
- **V^π(s) = Σ_a π(a|s) Q^π(s,a)** (for stochastic policies)
- **V^π(s) = max_a Q^π(s,a)** (for deterministic policies)

Q-functions are particularly useful because:
- They directly tell us the value of taking each action
- Policy improvement is straightforward: choose arg max_a Q(s,a)
- Many algorithms (like Q-learning, DQN) directly learn Q-functions
- They enable model-free learning

The optimal Q-function Q*(s,a) represents the maximum expected return achievable from (s,a).`
      },
      {
        id: 'bellman-equations',
        title: '3.3 Bellman Equations',
        content: `The Bellman equations express the recursive relationship between value functions at successive time steps. They are the foundation of dynamic programming and many RL algorithms.

**Bellman Expectation Equations:**
For a given policy π:
- V^π(s) = Σ_a π(a|s) Σ_s' P(s'|s,a)[R(s,a,s') + γV^π(s')]
- Q^π(s,a) = Σ_s' P(s'|s,a)[R(s,a,s') + γΣ_a' π(a'|s')Q^π(s',a')]

**Bellman Optimality Equations:**
For the optimal policy π*:
- V*(s) = max_a Σ_s' P(s'|s,a)[R(s,a,s') + γV*(s')]
- Q*(s,a) = Σ_s' P(s'|s,a)[R(s,a,s') + γ max_a' Q*(s',a')]

Key insights:
- Value at current state depends on immediate reward + discounted future value
- These equations form a system that can be solved iteratively
- They enable bootstrapping: using estimates to improve estimates
- Form the basis for value iteration and policy iteration algorithms`
      },
      {
        id: 'value-estimation-methods',
        title: '3.4 TD Learning and Monte Carlo Methods',
        content: `There are two main approaches to estimate value functions from experience: Temporal Difference (TD) Learning and Monte Carlo (MC) methods.

**Monte Carlo Methods:**
- Wait until episode ends to update values
- Use actual returns G_t as targets
- Update: V(s) ← V(s) + α[G_t - V(s)]
- Unbiased but high variance
- Only works for episodic tasks

**Temporal Difference Learning:**
- Update values after each step
- Use bootstrapped estimate as target
- TD(0) update: V(s) ← V(s) + α[r + γV(s') - V(s)]
- Biased but lower variance
- Works for continuing tasks

**Key differences:**
- MC uses complete returns, TD uses estimates
- MC has no bias, TD has bootstrap bias
- MC has high variance, TD has lower variance
- TD can learn online, MC must wait for episode end

**TD(λ) - Best of Both Worlds:**
- Interpolates between TD(0) and MC
- λ=0: pure TD, λ=1: pure MC
- Eligibility traces provide efficient implementation`
      }
    ],
    quiz: [
      {
        id: 'q1',
        question: 'What is the key difference between V(s) and Q(s,a)?',
        options: [
          'V(s) is for continuous spaces, Q(s,a) is for discrete',
          'V(s) evaluates states, Q(s,a) evaluates state-action pairs',
          'V(s) uses neural networks, Q(s,a) uses tables',
          'There is no meaningful difference'
        ],
        correctAnswer: 1,
        explanation: 'V(s) gives the value of being in a state, while Q(s,a) gives the value of taking a specific action in that state. Q(s,a) provides more detailed information needed for action selection.'
      },
      {
        id: 'q2',
        question: 'What is the main advantage of TD learning over Monte Carlo?',
        options: [
          'TD is always more accurate',
          'TD uses less memory',
          'TD can learn online without waiting for episode end',
          'TD is easier to implement'
        ],
        correctAnswer: 2,
        explanation: 'TD learning can update value estimates after each step, enabling online learning and working with continuing tasks. Monte Carlo must wait until the episode ends to know the actual return.'
      },
      {
        id: 'q3',
        question: 'In the Bellman equation, what role does the discount factor γ play?',
        options: [
          'It speeds up learning',
          'It balances immediate vs. future rewards',
          'It reduces memory usage',
          'It prevents overfitting'
        ],
        correctAnswer: 1,
        explanation: 'The discount factor γ (gamma) determines how much we value future rewards compared to immediate rewards. γ close to 0 makes the agent myopic, while γ close to 1 makes it consider long-term consequences.'
      }
    ] as QuizQuestion[],
    exercises: [
      'Implement value iteration and policy iteration for a simple grid world. Compare their convergence rates.',
      'Create a TD(0) learning agent and a Monte Carlo agent for the same environment. Plot their learning curves and analyze the differences.',
      'Implement TD(λ) with different λ values (0, 0.5, 0.9, 1.0) and observe how it affects learning speed and stability.'
    ],
    keyTakeaways: [
      'Value functions V(s) and Q(s,a) quantify how good states and actions are',
      'Bellman equations express the recursive nature of value functions',
      'TD learning enables online learning with bootstrapping, while MC uses actual returns',
      'The bias-variance tradeoff is key when choosing between TD and MC methods'
    ]
  },
  '4': {
    title: 'Chapter 4: Policy Gradients',
    description: 'From value-based to policy-based methods: REINFORCE, baselines, and advanced techniques',
    objectives: [
      'Understand the motivation for policy gradient methods',
      'Master the policy gradient theorem and its derivation',
      'Implement the REINFORCE algorithm from scratch',
      'Learn variance reduction techniques with baselines',
      'Get introduced to natural policy gradient and TRPO'
    ],
    sections: [
      {
        id: 'policy-based-methods',
        title: '4.1 Why Policy-Based Methods?',
        content: `While value-based methods like Q-learning have been successful, they have limitations that policy gradient methods address elegantly.

**Advantages of Policy-Based Methods:**
- **Continuous Action Spaces**: Can naturally handle continuous actions without discretization
- **Stochastic Policies**: Directly learn probabilistic policies, useful for partial observability
- **Convergence Properties**: Better convergence guarantees in certain cases
- **Smooth Optimization**: Small changes in parameters lead to small changes in policy

**Key Differences from Value-Based Methods:**
- Value-based: Learn value function → derive policy (e.g., ε-greedy from Q-values)
- Policy-based: Directly parameterize and optimize the policy π_θ(a|s)

**When to Use Policy Gradients:**
- Continuous or high-dimensional action spaces
- When stochastic policies are needed (e.g., rock-paper-scissors)
- Problems where value function is complex but optimal policy is simple
- When we want guaranteed policy improvement`
      },
      {
        id: 'policy-gradient-theorem',
        title: '4.2 The Policy Gradient Theorem',
        content: `The policy gradient theorem is the foundation of all policy gradient methods. It tells us how to compute gradients of the expected return with respect to policy parameters.

**Objective Function:**
We want to maximize the expected return:
**J(θ) = E_τ~π_θ[R(τ)] = E_τ~π_θ[Σ_(t=0)^T γ^t r_t]**

**The Policy Gradient Theorem:**
The gradient of J(θ) is:
**∇_θ J(θ) = E_τ~π_θ[Σ_(t=0)^T ∇_θ log π_θ(a_t|s_t) G_t]**

Where:
- G_t is the return from time t onward
- ∇_θ log π_θ(a_t|s_t) is the score function

**Key Insight:**
We don't need to know the environment dynamics! The gradient only depends on:
1. The log probability of actions taken
2. The returns received

**Intuition:**
- If an action led to high return → increase its probability
- If an action led to low return → decrease its probability
- The magnitude of change is proportional to how "surprising" the action was (log probability)`
      },
      {
        id: 'reinforce-algorithm',
        title: '4.3 REINFORCE Algorithm',
        content: `REINFORCE is the simplest policy gradient algorithm, directly implementing the policy gradient theorem using Monte Carlo returns.

**Algorithm Steps:**
1. Initialize policy network π_θ(a|s) with random parameters θ
2. For each episode:
   - Generate trajectory τ = (s_0, a_0, r_0, ..., s_T) using π_θ
   - Calculate returns G_t for each timestep
   - Update parameters: θ ← θ + α ∇_θ J(θ)

**REINFORCE Update Rule:**
**θ ← θ + α Σ_(t=0)^T ∇_θ log π_θ(a_t|s_t) G_t**

**Properties:**
- Unbiased gradient estimates
- High variance (uses full episode returns)
- On-policy: must generate new data after each update
- Works with any differentiable policy parameterization

**Common Issues:**
- High variance leads to unstable training
- Sample inefficient (needs many episodes)
- Sensitive to hyperparameters (learning rate, initialization)`
      },
      {
        id: 'variance-reduction',
        title: '4.4 Variance Reduction with Baselines',
        content: `High variance is the main challenge with REINFORCE. Baselines are a powerful technique to reduce variance without introducing bias.

**The Problem:**
Even if all actions in an episode are good, REINFORCE will increase probabilities of all actions. We want to increase probabilities only for actions that are better than average.

**Baseline Subtraction:**
Modify the policy gradient to:
**∇_θ J(θ) = E_τ~π_θ[Σ_(t=0)^T ∇_θ log π_θ(a_t|s_t) (G_t - b(s_t))]**

Where b(s_t) is a baseline that depends only on the state.

**Common Baseline Choices:**
1. **Constant baseline**: b = average return
2. **Value function baseline**: b(s_t) = V^π(s_t)
3. **Moving average**: b = exponential moving average of returns

**Why It Works:**
- Subtracting baseline doesn't change expected gradient (remains unbiased)
- Reduces variance by centering the returns
- Actions better than baseline → positive gradient
- Actions worse than baseline → negative gradient

**Advantage Function:**
When using value function as baseline:
**A^π(s_t, a_t) = Q^π(s_t, a_t) - V^π(s_t) = G_t - V^π(s_t)**

This is called the advantage function, measuring how much better an action is compared to the average.`
      },
      {
        id: 'advanced-methods-preview',
        title: '4.5 Natural Policy Gradient and TRPO Preview',
        content: `Standard policy gradients can be inefficient because parameter space and policy space have different geometries. Advanced methods address this.

**Natural Policy Gradient:**
Instead of steepest descent in parameter space, take steepest descent in policy distribution space:
**θ_(k+1) = θ_k + α F^(-1) ∇_θ J(θ)**

Where F is the Fisher Information Matrix:
**F = E_[s~ρ^π, a~π][∇_θ log π_θ(a|s) ∇_θ log π_θ(a|s)^T]**

**Benefits:**
- Invariant to parameterization
- More stable and efficient updates
- Natural measure of distance between policies

**Trust Region Policy Optimization (TRPO):**
TRPO ensures monotonic improvement by constraining policy updates:
**maximize_θ E[A^π_old(s,a)]**
**subject to: KL(π_old || π_θ) ≤ δ**

This prevents destructively large policy updates.

**Key Ideas Leading to PPO:**
- Natural gradients provide better update directions
- Trust regions ensure stable learning
- Computational efficiency is crucial for practical use
- These ideas culminate in PPO, which we'll study in detail

**Why This Matters:**
Understanding these advanced methods helps appreciate why PPO makes the design choices it does, balancing theoretical soundness with practical efficiency.`
      }
    ],
    quiz: [
      {
        id: 'q1',
        question: 'What is the main advantage of policy gradient methods over value-based methods?',
        options: [
          'They always converge faster',
          'They can naturally handle continuous action spaces',
          'They require less memory',
          'They don\'t need a neural network'
        ],
        correctAnswer: 1,
        explanation: 'Policy gradient methods directly parameterize the policy, making them natural for continuous action spaces. Value-based methods would need to discretize continuous actions or use complex techniques like DDPG.'
      },
      {
        id: 'q2',
        question: 'Why do we use baselines in REINFORCE?',
        options: [
          'To make the algorithm faster',
          'To handle continuous states',
          'To reduce variance without introducing bias',
          'To enable off-policy learning'
        ],
        correctAnswer: 2,
        explanation: 'Baselines reduce the variance of gradient estimates by centering the returns. Crucially, when the baseline depends only on states (not actions), it doesn\'t introduce bias to the gradient.'
      },
      {
        id: 'q3',
        question: 'What does the score function ∇_θ log π_θ(a|s) represent?',
        options: [
          'The value of taking action a in state s',
          'The direction to change θ to increase probability of action a',
          'The reward for taking action a',
          'The advantage of action a over other actions'
        ],
        correctAnswer: 1,
        explanation: 'The score function ∇_θ log π_θ(a|s) is the gradient of log probability with respect to parameters. It points in the direction that would increase the probability of taking action a in state s.'
      }
    ] as QuizQuestion[],
    exercises: [
      'Implement REINFORCE for CartPole. Compare performance with and without a baseline. Plot the learning curves and variance of gradient estimates.',
      'Derive the policy gradient theorem step by step. Show why subtracting a state-dependent baseline doesn\'t introduce bias.',
      'Implement a Gaussian policy for continuous control. Experiment with different ways of parameterizing the variance (fixed, state-dependent, learned).',
      'Compare REINFORCE with a value-based method (like DQN) on the same environment. Analyze the trade-offs in terms of sample efficiency and final performance.'
    ],
    keyTakeaways: [
      'Policy gradient methods directly optimize the policy without learning a value function',
      'The policy gradient theorem provides an unbiased way to estimate gradients using samples',
      'REINFORCE is simple but suffers from high variance, making baselines essential',
      'Natural policy gradient and trust regions address efficiency and stability issues',
      'These concepts form the foundation for understanding PPO\'s design choices'
    ]
  },
  '5': {
    title: 'Chapter 5: Actor-Critic',
    description: 'Combining value and policy methods for powerful hybrid algorithms',
    objectives: [
      'Understand why combining actor and critic is beneficial',
      'Master the A2C (Advantage Actor-Critic) algorithm',
      'Learn A3C (Asynchronous Advantage Actor-Critic) for parallel training',
      'Understand Generalized Advantage Estimation (GAE) and its bias-variance tradeoff',
      'See how actor-critic methods lead naturally to PPO'
    ],
    sections: [
      {
        id: 'actor-critic-motivation',
        title: '5.1 Why Actor-Critic Methods?',
        content: `Actor-critic methods combine the best of both worlds: policy gradient methods (actor) and value function methods (critic). This hybrid approach addresses key limitations of each individual approach.

**Limitations of Pure Policy Gradient Methods:**
- High variance in gradient estimates (REINFORCE)
- Sample inefficient - needs many episodes
- No bootstrapping - must wait for episode completion
- Difficult to judge action quality without baseline

**Limitations of Pure Value-Based Methods:**
- Indirect policy representation (need to derive from values)
- Difficulty with continuous actions
- Can be unstable with function approximation
- Exploration often requires ad-hoc methods (ε-greedy)

**Actor-Critic Solution:**
The actor-critic architecture uses:
- **Actor**: Policy network π_θ(a|s) that selects actions
- **Critic**: Value network V_φ(s) that evaluates states
- The critic provides a learned baseline for variance reduction
- The actor directly optimizes the policy

**Key Benefits:**
1. **Lower Variance**: Critic provides state-dependent baseline
2. **Online Learning**: Can use TD learning, no need to wait for episode end
3. **Stable Learning**: Critic stabilizes actor updates
4. **Natural Architecture**: Separates "what to do" from "how good is it"`
      },
      {
        id: 'a2c-algorithm',
        title: '5.2 A2C (Advantage Actor-Critic)',
        content: `A2C is the synchronous version of actor-critic that updates both networks using advantages computed from the critic's value estimates.

**Core Components:**
1. **Actor Network**: π_θ(a|s) - outputs action probabilities
2. **Critic Network**: V_φ(s) - outputs state value estimate
3. **Advantage Function**: A(s,a) = R + γV(s') - V(s)

**The A2C Algorithm:**
1. Collect batch of experiences using current policy
2. Compute returns and advantages
3. Update critic to minimize value prediction error
4. Update actor using policy gradient with advantages
5. Repeat

**Mathematical Framework:**
- Critic Loss: L_critic = E[(R + γV_φ(s') - V_φ(s))²]
- Actor Loss: L_actor = -E[log π_θ(a|s) * A(s,a)]
- Total Loss: L = L_actor + c₁*L_critic - c₂*H(π)

Where H(π) is entropy bonus for exploration.

**Key Design Choices:**
- Shared or separate networks for actor/critic
- n-step returns vs 1-step TD
- Entropy regularization strength
- Gradient clipping for stability`
      },
      {
        id: 'a3c-algorithm',
        title: '5.3 A3C (Asynchronous Advantage Actor-Critic)',
        content: `A3C parallelizes actor-critic training across multiple workers, each exploring different parts of the environment simultaneously.

**Architecture:**
- Multiple worker threads/processes
- Each worker has a copy of actor-critic networks
- Workers interact with separate environment instances
- Asynchronously update shared global networks

**A3C Algorithm Flow:**
1. Each worker:
   - Copies global network parameters
   - Collects trajectory using local policy
   - Computes gradients locally
   - Updates global network asynchronously
2. Global network aggregates updates from all workers

**Benefits of Asynchronous Training:**
- **Decorrelation**: Workers explore different states
- **Speed**: Parallel environment interaction
- **Stability**: Natural exploration from different workers
- **Robustness**: Less sensitive to individual trajectories

**Implementation Considerations:**
- Thread/process synchronization
- Gradient accumulation strategies
- Worker update frequency
- Handling stale gradients

**Modern Alternative - A2C with Parallel Environments:**
Instead of asynchronous updates, use synchronous updates with parallel environment collection. This is often more stable and easier to debug.`
      },
      {
        id: 'gae-generalized-advantage',
        title: '5.4 Generalized Advantage Estimation (GAE)',
        content: `GAE is a crucial innovation that provides a principled way to balance bias and variance in advantage estimation. It's a key component of modern algorithms including PPO.

**The Bias-Variance Tradeoff in Advantages:**

**High Variance (Monte Carlo):**
A^MC = G_t - V(s_t) = Σ_(k=0)^∞ γ^k r_(t+k) - V(s_t)
- Unbiased but high variance
- Uses actual returns

**High Bias (1-step TD):**
A^TD = r_t + γV(s_(t+1)) - V(s_t)
- Low variance but biased
- Uses bootstrapped value estimates

**GAE - Best of Both Worlds:**
GAE uses exponentially weighted average of n-step advantages:
**A^GAE(γ,λ) = Σ_(l=0)^∞ (γλ)^l δ_(t+l)**

Where δ_t = r_t + γV(s_(t+1)) - V(s_t) is the TD residual.

**Expanded Form:**
A^GAE = δ_t + (γλ)δ_(t+1) + (γλ)²δ_(t+2) + ...

**Key Properties:**
- λ = 0: Reduces to 1-step TD (high bias, low variance)
- λ = 1: Reduces to Monte Carlo (low bias, high variance)
- λ ∈ (0,1): Interpolates between extremes

**Why GAE Works:**
1. Exponential weighting emphasizes near-term advantages
2. Parameter λ allows fine-tuning bias-variance tradeoff
3. Efficient recursive computation
4. Empirically more stable than alternatives

**Practical Tips:**
- Typical λ values: 0.95-0.99
- Lower λ for noisy environments
- Higher λ for deterministic environments
- Critical for PPO's performance`
      },
      {
        id: 'implementation-details',
        title: '5.5 Implementation and Practical Considerations',
        content: `Implementing actor-critic methods requires careful attention to details that significantly impact performance.

**Network Architecture Choices:**

**Shared Architecture:**
- Actor and critic share feature extraction layers
- Separate heads for policy and value
- More parameter efficient
- Can lead to conflicting gradients

**Separate Architecture:**
- Independent networks for actor and critic
- More flexible optimization
- Higher memory usage
- Often more stable

**Normalization Techniques:**
1. **Advantage Normalization**: Normalize advantages per batch
2. **Value Function Scaling**: Normalize returns for critic training
3. **Observation Normalization**: Running statistics of inputs
4. **Gradient Clipping**: Prevent destructive updates

**Common Pitfalls and Solutions:**
1. **Critic Overfitting**: 
   - Use separate learning rates (critic often needs higher)
   - Early stopping for critic
   - Regularization (L2, dropout)

2. **Entropy Collapse**:
   - Maintain minimum entropy
   - Adaptive entropy coefficient
   - Action noise for exploration

3. **Advantage Estimation Issues**:
   - Ensure correct GAE implementation
   - Verify value function quality
   - Check for numerical instabilities

**Hyperparameter Guidelines:**
- Actor LR: 3e-4 (typically)
- Critic LR: 1e-3 (often higher than actor)
- Entropy coefficient: 0.01 (environment dependent)
- GAE λ: 0.95 (good default)
- Discount γ: 0.99 (for long horizons)`
      }
    ],
    quiz: [
      {
        id: 'q1',
        question: 'What is the main advantage of actor-critic over pure policy gradient methods?',
        options: [
          'It can handle discrete actions better',
          'It reduces variance through learned value baselines',
          'It uses less memory',
          'It doesn\'t require neural networks'
        ],
        correctAnswer: 1,
        explanation: 'Actor-critic methods use the critic\'s value estimates as a learned baseline, significantly reducing variance compared to REINFORCE while maintaining the benefits of direct policy optimization.'
      },
      {
        id: 'q2',
        question: 'In GAE, what happens when λ = 0?',
        options: [
          'It becomes equivalent to Monte Carlo returns',
          'It becomes equivalent to 1-step TD',
          'The advantages become zero',
          'Training becomes unstable'
        ],
        correctAnswer: 1,
        explanation: 'When λ = 0, GAE reduces to 1-step TD advantages: A = r + γV(s\') - V(s). This has low variance but high bias due to bootstrapping.'
      },
      {
        id: 'q3',
        question: 'Why might A3C\'s asynchronous updates be problematic?',
        options: [
          'They use too much memory',
          'They can\'t handle continuous actions',
          'Stale gradients and synchronization issues can hurt stability',
          'They only work with discrete action spaces'
        ],
        correctAnswer: 2,
        explanation: 'A3C\'s asynchronous updates can lead to stale gradients where workers update based on old network parameters. This is why many practitioners now prefer synchronous A2C with parallel environments.'
      }
    ] as QuizQuestion[],
    exercises: [
      'Implement A2C for CartPole. Compare performance with and without GAE. Plot learning curves and advantage variance.',
      'Create an A3C implementation with multiple workers. Compare convergence speed and stability with single-threaded A2C.',
      'Experiment with different GAE λ values (0, 0.5, 0.9, 0.95, 0.99, 1.0) on a continuous control task. Analyze the bias-variance tradeoff.',
      'Implement both shared and separate actor-critic architectures. Compare their performance and gradient conflicts.'
    ],
    keyTakeaways: [
      'Actor-critic combines policy gradients with value function approximation for lower variance',
      'A2C provides stable synchronous updates while A3C offers parallel asynchronous training',
      'GAE is crucial for balancing bias and variance in advantage estimation',
      'The critic acts as a learned baseline, making policy gradient updates more efficient',
      'These concepts directly lead to PPO, which adds trust region constraints to actor-critic'
    ]
  },
  '6': {
    title: 'Chapter 6: PPO Algorithm',
    description: 'The complete PPO algorithm - understanding clipping, trust regions, and implementation',
    objectives: [
      'Master the PPO objective function with clipping mechanism',
      'Understand why clipping creates an implicit trust region',
      'Learn the complete PPO algorithm with multiple epochs and minibatches',
      'Implement a production-ready PPO from scratch',
      'Master hyperparameter tuning and best practices'
    ],
    sections: [
      {
        id: 'ppo-objective',
        title: '6.1 The PPO Objective Function',
        content: `PPO's genius lies in its clipped objective function, which prevents destructive policy updates while maintaining simplicity. Let's build up to it step by step.

**The Problem with Vanilla Policy Gradient:**
In standard policy gradient methods, we optimize:
L^PG(θ) = E[log π_θ(a|s) * A(s,a)]

This can lead to catastrophically large updates if advantages are large, destroying previously learned behaviors.

**The Importance Sampling Ratio:**
PPO uses the ratio between new and old policies:
r(θ) = π_θ(a|s) / π_θ_old(a|s)

This allows us to reuse old data multiple times (crucial for sample efficiency).

**The Clipped Surrogate Objective:**
PPO's key innovation is clipping this ratio:

L^CLIP(θ) = E[min(r(θ) * A(s,a), clip(r(θ), 1-ε, 1+ε) * A(s,a))]

Where ε (epsilon) is typically 0.2.

**Breaking Down the Clipping:**
- When A > 0 (good action): Prevents ratio from exceeding 1+ε
- When A < 0 (bad action): Prevents ratio from going below 1-ε
- This creates a "trust region" without complex constraints

**The Complete PPO Loss:**
L^PPO(θ) = E[L^CLIP(θ) - c₁ * L^VF(θ) + c₂ * S[π_θ]]

Where:
- L^VF: Value function loss (MSE)
- S: Entropy bonus for exploration
- c₁, c₂: Coefficients (typically 0.5, 0.01)

**Why This Works:**
1. Prevents destructive updates by limiting policy change
2. Allows multiple epochs of updates on same data
3. Computationally efficient (no second-order optimization)
4. Empirically robust across many domains`
      },
      {
        id: 'trust-region-intuition',
        title: '6.2 Trust Region Intuition',
        content: `Understanding why PPO's clipping creates an effective trust region is crucial for mastering the algorithm.

**What is a Trust Region?**
A trust region is a constraint on how much the policy can change in a single update. It ensures we don't take steps so large that our approximations become invalid.

**The Problem with Large Policy Updates:**
1. **Approximation Error**: Policy gradient uses first-order approximation
2. **Distribution Shift**: New policy generates different state distribution
3. **Catastrophic Forgetting**: Can destroy good behaviors learned earlier

**How Clipping Creates a Trust Region:**

Consider the clipped objective for a single state-action pair:
- Original: r(θ) * A
- Clipped: min(r(θ) * A, clip(r(θ), 1-ε, 1+ε) * A)

**Case Analysis:**

**When Advantage is Positive (A > 0):**
- We want to increase π(a|s)
- But only until r(θ) = 1+ε
- Beyond that, no gradient signal
- This prevents greedy exploitation

**When Advantage is Negative (A < 0):**
- We want to decrease π(a|s)
- But only until r(θ) = 1-ε
- This prevents over-correction

**Visualizing the Objective:**
The clipped objective creates a "pessimistic" bound:
- Takes minimum of clipped and unclipped objectives
- Creates flat regions where updates stop
- Automatically adjusts constraint based on advantage sign

**Comparison with TRPO:**
- TRPO: Hard KL constraint, complex optimization
- PPO: Soft constraint via clipping, simple implementation
- Both achieve similar goals, PPO is more practical

**Adaptive Trust Region:**
PPO's trust region adapts based on:
- Advantage magnitude (larger advantages → stronger effect)
- Current policy ratio (far from old policy → clipping activates)
- This creates a dynamic, context-aware constraint`
      },
      {
        id: 'complete-ppo-algorithm',
        title: '6.3 The Complete PPO Algorithm',
        content: `Here's the full PPO algorithm, combining all concepts from previous chapters into a powerful, practical method.

**PPO Algorithm Overview:**

1. **Initialize** actor-critic networks π_θ and V_φ
2. **For** each iteration:
   a. Collect trajectories using current policy
   b. Compute advantages using GAE
   c. Optimize for K epochs using minibatches
   d. Update old policy for next iteration

**Detailed Algorithm:**

\`\`\`
Algorithm: Proximal Policy Optimization (PPO)
────────────────────────────────────────────
Initialize: policy π_θ, value function V_φ
Hyperparameters: 
  - T: trajectory length
  - K: optimization epochs
  - M: minibatch size
  - ε: clipping parameter
  - γ: discount factor
  - λ: GAE parameter

for iteration = 1, 2, ... do
  # Collect trajectories
  for actor = 1, ..., N do
    Run policy π_θ_old for T timesteps
    Store {s_t, a_t, r_t, logprob_t, v_t}
  
  # Compute advantages
  for t in [T-1, T-2, ..., 0] do
    δ_t = r_t + γ * V(s_(t+1)) - V(s_t)
    A_t = δ_t + (γλ) * A_(t+1)
  
  # Normalize advantages
  A = (A - mean(A)) / (std(A) + ε)
  
  # Optimization phase
  θ_old ← θ
  for epoch = 1, ..., K do
    Shuffle data into minibatches
    for minibatch in minibatches do
      # Compute ratio
      r(θ) = π_θ(a|s) / π_θ_old(a|s)
      
      # Clipped objective
      L_clip = min(r(θ)*A, clip(r(θ), 1-ε, 1+ε)*A)
      
      # Value loss
      L_vf = (V_φ(s) - R)²
      
      # Entropy bonus
      L_entropy = -S[π_θ]
      
      # Total loss
      L = -L_clip + c₁*L_vf - c₂*L_entropy
      
      # Update networks
      θ ← θ - α * ∇_θ L
      φ ← φ - α * ∇_φ L
end algorithm
\`\`\`

**Key Implementation Details:**

1. **Parallel Environment Collection:**
   - Run N parallel environments
   - Decorrelates samples
   - Improves efficiency

2. **Advantage Normalization:**
   - Stabilizes training
   - Makes hyperparameters more robust
   - Per-minibatch or per-epoch

3. **Early Stopping:**
   - Monitor KL divergence
   - Stop epochs if KL > target
   - Prevents overfitting to old data

4. **Gradient Clipping:**
   - Clip gradients by global norm
   - Typical value: 0.5
   - Prevents exploding gradients

5. **Learning Rate Schedule:**
   - Linear or cosine annealing
   - Helps convergence
   - Prevents late-stage instability`
      },
      {
        id: 'implementation-details',
        title: '6.4 Implementation Details and Hyperparameters',
        content: `Successful PPO implementation requires careful attention to details and proper hyperparameter selection.

**Critical Implementation Details:**

**1. Action Sampling and Log Probabilities:**
\`\`\`python
# Discrete actions
logits = actor_network(state)
dist = Categorical(logits=logits)
action = dist.sample()
log_prob = dist.log_prob(action)

# Continuous actions
mean, std = actor_network(state)
dist = Normal(mean, std)
action = dist.sample()
log_prob = dist.log_prob(action).sum(-1)  # Sum over action dimensions
\`\`\`

**2. Advantage Computation with GAE:**
\`\`\`python
def compute_gae(rewards, values, dones, gamma=0.99, lam=0.95):
    advantages = torch.zeros_like(rewards)
    lastgaelam = 0
    
    for t in reversed(range(len(rewards))):
        if t == len(rewards) - 1:
            nextnonterminal = 1.0 - dones[t]
            nextvalues = 0  # Bootstrap from last state
        else:
            nextnonterminal = 1.0 - dones[t]
            nextvalues = values[t + 1]
        
        delta = rewards[t] + gamma * nextvalues * nextnonterminal - values[t]
        advantages[t] = lastgaelam = delta + gamma * lam * nextnonterminal * lastgaelam
    
    returns = advantages + values
    return advantages, returns
\`\`\`

**3. Ratio Calculation with Numerical Stability:**
\`\`\`python
def compute_ratio(new_log_prob, old_log_prob):
    # Use log-space for numerical stability
    ratio = torch.exp(new_log_prob - old_log_prob)
    # Clip extreme ratios for additional safety
    ratio = torch.clamp(ratio, min=0.01, max=100.0)
    return ratio
\`\`\`

**Key Hyperparameters and Typical Values:**

**Core PPO Parameters:**
- clip_epsilon: 0.2 (range: 0.1-0.3)
- epochs: 4 (range: 3-10)
- minibatch_size: 64 (range: 32-256)
- gamma: 0.99 (discount factor)
- lambda_gae: 0.95 (GAE parameter)

**Optimization Parameters:**
- learning_rate: 3e-4 (often with annealing)
- value_loss_coef: 0.5
- entropy_coef: 0.01 (task-dependent)
- max_grad_norm: 0.5 (gradient clipping)

**Environment Interaction:**
- num_envs: 8-64 (parallel environments)
- rollout_length: 128-2048 steps
- total_timesteps: 1e6-1e8 (task-dependent)

**Common Tricks and Best Practices:**

1. **Orthogonal Initialization:**
   - Initialize weights with orthogonal matrices
   - Helps gradient flow
   - Especially important for LSTM/GRU

2. **Feature Normalization:**
   - Running mean/std of observations
   - Or use layer normalization
   - Critical for some environments

3. **Reward Scaling/Clipping:**
   - Clip rewards to [-1, 1] or [-10, 10]
   - Or use reward normalization
   - Prevents value function explosion

4. **Shared vs Separate Networks:**
   - Shared: More efficient, good for simple tasks
   - Separate: Better for complex tasks
   - Consider task requirements

5. **Invalid Action Masking:**
   - For environments with invalid actions
   - Mask logits before softmax
   - Maintains exploration in valid space`
      },
      {
        id: 'full-implementation',
        title: '6.5 Complete PPO Implementation',
        content: `Let's implement a production-ready PPO that brings together everything we've learned.

**Full PPO Implementation:**

\`\`\`python
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical, Normal
import numpy as np
from collections import deque

class PPO:
    def __init__(
        self,
        env_fn,
        actor_critic,
        lr=3e-4,
        gamma=0.99,
        lambda_gae=0.95,
        clip_epsilon=0.2,
        value_loss_coef=0.5,
        entropy_coef=0.01,
        max_grad_norm=0.5,
        num_epochs=4,
        num_minibatches=4,
        rollout_length=2048,
        num_envs=8,
        device='cuda'
    ):
        self.device = device
        self.num_envs = num_envs
        self.rollout_length = rollout_length
        self.gamma = gamma
        self.lambda_gae = lambda_gae
        self.clip_epsilon = clip_epsilon
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.num_epochs = num_epochs
        self.num_minibatches = num_minibatches
        
        # Create environments
        self.envs = [env_fn() for _ in range(num_envs)]
        
        # Networks
        self.actor_critic = actor_critic.to(device)
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=lr)
        
        # Storage
        self.storage = RolloutStorage(
            rollout_length, num_envs,
            self.envs[0].observation_space.shape,
            self.envs[0].action_space,
            device
        )
        
    def collect_rollouts(self):
        """Collect trajectories using current policy."""
        for step in range(self.rollout_length):
            with torch.no_grad():
                obs_tensor = self.storage.obs[step]
                
                # Get action distribution and value
                if hasattr(self.envs[0].action_space, 'n'):
                    # Discrete action space
                    logits, values = self.actor_critic(obs_tensor)
                    dist = Categorical(logits=logits)
                    actions = dist.sample()
                    log_probs = dist.log_prob(actions)
                else:
                    # Continuous action space
                    mean, std, values = self.actor_critic(obs_tensor)
                    dist = Normal(mean, std)
                    actions = dist.sample()
                    log_probs = dist.log_prob(actions).sum(-1)
                
            # Execute actions in environments
            cpu_actions = actions.cpu().numpy()
            obs_list, rewards, dones, infos = [], [], [], []
            
            for i, env in enumerate(self.envs):
                obs, reward, done, info = env.step(cpu_actions[i])
                if done:
                    obs = env.reset()
                obs_list.append(obs)
                rewards.append(reward)
                dones.append(done)
                infos.append(info)
            
            # Store transition
            self.storage.insert(
                torch.tensor(obs_list, device=self.device),
                actions,
                log_probs,
                values,
                torch.tensor(rewards, device=self.device).unsqueeze(1),
                torch.tensor(dones, device=self.device).unsqueeze(1)
            )
    
    def compute_returns(self):
        """Compute returns and advantages using GAE."""
        with torch.no_grad():
            next_obs = self.storage.obs[-1]
            _, next_value = self.actor_critic(next_obs)
            self.storage.compute_returns_and_advantages(
                next_value, self.gamma, self.lambda_gae
            )
    
    def update(self):
        """Update policy and value networks."""
        # Prepare data
        obs_shape = self.storage.obs.shape[2:]
        action_shape = self.storage.actions.shape[2:]
        rollout_data = self.storage.get_rollout_data()
        
        # Optimization epochs
        for epoch in range(self.num_epochs):
            # Generate minibatches
            batch_size = self.num_envs * self.rollout_length
            minibatch_size = batch_size // self.num_minibatches
            indices = np.random.permutation(batch_size)
            
            for start in range(0, batch_size, minibatch_size):
                end = start + minibatch_size
                mb_indices = indices[start:end]
                
                mb_obs = rollout_data.obs[mb_indices]
                mb_actions = rollout_data.actions[mb_indices]
                mb_old_log_probs = rollout_data.log_probs[mb_indices]
                mb_advantages = rollout_data.advantages[mb_indices]
                mb_returns = rollout_data.returns[mb_indices]
                
                # Forward pass
                if hasattr(self.envs[0].action_space, 'n'):
                    logits, values = self.actor_critic(mb_obs)
                    dist = Categorical(logits=logits)
                else:
                    mean, std, values = self.actor_critic(mb_obs)
                    dist = Normal(mean, std)
                
                # Compute losses
                new_log_probs = dist.log_prob(mb_actions)
                if len(new_log_probs.shape) > 1:
                    new_log_probs = new_log_probs.sum(-1)
                
                # Ratio and clipped objective
                ratio = torch.exp(new_log_probs - mb_old_log_probs)
                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * mb_advantages
                actor_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss
                value_loss = F.mse_loss(values.squeeze(), mb_returns)
                
                # Entropy loss
                entropy_loss = -dist.entropy().mean()
                
                # Total loss
                loss = actor_loss + self.value_loss_coef * value_loss + self.entropy_coef * entropy_loss
                
                # Optimize
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
                self.optimizer.step()
    
    def train(self, total_timesteps):
        """Main training loop."""
        num_updates = total_timesteps // (self.num_envs * self.rollout_length)
        
        # Initialize environments
        obs_list = [env.reset() for env in self.envs]
        self.storage.obs[0] = torch.tensor(obs_list, device=self.device)
        
        for update in range(num_updates):
            # Collect rollouts
            self.collect_rollouts()
            
            # Compute returns and advantages
            self.compute_returns()
            
            # Update networks
            self.update()
            
            # Copy first observation of next rollout
            self.storage.obs[0].copy_(self.storage.obs[-1])
            
            # Logging (every 10 updates)
            if update % 10 == 0:
                print(f"Update {update}/{num_updates}")


class RolloutStorage:
    """Storage for rollout data."""
    def __init__(self, rollout_length, num_envs, obs_shape, action_space, device):
        self.rollout_length = rollout_length
        self.num_envs = num_envs
        self.device = device
        
        # Storage buffers
        self.obs = torch.zeros(rollout_length + 1, num_envs, *obs_shape, device=device)
        if hasattr(action_space, 'n'):
            self.actions = torch.zeros(rollout_length, num_envs, 1, dtype=torch.long, device=device)
        else:
            self.actions = torch.zeros(rollout_length, num_envs, action_space.shape[0], device=device)
        self.log_probs = torch.zeros(rollout_length, num_envs, 1, device=device)
        self.values = torch.zeros(rollout_length, num_envs, 1, device=device)
        self.rewards = torch.zeros(rollout_length, num_envs, 1, device=device)
        self.dones = torch.zeros(rollout_length, num_envs, 1, device=device)
        self.advantages = torch.zeros(rollout_length, num_envs, 1, device=device)
        self.returns = torch.zeros(rollout_length, num_envs, 1, device=device)
        
        self.step = 0
    
    def insert(self, obs, actions, log_probs, values, rewards, dones):
        """Insert transition into storage."""
        self.obs[self.step + 1] = obs
        self.actions[self.step] = actions.unsqueeze(-1) if actions.dim() == 1 else actions
        self.log_probs[self.step] = log_probs.unsqueeze(-1)
        self.values[self.step] = values
        self.rewards[self.step] = rewards
        self.dones[self.step] = dones
        self.step = (self.step + 1) % self.rollout_length
    
    def compute_returns_and_advantages(self, next_value, gamma, lambda_gae):
        """Compute returns and GAE advantages."""
        self.values[-1] = next_value
        gae = 0
        
        for step in reversed(range(self.rollout_length)):
            next_value = self.values[step + 1]
            next_non_terminal = 1.0 - self.dones[step]
            
            delta = self.rewards[step] + gamma * next_value * next_non_terminal - self.values[step]
            gae = delta + gamma * lambda_gae * next_non_terminal * gae
            self.advantages[step] = gae
            self.returns[step] = gae + self.values[step]
        
        # Normalize advantages
        self.advantages = (self.advantages - self.advantages.mean()) / (self.advantages.std() + 1e-8)
    
    def get_rollout_data(self):
        """Get all rollout data in a flat format."""
        return SimpleNamespace(
            obs=self.obs[:-1].reshape(-1, *self.obs.shape[2:]),
            actions=self.actions.reshape(-1, *self.actions.shape[2:]),
            log_probs=self.log_probs.reshape(-1, 1),
            advantages=self.advantages.reshape(-1, 1),
            returns=self.returns.reshape(-1, 1)
        )
\`\`\`

**Usage Example:**
\`\`\`python
# Create actor-critic network
actor_critic = ActorCritic(
    obs_dim=env.observation_space.shape[0],
    action_dim=env.action_space.n,
    hidden_dim=64
)

# Initialize PPO
ppo = PPO(
    env_fn=lambda: gym.make('CartPole-v1'),
    actor_critic=actor_critic,
    num_envs=8,
    rollout_length=128
)

# Train
ppo.train(total_timesteps=1_000_000)
\`\`\`

This implementation includes all the critical components for a working PPO agent!`
      }
    ],
    codeExamples: [
      {
        title: 'PPO Clipping Mechanism',
        language: 'python',
        code: `def ppo_clip_objective(
    new_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    clip_epsilon: float = 0.2
) -> torch.Tensor:
    """
    Compute PPO's clipped surrogate objective.
    
    The clipping prevents large policy updates by creating
    a pessimistic bound on the objective.
    """
    # Compute probability ratio
    ratio = torch.exp(new_log_probs - old_log_probs)
    
    # Unclipped objective
    surr1 = ratio * advantages
    
    # Clipped objective
    surr2 = torch.clamp(
        ratio,
        1.0 - clip_epsilon,
        1.0 + clip_epsilon
    ) * advantages
    
    # Take minimum (pessimistic bound)
    clipped_objective = torch.min(surr1, surr2)
    
    return clipped_objective.mean()

# Visualization of clipping behavior
def visualize_clipping(advantage=1.0, epsilon=0.2):
    """Show how clipping affects the objective."""
    ratios = torch.linspace(0.5, 2.0, 100)
    
    # Unclipped objective
    unclipped = ratios * advantage
    
    # Clipped objective
    if advantage > 0:
        clipped = torch.minimum(
            ratios * advantage,
            (1 + epsilon) * advantage * torch.ones_like(ratios)
        )
    else:
        clipped = torch.maximum(
            ratios * advantage,
            (1 - epsilon) * advantage * torch.ones_like(ratios)
        )
    
    # The actual PPO objective takes the minimum
    ppo_objective = torch.minimum(unclipped, clipped)
    
    return ratios, unclipped, clipped, ppo_objective`
      },
      {
        title: 'Actor-Critic Network for PPO',
        language: 'python',
        code: `class ActorCritic(nn.Module):
    """
    Combined actor-critic network for PPO.
    Can handle both discrete and continuous actions.
    """
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int = 64,
        continuous: bool = False,
        activation: nn.Module = nn.Tanh
    ):
        super().__init__()
        self.continuous = continuous
        
        # Shared feature extractor
        self.features = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            activation(),
            nn.Linear(hidden_dim, hidden_dim),
            activation()
        )
        
        # Actor head
        if continuous:
            self.actor_mean = nn.Linear(hidden_dim, action_dim)
            self.actor_logstd = nn.Parameter(
                torch.zeros(1, action_dim)
            )
        else:
            self.actor = nn.Linear(hidden_dim, action_dim)
        
        # Critic head
        self.critic = nn.Linear(hidden_dim, 1)
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        """Orthogonal initialization for better training."""
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
            nn.init.constant_(module.bias, 0.0)
    
    def forward(self, obs):
        """Forward pass returning action distribution and value."""
        features = self.features(obs)
        
        if self.continuous:
            mean = self.actor_mean(features)
            std = torch.exp(self.actor_logstd).expand_as(mean)
            value = self.critic(features)
            return mean, std, value
        else:
            logits = self.actor(features)
            value = self.critic(features)
            return logits, value
    
    def get_action_and_value(self, obs):
        """Helper method for rollout collection."""
        if self.continuous:
            mean, std, value = self(obs)
            dist = Normal(mean, std)
            action = dist.sample()
            log_prob = dist.log_prob(action).sum(-1)
        else:
            logits, value = self(obs)
            dist = Categorical(logits=logits)
            action = dist.sample()
            log_prob = dist.log_prob(action)
        
        return action, log_prob, value.squeeze(-1), dist.entropy()`
      },
      {
        title: 'PPO Training Utilities',
        language: 'python',
        code: `class PPOTrainer:
    """Utilities for PPO training."""
    
    @staticmethod
    def compute_gae(
        rewards: torch.Tensor,
        values: torch.Tensor,
        dones: torch.Tensor,
        next_value: torch.Tensor,
        gamma: float = 0.99,
        lambda_: float = 0.95
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute Generalized Advantage Estimation (GAE).
        
        GAE balances bias and variance in advantage estimates
        using exponentially-weighted averages of TD residuals.
        """
        advantages = torch.zeros_like(rewards)
        last_gae_lambda = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_non_terminal = 1.0 - dones[t]
                next_values = next_value
            else:
                next_non_terminal = 1.0 - dones[t]
                next_values = values[t + 1]
            
            # TD residual
            delta = (
                rewards[t] +
                gamma * next_values * next_non_terminal -
                values[t]
            )
            
            # GAE
            advantages[t] = last_gae_lambda = (
                delta +
                gamma * lambda_ * next_non_terminal * last_gae_lambda
            )
        
        returns = advantages + values
        return advantages, returns
    
    @staticmethod
    def create_minibatches(
        data: Dict[str, torch.Tensor],
        batch_size: int,
        num_minibatches: int
    ) -> Generator:
        """Create minibatches for PPO updates."""
        indices = np.random.permutation(batch_size)
        minibatch_size = batch_size // num_minibatches
        
        for start in range(0, batch_size, minibatch_size):
            end = start + minibatch_size
            mb_indices = indices[start:end]
            
            minibatch = {}
            for key, tensor in data.items():
                minibatch[key] = tensor[mb_indices]
            
            yield minibatch
    
    @staticmethod
    def explained_variance(
        values: torch.Tensor,
        returns: torch.Tensor
    ) -> float:
        """
        Compute explained variance for monitoring value function quality.
        
        Higher is better. > 0.8 is good, < 0 means value function
        is making predictions worse than using the mean.
        """
        var_returns = torch.var(returns)
        if var_returns == 0:
            return 0.0
        return 1 - torch.var(returns - values) / var_returns`
      }
    ],
    interactiveDemos: [
      {
        title: 'PPO Clipping Visualization',
        description: 'See how PPO\'s clipping mechanism creates an implicit trust region',
        component: 'PPOClippingDemo'
      },
      {
        title: 'PPO vs Policy Gradient',
        description: 'Compare update behavior between vanilla PG and PPO',
        component: 'PPOComparisonDemo'
      },
      {
        title: 'Hyperparameter Effects',
        description: 'Explore how different hyperparameters affect PPO performance',
        component: 'PPOHyperparameterDemo'
      }
    ],
    confusionClarifiers: [
      {
        id: 'clip-vs-kl',
        question: 'How does PPO\'s clipping relate to KL divergence constraints?',
        answer: `PPO's clipping creates an implicit KL constraint without explicitly computing KL divergence:

**TRPO Approach:**
- Explicitly constrains: KL(π_new || π_old) ≤ δ
- Requires second-order optimization (expensive)
- Computes Fisher information matrix

**PPO Approach:**
- Clips ratio: 1-ε ≤ π_new/π_old ≤ 1+ε
- First-order optimization (simple gradient descent)
- Achieves similar effect with less computation

**The Connection:**
- Small ε ≈ small KL constraint
- Clipping prevents ratios that would violate trust region
- Empirically achieves similar performance to TRPO

**Advantages of Clipping:**
1. No KL computation needed
2. Works with any optimizer
3. Easier to implement and tune
4. More stable in practice`
      },
      {
        id: 'multiple-epochs',
        question: 'Why can PPO reuse data for multiple epochs when other methods can\'t?',
        answer: `PPO's clipping mechanism makes it safe to reuse data multiple times:

**Standard Policy Gradient Problem:**
- After one update, π_new ≠ π_old
- Data collected with π_old becomes "stale"
- Further updates using old data → wrong gradient direction
- Can lead to policy collapse

**PPO's Solution:**
- Clipping limits how much policy can change
- Even after updates, π_new ≈ π_old (within trust region)
- Data remains "fresh enough" for multiple epochs
- Importance sampling ratio corrects for distribution shift

**Key Insights:**
1. Clipping prevents destructive updates
2. Multiple epochs improve sample efficiency
3. Typically 3-10 epochs work well
4. Monitor KL divergence for early stopping

**Best Practices:**
- Start with 4 epochs
- Use minibatches to decorrelate updates
- Stop if KL divergence exceeds threshold
- Smaller ε → fewer safe epochs`
      },
      {
        id: 'advantage-normalization',
        question: 'Why normalize advantages in PPO? What\'s the impact?',
        answer: `Advantage normalization is crucial for stable PPO training:

**Why Normalize:**
1. **Scale Independence**: Makes learning rate robust to reward scale
2. **Gradient Stability**: Prevents exploding/vanishing gradients
3. **Faster Learning**: Normalized advantages → consistent update sizes

**Implementation:**
\`\`\`python
# Per-minibatch normalization (recommended)
advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

# Alternative: Per-iteration normalization
# Normalizes across all collected data
\`\`\`

**Effects:**
- Without normalization: Need to tune learning rate per environment
- With normalization: Same hyperparameters work across tasks
- Makes optimization landscape more consistent

**Important Considerations:**
1. Always normalize AFTER computing returns
2. Use small epsilon (1e-8) to prevent division by zero
3. Don't normalize returns (only advantages)
4. Normalization doesn't change relative advantage ordering`
      }
    ],
    quiz: [
      {
        id: 'q1',
        question: 'What is the primary purpose of PPO\'s clipping mechanism?',
        options: [
          'To make gradients smaller',
          'To create an implicit trust region preventing destructive updates',
          'To reduce computational cost',
          'To improve exploration'
        ],
        correctAnswer: 1,
        explanation: 'PPO\'s clipping creates an implicit trust region by limiting how much the policy can change, preventing catastrophically large updates that could destroy learned behaviors.'
      },
      {
        id: 'q2',
        question: 'In the PPO objective min(r(θ)A, clip(r(θ), 1-ε, 1+ε)A), what happens when A > 0 and r(θ) > 1+ε?',
        options: [
          'The gradient becomes zero, stopping further increase of π(a|s)',
          'The gradient becomes negative, decreasing π(a|s)',
          'The gradient doubles in magnitude',
          'The advantage is recalculated'
        ],
        correctAnswer: 0,
        explanation: 'When advantage is positive and the ratio exceeds 1+ε, the clipped objective becomes flat, producing zero gradient. This prevents greedy exploitation of positive advantages.'
      },
      {
        id: 'q3',
        question: 'Why can PPO safely perform multiple epochs of updates on the same data?',
        options: [
          'It uses a replay buffer to store experiences',
          'The value function prevents overfitting',
          'Clipping limits policy change, keeping data relevant',
          'It uses importance sampling to correct for any distribution shift'
        ],
        correctAnswer: 2,
        explanation: 'PPO\'s clipping mechanism limits how much the policy can change per update, ensuring that data collected with the old policy remains relevant for multiple epochs of optimization.'
      }
    ] as QuizQuestion[],
    exercises: [
      'Implement PPO for a continuous control task (e.g., Pendulum or LunarLander). Compare performance with different clipping values (ε = 0.1, 0.2, 0.3) and analyze the trade-off between sample efficiency and final performance.',
      'Modify the PPO implementation to include adaptive KL penalty (PPO-Penalty variant). Compare the clipped and penalty variants in terms of implementation complexity and performance.',
      'Implement PPO with LSTM policy for a partially observable environment. Pay attention to handling hidden states during rollout collection and training.',
      'Create a visualization showing how the clipping mechanism affects gradients during training. Plot the policy ratio, advantage, and resulting gradient for different scenarios.'
    ],
    keyTakeaways: [
      'PPO\'s clipped objective creates an implicit trust region without complex second-order optimization',
      'The clipping mechanism prevents destructive updates while allowing multiple epochs of reuse',
      'PPO combines the best ideas from TRPO with the simplicity of standard gradient methods',
      'Proper implementation details (GAE, normalization, parallel environments) are crucial for performance',
      'PPO is the go-to algorithm for many RL applications due to its robustness and simplicity'
    ]
  },
  '7': {
    title: 'Chapter 7: Advanced Topics - RLHF and Modern Applications',
    description: 'Reinforcement Learning from Human Feedback and PPO in Large Language Model training',
    objectives: [
      'Understand RLHF (Reinforcement Learning from Human Feedback) principles',
      'Master the Bradley-Terry model for preference learning',
      'Learn how to train reward models from human preferences',
      'Apply PPO for fine-tuning language models',
      'Understand challenges and best practices in RLHF implementation'
    ],
    sections: [
      {
        id: 'rlhf-introduction',
        title: '7.1 Introduction to RLHF',
        content: `Reinforcement Learning from Human Feedback (RLHF) has revolutionized how we train large language models to be helpful, harmless, and honest. This technique combines supervised learning, reward modeling, and reinforcement learning to align AI systems with human values.

The RLHF pipeline consists of three main stages:
1. **Supervised Fine-tuning (SFT)**: Train a base model on high-quality demonstrations
2. **Reward Model Training**: Learn to predict human preferences from comparison data
3. **RL Fine-tuning**: Use PPO to optimize the model based on the learned reward function

This approach has been instrumental in creating models like ChatGPT, Claude, and other modern AI assistants.`
      },
      {
        id: 'preference-learning',
        title: '7.2 Preference Learning and the Bradley-Terry Model',
        content: `Human preferences are often easier to provide than absolute scores. The Bradley-Terry model provides a principled way to learn from pairwise comparisons.

**The Bradley-Terry Model**
Given two responses A and B, the probability that A is preferred over B is:

P(A > B) = σ(r(A) - r(B)) = 1 / (1 + exp(-(r(A) - r(B))))

Where:
- r(A) and r(B) are the reward scores for responses A and B
- σ is the sigmoid function

This elegant formulation allows us to train a reward model using binary cross-entropy loss on preference pairs.`
      },
      {
        id: 'reward-model-training',
        title: '7.3 Training Reward Models',
        content: `The reward model is crucial for RLHF as it serves as a proxy for human judgment. Here's how we train it:

**Data Collection**
1. Generate pairs of responses for the same prompt
2. Have humans label which response is preferred
3. Create a dataset of (prompt, chosen, rejected) tuples

**Model Architecture**
The reward model typically shares the same architecture as the language model but replaces the final layer with a scalar output head.

**Training Process**
We optimize the Bradley-Terry loss to maximize the likelihood of human preferences.`
      },
      {
        id: 'ppo-for-llms',
        title: '7.4 PPO for Language Model Fine-tuning',
        content: `PPO is the algorithm of choice for RLHF due to its stability and sample efficiency. Here's how it's adapted for language models:

**Key Adaptations**
1. **Token-level Actions**: Each token generation is an action
2. **Sparse Rewards**: Reward is typically only given at the end of generation
3. **KL Penalty**: Prevents the model from diverging too far from the SFT baseline
4. **Advantage Estimation**: Uses GAE to handle credit assignment across long sequences

**The PPO Objective for LLMs**
The total objective combines three components:
- **Policy Loss**: Standard PPO clipped objective
- **Value Loss**: For the critic network
- **KL Penalty**: β * KL(π_θ || π_ref) to maintain similarity to reference model`
      },
      {
        id: 'challenges-best-practices',
        title: '7.5 Challenges and Best Practices',
        content: `Implementing RLHF comes with unique challenges:

**Common Challenges**
1. **Reward Hacking**: Models finding ways to maximize reward without truly improving
2. **Distribution Shift**: Training on model outputs vs human demonstrations
3. **Computational Cost**: Requires multiple models running simultaneously
4. **Reward Model Accuracy**: Limited by quality and diversity of preference data

**Best Practices**
1. **Diverse Preference Data**: Collect preferences across many domains and use cases
2. **Careful Hyperparameter Tuning**: Especially the KL penalty coefficient
3. **Regular Evaluation**: Monitor for degradation in capabilities
4. **Iterative Improvement**: Multiple rounds of RLHF with fresh data`
      }
    ],
    keyTakeaways: [
      'RLHF combines supervised learning, preference modeling, and reinforcement learning to align AI with human values',
      'The Bradley-Terry model provides a principled framework for learning from pairwise preferences',
      'PPO is adapted for LLM fine-tuning with token-level actions and KL penalties',
      'Successful RLHF requires careful attention to data quality, reward modeling, and preventing reward hacking',
      'This approach has enabled the creation of helpful, harmless, and honest AI assistants'
    ],
    quiz: [
      {
        id: 'q1',
        question: 'What is the primary advantage of RLHF over supervised fine-tuning alone?',
        options: [
          'RLHF is computationally cheaper than supervised learning',
          'RLHF allows models to optimize for qualities that are easier to judge than demonstrate',
          'RLHF requires less training data',
          'RLHF eliminates the need for human feedback'
        ],
        correctAnswer: 1,
        explanation: 'RLHF enables optimization for subjective qualities like helpfulness, humor, or creativity that are difficult to demonstrate but easy for humans to judge through comparisons.'
      },
      {
        id: 'q2',
        question: 'In the Bradley-Terry model, what does P(A > B) = σ(r(A) - r(B)) represent?',
        options: [
          'The probability that response A has a higher token count than B',
          'The probability that response A is preferred over response B',
          'The probability that response A was generated first',
          'The probability that response A contains fewer errors than B'
        ],
        correctAnswer: 1,
        explanation: 'The Bradley-Terry model uses the sigmoid of the reward difference to model the probability that one response is preferred over another in a pairwise comparison.'
      },
      {
        id: 'q3',
        question: 'Why is a KL penalty used when applying PPO to language models in RLHF?',
        options: [
          'To make training faster by reducing the model size',
          'To prevent the model from diverging too far from the initial supervised model',
          'To increase the diversity of generated responses',
          'To reduce the computational cost of reward calculation'
        ],
        correctAnswer: 1,
        explanation: 'The KL penalty prevents the policy from deviating too much from a reference model, maintaining linguistic coherence and preventing mode collapse while still allowing improvement through RL.'
      }
    ]
  },
  '8': {
    title: 'Chapter 8: VERL System',
    description: 'Distributed RL with separated Actor, Critic, and Rollout components',
    objectives: [
      'Understand the VERL distributed architecture',
      'Learn HybridFlow: separation of control and computation',
      'Master distributed training with Ray',
      'Scale PPO to production systems',
      'Optimize performance and deployment'
    ],
    sections: [
      {
        id: 'verl-overview',
        title: '8.1 VERL Architecture Overview',
        content: `VERL (Versatile Reinforcement Learning) is a distributed RL system designed for production-scale training. It separates concerns into distinct components for maximum efficiency and scalability.

Key components:
- **Actor**: Generates actions and collects experiences
- **Critic**: Evaluates states and computes advantages
- **Rollout Worker**: Manages environment interactions
- **Reference Policy**: Maintains stable baseline for KL constraints
- **Reward Model**: Provides learned rewards (for RLHF)

This separation allows each component to scale independently and optimize for its specific computational needs.`
      },
      {
        id: 'hybridflow',
        title: '8.2 HybridFlow Architecture',
        content: `HybridFlow is VERL's key innovation: separating control flow from computation flow.

**Control Flow**: Lightweight Python logic for orchestration
- Easy to debug and modify
- Handles complex training logic
- Manages distributed coordination

**Computation Flow**: Heavy lifting in optimized backends
- Leverages JAX/PyTorch for efficiency
- Enables massive parallelization
- Optimizes device utilization

This separation provides both flexibility and performance, crucial for large-scale RL.`
      },
      {
        id: 'distributed-training',
        title: '8.3 Distributed Training with Ray',
        content: `VERL uses Ray for distributed computing, enabling scaling to thousands of GPUs.

Key concepts:
- **Ray Actors**: Stateful workers for different components
- **Object Store**: Efficient data sharing between processes
- **Placement Groups**: Co-location for reduced latency
- **Fault Tolerance**: Automatic recovery from failures

The distributed architecture handles:
- Parallel rollout collection
- Distributed advantage computation
- Synchronized policy updates
- Efficient data transfer between components`
      },
      {
        id: 'scaling-ppo',
        title: '8.4 Scaling PPO to Production',
        content: `Scaling PPO requires careful attention to several aspects:

**Data Efficiency**:
- Larger batch sizes for stable updates
- Experience replay with importance sampling
- Efficient advantage computation

**Computational Efficiency**:
- Mixed precision training
- Gradient accumulation
- Model parallelism for large networks

**System Design**:
- Asynchronous rollout collection
- Pipelined training steps
- Resource allocation optimization`
      },
      {
        id: 'implementation',
        title: '8.5 Implementation and Best Practices',
        content: `Implementing a VERL-style system requires careful design and testing.

Best practices:
- Start with single-node implementation
- Profile and identify bottlenecks
- Scale gradually with monitoring
- Implement comprehensive logging
- Use fault-tolerant designs

Common pitfalls:
- Network bandwidth limitations
- Synchronization overhead
- Memory management issues
- Debugging distributed systems`
      }
    ],
    quiz: [
      {
        id: 'q1',
        question: 'What is the main advantage of separating Actor, Critic, and Rollout in VERL?',
        options: [
          'It makes the code easier to write',
          'Each component can scale independently based on its computational needs',
          'It reduces the total amount of computation required',
          'It eliminates the need for GPUs'
        ],
        correctAnswer: 1,
        explanation: 'Separating components allows each to scale independently - for example, rollout workers can run on CPUs while critics use GPUs, and you can have different numbers of each based on bottlenecks.'
      },
      {
        id: 'q2',
        question: 'What does HybridFlow separate in the VERL architecture?',
        options: [
          'Training data from validation data',
          'Actor networks from critic networks',
          'Control flow (orchestration) from computation flow (heavy lifting)',
          'CPU computation from GPU computation'
        ],
        correctAnswer: 2,
        explanation: 'HybridFlow separates control flow (Python orchestration logic) from computation flow (optimized backend computations), providing both flexibility and performance.'
      },
      {
        id: 'q3',
        question: 'Why is Ray used in the VERL system?',
        options: [
          'It provides the best single-GPU performance',
          'It offers distributed computing with efficient data sharing and fault tolerance',
          'It is required for implementing PPO',
          'It automatically writes the training code'
        ],
        correctAnswer: 1,
        explanation: 'Ray provides distributed computing capabilities with features like efficient object storage, fault tolerance, and easy scaling, making it ideal for large-scale RL systems.'
      }
    ]
  }
};

export default function ChapterPage() {
  const params = useParams();
  const router = useRouter();
  const chapterId = params?.id as string;
  const [progress, setProgress] = useState(0);
  const [completedSections, setCompletedSections] = useState<Set<string>>(new Set());
  const sectionsRef = useRef<{ [key: string]: HTMLElement | null }>({});

  const chapter = chaptersContent[chapterId];

  useEffect(() => {
    if (!chapter) return;

    const observerOptions = {
      root: null,
      rootMargin: '-50% 0px',
      threshold: 0
    };

    const observerCallback = (entries: IntersectionObserverEntry[]) => {
      entries.forEach(entry => {
        if (entry.isIntersecting) {
          const sectionId = entry.target.getAttribute('data-section-id');
          if (sectionId) {
            setCompletedSections(prev => new Set([...prev, sectionId]));
          }
        }
      });
    };

    const observer = new IntersectionObserver(observerCallback, observerOptions);

    Object.values(sectionsRef.current).forEach(section => {
      if (section) observer.observe(section);
    });

    return () => observer.disconnect();
  }, [chapter]);

  useEffect(() => {
    if (chapter) {
      const progressPercentage = (completedSections.size / chapter.sections.length) * 100;
      setProgress(progressPercentage);
    }
  }, [completedSections, chapter]);

  if (!chapter) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <p className="text-xl text-gray-600">Chapter not found</p>
      </div>
    );
  }

  const pyTorchExample = `import torch
import torch.nn as nn
import torch.optim as optim

# Define a simple neural network
class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Initialize network
net = SimpleNN(input_size=10, hidden_size=20, output_size=2)
optimizer = optim.Adam(net.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()`;

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <div className="bg-white shadow-sm sticky top-0 z-10">
        <div className="max-w-4xl mx-auto px-4 py-4">
          <div className="flex items-center justify-between">
            <Link
              href="/chapters"
              className="flex items-center gap-2 text-gray-600 hover:text-gray-900"
            >
              <ArrowLeft size={20} />
              Back to Chapters
            </Link>
            
            <div className="flex items-center gap-4">
              <div className="flex items-center gap-2">
                <span className="text-sm text-gray-600">Progress</span>
                <div className="w-32 bg-gray-200 rounded-full h-2">
                  <motion.div
                    className="bg-blue-600 h-2 rounded-full"
                    initial={{ width: 0 }}
                    animate={{ width: `${progress}%` }}
                    transition={{ duration: 0.5 }}
                  />
                </div>
                <span className="text-sm font-medium">{Math.round(progress)}%</span>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Content */}
      <div className="max-w-4xl mx-auto px-4 py-8">
        {/* Chapter Header */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="mb-8"
        >
          <h1 className="text-4xl font-bold mb-4">{chapter.title}</h1>
          <p className="text-xl text-gray-600">{chapter.description}</p>
        </motion.div>

        {/* Learning Objectives */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.1 }}
          className="bg-blue-50 rounded-lg p-6 mb-8"
        >
          <h2 className="text-2xl font-bold mb-4 flex items-center gap-2">
            <Target className="text-blue-600" />
            Learning Objectives
          </h2>
          <ul className="space-y-2">
            {chapter.objectives.map((objective: string, index: number) => (
              <li key={index} className="flex items-start gap-2">
                <CheckCircle className="w-5 h-5 text-green-500 mt-0.5" />
                <span>{objective}</span>
              </li>
            ))}
          </ul>
        </motion.div>

        {/* Sections */}
        {chapter.sections.map((section: any, index: number) => (
          <motion.section
            key={section.id}
            ref={el => sectionsRef.current[section.id] = el}
            data-section-id={section.id}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.2 + index * 0.1 }}
            className="mb-12"
          >
            <h2 className="text-2xl font-bold mb-4">{section.title}</h2>
            
            <div className="prose prose-lg max-w-none mb-6">
              {section.content.split('\n\n').map((paragraph: string, pIndex: number) => (
                <p key={pIndex} className="mb-4 text-gray-700 leading-relaxed">
                  {paragraph.split('**').map((part: string, partIndex: number) => 
                    partIndex % 2 === 1 ? <strong key={partIndex}>{part}</strong> : part
                  )}
                </p>
              ))}
            </div>

            {/* Mathematical equations for neural networks section */}
            {section.id === 'neural-networks' && (
              <>
                <div className="bg-gray-100 rounded-lg p-4 mb-6 font-mono">
                  <p className="mb-2">Forward propagation:</p>
                  <p className="mb-1">z = Wx + b</p>
                  <p>a = σ(z)</p>
                </div>

                <InteractiveDemo demoType="neural-network" className="mb-6" />

                <ConfusionClarifier
                  id="nn-confusion-1"
                  title="Weights vs. Biases"
                  confusion="Many beginners confuse the roles of weights and biases in neural networks."
                  clarification="Weights control the strength of connections between neurons, while biases allow neurons to activate even when all inputs are zero. Think of weights as 'slopes' and biases as 'intercepts' in linear equations."
                  example="For input x=0: output = W*0 + b = b (bias determines the output)"
                />
              </>
            )}

            {/* Backpropagation demo */}
            {section.id === 'backpropagation' && (
              <>
                <InteractiveDemo demoType="backpropagation" className="mb-6" />
                
                <ConfusionClarifier
                  id="backprop-confusion-1"
                  title="Gradient Flow Direction"
                  confusion="Students often think gradients flow forward through the network."
                  clarification="Gradients flow backward from the loss to the inputs, opposite to the forward pass. This is why it's called 'back'propagation."
                  type="warning"
                />
              </>
            )}

            {/* Gradient descent demo */}
            {section.id === 'gradient-descent' && (
              <>
                <InteractiveDemo demoType="gradient-descent" className="mb-6" />
                
                <ConfusionClarifier
                  id="gd-tip-1"
                  title="Learning Rate Selection"
                  confusion=""
                  clarification="Start with a learning rate of 0.001 for Adam optimizer. If loss explodes, decrease by 10x. If training is too slow, increase by 2-5x."
                  type="tip"
                />
              </>
            )}

            {/* PyTorch code example */}
            {section.id === 'pytorch-basics' && (
              <CodeExample
                code={pyTorchExample}
                language="python"
                title="Basic Neural Network in PyTorch"
                runnable={true}
                className="mb-6"
              />
            )}

            {/* Chapter 2 specific content */}
            {/* MDP Framework demo */}
            {section.id === 'mdp-framework' && (
              <>
                <InteractiveDemo demoType="mdp-visualization" className="mb-6" />
                
                <ConfusionClarifier
                  id="mdp-confusion-1"
                  title="State vs. Observation"
                  confusion="Many beginners confuse state and observation in RL."
                  clarification="State is the true underlying condition of the environment, while observation is what the agent perceives. In partially observable environments, the observation may not contain all state information."
                  example="In poker, the state includes all cards, but your observation only includes your hand and community cards."
                />
              </>
            )}

            {/* Policies visualization */}
            {section.id === 'policies' && (
              <>
                <CodeExample
                  code={`# Deterministic Policy
def deterministic_policy(state):
    # Always returns the same action for a given state
    if state < 0.5:
        return 0  # action 0
    else:
        return 1  # action 1

# Stochastic Policy
def stochastic_policy(state):
    # Returns probability distribution over actions
    if state < 0.5:
        return [0.8, 0.2]  # 80% action 0, 20% action 1
    else:
        return [0.3, 0.7]  # 30% action 0, 70% action 1`}
                  language="python"
                  title="Policy Examples"
                  className="mb-6"
                />
                
                <ConfusionClarifier
                  id="policy-tip-1"
                  title="When to Use Stochastic Policies"
                  confusion=""
                  clarification="Stochastic policies are essential for exploration during training and can be optimal in partially observable or competitive environments. Deterministic policies are often used during deployment for consistent behavior."
                  type="tip"
                />
              </>
            )}

            {/* Reward design examples */}
            {section.id === 'rewards' && (
              <>
                <ConfusionClarifier
                  id="reward-confusion-1"
                  title="Reward Hacking"
                  confusion="Agents often find unexpected ways to maximize rewards that don't align with intended behavior."
                  clarification="This happens when reward functions don't fully capture the desired objective. Always test your reward function thoroughly and consider edge cases."
                  example="A cleaning robot rewarded for picking up trash might learn to create trash just to pick it up again!"
                  type="warning"
                />
              </>
            )}

            {/* Exploration strategies code */}
            {section.id === 'exploration' && (
              <>
                <CodeExample
                  code={`import numpy as np

# Epsilon-greedy exploration
def epsilon_greedy(q_values, epsilon=0.1):
    if np.random.random() < epsilon:
        return np.random.randint(len(q_values))  # Random action
    else:
        return np.argmax(q_values)  # Greedy action

# Boltzmann exploration
def boltzmann_exploration(q_values, temperature=1.0):
    exp_values = np.exp(q_values / temperature)
    probabilities = exp_values / np.sum(exp_values)
    return np.random.choice(len(q_values), p=probabilities)`}
                  language="python"
                  title="Exploration Strategies"
                  runnable={true}
                  className="mb-6"
                />
              </>
            )}

            {/* Chapter 3 specific content */}
            {/* State Value Functions demo */}
            {section.id === 'state-value-functions' && (
              <>
                <div className="bg-gray-100 rounded-lg p-4 mb-6 font-mono">
                  <p className="mb-2">Value Function Definition:</p>
                  <p className="mb-1">V^π(s) = E[G_t | S_t = s]</p>
                  <p className="mb-1">G_t = R_(t+1) + γR_(t+2) + γ²R_(t+3) + ...</p>
                  <p>where γ ∈ [0,1] is the discount factor</p>
                </div>

                <InteractiveDemo demoType="value-function-grid" className="mb-6" />

                <CodeExample
                  code={`import numpy as np

# Simple grid world value function
class ValueFunction:
    def __init__(self, n_states, gamma=0.9):
        self.V = np.zeros(n_states)  # Initialize values to 0
        self.gamma = gamma
    
    def evaluate_policy(self, policy, rewards, transitions, n_iterations=100):
        """Iterative policy evaluation"""
        for _ in range(n_iterations):
            V_new = np.zeros_like(self.V)
            for s in range(len(self.V)):
                # Expected value under policy
                for a in range(len(policy[s])):
                    for s_prime in range(len(self.V)):
                        # V(s) = Σ π(a|s) * P(s'|s,a) * [R + γV(s')]
                        V_new[s] += (policy[s][a] * 
                                    transitions[s][a][s_prime] * 
                                    (rewards[s][a] + self.gamma * self.V[s_prime]))
            self.V = V_new
        return self.V`}
                  language="python"
                  title="Value Function Computation"
                  runnable={true}
                  className="mb-6"
                />

                <ConfusionClarifier
                  id="value-confusion-1"
                  title="Value vs. Reward"
                  confusion="Students often confuse immediate rewards with state values."
                  clarification="Reward is the immediate feedback from one transition, while value is the expected sum of all future rewards from that state. A state might have low immediate reward but high value if it leads to better future states."
                  example="A chess position might not capture a piece (low immediate reward) but leads to checkmate in 3 moves (high value)."
                />
              </>
            )}

            {/* Action Value Functions demo */}
            {section.id === 'action-value-functions' && (
              <>
                <CodeExample
                  code={`# Q-function and its relationship to V-function
import numpy as np

class ActionValueFunction:
    def __init__(self, n_states, n_actions, gamma=0.9):
        self.Q = np.zeros((n_states, n_actions))
        self.gamma = gamma
    
    def q_to_v(self, policy):
        """Convert Q-values to V-values given a policy"""
        n_states = self.Q.shape[0]
        V = np.zeros(n_states)
        
        for s in range(n_states):
            # V(s) = Σ_a π(a|s) * Q(s,a)
            V[s] = np.sum(policy[s] * self.Q[s])
        return V
    
    def greedy_policy_from_q(self):
        """Extract greedy policy from Q-values"""
        return np.argmax(self.Q, axis=1)
    
    def epsilon_greedy_action(self, state, epsilon=0.1):
        """Select action using ε-greedy strategy"""
        if np.random.random() < epsilon:
            return np.random.randint(self.Q.shape[1])
        else:
            return np.argmax(self.Q[state])`}
                  language="python"
                  title="Q-Function Implementation"
                  className="mb-6"
                />

                <InteractiveDemo demoType="q-value-visualization" className="mb-6" />

                <ConfusionClarifier
                  id="q-confusion-1"
                  title="When to Use V(s) vs Q(s,a)"
                  confusion="It's not always clear when to use state values vs action values."
                  clarification="Use V(s) when you have a fixed policy to evaluate. Use Q(s,a) when you need to compare actions for policy improvement or don't know the environment dynamics (model-free learning)."
                  type="tip"
                />
              </>
            )}

            {/* Bellman Equations demo */}
            {section.id === 'bellman-equations' && (
              <>
                <div className="bg-gray-100 rounded-lg p-4 mb-6 font-mono text-sm">
                  <p className="mb-2 font-bold">Bellman Expectation Equations:</p>
                  <p className="mb-1">V^π(s) = Σ_a π(a|s) Σ_s' P(s'|s,a)[R(s,a,s') + γV^π(s')]</p>
                  <p className="mb-3">Q^π(s,a) = Σ_s' P(s'|s,a)[R(s,a,s') + γΣ_a' π(a'|s')Q^π(s',a')]</p>
                  
                  <p className="mb-2 font-bold">Bellman Optimality Equations:</p>
                  <p className="mb-1">V*(s) = max_a Σ_s' P(s'|s,a)[R(s,a,s') + γV*(s')]</p>
                  <p>Q*(s,a) = Σ_s' P(s'|s,a)[R(s,a,s') + γ max_a' Q*(s',a')]</p>
                </div>

                <CodeExample
                  code={`# Value Iteration using Bellman Optimality Equation
def value_iteration(states, actions, transitions, rewards, gamma=0.9, theta=1e-6):
    """
    Solve MDP using value iteration
    Returns: optimal value function and policy
    """
    V = np.zeros(len(states))
    
    while True:
        delta = 0
        V_old = V.copy()
        
        for s in range(len(states)):
            # Bellman optimality update
            q_values = []
            for a in range(len(actions)):
                q_sa = 0
                for s_prime in range(len(states)):
                    # Q(s,a) = Σ P(s'|s,a)[R + γV(s')]
                    q_sa += transitions[s][a][s_prime] * (
                        rewards[s][a] + gamma * V_old[s_prime]
                    )
                q_values.append(q_sa)
            
            # V(s) = max_a Q(s,a)
            V[s] = max(q_values)
            delta = max(delta, abs(V[s] - V_old[s]))
        
        if delta < theta:
            break
    
    # Extract optimal policy
    policy = np.zeros(len(states), dtype=int)
    for s in range(len(states)):
        q_values = []
        for a in range(len(actions)):
            q_sa = sum(transitions[s][a][s_prime] * 
                      (rewards[s][a] + gamma * V[s_prime])
                      for s_prime in range(len(states)))
            q_values.append(q_sa)
        policy[s] = np.argmax(q_values)
    
    return V, policy`}
                  language="python"
                  title="Value Iteration Algorithm"
                  runnable={true}
                  className="mb-6"
                />

                <ConfusionClarifier
                  id="bellman-confusion-1"
                  title="Expectation vs Optimality"
                  confusion="The difference between Bellman expectation and optimality equations is subtle but crucial."
                  clarification="Expectation equations describe values under a specific policy (what you get by following π). Optimality equations describe values under the best possible policy (what you get by always choosing the best action)."
                  example="Expectation: 'If I follow my current strategy...' vs Optimality: 'If I play perfectly...'"
                  type="warning"
                />
              </>
            )}

            {/* TD Learning and Monte Carlo demo */}
            {section.id === 'value-estimation-methods' && (
              <>
                <CodeExample
                  code={`import numpy as np

class ValueEstimation:
    def __init__(self, n_states, alpha=0.1, gamma=0.9):
        self.V = np.zeros(n_states)
        self.alpha = alpha  # learning rate
        self.gamma = gamma  # discount factor
    
    def monte_carlo_update(self, episode):
        """Update values using Monte Carlo method"""
        states, rewards = episode
        G = 0  # return
        
        # Work backwards through episode
        for t in reversed(range(len(states))):
            G = rewards[t] + self.gamma * G
            state = states[t]
            # MC update: V(s) ← V(s) + α[G - V(s)]
            self.V[state] += self.alpha * (G - self.V[state])
    
    def td_update(self, state, reward, next_state):
        """Update value using TD(0) learning"""
        # TD error: δ = r + γV(s') - V(s)
        td_error = reward + self.gamma * self.V[next_state] - self.V[state]
        # TD update: V(s) ← V(s) + α * δ
        self.V[state] += self.alpha * td_error
        return td_error
    
    def td_lambda_update(self, state, reward, next_state, eligibility_traces, lambda_=0.9):
        """TD(λ) update with eligibility traces"""
        td_error = reward + self.gamma * self.V[next_state] - self.V[state]
        
        # Update eligibility traces
        eligibility_traces *= self.gamma * lambda_
        eligibility_traces[state] += 1
        
        # Update all states proportional to eligibility
        self.V += self.alpha * td_error * eligibility_traces
        
        return td_error, eligibility_traces

# Example comparison
def compare_methods():
    # Simulate a simple chain MDP
    n_states = 5
    mc_estimator = ValueEstimation(n_states)
    td_estimator = ValueEstimation(n_states)
    
    # Example episode: [s0 -> s1 -> s2 -> s3 -> s4 (terminal)]
    # with rewards: [0, 0, 0, 0, 1]
    
    # Monte Carlo waits until episode end
    episode = ([0, 1, 2, 3, 4], [0, 0, 0, 0, 1])
    mc_estimator.monte_carlo_update(episode)
    
    # TD learns at each step
    transitions = [(0, 0, 1), (1, 0, 2), (2, 0, 3), (3, 0, 4), (4, 1, 4)]
    for state, reward, next_state in transitions:
        td_estimator.td_update(state, reward, next_state)
    
    print("MC values:", mc_estimator.V)
    print("TD values:", td_estimator.V)`}
                  language="python"
                  title="TD Learning vs Monte Carlo"
                  runnable={true}
                  className="mb-6"
                />

                <InteractiveDemo demoType="td-vs-mc" className="mb-6" />

                <ConfusionClarifier
                  id="td-mc-confusion-1"
                  title="Bias-Variance Tradeoff"
                  confusion="Why would we use TD if it's biased, when MC is unbiased?"
                  clarification="TD's bias comes from using estimated values (bootstrapping), but this also reduces variance. In practice, lower variance often leads to faster learning, even with some bias. MC's high variance can make learning unstable, especially with limited data."
                  example="Think of it like weather prediction: TD uses today's forecast to update tomorrow's (some bias but consistent), while MC waits for the actual weather all week (accurate but variable)."
                />

                <ConfusionClarifier
                  id="td-tip-1"
                  title="Choosing α (Learning Rate)"
                  confusion=""
                  clarification="For TD methods, start with α=0.1. For MC, often use α=0.01 since returns have higher variance. In practice, decay α over time for convergence: α_t = α_0 / (1 + decay_rate * t)"
                  type="tip"
                />
              </>
            )}

            {/* Chapter 4 specific content */}
            {/* Policy-Based Methods comparison */}
            {section.id === 'policy-based-methods' && (
              <>
                <InteractiveDemo demoType="value-vs-policy" className="mb-6" />
                
                <CodeExample
                  code={`# Value-based approach (e.g., Q-learning)
class ValueBasedAgent:
    def __init__(self, n_states, n_actions):
        self.Q = np.zeros((n_states, n_actions))
    
    def get_action(self, state, epsilon=0.1):
        # Derive policy from Q-values
        if np.random.random() < epsilon:
            return np.random.randint(self.Q.shape[1])
        return np.argmax(self.Q[state])

# Policy-based approach
class PolicyBasedAgent:
    def __init__(self, state_dim, action_dim):
        # Directly parameterize policy
        self.theta = np.random.randn(state_dim, action_dim) * 0.01
    
    def get_action_probabilities(self, state):
        # Softmax policy
        logits = state @ self.theta
        exp_logits = np.exp(logits - np.max(logits))
        return exp_logits / np.sum(exp_logits)
    
    def get_action(self, state):
        probs = self.get_action_probabilities(state)
        return np.random.choice(len(probs), p=probs)`}
                  language="python"
                  title="Value-Based vs Policy-Based Methods"
                  className="mb-6"
                />

                <ConfusionClarifier
                  id="policy-confusion-1"
                  title="Policy Parameterization"
                  confusion="How do we represent a policy with neural networks?"
                  clarification="For discrete actions, output a softmax layer with probabilities for each action. For continuous actions, output mean and variance of a Gaussian distribution. The network learns to map states to action distributions."
                  example="Discrete: π(a|s) = softmax(f_θ(s)) | Continuous: π(a|s) = N(μ_θ(s), σ_θ(s))"
                />
              </>
            )}

            {/* Policy Gradient Theorem visualization */}
            {section.id === 'policy-gradient-theorem' && (
              <>
                <div className="bg-gray-100 rounded-lg p-4 mb-6 font-mono text-sm">
                  <p className="mb-2 font-bold">Policy Gradient Theorem:</p>
                  <p className="mb-1">∇_θ J(θ) = E_τ~π_θ[Σ_t ∇_θ log π_θ(a_t|s_t) G_t]</p>
                  <p className="mb-3">where G_t = Σ_(k=t)^T γ^(k-t) r_k</p>
                  
                  <p className="mb-2 font-bold">Log-derivative trick:</p>
                  <p className="mb-1">∇_θ π_θ(a|s) = π_θ(a|s) ∇_θ log π_θ(a|s)</p>
                  <p>This allows us to estimate gradients from samples!</p>
                </div>

                <InteractiveDemo demoType="policy-gradient-intuition" className="mb-6" />

                <CodeExample
                  code={`import torch
import torch.nn as nn
import torch.nn.functional as F

class PolicyNetwork(nn.Module):
    """Simple policy network for discrete actions"""
    def __init__(self, state_dim, hidden_dim, action_dim):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
    
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        logits = self.fc3(x)
        return F.softmax(logits, dim=-1)
    
    def get_log_prob(self, state, action):
        """Compute log π_θ(a|s) for the policy gradient"""
        probs = self.forward(state)
        return torch.log(probs.gather(1, action.unsqueeze(1)))

# Example gradient computation
policy = PolicyNetwork(state_dim=4, hidden_dim=64, action_dim=2)
optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)

# Simulated trajectory data
states = torch.randn(10, 4)  # 10 timesteps, 4-dim states
actions = torch.randint(0, 2, (10,))  # discrete actions
returns = torch.tensor([1.5, 1.3, 1.1, 0.9, 0.7, 0.5, 0.3, 0.1, -0.1, -0.3])

# Policy gradient update
log_probs = policy.get_log_prob(states, actions)
loss = -(log_probs.squeeze() * returns).mean()  # negative for gradient ascent

optimizer.zero_grad()
loss.backward()
optimizer.step()`}
                  language="python"
                  title="Policy Gradient Implementation"
                  runnable={true}
                  className="mb-6"
                />

                <ConfusionClarifier
                  id="pg-confusion-1"
                  title="Why Log Probabilities?"
                  confusion="Why do we use log π(a|s) instead of π(a|s) in the gradient?"
                  clarification="The log-derivative trick transforms the expectation of gradients (hard to compute) into an expectation of log-gradients times returns (easy to estimate with samples). Also, log probabilities are numerically more stable."
                  type="warning"
                />
              </>
            )}

            {/* REINFORCE Algorithm implementation */}
            {section.id === 'reinforce-algorithm' && (
              <>
                <CodeExample
                  code={`import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import deque

class REINFORCE:
    def __init__(self, state_dim, action_dim, lr=1e-3, gamma=0.99):
        self.policy = PolicyNetwork(state_dim, 64, action_dim)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        self.gamma = gamma
        
    def get_action(self, state):
        state = torch.FloatTensor(state)
        probs = self.policy(state)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action)
    
    def compute_returns(self, rewards):
        """Calculate discounted returns G_t for each timestep"""
        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + self.gamma * G
            returns.insert(0, G)
        return torch.tensor(returns)
    
    def update(self, states, actions, rewards):
        """REINFORCE update using collected episode data"""
        returns = self.compute_returns(rewards)
        
        # Normalize returns for stability
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        # Calculate loss
        loss = 0
        for state, action, G in zip(states, actions, returns):
            state = torch.FloatTensor(state)
            log_prob = self.policy.get_log_prob(state.unsqueeze(0), 
                                               torch.tensor([action]))
            loss -= log_prob * G  # negative for gradient ascent
        
        # Backprop and update
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()

# Training loop example
def train_reinforce(env, agent, n_episodes=1000):
    scores = deque(maxlen=100)
    
    for episode in range(n_episodes):
        state = env.reset()
        states, actions, rewards = [], [], []
        
        # Collect episode
        done = False
        while not done:
            action, _ = agent.get_action(state)
            next_state, reward, done, _ = env.step(action)
            
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            
            state = next_state
        
        # Update policy
        loss = agent.update(states, actions, rewards)
        scores.append(sum(rewards))
        
        if episode % 100 == 0:
            print(f"Episode {episode}, Average Score: {np.mean(scores):.2f}")`}
                  language="python"
                  title="Complete REINFORCE Implementation"
                  runnable={true}
                  className="mb-6"
                />

                <InteractiveDemo demoType="reinforce-training" className="mb-6" />

                <ConfusionClarifier
                  id="reinforce-tip-1"
                  title="Normalizing Returns"
                  confusion=""
                  clarification="Always normalize returns (subtract mean, divide by std) before computing gradients. This centers the gradients and prevents the optimization from being dominated by the scale of rewards, leading to more stable training."
                  type="tip"
                />
              </>
            )}

            {/* Variance Reduction techniques */}
            {section.id === 'variance-reduction' && (
              <>
                <CodeExample
                  code={`class REINFORCEWithBaseline:
    def __init__(self, state_dim, action_dim, lr=1e-3, gamma=0.99):
        # Policy network
        self.policy = PolicyNetwork(state_dim, 64, action_dim)
        
        # Value network as baseline
        self.value = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        self.value_optimizer = torch.optim.Adam(self.value.parameters(), lr=lr)
        self.gamma = gamma
    
    def update(self, states, actions, rewards):
        returns = self.compute_returns(rewards)
        states_tensor = torch.FloatTensor(states)
        
        # Compute baseline values
        values = self.value(states_tensor).squeeze()
        
        # Advantage = Return - Baseline
        advantages = returns - values.detach()
        
        # Update policy (using advantages)
        policy_loss = 0
        for state, action, advantage in zip(states, actions, advantages):
            state = torch.FloatTensor(state)
            log_prob = self.policy.get_log_prob(state.unsqueeze(0), 
                                               torch.tensor([action]))
            policy_loss -= log_prob * advantage
        
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()
        
        # Update value function (minimize MSE)
        value_loss = F.mse_loss(values, returns)
        self.value_optimizer.zero_grad()
        value_loss.backward()
        self.value_optimizer.step()
        
        return policy_loss.item(), value_loss.item()

# Comparison of different baselines
def compare_baselines():
    # No baseline
    returns = [10, 8, 6, 4, 2]
    print("No baseline - gradients:", returns)
    
    # Constant baseline (average)
    baseline_const = np.mean(returns)
    advantages_const = [r - baseline_const for r in returns]
    print(f"Constant baseline ({baseline_const:.1f}) - gradients:", 
          [f"{a:.1f}" for a in advantages_const])
    
    # Value function baseline (example values)
    values = [9, 7, 5, 3, 1]
    advantages_value = [r - v for r, v in zip(returns, values)]
    print("Value baseline - gradients:", advantages_value)
    
compare_baselines()`}
                  language="python"
                  title="REINFORCE with Baseline"
                  runnable={true}
                  className="mb-6"
                />

                <InteractiveDemo demoType="baseline-variance" className="mb-6" />

                <ConfusionClarifier
                  id="baseline-confusion-1"
                  title="Baseline Bias Concern"
                  confusion="Doesn't subtracting a baseline introduce bias to our gradient estimates?"
                  clarification="No! The key insight is that E[∇log π(a|s) * b(s)] = 0 when b depends only on state s, not on action a. This means subtracting any state-dependent baseline preserves the expected gradient while reducing variance."
                  example="Think of it as adjusting scores: if everyone gets +100 points, relative rankings stay the same."
                  type="warning"
                />
              </>
            )}

            {/* Natural Policy Gradient and TRPO preview */}
            {section.id === 'advanced-methods-preview' && (
              <>
                <div className="bg-blue-50 rounded-lg p-6 mb-6">
                  <h3 className="text-xl font-bold mb-4">Evolution of Policy Gradient Methods</h3>
                  <div className="space-y-4">
                    <div className="flex items-start gap-3">
                      <div className="w-8 h-8 bg-blue-500 text-white rounded-full flex items-center justify-center flex-shrink-0">1</div>
                      <div>
                        <strong>REINFORCE</strong>: Simple but high variance
                        <p className="text-sm text-gray-600">Basic policy gradient with Monte Carlo returns</p>
                      </div>
                    </div>
                    <div className="flex items-start gap-3">
                      <div className="w-8 h-8 bg-blue-500 text-white rounded-full flex items-center justify-center flex-shrink-0">2</div>
                      <div>
                        <strong>Natural Policy Gradient</strong>: Better update direction
                        <p className="text-sm text-gray-600">Accounts for policy geometry using Fisher information</p>
                      </div>
                    </div>
                    <div className="flex items-start gap-3">
                      <div className="w-8 h-8 bg-blue-500 text-white rounded-full flex items-center justify-center flex-shrink-0">3</div>
                      <div>
                        <strong>TRPO</strong>: Guaranteed improvement
                        <p className="text-sm text-gray-600">Constrains policy updates using KL divergence</p>
                      </div>
                    </div>
                    <div className="flex items-start gap-3">
                      <div className="w-8 h-8 bg-green-500 text-white rounded-full flex items-center justify-center flex-shrink-0">4</div>
                      <div>
                        <strong>PPO</strong>: Practical and efficient
                        <p className="text-sm text-gray-600">Simplifies TRPO while maintaining performance</p>
                      </div>
                    </div>
                  </div>
                </div>

                <CodeExample
                  code={`# Conceptual comparison of update rules

# Standard Policy Gradient
def standard_pg_update(theta, grad, lr):
    return theta + lr * grad

# Natural Policy Gradient
def natural_pg_update(theta, grad, fisher_matrix, lr):
    # F^(-1) accounts for parameter space geometry
    natural_grad = np.linalg.inv(fisher_matrix) @ grad
    return theta + lr * natural_grad

# TRPO (simplified)
def trpo_update(theta_old, grad, fisher_matrix, delta):
    # Solve: max g^T (theta - theta_old)
    # s.t. 0.5 * (theta - theta_old)^T F (theta - theta_old) <= delta
    
    natural_grad = np.linalg.inv(fisher_matrix) @ grad
    step_size = np.sqrt(2 * delta / (grad @ natural_grad))
    return theta_old + step_size * natural_grad

# PPO (simplified)
def ppo_update(theta, grad, lr, epsilon=0.2):
    # Clipped objective prevents large updates
    # Implementation uses ratio clipping, shown in next chapter
    return theta + lr * grad  # With special objective function`}
                  language="python"
                  title="Policy Update Methods Comparison"
                  className="mb-6"
                />

                <ConfusionClarifier
                  id="npg-confusion-1"
                  title="Why Natural Gradients?"
                  confusion="What's wrong with standard gradients?"
                  clarification="Standard gradients treat all parameter directions equally, but small changes in parameters can cause large changes in policy behavior. Natural gradients account for how parameter changes affect the policy distribution, leading to more stable updates."
                  example="Imagine navigating on a curved surface (policy space) vs flat surface (parameter space) - the shortest path looks different!"
                />

                <div className="bg-yellow-50 border-l-4 border-yellow-400 p-4 mb-6">
                  <div className="flex">
                    <AlertCircle className="h-6 w-6 text-yellow-400 mr-3 flex-shrink-0" />
                    <div>
                      <h4 className="font-bold">Looking Ahead to PPO</h4>
                      <p className="text-sm mt-1">
                        PPO combines the best ideas from these methods: the simplicity of REINFORCE, 
                        the variance reduction of baselines, and the stability of trust regions, 
                        all while being computationally efficient. The next chapters will show exactly how!
                      </p>
                    </div>
                  </div>
                </div>
              </>
            )}

            {/* Chapter 5 specific content */}
            {/* Actor-Critic Motivation demo */}
            {section.id === 'actor-critic-motivation' && (
              <>
                <InteractiveDemo demoType="actor-critic-architecture" className="mb-6" />
                
                <CodeExample
                  code={`# Comparison: Pure Policy Gradient vs Actor-Critic
import torch
import torch.nn as nn
import numpy as np

# Pure Policy Gradient (REINFORCE)
class PolicyGradientAgent:
    def __init__(self, state_dim, action_dim):
        self.policy = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
            nn.Softmax(dim=-1)
        )
        self.saved_log_probs = []
        self.rewards = []
    
    def update(self):
        # Must wait for full episode
        returns = self.calculate_returns()  # High variance!
        policy_loss = []
        for log_prob, R in zip(self.saved_log_probs, returns):
            policy_loss.append(-log_prob * R)
        # ... update

# Actor-Critic
class ActorCriticAgent:
    def __init__(self, state_dim, action_dim):
        # Shared feature extractor
        self.features = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU()
        )
        # Actor head
        self.actor = nn.Linear(128, action_dim)
        # Critic head
        self.critic = nn.Linear(128, 1)
    
    def forward(self, state):
        features = self.features(state)
        policy = torch.softmax(self.actor(features), dim=-1)
        value = self.critic(features)
        return policy, value
    
    def update(self, state, action, reward, next_state, done):
        # Can update online with TD learning!
        _, value = self.forward(state)
        _, next_value = self.forward(next_state)
        
        # TD error for advantage
        td_target = reward + (1-done) * 0.99 * next_value
        advantage = td_target - value  # Lower variance!
        
        # Update both actor and critic
        # ... update logic`}
                  language="python"
                  title="Policy Gradient vs Actor-Critic"
                  className="mb-6"
                />

                <ConfusionClarifier
                  id="ac-confusion-1"
                  title="Shared vs Separate Networks"
                  confusion="Should I use shared or separate networks for actor and critic?"
                  clarification="Start with shared feature extraction layers and separate heads. This is more parameter efficient and often works well. Only use fully separate networks if you see training instability or conflicting gradients."
                  type="tip"
                />
              </>
            )}

            {/* A2C Algorithm implementation */}
            {section.id === 'a2c-algorithm' && (
              <>
                <CodeExample
                  code={`import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class A2C(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(A2C, self).__init__()
        
        # Shared layers
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        
        # Actor head
        self.actor = nn.Linear(hidden_dim, action_dim)
        
        # Critic head
        self.critic = nn.Linear(hidden_dim, 1)
        
        self.optimizer = torch.optim.Adam(self.parameters(), lr=3e-4)
        
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        
        # Actor: action probabilities
        logits = self.actor(x)
        action_probs = F.softmax(logits, dim=-1)
        
        # Critic: state value
        state_value = self.critic(x)
        
        return action_probs, state_value
    
    def get_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        action_probs, value = self.forward(state)
        
        # Sample action from distribution
        dist = torch.distributions.Categorical(action_probs)
        action = dist.sample()
        
        return action.item(), dist.log_prob(action), value
    
    def compute_loss(self, states, actions, rewards, next_states, dones, gamma=0.99):
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones)
        
        # Get current values and action probabilities
        action_probs, values = self.forward(states)
        _, next_values = self.forward(next_states)
        
        # Calculate TD targets
        td_targets = rewards + gamma * next_values.squeeze() * (1 - dones)
        advantages = td_targets.detach() - values.squeeze()
        
        # Actor loss (policy gradient)
        dist = torch.distributions.Categorical(action_probs)
        log_probs = dist.log_prob(actions)
        actor_loss = -(log_probs * advantages.detach()).mean()
        
        # Critic loss (value function)
        critic_loss = F.mse_loss(values.squeeze(), td_targets.detach())
        
        # Entropy bonus for exploration
        entropy = dist.entropy().mean()
        
        # Total loss
        loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy
        
        return loss, actor_loss.item(), critic_loss.item(), entropy.item()
    
    def update(self, trajectory):
        states, actions, rewards, next_states, dones = trajectory
        
        loss, actor_loss, critic_loss, entropy = self.compute_loss(
            states, actions, rewards, next_states, dones
        )
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), 0.5)
        self.optimizer.step()
        
        return {
            'total_loss': loss.item(),
            'actor_loss': actor_loss,
            'critic_loss': critic_loss,
            'entropy': entropy
        }

# Training loop
def train_a2c(env, n_episodes=1000, batch_size=32):
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    agent = A2C(state_dim, action_dim)
    
    for episode in range(n_episodes):
        states, actions, rewards, next_states, dones = [], [], [], [], []
        
        state = env.reset()
        done = False
        episode_reward = 0
        
        # Collect trajectory
        while not done and len(states) < batch_size:
            action, _, _ = agent.get_action(state)
            next_state, reward, done, _ = env.step(action)
            
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            dones.append(done)
            
            state = next_state
            episode_reward += reward
        
        # Update agent
        trajectory = (states, actions, rewards, next_states, dones)
        metrics = agent.update(trajectory)
        
        if episode % 100 == 0:
            print(f"Episode {episode}: Reward = {episode_reward:.2f}, {metrics}")`}
                  language="python"
                  title="Complete A2C Implementation"
                  runnable={true}
                  className="mb-6"
                />

                <InteractiveDemo demoType="a2c-training" className="mb-6" />

                <ConfusionClarifier
                  id="a2c-confusion-1"
                  title="Entropy Regularization"
                  confusion="Why do we subtract entropy from the loss?"
                  clarification="Subtracting entropy encourages the policy to remain stochastic, promoting exploration. Without it, the policy might become deterministic too quickly, getting stuck in local optima. The entropy coefficient (typically 0.01) controls the exploration-exploitation balance."
                  type="warning"
                />
              </>
            )}

            {/* A3C Algorithm visualization */}
            {section.id === 'a3c-algorithm' && (
              <>
                <InteractiveDemo demoType="a3c-architecture" className="mb-6" />
                
                <CodeExample
                  code={`import torch
import torch.nn as nn
import torch.multiprocessing as mp
from torch.distributions import Categorical
import numpy as np

# Global shared model
class SharedA3C(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(SharedA3C, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.actor = nn.Linear(128, action_dim)
        self.critic = nn.Linear(128, 1)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        policy = torch.softmax(self.actor(x), dim=-1)
        value = self.critic(x)
        return policy, value

def worker(worker_id, shared_model, optimizer, env_fn, T_max=20):
    """A3C worker process"""
    env = env_fn()
    local_model = SharedA3C(env.observation_space.shape[0], 
                            env.action_space.n)
    
    t = 0
    while True:  # Run until stopped
        # Sync with shared model
        local_model.load_state_dict(shared_model.state_dict())
        
        # Collect trajectory
        states, actions, rewards, values = [], [], [], []
        state = env.reset()
        done = False
        
        for _ in range(T_max):
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            policy, value = local_model(state_tensor)
            
            # Sample action
            dist = Categorical(policy)
            action = dist.sample()
            
            next_state, reward, done, _ = env.step(action.item())
            
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            values.append(value)
            
            state = next_state
            t += 1
            
            if done:
                break
        
        # Compute returns and advantages
        R = 0 if done else local_model(
            torch.FloatTensor(state).unsqueeze(0))[1].item()
        
        returns = []
        for r in reversed(rewards):
            R = r + 0.99 * R
            returns.insert(0, R)
        
        returns = torch.tensor(returns)
        values = torch.cat(values)
        advantages = returns - values.squeeze()
        
        # Compute gradients
        states = torch.FloatTensor(states)
        actions = torch.stack(actions)
        
        policies, _ = local_model(states)
        dist = Categorical(policies)
        log_probs = dist.log_prob(actions.squeeze())
        
        actor_loss = -(log_probs * advantages.detach()).mean()
        critic_loss = advantages.pow(2).mean()
        entropy = dist.entropy().mean()
        
        loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy
        
        # Update shared model
        optimizer.zero_grad()
        loss.backward()
        
        # Copy gradients to shared model
        for shared_param, local_param in zip(shared_model.parameters(), 
                                           local_model.parameters()):
            shared_param._grad = local_param.grad
        
        optimizer.step()

# A2C with parallel environments (modern approach)
class ParallelA2C:
    """Synchronous A2C with multiple parallel environments"""
    def __init__(self, env_fns, state_dim, action_dim):
        self.envs = [fn() for fn in env_fns]
        self.n_envs = len(self.envs)
        self.model = A2C(state_dim, action_dim)
        
    def collect_rollouts(self, n_steps):
        """Collect data from all environments synchronously"""
        states = [env.reset() for env in self.envs]
        
        batch_states, batch_actions, batch_rewards = [], [], []
        batch_next_states, batch_dones = [], []
        
        for _ in range(n_steps):
            # Get actions for all environments
            actions = []
            for state in states:
                action, _, _ = self.model.get_action(state)
                actions.append(action)
            
            # Step all environments
            next_states, rewards, dones = [], [], []
            for i, (env, action) in enumerate(zip(self.envs, actions)):
                next_state, reward, done, _ = env.step(action)
                if done:
                    next_state = env.reset()
                
                next_states.append(next_state)
                rewards.append(reward)
                dones.append(done)
            
            # Store transitions
            batch_states.extend(states)
            batch_actions.extend(actions)
            batch_rewards.extend(rewards)
            batch_next_states.extend(next_states)
            batch_dones.extend(dones)
            
            states = next_states
        
        return (batch_states, batch_actions, batch_rewards, 
                batch_next_states, batch_dones)`}
                  language="python"
                  title="A3C vs Parallel A2C Implementation"
                  className="mb-6"
                />

                <ConfusionClarifier
                  id="a3c-tip-1"
                  title="Modern Best Practice"
                  confusion=""
                  clarification="Most practitioners now prefer synchronous parallel collection (like Parallel A2C) over asynchronous updates (A3C). It's more stable, easier to debug, and often performs better. Libraries like Stable-Baselines3 use this approach."
                  type="tip"
                />
              </>
            )}

            {/* GAE implementation and visualization */}
            {section.id === 'gae-generalized-advantage' && (
              <>
                <div className="bg-gray-100 rounded-lg p-4 mb-6 font-mono text-sm">
                  <p className="mb-2 font-bold">GAE Formula:</p>
                  <p className="mb-1">δ_t = r_t + γV(s_(t+1)) - V(s_t)</p>
                  <p className="mb-1">A^GAE_t = δ_t + (γλ)δ_(t+1) + (γλ)²δ_(t+2) + ...</p>
                  <p>= Σ_(l=0)^∞ (γλ)^l δ_(t+l)</p>
                </div>

                <CodeExample
                  code={`import torch
import numpy as np

def compute_gae(rewards, values, next_values, dones, gamma=0.99, lam=0.95):
    """
    Compute Generalized Advantage Estimation (GAE)
    
    Args:
        rewards: list of rewards for each timestep
        values: value estimates V(s_t)
        next_values: value estimates V(s_(t+1))
        dones: episode termination flags
        gamma: discount factor
        lam: GAE lambda parameter
    
    Returns:
        advantages: GAE advantages
        returns: discounted returns (for value function targets)
    """
    advantages = []
    gae = 0
    
    # Work backwards through the trajectory
    for t in reversed(range(len(rewards))):
        # TD residual: δ_t = r_t + γV(s_(t+1)) - V(s_t)
        if t == len(rewards) - 1:
            next_value = next_values[t]
        else:
            next_value = values[t + 1]
        
        delta = rewards[t] + gamma * next_value * (1 - dones[t]) - values[t]
        
        # GAE: A_t = δ_t + (γλ)δ_(t+1) + (γλ)²δ_(t+2) + ...
        gae = delta + gamma * lam * (1 - dones[t]) * gae
        advantages.insert(0, gae)
    
    advantages = torch.tensor(advantages)
    returns = advantages + values  # A = G - V, so G = A + V
    
    return advantages, returns

# Example: Compare different lambda values
def compare_gae_lambda():
    # Simulated trajectory
    rewards = [0, 0, 0, 1, 0, 0, 0, 10]  # Sparse rewards
    values = torch.tensor([0.1, 0.2, 0.5, 1.0, 0.8, 0.6, 0.4, 8.0])
    next_values = torch.tensor([0.2, 0.5, 1.0, 0.8, 0.6, 0.4, 8.0, 0.0])
    dones = [0, 0, 0, 0, 0, 0, 0, 1]
    
    # Compare different lambda values
    lambdas = [0.0, 0.5, 0.9, 0.95, 0.99, 1.0]
    
    print("Comparing GAE with different λ values:\n")
    print("Rewards:", rewards)
    print("Values:", values.numpy())
    print("\n")
    
    for lam in lambdas:
        advantages, returns = compute_gae(
            rewards, values, next_values, dones, gamma=0.99, lam=lam
        )
        print(f"λ = {lam}:")
        print(f"  Advantages: {advantages.numpy()}")
        print(f"  Variance: {advantages.var():.4f}")
        if lam == 0.0:
            print("  (Pure TD - high bias, low variance)")
        elif lam == 1.0:
            print("  (Pure MC - low bias, high variance)")
        print()

# Efficient GAE implementation for batched data
class GAEBuffer:
    def __init__(self, gamma=0.99, lam=0.95):
        self.gamma = gamma
        self.lam = lam
        self.clear()
    
    def clear(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.dones = []
        
    def add(self, state, action, reward, value, done):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)
    
    def compute_returns_and_advantages(self, last_value):
        """Compute GAE and returns for the entire buffer"""
        rewards = np.array(self.rewards)
        values = np.array(self.values)
        dones = np.array(self.dones)
        
        # Append last value for proper GAE computation
        values = np.append(values, last_value)
        
        advantages = np.zeros_like(rewards)
        last_gae_lam = 0
        
        for t in reversed(range(len(rewards))):
            next_non_terminal = 1.0 - dones[t]
            delta = rewards[t] + self.gamma * values[t + 1] * next_non_terminal - values[t]
            advantages[t] = last_gae_lam = delta + self.gamma * self.lam * next_non_terminal * last_gae_lam
        
        returns = advantages + values[:-1]
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        return returns, advantages

compare_gae_lambda()`}
                  language="python"
                  title="GAE Implementation and Analysis"
                  runnable={true}
                  className="mb-6"
                />

                <InteractiveDemo demoType="gae-visualization" className="mb-6" />

                <ConfusionClarifier
                  id="gae-confusion-1"
                  title="GAE vs Simple Advantages"
                  confusion="Why not just use A = R - V(s) like in basic actor-critic?"
                  clarification="Simple advantages (R - V) have high variance because they use full Monte Carlo returns. GAE with λ < 1 reduces variance by incorporating the critic's predictions, while λ controls how much we trust these predictions. This dramatically improves learning stability."
                  example="Think of GAE as a 'soft' version of n-step returns that smoothly blends all possible n-step advantages."
                  type="warning"
                />
              </>
            )}

            {/* Implementation details and best practices */}
            {section.id === 'implementation-details' && (
              <>
                <CodeExample
                  code={`import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Shared architecture with careful initialization
class SharedActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super().__init__()
        
        # Shared layers
        self.shared_fc1 = nn.Linear(state_dim, hidden_dim)
        self.shared_fc2 = nn.Linear(hidden_dim, hidden_dim)
        
        # Separate heads
        self.actor_head = nn.Linear(hidden_dim, action_dim)
        self.critic_head = nn.Linear(hidden_dim, 1)
        
        # Initialize weights
        self._initialize_weights()
        
    def _initialize_weights(self):
        # Initialize shared layers
        for m in [self.shared_fc1, self.shared_fc2]:
            nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
            nn.init.constant_(m.bias, 0)
        
        # Initialize actor head with smaller weights
        nn.init.orthogonal_(self.actor_head.weight, gain=0.01)
        nn.init.constant_(self.actor_head.bias, 0)
        
        # Initialize critic head
        nn.init.orthogonal_(self.critic_head.weight, gain=1.0)
        nn.init.constant_(self.critic_head.bias, 0)
    
    def forward(self, state):
        # Shared computation
        x = F.relu(self.shared_fc1(state))
        x = F.relu(self.shared_fc2(x))
        
        # Heads
        logits = self.actor_head(x)
        value = self.critic_head(x)
        
        return logits, value

# Separate architecture for comparison
class SeparateActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super().__init__()
        
        # Actor network
        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        
        # Critic network
        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Different learning rates for actor and critic
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=1e-3)

# Advanced normalization techniques
class NormalizedActorCritic(SharedActorCritic):
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super().__init__(state_dim, action_dim, hidden_dim)
        
        # Running statistics for observation normalization
        self.obs_mean = nn.Parameter(torch.zeros(state_dim), requires_grad=False)
        self.obs_std = nn.Parameter(torch.ones(state_dim), requires_grad=False)
        self.obs_count = nn.Parameter(torch.tensor(0.0), requires_grad=False)
        
        # Running statistics for reward normalization
        self.reward_mean = nn.Parameter(torch.tensor(0.0), requires_grad=False)
        self.reward_std = nn.Parameter(torch.tensor(1.0), requires_grad=False)
        
    def update_obs_stats(self, obs):
        """Update running statistics for observations"""
        batch_mean = obs.mean(dim=0)
        batch_std = obs.std(dim=0)
        batch_count = obs.shape[0]
        
        delta = batch_mean - self.obs_mean
        total_count = self.obs_count + batch_count
        
        # Update mean
        self.obs_mean.data = self.obs_mean + delta * batch_count / total_count
        
        # Update std (Welford's online algorithm)
        m_a = self.obs_std.pow(2) * self.obs_count
        m_b = batch_std.pow(2) * batch_count
        M2 = m_a + m_b + delta.pow(2) * self.obs_count * batch_count / total_count
        self.obs_std.data = torch.sqrt(M2 / total_count)
        
        self.obs_count.data = total_count
    
    def normalize_obs(self, obs):
        """Normalize observations using running statistics"""
        return (obs - self.obs_mean) / (self.obs_std + 1e-8)
    
    def forward(self, state):
        # Normalize input
        state = self.normalize_obs(state)
        return super().forward(state)

# Entropy scheduling
class EntropyScheduler:
    def __init__(self, initial_coef=0.01, final_coef=0.001, 
                 decay_steps=1000000, decay_type='linear'):
        self.initial_coef = initial_coef
        self.final_coef = final_coef
        self.decay_steps = decay_steps
        self.decay_type = decay_type
        self.step = 0
    
    def get_coef(self):
        if self.decay_type == 'linear':
            progress = min(self.step / self.decay_steps, 1.0)
            return self.initial_coef + (self.final_coef - self.initial_coef) * progress
        elif self.decay_type == 'exponential':
            decay_rate = (self.final_coef / self.initial_coef) ** (1 / self.decay_steps)
            return self.initial_coef * (decay_rate ** self.step)
    
    def step_update(self):
        self.step += 1

# Complete training setup with best practices
def train_actor_critic_best_practices(env, n_steps=2048, n_epochs=10, 
                                     batch_size=64, n_updates=1000000):
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    # Model with normalization
    model = NormalizedActorCritic(state_dim, action_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    
    # Entropy scheduler
    entropy_scheduler = EntropyScheduler()
    
    # GAE buffer
    buffer = GAEBuffer(gamma=0.99, lam=0.95)
    
    state = env.reset()
    
    for update in range(n_updates):
        # Collect rollout
        buffer.clear()
        
        for _ in range(n_steps):
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            
            with torch.no_grad():
                logits, value = model(state_tensor)
                dist = torch.distributions.Categorical(logits=logits)
                action = dist.sample()
            
            next_state, reward, done, _ = env.step(action.item())
            
            buffer.add(state, action.item(), reward, value.item(), done)
            
            state = next_state if not done else env.reset()
        
        # Compute returns and advantages
        with torch.no_grad():
            _, last_value = model(torch.FloatTensor(state).unsqueeze(0))
        
        returns, advantages = buffer.compute_returns_and_advantages(last_value.item())
        
        # Convert to tensors
        states = torch.FloatTensor(buffer.states)
        actions = torch.LongTensor(buffer.actions)
        returns = torch.FloatTensor(returns)
        advantages = torch.FloatTensor(advantages)
        
        # Update observation statistics
        model.update_obs_stats(states)
        
        # Mini-batch updates
        indices = np.arange(n_steps)
        
        for _ in range(n_epochs):
            np.random.shuffle(indices)
            
            for start in range(0, n_steps, batch_size):
                end = start + batch_size
                batch_indices = indices[start:end]
                
                # Get batch
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_returns = returns[batch_indices]
                batch_advantages = advantages[batch_indices]
                
                # Forward pass
                logits, values = model(batch_states)
                dist = torch.distributions.Categorical(logits=logits)
                log_probs = dist.log_prob(batch_actions)
                
                # Losses
                actor_loss = -(log_probs * batch_advantages).mean()
                critic_loss = F.mse_loss(values.squeeze(), batch_returns)
                entropy = dist.entropy().mean()
                
                # Total loss with entropy bonus
                entropy_coef = entropy_scheduler.get_coef()
                loss = actor_loss + 0.5 * critic_loss - entropy_coef * entropy
                
                # Update
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                optimizer.step()
        
        entropy_scheduler.step_update()
        
        if update % 100 == 0:
            print(f"Update {update}: Entropy coef = {entropy_coef:.4f}")`}
                  language="python"
                  title="Actor-Critic Best Practices"
                  runnable={true}
                  className="mb-6"
                />

                <ConfusionClarifier
                  id="impl-tip-1"
                  title="Gradient Clipping"
                  confusion=""
                  clarification="Always use gradient clipping (typically 0.5 or 1.0) in actor-critic methods. The interaction between actor and critic losses can create large gradients that destabilize training. Clipping prevents these destructive updates."
                  type="tip"
                />

                <ConfusionClarifier
                  id="impl-confusion-1"
                  title="Value Function Scale"
                  confusion="My critic loss is orders of magnitude larger than actor loss. Is this a problem?"
                  clarification="Yes! Large value scales can dominate the gradients. Either normalize returns before training the critic, or use separate optimizers with different learning rates (critic often needs larger LR). Some implementations also clip value predictions to a reasonable range."
                  type="warning"
                />
              </>
            )}

            {/* Chapter 7: RLHF and Modern Applications */}
            {/* RLHF Introduction */}
            {section.id === 'rlhf-introduction' && (
              <>
                <InteractiveDemo demoType="rlhf-pipeline" className="mb-6" />
                
                <ConfusionClarifier
                  id="rlhf-confusion-1"
                  title="Why Not Just Supervised Learning?"
                  confusion="If we have human feedback, why not just do supervised learning on the best responses?"
                  clarification="Supervised learning only teaches the model to imitate demonstrations. RLHF allows the model to learn from preferences and optimize for qualities that are hard to demonstrate but easy to judge (like humor, creativity, or safety). It also enables the model to potentially surpass human demonstrations."
                  example="It's easier to judge if a joke is funny than to write the perfect joke yourself."
                />
              </>
            )}

            {/* Bradley-Terry Model */}
            {section.id === 'preference-learning' && (
              <>
                <CodeExample
                  code={`import torch
import torch.nn as nn
import torch.nn.functional as F

class BradleyTerryLoss(nn.Module):
    """Bradley-Terry model for preference learning"""
    def __init__(self):
        super().__init__()
        
    def forward(self, chosen_rewards, rejected_rewards):
        """
        Compute Bradley-Terry loss for preference pairs
        
        Args:
            chosen_rewards: Rewards for chosen responses [batch_size]
            rejected_rewards: Rewards for rejected responses [batch_size]
        
        Returns:
            loss: Scalar loss value
        """
        # Compute preference probability
        # P(chosen > rejected) = sigmoid(r_chosen - r_rejected)
        logits = chosen_rewards - rejected_rewards
        
        # Binary cross-entropy with target = 1 (chosen is preferred)
        loss = -F.logsigmoid(logits).mean()
        
        # Compute accuracy for monitoring
        accuracy = (logits > 0).float().mean()
        
        return loss, accuracy

# Example usage
def preference_learning_example():
    # Simulated reward scores
    chosen_rewards = torch.tensor([0.8, 0.6, 0.9, 0.7])
    rejected_rewards = torch.tensor([0.3, 0.5, 0.2, 0.4])
    
    # Compute loss
    criterion = BradleyTerryLoss()
    loss, accuracy = criterion(chosen_rewards, rejected_rewards)
    
    print(f"Bradley-Terry Loss: {loss:.4f}")
    print(f"Preference Accuracy: {accuracy:.2%}")
    print(f"Average margin: {(chosen_rewards - rejected_rewards).mean():.3f}")

preference_learning_example()`}
                  language="python"
                  title="Bradley-Terry Model Implementation"
                  runnable={true}
                  className="mb-6"
                />
                
                <InteractiveDemo demoType="preference-comparison" className="mb-6" />
                
                <ConfusionClarifier
                  id="bt-tip-1"
                  title="Calibrating Reward Differences"
                  confusion=""
                  clarification="The Bradley-Terry model assumes reward differences map to preference probabilities via sigmoid. In practice, you may need to scale rewards or add a temperature parameter: P(A > B) = σ((r(A) - r(B))/τ) to calibrate the model's confidence."
                  type="tip"
                />
              </>
            )}

            {/* Reward Model Training */}
            {section.id === 'reward-model-training' && (
              <>
                <CodeExample
                  code={`import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

class RewardModel(nn.Module):
    """Reward model for RLHF"""
    def __init__(self, base_model_name="gpt2", hidden_size=768):
        super().__init__()
        # Load pretrained transformer
        self.transformer = AutoModel.from_pretrained(base_model_name)
        
        # Freeze early layers for efficiency (optional)
        for param in self.transformer.embeddings.parameters():
            param.requires_grad = False
            
        # Reward head: maps hidden states to scalar reward
        self.reward_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, 1)
        )
        
    def forward(self, input_ids, attention_mask):
        # Get transformer outputs
        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        
        # Use last hidden state of the last token
        hidden_states = outputs.last_hidden_state  # [batch, seq_len, hidden]
        
        # Find last non-padding token for each sequence
        sequence_lengths = attention_mask.sum(dim=1) - 1  # [batch]
        batch_idx = torch.arange(hidden_states.size(0))
        last_hidden = hidden_states[batch_idx, sequence_lengths]  # [batch, hidden]
        
        # Compute reward
        reward = self.reward_head(last_hidden).squeeze(-1)  # [batch]
        
        return reward

def train_reward_model(model, dataloader, num_epochs=3):
    """Train reward model on preference data"""
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    criterion = BradleyTerryLoss()
    
    for epoch in range(num_epochs):
        total_loss = 0
        total_accuracy = 0
        
        for batch in dataloader:
            # Get chosen and rejected responses
            chosen_ids = batch['chosen_input_ids']
            chosen_mask = batch['chosen_attention_mask']
            rejected_ids = batch['rejected_input_ids']
            rejected_mask = batch['rejected_attention_mask']
            
            # Forward pass
            chosen_rewards = model(chosen_ids, chosen_mask)
            rejected_rewards = model(rejected_ids, rejected_mask)
            
            # Compute loss
            loss, accuracy = criterion(chosen_rewards, rejected_rewards)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            total_accuracy += accuracy.item()
        
        avg_loss = total_loss / len(dataloader)
        avg_accuracy = total_accuracy / len(dataloader)
        print(f"Epoch {epoch+1}: Loss={avg_loss:.4f}, Accuracy={avg_accuracy:.2%}")

# Example: Creating preference dataset
def create_preference_pairs():
    """Example of creating preference training data"""
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    # Example preference pairs
    data = [
        {
            "prompt": "What is the capital of France?",
            "chosen": "The capital of France is Paris, a beautiful city known for the Eiffel Tower, art museums, and rich culture.",
            "rejected": "Paris."
        },
        {
            "prompt": "Explain photosynthesis",
            "chosen": "Photosynthesis is the process by which plants convert sunlight, water, and carbon dioxide into glucose and oxygen. This process occurs in chloroplasts and is essential for life on Earth.",
            "rejected": "Plants make food from sun."
        }
    ]
    
    # Tokenize
    tokenized_data = []
    for item in data:
        prompt = item["prompt"]
        chosen_text = f"{prompt}\\n\\nAssistant: {item['chosen']}"
        rejected_text = f"{prompt}\\n\\nAssistant: {item['rejected']}"
        
        chosen_tokens = tokenizer(chosen_text, padding=True, truncation=True, return_tensors="pt")
        rejected_tokens = tokenizer(rejected_text, padding=True, truncation=True, return_tensors="pt")
        
        tokenized_data.append({
            'chosen_input_ids': chosen_tokens['input_ids'],
            'chosen_attention_mask': chosen_tokens['attention_mask'],
            'rejected_input_ids': rejected_tokens['input_ids'],
            'rejected_attention_mask': rejected_tokens['attention_mask']
        })
    
    return tokenized_data`}
                  language="python"
                  title="Complete Reward Model Implementation"
                  runnable={true}
                  className="mb-6"
                />
                
                <ConfusionClarifier
                  id="rm-confusion-1"
                  title="Reward Hacking in Reward Models"
                  confusion="Can the reward model be gamed or exploited?"
                  clarification="Yes! Reward models can be exploited through adversarial inputs, out-of-distribution responses, or by finding patterns that correlate with high reward but don't reflect true quality. This is why KL penalties and regular model updates with fresh preference data are crucial."
                  example="A model might learn that longer responses get higher rewards and start padding responses unnecessarily."
                  type="warning"
                />
              </>
            )}

            {/* PPO for LLMs */}
            {section.id === 'ppo-for-llms' && (
              <>
                <CodeExample
                  code={`import torch
import torch.nn.functional as F
from torch.distributions import Categorical

class PPOTrainer:
    """PPO trainer adapted for language models"""
    def __init__(self, actor_model, critic_model, ref_model, reward_model, 
                 lr=1e-5, clip_epsilon=0.2, kl_coef=0.01):
        self.actor = actor_model
        self.critic = critic_model
        self.ref_model = ref_model  # Reference model for KL penalty
        self.reward_model = reward_model
        
        self.actor_optimizer = torch.optim.AdamW(actor_model.parameters(), lr=lr)
        self.critic_optimizer = torch.optim.AdamW(critic_model.parameters(), lr=lr*2)
        
        self.clip_epsilon = clip_epsilon
        self.kl_coef = kl_coef
        
    def compute_rewards(self, input_ids, attention_mask):
        """Compute rewards including KL penalty"""
        with torch.no_grad():
            # Get reward from reward model
            rewards = self.reward_model(input_ids, attention_mask)
            
            # Compute KL penalty
            actor_logits = self.actor(input_ids, attention_mask).logits
            ref_logits = self.ref_model(input_ids, attention_mask).logits
            
            # KL divergence at token level
            actor_logprobs = F.log_softmax(actor_logits, dim=-1)
            ref_logprobs = F.log_softmax(ref_logits, dim=-1)
            
            # Only compute KL for generated tokens (not prompt)
            kl_penalty = torch.sum(
                torch.exp(actor_logprobs) * (actor_logprobs - ref_logprobs),
                dim=-1
            )
            
            # Apply KL penalty only to generated portion
            response_mask = self._get_response_mask(attention_mask)
            kl_penalty = kl_penalty * response_mask
            kl_penalty = kl_penalty.sum(dim=1) / response_mask.sum(dim=1)
            
            # Final reward = reward_model_score - β * KL(π||π_ref)
            final_rewards = rewards - self.kl_coef * kl_penalty
            
        return final_rewards, rewards, kl_penalty
    
    def ppo_step(self, states, actions, advantages, old_log_probs):
        """Single PPO update step"""
        # Actor update
        actor_logits = self.actor(states).logits
        dist = Categorical(logits=actor_logits)
        log_probs = dist.log_prob(actions)
        
        # PPO clipped objective
        ratio = torch.exp(log_probs - old_log_probs)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
        actor_loss = -torch.min(surr1, surr2).mean()
        
        # Entropy bonus for exploration
        entropy = dist.entropy().mean()
        actor_loss = actor_loss - 0.01 * entropy
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.actor_optimizer.step()
        
        # Critic update
        values = self.critic(states).squeeze(-1)
        returns = advantages + values.detach()  # Bootstrap from old values
        critic_loss = F.mse_loss(values, returns)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        return actor_loss.item(), critic_loss.item(), entropy.item()

    def generate_and_score(self, prompts, max_length=100):
        """Generate responses and compute rewards"""
        # This is simplified - in practice, you'd use proper generation
        input_ids = self.tokenizer(prompts, return_tensors="pt").input_ids
        
        with torch.no_grad():
            # Generate response
            outputs = self.actor.generate(
                input_ids, 
                max_length=max_length,
                do_sample=True,
                top_p=0.9,
                temperature=0.8,
                pad_token_id=self.tokenizer.pad_token_id
            )
            
            # Compute rewards
            attention_mask = (outputs != self.tokenizer.pad_token_id).float()
            rewards, rm_scores, kl_penalties = self.compute_rewards(outputs, attention_mask)
            
        return outputs, rewards, rm_scores, kl_penalties

# Training loop example
def train_ppo_rlhf(ppo_trainer, prompts, num_iterations=1000):
    """Main RLHF training loop"""
    for iteration in range(num_iterations):
        # Generate responses for batch of prompts
        responses, rewards, rm_scores, kl_penalties = ppo_trainer.generate_and_score(prompts)
        
        # Compute advantages using GAE or simple advantage estimation
        advantages = compute_advantages(rewards, ppo_trainer.critic(responses))
        
        # Multiple PPO epochs on the same data
        for ppo_epoch in range(4):
            actor_loss, critic_loss, entropy = ppo_trainer.ppo_step(
                responses, actions, advantages, old_log_probs
            )
        
        if iteration % 10 == 0:
            print(f"Iteration {iteration}:")
            print(f"  Avg Reward: {rewards.mean():.3f}")
            print(f"  Avg RM Score: {rm_scores.mean():.3f}")
            print(f"  Avg KL Penalty: {kl_penalties.mean():.3f}")
            print(f"  Actor Loss: {actor_loss:.3f}")
            print(f"  Entropy: {entropy:.3f}")`}
                  language="python"
                  title="PPO for Language Model Fine-tuning"
                  runnable={true}
                  className="mb-6"
                />
                
                <InteractiveDemo demoType="ppo-llm-training" className="mb-6" />
                
                <ConfusionClarifier
                  id="ppo-llm-tip-1"
                  title="KL Coefficient Tuning"
                  confusion=""
                  clarification="The KL coefficient (β) is crucial for balancing reward optimization with maintaining model behavior. Start with β=0.01-0.1. If the model diverges too much from the reference (gibberish, repetition), increase β. If it's not improving, decrease β. Monitor KL divergence throughout training."
                  type="tip"
                />

                <ConfusionClarifier
                  id="ppo-llm-confusion-1"
                  title="Response-Level vs Token-Level Rewards"
                  confusion="Should rewards be assigned to individual tokens or the entire response?"
                  clarification="In practice, the reward model typically produces a single score for the entire response, but PPO needs token-level advantages. Common approaches: 1) Assign the final reward to the last token, 2) Distribute reward across all generated tokens, or 3) Use reward shaping to provide intermediate rewards."
                  example="Option 2 is often most stable: advantage[t] = (total_reward / num_generated_tokens) - value[t]"
                />
              </>
            )}

            {/* Challenges and Best Practices */}
            {section.id === 'challenges-best-practices' && (
              <>
                <CodeExample
                  code={`# Common RLHF Implementation Pitfalls and Solutions

class RLHFBestPractices:
    """Collection of best practices for RLHF implementation"""
    
    @staticmethod
    def prevent_reward_hacking(reward_model, responses):
        """Techniques to prevent reward hacking"""
        # 1. Length normalization
        response_lengths = (responses != pad_token_id).sum(dim=1)
        rewards = reward_model(responses)
        normalized_rewards = rewards / torch.sqrt(response_lengths.float())
        
        # 2. Diversity bonus
        unique_tokens = len(set(responses.flatten().tolist()))
        diversity_bonus = unique_tokens / responses.numel()
        
        # 3. Repetition penalty
        def compute_repetition_penalty(tokens):
            # Penalize repeated n-grams
            penalty = 0
            for n in [2, 3, 4]:  # bigrams, trigrams, 4-grams
                ngrams = [tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)]
                unique_ratio = len(set(ngrams)) / len(ngrams)
                penalty += (1 - unique_ratio)
            return penalty / 3
        
        rep_penalty = torch.tensor([
            compute_repetition_penalty(resp.tolist()) 
            for resp in responses
        ])
        
        # Combined reward
        final_rewards = normalized_rewards + 0.1 * diversity_bonus - 0.2 * rep_penalty
        return final_rewards
    
    @staticmethod
    def robust_preference_collection():
        """Guidelines for collecting high-quality preferences"""
        guidelines = """
        1. Diverse Rater Pool:
           - Multiple raters per comparison
           - Diverse backgrounds and expertise
           - Clear rating guidelines
        
        2. Quality Control:
           - Inter-rater agreement metrics
           - Catch trials with known preferences  
           - Regular calibration sessions
        
        3. Prompt Diversity:
           - Cover various domains and tasks
           - Include edge cases and adversarial prompts
           - Balance different difficulty levels
        
        4. Response Generation:
           - Use different sampling strategies
           - Include human-written responses
           - Vary response lengths and styles
        """
        return guidelines
    
    @staticmethod
    def iterative_rlhf_training():
        """Iterative RLHF training loop"""
        for round_num in range(num_rounds):
            # 1. Generate new responses with current policy
            responses = generate_responses(current_model, prompts)
            
            # 2. Collect new preferences
            preferences = collect_human_preferences(responses)
            
            # 3. Update reward model with all data
            all_preferences = previous_preferences + preferences
            reward_model = train_reward_model(all_preferences)
            
            # 4. Evaluate reward model quality
            rm_accuracy = evaluate_reward_model(reward_model, test_preferences)
            if rm_accuracy < 0.65:  # Below human agreement
                print("Warning: Reward model accuracy too low")
                # Consider collecting more data or debugging
            
            # 5. Run PPO with new reward model
            current_model = ppo_training(
                current_model, 
                reward_model,
                ref_model=previous_model  # Update reference periodically
            )
            
            # 6. Safety and quality checks
            if round_num % 2 == 0:
                safety_score = evaluate_safety(current_model)
                capability_score = evaluate_capabilities(current_model)
                
                if safety_score < threshold or capability_score < previous_score * 0.95:
                    print("Rolling back to previous model")
                    current_model = previous_model
                else:
                    previous_model = current_model
            
            previous_preferences = all_preferences

# Monitoring and debugging utilities
class RLHFMonitor:
    """Monitor RLHF training for common issues"""
    
    def __init__(self):
        self.metrics = {
            'kl_divergence': [],
            'reward_scores': [],
            'response_lengths': [],
            'entropy': [],
            'gradient_norms': []
        }
    
    def check_training_health(self):
        """Check for common RLHF training issues"""
        issues = []
        
        # Check for reward hacking
        if len(self.metrics['reward_scores']) > 100:
            recent_rewards = self.metrics['reward_scores'][-100:]
            recent_lengths = self.metrics['response_lengths'][-100:]
            
            correlation = np.corrcoef(recent_rewards, recent_lengths)[0, 1]
            if correlation > 0.8:
                issues.append("High correlation between reward and length - possible length hacking")
        
        # Check for mode collapse
        recent_entropy = self.metrics['entropy'][-50:]
        if len(recent_entropy) > 0 and np.mean(recent_entropy) < 0.1:
            issues.append("Low entropy - possible mode collapse")
        
        # Check for policy divergence  
        recent_kl = self.metrics['kl_divergence'][-50:]
        if len(recent_kl) > 0 and np.mean(recent_kl) > 10.0:
            issues.append("High KL divergence - policy diverging too much from reference")
        
        # Check for training instability
        recent_grads = self.metrics['gradient_norms'][-50:]
        if len(recent_grads) > 0:
            grad_variance = np.var(recent_grads)
            if grad_variance > 100:
                issues.append("High gradient variance - training may be unstable")
        
        return issues`}
                  language="python"
                  title="RLHF Best Practices and Monitoring"
                  runnable={true}
                  className="mb-6"
                />
                
                <InteractiveDemo demoType="rlhf-debugging" className="mb-6" />
                
                <ConfusionClarifier
                  id="rlhf-tip-2"
                  title="When to Update the Reference Model"
                  confusion=""
                  clarification="The reference model in the KL penalty should be updated periodically (e.g., every 1000 steps or when KL divergence is consistently low). This allows for controlled exploration while preventing the policy from reverting to the initial behavior. Think of it as gradually moving the 'anchor point' as the model improves."
                  type="tip"
                />
              </>
            )}

            {/* Chapter 8 specific content */}
            {/* VERL Overview */}
            {section.id === 'verl-overview' && (
              <>
                <CodeExample
                  code={`# VERL Component Architecture
class VERLSystem:
    """Distributed RL system with separated components"""
    
    def __init__(self):
        # Independent components that can scale separately
        self.actor = ActorWorker()          # GPU for inference
        self.critic = CriticWorker()        # GPU for value computation
        self.rollout = RolloutWorker()      # CPU for env simulation
        self.reference = ReferencePolicy()   # GPU for KL computation
        self.reward_model = RewardModel()    # GPU for reward prediction
        
    def architecture_benefits(self):
        return {
            "actor": "Can use model parallelism for large policies",
            "critic": "Can use different architecture than actor",
            "rollout": "Can scale to 1000s of CPU workers",
            "reference": "Can be quantized for efficiency",
            "reward_model": "Can be updated independently"
        }

# Example deployment configuration
deployment_config = {
    "actor": {"num_gpus": 8, "num_replicas": 2},
    "critic": {"num_gpus": 4, "num_replicas": 1},
    "rollout": {"num_cpus": 128, "num_replicas": 16},
    "reference": {"num_gpus": 1, "num_replicas": 1},
    "reward_model": {"num_gpus": 2, "num_replicas": 1}
}`}
                  language="python"
                  title="VERL Component Architecture"
                  className="mb-6"
                />
                
                <InteractiveDemo demoType="verl-architecture" className="mb-6" />
                
                <ConfusionClarifier
                  id="verl-confusion-1"
                  title="Why Separate Actor and Critic?"
                  confusion="In standard implementations, actor and critic share parameters. Why separate them?"
                  clarification="Separation allows: 1) Different model architectures (e.g., larger critic), 2) Different update frequencies, 3) Independent scaling based on computational bottlenecks, 4) Easier debugging and monitoring."
                  example="The actor might use a DistilBERT for fast inference, while the critic uses a full BERT for better value estimates."
                />
              </>
            )}

            {/* HybridFlow */}
            {section.id === 'hybridflow' && (
              <>
                <CodeExample
                  code={`# HybridFlow Example: Separating Control and Computation
import ray
import jax
import torch

class HybridFlowTrainer:
    """Demonstrates control/computation separation"""
    
    def __init__(self):
        # Control flow: Python orchestration
        self.training_loop = self.python_control_flow
        
        # Computation flow: Optimized backends
        self.actor_compute = JaxActorCompute()
        self.critic_compute = TorchCriticCompute()
    
    def python_control_flow(self):
        """Lightweight Python logic for orchestration"""
        while not self.done:
            # Control flow handles logic
            if self.should_collect_rollouts():
                rollouts = self.collect_rollouts()
            
            if self.should_update_critic():
                # Dispatch heavy computation
                critic_loss = self.critic_compute.update(rollouts)
            
            if self.should_update_actor():
                # Different backend for actor
                actor_loss = self.actor_compute.update(rollouts)
            
            # Python handles coordination
            self.log_metrics(critic_loss, actor_loss)
            self.checkpoint_if_needed()
    
    def benefits(self):
        return [
            "Easy debugging with Python control flow",
            "Flexibility to change training logic",
            "Optimal backend for each computation",
            "Clear separation of concerns"
        ]

# Example: Different backends for different components
class JaxActorCompute:
    """JAX backend for actor updates (good for TPUs)"""
    @jax.jit
    def update(self, batch):
        # Optimized JAX computation
        return actor_loss

class TorchCriticCompute:
    """PyTorch backend for critic (good for dynamic graphs)"""
    def update(self, batch):
        # PyTorch computation with autograd
        return critic_loss`}
                  language="python"
                  title="HybridFlow Architecture"
                  className="mb-6"
                />
                
                <ConfusionClarifier
                  id="hybridflow-tip-1"
                  title="Choosing Computation Backends"
                  confusion=""
                  clarification="JAX excels at JIT compilation and TPU support, making it ideal for large-scale actor inference. PyTorch's dynamic graphs are perfect for critics that might have varying architectures. Use the right tool for each job!"
                  type="tip"
                />
              </>
            )}

            {/* Distributed Training */}
            {section.id === 'distributed-training' && (
              <>
                <CodeExample
                  code={`import ray
import torch
from ray import train
from ray.train.torch import TorchTrainer
from ray.air import ScalingConfig

# Initialize Ray
ray.init()

@ray.remote(num_gpus=1)
class ActorWorker:
    """Ray actor for policy inference"""
    def __init__(self, model_config):
        self.device = torch.device("cuda")
        self.model = PolicyNetwork(model_config).to(self.device)
        
    def get_actions(self, observations):
        with torch.no_grad():
            obs_tensor = torch.tensor(observations).to(self.device)
            actions, log_probs = self.model.act(obs_tensor)
        return actions.cpu().numpy(), log_probs.cpu().numpy()
    
    def update_weights(self, weights):
        self.model.load_state_dict(weights)

@ray.remote(num_cpus=2)
class RolloutWorker:
    """CPU worker for environment rollouts"""
    def __init__(self, env_name, actor_handle):
        self.env = gym.make(env_name)
        self.actor = actor_handle
        
    async def collect_rollout(self, num_steps):
        observations = []
        actions = []
        rewards = []
        
        obs = self.env.reset()
        for _ in range(num_steps):
            # Get action from actor (async)
            action, log_prob = await self.actor.get_actions.remote(obs)
            
            # Step environment
            next_obs, reward, done, _ = self.env.step(action)
            
            observations.append(obs)
            actions.append(action)
            rewards.append(reward)
            
            if done:
                obs = self.env.reset()
            else:
                obs = next_obs
                
        return {
            "observations": np.array(observations),
            "actions": np.array(actions),
            "rewards": np.array(rewards)
        }

# Distributed PPO Trainer
class DistributedPPO:
    def __init__(self, num_actors=4, num_rollout_workers=32):
        # Create actors
        self.actors = [ActorWorker.remote(model_config) 
                      for _ in range(num_actors)]
        
        # Create rollout workers
        self.rollout_workers = []
        for i in range(num_rollout_workers):
            # Assign workers to actors round-robin
            actor = self.actors[i % num_actors]
            worker = RolloutWorker.remote(env_name, actor)
            self.rollout_workers.append(worker)
        
    def train(self):
        for iteration in range(num_iterations):
            # Parallel rollout collection
            rollout_futures = [
                worker.collect_rollout.remote(rollout_length)
                for worker in self.rollout_workers
            ]
            
            # Wait for all rollouts
            rollouts = ray.get(rollout_futures)
            
            # Update models
            self.update_models(rollouts)
            
            # Broadcast new weights
            new_weights = self.get_model_weights()
            update_futures = [
                actor.update_weights.remote(new_weights)
                for actor in self.actors
            ]
            ray.get(update_futures)`}
                  language="python"
                  title="Distributed Training with Ray"
                  runnable={true}
                  className="mb-6"
                />
                
                <InteractiveDemo demoType="distributed-training" className="mb-6" />
                
                <ConfusionClarifier
                  id="ray-confusion-1"
                  title="Ray Actors vs Tasks"
                  confusion="When should I use Ray actors vs Ray tasks?"
                  clarification="Use Ray actors for stateful components (models, environments) that persist across multiple operations. Use Ray tasks for stateless computations (preprocessing, one-off calculations). Actors have overhead but enable efficient stateful operations."
                  example="Actor: Model that serves many inference requests. Task: Computing advantages for a batch."
                />
              </>
            )}

            {/* Scaling PPO */}
            {section.id === 'scaling-ppo' && (
              <>
                <CodeExample
                  code={`# Scaling PPO to Production
class ScalablePPO:
    """Production-ready PPO with scaling optimizations"""
    
    def __init__(self, config):
        self.config = config
        self.setup_distributed_components()
        
    def scaling_optimizations(self):
        return {
            # Data Efficiency
            "large_batch_training": {
                "batch_size": 65536,  # Very large batches
                "mini_batch_size": 2048,
                "gradient_accumulation_steps": 4
            },
            
            # Computational Efficiency  
            "mixed_precision": {
                "enabled": True,
                "opt_level": "O2",  # FP16 with FP32 master weights
                "loss_scale": "dynamic"
            },
            
            # Memory Efficiency
            "gradient_checkpointing": {
                "enabled": True,
                "checkpoint_segments": 4
            },
            
            # Communication Efficiency
            "all_reduce_optimization": {
                "bucket_size_mb": 25,
                "allreduce_batched": True,
                "gradient_predivide_factor": 1.0
            }
        }
    
    def distributed_advantage_computation(self, rollouts):
        """Compute advantages in parallel across workers"""
        # Shard rollouts across workers
        rollout_shards = self.shard_rollouts(rollouts)
        
        # Parallel advantage computation
        advantage_futures = []
        for worker, shard in zip(self.critic_workers, rollout_shards):
            future = worker.compute_advantages.remote(shard)
            advantage_futures.append(future)
        
        # Gather results
        advantages = ray.get(advantage_futures)
        return self.combine_advantages(advantages)
    
    def pipelined_training_step(self):
        """Pipeline data collection and training"""
        # Start collecting next batch while training
        next_rollout_future = self.collect_rollouts_async()
        
        # Train on current batch
        while self.current_batch is not None:
            # Update critic and actor
            critic_loss = self.update_critic(self.current_batch)
            actor_loss = self.update_actor(self.current_batch)
            
            # Check if next batch is ready
            if next_rollout_future.done():
                self.current_batch = next_rollout_future.result()
                next_rollout_future = self.collect_rollouts_async()
        
    def resource_allocation_strategy(self):
        """Dynamic resource allocation based on bottlenecks"""
        metrics = self.get_performance_metrics()
        
        if metrics["rollout_time"] > metrics["training_time"]:
            # Rollout is bottleneck - allocate more workers
            self.scale_rollout_workers(factor=1.5)
        elif metrics["critic_time"] > metrics["actor_time"]:
            # Critic is bottleneck - allocate more GPUs
            self.scale_critic_resources(factor=1.2)
            
        return self.get_current_allocation()`}
                  language="python"
                  title="Scaling PPO to Production"
                  className="mb-6"
                />
                
                <ConfusionClarifier
                  id="scaling-tip-1"
                  title="Batch Size Scaling"
                  confusion=""
                  clarification="When scaling batch size, scale the learning rate proportionally (linear scaling rule). For very large batches (>10k), consider using a warmup period and adaptive learning rate schedules. Monitor the effective batch size after gradient accumulation."
                  type="tip"
                />
              </>
            )}

            {/* Implementation Best Practices */}
            {section.id === 'implementation' && (
              <>
                <CodeExample
                  code={`# VERL Implementation Best Practices
import logging
import wandb
from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class VERLConfig:
    """Configuration for VERL system"""
    # Component configs
    num_actors: int = 4
    num_critics: int = 2  
    num_rollout_workers: int = 32
    
    # Resource allocation
    actor_gpus: float = 1.0
    critic_gpus: float = 1.0
    rollout_cpus: int = 2
    
    # Training config
    rollout_length: int = 512
    batch_size: int = 8192
    num_epochs: int = 4
    
    # Monitoring
    log_interval: int = 10
    checkpoint_interval: int = 100
    profile_enabled: bool = True

class VERLMonitor:
    """Comprehensive monitoring for distributed RL"""
    
    def __init__(self, config: VERLConfig):
        self.config = config
        self.setup_logging()
        self.setup_metrics()
        
    def setup_logging(self):
        """Setup distributed logging"""
        logging.basicConfig(
            level=logging.INFO,
            format='[%(asctime)s] [%(name)s] %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler('verl_training.log')
            ]
        )
        
        # Initialize wandb for experiment tracking
        wandb.init(
            project="verl-ppo",
            config=self.config.__dict__
        )
    
    def log_system_metrics(self, metrics: Dict[str, Any]):
        """Log system-wide metrics"""
        # Component utilization
        wandb.log({
            "system/actor_gpu_util": metrics["actor_gpu_util"],
            "system/critic_gpu_util": metrics["critic_gpu_util"],
            "system/rollout_cpu_util": metrics["rollout_cpu_util"],
            "system/network_bandwidth_gb": metrics["network_bandwidth"],
        })
        
        # Timing metrics
        wandb.log({
            "timing/rollout_ms": metrics["rollout_time"],
            "timing/actor_update_ms": metrics["actor_update_time"],
            "timing/critic_update_ms": metrics["critic_update_time"],
            "timing/total_iteration_ms": metrics["total_time"],
        })
        
        # Training metrics
        wandb.log({
            "train/actor_loss": metrics["actor_loss"],
            "train/critic_loss": metrics["critic_loss"],
            "train/kl_divergence": metrics["kl_div"],
            "train/explained_variance": metrics["explained_var"],
        })

class FaultTolerantVERL:
    """VERL with fault tolerance"""
    
    def __init__(self, config: VERLConfig):
        self.config = config
        self.setup_checkpointing()
        
    def setup_checkpointing(self):
        """Setup distributed checkpointing"""
        self.checkpoint_manager = CheckpointManager(
            checkpoint_dir="gs://verl-checkpoints",
            keep_last_n=5,
            upload_async=True
        )
    
    @ray.remote
    def fault_tolerant_worker(self, worker_id: int):
        """Worker with automatic recovery"""
        try:
            # Normal operation
            return self.worker_function(worker_id)
        except Exception as e:
            logging.error(f"Worker {worker_id} failed: {e}")
            
            # Attempt recovery
            if self.can_recover(e):
                logging.info(f"Recovering worker {worker_id}")
                self.recover_worker_state(worker_id)
                return self.worker_function(worker_id)
            else:
                # Escalate if cannot recover
                raise
    
    def can_recover(self, exception: Exception) -> bool:
        """Determine if we can recover from this exception"""
        recoverable_errors = [
            ConnectionError,
            TimeoutError,
            torch.cuda.OutOfMemoryError
        ]
        return any(isinstance(exception, err) for err in recoverable_errors)

# Production deployment checklist
def production_checklist():
    return """
    VERL Production Deployment Checklist:
    
    1. Monitoring & Logging
       ✓ Setup distributed logging aggregation
       ✓ Configure metrics dashboards
       ✓ Set up alerts for key metrics
       ✓ Enable distributed tracing
    
    2. Fault Tolerance
       ✓ Implement checkpoint/recovery
       ✓ Handle worker failures gracefully
       ✓ Test disaster recovery procedures
       ✓ Configure automatic restarts
    
    3. Performance
       ✓ Profile and optimize bottlenecks
       ✓ Tune batch sizes and parallelism
       ✓ Optimize network communication
       ✓ Enable mixed precision training
    
    4. Resource Management
       ✓ Configure autoscaling policies
       ✓ Set resource limits and quotas
       ✓ Implement cost monitoring
       ✓ Plan for peak capacity
    
    5. Testing
       ✓ Load test with expected scale
       ✓ Chaos engineering tests
       ✓ Performance regression tests
       ✓ End-to-end integration tests
    """`}
                  language="python"
                  title="VERL Implementation Best Practices"
                  className="mb-6"
                />
                
                <InteractiveDemo demoType="verl-monitoring" className="mb-6" />
                
                <ConfusionClarifier
                  id="verl-debug-tip"
                  title="Debugging Distributed RL"
                  confusion="Distributed systems are hard to debug. How do I troubleshoot VERL?"
                  clarification="Start with single-worker debugging, then scale gradually. Use Ray's dashboard for system monitoring. Add extensive logging at component boundaries. Implement deterministic testing modes. Most importantly: reproduce issues at the smallest scale possible before debugging at scale."
                  type="tip"
                />
              </>
            )}
          </motion.section>
        ))}

        {/* Exercises */}
        {chapter.exercises && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="bg-purple-50 rounded-lg p-6 mb-8"
          >
            <h2 className="text-2xl font-bold mb-4">Exercises</h2>
            <ol className="space-y-4">
              {chapter.exercises.map((exercise: string, index: number) => (
                <li key={index}>
                  <strong>Exercise {index + 1}:</strong> {exercise}
                </li>
              ))}
            </ol>
          </motion.div>
        )}

        {/* Key Takeaways */}
        {chapter.keyTakeaways && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="bg-green-50 rounded-lg p-6 mb-8"
            data-testid="key-takeaways"
          >
            <h2 className="text-2xl font-bold mb-4">Key Takeaways</h2>
            <ul className="space-y-2">
              {chapter.keyTakeaways.map((takeaway: string, index: number) => (
                <li key={index} className="flex items-start gap-2">
                  <CheckCircle className="w-5 h-5 text-green-500 mt-0.5" />
                  <span>{takeaway}</span>
                </li>
              ))}
            </ul>
          </motion.div>
        )}

        {/* Quiz */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="mb-8"
        >
          <QuizSection 
            questions={chapter.quiz}
            onComplete={(score) => {
              console.log(`Quiz completed with score: ${score}`);
            }}
          />
        </motion.div>

        {/* Navigation */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="flex justify-between items-center pt-8 border-t"
        >
          <Link
            href="/chapters"
            className="flex items-center gap-2 text-gray-600 hover:text-gray-900"
          >
            <ArrowLeft size={20} />
            Back to Chapters
          </Link>
          
          {parseInt(chapterId) < Object.keys(chaptersContent).length && (
            <Link
              href={`/chapters/${parseInt(chapterId) + 1}`}
              className="flex items-center gap-2 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700"
            >
              Next: Chapter {parseInt(chapterId) + 1}
              <ArrowRight size={20} />
            </Link>
          )}
        </motion.div>
      </div>
    </div>
  );
}