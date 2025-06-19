# RLHF (Reinforcement Learning from Human Feedback) Deep Research

## Executive Summary

Reinforcement Learning from Human Feedback (RLHF) has emerged as the dominant paradigm for aligning large language models with human preferences. This research explores the complete RLHF pipeline, implementation details, theoretical foundations, and practical insights from VERL's implementation.

## Table of Contents
1. [RLHF Pipeline Overview](#rlhf-pipeline-overview)
2. [Theoretical Foundations](#theoretical-foundations)
3. [Implementation Architecture](#implementation-architecture)
4. [Critical Components Analysis](#critical-components-analysis)
5. [Advanced Techniques](#advanced-techniques)
6. [Common Pitfalls and Solutions](#common-pitfalls-and-solutions)
7. [Educational Insights](#educational-insights)

## 1. RLHF Pipeline Overview

### 1.1 Three-Stage Training Process

RLHF consists of three critical stages, each building upon the previous:

#### Stage 1: Supervised Fine-Tuning (SFT)
- **Purpose**: Establish a base capability for the model to follow instructions
- **Data**: Human demonstrations of desired behavior
- **Loss**: Standard cross-entropy loss for next-token prediction
- **Key Insight**: Both chosen and rejected responses are used for training diversity

#### Stage 2: Reward Model Training
- **Purpose**: Learn human preferences from comparison data
- **Data**: Pairs of (prompt, chosen_response, rejected_response)
- **Loss**: Bradley-Terry model for preference learning
- **Architecture**: Language model with scalar reward head

#### Stage 3: PPO Optimization
- **Purpose**: Fine-tune the policy to maximize reward while staying close to reference
- **Algorithm**: Proximal Policy Optimization with KL penalty
- **Components**: Actor (policy), Critic (value function), Reference Model

### 1.2 Data Flow Through Pipeline

```
Raw Preferences → Data Preprocessing → Three Training Stages
                                      ↓
                                   SFT Data (224K samples)
                                   RM Data (112K train/test)
                                   RL Data (112K prompts)
```

## 2. Theoretical Foundations

### 2.1 Bradley-Terry Model for Preferences

The Bradley-Terry model is the theoretical foundation for reward model training:

```python
P(response_A > response_B) = exp(r_A) / (exp(r_A) + exp(r_B))
                           = sigmoid(r_A - r_B)
```

**Key Properties**:
- Transitivity: If A > B and B > C, then A > C
- Scale invariance: Only relative differences matter
- Maximum likelihood estimation via logistic regression

### 2.2 PPO Objective Function

The PPO objective balances reward maximization with distribution proximity:

```python
L_PPO = E_t[min(r_t * A_t, clip(r_t, 1-ε, 1+ε) * A_t)]
      - β * KL[π_θ || π_ref]
      + γ * H[π_θ]
```

Where:
- `r_t = π_θ(a_t|s_t) / π_old(a_t|s_t)`: Probability ratio
- `A_t`: Advantage estimate
- `β`: KL penalty coefficient
- `γ`: Entropy bonus coefficient

### 2.3 KL Divergence Constraint

The KL constraint prevents the policy from deviating too far:

```python
KL[π_θ || π_ref] = E_π_θ[log(π_θ/π_ref)]
```

This ensures:
- Stability during training
- Preservation of language modeling capabilities
- Prevention of reward hacking

## 3. Implementation Architecture

### 3.1 Model Architecture Components

#### Base Language Model
- Transformer architecture (e.g., Llama, DeepSeek, Qwen)
- Hidden size: 4096 (for 7B models)
- Vocabulary size: 32000-152000

#### Reward Head
```python
class RewardHead(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.linear = nn.Linear(hidden_size, 1, bias=False)
    
    def forward(self, hidden_states):
        # Extract last token hidden state
        last_hidden = hidden_states[:, -1, :]  # [batch, hidden_size]
        return self.linear(last_hidden)         # [batch, 1]
```

#### Value Head (Critic)
```python
class ValueHead(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.ReLU()
        self.output = nn.Linear(hidden_size, 1)
    
    def forward(self, hidden_states):
        # Per-token value estimation
        h = self.activation(self.dense(hidden_states))
        return self.output(h)  # [batch, seq_len, 1]
```

### 3.2 Distributed Training Architecture

#### Worker Group Separation
1. **ActorRolloutRef Worker**
   - Manages policy model, generation, and reference
   - Can be colocated for efficiency
   - Handles log probability computation

2. **Critic Worker**
   - Value function estimation
   - Advantage computation support
   - Often smaller model than actor

3. **Reward Model Worker**
   - Scores generated responses
   - Can be model-based or rule-based
   - Critical for preference learning

#### Parallelism Strategies
- **Data Parallelism**: Split batch across GPUs
- **Tensor Parallelism**: Split model layers
- **Pipeline Parallelism**: Split model stages
- **Hybrid**: Combine strategies for large models

## 4. Critical Components Analysis

### 4.1 SFT Implementation Details

#### Loss Masking Strategy
```python
def compute_sft_loss(logits, labels, prompt_mask):
    # Only compute loss on response tokens
    shift_logits = logits[..., :-1, :]
    shift_labels = labels[..., 1:]
    
    # Mask prompt tokens
    loss_mask = ~prompt_mask[..., 1:]
    
    # Cross-entropy with masking
    loss = F.cross_entropy(
        shift_logits[loss_mask],
        shift_labels[loss_mask]
    )
    return loss
```

#### Data Augmentation
- Include both chosen and rejected responses
- Maintains response diversity
- Prevents mode collapse

### 4.2 Reward Model Training Gap

**Critical Finding**: VERL's current implementation has a placeholder loss function:

```python
# Current (non-functional)
def loss_func(output):
    return torch.tensor(1.0, device=output.device), output

# Proper implementation needed
def bradley_terry_loss(chosen_logits, rejected_logits):
    # Preference probability
    preference_logits = chosen_logits - rejected_logits
    
    # Binary cross-entropy with implicit target=1
    loss = F.binary_cross_entropy_with_logits(
        preference_logits,
        torch.ones_like(preference_logits)
    )
    return loss
```

### 4.3 PPO Implementation Details

#### Advantage Estimation (GAE)
```python
def compute_gae(rewards, values, gamma=0.99, lam=0.95):
    advantages = torch.zeros_like(rewards)
    lastgaelam = 0
    
    for t in reversed(range(len(rewards))):
        if t == len(rewards) - 1:
            nextnonterminal = 0
            nextvalues = 0
        else:
            nextnonterminal = 1
            nextvalues = values[t + 1]
        
        delta = rewards[t] + gamma * nextvalues * nextnonterminal - values[t]
        advantages[t] = lastgaelam = delta + gamma * lam * nextnonterminal * lastgaelam
    
    returns = advantages + values
    return advantages, returns
```

#### Clipped Surrogate Objective
```python
def ppo_loss(log_probs, old_log_probs, advantages, clip_range=0.2):
    # Probability ratio
    ratio = torch.exp(log_probs - old_log_probs)
    
    # Clipped and unclipped objectives
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range) * advantages
    
    # Take minimum (pessimistic bound)
    policy_loss = -torch.min(surr1, surr2).mean()
    
    # Additional metrics
    clip_fraction = (torch.abs(ratio - 1.0) > clip_range).float().mean()
    
    return policy_loss, clip_fraction
```

## 5. Advanced Techniques

### 5.1 Rejection Sampling for Response Quality

```python
def rejection_sampling(model, prompts, reward_model, temperature=0.7, top_p=0.9, best_of=4):
    all_responses = []
    all_rewards = []
    
    for _ in range(best_of):
        # Generate responses
        responses = model.generate(
            prompts,
            temperature=temperature,
            top_p=top_p
        )
        
        # Score with reward model
        rewards = reward_model(responses)
        
        all_responses.append(responses)
        all_rewards.append(rewards)
    
    # Select best responses
    all_rewards = torch.stack(all_rewards, dim=1)  # [batch, best_of]
    best_indices = all_rewards.argmax(dim=1)       # [batch]
    
    best_responses = []
    for i, idx in enumerate(best_indices):
        best_responses.append(all_responses[idx][i])
    
    return best_responses
```

### 5.2 Reward Shaping and Normalization

```python
class RewardNormalizer:
    def __init__(self, gamma=0.99):
        self.gamma = gamma
        self.mean = 0
        self.var = 1
        self.count = 1e-4
    
    def update(self, rewards):
        batch_mean = rewards.mean()
        batch_var = rewards.var()
        batch_count = rewards.numel()
        
        delta = batch_mean - self.mean
        total_count = self.count + batch_count
        
        self.mean += delta * batch_count / total_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + delta**2 * self.count * batch_count / total_count
        self.var = M2 / total_count
        self.count = total_count
    
    def normalize(self, rewards):
        return (rewards - self.mean) / (self.var**0.5 + 1e-8)
```

### 5.3 Adaptive KL Control

```python
class AdaptiveKLController:
    def __init__(self, init_kl_coef=0.1, target_kl=6.0, horizon=10000):
        self.kl_coef = init_kl_coef
        self.target_kl = target_kl
        self.horizon = horizon
    
    def update(self, current_kl, n_steps):
        # Proportional control
        proportional_error = np.clip(current_kl / self.target_kl - 1, -0.2, 0.2)
        
        # Integral term
        mult = 1 + proportional_error * n_steps / self.horizon
        
        # Update coefficient
        self.kl_coef *= mult
        
        return self.kl_coef
```

## 6. Common Pitfalls and Solutions

### 6.1 Reward Hacking

**Problem**: Model finds unintended ways to maximize reward

**Solutions**:
1. **Diverse Reward Models**: Train ensemble of reward models
2. **Reward Clipping**: Limit maximum reward magnitude
3. **KL Penalty**: Constrain deviation from base model
4. **Human-in-the-loop**: Periodic evaluation and adjustment

### 6.2 Distribution Shift

**Problem**: Training distribution differs from generation distribution

**Solutions**:
1. **Online Generation**: Generate training data on-policy
2. **Importance Sampling**: Weight samples by probability ratio
3. **Replay Buffer**: Mix on-policy and off-policy data

### 6.3 Catastrophic Forgetting

**Problem**: Model loses pre-training capabilities

**Solutions**:
1. **Reference Model**: KL penalty from original model
2. **Task Mixing**: Include pre-training tasks
3. **Elastic Weight Consolidation**: Protect important weights

### 6.4 Optimization Instability

**Problem**: Training diverges or oscillates

**Solutions**:
1. **Gradient Clipping**: Limit gradient magnitude
2. **Learning Rate Scheduling**: Careful decay
3. **Early Stopping**: Monitor validation metrics
4. **PPO Clipping**: Limit policy updates

## 7. Educational Insights

### 7.1 Key Concepts for Course

1. **Preference Learning**
   - Bradley-Terry model intuition
   - Pairwise vs. pointwise feedback
   - Transitivity and consistency

2. **Policy Optimization**
   - Trust region methods
   - Importance sampling
   - Advantage estimation

3. **Distributed Training**
   - Model parallelism strategies
   - Communication patterns
   - Synchronization challenges

### 7.2 Interactive Demonstrations

1. **Preference Visualization**
   - Show how reward model learns preferences
   - Interactive Bradley-Terry probability calculator
   - Preference graph construction

2. **PPO Mechanics**
   - Visualize policy updates with clipping
   - Show KL divergence effects
   - Advantage landscape exploration

3. **Training Dynamics**
   - Real-time loss curves
   - Reward distribution evolution
   - Policy drift visualization

### 7.3 Hands-on Exercises

1. **Implement Bradley-Terry Loss**
   ```python
   def exercise_bradley_terry(chosen_rewards, rejected_rewards):
       # TODO: Implement preference probability
       # TODO: Compute negative log-likelihood
       pass
   ```

2. **Build Reward Head**
   ```python
   class ExerciseRewardModel(nn.Module):
       def __init__(self, base_model):
           # TODO: Initialize reward head
           # TODO: Handle last token extraction
           pass
   ```

3. **PPO Update Step**
   ```python
   def exercise_ppo_step(policy, old_log_probs, advantages):
       # TODO: Compute new log probabilities
       # TODO: Calculate clipped objective
       # TODO: Add KL penalty
       pass
   ```

### 7.4 Common Misconceptions

1. **"Reward Model is Always Right"**
   - Reality: Reward model has limited coverage
   - Solution: Uncertainty estimation, ensemble methods

2. **"Higher Reward is Always Better"**
   - Reality: Can lead to reward hacking
   - Solution: Multi-objective optimization, constraints

3. **"RLHF Removes All Biases"**
   - Reality: Reflects annotator preferences
   - Solution: Diverse annotation, bias monitoring

## Conclusion

RLHF represents a powerful paradigm for aligning language models with human preferences, but requires careful implementation of all components. The three-stage pipeline - SFT, reward modeling, and PPO - each play crucial roles in the final model behavior. Understanding the theoretical foundations, implementation details, and common pitfalls is essential for successful RLHF deployment.

Key takeaways:
1. Proper reward model training with Bradley-Terry loss is critical
2. PPO provides stable policy optimization with appropriate constraints
3. Distributed training enables scaling to large models
4. Careful monitoring prevents reward hacking and distribution shift
5. The approach reflects human annotator preferences and biases