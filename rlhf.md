# HH-RLHF Code Tutorial: Complete Implementation Guide

This tutorial provides a comprehensive code walkthrough of Human Feedback Reinforcement Learning (HH-RLHF) implementation in VERL, with detailed code snippets, file references, and numerical examples for each training stage.

## Table of Contents
1. [Data Preprocessing with Code Examples](#data-preprocessing)
2. [SFT Training with Examples](#sft-training)
3. [Reward Model Training with Bradley-Terry Loss](#reward-model-training)
4. [PPO Training with Examples](#ppo-training)
5. [Architecture Implementation Details](#architecture-details)

---

## Data Preprocessing

### Code Implementation
**File**: [`verl/examples/data_preprocess/full_hh_rlhf.py`](verl/examples/data_preprocess/full_hh_rlhf.py)

```python
def process_sft_data(data):
    """Convert preference pairs to SFT format"""
    sft_data = []
    for item in data:
        # Add chosen response
        sft_data.append({
            'prompt': item['prompt'],
            'response': item['chosen']
        })
        # Add rejected response  
        sft_data.append({
            'prompt': item['prompt'], 
            'response': item['rejected']
        })
    return sft_data

def process_rm_data(data, train_ratio=0.75):
    """Process data for reward model training"""
    random.shuffle(data)
    split_idx = int(len(data) * train_ratio)
    
    train_data = []
    for item in data[:split_idx]:
        train_data.append({
            'prompt': item['prompt'],
            'chosen': item['chosen'],
            'rejected': item['rejected']
        })
    
    test_data = data[split_idx:]
    return train_data, test_data
```

### Numerical Example: Data Preprocessing

**Input**: Raw HH-RLHF dataset sample
```python
raw_sample = {
    "prompt": "What is the meaning of life?",
    "chosen": "The meaning of life is a complex philosophical question that has been pondered for centuries. Many find meaning through relationships, personal growth, and contributing to something larger than themselves.",
    "rejected": "42"
}
```

**Output**: Three processed datasets

**1. SFT Dataset** (`~/data/full_hh_rlhf/sft/train.parquet`)
```python
sft_samples = [
    {
        "prompt": "What is the meaning of life?",
        "response": "The meaning of life is a complex philosophical question..."
    },
    {
        "prompt": "What is the meaning of life?", 
        "response": "42"
    }
]
# Total: 224,104 samples (both chosen + rejected)
```

**2. RM Dataset** (`~/data/full_hh_rlhf/rm/train.parquet`)
```python
rm_sample = {
    "prompt": "What is the meaning of life?",
    "chosen": "The meaning of life is a complex philosophical question...",
    "rejected": "42"
}
# Train: 84,039 samples, Test: 28,013 samples
```

**3. RL Dataset** (`~/data/full_hh_rlhf/rl/train.parquet`)
```python
rl_sample = {
    "prompt": [
        {"role": "user", "content": "What is the meaning of life?"}
    ],
    "chosen": "The meaning of life is a complex philosophical question...",
    "rejected": "42",
    "data_source": "Dahoas/full-hh-rlhf",
    "ability": "alignment",
    "reward_model": {
        "style": "model",
        "ground_truth": 1.0  # Chosen response preference
    }
}
# Total: 112,052 samples
```

---

## SFT Training

### Code Implementation
**File**: [`verl/verl/trainer/fsdp_sft_trainer.py`](verl/verl/trainer/fsdp_sft_trainer.py:297)

```python
def _compute_loss_and_backward(self, batch, do_backward=True):
    """Compute SFT loss with cross-entropy"""
    input_ids = batch['input_ids']  # [batch_size, seq_len]
    labels = batch['labels']        # [batch_size, seq_len] 
    attention_mask = batch['attention_mask']
    
    # Forward pass through model
    outputs = self.model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels
    )
    
    # Cross-entropy loss for next token prediction
    loss = outputs.loss
    
    if do_backward:
        loss.backward()
    
    return loss
```

### Numerical Example: SFT Training

**Input**: Tokenized prompt + response
```python
# Tokenization
prompt = "What is the meaning of life?"
response = "The meaning of life is a complex philosophical question..."

# Combined input
input_text = f"<|user|>\n{prompt}\n<|assistant|>\n{response}<|end|>"
input_ids = tokenizer.encode(input_text)  # [1, 2847, 374, 279, 7438, ...]
labels = input_ids.copy()
labels[:len(prompt_tokens)] = -100  # Mask prompt tokens

# Shape: [batch_size=4, seq_len=128]
batch = {
    'input_ids': torch.tensor([[1, 2847, 374, ...]]),      # Token IDs
    'labels': torch.tensor([[-100, -100, 374, ...]]),      # Labels (prompt masked)
    'attention_mask': torch.tensor([[1, 1, 1, ...]])       # Attention mask
}
```

**Forward Pass**:
```python
# Model forward pass
logits = model(input_ids)  # Shape: [1, 128, 32000] (vocab_size=32000)

# Cross-entropy loss computation
shift_logits = logits[..., :-1, :].contiguous()  # [1, 127, 32000]
shift_labels = labels[..., 1:].contiguous()      # [1, 127]

loss_fct = CrossEntropyLoss(ignore_index=-100)
loss = loss_fct(shift_logits.view(-1, 32000), shift_labels.view(-1))

# Example loss value
loss = 2.847  # Cross-entropy loss
```

**Training Step**:
```python
# Backward pass
loss.backward()  # Compute gradients

# Optimizer step  
optimizer.step()  # Update parameters
optimizer.zero_grad()  # Clear gradients

# Learning: Model learns to predict next tokens in responses
```

---

## Reward Model Training

### Code Implementation
**File**: [`verl/verl/workers/reward_model/megatron/reward_model.py`](verl/verl/workers/reward_model/megatron/reward_model.py:246)

```python
def loss_func(output):
    """Reward model loss function (placeholder)"""
    # Note: Actual Bradley-Terry loss not implemented in this file
    return torch.tensor(1.0, device=output.device), output

# Reward head architecture
def create_reward_head(hidden_size=4096):
    """Create reward head for scoring"""
    return torch.nn.Linear(hidden_size, 1)  # 4096 -> 1 scalar
```

### Bradley-Terry Loss Implementation (Conceptual)
**Note**: This is the theoretical implementation that should exist but is missing:

```python
def bradley_terry_loss(chosen_rewards, rejected_rewards):
    """
    Bradley-Terry model for preference learning
    P(chosen > rejected) = sigmoid(r_chosen - r_rejected)
    """
    # Compute preference probability
    logits = chosen_rewards - rejected_rewards  # [batch_size, 1]
    
    # Bradley-Terry loss (binary cross-entropy with target=1)
    loss = -torch.log(torch.sigmoid(logits))  # [batch_size, 1]
    
    return loss.mean()

def compute_reward_model_loss(model, batch):
    """Complete reward model training step"""
    # Get chosen and rejected responses
    chosen_input_ids = batch['chosen_input_ids']    # [batch_size, seq_len]
    rejected_input_ids = batch['rejected_input_ids'] # [batch_size, seq_len]
    
    # Forward pass for both responses
    chosen_rewards = model(chosen_input_ids)    # [batch_size, 1]
    rejected_rewards = model(rejected_input_ids) # [batch_size, 1]
    
    # Bradley-Terry loss
    loss = bradley_terry_loss(chosen_rewards, rejected_rewards)
    
    return loss
```

### Numerical Example: Reward Model Training

**Input**: Preference pair
```python
prompt = "What is the meaning of life?"
chosen = "The meaning of life is a complex philosophical question that has been pondered for centuries..."
rejected = "42"

# Tokenize both responses
chosen_input = f"<|user|>\n{prompt}\n<|assistant|>\n{chosen}<|end|>"
rejected_input = f"<|user|>\n{prompt}\n<|assistant|>\n{rejected}<|end|>"

chosen_ids = tokenizer.encode(chosen_input)    # [1, 2847, 374, ..., 2564]
rejected_ids = tokenizer.encode(rejected_input) # [1, 2847, 374, ..., 2983]
```

**Forward Pass**:
```python
# Model forward pass (with reward head)
chosen_hidden = model.transformer(chosen_ids)     # [1, seq_len, 4096]
rejected_hidden = model.transformer(rejected_ids) # [1, seq_len, 4096]

# Extract final hidden states
chosen_final = chosen_hidden[:, -1, :]    # [1, 4096] (last token)
rejected_final = rejected_hidden[:, -1, :] # [1, 4096] (last token)

# Reward head (Linear: 4096 -> 1)
reward_head = torch.nn.Linear(4096, 1)
chosen_reward = reward_head(chosen_final)    # [1, 1] -> 0.73
rejected_reward = reward_head(rejected_final) # [1, 1] -> 0.21
```

**Bradley-Terry Loss Calculation**:
```python
# Preference logit
logit = chosen_reward - rejected_reward  # 0.73 - 0.21 = 0.52

# Bradley-Terry probability
prob_chosen = torch.sigmoid(logit)  # sigmoid(0.52) = 0.627

# Loss (negative log-likelihood)
loss = -torch.log(prob_chosen)  # -log(0.627) = 0.467

# Backward pass
loss.backward()  # Gradients: increase chosen_reward, decrease rejected_reward
```

**Training Effect**:
```python
# After training:
# chosen_reward increases: 0.73 -> 0.85
# rejected_reward decreases: 0.21 -> 0.15
# Better separation between chosen/rejected responses
```

---

## PPO Training

### Code Implementation
**File**: [`verl/verl/trainer/ppo/core_algos.py`](verl/verl/trainer/ppo/core_algos.py:533)

```python
def compute_policy_loss(
    old_log_prob,      # [batch_size, seq_len] 
    log_prob,          # [batch_size, seq_len]
    advantages,        # [batch_size, seq_len]
    response_mask,     # [batch_size, seq_len]
    cliprange=0.2,     # PPO clipping parameter
    loss_agg_mode="token-mean"
):
    """Compute PPO policy loss with clipping"""
    
    # Compute probability ratio
    ratio = torch.exp(log_prob - old_log_prob)  # π_θ(a|s) / π_θ_old(a|s)
    
    # Clipped objective
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1 - cliprange, 1 + cliprange) * advantages
    
    # PPO loss (negative because we want to maximize)
    policy_loss = -torch.min(surr1, surr2)
    
    # Aggregate loss over tokens
    return agg_loss(policy_loss, response_mask, loss_agg_mode)

def compute_value_loss(vpreds, returns, values, response_mask, cliprange_value=0.2):
    """Compute value function loss"""
    
    # Clipped value loss
    vpredclipped = values + torch.clamp(vpreds - values, -cliprange_value, cliprange_value)
    
    vf_losses1 = (vpreds - returns) ** 2
    vf_losses2 = (vpredclipped - returns) ** 2
    
    vf_loss = torch.max(vf_losses1, vf_losses2)
    
    return agg_loss(vf_loss, response_mask, "token-mean")
```

### Reward Model Scoring
**File**: [`verl/verl/workers/reward_model/megatron/reward_model.py`](verl/verl/workers/reward_model/megatron/reward_model.py:175)

```python
def compute_reward(self, input_ids, attention_mask, position_ids=None):
    """Compute reward scores for responses"""
    
    # Forward pass through transformer
    outputs = self.model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids
    )
    
    # Get token-level rewards from reward head
    token_level_rewards = outputs.logits  # [batch_size, seq_len, 1]
    token_level_rewards = token_level_rewards.squeeze(-1)  # [batch_size, seq_len]
    
    # Extract reward at last valid token position
    ends = attention_mask.cumsum(dim=-1).argmax(dim=-1).view(-1, 1)  # [batch_size, 1]
    rewards = torch.gather(token_level_rewards, dim=1, index=ends)   # [batch_size, 1]
    
    return rewards.squeeze(-1)  # [batch_size]
```

### Numerical Example: PPO Training

**Input**: Actor generates response to prompt
```python
prompt = "What is the meaning of happiness?"
# Actor generates response
generated_response = "Happiness is a positive emotional state characterized by joy, satisfaction, and fulfillment. It can come from meaningful relationships, personal achievements, and living according to one's values."

# Tokenize
input_text = f"<|user|>\n{prompt}\n<|assistant|>\n{generated_response}<|end|>"
input_ids = tokenizer.encode(input_text)  # [1, 2847, 374, ...]
response_start = len(tokenizer.encode(f"<|user|>\n{prompt}\n<|assistant|>\n"))
response_mask = torch.zeros_like(input_ids)
response_mask[response_start:] = 1  # Mask only response tokens
```

**Reward Model Scoring**:
```python
# Forward pass through reward model
reward_outputs = reward_model(input_ids)  # [1, seq_len, 1]

# Extract reward at last token
last_token_idx = len(input_ids) - 1
reward_score = reward_outputs[0, last_token_idx, 0]  # Scalar reward

# Example reward score
reward_score = 0.73  # Higher is better
```

**PPO Loss Calculation**:
```python
# Actor probabilities
old_log_probs = torch.tensor([-2.1, -1.8, -2.3, -1.9])  # Old policy log probs
new_log_probs = torch.tensor([-2.0, -1.7, -2.2, -1.8])  # New policy log probs

# Advantage estimation (from reward and value function)
rewards = torch.tensor([0.73])  # Reward model score
values = torch.tensor([0.65])   # Critic value estimate
advantages = rewards - values   # 0.73 - 0.65 = 0.08

# Expand advantages to token level
token_advantages = torch.tensor([0.08, 0.08, 0.08, 0.08])  # Per token

# PPO ratio calculation
ratios = torch.exp(new_log_probs - old_log_probs)
# exp([-2.0 - (-2.1), -1.7 - (-1.8), ...]) = [1.105, 1.105, 1.105, 1.105]

# PPO loss computation
cliprange = 0.2
surr1 = ratios * token_advantages  # [0.088, 0.088, 0.088, 0.088]
surr2 = torch.clamp(ratios, 1-0.2, 1+0.2) * token_advantages  # Clipped
policy_loss = -torch.min(surr1, surr2).mean()  # -0.088
```

**Training Step**:
```python
# Total loss
total_loss = policy_loss + 0.5 * value_loss + 0.01 * entropy_loss
# Example: -0.088 + 0.5 * 0.02 + 0.01 * 1.2 = -0.066

# Backward pass
total_loss.backward()

# Update actor parameters
actor_optimizer.step()
actor_optimizer.zero_grad()

# Effect: Actor learns to generate responses with higher reward scores
```

---

## Architecture Implementation Details

### Reward Head Architecture
**File**: [`verl/verl/utils/model.py`](verl/verl/utils/model.py:469)

```python
# Replace language modeling head with reward head
parallel_model.output_layer = LinearForLastLayer(
    input_size=tfconfig.hidden_size,  # 4096 for deepseek-7b
    output_size=1,                    # Single scalar reward
    config=tfconfig
)

class LinearForLastLayer(torch.nn.Module):
    """Linear layer for reward head"""
    def __init__(self, input_size, output_size, config):
        super().__init__()
        self.linear = torch.nn.Linear(input_size, output_size, bias=False)
        
    def forward(self, hidden_states):
        # hidden_states: [batch_size, seq_len, hidden_size]
        return self.linear(hidden_states)  # [batch_size, seq_len, 1]
```

### Megatron Tensor Parallelism
**File**: [`examples/ppo_trainer/run_deepseek_full_hh_rlhf.sh`](examples/ppo_trainer/run_deepseek_full_hh_rlhf.sh)

```bash
# Tensor parallelism configuration
actor_rollout_ref.rollout.tensor_model_parallel_size=4  # 4-way TP
reward_model.megatron.tensor_model_parallel_size=4      # 4-way TP  
critic.megatron.tensor_model_parallel_size=1            # No TP

# Memory configuration
trainer.n_gpus_per_node=8                               # 8 GPUs total
actor_rollout_ref.rollout.gpu_memory_utilization=0.4   # 40% GPU memory
```

### Distributed Training Setup
```python
# Tensor parallel weight distribution
# Original weight: [4096, 32000] (hidden_size x vocab_size)
# With 4-way TP: Each GPU gets [4096, 8000] (vocab_size / 4)

class TensorParallelLinear:
    def __init__(self, input_size, output_size, tp_size=4):
        self.tp_size = tp_size
        self.output_size_per_partition = output_size // tp_size
        
        # Each GPU gets a slice of the weight matrix
        self.weight = torch.randn(input_size, self.output_size_per_partition)
    
    def forward(self, x):
        # x: [batch_size, seq_len, input_size]
        local_output = torch.matmul(x, self.weight)  # Local computation
        
        # All-gather to combine results from all GPUs
        global_output = all_gather(local_output)  # Communication
        
        return global_output
```

---

## Complete Training Pipeline Example

### Full Numerical Walkthrough

**1. Data Sample**:
```python
sample = {
    "prompt": "How can I improve my productivity?",
    "chosen": "To improve productivity, try time-blocking, eliminating distractions, taking regular breaks, and focusing on high-priority tasks first.",
    "rejected": "Just work harder and sleep less."
}
```

**2. SFT Training**:
```python
# Both responses used for SFT
sft_loss_chosen = 1.23   # Cross-entropy loss for chosen response
sft_loss_rejected = 2.45 # Cross-entropy loss for rejected response
# Model learns to generate both types of responses
```

**3. Reward Model Training** (Missing in current implementation):
```python
# Should train reward model to prefer chosen over rejected
chosen_reward = 0.85     # Reward for chosen response
rejected_reward = 0.23   # Reward for rejected response
bradley_terry_loss = -log(sigmoid(0.85 - 0.23)) = 0.38
# Model learns: chosen > rejected
```

**4. PPO Training**:
```python
# Actor generates new response
new_response = "Productivity can be enhanced through effective time management, setting clear goals, minimizing interruptions, and maintaining work-life balance."

# Reward model scores new response
reward_score = 0.78  # Good response gets high reward

# PPO updates actor to generate higher-reward responses
advantage = reward_score - baseline_value = 0.78 - 0.65 = 0.13
policy_loss = -0.13 * probability_ratio  # Negative for maximization
# Actor learns to generate responses similar to this high-reward one
```

---

## Key Files Reference

### Core Implementation Files
- **Data Preprocessing**: [`verl/examples/data_preprocess/full_hh_rlhf.py`](verl/examples/data_preprocess/full_hh_rlhf.py)
- **PPO Algorithms**: [`verl/verl/trainer/ppo/core_algos.py`](verl/verl/trainer/ppo/core_algos.py)
- **Reward Model**: [`verl/verl/workers/reward_model/megatron/reward_model.py`](verl/verl/workers/reward_model/megatron/reward_model.py)
- **Training Script**: [`examples/ppo_trainer/run_deepseek_full_hh_rlhf.sh`](examples/ppo_trainer/run_deepseek_full_hh_rlhf.sh)
- **Model Utils**: [`verl/verl/utils/model.py`](verl/verl/utils/model.py)
- **SFT Trainer**: [`verl/verl/trainer/fsdp_sft_trainer.py`](verl/verl/trainer/fsdp_sft_trainer.py)

### Configuration Files
- **PPO Config**: [`verl/verl/trainer/config/ppo_megatron_trainer.yaml`](verl/verl/trainer/config/ppo_megatron_trainer.yaml)

### Generated Data Files
- **SFT Data**: `~/data/full_hh_rlhf/sft/train.parquet`
- **RM Data**: `~/data/full_hh_rlhf/rm/train.parquet`, `~/data/full_hh_rlhf/rm/test.parquet`
- **RL Data**: `~/data/full_hh_rlhf/rl/train.parquet`

---

## Critical Implementation Gap

**Missing Component**: The Bradley-Terry loss for reward model training is not implemented in the current codebase. The reward model loss function in [`verl/verl/workers/reward_model/megatron/reward_model.py:246`](verl/verl/workers/reward_model/megatron/reward_model.py:246) is a placeholder:

```python
def loss_func(output):
    return torch.tensor(1.0, device=output.device), output
```

This means the reward model is never actually trained on preference data, leading to the self-judgment problem identified in our analysis.

**Proper Implementation Would Require**:
```python
def bradley_terry_loss_func(chosen_output, rejected_output):
    """Proper reward model training loss"""
    chosen_rewards = chosen_output.logits.squeeze(-1)    # [batch_size]
    rejected_rewards = rejected_output.logits.squeeze(-1) # [batch_size]
    
    # Bradley-Terry loss
    logits = chosen_rewards - rejected_rewards
    loss = -torch.log(torch.sigmoid(logits)).mean()
    
    return loss, {"chosen_rewards": chosen_rewards, "rejected_rewards": rejected_rewards}
```

This tutorial demonstrates both the sophisticated technical implementation of VERL and the critical gap that prevents it from achieving true human preference alignment in the current configuration.