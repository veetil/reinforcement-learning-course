# VERL (Versatile Reinforcement Learning) Distributed System Architecture Research

## Executive Summary

VERL (Volcano Engine Reinforcement Learning) is a flexible, efficient, and production-ready RL training framework designed for large language models (LLMs). It implements the HybridFlow architecture that decouples control flow from computation flow, enabling efficient distributed training while maintaining programming simplicity.

## 1. Core Architecture: Separation of Worker Groups

### 1.1 Worker Group Organization

VERL separates its distributed components into distinct worker groups, each responsible for specific computational tasks:

#### **ActorRolloutRef Worker Group**
- **Purpose**: Manages actor model, rollout generation, and reference policy
- **Flexibility**: Can be instantiated as:
  - Single actor
  - Single rollout
  - Single reference policy
  - Combined actor/rollout (for fast weight transfer using NCCL)
  - Combined actor/rollout/ref (for efficient LoRA PPO implementation)
- **Key Operations**:
  - `generate_sequences`: Sample generation from prompts
  - `compute_log_prob`: Log-probability computation using actor
  - `compute_ref_log_prob`: Log-probability computation using reference policy
  - `save_checkpoint`: Model checkpointing

#### **Critic Worker Group**
- **Purpose**: Handles value function computation for advantage estimation
- **Operations**: 
  - `compute_values`: Estimate state values for PPO advantage calculation
  - `update_critic`: Update critic network parameters

#### **Reward Model Worker Group**
- **Purpose**: Computes reward scores for generated sequences
- **Flexibility**: Supports both:
  - Model-based rewards (using neural networks)
  - Function-based rewards (rule-based scoring for math, coding, etc.)
- **Operations**:
  - `compute_scores`: Calculate rewards for generated responses

#### **Reference Policy Worker Group**
- **Purpose**: Maintains the original policy for KL divergence computation
- **Note**: Often colocated with Actor for memory efficiency in LoRA implementations

## 2. HybridFlow Architecture

### 2.1 Design Philosophy

The HybridFlow architecture addresses a fundamental challenge in LLM-scale RL: how to maintain both computational efficiency and programming flexibility when dealing with multi-process neural network training.

#### **Two-Level Dataflow Problem**

1. **Control Flow** (High Level):
   - Defines RL algorithm logic (e.g., PPO sequence: rollout → advantage computation → training)
   - Expresses core algorithmic patterns
   - Runs on a single controller process

2. **Computation Flow** (Low Level):
   - Neural network operations (forward/backward/optimization)
   - Distributed across multiple GPUs/processes
   - Handled by computation backends (FSDP, Megatron-LM, vLLM)

### 2.2 Architecture Benefits

#### **Flexibility**
- Easy algorithm extension: New RL algorithms can be implemented in few lines of code
- Backend agnostic: Switch between FSDP, Megatron-LM without changing algorithm code
- Flexible device mapping: Models can be placed on different GPU sets dynamically

#### **Efficiency**
- Asynchronous execution: Operations trigger automatically when inputs are available
- Minimal communication overhead through careful data movement design
- 3D-HybridEngine for efficient model resharding

## 3. Asynchronous Execution Model

### 3.1 Execution Pattern

```
Controller (Single Process)          Workers (Multiple Processes)
     │                                      │
     ├─[1]─> Dispatch Data ─────────────────┤
     │                                      ├─> Worker 1 (GPU 0)
     │                                      ├─> Worker 2 (GPU 1)
     │       (Asynchronous Execution)       ├─> Worker 3 (GPU 2)
     │                                      └─> Worker N (GPU N)
     │                                      │
     └─[2]─< Collect Results <──────────────┘
```

### 3.2 Key Features

1. **Data Futures**: Operations return immediately with futures, allowing controller to initiate new operations while previous ones execute

2. **Automatic Triggering**: When models are on separate devices, execution triggers as soon as inputs become available

3. **Overlapped Computation**: Controller can dispatch to critic while actor is still processing, maximizing GPU utilization

## 4. Resource Pool Management and GPU Allocation

### 4.1 Resource Pool Architecture

```python
resource_pool_spec = {
    'global_pool': [n_gpus_per_node] * nnodes,
    'actor_pool': [8] * 4,  # 32 GPUs for actor
    'critic_pool': [4] * 4,  # 16 GPUs for critic
}

mapping = {
    Role.ActorRollout: 'actor_pool',
    Role.Critic: 'critic_pool',
    Role.RefPolicy: 'actor_pool',  # Colocated with actor
}
```

### 4.2 Placement Strategies

#### **Colocated Placement**
- All models share same GPU set
- Advantages: Minimal communication overhead, memory efficiency
- Disadvantages: Less parallelism, potential memory constraints

#### **Split Placement**
- Different models on different GPU sets
- Advantages: Better parallelism, can optimize each model's parallelism strategy
- Disadvantages: Higher communication overhead

#### **Hybrid Placement**
- Strategic colocation (e.g., Actor+Reference for LoRA)
- Balance between efficiency and flexibility

### 4.3 Dynamic Resource Allocation

VERL supports runtime resource allocation adjustments:
- Models can be moved between resource pools
- Parallelism strategies can be changed (TP/PP/DP sizes)
- Automatic load balancing based on workload

## 5. Data Pipeline Design for High Throughput

### 5.1 Data Movement Optimization

#### **Dispatch Modes**

1. **DP_COMPUTE_PROTO**: 
   - Splits data by data parallel dimension
   - Each worker processes 1/DP of the data
   - Results concatenated on controller

2. **MEGATRON_COMPUTE**:
   - Handles complex 3D parallelism (TP/PP/DP)
   - Ensures correct data routing across pipeline stages

3. **ONE_TO_ALL**:
   - Broadcasts same data to all workers
   - Used for operations like model synchronization

### 5.2 Protocol Design

```python
@register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
def generate_sequences(self, prompts: DataProto) -> DataProto:
    # Automatically handles:
    # 1. Data splitting across DP dimension
    # 2. Remote dispatch to workers
    # 3. Result collection and concatenation
    pass
```

### 5.3 3D-HybridEngine for Resharding

The 3D-HybridEngine provides zero memory redundancy and minimal communication overhead during model parameter resharding between training and generation phases:

1. **Memory Efficiency**: No duplicate storage of weights
2. **Communication Optimization**: Minimal data movement during transitions
3. **Performance**: 1.53× throughput improvement vs baselines

## 6. Synchronization Patterns

### 6.1 Worker Group Synchronization

```python
# Driver process synchronization pattern
for prompt in dataloader:
    # All operations return futures immediately
    output_future = actor_rollout_ref_wg.generate_sequences(prompt)
    
    # Can start other operations while generation runs
    old_log_prob_future = actor_rollout_ref_wg.compute_log_prob(output_future)
    values_future = critic_wg.compute_values(output_future)
    
    # Synchronization happens only when needed
    output = ray.get(output_future)
    old_log_prob = ray.get(old_log_prob_future)
    values = ray.get(values_future)
```

### 6.2 Hierarchical Synchronization

1. **Intra-Worker**: NCCL for tensor parallelism within a model
2. **Inter-Worker**: Ray futures for cross-worker communication
3. **Controller-Worker**: Asynchronous dispatch with explicit synchronization points

## 7. Scaling Strategies and Performance Optimizations

### 7.1 Parallelism Strategies

#### **Data Parallelism (DP)**
- Each worker has full model replica
- Processes different data batches
- Scales well for smaller models

#### **Tensor Parallelism (TP)**
- Model layers split across GPUs
- Required for models that don't fit on single GPU
- Higher communication overhead

#### **Pipeline Parallelism (PP)**
- Model stages distributed across GPUs
- Enables training of very large models
- Complex scheduling requirements

#### **3D Parallelism**
- Combines DP, TP, and PP
- Optimal for very large models (70B+)
- Supported up to 671B parameters

### 7.2 Performance Optimizations

1. **Sequence Packing**: Remove padding for efficient computation
2. **Dynamic Batch Sizing**: Process similar token counts per batch
3. **Activation Offloading**: Move activations to CPU during backward pass
4. **Flash Attention 2**: Efficient attention computation
5. **Ulysses Sequence Parallelism**: For long context (32K+) training

### 7.3 Rollout Optimization

Key tuning parameters:
- `gpu_memory_utilization`: 0.5-0.7 for balance
- `max_num_seqs`: Increase for higher throughput
- `max_num_batched_tokens`: > 2048 recommended
- Smaller `tensor_parallel_size` for more vLLM replicas

## 8. Best Practices for Distributed PPO Training

### 8.1 Algorithm Implementation

```python
# Simple PPO loop with distributed execution
for epoch in range(num_epochs):
    for prompt_batch in dataloader:
        # Rollout phase
        trajectories = actor_rollout_ref_wg.generate_sequences(prompt_batch)
        
        # Compute required values
        old_log_probs = actor_rollout_ref_wg.compute_log_prob(trajectories)
        ref_log_probs = actor_rollout_ref_wg.compute_ref_log_prob(trajectories)
        values = critic_wg.compute_values(trajectories)
        rewards = reward_wg.compute_scores(trajectories)
        
        # Advantage computation (on controller)
        advantages = compute_advantages(values, rewards)
        
        # Update phase
        actor_rollout_ref_wg.update_actor(trajectories, advantages)
        critic_wg.update_critic(trajectories, advantages)
```

### 8.2 Configuration Best Practices

1. **Batch Sizing**:
   - Use `*micro_batch_size_per_gpu` for local control
   - Global batch sizes are automatically normalized
   - Forward-only operations can use larger batches

2. **Memory Management**:
   - Enable gradient checkpointing for larger batches
   - Use activation offloading with FSDP
   - Consider LoRA for memory-constrained scenarios

3. **Communication Optimization**:
   - Colocate models that frequently exchange data
   - Use appropriate parallelism strategy for model size
   - Minimize controller-worker communication frequency

### 8.3 Debugging and Monitoring

1. **Enable Logging**:
   ```yaml
   actor_rollout_ref:
     rollout:
       disable_log_stats: false  # Enable rollout statistics
   ```

2. **Performance Metrics**:
   - Monitor GPU utilization
   - Track communication overhead
   - Measure time per PPO iteration

3. **Common Issues**:
   - OOM: Reduce batch size or increase gradient checkpointing
   - Low throughput: Check parallelism strategy and batch sizes
   - Communication bottlenecks: Consider colocation

## 9. Educational Insights

### 9.1 Key Design Lessons

1. **Separation of Concerns**: Decoupling control and computation flows enables both flexibility and efficiency

2. **Abstraction Layers**: Well-designed abstractions (WorkerGroup, DataProto) hide complexity while maintaining performance

3. **Hybrid Approaches**: Combining single-controller simplicity with multi-controller efficiency

### 9.2 Architecture Patterns

1. **Producer-Consumer Pattern**: Controller produces work, workers consume and process
2. **Future-Based Concurrency**: Asynchronous operations with explicit synchronization
3. **Resource Pooling**: Dynamic allocation and sharing of computational resources

### 9.3 Scalability Principles

1. **Hierarchical Parallelism**: Different parallelism strategies at different levels
2. **Communication Minimization**: Colocate frequently communicating components
3. **Load Balancing**: Distribute work evenly across available resources

## 10. Implementation Examples

### 10.1 Custom Worker Implementation

```python
class CustomActorWorker(BaseWorker):
    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def custom_operation(self, data: DataProto) -> DataProto:
        # Your custom implementation
        # Automatic handling of:
        # - Data distribution
        # - Remote execution
        # - Result collection
        pass
```

### 10.2 Resource Pool Configuration

```python
# Example: Separate pools for different model sizes
resource_pool_spec = {
    'large_model_pool': [8] * 8,   # 64 GPUs for 70B model
    'small_model_pool': [4] * 4,   # 16 GPUs for 7B model
    'reward_pool': [2] * 2,        # 4 GPUs for reward model
}
```

## Conclusion

VERL's distributed architecture represents a sophisticated approach to scaling RL training for LLMs. By separating control and computation flows, implementing efficient data movement patterns, and providing flexible resource management, VERL enables researchers and practitioners to train state-of-the-art models while maintaining code simplicity and reusability.

The framework's design choices reflect deep understanding of the challenges in distributed RL training and provide elegant solutions that balance performance, flexibility, and usability. As models continue to grow and RL algorithms become more complex, architectures like VERL's HybridFlow will become increasingly important for advancing the field.