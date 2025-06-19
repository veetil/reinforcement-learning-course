# Reward Model Training: Deep Research and Best Practices

## Executive Summary

Reward model training is the critical bridge between human preferences and AI behavior in RLHF systems. This research explores the theoretical foundations, practical implementations, advanced techniques, and common challenges in training effective reward models for language model alignment.

## Table of Contents
1. [Theoretical Foundations](#theoretical-foundations)
2. [Architecture Design](#architecture-design)
3. [Training Methodology](#training-methodology)
4. [Data Collection and Quality](#data-collection-and-quality)
5. [Advanced Techniques](#advanced-techniques)
6. [Evaluation Metrics](#evaluation-metrics)
7. [Common Challenges and Solutions](#common-challenges-and-solutions)
8. [Best Practices](#best-practices)

## 1. Theoretical Foundations

### 1.1 Preference Modeling Theory

#### Bradley-Terry Model
The Bradley-Terry model provides the mathematical foundation for learning from pairwise comparisons:

```python
# Probability that response A is preferred over response B
P(A > B) = exp(r(A)) / (exp(r(A)) + exp(r(B)))
         = sigmoid(r(A) - r(B))

where r(·) is the reward function
```

#### Plackett-Luce Model (for K-wise comparisons)
Extends to rankings over multiple options:

```python
P(ranking) = ∏_{i=1}^{K-1} exp(r(a_i)) / ∑_{j=i}^{K} exp(r(a_j))
```

#### Elo Rating System
Alternative approach based on chess ratings:

```python
# Expected score
E_A = 1 / (1 + 10^((R_B - R_A)/400))

# Rating update
R'_A = R_A + K(S_A - E_A)
```

### 1.2 Loss Functions

#### Standard Bradley-Terry Loss
```python
def bradley_terry_loss(r_chosen, r_rejected):
    """Binary cross-entropy formulation"""
    return -torch.log(torch.sigmoid(r_chosen - r_rejected))
```

#### Margin-based Loss
```python
def margin_loss(r_chosen, r_rejected, margin=0.1):
    """Ensures minimum separation between preferences"""
    return torch.relu(margin - (r_chosen - r_rejected))
```

#### Focal Loss for Imbalanced Data
```python
def focal_bradley_terry_loss(r_chosen, r_rejected, gamma=2.0):
    """Focuses on hard examples"""
    p = torch.sigmoid(r_chosen - r_rejected)
    return -(1 - p)**gamma * torch.log(p)
```

## 2. Architecture Design

### 2.1 Base Model Selection

#### Option 1: Shared Backbone with Policy
```python
class SharedBackboneRewardModel(nn.Module):
    def __init__(self, base_model, hidden_size):
        super().__init__()
        self.base_model = base_model  # Shared with policy
        self.reward_head = nn.Linear(hidden_size, 1)
        
    def forward(self, input_ids, attention_mask):
        outputs = self.base_model(input_ids, attention_mask)
        last_hidden = outputs.last_hidden_state
        
        # Get last non-padding token
        sequence_lengths = attention_mask.sum(dim=1) - 1
        last_tokens = last_hidden[torch.arange(len(last_hidden)), sequence_lengths]
        
        return self.reward_head(last_tokens)
```

#### Option 2: Separate Smaller Model
```python
class EfficientRewardModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Smaller transformer (e.g., 1B params vs 7B policy)
        self.transformer = AutoModel.from_pretrained(
            "microsoft/deberta-v3-base"
        )
        self.reward_head = RewardHead(self.transformer.config.hidden_size)
        
    def forward(self, input_ids, attention_mask):
        outputs = self.transformer(input_ids, attention_mask)
        return self.reward_head(outputs.last_hidden_state, attention_mask)
```

### 2.2 Reward Head Architectures

#### Simple Linear Head
```python
class LinearRewardHead(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.linear = nn.Linear(hidden_size, 1)
        
    def forward(self, hidden_states, attention_mask):
        # Extract last token
        lengths = attention_mask.sum(dim=1) - 1
        last_hidden = hidden_states[torch.arange(len(hidden_states)), lengths]
        return self.linear(last_hidden)
```

#### MLP Head with Dropout
```python
class MLPRewardHead(nn.Module):
    def __init__(self, hidden_size, intermediate_size=None, dropout=0.1):
        super().__init__()
        intermediate_size = intermediate_size or hidden_size * 2
        
        self.dense = nn.Linear(hidden_size, intermediate_size)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(intermediate_size, 1)
        
    def forward(self, hidden_states, attention_mask):
        lengths = attention_mask.sum(dim=1) - 1
        last_hidden = hidden_states[torch.arange(len(hidden_states)), lengths]
        
        h = self.dense(last_hidden)
        h = self.activation(h)
        h = self.dropout(h)
        return self.output(h)
```

#### Multi-Head Attention Pooling
```python
class AttentionPoolingRewardHead(nn.Module):
    def __init__(self, hidden_size, num_heads=8):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            hidden_size, num_heads, batch_first=True
        )
        self.query = nn.Parameter(torch.randn(1, 1, hidden_size))
        self.output = nn.Linear(hidden_size, 1)
        
    def forward(self, hidden_states, attention_mask):
        # Use learned query to attend over sequence
        batch_size = hidden_states.size(0)
        query = self.query.expand(batch_size, -1, -1)
        
        # Apply attention with mask
        attn_mask = ~attention_mask.bool()
        pooled, _ = self.attention(
            query, hidden_states, hidden_states,
            key_padding_mask=attn_mask
        )
        
        return self.output(pooled.squeeze(1))
```

## 3. Training Methodology

### 3.1 Data Preprocessing

#### Preference Pair Construction
```python
def construct_preference_pairs(dataset):
    pairs = []
    for sample in dataset:
        prompt = sample['prompt']
        chosen = sample['chosen']
        rejected = sample['rejected']
        
        # Tokenize with proper formatting
        chosen_text = format_conversation(prompt, chosen)
        rejected_text = format_conversation(prompt, rejected)
        
        chosen_tokens = tokenizer(chosen_text, truncation=True, max_length=512)
        rejected_tokens = tokenizer(rejected_text, truncation=True, max_length=512)
        
        pairs.append({
            'chosen_input_ids': chosen_tokens['input_ids'],
            'chosen_attention_mask': chosen_tokens['attention_mask'],
            'rejected_input_ids': rejected_tokens['input_ids'],
            'rejected_attention_mask': rejected_tokens['attention_mask'],
            'metadata': sample.get('metadata', {})
        })
    
    return pairs
```

#### Balanced Sampling
```python
class BalancedPreferenceSampler:
    def __init__(self, dataset, attributes=['length', 'domain', 'difficulty']):
        self.dataset = dataset
        self.attributes = attributes
        self.buckets = self._create_buckets()
        
    def _create_buckets(self):
        buckets = defaultdict(list)
        for idx, sample in enumerate(self.dataset):
            key = tuple(sample['metadata'].get(attr) for attr in self.attributes)
            buckets[key].append(idx)
        return buckets
    
    def sample_batch(self, batch_size):
        # Sample uniformly across buckets
        bucket_keys = list(self.buckets.keys())
        samples_per_bucket = batch_size // len(bucket_keys)
        
        indices = []
        for key in bucket_keys:
            bucket_indices = self.buckets[key]
            sampled = np.random.choice(
                bucket_indices, 
                size=min(samples_per_bucket, len(bucket_indices)),
                replace=False
            )
            indices.extend(sampled)
        
        return [self.dataset[i] for i in indices]
```

### 3.2 Training Loop Implementation

```python
class RewardModelTrainer:
    def __init__(self, model, optimizer, scheduler, config):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.config = config
        
        # Initialize metrics
        self.accuracy_tracker = AccuracyTracker()
        self.loss_tracker = ExponentialMovingAverage(alpha=0.99)
        
    def train_step(self, batch):
        self.model.train()
        
        # Forward pass for chosen responses
        chosen_rewards = self.model(
            input_ids=batch['chosen_input_ids'],
            attention_mask=batch['chosen_attention_mask']
        )
        
        # Forward pass for rejected responses  
        rejected_rewards = self.model(
            input_ids=batch['rejected_input_ids'],
            attention_mask=batch['rejected_attention_mask']
        )
        
        # Compute loss
        loss = self.compute_loss(chosen_rewards, rejected_rewards, batch)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        if self.config.gradient_clip_val > 0:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), 
                self.config.gradient_clip_val
            )
        
        # Optimizer step
        self.optimizer.step()
        self.scheduler.step()
        self.optimizer.zero_grad()
        
        # Update metrics
        accuracy = (chosen_rewards > rejected_rewards).float().mean()
        self.accuracy_tracker.update(accuracy)
        self.loss_tracker.update(loss.item())
        
        return {
            'loss': loss.item(),
            'accuracy': accuracy.item(),
            'chosen_reward_mean': chosen_rewards.mean().item(),
            'rejected_reward_mean': rejected_rewards.mean().item(),
            'reward_diff': (chosen_rewards - rejected_rewards).mean().item()
        }
    
    def compute_loss(self, chosen_rewards, rejected_rewards, batch):
        # Basic Bradley-Terry loss
        bt_loss = -torch.log(torch.sigmoid(chosen_rewards - rejected_rewards)).mean()
        
        # Optional: Add margin loss
        if self.config.use_margin_loss:
            margin_loss = torch.relu(
                self.config.margin - (chosen_rewards - rejected_rewards)
            ).mean()
            bt_loss = bt_loss + self.config.margin_weight * margin_loss
        
        # Optional: Add regularization
        if self.config.use_reward_regularization:
            # Penalize extreme rewards
            reg_loss = (
                chosen_rewards.pow(2).mean() + 
                rejected_rewards.pow(2).mean()
            ) * self.config.reg_weight
            bt_loss = bt_loss + reg_loss
        
        return bt_loss
```

### 3.3 Curriculum Learning

```python
class CurriculumRewardTrainer:
    def __init__(self, easy_data, medium_data, hard_data):
        self.datasets = {
            'easy': easy_data,
            'medium': medium_data,
            'hard': hard_data
        }
        self.current_difficulty = 'easy'
        self.performance_threshold = 0.85
        
    def get_batch(self, batch_size):
        # Sample from current difficulty
        dataset = self.datasets[self.current_difficulty]
        return dataset.sample(batch_size)
    
    def update_curriculum(self, accuracy):
        if accuracy > self.performance_threshold:
            if self.current_difficulty == 'easy':
                self.current_difficulty = 'medium'
                print("Advancing to medium difficulty")
            elif self.current_difficulty == 'medium':
                self.current_difficulty = 'hard'
                print("Advancing to hard difficulty")
```

## 4. Data Collection and Quality

### 4.1 Annotation Strategies

#### Pairwise Comparison
```python
class PairwiseAnnotationInterface:
    def __init__(self, prompt, response_a, response_b):
        self.prompt = prompt
        self.response_a = response_a
        self.response_b = response_b
        
    def display(self):
        print(f"Prompt: {self.prompt}")
        print(f"\nResponse A: {self.response_a}")
        print(f"\nResponse B: {self.response_b}")
        print("\nWhich response is better? (A/B/Tie)")
        
    def collect_annotation(self, annotator_id):
        choice = input().strip().upper()
        confidence = float(input("Confidence (0-1): "))
        
        return {
            'annotator_id': annotator_id,
            'choice': choice,
            'confidence': confidence,
            'timestamp': datetime.now()
        }
```

#### Likert Scale Rating
```python
class LikertRatingInterface:
    def __init__(self, prompt, response):
        self.prompt = prompt
        self.response = response
        self.criteria = [
            'helpfulness',
            'harmlessness', 
            'honesty',
            'clarity',
            'relevance'
        ]
        
    def collect_ratings(self, annotator_id):
        ratings = {}
        for criterion in self.criteria:
            rating = int(input(f"Rate {criterion} (1-5): "))
            ratings[criterion] = rating
            
        return {
            'annotator_id': annotator_id,
            'ratings': ratings,
            'overall': np.mean(list(ratings.values()))
        }
```

### 4.2 Quality Control

#### Inter-Annotator Agreement
```python
def compute_krippendorff_alpha(annotations):
    """
    Compute Krippendorff's alpha for inter-annotator agreement
    """
    # Create reliability data matrix
    # Each row is an annotator, each column is an item
    data_matrix = create_annotation_matrix(annotations)
    
    # Compute alpha
    alpha = krippendorff.alpha(
        reliability_data=data_matrix,
        level_of_measurement='ordinal'
    )
    
    return alpha

def filter_by_agreement(preference_pairs, min_agreement=0.7):
    """
    Filter pairs with high annotator agreement
    """
    filtered_pairs = []
    
    for pair in preference_pairs:
        annotations = pair['annotations']
        
        # Count votes
        votes = Counter([a['choice'] for a in annotations])
        total_votes = sum(votes.values())
        
        # Check if there's sufficient agreement
        max_votes = max(votes.values())
        agreement = max_votes / total_votes
        
        if agreement >= min_agreement:
            # Use majority vote
            pair['chosen'] = pair['response_a'] if votes['A'] > votes['B'] else pair['response_b']
            pair['rejected'] = pair['response_b'] if votes['A'] > votes['B'] else pair['response_a']
            pair['agreement_score'] = agreement
            filtered_pairs.append(pair)
    
    return filtered_pairs
```

#### Annotator Calibration
```python
class AnnotatorCalibration:
    def __init__(self, gold_standard_pairs):
        self.gold_standard = gold_standard_pairs
        self.annotator_scores = defaultdict(list)
        
    def evaluate_annotator(self, annotator_id, annotations):
        correct = 0
        total = 0
        
        for annotation in annotations:
            pair_id = annotation['pair_id']
            if pair_id in self.gold_standard:
                gold_choice = self.gold_standard[pair_id]
                if annotation['choice'] == gold_choice:
                    correct += 1
                total += 1
        
        accuracy = correct / total if total > 0 else 0
        self.annotator_scores[annotator_id].append(accuracy)
        
        return accuracy
    
    def get_reliable_annotators(self, min_accuracy=0.8):
        reliable = []
        for annotator_id, scores in self.annotator_scores.items():
            avg_accuracy = np.mean(scores)
            if avg_accuracy >= min_accuracy:
                reliable.append(annotator_id)
        return reliable
```

## 5. Advanced Techniques

### 5.1 Ensemble Reward Models

```python
class EnsembleRewardModel(nn.Module):
    def __init__(self, model_configs, aggregation='mean'):
        super().__init__()
        self.models = nn.ModuleList([
            create_reward_model(config) for config in model_configs
        ])
        self.aggregation = aggregation
        
        if aggregation == 'learned':
            self.aggregation_weights = nn.Parameter(
                torch.ones(len(self.models)) / len(self.models)
            )
            
    def forward(self, input_ids, attention_mask):
        rewards = []
        
        for model in self.models:
            reward = model(input_ids, attention_mask)
            rewards.append(reward)
            
        rewards = torch.stack(rewards, dim=1)  # [batch, n_models]
        
        if self.aggregation == 'mean':
            return rewards.mean(dim=1)
        elif self.aggregation == 'median':
            return rewards.median(dim=1).values
        elif self.aggregation == 'learned':
            weights = torch.softmax(self.aggregation_weights, dim=0)
            return (rewards * weights).sum(dim=1)
        else:
            raise ValueError(f"Unknown aggregation: {self.aggregation}")
    
    def compute_uncertainty(self, input_ids, attention_mask):
        with torch.no_grad():
            rewards = []
            for model in self.models:
                reward = model(input_ids, attention_mask)
                rewards.append(reward)
                
            rewards = torch.stack(rewards, dim=1)
            mean_reward = rewards.mean(dim=1)
            std_reward = rewards.std(dim=1)
            
        return mean_reward, std_reward
```

### 5.2 Uncertainty-Aware Training

```python
class UncertaintyAwareRewardModel(nn.Module):
    def __init__(self, base_model, hidden_size, num_dropout_samples=10):
        super().__init__()
        self.base_model = base_model
        self.num_dropout_samples = num_dropout_samples
        
        # Dual head for mean and log variance
        self.mean_head = nn.Linear(hidden_size, 1)
        self.log_var_head = nn.Linear(hidden_size, 1)
        
        # MC Dropout layers
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, input_ids, attention_mask, return_uncertainty=False):
        outputs = self.base_model(input_ids, attention_mask)
        hidden = self.extract_features(outputs, attention_mask)
        
        # Apply dropout
        hidden = self.dropout(hidden)
        
        # Predict mean and variance
        mean = self.mean_head(hidden)
        log_var = self.log_var_head(hidden)
        
        if return_uncertainty:
            # Monte Carlo sampling
            samples = []
            for _ in range(self.num_dropout_samples):
                h = self.dropout(hidden)
                sample = self.mean_head(h)
                samples.append(sample)
            
            samples = torch.stack(samples, dim=1)
            epistemic_uncertainty = samples.var(dim=1)
            aleatoric_uncertainty = torch.exp(log_var)
            
            return mean, epistemic_uncertainty, aleatoric_uncertainty
        
        return mean
```

### 5.3 Contrastive Learning

```python
class ContrastiveRewardModel(nn.Module):
    def __init__(self, base_model, hidden_size, temperature=0.07):
        super().__init__()
        self.base_model = base_model
        self.projection = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 128)  # Project to smaller space
        )
        self.reward_head = nn.Linear(128, 1)
        self.temperature = temperature
        
    def forward(self, chosen_ids, rejected_ids, chosen_mask, rejected_mask):
        # Encode both responses
        chosen_hidden = self.encode(chosen_ids, chosen_mask)
        rejected_hidden = self.encode(rejected_ids, rejected_mask)
        
        # Contrastive loss in projection space
        chosen_proj = self.projection(chosen_hidden)
        rejected_proj = self.projection(rejected_hidden)
        
        # Normalize
        chosen_proj = F.normalize(chosen_proj, dim=1)
        rejected_proj = F.normalize(rejected_proj, dim=1)
        
        # Compute similarity
        similarity = torch.matmul(chosen_proj, rejected_proj.T) / self.temperature
        
        # Contrastive loss (chosen should be far from rejected)
        labels = torch.arange(len(chosen_proj), device=chosen_proj.device)
        contrastive_loss = F.cross_entropy(similarity, labels)
        
        # Compute rewards
        chosen_rewards = self.reward_head(chosen_proj)
        rejected_rewards = self.reward_head(rejected_proj)
        
        return chosen_rewards, rejected_rewards, contrastive_loss
```

## 6. Evaluation Metrics

### 6.1 Accuracy Metrics

```python
class RewardModelEvaluator:
    def __init__(self, model, test_dataset):
        self.model = model
        self.test_dataset = test_dataset
        
    def evaluate(self):
        self.model.eval()
        metrics = defaultdict(list)
        
        with torch.no_grad():
            for batch in self.test_dataset:
                chosen_rewards = self.model(
                    batch['chosen_input_ids'],
                    batch['chosen_attention_mask']
                )
                rejected_rewards = self.model(
                    batch['rejected_input_ids'],
                    batch['rejected_attention_mask']
                )
                
                # Basic accuracy
                correct = (chosen_rewards > rejected_rewards).float()
                metrics['accuracy'].append(correct.mean().item())
                
                # Margin
                margin = (chosen_rewards - rejected_rewards)
                metrics['mean_margin'].append(margin.mean().item())
                metrics['min_margin'].append(margin.min().item())
                
                # Confidence calibration
                probs = torch.sigmoid(margin)
                metrics['mean_confidence'].append(probs.mean().item())
                
        # Aggregate metrics
        results = {}
        for key, values in metrics.items():
            results[key] = np.mean(values)
            results[f'{key}_std'] = np.std(values)
            
        return results
```

### 6.2 Correlation Metrics

```python
def compute_human_correlation(model_rewards, human_ratings):
    """
    Compute correlation between model and human preferences
    """
    # Pearson correlation
    pearson_r, pearson_p = scipy.stats.pearsonr(
        model_rewards, human_ratings
    )
    
    # Spearman rank correlation
    spearman_r, spearman_p = scipy.stats.spearmanr(
        model_rewards, human_ratings
    )
    
    # Kendall's tau
    kendall_tau, kendall_p = scipy.stats.kendalltau(
        model_rewards, human_ratings
    )
    
    return {
        'pearson_r': pearson_r,
        'pearson_p': pearson_p,
        'spearman_r': spearman_r,
        'spearman_p': spearman_p,
        'kendall_tau': kendall_tau,
        'kendall_p': kendall_p
    }
```

### 6.3 Robustness Metrics

```python
class RobustnessEvaluator:
    def __init__(self, model, perturbation_types=['typo', 'paraphrase', 'truncate']):
        self.model = model
        self.perturbation_types = perturbation_types
        
    def evaluate_robustness(self, test_pairs):
        results = defaultdict(list)
        
        for pair in test_pairs:
            original_chosen = pair['chosen']
            original_rejected = pair['rejected']
            
            # Get original rewards
            orig_chosen_reward = self.get_reward(pair['prompt'], original_chosen)
            orig_rejected_reward = self.get_reward(pair['prompt'], original_rejected)
            orig_correct = orig_chosen_reward > orig_rejected_reward
            
            # Test perturbations
            for perturb_type in self.perturbation_types:
                perturbed_chosen = self.perturb(original_chosen, perturb_type)
                perturbed_rejected = self.perturb(original_rejected, perturb_type)
                
                pert_chosen_reward = self.get_reward(pair['prompt'], perturbed_chosen)
                pert_rejected_reward = self.get_reward(pair['prompt'], perturbed_rejected)
                pert_correct = pert_chosen_reward > pert_rejected_reward
                
                # Check consistency
                consistent = orig_correct == pert_correct
                results[f'{perturb_type}_consistency'].append(consistent)
                
                # Check reward stability
                chosen_diff = abs(orig_chosen_reward - pert_chosen_reward)
                rejected_diff = abs(orig_rejected_reward - pert_rejected_reward)
                results[f'{perturb_type}_chosen_stability'].append(chosen_diff)
                results[f'{perturb_type}_rejected_stability'].append(rejected_diff)
        
        # Aggregate results
        summary = {}
        for key, values in results.items():
            if 'consistency' in key:
                summary[key] = np.mean(values)
            else:
                summary[f'{key}_mean'] = np.mean(values)
                summary[f'{key}_max'] = np.max(values)
                
        return summary
```

## 7. Common Challenges and Solutions

### 7.1 Reward Hacking Prevention

```python
class RewardHackingDetector:
    def __init__(self, reward_model, reference_model):
        self.reward_model = reward_model
        self.reference_model = reference_model
        self.anomaly_threshold = 3.0  # Standard deviations
        
    def detect_hacking(self, responses):
        # Get rewards
        rewards = self.reward_model(responses)
        
        # Compute perplexity under reference model
        with torch.no_grad():
            log_probs = self.reference_model.compute_log_probs(responses)
            perplexities = torch.exp(-log_probs.mean(dim=1))
        
        # Detect anomalies
        reward_zscore = (rewards - rewards.mean()) / rewards.std()
        perplexity_zscore = (perplexities - perplexities.mean()) / perplexities.std()
        
        # High reward + high perplexity = potential hacking
        hacking_score = reward_zscore + perplexity_zscore
        potentially_hacked = hacking_score > self.anomaly_threshold
        
        return potentially_hacked, hacking_score
```

### 7.2 Distribution Shift Handling

```python
class DistributionShiftAdapter:
    def __init__(self, initial_reward_model):
        self.reward_model = initial_reward_model
        self.distribution_monitor = DistributionMonitor()
        self.adaptation_buffer = []
        self.adaptation_threshold = 100
        
    def score_with_adaptation(self, prompt, response):
        # Get base reward
        reward = self.reward_model(prompt, response)
        
        # Monitor distribution
        shift_detected = self.distribution_monitor.check_shift(prompt, response)
        
        if shift_detected:
            # Add to adaptation buffer
            self.adaptation_buffer.append({
                'prompt': prompt,
                'response': response,
                'initial_reward': reward
            })
            
            # Trigger adaptation if buffer is full
            if len(self.adaptation_buffer) >= self.adaptation_threshold:
                self.adapt_model()
                
        return reward
    
    def adapt_model(self):
        # Collect human feedback on buffer
        annotations = self.collect_annotations(self.adaptation_buffer)
        
        # Fine-tune on new data
        self.reward_model = self.fine_tune(
            self.reward_model,
            annotations,
            epochs=5
        )
        
        # Clear buffer
        self.adaptation_buffer = []
```

### 7.3 Multi-Objective Balancing

```python
class MultiObjectiveRewardModel(nn.Module):
    def __init__(self, base_model, objectives=['helpfulness', 'harmlessness', 'honesty']):
        super().__init__()
        self.base_model = base_model
        self.objectives = objectives
        
        # Separate heads for each objective
        hidden_size = base_model.config.hidden_size
        self.objective_heads = nn.ModuleDict({
            obj: nn.Linear(hidden_size, 1) for obj in objectives
        })
        
        # Learnable weights for combining objectives
        self.objective_weights = nn.Parameter(
            torch.ones(len(objectives)) / len(objectives)
        )
        
    def forward(self, input_ids, attention_mask, return_all_objectives=False):
        # Get base representations
        outputs = self.base_model(input_ids, attention_mask)
        hidden = self.extract_features(outputs, attention_mask)
        
        # Compute objective-specific rewards
        objective_rewards = {}
        for obj, head in self.objective_heads.items():
            objective_rewards[obj] = head(hidden)
        
        if return_all_objectives:
            return objective_rewards
        
        # Combine objectives
        weights = torch.softmax(self.objective_weights, dim=0)
        combined_reward = sum(
            weights[i] * objective_rewards[obj]
            for i, obj in enumerate(self.objectives)
        )
        
        return combined_reward
```

## 8. Best Practices

### 8.1 Training Recipe

```python
# Recommended hyperparameters
REWARD_MODEL_CONFIG = {
    'learning_rate': 1e-5,
    'warmup_steps': 100,
    'batch_size': 32,
    'gradient_accumulation_steps': 4,
    'max_epochs': 3,
    'weight_decay': 0.01,
    'gradient_clip': 1.0,
    'label_smoothing': 0.1,
    'dropout': 0.1,
    'margin': 0.1,
    'ensemble_size': 3
}

# Training schedule
SCHEDULE = [
    {'phase': 'warmup', 'epochs': 0.5, 'lr_multiplier': 0.1},
    {'phase': 'main', 'epochs': 2.0, 'lr_multiplier': 1.0},
    {'phase': 'fine_tune', 'epochs': 0.5, 'lr_multiplier': 0.1}
]
```

### 8.2 Data Quality Checklist

```python
class DataQualityChecker:
    def __init__(self):
        self.checks = [
            self.check_length_distribution,
            self.check_preference_consistency,
            self.check_annotation_quality,
            self.check_domain_coverage,
            self.check_difficulty_balance
        ]
    
    def validate_dataset(self, dataset):
        report = {}
        
        for check in self.checks:
            check_name = check.__name__
            passed, details = check(dataset)
            report[check_name] = {
                'passed': passed,
                'details': details
            }
            
        return report
    
    def check_length_distribution(self, dataset):
        # Ensure balanced length distribution
        lengths = [len(d['chosen']) + len(d['rejected']) for d in dataset]
        cv = np.std(lengths) / np.mean(lengths)
        
        passed = cv < 0.5  # Coefficient of variation < 0.5
        details = {
            'mean_length': np.mean(lengths),
            'std_length': np.std(lengths),
            'cv': cv
        }
        
        return passed, details
```

### 8.3 Deployment Considerations

```python
class ProductionRewardModel:
    def __init__(self, model_path, device='cuda'):
        self.device = device
        self.model = self.load_model(model_path)
        self.cache = LRUCache(maxsize=10000)
        self.batch_queue = []
        self.batch_size = 32
        
    def score_async(self, prompt, response):
        # Check cache
        cache_key = hash((prompt, response))
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        # Add to batch queue
        future = asyncio.Future()
        self.batch_queue.append({
            'prompt': prompt,
            'response': response,
            'future': future,
            'cache_key': cache_key
        })
        
        # Process batch if full
        if len(self.batch_queue) >= self.batch_size:
            asyncio.create_task(self.process_batch())
            
        return future
    
    async def process_batch(self):
        batch = self.batch_queue[:self.batch_size]
        self.batch_queue = self.batch_queue[self.batch_size:]
        
        # Batch inference
        prompts = [b['prompt'] for b in batch]
        responses = [b['response'] for b in batch]
        rewards = self.model.score_batch(prompts, responses)
        
        # Update cache and resolve futures
        for item, reward in zip(batch, rewards):
            self.cache[item['cache_key']] = reward
            item['future'].set_result(reward)
```

## Conclusion

Reward model training is a critical component of RLHF that requires careful attention to:

1. **Architecture Design**: Choose appropriate model size and head design
2. **Data Quality**: Ensure high-quality preference annotations
3. **Training Methodology**: Use proper loss functions and optimization
4. **Evaluation**: Comprehensive metrics beyond simple accuracy
5. **Robustness**: Handle distribution shift and prevent reward hacking
6. **Production**: Efficient deployment with caching and batching

Key takeaways:
- Bradley-Terry loss is the foundation, but enhancements improve performance
- Ensemble methods provide uncertainty estimates and robustness
- Data quality is paramount - invest in annotation quality control
- Monitor for reward hacking and distribution shift
- Multi-objective balancing prevents over-optimization