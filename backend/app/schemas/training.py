from pydantic import BaseModel
from typing import Dict, List, Optional
from datetime import datetime

class TrainingConfig(BaseModel):
    environment: str
    algorithm: str = "PPO"
    total_steps: int = 10000
    learning_rate: float = 0.0003
    batch_size: int = 64
    n_epochs: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01

class TrainingStatus(BaseModel):
    session_id: str
    status: str  # initializing, running, completed, failed, stopped
    environment: str
    algorithm: str
    current_step: int
    total_steps: int
    created_at: datetime = datetime.utcnow()
    updated_at: datetime = datetime.utcnow()

class TrainingMetrics(BaseModel):
    step: int
    episode_reward: float
    episode_length: int
    value_loss: float
    policy_loss: float
    entropy: float
    kl_divergence: float
    learning_rate: float
    explained_variance: float

class TrainingResult(BaseModel):
    session_id: str
    status: str
    total_steps: int
    training_time: float
    final_reward: float
    best_reward: float
    metrics_history: List[TrainingMetrics]