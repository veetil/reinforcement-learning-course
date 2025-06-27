from pydantic import BaseModel
from typing import Dict, List, Optional
from datetime import datetime

class HyperParameters(BaseModel):
    learningRate: float = 0.0003
    clipRange: float = 0.2
    gamma: float = 0.99
    gaeBalance: float = 0.95
    nEpochs: int = 10
    batchSize: int = 64
    entropyCoef: float = 0.01
    valueCoef: float = 0.5

class TrainingConfig(BaseModel):
    environment: str
    algorithm: str = "PPO"
    hyperparameters: HyperParameters

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