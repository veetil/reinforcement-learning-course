from fastapi import APIRouter, HTTPException, Depends
from typing import Dict, List
import uuid
from app.schemas.training import TrainingConfig, TrainingStatus, TrainingResult
from app.services.rl_engine import RLEngine

router = APIRouter()

# In-memory storage for demo (use Redis in production)
training_sessions: Dict[str, TrainingStatus] = {}
rl_engine = RLEngine()

@router.post("/start", response_model=TrainingStatus)
async def start_training(config: TrainingConfig):
    """Start a new PPO training session"""
    session_id = str(uuid.uuid4())
    
    try:
        # Initialize training
        env_id = await rl_engine.create_environment(config.environment)
        
        training_sessions[session_id] = TrainingStatus(
            session_id=session_id,
            status="initializing",
            environment=config.environment,
            algorithm="PPO",
            current_step=0,
            total_steps=config.total_steps
        )
        
        # Start async training
        # In production, this would be handled by Celery or similar
        # For now, we'll simulate it
        training_sessions[session_id].status = "running"
        
        return training_sessions[session_id]
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/status/{session_id}", response_model=TrainingStatus)
async def get_training_status(session_id: str):
    """Get the status of a training session"""
    if session_id not in training_sessions:
        raise HTTPException(status_code=404, detail="Training session not found")
    
    return training_sessions[session_id]

@router.post("/stop/{session_id}")
async def stop_training(session_id: str):
    """Stop a training session"""
    if session_id not in training_sessions:
        raise HTTPException(status_code=404, detail="Training session not found")
    
    training_sessions[session_id].status = "stopped"
    return {"message": "Training stopped successfully"}

@router.get("/metrics/{session_id}")
async def get_training_metrics(session_id: str):
    """Get training metrics for a session"""
    if session_id not in training_sessions:
        raise HTTPException(status_code=404, detail="Training session not found")
    
    # Simulate some metrics
    return {
        "session_id": session_id,
        "metrics": {
            "episode_reward": [10, 15, 20, 25, 30],
            "loss": [0.5, 0.4, 0.35, 0.3, 0.28],
            "kl_divergence": [0.01, 0.012, 0.011, 0.013, 0.012],
            "entropy": [1.0, 0.9, 0.85, 0.8, 0.78]
        }
    }