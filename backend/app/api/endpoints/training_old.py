from fastapi import APIRouter, HTTPException, Depends
from typing import Dict, List
import uuid
from app.schemas.training import TrainingConfig
from app.services.rl_engine_mock import MockRLEngine

router = APIRouter()

# In-memory storage for demo (use Redis in production)
training_sessions: Dict[str, TrainingStatus] = {}
rl_engine = MockRLEngine()

@router.post("/start")
async def start_training(config: TrainingConfig):
    """Start a new PPO training session"""
    try:
        # Start training with mock engine
        session_id = await rl_engine.start_training(config.dict())
        
        return {"id": session_id}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/status/{session_id}")
async def get_training_status(session_id: str):
    """Get the status of a training session"""
    status = rl_engine.get_training_status(session_id)
    if not status:
        raise HTTPException(status_code=404, detail="Training session not found")
    
    return status

@router.post("/stop/{session_id}")
async def stop_training(session_id: str):
    """Stop a training session"""
    success = rl_engine.stop_training(session_id)
    if not success:
        raise HTTPException(status_code=404, detail="Training session not found")
    
    return {"message": "Training stopped successfully"}

@router.get("/environments")
async def get_environments():
    """Get available environments"""
    return rl_engine.get_available_environments()

@router.get("/algorithms")
async def get_algorithms():
    """Get available algorithms"""
    return rl_engine.get_available_algorithms()

@router.get("/metrics/{session_id}")
async def get_training_metrics(session_id: str):
    """Get training metrics for a session"""
    status = rl_engine.get_training_status(session_id)
    if not status:
        raise HTTPException(status_code=404, detail="Training session not found")
    
    return {
        "session_id": session_id,
        "metrics": status.get("metrics", {})
    }