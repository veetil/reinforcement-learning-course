from fastapi import FastAPI, HTTPException, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List, Optional
import asyncio
import uuid
import numpy as np
from datetime import datetime
import json

app = FastAPI(title="PPO Course API")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Models
class Hyperparameters(BaseModel):
    learningRate: float
    clipRange: float
    gamma: float
    gaeBalance: float
    nEpochs: int
    batchSize: int
    entropyCoef: float
    valueCoef: float

class TrainingConfig(BaseModel):
    environment: str
    algorithm: str
    hyperparameters: Hyperparameters

class TrainingStatus(BaseModel):
    id: str
    status: str  # 'pending', 'running', 'completed', 'failed'
    progress: float
    currentEpisode: int
    totalEpisodes: int
    metrics: Dict[str, float]

class CodeExecutionRequest(BaseModel):
    code: str
    language: str = "python"
    timeout: Optional[int] = 30000

# In-memory storage (replace with database in production)
training_sessions: Dict[str, TrainingStatus] = {}
active_connections: Dict[str, List[WebSocket]] = {}

# Mock training simulator
class TrainingSimulator:
    def __init__(self, config: TrainingConfig, training_id: str):
        self.config = config
        self.training_id = training_id
        self.current_episode = 0
        self.total_episodes = 1000
        self.is_running = True
        
    async def run(self):
        """Simulate training process"""
        while self.is_running and self.current_episode < self.total_episodes:
            # Simulate one training step
            await asyncio.sleep(0.1)  # 100ms per step
            
            self.current_episode += 1
            progress = (self.current_episode / self.total_episodes) * 100
            
            # Generate mock metrics
            metrics = {
                "meanReward": 50 + self.current_episode * 0.1 + np.random.randn() * 10,
                "meanEpisodeLength": 200 + np.random.randn() * 20,
                "policyLoss": 0.5 * np.exp(-self.current_episode / 100) + np.random.randn() * 0.01,
                "valueLoss": 1.0 * np.exp(-self.current_episode / 100) + np.random.randn() * 0.02,
                "entropy": 0.5 + np.random.randn() * 0.05,
                "klDivergence": 0.01 + np.random.randn() * 0.005,
                "clipFraction": 0.2 + np.random.randn() * 0.05
            }
            
            # Update status
            status = TrainingStatus(
                id=self.training_id,
                status="running",
                progress=progress,
                currentEpisode=self.current_episode,
                totalEpisodes=self.total_episodes,
                metrics=metrics
            )
            
            training_sessions[self.training_id] = status
            
            # Send updates to connected clients
            await self.broadcast_update(status)
            
        # Mark as completed
        if self.training_id in training_sessions:
            training_sessions[self.training_id].status = "completed"
    
    async def broadcast_update(self, status: TrainingStatus):
        """Send status update to all connected WebSocket clients"""
        if self.training_id in active_connections:
            message = json.dumps(status.dict())
            dead_connections = []
            
            for websocket in active_connections[self.training_id]:
                try:
                    await websocket.send_text(message)
                except:
                    dead_connections.append(websocket)
            
            # Remove dead connections
            for ws in dead_connections:
                active_connections[self.training_id].remove(ws)
    
    def stop(self):
        self.is_running = False

# Active training tasks
training_tasks: Dict[str, asyncio.Task] = {}

@app.get("/")
async def root():
    return {"message": "PPO Course API", "version": "1.0.0"}

@app.post("/api/training/start")
async def start_training(config: TrainingConfig):
    """Start a new training session"""
    training_id = str(uuid.uuid4())
    
    # Initialize status
    status = TrainingStatus(
        id=training_id,
        status="pending",
        progress=0,
        currentEpisode=0,
        totalEpisodes=1000,
        metrics={
            "meanReward": 0,
            "meanEpisodeLength": 0,
            "policyLoss": 0,
            "valueLoss": 0,
            "entropy": 0,
            "klDivergence": 0,
            "clipFraction": 0
        }
    )
    
    training_sessions[training_id] = status
    
    # Start training in background
    simulator = TrainingSimulator(config, training_id)
    task = asyncio.create_task(simulator.run())
    training_tasks[training_id] = task
    
    return {"id": training_id}

@app.get("/api/training/status/{training_id}")
async def get_training_status(training_id: str):
    """Get current training status"""
    if training_id not in training_sessions:
        raise HTTPException(status_code=404, detail="Training session not found")
    
    return training_sessions[training_id]

@app.post("/api/training/stop/{training_id}")
async def stop_training(training_id: str):
    """Stop a training session"""
    if training_id not in training_sessions:
        raise HTTPException(status_code=404, detail="Training session not found")
    
    # Cancel the training task
    if training_id in training_tasks:
        training_tasks[training_id].cancel()
        del training_tasks[training_id]
    
    # Update status
    training_sessions[training_id].status = "stopped"
    
    return {"message": "Training stopped"}

@app.post("/api/code/execute")
async def execute_code(request: CodeExecutionRequest):
    """Execute code in a sandboxed environment"""
    # This is a mock implementation
    # In production, use a proper sandboxed execution environment
    
    if request.language != "python":
        raise HTTPException(status_code=400, detail="Only Python is supported")
    
    # Mock execution
    output = f"# Mock execution of:\n{request.code}\n\n# Output:\nHello from PPO Course!"
    
    return {
        "output": output,
        "error": None,
        "executionTime": 0.123
    }

@app.get("/api/environments")
async def get_environments():
    """Get list of available environments"""
    return [
        "CartPole-v1",
        "MountainCar-v0",
        "LunarLander-v2",
        "BipedalWalker-v3",
        "HalfCheetah-v4"
    ]

@app.get("/api/algorithms")
async def get_algorithms():
    """Get list of available algorithms"""
    return [
        "PPO",
        "A2C",
        "DQN",
        "SAC",
        "TD3"
    ]

@app.websocket("/ws/training/{training_id}")
async def websocket_endpoint(websocket: WebSocket, training_id: str):
    """WebSocket endpoint for real-time training updates"""
    await websocket.accept()
    
    # Add to active connections
    if training_id not in active_connections:
        active_connections[training_id] = []
    active_connections[training_id].append(websocket)
    
    try:
        # Keep connection alive
        while True:
            # Wait for messages (ping/pong)
            await websocket.receive_text()
    except:
        # Remove from active connections
        if training_id in active_connections:
            active_connections[training_id].remove(websocket)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)