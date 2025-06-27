from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/api/v1/training/start")
async def start_training(config: dict):
    return {"id": "test-session-123"}

@app.get("/api/v1/training/status/{session_id}")
async def get_status(session_id: str):
    return {
        "id": session_id,
        "status": "running",
        "progress": 25,
        "currentEpisode": 250,
        "totalEpisodes": 1000,
        "metrics": {
            "meanReward": 150.5,
            "meanEpisodeLength": 200,
            "policyLoss": 0.25,
            "valueLoss": 0.15,
            "entropy": 1.2,
            "klDivergence": 0.012,
            "clipFraction": 0.2
        }
    }

@app.post("/api/v1/training/stop/{session_id}")
async def stop_training(session_id: str):
    return {"message": "Training stopped successfully"}

@app.get("/api/v1/training/environments")
async def get_environments():
    return ["CartPole-v1", "LunarLander-v2", "MountainCar-v0"]

@app.get("/api/v1/training/algorithms")
async def get_algorithms():
    return ["PPO", "A2C", "SAC"]

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)