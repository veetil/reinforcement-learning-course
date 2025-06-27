"""Mock RL Engine for demo purposes"""
import asyncio
import uuid
from typing import Dict, Any
import random
import time

class MockRLEngine:
    def __init__(self):
        self.environments = {}
        self.training_sessions = {}
    
    async def create_environment(self, env_name: str) -> str:
        """Create a mock environment"""
        env_id = str(uuid.uuid4())
        self.environments[env_id] = {
            "name": env_name,
            "state": "initialized",
            "observation_space": {"shape": [4] if "CartPole" in env_name else [8]},
            "action_space": {"n": 2 if "CartPole" in env_name else 4}
        }
        return env_id
    
    async def start_training(self, config: Dict[str, Any]) -> str:
        """Start mock training session"""
        session_id = str(uuid.uuid4())
        
        # Create environment
        env_id = await self.create_environment(config.get("environment", "CartPole-v1"))
        
        self.training_sessions[session_id] = {
            "id": session_id,
            "status": "running",
            "environment": config.get("environment", "CartPole-v1"),
            "algorithm": config.get("algorithm", "PPO"),
            "config": config,
            "env_id": env_id,
            "start_time": time.time(),
            "current_episode": 0,
            "total_episodes": 1000,
            "metrics": self._generate_initial_metrics()
        }
        
        # Start background simulation
        asyncio.create_task(self._simulate_training(session_id))
        
        return session_id
    
    def _generate_initial_metrics(self) -> Dict[str, float]:
        """Generate initial training metrics"""
        return {
            "meanReward": random.uniform(-50, 50),
            "meanEpisodeLength": random.uniform(50, 200),
            "policyLoss": random.uniform(0.1, 1.0),
            "valueLoss": random.uniform(0.1, 1.0),
            "entropy": random.uniform(0.5, 2.0),
            "klDivergence": random.uniform(0.001, 0.02),
            "clipFraction": random.uniform(0.1, 0.3)
        }
    
    async def _simulate_training(self, session_id: str):
        """Simulate training progress"""
        session = self.training_sessions.get(session_id)
        if not session:
            return
        
        while session["status"] == "running" and session["current_episode"] < session["total_episodes"]:
            await asyncio.sleep(2)  # Update every 2 seconds
            
            # Update episode count
            session["current_episode"] += random.randint(1, 5)
            session["progress"] = min(100, (session["current_episode"] / session["total_episodes"]) * 100)
            
            # Update metrics with some improvement trend
            episode_factor = session["current_episode"] / session["total_episodes"]
            
            session["metrics"] = {
                "meanReward": session["metrics"]["meanReward"] + random.uniform(-5, 10) * (1 + episode_factor),
                "meanEpisodeLength": session["metrics"]["meanEpisodeLength"] + random.uniform(-10, 20),
                "policyLoss": max(0.01, session["metrics"]["policyLoss"] * random.uniform(0.98, 1.02)),
                "valueLoss": max(0.01, session["metrics"]["valueLoss"] * random.uniform(0.98, 1.02)),
                "entropy": max(0.1, session["metrics"]["entropy"] * random.uniform(0.995, 1.005)),
                "klDivergence": session["metrics"]["klDivergence"] * random.uniform(0.95, 1.05),
                "clipFraction": min(1.0, max(0.01, session["metrics"]["clipFraction"] + random.uniform(-0.02, 0.02)))
            }
            
            # Cap metrics at reasonable bounds
            session["metrics"]["meanReward"] = min(500, session["metrics"]["meanReward"])
            session["metrics"]["meanEpisodeLength"] = min(1000, max(10, session["metrics"]["meanEpisodeLength"]))
            session["metrics"]["klDivergence"] = min(0.05, max(0.001, session["metrics"]["klDivergence"]))
        
        # Mark as completed if finished
        if session["current_episode"] >= session["total_episodes"]:
            session["status"] = "completed"
    
    def get_training_status(self, session_id: str) -> Dict[str, Any]:
        """Get training session status"""
        session = self.training_sessions.get(session_id)
        if not session:
            return None
        
        return {
            "id": session_id,
            "status": session["status"],
            "progress": session.get("progress", 0),
            "currentEpisode": session["current_episode"],
            "totalEpisodes": session["total_episodes"],
            "metrics": session["metrics"]
        }
    
    def stop_training(self, session_id: str) -> bool:
        """Stop training session"""
        if session_id in self.training_sessions:
            self.training_sessions[session_id]["status"] = "stopped"
            return True
        return False
    
    def get_available_environments(self) -> list:
        """Get list of available environments"""
        return [
            "CartPole-v1",
            "LunarLander-v2", 
            "MountainCar-v0",
            "Acrobot-v1",
            "Pendulum-v1"
        ]
    
    def get_available_algorithms(self) -> list:
        """Get list of available algorithms"""
        return ["PPO", "A2C", "SAC", "TD3", "DDPG"]