import asyncio
from typing import Dict, Optional, Any
import uuid
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback

class TrainingCallback(BaseCallback):
    """Custom callback for tracking training metrics"""
    def __init__(self, callback_fn=None):
        super().__init__()
        self.callback_fn = callback_fn
        self.metrics = []
        
    def _on_step(self) -> bool:
        # Collect metrics
        if self.n_calls % 100 == 0:  # Every 100 steps
            metrics = {
                'step': self.n_calls,
                'episode_reward': self.locals.get('rewards', [0])[0],
                'value_loss': self.logger.name_to_value.get('train/value_loss', 0),
                'policy_loss': self.logger.name_to_value.get('train/policy_gradient_loss', 0),
                'entropy': self.logger.name_to_value.get('train/entropy_loss', 0),
                'kl_divergence': self.logger.name_to_value.get('train/approx_kl', 0),
                'learning_rate': self.logger.name_to_value.get('train/learning_rate', 0)
            }
            self.metrics.append(metrics)
            
            # Call async callback if provided
            if self.callback_fn:
                asyncio.create_task(self.callback_fn(metrics))
                
        return True

class RLEngine:
    def __init__(self):
        self.environments: Dict[str, Any] = {}
        self.agents: Dict[str, PPO] = {}
        self.training_sessions: Dict[str, Dict] = {}
        
    def _generate_id(self) -> str:
        return str(uuid.uuid4())
        
    async def create_environment(self, env_name: str) -> str:
        """Create and register a new environment"""
        env_id = self._generate_id()
        
        try:
            # Create environment
            if env_name == "GridWorld":
                # Custom GridWorld environment
                env = self._create_grid_world()
            else:
                # Use Gymnasium environment
                env = gym.make(env_name)
                
            self.environments[env_id] = env
            return env_id
            
        except Exception as e:
            raise Exception(f"Failed to create environment: {str(e)}")
            
    def _create_grid_world(self):
        """Create a simple grid world environment"""
        # For demo purposes, we'll use CartPole as a placeholder
        # In production, implement custom GridWorld
        return gym.make("CartPole-v1")
        
    async def create_agent(self, env_id: str, config: Dict[str, Any]) -> str:
        """Create a PPO agent for the environment"""
        if env_id not in self.environments:
            raise ValueError(f"Environment {env_id} not found")
            
        agent_id = self._generate_id()
        env = self.environments[env_id]
        
        # Wrap in vectorized environment
        vec_env = DummyVecEnv([lambda: env])
        
        # Create PPO agent
        agent = PPO(
            "MlpPolicy",
            vec_env,
            learning_rate=config.get("learning_rate", 0.0003),
            n_steps=config.get("n_steps", 2048),
            batch_size=config.get("batch_size", 64),
            n_epochs=config.get("n_epochs", 10),
            gamma=config.get("gamma", 0.99),
            gae_lambda=config.get("gae_lambda", 0.95),
            clip_range=config.get("clip_range", 0.2),
            verbose=1
        )
        
        self.agents[agent_id] = agent
        return agent_id
        
    async def train_agent(
        self,
        agent_id: str,
        total_timesteps: int,
        callback: Optional[Any] = None
    ) -> Dict[str, Any]:
        """Train the PPO agent"""
        if agent_id not in self.agents:
            raise ValueError(f"Agent {agent_id} not found")
            
        agent = self.agents[agent_id]
        session_id = self._generate_id()
        
        # Create callback
        training_callback = TrainingCallback(callback_fn=callback)
        
        # Store session info
        self.training_sessions[session_id] = {
            'agent_id': agent_id,
            'status': 'running',
            'metrics': []
        }
        
        try:
            # Train agent
            agent.learn(
                total_timesteps=total_timesteps,
                callback=training_callback
            )
            
            self.training_sessions[session_id]['status'] = 'completed'
            self.training_sessions[session_id]['metrics'] = training_callback.metrics
            
            return {
                'session_id': session_id,
                'status': 'completed',
                'metrics': training_callback.metrics
            }
            
        except Exception as e:
            self.training_sessions[session_id]['status'] = 'failed'
            self.training_sessions[session_id]['error'] = str(e)
            raise
            
    async def evaluate_agent(
        self,
        agent_id: str,
        num_episodes: int = 10
    ) -> Dict[str, Any]:
        """Evaluate the trained agent"""
        if agent_id not in self.agents:
            raise ValueError(f"Agent {agent_id} not found")
            
        agent = self.agents[agent_id]
        env = agent.get_env()
        
        episode_rewards = []
        episode_lengths = []
        
        for _ in range(num_episodes):
            obs = env.reset()
            done = False
            episode_reward = 0
            episode_length = 0
            
            while not done:
                action, _ = agent.predict(obs, deterministic=True)
                obs, reward, done, _ = env.step(action)
                episode_reward += reward
                episode_length += 1
                
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            
        return {
            'mean_reward': np.mean(episode_rewards),
            'std_reward': np.std(episode_rewards),
            'mean_length': np.mean(episode_lengths),
            'std_length': np.std(episode_lengths),
            'episode_rewards': episode_rewards
        }
        
    async def get_agent_policy(self, agent_id: str) -> Dict[str, Any]:
        """Get the current policy parameters"""
        if agent_id not in self.agents:
            raise ValueError(f"Agent {agent_id} not found")
            
        agent = self.agents[agent_id]
        
        # Get policy network parameters
        policy_params = {}
        for name, param in agent.policy.named_parameters():
            policy_params[name] = param.detach().cpu().numpy().tolist()
            
        return {
            'agent_id': agent_id,
            'policy_params': policy_params
        }