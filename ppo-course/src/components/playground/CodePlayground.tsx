'use client';

import React, { useState, useCallback } from 'react';
import Editor from '@monaco-editor/react';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  Play, Download, Copy, Check, Terminal, 
  FileCode, AlertCircle, Loader2 
} from 'lucide-react';
import { useCodeExecution } from '@/hooks/useTraining';

interface CodeTemplate {
  name: string;
  description: string;
  code: string;
}

const codeTemplates: CodeTemplate[] = [
  {
    name: 'PPO Basic Implementation',
    description: 'Simple PPO algorithm implementation',
    code: `import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

class PPO:
    def __init__(self, state_dim, action_dim, lr=3e-4, gamma=0.99, 
                 epsilon=0.2, c1=0.5, c2=0.01):
        self.gamma = gamma
        self.epsilon = epsilon
        self.c1 = c1  # Value loss coefficient
        self.c2 = c2  # Entropy coefficient
        
        # Initialize actor-critic network
        self.policy = ActorCritic(state_dim, action_dim)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        
    def compute_gae(self, rewards, values, dones, gamma=0.99, lam=0.95):
        """Compute Generalized Advantage Estimation"""
        advantages = []
        gae = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0
            else:
                next_value = values[t + 1]
            
            delta = rewards[t] + gamma * next_value * (1 - dones[t]) - values[t]
            gae = delta + gamma * lam * (1 - dones[t]) * gae
            advantages.insert(0, gae)
            
        return torch.tensor(advantages)
    
    def update(self, states, actions, rewards, dones, old_log_probs):
        """Perform PPO update"""
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        old_log_probs = torch.FloatTensor(old_log_probs)
        
        # Get current policy outputs
        log_probs, values, entropy = self.policy.evaluate(states, actions)
        
        # Compute advantages
        advantages = self.compute_gae(rewards, values.detach().numpy(), dones)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Compute ratio
        ratio = torch.exp(log_probs - old_log_probs)
        
        # Compute clipped objective
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * advantages
        
        # Compute losses
        actor_loss = -torch.min(surr1, surr2).mean()
        critic_loss = self.c1 * nn.MSELoss()(values, returns)
        entropy_loss = -self.c2 * entropy.mean()
        
        # Total loss
        loss = actor_loss + critic_loss + entropy_loss
        
        # Update
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()

print("PPO implementation loaded successfully!")
`
  },
  {
    name: 'Actor-Critic Network',
    description: 'Neural network architecture for PPO',
    code: `import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super(ActorCritic, self).__init__()
        
        # Shared layers
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        
        # Actor head (policy)
        self.actor = nn.Linear(hidden_dim, action_dim)
        
        # Critic head (value function)
        self.critic = nn.Linear(hidden_dim, 1)
        
    def forward(self, state):
        """Forward pass returning policy logits and value"""
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        
        # Policy logits
        logits = self.actor(x)
        
        # Value estimate
        value = self.critic(x)
        
        return logits, value
    
    def act(self, state):
        """Sample action from policy"""
        logits, value = self.forward(state)
        
        # Create categorical distribution
        probs = F.softmax(logits, dim=-1)
        dist = Categorical(probs)
        
        # Sample action
        action = dist.sample()
        log_prob = dist.log_prob(action)
        
        return action.item(), log_prob.item(), value.item()
    
    def evaluate(self, states, actions):
        """Evaluate actions for PPO update"""
        logits, values = self.forward(states)
        
        # Get log probabilities
        probs = F.softmax(logits, dim=-1)
        dist = Categorical(probs)
        log_probs = dist.log_prob(actions)
        
        # Calculate entropy
        entropy = dist.entropy()
        
        return log_probs, values.squeeze(), entropy

# Test the network
net = ActorCritic(state_dim=4, action_dim=2)
print(f"Network architecture:\\n{net}")
print(f"\\nTotal parameters: {sum(p.numel() for p in net.parameters())}")
`
  },
  {
    name: 'GAE Calculation',
    description: 'Generalized Advantage Estimation implementation',
    code: `import numpy as np

def compute_gae(rewards, values, dones, gamma=0.99, lam=0.95):
    """
    Compute Generalized Advantage Estimation (GAE)
    
    Args:
        rewards: list of rewards
        values: list of value estimates V(s)
        dones: list of done flags
        gamma: discount factor
        lam: GAE lambda parameter
    
    Returns:
        advantages: GAE advantages
        returns: discounted returns
    """
    n = len(rewards)
    advantages = np.zeros(n)
    returns = np.zeros(n)
    
    # Start from the last timestep
    gae = 0
    next_value = 0  # Terminal state has value 0
    
    for t in reversed(range(n)):
        # TD error: δ_t = r_t + γ * V(s_{t+1}) - V(s_t)
        if t == n - 1:
            next_value = 0
        else:
            next_value = values[t + 1] * (1 - dones[t])
        
        delta = rewards[t] + gamma * next_value - values[t]
        
        # GAE: A_t = δ_t + (γλ)δ_{t+1} + (γλ)²δ_{t+2} + ...
        gae = delta + gamma * lam * (1 - dones[t]) * gae
        
        advantages[t] = gae
        returns[t] = advantages[t] + values[t]
    
    # Normalize advantages
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    
    return advantages, returns

# Example usage
rewards = [1, 1, 1, 1, 10]  # Reward of 10 at the end
values = [0, 2, 4, 6, 8]    # Value estimates
dones = [0, 0, 0, 0, 1]     # Episode ends at last step

advantages, returns = compute_gae(rewards, values, dones)

print("Rewards:", rewards)
print("Values:", values)
print("Advantages:", advantages)
print("Returns:", returns)
print(f"\\nNote: λ=0 gives one-step TD, λ=1 gives Monte Carlo")
`
  },
  {
    name: 'PPO Clipping Visualization',
    description: 'Visualize PPO clipping mechanism',
    code: `import numpy as np
import matplotlib.pyplot as plt

def ppo_clip_objective(ratio, advantage, epsilon=0.2):
    """Compute PPO clipped objective"""
    if advantage > 0:
        return min(ratio * advantage, (1 + epsilon) * advantage)
    else:
        return max(ratio * advantage, (1 - epsilon) * advantage)

# Generate data for visualization
ratios = np.linspace(0, 2, 100)
advantages = [1.0, 0.5, -0.5, -1.0]
epsilon = 0.2

plt.figure(figsize=(10, 6))

for adv in advantages:
    # Unclipped objective
    unclipped = ratios * adv
    
    # Clipped objective
    clipped = [ppo_clip_objective(r, adv, epsilon) for r in ratios]
    
    plt.plot(ratios, unclipped, '--', alpha=0.5, label=f'Unclipped (A={adv})')
    plt.plot(ratios, clipped, '-', linewidth=2, label=f'Clipped (A={adv})')

# Mark clipping regions
plt.axvline(x=1-epsilon, color='red', linestyle=':', alpha=0.5, label='Clip bounds')
plt.axvline(x=1+epsilon, color='red', linestyle=':', alpha=0.5)
plt.axvline(x=1, color='black', linestyle='-', alpha=0.3)

plt.xlabel('Probability Ratio r(θ)')
plt.ylabel('Objective Value')
plt.title(f'PPO Clipping with ε={epsilon}')
plt.legend()
plt.grid(True, alpha=0.3)
plt.xlim(0, 2)

# Highlight key insight
plt.text(0.5, 0.8, 'Clipping prevents\\nlarge policy updates', 
         transform=plt.gca().transAxes,
         bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))

plt.tight_layout()
plt.show()

print("Key insights:")
print("- Positive advantages (good actions) are clipped from above")
print("- Negative advantages (bad actions) are clipped from below")
print("- This creates an automatic trust region")
`
  }
];

export const CodePlayground: React.FC = () => {
  const [selectedTemplate, setSelectedTemplate] = useState(0);
  const [code, setCode] = useState(codeTemplates[0].code);
  const [copied, setCopied] = useState(false);
  
  const { executeCode, isExecuting, result, error } = useCodeExecution();

  const handleTemplateChange = (index: number) => {
    setSelectedTemplate(index);
    setCode(codeTemplates[index].code);
  };

  const handleRunCode = async () => {
    await executeCode(code, 'python');
  };

  const handleCopyCode = useCallback(() => {
    navigator.clipboard.writeText(code);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  }, [code]);

  const handleDownloadCode = useCallback(() => {
    const blob = new Blob([code], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${codeTemplates[selectedTemplate].name.toLowerCase().replace(/\s+/g, '-')}.py`;
    a.click();
    URL.revokeObjectURL(url);
  }, [code, selectedTemplate]);

  return (
    <div className="w-full h-full flex flex-col">
      {/* Header */}
      <div className="bg-white border-b p-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-4">
            <FileCode className="w-6 h-6 text-gray-600" />
            <select
              value={selectedTemplate}
              onChange={(e) => handleTemplateChange(Number(e.target.value))}
              className="px-3 py-2 border rounded-lg"
            >
              {codeTemplates.map((template, index) => (
                <option key={index} value={index}>
                  {template.name}
                </option>
              ))}
            </select>
            <p className="text-sm text-gray-600">
              {codeTemplates[selectedTemplate].description}
            </p>
          </div>
          
          <div className="flex items-center gap-2">
            <button
              onClick={handleCopyCode}
              className="p-2 text-gray-600 hover:bg-gray-100 rounded-lg"
              title="Copy code"
            >
              {copied ? <Check className="w-5 h-5 text-green-500" /> : <Copy className="w-5 h-5" />}
            </button>
            
            <button
              onClick={handleDownloadCode}
              className="p-2 text-gray-600 hover:bg-gray-100 rounded-lg"
              title="Download code"
            >
              <Download className="w-5 h-5" />
            </button>
            
            <button
              onClick={handleRunCode}
              disabled={isExecuting}
              className="px-4 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700 disabled:bg-gray-400 flex items-center gap-2"
            >
              {isExecuting ? (
                <>
                  <Loader2 className="w-4 h-4 animate-spin" />
                  Running...
                </>
              ) : (
                <>
                  <Play className="w-4 h-4" />
                  Run Code
                </>
              )}
            </button>
          </div>
        </div>
      </div>

      {/* Editor and Output */}
      <div className="flex-1 grid md:grid-cols-2 gap-0">
        {/* Code Editor */}
        <div className="border-r">
          <Editor
            height="100%"
            defaultLanguage="python"
            value={code}
            onChange={(value) => setCode(value || '')}
            theme="vs-dark"
            options={{
              minimap: { enabled: false },
              fontSize: 14,
              wordWrap: 'on',
              automaticLayout: true,
              scrollBeyondLastLine: false,
            }}
          />
        </div>

        {/* Output Panel */}
        <div className="bg-gray-900 text-white p-4 overflow-auto">
          <div className="flex items-center gap-2 mb-4">
            <Terminal className="w-5 h-5" />
            <h3 className="font-semibold">Output</h3>
          </div>
          
          <AnimatePresence mode="wait">
            {isExecuting && (
              <motion.div
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
                className="flex items-center gap-2 text-blue-400"
              >
                <Loader2 className="w-4 h-4 animate-spin" />
                Executing code...
              </motion.div>
            )}
            
            {error && !isExecuting && (
              <motion.div
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
                className="text-red-400"
              >
                <div className="flex items-center gap-2 mb-2">
                  <AlertCircle className="w-4 h-4" />
                  <span className="font-semibold">Error:</span>
                </div>
                <pre className="text-sm whitespace-pre-wrap">{error}</pre>
              </motion.div>
            )}
            
            {result && !isExecuting && !error && (
              <motion.div
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
              >
                <pre className="text-sm whitespace-pre-wrap font-mono">{result}</pre>
              </motion.div>
            )}
            
            {!isExecuting && !error && !result && (
              <motion.div
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                className="text-gray-500"
              >
                Click "Run Code" to execute the Python code
              </motion.div>
            )}
          </AnimatePresence>
        </div>
      </div>
    </div>
  );
};