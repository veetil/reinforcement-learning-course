'use client';

import React, { useState } from 'react';
import Link from 'next/link';
import { ArrowLeft, Github, FileCode, BookOpen, Zap, Brain, BarChart } from 'lucide-react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { SACVisualization } from '@/components/algorithms/SACVisualization';

export default function SACPage() {
  const [activeTab, setActiveTab] = useState('theory');
  return (
    <div className="container mx-auto py-8 px-4 max-w-7xl">
      {/* Header */}
      <div className="mb-8">
        <Link href="/algorithms">
          <Button variant="ghost" className="mb-4">
            <ArrowLeft className="w-4 h-4 mr-2" />
            Back to Algorithm Zoo
          </Button>
        </Link>
        
        <div className="flex items-start justify-between">
          <div>
            <h1 className="text-4xl font-bold mb-2">Soft Actor-Critic (SAC)</h1>
            <p className="text-xl text-muted-foreground">
              Maximum entropy reinforcement learning for continuous control
            </p>
          </div>
          <Badge className="bg-orange-100 text-orange-800">Advanced</Badge>
        </div>
      </div>

      {/* Key Features */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-8">
        <Card>
          <CardHeader className="pb-3">
            <CardTitle className="text-lg flex items-center gap-2">
              <Zap className="w-5 h-5 text-orange-500" />
              Off-Policy
            </CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-sm text-muted-foreground">
              Learns from replay buffer, making it highly sample-efficient
            </p>
          </CardContent>
        </Card>
        
        <Card>
          <CardHeader className="pb-3">
            <CardTitle className="text-lg flex items-center gap-2">
              <Brain className="w-5 h-5 text-purple-500" />
              Maximum Entropy
            </CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-sm text-muted-foreground">
              Encourages exploration by maximizing action entropy
            </p>
          </CardContent>
        </Card>
        
        <Card>
          <CardHeader className="pb-3">
            <CardTitle className="text-lg flex items-center gap-2">
              <BarChart className="w-5 h-5 text-blue-500" />
              Stable Learning
            </CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-sm text-muted-foreground">
              Twin Q-networks and automatic temperature tuning
            </p>
          </CardContent>
        </Card>
      </div>

      {/* Main Content Tabs */}
      <Tabs value={activeTab} onValueChange={setActiveTab} className="space-y-6">
        <TabsList className="grid w-full grid-cols-4">
          <TabsTrigger value="theory">Theory</TabsTrigger>
          <TabsTrigger value="visualization">Visualization</TabsTrigger>
          <TabsTrigger value="implementation">Implementation</TabsTrigger>
          <TabsTrigger value="comparison">Comparison</TabsTrigger>
        </TabsList>

        {/* Theory Tab */}
        <TabsContent value="theory" className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle>Maximum Entropy Reinforcement Learning</CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <p>
                SAC is an off-policy actor-critic algorithm that maximizes a trade-off between 
                expected return and entropy. This encourages the agent to explore more systematically 
                and learn more robust policies.
              </p>
              
              <div className="bg-muted p-4 rounded-lg">
                <h4 className="font-semibold mb-2">The SAC Objective</h4>
                <p className="font-mono text-sm">
                  J(π) = Σ E_(s,a)~ρ_π [r(s,a) + α H(π(·|s))]
                </p>
                <p className="text-sm mt-2 text-muted-foreground">
                  Where α is the temperature parameter that determines the relative importance 
                  of entropy versus reward.
                </p>
              </div>

              <div>
                <h4 className="font-semibold mb-2">Key Components</h4>
                <ul className="space-y-2">
                  <li className="flex gap-2">
                    <span className="font-medium">1.</span>
                    <div>
                      <strong>Actor Network:</strong> Outputs mean and std of Gaussian policy
                      <p className="text-sm text-muted-foreground">
                        π(a|s) = tanh(μ(s) + σ(s) · ε), where ε ~ N(0, I)
                      </p>
                    </div>
                  </li>
                  <li className="flex gap-2">
                    <span className="font-medium">2.</span>
                    <div>
                      <strong>Twin Critic Networks:</strong> Two Q-functions to mitigate overestimation
                      <p className="text-sm text-muted-foreground">
                        Q_target = min(Q₁(s', a'), Q₂(s', a')) - α log π(a'|s')
                      </p>
                    </div>
                  </li>
                  <li className="flex gap-2">
                    <span className="font-medium">3.</span>
                    <div>
                      <strong>Automatic Temperature Tuning:</strong> Learns α to maintain target entropy
                      <p className="text-sm text-muted-foreground">
                        J(α) = E_a~π [-α log π(a|s) - α H̄]
                      </p>
                    </div>
                  </li>
                </ul>
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>Algorithm Details</CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="bg-muted p-4 rounded-lg font-mono text-sm">
                <pre>{`Algorithm SAC:
  Initialize: Actor π_φ, Critics Q_θ1, Q_θ2, Target critics
  Initialize: Temperature α, Replay buffer D
  
  for each iteration:
    for each environment step:
      a_t ~ π_φ(·|s_t)  # Sample action
      s_{t+1} ~ p(·|s_t, a_t)  # Step environment
      D ← D ∪ {(s_t, a_t, r_t, s_{t+1})}
      
    for each gradient step:
      Sample batch from D
      
      # Update critics
      y = r + γ(min_i Q_θ̄i(s', a') - α log π(a'|s'))
      θ_i ← θ_i - λ_Q ∇_θi (Q_θi(s,a) - y)²
      
      # Update actor
      φ ← φ - λ_π ∇_φ (α log π_φ(a|s) - min_i Q_θi(s,a))
      
      # Update temperature
      α ← α - λ_α ∇_α (log π_φ(a|s) + H̄)
      
      # Soft update targets
      θ̄_i ← τθ_i + (1-τ)θ̄_i`}</pre>
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>Advantages & Use Cases</CardTitle>
            </CardHeader>
            <CardContent className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div>
                <h4 className="font-semibold mb-2">Advantages</h4>
                <ul className="list-disc pl-5 space-y-1 text-sm">
                  <li>High sample efficiency (off-policy)</li>
                  <li>No need for importance sampling</li>
                  <li>Automatic exploration via entropy</li>
                  <li>Stable convergence properties</li>
                  <li>Works well with continuous actions</li>
                  <li>Robust to hyperparameter choices</li>
                </ul>
              </div>
              <div>
                <h4 className="font-semibold mb-2">Best Use Cases</h4>
                <ul className="list-disc pl-5 space-y-1 text-sm">
                  <li>Robotics and continuous control</li>
                  <li>Tasks requiring exploration</li>
                  <li>Limited environment interactions</li>
                  <li>High-dimensional action spaces</li>
                  <li>Real-world applications with safety constraints</li>
                </ul>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        {/* Visualization Tab */}
        <TabsContent value="visualization">
          <SACVisualization />
        </TabsContent>

        {/* Implementation Tab */}
        <TabsContent value="implementation" className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle>Implementation Guide</CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div>
                <h4 className="font-semibold mb-2">1. Network Architecture</h4>
                <pre className="bg-muted p-4 rounded-lg overflow-x-auto text-sm">
{`# Actor Network (Gaussian Policy)
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)
        
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        mean = self.mean(x)
        log_std = self.log_std(x).clamp(-20, 2)
        return mean, log_std
        
    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()  # Reparameterization trick
        action = torch.tanh(x_t)
        
        # Compute log probability with correction for tanh
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        
        return action, log_prob`}</pre>
              </div>

              <div>
                <h4 className="font-semibold mb-2">2. Critic Networks</h4>
                <pre className="bg-muted p-4 rounded-lg overflow-x-auto text-sm">
{`# Twin Q-Networks
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()
        # Q1 architecture
        self.q1 = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Q2 architecture
        self.q2 = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, state, action):
        sa = torch.cat([state, action], 1)
        q1 = self.q1(sa)
        q2 = self.q2(sa)
        return q1, q2`}</pre>
              </div>

              <div>
                <h4 className="font-semibold mb-2">3. Training Loop</h4>
                <pre className="bg-muted p-4 rounded-lg overflow-x-auto text-sm">
{`def update_sac(batch, actor, critic, target_critic, 
               log_alpha, target_entropy):
    states, actions, rewards, next_states, dones = batch
    
    # Sample next actions and compute target Q
    with torch.no_grad():
        next_actions, next_log_probs = actor.sample(next_states)
        target_q1, target_q2 = target_critic(next_states, next_actions)
        target_q = torch.min(target_q1, target_q2)
        target_value = target_q - alpha * next_log_probs
        target_q = rewards + gamma * (1 - dones) * target_value
    
    # Update critics
    current_q1, current_q2 = critic(states, actions)
    critic_loss = F.mse_loss(current_q1, target_q) + \
                  F.mse_loss(current_q2, target_q)
    
    # Update actor
    new_actions, log_probs = actor.sample(states)
    q1_new, q2_new = critic(states, new_actions)
    q_new = torch.min(q1_new, q2_new)
    actor_loss = (alpha * log_probs - q_new).mean()
    
    # Update temperature
    alpha_loss = -(log_alpha * (log_probs + target_entropy).detach()).mean()
    
    return actor_loss, critic_loss, alpha_loss`}</pre>
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>Hyperparameters & Tips</CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div>
                  <h4 className="font-semibold mb-2">Recommended Hyperparameters</h4>
                  <ul className="space-y-1 text-sm">
                    <li><code>learning_rate</code>: 3e-4 for all networks</li>
                    <li><code>batch_size</code>: 256</li>
                    <li><code>buffer_size</code>: 1e6</li>
                    <li><code>gamma</code>: 0.99</li>
                    <li><code>tau</code>: 0.005 (soft update)</li>
                    <li><code>initial_alpha</code>: 0.2</li>
                    <li><code>target_entropy</code>: -dim(A)</li>
                  </ul>
                </div>
                <div>
                  <h4 className="font-semibold mb-2">Implementation Tips</h4>
                  <ul className="space-y-1 text-sm">
                    <li>• Use gradient clipping for stability</li>
                    <li>• Initialize networks with proper scaling</li>
                    <li>• Log-transform alpha for optimization</li>
                    <li>• Use double-Q trick to prevent overestimation</li>
                    <li>• Start collecting data before training</li>
                    <li>• Monitor entropy to ensure exploration</li>
                  </ul>
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        {/* Comparison Tab */}
        <TabsContent value="comparison" className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle>SAC vs PPO vs GRPO</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="overflow-x-auto">
                <table className="w-full border-collapse">
                  <thead>
                    <tr className="border-b">
                      <th className="text-left p-2">Aspect</th>
                      <th className="text-left p-2">SAC</th>
                      <th className="text-left p-2">PPO</th>
                      <th className="text-left p-2">GRPO</th>
                    </tr>
                  </thead>
                  <tbody className="text-sm">
                    <tr className="border-b">
                      <td className="p-2 font-medium">Learning Type</td>
                      <td className="p-2">Off-policy</td>
                      <td className="p-2">On-policy</td>
                      <td className="p-2">On-policy</td>
                    </tr>
                    <tr className="border-b">
                      <td className="p-2 font-medium">Sample Efficiency</td>
                      <td className="p-2 text-green-600">High</td>
                      <td className="p-2 text-orange-600">Medium</td>
                      <td className="p-2 text-orange-600">Medium</td>
                    </tr>
                    <tr className="border-b">
                      <td className="p-2 font-medium">Exploration</td>
                      <td className="p-2">Automatic (entropy)</td>
                      <td className="p-2">Manual tuning</td>
                      <td className="p-2">Group-based</td>
                    </tr>
                    <tr className="border-b">
                      <td className="p-2 font-medium">Action Space</td>
                      <td className="p-2">Continuous</td>
                      <td className="p-2">Both</td>
                      <td className="p-2">Both</td>
                    </tr>
                    <tr className="border-b">
                      <td className="p-2 font-medium">Stability</td>
                      <td className="p-2 text-green-600">Very High</td>
                      <td className="p-2 text-green-600">High</td>
                      <td className="p-2 text-green-600">High</td>
                    </tr>
                    <tr className="border-b">
                      <td className="p-2 font-medium">Hyperparameter Sensitivity</td>
                      <td className="p-2 text-green-600">Low</td>
                      <td className="p-2 text-orange-600">Medium</td>
                      <td className="p-2 text-orange-600">Medium</td>
                    </tr>
                    <tr className="border-b">
                      <td className="p-2 font-medium">Best For</td>
                      <td className="p-2">Robotics, Control</td>
                      <td className="p-2">General RL</td>
                      <td className="p-2">Multi-task, Diverse rewards</td>
                    </tr>
                  </tbody>
                </table>
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>Performance Comparison</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="h-64 flex items-center justify-center text-muted-foreground">
                [Interactive benchmark comparison would go here]
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>When to Use SAC</CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div>
                <h4 className="font-semibold text-green-600 mb-2">✓ Use SAC When:</h4>
                <ul className="list-disc pl-5 space-y-1 text-sm">
                  <li>You have continuous action spaces</li>
                  <li>Sample efficiency is critical</li>
                  <li>You can afford to store a large replay buffer</li>
                  <li>The environment is deterministic or near-deterministic</li>
                  <li>You need stable, robust performance</li>
                </ul>
              </div>
              
              <div>
                <h4 className="font-semibold text-red-600 mb-2">✗ Avoid SAC When:</h4>
                <ul className="list-disc pl-5 space-y-1 text-sm">
                  <li>You have discrete action spaces only</li>
                  <li>On-policy learning is required</li>
                  <li>Memory is severely constrained</li>
                  <li>You need the simplest possible implementation</li>
                  <li>The environment is highly stochastic</li>
                </ul>
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>

      {/* Resources */}
      <Card className="mt-8">
        <CardHeader>
          <CardTitle>Resources & References</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div>
              <h4 className="font-semibold mb-2">Papers</h4>
              <ul className="space-y-1 text-sm">
                <li>
                  <a href="https://arxiv.org/abs/1801.01290" className="text-blue-600 hover:underline">
                    Soft Actor-Critic: Off-Policy Maximum Entropy RL (Original)
                  </a>
                </li>
                <li>
                  <a href="https://arxiv.org/abs/1812.05905" className="text-blue-600 hover:underline">
                    SAC: Algorithms and Applications (Updated version)
                  </a>
                </li>
              </ul>
            </div>
            <div>
              <h4 className="font-semibold mb-2">Code & Examples</h4>
              <div className="flex gap-2">
                <a href="https://github.com/rail-berkeley/softlearning" target="_blank" rel="noopener noreferrer">
                  <Button size="sm" variant="outline">
                    <Github className="w-4 h-4 mr-2" />
                    Official Repo
                  </Button>
                </a>
                <Link href="/playground?algorithm=sac">
                  <Button size="sm" variant="outline">
                    <FileCode className="w-4 h-4 mr-2" />
                    Try in Playground
                  </Button>
                </Link>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}