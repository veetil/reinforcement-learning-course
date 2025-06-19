'use client';

import React, { useState } from 'react';
import Link from 'next/link';
import { ArrowLeft, Users, Brain, MessageSquare, Target, GitBranch, Network } from 'lucide-react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { MAPPOVisualization } from '@/components/algorithms/MAPPOVisualization';

export default function MAPPOPage() {
  const [activeTab, setActiveTab] = useState('overview');
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
            <div className="flex items-center gap-3 mb-4">
              <div className="p-2 bg-blue-100 rounded-lg">
                <Users className="w-8 h-8 text-blue-600" />
              </div>
              <div>
                <h1 className="text-4xl font-bold">MAPPO</h1>
                <p className="text-xl text-muted-foreground">
                  Multi-Agent Proximal Policy Optimization
                </p>
              </div>
            </div>
            <div className="flex items-center gap-2 mb-4">
              <Badge variant="secondary">Multi-Agent</Badge>
              <Badge variant="secondary">Policy Gradient</Badge>
              <Badge variant="secondary">Centralized Training</Badge>
              <Badge variant="outline">Advanced</Badge>
            </div>
          </div>
          
          <div className="text-right">
            <p className="text-sm text-muted-foreground mb-2">Difficulty</p>
            <Badge variant="secondary" className="bg-orange-100 text-orange-800">
              Advanced
            </Badge>
          </div>
        </div>
      </div>

      <Tabs value={activeTab} onValueChange={setActiveTab} className="space-y-6">
        <TabsList className="grid w-full grid-cols-5">
          <TabsTrigger value="overview">Overview</TabsTrigger>
          <TabsTrigger value="visualization">Visualization</TabsTrigger>
          <TabsTrigger value="theory">Theory</TabsTrigger>
          <TabsTrigger value="implementation">Implementation</TabsTrigger>
          <TabsTrigger value="research">Research</TabsTrigger>
        </TabsList>

        {/* Overview Tab */}
        <TabsContent value="overview" className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Brain className="w-5 h-5" />
                What is MAPPO?
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <p className="text-lg">
                Multi-Agent Proximal Policy Optimization (MAPPO) extends PPO to multi-agent environments 
                by using centralized training with decentralized execution. It addresses the challenges 
                of non-stationary environments and credit assignment in multi-agent settings.
              </p>
              
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div>
                  <h3 className="text-lg font-semibold mb-3">Key Features</h3>
                  <ul className="space-y-2">
                    <li className="flex items-start gap-2">
                      <div className="w-2 h-2 bg-blue-500 rounded-full mt-2"></div>
                      <span>Centralized training, decentralized execution (CTDE)</span>
                    </li>
                    <li className="flex items-start gap-2">
                      <div className="w-2 h-2 bg-blue-500 rounded-full mt-2"></div>
                      <span>Handles non-stationary multi-agent environments</span>
                    </li>
                    <li className="flex items-start gap-2">
                      <div className="w-2 h-2 bg-blue-500 rounded-full mt-2"></div>
                      <span>Optional parameter sharing across agents</span>
                    </li>
                    <li className="flex items-start gap-2">
                      <div className="w-2 h-2 bg-blue-500 rounded-full mt-2"></div>
                      <span>Communication protocols between agents</span>
                    </li>
                    <li className="flex items-start gap-2">
                      <div className="w-2 h-2 bg-blue-500 rounded-full mt-2"></div>
                      <span>Credit assignment mechanisms</span>
                    </li>
                  </ul>
                </div>
                
                <div>
                  <h3 className="text-lg font-semibold mb-3">Applications</h3>
                  <ul className="space-y-2">
                    <li className="flex items-start gap-2">
                      <Target className="w-4 h-4 text-green-500 mt-1" />
                      <span>Cooperative navigation tasks</span>
                    </li>
                    <li className="flex items-start gap-2">
                      <Target className="w-4 h-4 text-green-500 mt-1" />
                      <span>Multi-robot coordination</span>
                    </li>
                    <li className="flex items-start gap-2">
                      <Target className="w-4 h-4 text-green-500 mt-1" />
                      <span>Traffic management systems</span>
                    </li>
                    <li className="flex items-start gap-2">
                      <Target className="w-4 h-4 text-green-500 mt-1" />
                      <span>Autonomous vehicle coordination</span>
                    </li>
                    <li className="flex items-start gap-2">
                      <Target className="w-4 h-4 text-green-500 mt-1" />
                      <span>Multi-agent game environments</span>
                    </li>
                  </ul>
                </div>
              </div>
            </CardContent>
          </Card>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2 text-lg">
                  <Network className="w-5 h-5 text-blue-500" />
                  Architecture
                </CardTitle>
              </CardHeader>
              <CardContent>
                <p className="text-sm text-muted-foreground mb-3">
                  MAPPO uses separate actor networks for each agent but can share a centralized 
                  critic that has access to global state information.
                </p>
                <div className="space-y-2 text-sm">
                  <div className="flex justify-between">
                    <span>Actors:</span>
                    <span className="font-medium">Individual/Shared</span>
                  </div>
                  <div className="flex justify-between">
                    <span>Critics:</span>
                    <span className="font-medium">Centralized/Individual</span>
                  </div>
                  <div className="flex justify-between">
                    <span>Training:</span>
                    <span className="font-medium">Centralized</span>
                  </div>
                  <div className="flex justify-between">
                    <span>Execution:</span>
                    <span className="font-medium">Decentralized</span>
                  </div>
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2 text-lg">
                  <MessageSquare className="w-5 h-5 text-purple-500" />
                  Communication
                </CardTitle>
              </CardHeader>
              <CardContent>
                <p className="text-sm text-muted-foreground mb-3">
                  Agents can learn to communicate through differentiable communication 
                  protocols, sharing information to improve coordination.
                </p>
                <div className="space-y-2 text-sm">
                  <div className="flex justify-between">
                    <span>Protocol:</span>
                    <span className="font-medium">Learnable</span>
                  </div>
                  <div className="flex justify-between">
                    <span>Message Size:</span>
                    <span className="font-medium">Configurable</span>
                  </div>
                  <div className="flex justify-between">
                    <span>Range:</span>
                    <span className="font-medium">Local/Global</span>
                  </div>
                  <div className="flex justify-between">
                    <span>Bandwidth:</span>
                    <span className="font-medium">Limited</span>
                  </div>
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2 text-lg">
                  <GitBranch className="w-5 h-5 text-green-500" />
                  Scalability
                </CardTitle>
              </CardHeader>
              <CardContent>
                <p className="text-sm text-muted-foreground mb-3">
                  MAPPO scales well to many agents through parameter sharing and 
                  efficient gradient computation across the agent population.
                </p>
                <div className="space-y-2 text-sm">
                  <div className="flex justify-between">
                    <span>Max Agents:</span>
                    <span className="font-medium">100+</span>
                  </div>
                  <div className="flex justify-between">
                    <span>Memory:</span>
                    <span className="font-medium">O(n log n)</span>
                  </div>
                  <div className="flex justify-between">
                    <span>Computation:</span>
                    <span className="font-medium">Parallelizable</span>
                  </div>
                  <div className="flex justify-between">
                    <span>Sharing:</span>
                    <span className="font-medium">Optional</span>
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>

          <Card>
            <CardHeader>
              <CardTitle>MAPPO vs Other Multi-Agent Algorithms</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="overflow-x-auto">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="border-b">
                      <th className="text-left py-2">Algorithm</th>
                      <th className="text-left py-2">Training</th>
                      <th className="text-left py-2">Execution</th>
                      <th className="text-left py-2">Communication</th>
                      <th className="text-left py-2">Scalability</th>
                      <th className="text-left py-2">Stability</th>
                    </tr>
                  </thead>
                  <tbody className="text-sm">
                    <tr className="border-b">
                      <td className="py-2 font-medium">MAPPO</td>
                      <td className="py-2">Centralized</td>
                      <td className="py-2">Decentralized</td>
                      <td className="py-2">✓ Learnable</td>
                      <td className="py-2">High</td>
                      <td className="py-2">High</td>
                    </tr>
                    <tr className="border-b">
                      <td className="py-2 font-medium">MADDPG</td>
                      <td className="py-2">Centralized</td>
                      <td className="py-2">Decentralized</td>
                      <td className="py-2">✗ None</td>
                      <td className="py-2">Medium</td>
                      <td className="py-2">Medium</td>
                    </tr>
                    <tr className="border-b">
                      <td className="py-2 font-medium">QMIX</td>
                      <td className="py-2">Centralized</td>
                      <td className="py-2">Decentralized</td>
                      <td className="py-2">✗ None</td>
                      <td className="py-2">High</td>
                      <td className="py-2">High</td>
                    </tr>
                    <tr className="border-b">
                      <td className="py-2 font-medium">Independent PPO</td>
                      <td className="py-2">Decentralized</td>
                      <td className="py-2">Decentralized</td>
                      <td className="py-2">✗ None</td>
                      <td className="py-2">High</td>
                      <td className="py-2">Low</td>
                    </tr>
                  </tbody>
                </table>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        {/* Visualization Tab */}
        <TabsContent value="visualization">
          <MAPPOVisualization />
        </TabsContent>

        {/* Theory Tab */}
        <TabsContent value="theory" className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle>Mathematical Foundation</CardTitle>
            </CardHeader>
            <CardContent className="space-y-6">
              <div>
                <h3 className="text-lg font-semibold mb-3">Multi-Agent MDP Formulation</h3>
                <p className="mb-4">
                  MAPPO operates in a Multi-Agent Markov Decision Process (MAMDP) defined by the tuple 
                  <code className="bg-muted px-2 py-1 rounded mx-1">(S, A₁...Aₙ, P, R₁...Rₙ, γ)</code>:
                </p>
                <ul className="space-y-2 text-sm">
                  <li><strong>S</strong>: Global state space</li>
                  <li><strong>Aᵢ</strong>: Action space for agent i</li>
                  <li><strong>P</strong>: State transition probability</li>
                  <li><strong>Rᵢ</strong>: Reward function for agent i</li>
                  <li><strong>γ</strong>: Discount factor</li>
                </ul>
              </div>

              <div>
                <h3 className="text-lg font-semibold mb-3">Policy Gradient</h3>
                <p className="mb-4">
                  Each agent i learns a policy πᵢ(aᵢ|oᵢ) where oᵢ is the local observation. 
                  The policy gradient for agent i is:
                </p>
                <div className="bg-muted p-4 rounded-lg font-mono text-sm">
                  ∇θᵢ J(θᵢ) = E[∇θᵢ log πᵢ(aᵢ|oᵢ) Âᵢ]
                </div>
                <p className="mt-2 text-sm text-muted-foreground">
                  Where Âᵢ is the advantage estimate for agent i, computed using the centralized critic.
                </p>
              </div>

              <div>
                <h3 className="text-lg font-semibold mb-3">Centralized Value Function</h3>
                <p className="mb-4">
                  The centralized critic estimates the value function using global information:
                </p>
                <div className="bg-muted p-4 rounded-lg font-mono text-sm">
                  Vᵢ(s, a₁...aₙ) = E[∑ₜ γᵗ rᵢₜ | s₀=s, a₀=(a₁...aₙ)]
                </div>
                <p className="mt-2 text-sm text-muted-foreground">
                  This allows for better credit assignment and handling of non-stationarity.
                </p>
              </div>

              <div>
                <h3 className="text-lg font-semibold mb-3">PPO Clipping in Multi-Agent Setting</h3>
                <p className="mb-4">
                  The clipped surrogate objective for each agent becomes:
                </p>
                <div className="bg-muted p-4 rounded-lg font-mono text-sm">
                  L^CLIP_i(θᵢ) = E[min(rᵢ(θᵢ)Âᵢ, clip(rᵢ(θᵢ), 1-ε, 1+ε)Âᵢ)]
                </div>
                <p className="mt-2 text-sm text-muted-foreground">
                  Where rᵢ(θᵢ) = πᵢ(aᵢ|oᵢ)/πᵢ_old(aᵢ|oᵢ) is the probability ratio for agent i.
                </p>
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>Communication Mechanism</CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div>
                <h3 className="text-lg font-semibold mb-3">Message Passing</h3>
                <p className="mb-4">
                  Agents can send messages mᵢ to other agents. The communication-augmented policy becomes:
                </p>
                <div className="bg-muted p-4 rounded-lg font-mono text-sm">
                  πᵢ(aᵢ|oᵢ, m₁...mₙ₋₁)
                </div>
                <p className="mt-2 text-sm text-muted-foreground">
                  Messages are generated by a learned communication module and can be discrete or continuous.
                </p>
              </div>

              <div>
                <h3 className="text-lg font-semibold mb-3">Differentiable Communication</h3>
                <p className="mb-4">
                  For continuous communication, messages are produced by:
                </p>
                <div className="bg-muted p-4 rounded-lg font-mono text-sm">
                  mᵢ = gᵢ(oᵢ, hᵢ)
                </div>
                <p className="mt-2 text-sm text-muted-foreground">
                  Where gᵢ is a neural network and hᵢ is the hidden state. This allows end-to-end training 
                  of communication protocols.
                </p>
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>Credit Assignment</CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div>
                <h3 className="text-lg font-semibold mb-3">Counterfactual Reasoning</h3>
                <p className="mb-4">
                  MAPPO can use counterfactual reasoning to assign credit to individual agents:
                </p>
                <div className="bg-muted p-4 rounded-lg font-mono text-sm">
                  Δᵢ = Q(s, a₁...aₙ) - Q(s, a₁...aᵢ₋₁, cᵢ, aᵢ₊₁...aₙ)
                </div>
                <p className="mt-2 text-sm text-muted-foreground">
                  Where cᵢ is a default action for agent i, measuring the contribution of agent i's actual action.
                </p>
              </div>

              <div>
                <h3 className="text-lg font-semibold mb-3">Difference Rewards</h3>
                <p className="mb-4">
                  Another approach uses difference rewards:
                </p>
                <div className="bg-muted p-4 rounded-lg font-mono text-sm">
                  Dᵢ = G(τ) - G(τ⁻ⁱ)
                </div>
                <p className="mt-2 text-sm text-muted-foreground">
                  Where G(τ) is the global return and G(τ⁻ⁱ) is the return without agent i's contribution.
                </p>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        {/* Implementation Tab */}
        <TabsContent value="implementation" className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle>MAPPO Implementation Guide</CardTitle>
            </CardHeader>
            <CardContent className="space-y-6">
              <div>
                <h3 className="text-lg font-semibold mb-3">Network Architecture</h3>
                <div className="bg-muted p-4 rounded-lg">
                  <pre className="text-sm overflow-x-auto">
{`class MAPPO:
    def __init__(self, config):
        self.n_agents = config.n_agents
        self.centralized_critic = config.centralized_critic
        self.parameter_sharing = config.parameter_sharing
        
        # Build networks
        if self.parameter_sharing:
            self.shared_actor = self.build_actor()
            self.actors = [self.shared_actor] * self.n_agents
        else:
            self.actors = [self.build_actor() for _ in range(self.n_agents)]
        
        if self.centralized_critic:
            self.critic = self.build_centralized_critic()
        else:
            self.critics = [self.build_critic() for _ in range(self.n_agents)]`}
                  </pre>
                </div>
              </div>

              <div>
                <h3 className="text-lg font-semibold mb-3">Action Selection</h3>
                <div className="bg-muted p-4 rounded-lg">
                  <pre className="text-sm overflow-x-auto">
{`async selectActions(states: tf.Tensor[]): Promise<{
    actions: tf.Tensor[];
    logProbs: tf.Tensor[];
    values: tf.Tensor[];
}> {
    const actions: tf.Tensor[] = [];
    const logProbs: tf.Tensor[] = [];
    const values: tf.Tensor[] = [];
    
    // Generate communication messages
    let messages: tf.Tensor[] | undefined;
    if (this.communicationModule) {
        messages = states.map(state => 
            this.communicationModule!.predict(state) as tf.Tensor
        );
    }
    
    // Select actions for each agent
    for (let i = 0; i < this.config.nAgents; i++) {
        const actor = this.config.parameterSharing ? 
            this.sharedActor! : this.actors[i];
        
        let actionProbs: tf.Tensor;
        if (messages) {
            const avgMessage = tf.stack(messages).mean(0);
            const inputWithComm = tf.concat([states[i], avgMessage], 1);
            actionProbs = actor.predict(inputWithComm) as tf.Tensor;
        } else {
            actionProbs = actor.predict(states[i]) as tf.Tensor;
        }
        
        const action = tf.multinomial(actionProbs, 1);
        const logProb = tf.log(tf.add(
            actionProbs.gather(action, 1), 1e-8));
        
        actions.push(action);
        logProbs.push(logProb);
    }
    
    // Compute values
    if (this.config.centralizedCritic) {
        const globalState = tf.concat(states, 1);
        const allValues = this.centralizedCritic!.predict(globalState) as tf.Tensor;
        for (let i = 0; i < this.config.nAgents; i++) {
            values.push(allValues.slice([0, i], [-1, 1]));
        }
    } else {
        for (let i = 0; i < this.config.nAgents; i++) {
            const value = this.critics[i].predict(states[i]) as tf.Tensor;
            values.push(value);
        }
    }
    
    return { actions, logProbs, values };
}`}
                  </pre>
                </div>
              </div>

              <div>
                <h3 className="text-lg font-semibold mb-3">Training Loop</h3>
                <div className="bg-muted p-4 rounded-lg">
                  <pre className="text-sm overflow-x-auto">
{`async update(rollouts: MultiAgentRollout[]): Promise<UpdateInfo> {
    // Compute advantages and returns
    const processedData = await this.computeAdvantagesAndReturns(rollouts);
    
    // Prepare batch data for each agent
    const agentBatches = this.prepareAgentBatches(processedData);
    
    const losses = { actorLoss: [], criticLoss: [], entropy: [] };
    
    // Update networks for multiple epochs
    for (let epoch = 0; epoch < this.config.nEpochs; epoch++) {
        if (this.config.parameterSharing) {
            const { actorLoss, entropy } = await this.updateSharedActor(agentBatches);
            losses.actorLoss.push(actorLoss);
            losses.entropy.push(entropy);
        } else {
            for (let i = 0; i < this.config.nAgents; i++) {
                const { actorLoss, entropy } = await this.updateActor(i, agentBatches[i]);
                losses.actorLoss.push(actorLoss);
                losses.entropy.push(entropy);
            }
        }
        
        // Update critics
        if (this.config.centralizedCritic) {
            const criticLoss = await this.updateCentralizedCritic(agentBatches);
            losses.criticLoss.push(criticLoss);
        } else {
            for (let i = 0; i < this.config.nAgents; i++) {
                const criticLoss = await this.updateCritic(i, agentBatches[i]);
                losses.criticLoss.push(criticLoss);
            }
        }
    }
    
    return losses;
}`}
                  </pre>
                </div>
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>Best Practices</CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div>
                  <h3 className="text-lg font-semibold mb-3">Hyperparameter Tuning</h3>
                  <ul className="space-y-2 text-sm">
                    <li className="flex items-start gap-2">
                      <div className="w-2 h-2 bg-green-500 rounded-full mt-2"></div>
                      <span>Start with smaller learning rates (1e-4 to 3e-4)</span>
                    </li>
                    <li className="flex items-start gap-2">
                      <div className="w-2 h-2 bg-green-500 rounded-full mt-2"></div>
                      <span>Use larger batch sizes for stability</span>
                    </li>
                    <li className="flex items-start gap-2">
                      <div className="w-2 h-2 bg-green-500 rounded-full mt-2"></div>
                      <span>Adjust clip range based on environment complexity</span>
                    </li>
                    <li className="flex items-start gap-2">
                      <div className="w-2 h-2 bg-green-500 rounded-full mt-2"></div>
                      <span>Experiment with different GAE lambda values</span>
                    </li>
                  </ul>
                </div>
                
                <div>
                  <h3 className="text-lg font-semibold mb-3">Common Pitfalls</h3>
                  <ul className="space-y-2 text-sm">
                    <li className="flex items-start gap-2">
                      <div className="w-2 h-2 bg-red-500 rounded-full mt-2"></div>
                      <span>Non-stationary environments from other agents</span>
                    </li>
                    <li className="flex items-start gap-2">
                      <div className="w-2 h-2 bg-red-500 rounded-full mt-2"></div>
                      <span>Credit assignment becomes harder with more agents</span>
                    </li>
                    <li className="flex items-start gap-2">
                      <div className="w-2 h-2 bg-red-500 rounded-full mt-2"></div>
                      <span>Communication can become noisy or degenerate</span>
                    </li>
                    <li className="flex items-start gap-2">
                      <div className="w-2 h-2 bg-red-500 rounded-full mt-2"></div>
                      <span>Scalability issues with centralized critic</span>
                    </li>
                  </ul>
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        {/* Research Tab */}
        <TabsContent value="research" className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle>Recent Research & Developments</CardTitle>
            </CardHeader>
            <CardContent className="space-y-6">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div>
                  <h3 className="text-lg font-semibold mb-3">Key Papers</h3>
                  <div className="space-y-4">
                    <div className="border-l-4 border-blue-500 pl-4">
                      <h4 className="font-medium">Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments</h4>
                      <p className="text-sm text-muted-foreground">Lowe et al., 2017</p>
                      <p className="text-sm">Introduced MADDPG and centralized training paradigm</p>
                    </div>
                    
                    <div className="border-l-4 border-blue-500 pl-4">
                      <h4 className="font-medium">The Surprising Effectiveness of MAPPO in Cooperative Multi-Agent Games</h4>
                      <p className="text-sm text-muted-foreground">Yu et al., 2021</p>
                      <p className="text-sm">Demonstrated MAPPO's superior performance across environments</p>
                    </div>
                    
                    <div className="border-l-4 border-blue-500 pl-4">
                      <h4 className="font-medium">Learning Multiagent Communication with Backpropagation</h4>
                      <p className="text-sm text-muted-foreground">Sukhbaatar et al., 2016</p>
                      <p className="text-sm">Foundational work on differentiable communication</p>
                    </div>
                  </div>
                </div>
                
                <div>
                  <h3 className="text-lg font-semibold mb-3">Recent Advances</h3>
                  <div className="space-y-4">
                    <div className="border-l-4 border-green-500 pl-4">
                      <h4 className="font-medium">Hierarchical MAPPO</h4>
                      <p className="text-sm">Extends MAPPO to hierarchical action spaces for complex coordination</p>
                    </div>
                    
                    <div className="border-l-4 border-green-500 pl-4">
                      <h4 className="font-medium">Meta-Learning MAPPO</h4>
                      <p className="text-sm">Enables quick adaptation to new multi-agent environments</p>
                    </div>
                    
                    <div className="border-l-4 border-green-500 pl-4">
                      <h4 className="font-medium">Attention-based Communication</h4>
                      <p className="text-sm">Uses attention mechanisms for selective agent communication</p>
                    </div>
                    
                    <div className="border-l-4 border-green-500 pl-4">
                      <h4 className="font-medium">Distributed MAPPO</h4>
                      <p className="text-sm">Scales to hundreds of agents using distributed training</p>
                    </div>
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>Future Directions</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div>
                  <h3 className="text-lg font-semibold mb-3">Technical Challenges</h3>
                  <ul className="space-y-2 text-sm">
                    <li className="flex items-start gap-2">
                      <div className="w-2 h-2 bg-orange-500 rounded-full mt-2"></div>
                      <span>Scalability to thousands of agents</span>
                    </li>
                    <li className="flex items-start gap-2">
                      <div className="w-2 h-2 bg-orange-500 rounded-full mt-2"></div>
                      <span>Partial observability in complex environments</span>
                    </li>
                    <li className="flex items-start gap-2">
                      <div className="w-2 h-2 bg-orange-500 rounded-full mt-2"></div>
                      <span>Dynamic agent populations</span>
                    </li>
                    <li className="flex items-start gap-2">
                      <div className="w-2 h-2 bg-orange-500 rounded-full mt-2"></div>
                      <span>Continual learning and adaptation</span>
                    </li>
                  </ul>
                </div>
                
                <div>
                  <h3 className="text-lg font-semibold mb-3">Application Areas</h3>
                  <ul className="space-y-2 text-sm">
                    <li className="flex items-start gap-2">
                      <div className="w-2 h-2 bg-purple-500 rounded-full mt-2"></div>
                      <span>Autonomous vehicle coordination</span>
                    </li>
                    <li className="flex items-start gap-2">
                      <div className="w-2 h-2 bg-purple-500 rounded-full mt-2"></div>
                      <span>Smart city traffic management</span>
                    </li>
                    <li className="flex items-start gap-2">
                      <div className="w-2 h-2 bg-purple-500 rounded-full mt-2"></div>
                      <span>Distributed energy systems</span>
                    </li>
                    <li className="flex items-start gap-2">
                      <div className="w-2 h-2 bg-purple-500 rounded-full mt-2"></div>
                      <span>Multi-robot manufacturing</span>
                    </li>
                  </ul>
                </div>
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>Benchmarks & Evaluation</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                <div>
                  <h3 className="text-lg font-semibold mb-3">Standard Environments</h3>
                  <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                    <div className="text-center p-3 border rounded-lg">
                      <h4 className="font-medium text-sm">MPE</h4>
                      <p className="text-xs text-muted-foreground">Multi-Particle Env</p>
                    </div>
                    <div className="text-center p-3 border rounded-lg">
                      <h4 className="font-medium text-sm">SMAC</h4>
                      <p className="text-xs text-muted-foreground">StarCraft II</p>
                    </div>
                    <div className="text-center p-3 border rounded-lg">
                      <h4 className="font-medium text-sm">MAgent</h4>
                      <p className="text-xs text-muted-foreground">Large-scale</p>
                    </div>
                    <div className="text-center p-3 border rounded-lg">
                      <h4 className="font-medium text-sm">Hanabi</h4>
                      <p className="text-xs text-muted-foreground">Card Game</p>
                    </div>
                  </div>
                </div>
                
                <div>
                  <h3 className="text-lg font-semibold mb-3">Evaluation Metrics</h3>
                  <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                    <div>
                      <h4 className="font-medium mb-2">Performance</h4>
                      <ul className="text-sm space-y-1">
                        <li>• Episode return</li>
                        <li>• Success rate</li>
                        <li>• Sample efficiency</li>
                      </ul>
                    </div>
                    <div>
                      <h4 className="font-medium mb-2">Coordination</h4>
                      <ul className="text-sm space-y-1">
                        <li>• Cooperation score</li>
                        <li>• Communication usage</li>
                        <li>• Collective behavior</li>
                      </ul>
                    </div>
                    <div>
                      <h4 className="font-medium mb-2">Scalability</h4>
                      <ul className="text-sm space-y-1">
                        <li>• Training time</li>
                        <li>• Memory usage</li>
                        <li>• Agent count scaling</li>
                      </ul>
                    </div>
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
}