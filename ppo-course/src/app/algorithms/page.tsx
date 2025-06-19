'use client';

import React from 'react';
import Link from 'next/link';
import { ArrowLeft, Brain, Users, Zap, BarChart3, Sparkles, ChevronRight, GitCompare } from 'lucide-react';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';

const algorithms = [
  {
    id: 'grpo',
    name: 'GRPO',
    fullName: 'Group Relative Policy Optimization',
    description: 'Advanced PPO variant with group-based advantage normalization for multi-task and diverse reward environments.',
    icon: Users,
    tags: ['Policy Gradient', 'Multi-Task', 'Advanced'],
    difficulty: 'Advanced',
    href: '/algorithms/grpo',
  },
  {
    id: 'ppo',
    name: 'PPO',
    fullName: 'Proximal Policy Optimization',
    description: 'The foundation algorithm - a stable and efficient policy gradient method with clipped objective.',
    icon: Brain,
    tags: ['Policy Gradient', 'On-Policy', 'Foundational'],
    difficulty: 'Intermediate',
    href: '/chapters/6',
  },
  {
    id: 'sac',
    name: 'SAC',
    fullName: 'Soft Actor-Critic',
    description: 'Off-policy algorithm that maximizes both reward and entropy for robust learning.',
    icon: Zap,
    tags: ['Actor-Critic', 'Off-Policy', 'Continuous'],
    difficulty: 'Advanced',
    href: '/algorithms/sac',
  },
  {
    id: 'mappo',
    name: 'MAPPO',
    fullName: 'Multi-Agent Proximal Policy Optimization',
    description: 'PPO extended to multi-agent environments with centralized training and decentralized execution.',
    icon: Users,
    tags: ['Multi-Agent', 'Policy Gradient', 'Centralized Training'],
    difficulty: 'Advanced',
    href: '/algorithms/mappo',
  },
  {
    id: 'dqn',
    name: 'DQN',
    fullName: 'Deep Q-Network',
    description: 'Value-based method using neural networks to approximate Q-values.',
    icon: BarChart3,
    tags: ['Value-Based', 'Off-Policy', 'Discrete'],
    difficulty: 'Beginner',
    href: '#',
    comingSoon: true,
  },
  {
    id: 'muzero',
    name: 'MuZero',
    fullName: 'Model-Based Planning',
    description: 'Combines model-based planning with model-free RL without knowing the environment dynamics.',
    icon: Sparkles,
    tags: ['Model-Based', 'Planning', 'State-of-the-Art'],
    difficulty: 'Expert',
    href: '#',
    comingSoon: true,
  },
];

const difficultyColors = {
  Beginner: 'bg-green-100 text-green-800',
  Intermediate: 'bg-blue-100 text-blue-800',
  Advanced: 'bg-orange-100 text-orange-800',
  Expert: 'bg-red-100 text-red-800',
};

export default function AlgorithmsPage() {
  return (
    <div className="container mx-auto py-8 px-4 max-w-7xl">
      <div className="mb-8">
        <Link href="/">
          <Button variant="ghost" className="mb-4">
            <ArrowLeft className="w-4 h-4 mr-2" />
            Back to Home
          </Button>
        </Link>
        
        <div className="flex items-start justify-between">
          <div>
            <h1 className="text-4xl font-bold mb-4">Algorithm Zoo</h1>
            <p className="text-xl text-muted-foreground">
              Explore advanced reinforcement learning algorithms beyond PPO. Each algorithm 
              includes interactive visualizations, implementations, and comparisons.
            </p>
          </div>
          <Link href="/algorithms/compare">
            <Button>
              <GitCompare className="w-4 h-4 mr-2" />
              Compare Algorithms
            </Button>
          </Link>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {algorithms.map((algo) => {
          const Icon = algo.icon;
          return (
            <Card key={algo.id} className="hover:shadow-lg transition-shadow">
              <CardHeader>
                <div className="flex items-start justify-between mb-2">
                  <Icon className="w-8 h-8 text-primary" />
                  <Badge 
                    variant="secondary" 
                    className={difficultyColors[algo.difficulty as keyof typeof difficultyColors]}
                  >
                    {algo.difficulty}
                  </Badge>
                </div>
                <CardTitle className="text-2xl">{algo.name}</CardTitle>
                <CardDescription className="text-sm">
                  {algo.fullName}
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <p className="text-sm">{algo.description}</p>
                
                <div className="flex flex-wrap gap-2">
                  {algo.tags.map((tag) => (
                    <Badge key={tag} variant="outline" className="text-xs">
                      {tag}
                    </Badge>
                  ))}
                </div>
                
                {algo.comingSoon ? (
                  <Button className="w-full" disabled>
                    Coming Soon
                  </Button>
                ) : (
                  <Link href={algo.href} className="block">
                    <Button className="w-full">
                      Explore {algo.name}
                      <ChevronRight className="w-4 h-4 ml-2" />
                    </Button>
                  </Link>
                )}
              </CardContent>
            </Card>
          );
        })}
      </div>

      <Card className="mt-12">
        <CardHeader>
          <CardTitle>More Algorithms Coming Soon</CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <p>
            We're continuously expanding our algorithm collection. Upcoming additions include:
          </p>
          
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <h4 className="font-semibold mb-2">Policy Gradient Methods</h4>
              <ul className="list-disc pl-6 space-y-1 text-sm text-muted-foreground">
                <li>TRPO (Trust Region Policy Optimization)</li>
                <li>IMPALA (Importance Weighted Actor-Learner)</li>
                <li>V-MPO (Maximum a Posteriori Policy Optimization)</li>
                <li>AWR (Advantage Weighted Regression)</li>
              </ul>
            </div>
            
            <div>
              <h4 className="font-semibold mb-2">Value-Based & Model-Based</h4>
              <ul className="list-disc pl-6 space-y-1 text-sm text-muted-foreground">
                <li>Rainbow DQN</li>
                <li>C51 (Categorical DQN)</li>
                <li>Dreamer & DreamerV3</li>
                <li>World Models</li>
              </ul>
            </div>
          </div>
          
          <p className="text-sm text-muted-foreground">
            Each algorithm will include the same comprehensive treatment: theory, implementation, 
            visualization, and production tips.
          </p>
        </CardContent>
      </Card>
    </div>
  );
}