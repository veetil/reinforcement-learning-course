'use client';

import React from 'react';
import { motion } from 'framer-motion';
import { useRouter } from 'next/navigation';
import Link from 'next/link';
import {
  Brain, Zap, Target, TrendingUp, Cpu, Award, BookOpen, Network,
  Clock, BarChart, Play
} from 'lucide-react';

interface Chapter {
  id: number;
  title: string;
  description: string;
  topics: string[];
  icon: React.ElementType;
  estimatedTime: string;
  difficulty: 'Beginner' | 'Intermediate' | 'Advanced' | 'Expert';
  color: string;
}

const chapters: Chapter[] = [
  {
    id: 1,
    title: 'Chapter 1: Foundations',
    description: 'Neural networks, optimization, and mathematical prerequisites',
    topics: ['Neural Networks', 'Backpropagation', 'Optimization', 'PyTorch Basics'],
    icon: Brain,
    estimatedTime: '45 min',
    difficulty: 'Beginner',
    color: 'bg-blue-500'
  },
  {
    id: 2,
    title: 'Chapter 2: RL Fundamentals',
    description: 'MDP framework, policies, rewards, and exploration strategies',
    topics: ['Markov Chains', 'MDP', 'Policies', 'Exploration'],
    icon: Target,
    estimatedTime: '60 min',
    difficulty: 'Beginner',
    color: 'bg-green-500'
  },
  {
    id: 3,
    title: 'Chapter 3: Value Functions',
    description: 'State values, action values, and Bellman equations explained',
    topics: ['V(s) and Q(s,a)', 'Bellman Equations', 'TD Learning', 'Monte Carlo'],
    icon: BarChart,
    estimatedTime: '50 min',
    difficulty: 'Intermediate',
    color: 'bg-purple-500'
  },
  {
    id: 4,
    title: 'Chapter 4: Policy Gradient',
    description: 'From REINFORCE to natural policy gradient methods',
    topics: ['REINFORCE', 'Baselines', 'Natural Gradient', 'TRPO'],
    icon: TrendingUp,
    estimatedTime: '55 min',
    difficulty: 'Intermediate',
    color: 'bg-orange-500'
  },
  {
    id: 5,
    title: 'Chapter 5: Actor-Critic',
    description: 'Combining value and policy methods for stable learning',
    topics: ['A2C', 'A3C', 'GAE', 'Advantage Functions'],
    icon: Zap,
    estimatedTime: '65 min',
    difficulty: 'Intermediate',
    color: 'bg-yellow-500'
  },
  {
    id: 6,
    title: 'Chapter 6: PPO Algorithm',
    description: 'Deep dive into clipping, advantages, and implementation',
    topics: ['PPO Clipping', 'Multiple Epochs', 'Implementation', 'Hyperparameters'],
    icon: Award,
    estimatedTime: '70 min',
    difficulty: 'Advanced',
    color: 'bg-red-500'
  },
  {
    id: 7,
    title: 'Chapter 7: Advanced Topics',
    description: 'RLHF, reward modeling, and preference learning',
    topics: ['RLHF', 'Bradley-Terry', 'Reward Models', 'Human Feedback'],
    icon: BookOpen,
    estimatedTime: '60 min',
    difficulty: 'Advanced',
    color: 'bg-indigo-500'
  },
  {
    id: 8,
    title: 'Chapter 8: VERL System',
    description: 'Distributed RL with separated Actor, Critic, and Rollout',
    topics: ['VERL Architecture', 'HybridFlow', 'Distributed Training', 'Ray Integration'],
    icon: Network,
    estimatedTime: '75 min',
    difficulty: 'Expert',
    color: 'bg-pink-500'
  }
];

const difficultyColors = {
  Beginner: 'text-green-600 bg-green-100',
  Intermediate: 'text-yellow-600 bg-yellow-100',
  Advanced: 'text-orange-600 bg-orange-100',
  Expert: 'text-red-600 bg-red-100'
};

export default function ChaptersPage() {
  const router = useRouter();

  const handleChapterClick = (chapterId: number) => {
    router.push(`/chapters/${chapterId}`);
  };

  return (
    <div className="min-h-screen bg-gray-50 py-8">
      <div className="max-w-7xl mx-auto px-4">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="mb-8"
        >
          <h1 className="text-4xl font-bold mb-4">Course Chapters</h1>
          <p className="text-xl text-gray-600">
            Master PPO through structured, interactive lessons
          </p>
        </motion.div>

        {/* Chapters Grid */}
        <div className="grid md:grid-cols-2 gap-6 mb-12">
          {chapters.map((chapter, index) => {
            const Icon = chapter.icon;
            return (
              <motion.div
                key={chapter.id}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: index * 0.1 }}
                data-testid={`chapter-card-${chapter.id}`}
                onClick={() => handleChapterClick(chapter.id)}
                className="bg-white rounded-lg shadow-lg overflow-hidden cursor-pointer hover:shadow-xl transform hover:-translate-y-1 transition-all"
              >
                <div className={`h-2 ${chapter.color}`} />
                
                <div className="p-6">
                  <div className="flex items-start justify-between mb-4">
                    <div className="flex items-center gap-3">
                      <div className={`p-3 rounded-lg ${chapter.color} bg-opacity-20`}>
                        <Icon className={`w-6 h-6 ${chapter.color.replace('bg-', 'text-')}`} />
                      </div>
                      <div>
                        <h3 className="text-xl font-bold">{chapter.title}</h3>
                        <p className="text-gray-600 text-sm mt-1">{chapter.description}</p>
                      </div>
                    </div>
                  </div>

                  <div className="flex items-center gap-4 mb-4 text-sm">
                    <div className="flex items-center gap-1">
                      <Clock className="w-4 h-4 text-gray-400" />
                      <span className="text-gray-600">{chapter.estimatedTime}</span>
                    </div>
                    <div className={`px-2 py-1 rounded-full text-xs font-medium ${difficultyColors[chapter.difficulty]}`}>
                      {chapter.difficulty}
                    </div>
                  </div>

                  <div className="flex flex-wrap gap-2">
                    {chapter.topics.map((topic, topicIndex) => (
                      <span
                        key={topicIndex}
                        className="px-3 py-1 bg-gray-100 rounded-full text-xs text-gray-700"
                      >
                        {topic}
                      </span>
                    ))}
                  </div>
                </div>
              </motion.div>
            );
          })}
        </div>

        {/* Learning Path */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.8 }}
          className="bg-gradient-to-r from-blue-50 to-purple-50 rounded-lg p-8 mb-8"
        >
          <h2 className="text-2xl font-bold mb-6">🎯 Learning Path</h2>
          
          <div className="grid md:grid-cols-3 gap-6">
            <div>
              <h3 className="font-semibold text-lg mb-3">Prerequisites</h3>
              <ul className="space-y-2 text-sm">
                <li className="flex items-center gap-2">
                  <span className="text-green-500">✓</span>
                  Basic calculus and linear algebra
                </li>
                <li className="flex items-center gap-2">
                  <span className="text-green-500">✓</span>
                  Python programming experience
                </li>
              </ul>
            </div>
            
            <div>
              <h3 className="font-semibold text-lg mb-3">Core Concepts</h3>
              <ul className="space-y-2 text-sm">
                <li className="flex items-center gap-2">
                  <span className="text-green-500">✓</span>
                  Markov Decision Processes
                </li>
                <li className="flex items-center gap-2">
                  <span className="text-green-500">✓</span>
                  Value and Policy Methods
                </li>
                <li className="flex items-center gap-2">
                  <span className="text-green-500">✓</span>
                  PPO implementation from scratch
                </li>
              </ul>
            </div>
            
            <div>
              <h3 className="font-semibold text-lg mb-3">Advanced Topics</h3>
              <ul className="space-y-2 text-sm">
                <li className="flex items-center gap-2">
                  <span className="text-green-500">✓</span>
                  RLHF and preference learning
                </li>
                <li className="flex items-center gap-2">
                  <span className="text-green-500">✓</span>
                  Distributed training systems
                </li>
              </ul>
            </div>
          </div>
        </motion.div>

        {/* Quick Links */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.9 }}
          className="bg-white rounded-lg shadow p-6"
        >
          <h2 className="text-xl font-bold mb-4">⚡ Quick Links</h2>
          
          <div className="grid md:grid-cols-3 gap-4">
            <Link
              href="/interactive"
              className="flex items-center gap-3 p-4 bg-gray-50 rounded-lg hover:bg-gray-100 transition-colors"
            >
              <Play className="w-5 h-5 text-blue-500" />
              <span className="font-medium">Interactive Grid World</span>
            </Link>
            
            <Link
              href="/playground"
              className="flex items-center gap-3 p-4 bg-gray-50 rounded-lg hover:bg-gray-100 transition-colors"
            >
              <Cpu className="w-5 h-5 text-green-500" />
              <span className="font-medium">Code Playground</span>
            </Link>
            
            <Link
              href="/training"
              className="flex items-center gap-3 p-4 bg-gray-50 rounded-lg hover:bg-gray-100 transition-colors"
            >
              <BarChart className="w-5 h-5 text-purple-500" />
              <span className="font-medium">Live Training</span>
            </Link>
          </div>
        </motion.div>
      </div>
    </div>
  );
}