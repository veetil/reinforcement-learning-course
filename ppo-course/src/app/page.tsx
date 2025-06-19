'use client'

import React from 'react'
import Link from 'next/link'
import { motion } from 'framer-motion'
import { ArrowRight, Brain, Zap, Users, Award, Sparkles } from 'lucide-react'
import NeuralNetworkVisualizer from '@/components/visualization/NeuralNetworkVisualizer'

const features = [
  {
    icon: Brain,
    title: 'Interactive Visualizations',
    description: 'See neural networks come to life with real-time animations and interactive controls'
  },
  {
    icon: Zap,
    title: 'Learn by Doing',
    description: 'Build your own PPO implementation with guided exercises and instant feedback'
  },
  {
    icon: Users,
    title: 'Community Learning',
    description: 'Collaborate with peers, share insights, and learn together'
  },
  {
    icon: Award,
    title: 'Industry Certification',
    description: 'Earn a recognized certificate upon successful completion'
  }
]

export default function HomePage() {
  return (
    <div className="relative">
      {/* Hero Section */}
      <section className="relative overflow-hidden bg-gradient-to-br from-blue-50 to-indigo-100 py-20">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-12 items-center">
            <motion.div
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.5 }}
            >
              <h1 className="text-4xl sm:text-5xl font-bold text-gray-900 mb-6">
                Master <span className="text-blue-600">PPO Algorithm</span> Through Interactive Learning
              </h1>
              <p className="text-xl text-gray-600 mb-8">
                Learn Proximal Policy Optimization with hands-on visualizations, 
                practical coding exercises, and real-world applications. 
                No more passive video watching - build real understanding.
              </p>
              <div className="flex flex-col sm:flex-row gap-4">
                <Link
                  href="/chapters/1"
                  className="inline-flex items-center justify-center px-6 py-3 bg-blue-600 text-white font-medium rounded-lg hover:bg-blue-700 transition-colors"
                >
                  Start Learning
                  <ArrowRight className="ml-2" size={20} />
                </Link>
                <Link
                  href="/playground"
                  className="inline-flex items-center justify-center px-6 py-3 bg-white text-gray-700 font-medium rounded-lg border border-gray-300 hover:bg-gray-50 transition-colors"
                >
                  Try Playground
                </Link>
                <Link
                  href="/algorithms"
                  className="inline-flex items-center justify-center px-6 py-3 bg-purple-600 text-white font-medium rounded-lg hover:bg-purple-700 transition-colors"
                >
                  <Sparkles className="mr-2" size={20} />
                  Algorithm Zoo
                </Link>
              </div>
              <div className="flex flex-col sm:flex-row gap-4 mt-4">
                <Link
                  href="/benchmarks"
                  className="inline-flex items-center justify-center px-6 py-3 bg-yellow-600 text-white font-medium rounded-lg hover:bg-yellow-700 transition-colors"
                >
                  <Award className="mr-2" size={20} />
                  Benchmark Suite
                </Link>
              </div>
            </motion.div>

            <motion.div
              initial={{ opacity: 0, scale: 0.9 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ duration: 0.5, delay: 0.2 }}
              className="relative h-[400px] lg:h-[500px]"
            >
              <div className="absolute inset-0 bg-white rounded-lg shadow-xl p-4">
                <NeuralNetworkVisualizer
                  layers={[
                    { neurons: 3, type: 'input', label: 'State' },
                    { neurons: 4, type: 'hidden', label: 'Hidden' },
                    { neurons: 2, type: 'output', label: 'Action' }
                  ]}
                  activations={[
                    [0.8, 0.3, 0.6],
                    [0.9, 0.2, 0.7, 0.5],
                    [0.6, 0.4]
                  ]}
                  animated={true}
                  showValues={true}
                />
              </div>
            </motion.div>
          </div>
        </div>
      </section>

      {/* Features Section */}
      <section className="py-20 bg-white">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5 }}
            viewport={{ once: true }}
            className="text-center mb-12"
          >
            <h2 className="text-3xl font-bold text-gray-900 mb-4">
              Why This Course is Different
            </h2>
            <p className="text-xl text-gray-600 max-w-3xl mx-auto">
              We believe in learning by doing. Every concept is paired with 
              interactive visualizations and hands-on exercises.
            </p>
          </motion.div>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-8">
            {features.map((feature, index) => {
              const Icon = feature.icon
              return (
                <motion.div
                  key={feature.title}
                  initial={{ opacity: 0, y: 20 }}
                  whileInView={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.5, delay: index * 0.1 }}
                  viewport={{ once: true }}
                  className="bg-gray-50 rounded-lg p-6 hover:shadow-lg transition-shadow"
                >
                  <div className="w-12 h-12 bg-blue-100 rounded-lg flex items-center justify-center mb-4">
                    <Icon className="text-blue-600" size={24} />
                  </div>
                  <h3 className="text-lg font-semibold text-gray-900 mb-2">
                    {feature.title}
                  </h3>
                  <p className="text-gray-600">
                    {feature.description}
                  </p>
                </motion.div>
              )
            })}
          </div>
        </div>
      </section>

      {/* Course Structure */}
      <section className="py-20 bg-gray-50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5 }}
            viewport={{ once: true }}
            className="text-center mb-12"
          >
            <h2 className="text-3xl font-bold text-gray-900 mb-4">
              Course Structure
            </h2>
            <p className="text-xl text-gray-600 max-w-3xl mx-auto">
              14 comprehensive chapters taking you from RL basics to production-ready PPO systems
            </p>
          </motion.div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
            <motion.div
              initial={{ opacity: 0, x: -20 }}
              whileInView={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.5 }}
              viewport={{ once: true }}
              className="bg-white rounded-lg shadow-lg p-6"
            >
              <div className="text-blue-600 font-semibold mb-2">Phase 1</div>
              <h3 className="text-xl font-bold text-gray-900 mb-4">
                Foundation & Core Concepts
              </h3>
              <ul className="space-y-2 text-gray-600">
                <li>• Introduction to Reinforcement Learning</li>
                <li>• Value Functions and Critics</li>
                <li>• Actor-Critic Architecture</li>
                <li>• Introduction to PPO</li>
              </ul>
            </motion.div>

            <motion.div
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5, delay: 0.1 }}
              viewport={{ once: true }}
              className="bg-white rounded-lg shadow-lg p-6"
            >
              <div className="text-blue-600 font-semibold mb-2">Phase 2</div>
              <h3 className="text-xl font-bold text-gray-900 mb-4">
                PPO Deep Dive
              </h3>
              <ul className="space-y-2 text-gray-600">
                <li>• PPO Objective Function</li>
                <li>• Advantage Estimation</li>
                <li>• Implementation Architecture</li>
                <li>• Mini-batch Training</li>
                <li>• PPO for Language Models</li>
              </ul>
            </motion.div>

            <motion.div
              initial={{ opacity: 0, x: 20 }}
              whileInView={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.5, delay: 0.2 }}
              viewport={{ once: true }}
              className="bg-white rounded-lg shadow-lg p-6"
            >
              <div className="text-blue-600 font-semibold mb-2">Phase 3</div>
              <h3 className="text-xl font-bold text-gray-900 mb-4">
                Advanced Applications
              </h3>
              <ul className="space-y-2 text-gray-600">
                <li>• Scaling PPO Systems</li>
                <li>• Advanced Reward Modeling</li>
                <li>• Complex Domains</li>
                <li>• Production Deployment</li>
                <li>• PPO Variants</li>
              </ul>
            </motion.div>
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="py-20 bg-blue-600">
        <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 text-center">
          <motion.div
            initial={{ opacity: 0, scale: 0.9 }}
            whileInView={{ opacity: 1, scale: 1 }}
            transition={{ duration: 0.5 }}
            viewport={{ once: true }}
          >
            <h2 className="text-3xl font-bold text-white mb-4">
              Ready to Master PPO?
            </h2>
            <p className="text-xl text-blue-100 mb-8">
              Join thousands of learners who have transformed their understanding of reinforcement learning
            </p>
            <Link
              href="/chapters/1"
              className="inline-flex items-center px-8 py-4 bg-white text-blue-600 font-semibold rounded-lg hover:bg-gray-100 transition-colors"
            >
              Start Your Journey
              <ArrowRight className="ml-2" size={20} />
            </Link>
          </motion.div>
        </div>
      </section>
    </div>
  )
}