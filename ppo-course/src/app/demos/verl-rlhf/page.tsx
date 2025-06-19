'use client';

import React, { useState } from 'react';
import { VERLSystemVisualizer } from '@/components/visualization/VERLSystemVisualizer';
import { RLHFPipelineVisualizer } from '@/components/visualization/RLHFPipelineVisualizer';
import { BradleyTerryCalculator } from '@/components/interactive/BradleyTerryCalculator';
import { motion } from 'framer-motion';
import { ChevronDown, ChevronUp, BookOpen, Cpu, Calculator } from 'lucide-react';

interface Section {
  id: string;
  title: string;
  description: string;
  icon: React.ReactNode;
  component: React.ReactNode;
}

const sections: Section[] = [
  {
    id: 'verl-system',
    title: 'VERL Distributed System Architecture',
    description: 'Explore how VERL separates workers and manages GPU resources for distributed RL training',
    icon: <Cpu className="w-5 h-5" />,
    component: <VERLSystemVisualizer className="h-[600px]" />,
  },
  {
    id: 'rlhf-pipeline',
    title: 'RLHF Training Pipeline',
    description: 'Watch data flow through the three stages of RLHF and discover the critical implementation issue',
    icon: <BookOpen className="w-5 h-5" />,
    component: <RLHFPipelineVisualizer />,
  },
  {
    id: 'bradley-terry',
    title: 'Bradley-Terry Model Calculator',
    description: 'Understand how preference learning works with interactive probability calculations',
    icon: <Calculator className="w-5 h-5" />,
    component: <BradleyTerryCalculator />,
  },
];

export default function VERLRLHFDemoPage() {
  const [expandedSections, setExpandedSections] = useState<Set<string>>(
    new Set(['verl-system']) // Start with first section expanded
  );

  const toggleSection = (sectionId: string) => {
    setExpandedSections((prev) => {
      const newSet = new Set(prev);
      if (newSet.has(sectionId)) {
        newSet.delete(sectionId);
      } else {
        newSet.add(sectionId);
      }
      return newSet;
    });
  };

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <div className="bg-white shadow-sm border-b">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
          <h1 className="text-3xl font-bold text-gray-900">
            VERL & RLHF Interactive Demos
          </h1>
          <p className="mt-2 text-lg text-gray-600">
            Explore distributed RL systems and understand RLHF implementation details
          </p>
        </div>
      </div>

      {/* Main Content */}
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <div className="space-y-6">
          {sections.map((section, index) => (
            <motion.div
              key={section.id}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: index * 0.1 }}
              className="bg-white rounded-lg shadow-lg overflow-hidden"
            >
              {/* Section Header */}
              <button
                onClick={() => toggleSection(section.id)}
                className="w-full px-6 py-4 flex items-center justify-between hover:bg-gray-50 transition-colors"
              >
                <div className="flex items-center space-x-3">
                  <div className="p-2 bg-blue-100 rounded-lg text-blue-600">
                    {section.icon}
                  </div>
                  <div className="text-left">
                    <h2 className="text-xl font-semibold text-gray-900">
                      {section.title}
                    </h2>
                    <p className="text-sm text-gray-600 mt-1">
                      {section.description}
                    </p>
                  </div>
                </div>
                <div className="text-gray-400">
                  {expandedSections.has(section.id) ? (
                    <ChevronUp className="w-6 h-6" />
                  ) : (
                    <ChevronDown className="w-6 h-6" />
                  )}
                </div>
              </button>

              {/* Section Content */}
              <motion.div
                initial={false}
                animate={{
                  height: expandedSections.has(section.id) ? 'auto' : 0,
                  opacity: expandedSections.has(section.id) ? 1 : 0,
                }}
                transition={{ duration: 0.3 }}
                className="overflow-hidden"
              >
                <div className="p-6 border-t">{section.component}</div>
              </motion.div>
            </motion.div>
          ))}
        </div>

        {/* Learning Path */}
        <div className="mt-12 bg-blue-50 rounded-lg p-6 border-2 border-blue-200">
          <h3 className="text-lg font-semibold mb-3">Suggested Learning Path</h3>
          <ol className="space-y-2 text-sm">
            <li>1. Start with <strong>VERL System Architecture</strong> to understand distributed training</li>
            <li>2. Explore the <strong>RLHF Pipeline</strong> to see the complete training process</li>
            <li>3. Use the <strong>Bradley-Terry Calculator</strong> to understand preference learning math</li>
            <li>4. Try different configurations and observe how they affect performance</li>
          </ol>
        </div>

        {/* Key Takeaways */}
        <div className="mt-8 bg-yellow-50 rounded-lg p-6 border-2 border-yellow-200">
          <h3 className="text-lg font-semibold mb-3">Key Takeaways</h3>
          <ul className="space-y-2 text-sm">
            <li>• VERL uses a HybridFlow architecture to separate control and computation</li>
            <li>• Worker placement strategies significantly impact training performance</li>
            <li>• The Bradley-Terry model is essential for learning human preferences</li>
            <li>• VERL's current reward model implementation has a critical bug that prevents proper training</li>
            <li>• Understanding these systems is crucial for implementing production RLHF</li>
          </ul>
        </div>
      </div>
    </div>
  );
}