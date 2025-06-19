'use client';

import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { AlertTriangle, ChevronDown, ChevronUp, Lightbulb } from 'lucide-react';

interface ConfusionClarifierProps {
  id: string;
  title: string;
  confusion: string;
  clarification: string;
  example?: string;
  type?: 'warning' | 'tip';
}

export const ConfusionClarifier: React.FC<ConfusionClarifierProps> = ({
  id,
  title,
  confusion,
  clarification,
  example,
  type = 'warning'
}) => {
  const [isExpanded, setIsExpanded] = useState(false);

  const icon = type === 'warning' ? AlertTriangle : Lightbulb;
  const bgColor = type === 'warning' ? 'bg-yellow-50' : 'bg-blue-50';
  const borderColor = type === 'warning' ? 'border-yellow-200' : 'border-blue-200';
  const iconColor = type === 'warning' ? 'text-yellow-600' : 'text-blue-600';
  const prefix = type === 'warning' ? '⚠️ Common Confusion Point' : '💡 Pro Tip';

  return (
    <div
      data-testid={`clarifier-${id}`}
      className={`${bgColor} ${borderColor} border rounded-lg p-4 my-4`}
    >
      <button
        onClick={() => setIsExpanded(!isExpanded)}
        className="w-full flex items-center justify-between text-left"
      >
        <div className="flex items-center gap-3">
          {React.createElement(icon, { className: `w-5 h-5 ${iconColor}` })}
          <div>
            <p className="font-semibold text-sm">{prefix}</p>
            <p className="text-gray-800">{title}</p>
          </div>
        </div>
        {isExpanded ? (
          <ChevronUp className="w-5 h-5 text-gray-600" />
        ) : (
          <ChevronDown className="w-5 h-5 text-gray-600" />
        )}
      </button>

      <AnimatePresence>
        {isExpanded && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            exit={{ opacity: 0, height: 0 }}
            transition={{ duration: 0.2 }}
            className="mt-4 space-y-3"
          >
            <div>
              <p className="font-medium text-sm text-gray-700 mb-1">Common Confusion:</p>
              <p className="text-gray-700">{confusion}</p>
            </div>

            <div>
              <p className="font-medium text-sm text-gray-700 mb-1">Clarification:</p>
              <p className="text-gray-700">{clarification}</p>
            </div>

            {example && (
              <div className="bg-white rounded p-3">
                <p className="font-medium text-sm text-gray-700 mb-1">Example:</p>
                <pre className="text-sm text-gray-700 whitespace-pre-wrap">{example}</pre>
              </div>
            )}
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
};