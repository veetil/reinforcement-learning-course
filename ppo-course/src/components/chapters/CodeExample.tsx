'use client';

import React, { useState } from 'react';
import { motion } from 'framer-motion';
import { Copy, Check, Play } from 'lucide-react';

interface CodeExampleProps {
  code: string;
  language: string;
  title?: string;
  runnable?: boolean;
  className?: string;
}

export const CodeExample: React.FC<CodeExampleProps> = ({
  code,
  language,
  title,
  runnable = false,
  className = ''
}) => {
  const [copied, setCopied] = useState(false);
  const [isRunning, setIsRunning] = useState(false);
  const [output, setOutput] = useState<string | null>(null);

  const handleCopy = () => {
    navigator.clipboard.writeText(code);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const handleRun = async () => {
    if (!runnable) return;
    
    setIsRunning(true);
    setOutput(null);
    
    // Simulate code execution
    setTimeout(() => {
      setOutput("Output: Neural network initialized successfully!");
      setIsRunning(false);
    }, 1000);
  };

  return (
    <div className={`rounded-lg overflow-hidden ${className}`}>
      <div className="bg-gray-800 text-white px-4 py-2 flex justify-between items-center">
        <div className="flex items-center gap-2">
          <span className="text-sm font-mono">{language}</span>
          {title && <span className="text-sm text-gray-400">• {title}</span>}
        </div>
        <div className="flex items-center gap-2">
          {runnable && (
            <button
              onClick={handleRun}
              disabled={isRunning}
              className="px-3 py-1 bg-green-600 text-white rounded text-sm hover:bg-green-700 disabled:bg-gray-600 flex items-center gap-1"
            >
              <Play size={14} />
              {isRunning ? 'Running...' : 'Run'}
            </button>
          )}
          <button
            onClick={handleCopy}
            className="p-1.5 hover:bg-gray-700 rounded transition-colors"
            title="Copy code"
          >
            {copied ? (
              <Check size={16} className="text-green-400" />
            ) : (
              <Copy size={16} />
            )}
          </button>
        </div>
      </div>
      
      <div className="bg-gray-900 p-4 overflow-x-auto">
        <pre className="text-sm text-gray-300">
          <code>{code}</code>
        </pre>
      </div>
      
      {output && (
        <motion.div
          initial={{ opacity: 0, height: 0 }}
          animate={{ opacity: 1, height: 'auto' }}
          className="bg-gray-950 border-t border-gray-700 p-4"
        >
          <pre className="text-sm text-green-400">{output}</pre>
        </motion.div>
      )}
    </div>
  );
};