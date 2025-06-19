'use client';

import React, { useState } from 'react';
import { Paper } from '@/lib/paper-vault/arxiv-crawler';
import { 
  ArrowLeft, Download, ExternalLink, Copy, CheckCircle, 
  FileText, Code, BookOpen, Lightbulb 
} from 'lucide-react';

interface PaperReaderProps {
  paper: Paper;
  onBack: () => void;
}

export function PaperReader({ paper, onBack }: PaperReaderProps) {
  const [activeTab, setActiveTab] = useState<'summary' | 'implementation' | 'notes'>('summary');
  const [copied, setCopied] = useState(false);
  const [notes, setNotes] = useState('');

  const copyBibtex = () => {
    const bibtex = `@article{${paper.arxivId},
  title={${paper.title}},
  author={${paper.authors.map(a => a.name).join(' and ')}},
  journal={arXiv preprint arXiv:${paper.arxivId}},
  year={${new Date(paper.publishedDate).getFullYear()}}
}`;
    navigator.clipboard.writeText(bibtex);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const generateImplementationTemplate = () => {
    return `"""
Implementation of: ${paper.title}
ArXiv: ${paper.arxivId}
Authors: ${paper.authors.map(a => a.name).join(', ')}

Key Concepts:
${paper.abstract.split('.').slice(0, 2).join('.')}
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class ${paper.title.split(' ')[0].replace(/[^a-zA-Z]/g, '')}Model(nn.Module):
    """
    Implementation of the model described in the paper.
    """
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        # TODO: Implement architecture from paper
        
    def forward(self, x):
        # TODO: Implement forward pass
        pass

class ${paper.title.split(' ')[0].replace(/[^a-zA-Z]/g, '')}Agent:
    """
    Agent implementation based on the paper's algorithm.
    """
    def __init__(self, env, model, config):
        self.env = env
        self.model = model
        self.config = config
        
    def train(self, num_episodes):
        # TODO: Implement training loop from paper
        pass
        
    def act(self, state):
        # TODO: Implement action selection
        pass

# Example usage
if __name__ == "__main__":
    # TODO: Set up environment and hyperparameters
    # TODO: Initialize model and agent
    # TODO: Run training
    pass
`;
  };

  return (
    <div className="max-w-4xl mx-auto space-y-6">
      {/* Header */}
      <div className="flex items-center gap-4">
        <button
          onClick={onBack}
          className="flex items-center gap-2 px-3 py-2 text-gray-600 hover:text-gray-900"
        >
          <ArrowLeft className="w-4 h-4" />
          Back to Papers
        </button>
      </div>

      {/* Paper Info */}
      <div className="bg-white border rounded-lg p-6 space-y-4">
        <h1 className="text-2xl font-bold text-gray-900">{paper.title}</h1>
        
        <div className="flex flex-wrap gap-4 text-sm text-gray-600">
          <span>{paper.authors.map(a => a.name).join(', ')}</span>
          <span>•</span>
          <span>{new Date(paper.publishedDate).toLocaleDateString()}</span>
          <span>•</span>
          <span>arXiv:{paper.arxivId}</span>
        </div>

        <div className="flex gap-4">
          <a
            href={paper.pdfUrl}
            target="_blank"
            rel="noopener noreferrer"
            className="flex items-center gap-2 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700"
          >
            <ExternalLink className="w-4 h-4" />
            View PDF
          </a>
          <button
            onClick={copyBibtex}
            className="flex items-center gap-2 px-4 py-2 border rounded-lg hover:bg-gray-50"
          >
            {copied ? <CheckCircle className="w-4 h-4 text-green-600" /> : <Copy className="w-4 h-4" />}
            {copied ? 'Copied!' : 'Copy BibTeX'}
          </button>
        </div>
      </div>

      {/* Tabs */}
      <div className="border-b">
        <div className="flex gap-6">
          <button
            onClick={() => setActiveTab('summary')}
            className={`pb-2 px-1 border-b-2 transition-colors ${
              activeTab === 'summary' 
                ? 'border-blue-600 text-blue-600' 
                : 'border-transparent text-gray-600 hover:text-gray-900'
            }`}
          >
            <BookOpen className="w-4 h-4 inline mr-2" />
            Summary
          </button>
          <button
            onClick={() => setActiveTab('implementation')}
            className={`pb-2 px-1 border-b-2 transition-colors ${
              activeTab === 'implementation' 
                ? 'border-blue-600 text-blue-600' 
                : 'border-transparent text-gray-600 hover:text-gray-900'
            }`}
          >
            <Code className="w-4 h-4 inline mr-2" />
            Implementation
          </button>
          <button
            onClick={() => setActiveTab('notes')}
            className={`pb-2 px-1 border-b-2 transition-colors ${
              activeTab === 'notes' 
                ? 'border-blue-600 text-blue-600' 
                : 'border-transparent text-gray-600 hover:text-gray-900'
            }`}
          >
            <FileText className="w-4 h-4 inline mr-2" />
            Notes
          </button>
        </div>
      </div>

      {/* Tab Content */}
      <div className="bg-white border rounded-lg p-6">
        {activeTab === 'summary' && (
          <div className="space-y-6">
            <div>
              <h3 className="text-lg font-semibold mb-3">Abstract</h3>
              <p className="text-gray-700 leading-relaxed">{paper.abstract}</p>
            </div>

            <div>
              <h3 className="text-lg font-semibold mb-3">Key Takeaways</h3>
              <div className="space-y-3">
                <div className="flex gap-3">
                  <Lightbulb className="w-5 h-5 text-yellow-500 flex-shrink-0 mt-0.5" />
                  <div>
                    <p className="font-medium">Main Contribution</p>
                    <p className="text-gray-600 text-sm">
                      {paper.abstract.split('.')[0]}.
                    </p>
                  </div>
                </div>
                <div className="flex gap-3">
                  <Lightbulb className="w-5 h-5 text-yellow-500 flex-shrink-0 mt-0.5" />
                  <div>
                    <p className="font-medium">Technical Approach</p>
                    <p className="text-gray-600 text-sm">
                      {paper.abstract.split('.')[1] || 'See paper for details'}.
                    </p>
                  </div>
                </div>
              </div>
            </div>

            <div>
              <h3 className="text-lg font-semibold mb-3">Categories</h3>
              <div className="flex flex-wrap gap-2">
                {paper.categories.map((cat, i) => (
                  <span key={i} className="px-3 py-1 bg-gray-100 text-gray-700 rounded-full text-sm">
                    {cat}
                  </span>
                ))}
              </div>
            </div>
          </div>
        )}

        {activeTab === 'implementation' && (
          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <h3 className="text-lg font-semibold">Implementation Template</h3>
              <button
                onClick={() => {
                  const template = generateImplementationTemplate();
                  navigator.clipboard.writeText(template);
                }}
                className="flex items-center gap-2 px-3 py-1 text-sm bg-gray-100 hover:bg-gray-200 rounded"
              >
                <Copy className="w-4 h-4" />
                Copy Template
              </button>
            </div>
            <pre className="bg-gray-50 p-4 rounded-lg overflow-x-auto text-sm">
              <code>{generateImplementationTemplate()}</code>
            </pre>
          </div>
        )}

        {activeTab === 'notes' && (
          <div className="space-y-4">
            <h3 className="text-lg font-semibold">Your Notes</h3>
            <textarea
              value={notes}
              onChange={(e) => setNotes(e.target.value)}
              placeholder="Add your notes about this paper..."
              className="w-full h-64 p-4 border rounded-lg resize-none focus:outline-none focus:ring-2 focus:ring-blue-500"
            />
            <p className="text-sm text-gray-500">
              Notes are stored locally in your browser.
            </p>
          </div>
        )}
      </div>
    </div>
  );
}