'use client';

import React from 'react';
import { Paper, PaperMetadata } from '@/lib/paper-vault/arxiv-crawler';
import { Calendar, Users, Tag, FileText, Star, ExternalLink } from 'lucide-react';

interface PaperCardProps {
  paper: Paper;
  metadata: PaperMetadata;
  onSelect?: (paper: Paper) => void;
}

export function PaperCard({ paper, metadata, onSelect }: PaperCardProps) {
  const getDifficultyColor = (difficulty: string) => {
    switch (difficulty) {
      case 'beginner': return 'text-green-600 bg-green-100';
      case 'intermediate': return 'text-blue-600 bg-blue-100';
      case 'advanced': return 'text-orange-600 bg-orange-100';
      case 'expert': return 'text-red-600 bg-red-100';
      default: return 'text-gray-600 bg-gray-100';
    }
  };

  const getCategoryColor = (category: string) => {
    switch (category) {
      case 'policy-gradient': return 'text-purple-600 bg-purple-100';
      case 'model-based': return 'text-indigo-600 bg-indigo-100';
      case 'offline-rl': return 'text-pink-600 bg-pink-100';
      case 'multi-agent': return 'text-teal-600 bg-teal-100';
      case 'rlhf': return 'text-amber-600 bg-amber-100';
      default: return 'text-gray-600 bg-gray-100';
    }
  };

  const formatDate = (date: Date) => {
    return new Date(date).toLocaleDateString('en-US', {
      year: 'numeric',
      month: 'short',
      day: 'numeric'
    });
  };

  const qualityStars = Math.round(metadata.qualityScore * 5);

  return (
    <div 
      className="border rounded-lg p-6 hover:shadow-lg transition-shadow cursor-pointer bg-white"
      onClick={() => onSelect?.(paper)}
    >
      <div className="space-y-4">
        {/* Title and Quality */}
        <div>
          <div className="flex items-start justify-between gap-4">
            <h3 className="text-lg font-semibold text-gray-900 line-clamp-2">
              {paper.title}
            </h3>
            <div className="flex items-center gap-1 flex-shrink-0">
              {[...Array(5)].map((_, i) => (
                <Star 
                  key={i} 
                  className={`w-4 h-4 ${i < qualityStars ? 'text-yellow-500 fill-current' : 'text-gray-300'}`} 
                />
              ))}
            </div>
          </div>
        </div>

        {/* Authors */}
        <div className="flex items-center gap-2 text-sm text-gray-600">
          <Users className="w-4 h-4" />
          <span className="line-clamp-1">
            {paper.authors.map(a => a.name).join(', ')}
          </span>
        </div>

        {/* Abstract */}
        <p className="text-sm text-gray-700 line-clamp-3">
          {paper.abstract}
        </p>

        {/* Metadata Pills */}
        <div className="flex flex-wrap gap-2">
          <span className={`px-2 py-1 text-xs font-medium rounded-full ${getDifficultyColor(metadata.difficulty)}`}>
            {metadata.difficulty}
          </span>
          <span className={`px-2 py-1 text-xs font-medium rounded-full ${getCategoryColor(metadata.category)}`}>
            {metadata.category}
          </span>
          {metadata.implementationAvailable && (
            <span className="px-2 py-1 text-xs font-medium rounded-full text-green-600 bg-green-100">
              <FileText className="w-3 h-3 inline mr-1" />
              Code Available
            </span>
          )}
        </div>

        {/* Key Concepts */}
        <div className="flex flex-wrap gap-1">
          {metadata.keyConcepts.slice(0, 5).map((concept, i) => (
            <span key={i} className="text-xs px-2 py-1 bg-gray-100 text-gray-700 rounded">
              {concept}
            </span>
          ))}
          {metadata.keyConcepts.length > 5 && (
            <span className="text-xs px-2 py-1 text-gray-500">
              +{metadata.keyConcepts.length - 5} more
            </span>
          )}
        </div>

        {/* Footer */}
        <div className="flex items-center justify-between text-sm text-gray-500 pt-2 border-t">
          <div className="flex items-center gap-1">
            <Calendar className="w-4 h-4" />
            <span>{formatDate(paper.publishedDate)}</span>
          </div>
          <a 
            href={paper.pdfUrl}
            target="_blank"
            rel="noopener noreferrer"
            className="flex items-center gap-1 text-blue-600 hover:text-blue-800"
            onClick={(e) => e.stopPropagation()}
          >
            <ExternalLink className="w-4 h-4" />
            PDF
          </a>
        </div>
      </div>
    </div>
  );
}