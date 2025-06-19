'use client';

import React, { useState, useEffect } from 'react';
import { Paper, PaperMetadata, ArxivRLCrawler } from '@/lib/paper-vault/arxiv-crawler';
import { PaperCard } from './PaperCard';
import { PaperReader } from './PaperReader';
import { CitationGraph } from './CitationGraph';
import { Search, Filter, RefreshCw, TrendingUp, Clock, Star, Network, Grid3X3 } from 'lucide-react';

interface PaperWithMetadata {
  paper: Paper;
  metadata: PaperMetadata;
}

type SortOption = 'quality' | 'date' | 'relevance';
type FilterCategory = 'all' | 'policy-gradient' | 'model-based' | 'offline-rl' | 'multi-agent' | 'rlhf';
type FilterDifficulty = 'all' | 'beginner' | 'intermediate' | 'advanced' | 'expert';

export function PaperVault() {
  const [papers, setPapers] = useState<PaperWithMetadata[]>([]);
  const [loading, setLoading] = useState(false);
  const [selectedPaper, setSelectedPaper] = useState<Paper | null>(null);
  const [searchTerm, setSearchTerm] = useState('');
  const [sortBy, setSortBy] = useState<SortOption>('quality');
  const [categoryFilter, setCategoryFilter] = useState<FilterCategory>('all');
  const [difficultyFilter, setDifficultyFilter] = useState<FilterDifficulty>('all');
  const [showFilters, setShowFilters] = useState(false);
  const [viewMode, setViewMode] = useState<'grid' | 'graph'>('grid');

  const crawler = React.useMemo(() => new ArxivRLCrawler(), []);

  const fetchPapers = async () => {
    setLoading(true);
    try {
      const fetchedPapers = await crawler.fetchNewPapers();
      const papersWithMetadata = fetchedPapers.map(paper => ({
        paper,
        metadata: crawler.extractMetadata(paper)
      }));
      setPapers(papersWithMetadata);
    } catch (error) {
      console.error('Error fetching papers:', error);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    // Load sample papers on mount
    const samplePapers: PaperWithMetadata[] = [
      {
        paper: {
          arxivId: '2307.04964',
          title: 'RLHF: Reinforcement Learning from Human Feedback',
          abstract: 'We present a comprehensive study on training language models using reinforcement learning from human feedback. Our approach significantly improves model alignment with human preferences.',
          authors: [
            { name: 'John Doe', affiliation: 'OpenAI' },
            { name: 'Jane Smith', affiliation: 'Stanford' }
          ],
          publishedDate: new Date('2023-07-10'),
          categories: ['cs.LG', 'cs.AI'],
          pdfUrl: 'https://arxiv.org/pdf/2307.04964.pdf'
        },
        metadata: {
          keyConcepts: ['rlhf', 'policy gradient', 'ppo', 'human feedback'],
          category: 'rlhf',
          difficulty: 'advanced',
          implementationAvailable: true,
          qualityScore: 0.9
        }
      },
      {
        paper: {
          arxivId: '2106.01345',
          title: 'Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning',
          abstract: 'We propose soft actor-critic (SAC), an off-policy actor-critic deep RL algorithm based on the maximum entropy reinforcement learning framework.',
          authors: [
            { name: 'Tuomas Haarnoja', affiliation: 'UC Berkeley' }
          ],
          publishedDate: new Date('2021-06-02'),
          categories: ['cs.LG'],
          pdfUrl: 'https://arxiv.org/pdf/2106.01345.pdf'
        },
        metadata: {
          keyConcepts: ['sac', 'actor-critic', 'off-policy', 'maximum entropy'],
          category: 'policy-gradient',
          difficulty: 'intermediate',
          implementationAvailable: true,
          qualityScore: 0.85
        }
      }
    ];
    setPapers(samplePapers);
  }, []);

  // Filter and sort papers
  const filteredPapers = papers.filter(({ paper, metadata }) => {
    // Search filter
    if (searchTerm) {
      const searchLower = searchTerm.toLowerCase();
      const matchesSearch = 
        paper.title.toLowerCase().includes(searchLower) ||
        paper.abstract.toLowerCase().includes(searchLower) ||
        metadata.keyConcepts.some(c => c.includes(searchLower));
      if (!matchesSearch) return false;
    }

    // Category filter
    if (categoryFilter !== 'all' && metadata.category !== categoryFilter) {
      return false;
    }

    // Difficulty filter
    if (difficultyFilter !== 'all' && metadata.difficulty !== difficultyFilter) {
      return false;
    }

    return true;
  });

  const sortedPapers = [...filteredPapers].sort((a, b) => {
    switch (sortBy) {
      case 'quality':
        return b.metadata.qualityScore - a.metadata.qualityScore;
      case 'date':
        return b.paper.publishedDate.getTime() - a.paper.publishedDate.getTime();
      case 'relevance':
        // Simple relevance based on keyword matches
        const aRelevance = a.metadata.keyConcepts.filter(c => 
          searchTerm.toLowerCase().includes(c)
        ).length;
        const bRelevance = b.metadata.keyConcepts.filter(c => 
          searchTerm.toLowerCase().includes(c)
        ).length;
        return bRelevance - aRelevance;
      default:
        return 0;
    }
  });

  if (selectedPaper) {
    return (
      <PaperReader 
        paper={selectedPaper} 
        onBack={() => setSelectedPaper(null)} 
      />
    );
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <h2 className="text-2xl font-bold">Paper Vault</h2>
        <div className="flex items-center gap-4">
          {/* View Mode Toggle */}
          <div className="flex items-center bg-gray-100 rounded-lg p-1">
            <button
              onClick={() => setViewMode('grid')}
              className={`flex items-center gap-2 px-3 py-1 rounded ${
                viewMode === 'grid' 
                  ? 'bg-white shadow text-gray-900' 
                  : 'text-gray-600 hover:text-gray-900'
              }`}
            >
              <Grid3X3 className="w-4 h-4" />
              Grid
            </button>
            <button
              onClick={() => setViewMode('graph')}
              className={`flex items-center gap-2 px-3 py-1 rounded ${
                viewMode === 'graph' 
                  ? 'bg-white shadow text-gray-900' 
                  : 'text-gray-600 hover:text-gray-900'
              }`}
            >
              <Network className="w-4 h-4" />
              Graph
            </button>
          </div>
          <button
            onClick={fetchPapers}
            disabled={loading}
            className="flex items-center gap-2 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50"
          >
            <RefreshCw className={`w-4 h-4 ${loading ? 'animate-spin' : ''}`} />
            Fetch New Papers
          </button>
        </div>
      </div>

      {/* Search and Filters */}
      <div className="space-y-4">
        <div className="flex gap-4">
          <div className="flex-1 relative">
            <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 w-5 h-5 text-gray-400" />
            <input
              type="text"
              placeholder="Search papers by title, abstract, or concepts..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              className="w-full pl-10 pr-4 py-2 border rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
            />
          </div>
          <button
            onClick={() => setShowFilters(!showFilters)}
            className="flex items-center gap-2 px-4 py-2 border rounded-lg hover:bg-gray-50"
          >
            <Filter className="w-4 h-4" />
            Filters
          </button>
        </div>

        {/* Filter Panel */}
        {showFilters && (
          <div className="p-4 border rounded-lg bg-gray-50 space-y-4">
            <div className="grid grid-cols-3 gap-4">
              {/* Sort By */}
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Sort By
                </label>
                <select
                  value={sortBy}
                  onChange={(e) => setSortBy(e.target.value as SortOption)}
                  className="w-full px-3 py-2 border rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
                >
                  <option value="quality">Quality Score</option>
                  <option value="date">Publication Date</option>
                  <option value="relevance">Relevance</option>
                </select>
              </div>

              {/* Category Filter */}
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Category
                </label>
                <select
                  value={categoryFilter}
                  onChange={(e) => setCategoryFilter(e.target.value as FilterCategory)}
                  className="w-full px-3 py-2 border rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
                >
                  <option value="all">All Categories</option>
                  <option value="policy-gradient">Policy Gradient</option>
                  <option value="model-based">Model-Based</option>
                  <option value="offline-rl">Offline RL</option>
                  <option value="multi-agent">Multi-Agent</option>
                  <option value="rlhf">RLHF</option>
                </select>
              </div>

              {/* Difficulty Filter */}
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Difficulty
                </label>
                <select
                  value={difficultyFilter}
                  onChange={(e) => setDifficultyFilter(e.target.value as FilterDifficulty)}
                  className="w-full px-3 py-2 border rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
                >
                  <option value="all">All Levels</option>
                  <option value="beginner">Beginner</option>
                  <option value="intermediate">Intermediate</option>
                  <option value="advanced">Advanced</option>
                  <option value="expert">Expert</option>
                </select>
              </div>
            </div>

            {/* Quick Stats */}
            <div className="flex gap-4 text-sm text-gray-600">
              <span>{sortedPapers.length} papers found</span>
              <span>•</span>
              <span>{papers.filter(p => p.metadata.implementationAvailable).length} with code</span>
              <span>•</span>
              <span>{papers.filter(p => p.metadata.qualityScore > 0.8).length} high quality</span>
            </div>
          </div>
        )}
      </div>

      {/* Content Area */}
      {loading ? (
        <div className="flex items-center justify-center py-12">
          <RefreshCw className="w-8 h-8 animate-spin text-blue-600" />
        </div>
      ) : sortedPapers.length === 0 ? (
        <div className="text-center py-12 text-gray-500">
          No papers found matching your criteria.
        </div>
      ) : viewMode === 'grid' ? (
        <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-3">
          {sortedPapers.map(({ paper, metadata }) => (
            <PaperCard
              key={paper.arxivId}
              paper={paper}
              metadata={metadata}
              onSelect={setSelectedPaper}
            />
          ))}
        </div>
      ) : (
        <div className="h-[600px] border rounded-lg">
          <CitationGraph 
            papers={sortedPapers.map(p => p.paper)} 
            focusPaperId={selectedPaper?.arxivId}
          />
        </div>
      )}
    </div>
  );
}