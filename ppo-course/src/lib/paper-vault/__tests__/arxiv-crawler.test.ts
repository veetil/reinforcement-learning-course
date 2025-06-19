import { ArxivRLCrawler, Paper, PaperMetadata } from '../arxiv-crawler';
import { describe, test, expect, beforeEach, jest } from '@jest/globals';

// Mock fetch
global.fetch = jest.fn();

describe('ArxivRLCrawler', () => {
  let crawler: ArxivRLCrawler;

  beforeEach(() => {
    jest.clearAllMocks();
    crawler = new ArxivRLCrawler();
  });

  describe('Paper Fetching', () => {
    test('fetches papers from multiple categories', async () => {
      const mockResponse = {
        ok: true,
        text: jest.fn().mockResolvedValue(`
          <feed xmlns="http://www.w3.org/2005/Atom">
            <entry>
              <id>http://arxiv.org/abs/2024.01234</id>
              <title>Proximal Policy Optimization with Group Normalization</title>
              <summary>We propose GRPO, an enhancement to PPO...</summary>
              <author><name>John Doe</name></author>
              <author><name>Jane Smith</name></author>
              <published>2024-01-15T00:00:00Z</published>
              <arxiv:primary_category term="cs.LG"/>
            </entry>
          </feed>
        `),
      };

      (global.fetch as jest.Mock).mockResolvedValue(mockResponse);

      const papers = await crawler.fetchNewPapers();
      
      expect(papers.length).toBeGreaterThan(0);
      expect(papers[0].title).toContain('Proximal Policy Optimization');
      expect(papers[0].authors).toHaveLength(2);
    });

    test('filters papers by RL keywords', async () => {
      const mockPapers = [
        createMockPaper('Deep Learning for Image Recognition', 'CNN architectures...'),
        createMockPaper('PPO for Robotics', 'We apply PPO to robotic control...'),
        createMockPaper('Reinforcement Learning Survey', 'A comprehensive survey of RL...'),
      ];

      const filtered = crawler.filterRLPapers(mockPapers);
      
      expect(filtered).toHaveLength(2);
      expect(filtered[0].title).toContain('PPO');
      expect(filtered[1].title).toContain('Reinforcement Learning');
    });

    test('assesses paper quality based on multiple factors', () => {
      const highQualityPaper = createMockPaper(
        'RLHF: A Breakthrough in Language Model Alignment',
        'We present a novel approach to RLHF that significantly improves...',
        ['OpenAI Research', 'Stanford University'],
        50
      );

      const lowQualityPaper = createMockPaper(
        'Yet Another PPO Implementation',
        'This is my implementation of PPO...',
        ['Unknown University'],
        0
      );

      const highScore = crawler.assessQuality(highQualityPaper);
      const lowScore = crawler.assessQuality(lowQualityPaper);

      expect(highScore).toBeGreaterThan(0.7);
      expect(lowScore).toBeLessThan(0.5);
      expect(highScore).toBeGreaterThan(lowScore);
    });
  });

  describe('Metadata Extraction', () => {
    test('extracts key concepts from abstract', () => {
      const paper = createMockPaper(
        'PPO with Adaptive KL Penalty',
        'We propose an adaptive KL divergence penalty for PPO that improves sample efficiency. Our method uses GAE for advantage estimation and achieves state-of-the-art results on MuJoCo benchmarks.'
      );

      const metadata = crawler.extractMetadata(paper);
      
      expect(metadata.keyConcepts).toContain('PPO');
      expect(metadata.keyConcepts).toContain('KL divergence');
      expect(metadata.keyConcepts).toContain('GAE');
      expect(metadata.keyConcepts).toContain('sample efficiency');
    });

    test('identifies paper category', () => {
      const papers = [
        { 
          paper: createMockPaper('SAC: Off-Policy Maximum Entropy RL', 'Soft Actor-Critic...'),
          expectedCategory: 'policy-gradient'
        },
        {
          paper: createMockPaper('Model-Based Planning with MuZero', 'We present MuZero...'),
          expectedCategory: 'model-based'
        },
        {
          paper: createMockPaper('Offline RL with Conservative Q-Learning', 'CQL prevents...'),
          expectedCategory: 'offline-rl'
        },
      ];

      papers.forEach(({ paper, expectedCategory }) => {
        const metadata = crawler.extractMetadata(paper);
        expect(metadata.category).toBe(expectedCategory);
      });
    });

    test('calculates citation impact score', async () => {
      const paper = createMockPaper(
        'Attention is All You Need',
        'Transformers for RL...',
        ['Google Research'],
        1000
      );

      const impactScore = await crawler.calculateImpactScore(paper);
      
      expect(impactScore).toBeGreaterThan(0.9);
    });
  });

  describe('Batch Processing', () => {
    test('processes papers in batches to avoid rate limits', async () => {
      const papers = Array(50).fill(null).map((_, i) => 
        createMockPaper(`Paper ${i}`, `Abstract ${i}`)
      );

      const startTime = Date.now();
      await crawler.processPaperBatch(papers);
      const duration = Date.now() - startTime;

      // Should have delays between batches
      expect(duration).toBeGreaterThan(100);
      expect(crawler.processedCount).toBe(50);
    });

    test('handles API errors gracefully', async () => {
      (global.fetch as jest.Mock)
        .mockRejectedValueOnce(new Error('API rate limit'))
        .mockResolvedValueOnce({
          ok: true,
          text: jest.fn().mockResolvedValue('<feed></feed>'),
        });

      const papers = await crawler.fetchNewPapers();
      
      // Should retry and eventually succeed
      expect(papers).toBeDefined();
      expect(global.fetch).toHaveBeenCalledTimes(2);
    });
  });

  describe('Deduplication', () => {
    test('removes duplicate papers by arxiv ID', () => {
      const papers = [
        createMockPaper('PPO Paper', 'Abstract 1', [], 0, '2024.1234'),
        createMockPaper('PPO Paper Updated', 'Abstract 2', [], 0, '2024.1234'),
        createMockPaper('Different Paper', 'Abstract 3', [], 0, '2024.5678'),
      ];

      const deduplicated = crawler.deduplicatePapers(papers);
      
      expect(deduplicated).toHaveLength(2);
      expect(deduplicated[0].arxivId).toBe('2024.1234');
      expect(deduplicated[1].arxivId).toBe('2024.5678');
    });
  });
});

// Helper function to create mock papers
function createMockPaper(
  title: string,
  abstract: string,
  affiliations: string[] = ['Test University'],
  citations: number = 0,
  arxivId: string = Math.random().toString()
): Paper {
  return {
    arxivId,
    title,
    abstract,
    authors: affiliations.map((aff, i) => ({
      name: `Author ${i}`,
      affiliation: aff,
    })),
    publishedDate: new Date(),
    categories: ['cs.LG'],
    pdfUrl: `https://arxiv.org/pdf/${arxivId}.pdf`,
    citations,
  };
}