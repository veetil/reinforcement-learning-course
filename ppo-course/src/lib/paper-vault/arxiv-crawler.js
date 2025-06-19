class ArxivRLCrawler {
  constructor() {
    this.categories = ['cs.LG', 'cs.AI', 'stat.ML'];
    this.rlKeywords = [
      'reinforcement learning', 'policy gradient', 'q-learning',
      'actor-critic', 'PPO', 'SAC', 'offline RL', 'RLHF',
      'multi-agent', 'model-based RL', 'meta-RL', 'DQN',
      'TRPO', 'TD3', 'reward', 'MDP', 'bellman', 'value function',
      'advantage', 'policy optimization', 'exploration'
    ];
    
    this.qualityThreshold = 0.7;
    this.batchSize = 10;
    this.rateLimitDelay = 3000; // 3 seconds between API calls
    this.processedCount = 0;
  }

  async fetchNewPapers() {
    const allPapers = [];
    
    for (const category of this.categories) {
      try {
        const papers = await this.fetchCategoryPapers(category);
        allPapers.push(...papers);
        await this.delay(this.rateLimitDelay);
      } catch (error) {
        console.error(`Error fetching ${category}:`, error);
        // Retry once
        await this.delay(this.rateLimitDelay);
        try {
          const papers = await this.fetchCategoryPapers(category);
          allPapers.push(...papers);
        } catch (retryError) {
          console.error(`Retry failed for ${category}:`, retryError);
        }
      }
    }
    
    const rlPapers = this.filterRLPapers(allPapers);
    const qualityPapers = rlPapers.filter(paper => 
      this.assessQuality(paper) > this.qualityThreshold
    );
    
    return this.deduplicatePapers(qualityPapers);
  }

  async fetchCategoryPapers(category) {
    const url = `http://export.arxiv.org/api/query?search_query=cat:${category}&sortBy=submittedDate&sortOrder=descending&max_results=100`;
    
    const response = await fetch(url);
    const text = await response.text();
    
    return this.parseArxivResponse(text);
  }

  parseArxivResponse(xmlText) {
    const papers = [];
    
    // Simple XML parsing (in production, use a proper XML parser)
    const entries = xmlText.match(/<entry>[\s\S]*?<\/entry>/g) || [];
    
    for (const entry of entries) {
      const idMatch = entry.match(/<id>(.*?)<\/id>/);
      const titleMatch = entry.match(/<title>(.*?)<\/title>/);
      const summaryMatch = entry.match(/<summary>(.*?)<\/summary>/);
      const publishedMatch = entry.match(/<published>(.*?)<\/published>/);
      
      if (idMatch && titleMatch && summaryMatch) {
        const arxivId = idMatch[1].split('/').pop() || '';
        const authorMatches = entry.matchAll(/<author>[\s\S]*?<\/author>/g);
        const authors = [];
        
        for (const authorMatch of authorMatches) {
          const authorStr = authorMatch[0];
          const nameMatch = authorStr.match(/<name>(.*?)<\/name>/);
          const affiliationMatch = authorStr.match(/<affiliation>(.*?)<\/affiliation>/);
          
          if (nameMatch) {
            authors.push({
              name: nameMatch[1].trim(),
              affiliation: affiliationMatch ? affiliationMatch[1].trim() : undefined
            });
          }
        }
        
        papers.push({
          arxivId,
          title: titleMatch[1].trim(),
          abstract: summaryMatch[1].trim(),
          authors,
          publishedDate: publishedMatch ? new Date(publishedMatch[1]) : new Date(),
          categories: this.extractCategories(entry),
          pdfUrl: `https://arxiv.org/pdf/${arxivId}.pdf`,
        });
      }
    }
    
    return papers;
  }

  extractCategories(entry) {
    const categoryMatches = entry.matchAll(/term="(.*?)"/g);
    return Array.from(categoryMatches).map(match => match[1]);
  }

  filterRLPapers(papers) {
    return papers.filter(paper => {
      const text = `${paper.title} ${paper.abstract}`.toLowerCase();
      return this.rlKeywords.some(keyword => text.includes(keyword.toLowerCase()));
    });
  }

  assessQuality(paper) {
    let score = 0;
    
    // Author reputation (simplified - check for known institutions)
    const prestigiousInstitutions = [
      'openai', 'deepmind', 'google', 'stanford', 'mit', 'berkeley',
      'oxford', 'cambridge', 'cmu', 'toronto', 'montreal', 'mila'
    ];
    
    const authorText = paper.authors.map(a => a.affiliation || '').join(' ').toLowerCase();
    const hasPrestigiousAuthor = prestigiousInstitutions.some(inst => 
      authorText.includes(inst) || paper.abstract.toLowerCase().includes(inst)
    );
    
    if (hasPrestigiousAuthor) score += 0.3;
    
    // Abstract quality - length and technical depth
    const abstractLength = paper.abstract.split(' ').length;
    if (abstractLength > 100 && abstractLength < 500) score += 0.2;
    
    // Technical indicators
    const technicalTerms = ['theorem', 'proof', 'convergence', 'bound', 'optimal'];
    const hasTechnicalDepth = technicalTerms.some(term => 
      paper.abstract.toLowerCase().includes(term)
    );
    if (hasTechnicalDepth) score += 0.2;
    
    // Novelty indicators
    const noveltyTerms = ['novel', 'new', 'first', 'breakthrough', 'state-of-the-art'];
    const hasNovelty = noveltyTerms.some(term => 
      paper.abstract.toLowerCase().includes(term)
    );
    if (hasNovelty) score += 0.2;
    
    // Citation potential (based on title appeal and clarity)
    const titleWords = paper.title.split(' ').length;
    if (titleWords >= 3 && titleWords <= 12) score += 0.1;
    
    return Math.min(score, 1.0);
  }

  extractMetadata(paper) {
    const text = `${paper.title} ${paper.abstract}`.toLowerCase();
    
    // Extract key concepts
    const concepts = [];
    const conceptPatterns = [
      /ppo/gi, /sac/gi, /dqn/gi, /trpo/gi, /a3c/gi, /a2c/gi,
      /kl divergence/gi, /gae/gi, /advantage/gi, /policy gradient/gi,
      /value function/gi, /q-learning/gi, /actor-critic/gi,
      /sample efficiency/gi, /exploration/gi, /exploitation/gi
    ];
    
    conceptPatterns.forEach(pattern => {
      const matches = text.match(pattern);
      if (matches) {
        concepts.push(...matches.map(m => m.toLowerCase().trim()));
      }
    });
    
    // Determine category
    let category = 'general';
    if (text.includes('policy gradient') || text.includes('ppo') || text.includes('trpo') || 
        text.includes('sac') || text.includes('actor-critic') || text.includes('a2c') || text.includes('a3c')) {
      category = 'policy-gradient';
    } else if (text.includes('model-based') || text.includes('world model') || text.includes('muzero')) {
      category = 'model-based';
    } else if (text.includes('offline') || text.includes('batch rl') || text.includes('cql')) {
      category = 'offline-rl';
    } else if (text.includes('multi-agent') || text.includes('marl')) {
      category = 'multi-agent';
    } else if (text.includes('rlhf') || text.includes('human feedback')) {
      category = 'rlhf';
    }
    
    // Assess difficulty
    let difficulty = 'intermediate';
    if (text.includes('introduction') || text.includes('tutorial') || text.includes('survey')) {
      difficulty = 'beginner';
    } else if (text.includes('advanced') || text.includes('theoretical') || text.includes('proof')) {
      difficulty = 'advanced';
    } else if (text.includes('novel') && text.includes('theorem')) {
      difficulty = 'expert';
    }
    
    return {
      keyConcepts: [...new Set(concepts)],
      category,
      difficulty,
      implementationAvailable: text.includes('github') || text.includes('code available'),
      qualityScore: this.assessQuality(paper),
    };
  }

  async calculateImpactScore(paper) {
    // Simplified impact calculation
    const citationScore = Math.min((paper.citations || 0) / 1000, 1.0);
    const ageInDays = (Date.now() - paper.publishedDate.getTime()) / (1000 * 60 * 60 * 24);
    const recencyScore = Math.max(0, 1 - ageInDays / 365);
    
    return citationScore * 0.7 + recencyScore * 0.3;
  }

  async processPaperBatch(papers) {
    for (let i = 0; i < papers.length; i += this.batchSize) {
      const batch = papers.slice(i, i + this.batchSize);
      
      // Process batch
      await Promise.all(batch.map(paper => this.processPaper(paper)));
      
      this.processedCount += batch.length;
      
      // Delay between batches
      if (i + this.batchSize < papers.length) {
        await this.delay(100);
      }
    }
  }

  async processPaper(paper) {
    // Simulate processing
    await this.delay(10);
  }

  deduplicatePapers(papers) {
    const seen = new Set();
    return papers.filter(paper => {
      if (seen.has(paper.arxivId)) {
        return false;
      }
      seen.add(paper.arxivId);
      return true;
    });
  }

  delay(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
  }
}

module.exports = { ArxivRLCrawler };