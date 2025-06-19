export interface PDFMetadata {
  title?: string;
  authors?: string[];
  abstract?: string;
  keywords?: string[];
  sections?: PDFSection[];
  references?: Reference[];
  equations?: string[];
  figures?: FigureInfo[];
}

export interface PDFSection {
  title: string;
  content: string;
  level: number;
  pageNumber: number;
}

export interface Reference {
  id: string;
  title: string;
  authors: string[];
  year?: number;
  venue?: string;
}

export interface FigureInfo {
  caption: string;
  pageNumber: number;
  type: 'figure' | 'table' | 'algorithm';
}

export class PDFParser {
  /**
   * Extract metadata from PDF file
   * Note: This is a simplified implementation for demonstration
   * In production, you'd use a library like pdf.js or pdf-parse
   */
  async extractMetadata(file: File): Promise<PDFMetadata> {
    // For now, we'll simulate extraction
    // In a real implementation, you'd use pdf.js to parse the PDF
    
    const text = await this.extractTextFromPDF(file);
    
    return {
      title: this.extractTitle(text),
      authors: this.extractAuthors(text),
      abstract: this.extractAbstract(text),
      keywords: this.extractKeywords(text),
      sections: this.extractSections(text),
      references: this.extractReferences(text),
      equations: this.extractEquations(text),
      figures: this.extractFigures(text),
    };
  }

  private async extractTextFromPDF(file: File): Promise<string> {
    // Simulated text extraction
    // In production, use pdf.js or similar
    return `
      Title: Proximal Policy Optimization Algorithms
      
      Authors: John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, Oleg Klimov
      
      Abstract:
      We propose a new family of policy gradient methods for reinforcement learning, which alternate between sampling data through interaction with the environment, and optimizing a "surrogate" objective function using stochastic gradient ascent. Whereas standard policy gradient methods perform one gradient update per data sample, we propose a novel objective function that enables multiple epochs of minibatch updates. The new methods, which we call proximal policy optimization (PPO), have some of the benefits of trust region policy optimization (TRPO), but they are much simpler to implement, more general, and have better sample complexity (empirically). Our experiments test PPO on a collection of benchmark tasks, including simulated robotic locomotion and Atari game playing, and we show that PPO outperforms other online policy gradient methods, and overall strikes a favorable balance between sample complexity, simplicity, and wall-time.
      
      Keywords: Reinforcement Learning, Policy Gradient, Actor-Critic
      
      1. Introduction
      In recent years, several different approaches have been proposed for reinforcement learning...
      
      2. Background: Policy Optimization
      2.1 Policy Gradient Methods
      Policy gradient methods work by computing...
      
      References:
      [1] Schulman, J., Levine, S., Moritz, P., Jordan, M. I., & Abbeel, P. (2015). Trust region policy optimization.
      [2] Mnih, V., et al. (2016). Asynchronous methods for deep reinforcement learning.
    `;
  }

  private extractTitle(text: string): string {
    // Look for title pattern at the beginning
    const titleMatch = text.match(/Title:\s*(.+?)(?:\n|Authors:|Abstract:|$)/i);
    if (titleMatch) return titleMatch[1].trim();

    // Alternative: Look for the first line that's not empty
    const lines = text.split('\n').filter(l => l.trim());
    return lines[0] || 'Untitled';
  }

  private extractAuthors(text: string): string[] {
    const authorsMatch = text.match(/Authors?:\s*(.+?)(?:\n|Abstract:|Keywords:|$)/i);
    if (!authorsMatch) return [];

    const authorsStr = authorsMatch[1];
    // Split by common separators
    return authorsStr
      .split(/,|;|and/)
      .map(a => a.trim())
      .filter(a => a.length > 0);
  }

  private extractAbstract(text: string): string {
    const abstractMatch = text.match(/Abstract:?\s*(.+?)(?:\n\n|Keywords:|1\.|Introduction|$)/is);
    return abstractMatch ? abstractMatch[1].trim() : '';
  }

  private extractKeywords(text: string): string[] {
    const keywordsMatch = text.match(/Keywords?:?\s*(.+?)(?:\n|1\.|Introduction|$)/i);
    if (!keywordsMatch) return [];

    return keywordsMatch[1]
      .split(/,|;/)
      .map(k => k.trim())
      .filter(k => k.length > 0);
  }

  private extractSections(text: string): PDFSection[] {
    const sections: PDFSection[] = [];
    const lines = text.split('\n');
    
    // Common section patterns
    const sectionPattern = /^(\d+\.?\d*)\s+(.+)$/;
    const subsectionPattern = /^(\d+\.\d+\.?\d*)\s+(.+)$/;
    
    let currentSection = '';
    let currentLevel = 0;
    let pageNumber = 1;

    for (let i = 0; i < lines.length; i++) {
      const line = lines[i].trim();
      
      // Check for page breaks (simplified)
      if (line.includes('Page') || i % 50 === 0) {
        pageNumber++;
      }

      const sectionMatch = line.match(sectionPattern);
      if (sectionMatch) {
        const [, num, title] = sectionMatch;
        const level = num.split('.').length - 1;
        
        sections.push({
          title: title.trim(),
          content: '', // Would extract section content in full implementation
          level,
          pageNumber,
        });
      }
    }

    return sections;
  }

  private extractReferences(text: string): Reference[] {
    const references: Reference[] = [];
    
    // Look for references section
    const refsMatch = text.match(/References:?\s*(.+)$/is);
    if (!refsMatch) return references;

    const refsText = refsMatch[1];
    // Simple pattern for numbered references
    const refPattern = /\[(\d+)\]\s*(.+?)(?=\[\d+\]|$)/gs;
    
    let match;
    while ((match = refPattern.exec(refsText)) !== null) {
      const [, id, content] = match;
      
      // Extract basic info from reference
      const yearMatch = content.match(/\((\d{4})\)/);
      const year = yearMatch ? parseInt(yearMatch[1]) : undefined;
      
      // Extract title (usually before the first period)
      const titleMatch = content.match(/^(.+?)\./);
      const title = titleMatch ? titleMatch[1].trim() : content.trim();
      
      // Extract authors (before the title)
      const authorsMatch = content.match(/^(.+?)\s*\(/);
      const authors = authorsMatch 
        ? authorsMatch[1].split(/,|&/).map(a => a.trim())
        : [];

      references.push({
        id,
        title,
        authors,
        year,
      });
    }

    return references;
  }

  private extractEquations(text: string): string[] {
    const equations: string[] = [];
    
    // Look for LaTeX equations
    const equationPattern = /\$\$(.+?)\$\$|\\\[(.+?)\\\]|\\begin\{equation\}(.+?)\\end\{equation\}/gs;
    
    let match;
    while ((match = equationPattern.exec(text)) !== null) {
      const equation = match[1] || match[2] || match[3];
      if (equation) {
        equations.push(equation.trim());
      }
    }

    return equations;
  }

  private extractFigures(text: string): FigureInfo[] {
    const figures: FigureInfo[] = [];
    
    // Look for figure/table captions
    const figurePattern = /(Figure|Table|Algorithm)\s*(\d+):?\s*(.+?)(?=\n|Figure|Table|Algorithm|$)/gi;
    
    let match;
    let pageNumber = 1;
    
    while ((match = figurePattern.exec(text)) !== null) {
      const [, type, , caption] = match;
      
      figures.push({
        caption: caption.trim(),
        pageNumber,
        type: type.toLowerCase() as 'figure' | 'table' | 'algorithm',
      });
    }

    return figures;
  }

  /**
   * Extract key RL concepts from the paper
   */
  extractRLConcepts(metadata: PDFMetadata): string[] {
    const concepts: Set<string> = new Set();
    
    // Combine all text
    const fullText = [
      metadata.title || '',
      metadata.abstract || '',
      metadata.keywords?.join(' ') || '',
      metadata.sections?.map(s => s.content).join(' ') || '',
    ].join(' ').toLowerCase();

    // RL concept patterns
    const rlConcepts = [
      'policy gradient', 'value function', 'q-learning', 'actor-critic',
      'ppo', 'trpo', 'sac', 'dqn', 'a3c', 'a2c', 'ddpg', 'td3',
      'advantage', 'bellman', 'markov decision process', 'mdp',
      'exploration', 'exploitation', 'reward shaping', 'discount factor',
      'trajectory', 'rollout', 'episode', 'timestep', 'horizon',
      'on-policy', 'off-policy', 'model-based', 'model-free',
      'continuous action', 'discrete action', 'state space', 'action space',
      'convergence', 'sample efficiency', 'bias-variance',
    ];

    // Check for each concept
    rlConcepts.forEach(concept => {
      if (fullText.includes(concept)) {
        concepts.add(concept);
      }
    });

    return Array.from(concepts);
  }

  /**
   * Generate a summary of the paper suitable for the Paper Vault
   */
  generateSummary(metadata: PDFMetadata): string {
    const { title, authors, abstract, keywords } = metadata;
    
    let summary = '';
    
    if (title) {
      summary += `# ${title}\n\n`;
    }
    
    if (authors && authors.length > 0) {
      summary += `**Authors:** ${authors.join(', ')}\n\n`;
    }
    
    if (abstract) {
      summary += `**Abstract:** ${abstract}\n\n`;
    }
    
    if (keywords && keywords.length > 0) {
      summary += `**Keywords:** ${keywords.join(', ')}\n\n`;
    }
    
    if (metadata.sections && metadata.sections.length > 0) {
      summary += `**Sections:**\n`;
      metadata.sections.forEach(section => {
        const indent = '  '.repeat(section.level);
        summary += `${indent}- ${section.title} (p. ${section.pageNumber})\n`;
      });
      summary += '\n';
    }
    
    if (metadata.equations && metadata.equations.length > 0) {
      summary += `**Key Equations:** ${metadata.equations.length} equations found\n\n`;
    }
    
    if (metadata.figures && metadata.figures.length > 0) {
      summary += `**Figures & Tables:** ${metadata.figures.length} visual elements\n`;
    }
    
    return summary;
  }
}