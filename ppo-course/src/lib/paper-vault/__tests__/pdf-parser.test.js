const { PDFParser } = require('../pdf-parser');

describe('PDFParser', () => {
  let parser;

  beforeEach(() => {
    parser = new PDFParser();
  });

  describe('Metadata Extraction', () => {
    test('extracts paper title correctly', async () => {
      const mockFile = new File(['test content'], 'paper.pdf', { type: 'application/pdf' });
      const metadata = await parser.extractMetadata(mockFile);
      
      expect(metadata.title).toBe('Proximal Policy Optimization Algorithms');
    });

    test('extracts authors list', async () => {
      const mockFile = new File(['test content'], 'paper.pdf', { type: 'application/pdf' });
      const metadata = await parser.extractMetadata(mockFile);
      
      expect(metadata.authors).toHaveLength(5);
      expect(metadata.authors).toContain('John Schulman');
      expect(metadata.authors).toContain('Oleg Klimov');
    });

    test('extracts abstract', async () => {
      const mockFile = new File(['test content'], 'paper.pdf', { type: 'application/pdf' });
      const metadata = await parser.extractMetadata(mockFile);
      
      expect(metadata.abstract).toContain('We propose a new family of policy gradient methods');
      expect(metadata.abstract).toContain('proximal policy optimization (PPO)');
    });

    test('extracts keywords', async () => {
      const mockFile = new File(['test content'], 'paper.pdf', { type: 'application/pdf' });
      const metadata = await parser.extractMetadata(mockFile);
      
      expect(metadata.keywords).toContain('Reinforcement Learning');
      expect(metadata.keywords).toContain('Policy Gradient');
      expect(metadata.keywords).toContain('Actor-Critic');
    });

    test('extracts sections', async () => {
      const mockFile = new File(['test content'], 'paper.pdf', { type: 'application/pdf' });
      const metadata = await parser.extractMetadata(mockFile);
      
      expect(metadata.sections.length).toBeGreaterThan(0);
      
      const introSection = metadata.sections.find(s => s.title === 'Introduction');
      expect(introSection).toBeDefined();
      expect(introSection.level).toBe(1);
      
      const backgroundSection = metadata.sections.find(s => s.title.includes('Background'));
      expect(backgroundSection).toBeDefined();
    });

    test('extracts references', async () => {
      const mockFile = new File(['test content'], 'paper.pdf', { type: 'application/pdf' });
      const metadata = await parser.extractMetadata(mockFile);
      
      expect(metadata.references.length).toBeGreaterThan(0);
      
      const trpoRef = metadata.references.find(r => 
        r.title.toLowerCase().includes('trust region') || 
        r.title.includes('Schulman')
      );
      expect(trpoRef).toBeDefined();
      if (trpoRef) {
        expect(trpoRef.year).toBe(2015);
      }
    });
  });

  describe('RL Concept Extraction', () => {
    test('identifies RL concepts from metadata', async () => {
      const mockFile = new File(['test content'], 'paper.pdf', { type: 'application/pdf' });
      const metadata = await parser.extractMetadata(mockFile);
      const concepts = parser.extractRLConcepts(metadata);
      
      expect(concepts).toContain('policy gradient');
      expect(concepts).toContain('actor-critic');
      expect(concepts).toContain('ppo');
      // Check that we found some RL concepts
      expect(concepts.length).toBeGreaterThan(3);
    });
  });

  describe('Summary Generation', () => {
    test('generates comprehensive summary', async () => {
      const mockFile = new File(['test content'], 'paper.pdf', { type: 'application/pdf' });
      const metadata = await parser.extractMetadata(mockFile);
      const summary = parser.generateSummary(metadata);
      
      expect(summary).toContain('# Proximal Policy Optimization Algorithms');
      expect(summary).toContain('**Authors:**');
      expect(summary).toContain('**Abstract:**');
      expect(summary).toContain('**Keywords:**');
      expect(summary).toContain('**Sections:**');
    });
  });
});