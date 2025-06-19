import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import ChapterPage from '../page';
import { useParams } from 'next/navigation';

// Mock next/navigation
jest.mock('next/navigation', () => ({
  useParams: jest.fn(),
  useRouter: jest.fn(() => ({
    push: jest.fn(),
  })),
}));

// Mock framer-motion
jest.mock('framer-motion', () => ({
  motion: {
    div: ({ children, ...props }: any) => <div {...props}>{children}</div>,
    button: ({ children, ...props }: any) => <button {...props}>{children}</button>,
    pre: ({ children, ...props }: any) => <pre {...props}>{children}</pre>,
  },
  AnimatePresence: ({ children }: any) => <>{children}</>,
}));

// Mock components
jest.mock('@/components/chapters/InteractiveDemo', () => ({
  InteractiveDemo: ({ demoType }: any) => (
    <div data-testid={`demo-${demoType}`}>Interactive Demo: {demoType}</div>
  ),
}));

jest.mock('@/components/chapters/QuizSection', () => ({
  QuizSection: ({ questions }: any) => (
    <div data-testid="quiz-section">Quiz with {questions.length} questions</div>
  ),
}));

jest.mock('@/components/chapters/CodeExample', () => ({
  CodeExample: ({ code, language }: any) => (
    <div data-testid="code-example">
      <pre>{code}</pre>
    </div>
  ),
}));

describe('Chapter 1: Foundations', () => {
  beforeEach(() => {
    (useParams as jest.Mock).mockReturnValue({ id: '1' });
  });

  test('renders chapter title and introduction', () => {
    render(<ChapterPage />);
    
    expect(screen.getByText('Chapter 1: Foundations')).toBeInTheDocument();
    expect(screen.getByText(/Neural networks, optimization, and mathematical prerequisites/)).toBeInTheDocument();
  });

  test('displays learning objectives', () => {
    render(<ChapterPage />);
    
    expect(screen.getByText('Learning Objectives')).toBeInTheDocument();
    const objectives = [
      'Understand neural network fundamentals',
      'Master backpropagation algorithm',
      'Learn gradient descent optimization',
      'Get comfortable with PyTorch basics'
    ];
    
    objectives.forEach(objective => {
      expect(screen.getByText(objective)).toBeInTheDocument();
    });
  });

  test('shows progress tracker', () => {
    render(<ChapterPage />);
    
    expect(screen.getByText('Progress')).toBeInTheDocument();
    expect(screen.getByText('0%')).toBeInTheDocument();
  });

  test('renders neural network section', () => {
    render(<ChapterPage />);
    
    expect(screen.getByText('1.1 Neural Network Fundamentals')).toBeInTheDocument();
    expect(screen.getByText(/artificial neural networks are computational models/i)).toBeInTheDocument();
  });

  test('displays interactive neural network demo', () => {
    render(<ChapterPage />);
    
    expect(screen.getByTestId('demo-neural-network')).toBeInTheDocument();
  });

  test('shows mathematical equations', () => {
    render(<ChapterPage />);
    
    // Check for forward propagation equation
    expect(screen.getByText(/z = Wx \+ b/)).toBeInTheDocument();
    expect(screen.getByText(/a = σ\(z\)/)).toBeInTheDocument();
  });

  test('renders backpropagation section', () => {
    render(<ChapterPage />);
    
    expect(screen.getByText('1.2 Backpropagation')).toBeInTheDocument();
    expect(screen.getByText(/chain rule of calculus/i)).toBeInTheDocument();
  });

  test('displays gradient descent section', () => {
    render(<ChapterPage />);
    
    expect(screen.getByText('1.3 Gradient Descent Optimization')).toBeInTheDocument();
    expect(screen.getByText(/iterative optimization algorithm/i)).toBeInTheDocument();
  });

  test('shows PyTorch basics section', () => {
    render(<ChapterPage />);
    
    expect(screen.getByText('1.4 PyTorch Basics')).toBeInTheDocument();
    expect(screen.getByTestId('code-example')).toBeInTheDocument();
  });

  test('renders confusion clarifier boxes', () => {
    render(<ChapterPage />);
    
    const clarifiers = screen.getAllByTestId(/clarifier-/);
    expect(clarifiers.length).toBeGreaterThan(0);
  });

  test('displays quiz section at the end', () => {
    render(<ChapterPage />);
    
    expect(screen.getByTestId('quiz-section')).toBeInTheDocument();
    expect(screen.getByText(/Quiz with \d+ questions/)).toBeInTheDocument();
  });

  test('shows next chapter button', () => {
    render(<ChapterPage />);
    
    const nextButton = screen.getByText('Next: Chapter 2');
    expect(nextButton).toBeInTheDocument();
  });

  test('tracks section completion on scroll', async () => {
    render(<ChapterPage />);
    
    // Simulate scrolling
    const section = screen.getByText('1.1 Neural Network Fundamentals').closest('section');
    
    // Mock intersection observer
    const mockIntersectionObserver = jest.fn();
    mockIntersectionObserver.mockReturnValue({
      observe: () => null,
      unobserve: () => null,
      disconnect: () => null
    });
    window.IntersectionObserver = mockIntersectionObserver as any;
    
    // Initially progress should be 0%
    expect(screen.getByText('0%')).toBeInTheDocument();
  });

  test('displays warning for complex concepts', () => {
    render(<ChapterPage />);
    
    expect(screen.getByText(/⚠️ Common Confusion Point/)).toBeInTheDocument();
  });

  test('shows practical tips', () => {
    render(<ChapterPage />);
    
    expect(screen.getByText(/💡 Pro Tip/)).toBeInTheDocument();
  });

  test('includes exercises', () => {
    render(<ChapterPage />);
    
    expect(screen.getByText('Exercises')).toBeInTheDocument();
    expect(screen.getByText(/Exercise 1:/)).toBeInTheDocument();
  });

  test('renders key takeaways section', () => {
    render(<ChapterPage />);
    
    expect(screen.getByText('Key Takeaways')).toBeInTheDocument();
    const takeaways = screen.getByTestId('key-takeaways');
    expect(takeaways).toBeInTheDocument();
  });
});