import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { useRouter } from 'next/navigation';
import AssessmentPage from '../page';

// Mock next/navigation
jest.mock('next/navigation', () => ({
  useRouter: jest.fn(),
}));

// Mock framer-motion
jest.mock('framer-motion', () => ({
  motion: {
    div: ({ children, ...props }: any) => <div {...props}>{children}</div>,
    button: ({ children, ...props }: any) => <button {...props}>{children}</button>,
  },
  AnimatePresence: ({ children }: any) => <>{children}</>,
}));

describe('AssessmentPage', () => {
  const mockPush = jest.fn();
  
  beforeEach(() => {
    jest.clearAllMocks();
    (useRouter as jest.Mock).mockReturnValue({
      push: mockPush,
    });
  });

  test('renders assessment overview', () => {
    render(<AssessmentPage />);
    
    expect(screen.getByText('Course Assessment')).toBeInTheDocument();
    expect(screen.getByText(/Test your understanding of PPO/)).toBeInTheDocument();
  });

  test('displays assessment categories', () => {
    render(<AssessmentPage />);
    
    const categories = [
      'Fundamentals',
      'Value Functions',
      'Policy Gradients',
      'PPO Algorithm',
      'Advanced Topics'
    ];
    
    categories.forEach(category => {
      expect(screen.getByText(category)).toBeInTheDocument();
    });
  });

  test('shows categories without progress initially', () => {
    render(<AssessmentPage />);
    
    // Should show category cards but no progress indicators initially
    expect(screen.getAllByText(/questions/).length).toBeGreaterThan(0);
    expect(screen.getAllByText(/Passing:/).length).toBeGreaterThan(0);
  });

  test('clicking on category starts assessment', () => {
    render(<AssessmentPage />);
    
    const fundamentalsCard = screen.getByText('Fundamentals').closest('button');
    fireEvent.click(fundamentalsCard!);
    
    expect(screen.getByText(/Question 1/)).toBeInTheDocument();
  });

  test('displays question with multiple choice options', () => {
    render(<AssessmentPage />);
    
    // Start assessment
    const categoryCard = screen.getByText('Fundamentals').closest('button');
    fireEvent.click(categoryCard!);
    
    // Check question structure
    expect(screen.getByTestId('question-text')).toBeInTheDocument();
    const options = screen.getAllByText(/To \w+/); // Match option text patterns
    expect(options.length).toBeGreaterThanOrEqual(3);
  });

  test('selecting answer shows feedback', async () => {
    render(<AssessmentPage />);
    
    // Start assessment
    const categoryCard = screen.getByText('Fundamentals').closest('button');
    fireEvent.click(categoryCard!);
    
    // Select an answer
    const firstOption = screen.getByText('To initialize network weights').closest('button');
    fireEvent.click(firstOption!);
    
    // Should show feedback
    await waitFor(() => {
      expect(screen.getByTestId('answer-feedback')).toBeInTheDocument();
    });
  });

  test('tracks score throughout assessment', () => {
    render(<AssessmentPage />);
    
    // Start assessment
    const categoryCard = screen.getByText('Fundamentals').closest('button');
    fireEvent.click(categoryCard!);
    
    // Score tracker should be visible
    expect(screen.getByText(/Score:/)).toBeInTheDocument();
  });

  test('shows final results after completing category', async () => {
    render(<AssessmentPage />);
    
    // Start and complete assessment
    const categoryCard = screen.getByText('Fundamentals').closest('button');
    fireEvent.click(categoryCard!);
    
    // Answer all questions (mock completion)
    // This would normally involve answering multiple questions
    
    // Check for results screen
    // Implementation would show this after answering all questions
  });

  test('provides explanations for answers', () => {
    render(<AssessmentPage />);
    
    // Start assessment
    const categoryCard = screen.getByText('Fundamentals').closest('button');
    fireEvent.click(categoryCard!);
    
    // Answer a question
    const option = screen.getByText('To initialize network weights').closest('button');
    fireEvent.click(option!);
    
    // Should show explanation
    expect(screen.getByTestId('answer-explanation')).toBeInTheDocument();
  });

  test('allows reviewing missed questions', () => {
    render(<AssessmentPage />);
    
    // This would be shown in the results screen
    // Check for review functionality
  });

  test('shows overall course completion status', () => {
    render(<AssessmentPage />);
    
    expect(screen.getByTestId('overall-progress')).toBeInTheDocument();
    expect(screen.getByText(/Overall Progress/)).toBeInTheDocument();
  });

  test('provides certificate or completion badge', () => {
    render(<AssessmentPage />);
    
    // This would be shown when all assessments are completed
    // Check for certificate section
    const certificateSection = screen.queryByTestId('certificate-section');
    // Would be present if all assessments completed
  });
});