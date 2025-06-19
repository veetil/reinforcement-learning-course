import React from 'react';
import { render, screen, fireEvent } from '@testing-library/react';
import { useRouter } from 'next/navigation';
import ChaptersPage from '../page';

// Mock next/navigation
jest.mock('next/navigation', () => ({
  useRouter: jest.fn(),
}));

// Mock framer-motion
jest.mock('framer-motion', () => ({
  motion: {
    div: ({ children, ...props }: any) => <div {...props}>{children}</div>,
  },
  AnimatePresence: ({ children }: any) => <>{children}</>,
}));

describe('ChaptersPage', () => {
  const mockPush = jest.fn();
  
  beforeEach(() => {
    jest.clearAllMocks();
    (useRouter as jest.Mock).mockReturnValue({
      push: mockPush,
    });
  });

  test('renders page title and description', () => {
    render(<ChaptersPage />);
    
    expect(screen.getByText('Course Chapters')).toBeInTheDocument();
    expect(screen.getByText('Master PPO through structured, interactive lessons')).toBeInTheDocument();
  });

  test('renders all 8 chapters', () => {
    render(<ChaptersPage />);
    
    const chapters = [
      'Chapter 1: Foundations',
      'Chapter 2: RL Fundamentals',
      'Chapter 3: Value Functions',
      'Chapter 4: Policy Gradient',
      'Chapter 5: Actor-Critic',
      'Chapter 6: PPO Algorithm',
      'Chapter 7: Advanced Topics',
      'Chapter 8: VERL System'
    ];

    chapters.forEach(chapter => {
      expect(screen.getByText(chapter)).toBeInTheDocument();
    });
  });

  test('displays chapter descriptions', () => {
    render(<ChaptersPage />);
    
    const descriptions = [
      'Neural networks, optimization, and mathematical prerequisites',
      'MDP framework, policies, rewards, and exploration strategies',
      'State values, action values, and Bellman equations explained',
      'From REINFORCE to natural policy gradient methods',
      'Combining value and policy methods for stable learning',
      'Deep dive into clipping, advantages, and implementation',
      'RLHF, reward modeling, and preference learning',
      'Distributed RL with separated Actor, Critic, and Rollout'
    ];

    descriptions.forEach(desc => {
      expect(screen.getByText(desc)).toBeInTheDocument();
    });
  });

  test('displays topics for each chapter', () => {
    render(<ChaptersPage />);
    
    // Check some key topics
    expect(screen.getByText('Backpropagation')).toBeInTheDocument();
    expect(screen.getByText('Markov Chains')).toBeInTheDocument();
    expect(screen.getByText('TD Learning')).toBeInTheDocument();
    expect(screen.getByText('REINFORCE')).toBeInTheDocument();
    expect(screen.getByText('GAE')).toBeInTheDocument();
    expect(screen.getByText('PPO Clipping')).toBeInTheDocument();
    expect(screen.getByText('RLHF')).toBeInTheDocument();
    expect(screen.getByText('HybridFlow')).toBeInTheDocument();
  });

  test('clicking chapter card navigates to chapter page', () => {
    render(<ChaptersPage />);
    
    const chapter1Card = screen.getByText('Chapter 1: Foundations').closest('div.cursor-pointer');
    fireEvent.click(chapter1Card!);
    
    expect(mockPush).toHaveBeenCalledWith('/chapters/1');
  });

  test('displays learning path section', () => {
    render(<ChaptersPage />);
    
    expect(screen.getByText('🎯 Learning Path')).toBeInTheDocument();
    expect(screen.getByText('Prerequisites')).toBeInTheDocument();
    expect(screen.getByText('Core Concepts')).toBeInTheDocument();
    expect(screen.getByText('Advanced Topics')).toBeInTheDocument();
  });

  test('displays quick links section', () => {
    render(<ChaptersPage />);
    
    expect(screen.getByText('⚡ Quick Links')).toBeInTheDocument();
    expect(screen.getByRole('link', { name: /Interactive Grid World/i })).toBeInTheDocument();
    expect(screen.getByRole('link', { name: /Code Playground/i })).toBeInTheDocument();
    expect(screen.getByRole('link', { name: /Live Training/i })).toBeInTheDocument();
  });

  test('quick links have correct href attributes', () => {
    render(<ChaptersPage />);
    
    const gridWorldLink = screen.getByRole('link', { name: /Interactive Grid World/i });
    const playgroundLink = screen.getByRole('link', { name: /Code Playground/i });
    const trainingLink = screen.getByRole('link', { name: /Live Training/i });
    
    expect(gridWorldLink).toHaveAttribute('href', '/interactive');
    expect(playgroundLink).toHaveAttribute('href', '/playground');
    expect(trainingLink).toHaveAttribute('href', '/training');
  });

  test('displays estimated time for chapters', () => {
    render(<ChaptersPage />);
    
    const estimatedTimes = [
      { time: '45 min', count: 1 },
      { time: '60 min', count: 2 },
      { time: '50 min', count: 1 },
      { time: '55 min', count: 1 },
      { time: '65 min', count: 1 },
      { time: '70 min', count: 1 },
      { time: '75 min', count: 1 }
    ];

    estimatedTimes.forEach(({ time, count }) => {
      const elements = screen.getAllByText(time);
      expect(elements).toHaveLength(count);
    });
  });

  test('displays difficulty levels', () => {
    render(<ChaptersPage />);
    
    const difficulties = [
      'Beginner',
      'Beginner',
      'Intermediate',
      'Intermediate',
      'Intermediate',
      'Advanced',
      'Advanced',
      'Expert'
    ];

    // Count occurrences of each difficulty
    const beginnerElements = screen.getAllByText('Beginner');
    const intermediateElements = screen.getAllByText('Intermediate');
    const advancedElements = screen.getAllByText('Advanced');
    const expertElements = screen.getAllByText('Expert');
    
    expect(beginnerElements).toHaveLength(2);
    expect(intermediateElements).toHaveLength(3);
    expect(advancedElements).toHaveLength(2);
    expect(expertElements).toHaveLength(1);
  });

  test('chapter cards have appropriate icons', () => {
    render(<ChaptersPage />);
    
    // Check that each chapter card contains an icon element
    const chapterCards = screen.getAllByTestId(/chapter-card-/);
    expect(chapterCards).toHaveLength(8);
    
    chapterCards.forEach(card => {
      const svgIcon = card.querySelector('svg');
      expect(svgIcon).toBeInTheDocument();
    });
  });

  test('learning path items have checkmarks', () => {
    render(<ChaptersPage />);
    
    const pathItems = [
      'Basic calculus and linear algebra',
      'Python programming experience',
      'Markov Decision Processes',
      'Value and Policy Methods',
      'PPO implementation from scratch',
      'RLHF and preference learning',
      'Distributed training systems'
    ];

    pathItems.forEach(item => {
      const element = screen.getByText(item);
      const listItem = element.closest('li');
      expect(listItem?.textContent).toContain('✓');
    });
  });
});