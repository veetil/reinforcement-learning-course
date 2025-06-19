'use client';

import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { useRouter } from 'next/navigation';
import { 
  BookOpen, CheckCircle, Trophy, Star, Target, 
  Brain, BarChart, TrendingUp, Award, Zap,
  ArrowRight, ArrowLeft, RefreshCw, Download
} from 'lucide-react';

interface Question {
  id: string;
  text: string;
  options: string[];
  correctAnswer: number;
  explanation: string;
  difficulty: 'Easy' | 'Medium' | 'Hard';
}

interface AssessmentCategory {
  id: string;
  title: string;
  description: string;
  icon: React.ElementType;
  color: string;
  questions: Question[];
  passingScore: number;
}

const assessmentCategories: AssessmentCategory[] = [
  {
    id: 'fundamentals',
    title: 'Fundamentals',
    description: 'Neural networks, backpropagation, and optimization basics',
    icon: Brain,
    color: 'bg-blue-500',
    passingScore: 70,
    questions: [
      {
        id: 'f1',
        text: 'What is the primary purpose of backpropagation in neural networks?',
        options: [
          'To initialize network weights',
          'To compute gradients for parameter updates',
          'To make predictions on new data',
          'To evaluate model performance'
        ],
        correctAnswer: 1,
        explanation: 'Backpropagation computes gradients of the loss function with respect to network parameters, enabling gradient-based optimization.',
        difficulty: 'Easy'
      },
      {
        id: 'f2',
        text: 'Which activation function helps mitigate the vanishing gradient problem?',
        options: [
          'Sigmoid',
          'Tanh',
          'ReLU',
          'Linear'
        ],
        correctAnswer: 2,
        explanation: 'ReLU (Rectified Linear Unit) helps mitigate vanishing gradients because its derivative is either 0 or 1, preventing gradient decay in deep networks.',
        difficulty: 'Medium'
      },
      {
        id: 'f3',
        text: 'What happens when the learning rate is too high in gradient descent?',
        options: [
          'Training becomes slower',
          'The model converges to a better solution',
          'The loss may oscillate or diverge',
          'Memory usage increases'
        ],
        correctAnswer: 2,
        explanation: 'A learning rate that is too high can cause the optimizer to overshoot the minimum, leading to oscillating or diverging loss.',
        difficulty: 'Medium'
      }
    ]
  },
  {
    id: 'value-functions',
    title: 'Value Functions',
    description: 'State values, action values, and Bellman equations',
    icon: BarChart,
    color: 'bg-green-500',
    passingScore: 75,
    questions: [
      {
        id: 'v1',
        text: 'What does the state value function V(s) represent?',
        options: [
          'The immediate reward in state s',
          'The expected cumulative reward starting from state s',
          'The probability of being in state s',
          'The best action to take in state s'
        ],
        correctAnswer: 1,
        explanation: 'V(s) represents the expected cumulative (discounted) reward when starting from state s and following a given policy.',
        difficulty: 'Easy'
      },
      {
        id: 'v2',
        text: 'How does Q(s,a) relate to V(s)?',
        options: [
          'Q(s,a) = V(s) for all actions a',
          'V(s) = max_a Q(s,a) for optimal policy',
          'Q(s,a) = V(s) + reward',
          'They are unrelated'
        ],
        correctAnswer: 1,
        explanation: 'For an optimal policy, V(s) equals the maximum Q-value: V*(s) = max_a Q*(s,a). The optimal policy chooses the action with highest Q-value.',
        difficulty: 'Medium'
      },
      {
        id: 'v3',
        text: 'What is the key difference between TD learning and Monte Carlo methods?',
        options: [
          'TD uses function approximation, MC does not',
          'TD updates estimates online, MC waits for episode completion',
          'TD is only for continuous spaces',
          'MC is more computationally efficient'
        ],
        correctAnswer: 1,
        explanation: 'TD learning updates value estimates online using bootstrapping, while Monte Carlo methods wait for complete episodes to update estimates.',
        difficulty: 'Hard'
      }
    ]
  },
  {
    id: 'policy-gradients',
    title: 'Policy Gradients',
    description: 'REINFORCE, baselines, and policy optimization',
    icon: TrendingUp,
    color: 'bg-purple-500',
    passingScore: 75,
    questions: [
      {
        id: 'p1',
        text: 'What is the main advantage of policy gradient methods over value-based methods?',
        options: [
          'They are faster to compute',
          'They can handle continuous action spaces naturally',
          'They require less memory',
          'They always find the global optimum'
        ],
        correctAnswer: 1,
        explanation: 'Policy gradient methods can naturally handle continuous action spaces by directly parameterizing the policy, while value-based methods struggle with continuous actions.',
        difficulty: 'Medium'
      },
      {
        id: 'p2',
        text: 'Why do we use baselines in REINFORCE?',
        options: [
          'To reduce bias in gradient estimates',
          'To reduce variance in gradient estimates',
          'To increase the learning rate',
          'To handle continuous actions'
        ],
        correctAnswer: 1,
        explanation: 'Baselines reduce the variance of gradient estimates without introducing bias, leading to more stable and efficient learning.',
        difficulty: 'Medium'
      },
      {
        id: 'p3',
        text: 'What does the score function ∇log π(a|s) represent in policy gradients?',
        options: [
          'The probability of action a in state s',
          'The gradient of the log probability with respect to parameters',
          'The value of taking action a in state s',
          'The reward received after action a'
        ],
        correctAnswer: 1,
        explanation: 'The score function ∇log π(a|s) is the gradient of the log probability of action a in state s with respect to policy parameters.',
        difficulty: 'Hard'
      }
    ]
  },
  {
    id: 'ppo-algorithm',
    title: 'PPO Algorithm',
    description: 'Clipping, trust regions, and implementation details',
    icon: Award,
    color: 'bg-red-500',
    passingScore: 80,
    questions: [
      {
        id: 'ppo1',
        text: 'What is the primary innovation of PPO compared to standard policy gradients?',
        options: [
          'Using neural networks for function approximation',
          'Clipping the probability ratio to limit policy updates',
          'Computing advantages with GAE',
          'Using multiple parallel environments'
        ],
        correctAnswer: 1,
        explanation: 'PPO\'s key innovation is clipping the probability ratio r(θ) = π(a|s)/π_old(a|s) to prevent excessively large policy updates.',
        difficulty: 'Medium'
      },
      {
        id: 'ppo2',
        text: 'When advantages are positive, how does PPO clipping work?',
        options: [
          'Clips ratio at minimum of (1-ε)',
          'Clips ratio at maximum of (1+ε)',
          'No clipping is applied',
          'Clips the advantage directly'
        ],
        correctAnswer: 1,
        explanation: 'For positive advantages, PPO clips the ratio at (1+ε) to prevent the policy from changing too much in the good direction.',
        difficulty: 'Hard'
      },
      {
        id: 'ppo3',
        text: 'Why can PPO safely perform multiple epochs on the same data?',
        options: [
          'Because it uses experience replay',
          'Because of the clipping mechanism that limits updates',
          'Because it normalizes advantages',
          'Because it uses a separate critic network'
        ],
        correctAnswer: 1,
        explanation: 'PPO\'s clipping mechanism prevents the new policy from deviating too far from the old policy, making multiple epochs on the same data safe.',
        difficulty: 'Hard'
      }
    ]
  },
  {
    id: 'advanced-topics',
    title: 'Advanced Topics',
    description: 'RLHF, VERL system, and production deployment',
    icon: Zap,
    color: 'bg-indigo-500',
    passingScore: 75,
    questions: [
      {
        id: 'a1',
        text: 'What is the main advantage of RLHF over supervised fine-tuning for language models?',
        options: [
          'It requires less computational resources',
          'It can optimize for hard-to-specify human preferences',
          'It trains faster than supervised learning',
          'It doesn\'t require any human data'
        ],
        correctAnswer: 1,
        explanation: 'RLHF allows optimization for complex human preferences that are difficult to specify explicitly, going beyond what can be captured in supervised datasets.',
        difficulty: 'Medium'
      },
      {
        id: 'a2',
        text: 'In the VERL architecture, why separate the Actor, Critic, and Rollout workers?',
        options: [
          'To reduce total computation time',
          'Each component can scale independently based on bottlenecks',
          'To make the code easier to debug',
          'To reduce memory usage'
        ],
        correctAnswer: 1,
        explanation: 'Separating components allows independent scaling - e.g., you might need many CPU rollout workers but fewer GPU actor workers, optimizing resource allocation.',
        difficulty: 'Hard'
      },
      {
        id: 'a3',
        text: 'What does the KL penalty in PPO for language models accomplish?',
        options: [
          'Reduces computational cost',
          'Prevents the model from diverging too far from the reference model',
          'Increases generation diversity',
          'Improves training speed'
        ],
        correctAnswer: 1,
        explanation: 'The KL penalty keeps the fine-tuned model close to a reference model, maintaining language quality while allowing improvement through RL.',
        difficulty: 'Medium'
      }
    ]
  }
];

export default function AssessmentPage() {
  const router = useRouter();
  const [selectedCategory, setSelectedCategory] = useState<string | null>(null);
  const [currentQuestionIndex, setCurrentQuestionIndex] = useState(0);
  const [selectedAnswer, setSelectedAnswer] = useState<number | null>(null);
  const [showFeedback, setShowFeedback] = useState(false);
  const [categoryScores, setCategoryScores] = useState<Record<string, number>>({});
  const [answers, setAnswers] = useState<Record<string, number>>({});
  const [assessmentComplete, setAssessmentComplete] = useState(false);
  const [showResults, setShowResults] = useState(false);

  const currentCategory = assessmentCategories.find(cat => cat.id === selectedCategory);
  const currentQuestion = currentCategory?.questions[currentQuestionIndex];
  const totalQuestions = currentCategory?.questions.length || 0;

  useEffect(() => {
    // Load saved progress from localStorage
    const saved = localStorage.getItem('ppo-course-assessment');
    if (saved) {
      const data = JSON.parse(saved);
      setCategoryScores(data.categoryScores || {});
      setAnswers(data.answers || {});
    }
  }, []);

  const saveProgress = () => {
    localStorage.setItem('ppo-course-assessment', JSON.stringify({
      categoryScores,
      answers
    }));
  };

  const handleCategorySelect = (categoryId: string) => {
    setSelectedCategory(categoryId);
    setCurrentQuestionIndex(0);
    setSelectedAnswer(null);
    setShowFeedback(false);
    setShowResults(false);
  };

  const handleAnswerSelect = (answerIndex: number) => {
    if (selectedAnswer !== null) return;
    
    setSelectedAnswer(answerIndex);
    setShowFeedback(true);
    
    if (currentQuestion) {
      setAnswers(prev => ({
        ...prev,
        [currentQuestion.id]: answerIndex
      }));
    }
  };

  const handleNext = () => {
    if (currentQuestionIndex < totalQuestions - 1) {
      setCurrentQuestionIndex(currentQuestionIndex + 1);
      setSelectedAnswer(null);
      setShowFeedback(false);
    } else {
      // Complete category assessment
      completeAssessment();
    }
  };

  const completeAssessment = () => {
    if (!currentCategory) return;
    
    const correctAnswers = currentCategory.questions.filter(
      q => answers[q.id] === q.correctAnswer
    ).length;
    
    const score = Math.round((correctAnswers / totalQuestions) * 100);
    
    setCategoryScores(prev => ({
      ...prev,
      [currentCategory.id]: score
    }));
    
    setShowResults(true);
    saveProgress();
  };

  const resetCategory = () => {
    if (!currentCategory) return;
    
    // Remove answers for this category
    const newAnswers = { ...answers };
    currentCategory.questions.forEach(q => {
      delete newAnswers[q.id];
    });
    setAnswers(newAnswers);
    
    // Remove score
    const newScores = { ...categoryScores };
    delete newScores[currentCategory.id];
    setCategoryScores(newScores);
    
    // Reset state
    setCurrentQuestionIndex(0);
    setSelectedAnswer(null);
    setShowFeedback(false);
    setShowResults(false);
    
    saveProgress();
  };

  const backToOverview = () => {
    setSelectedCategory(null);
    setShowResults(false);
  };

  const overallProgress = () => {
    const totalCategories = assessmentCategories.length;
    const completedCategories = Object.keys(categoryScores).length;
    const totalScore = Object.values(categoryScores).reduce((a, b) => a + b, 0);
    const averageScore = completedCategories > 0 ? totalScore / completedCategories : 0;
    
    return {
      completed: completedCategories,
      total: totalCategories,
      percentage: Math.round((completedCategories / totalCategories) * 100),
      averageScore: Math.round(averageScore)
    };
  };

  const progress = overallProgress();
  const isAllComplete = progress.completed === progress.total && progress.averageScore >= 75;

  if (selectedCategory && !showResults) {
    return (
      <div className="min-h-screen bg-gray-50 py-8">
        <div className="max-w-4xl mx-auto px-4">
          {/* Header */}
          <div className="flex items-center justify-between mb-8">
            <button
              onClick={backToOverview}
              className="flex items-center gap-2 text-gray-600 hover:text-gray-900"
            >
              <ArrowLeft size={20} />
              Back to Assessment
            </button>
            
            <div className="text-center">
              <h1 className="text-2xl font-bold">{currentCategory?.title}</h1>
              <p className="text-gray-600">
                Question {currentQuestionIndex + 1} of {totalQuestions}
              </p>
            </div>
            
            <div className="text-sm text-gray-600">
              Score: {Object.keys(answers).filter(qId => 
                currentCategory?.questions.find(q => q.id === qId && answers[qId] === q.correctAnswer)
              ).length} / {currentQuestionIndex + (selectedAnswer !== null ? 1 : 0)}
            </div>
          </div>

          {/* Progress bar */}
          <div className="w-full bg-gray-200 rounded-full h-2 mb-8">
            <div
              className="bg-blue-600 h-2 rounded-full transition-all"
              style={{ width: `${((currentQuestionIndex + 1) / totalQuestions) * 100}%` }}
            />
          </div>

          {/* Question */}
          {currentQuestion && (
            <motion.div
              key={currentQuestion.id}
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              style={{}}
            >
              <div className="bg-white rounded-lg shadow-lg p-8">
              <div className="flex items-center gap-3 mb-6">
                <span className={`px-3 py-1 rounded-full text-xs font-medium ${
                  currentQuestion.difficulty === 'Easy' ? 'bg-green-100 text-green-700' :
                  currentQuestion.difficulty === 'Medium' ? 'bg-yellow-100 text-yellow-700' :
                  'bg-red-100 text-red-700'
                }`}>
                  {currentQuestion.difficulty}
                </span>
              </div>
              
              <h2 data-testid="question-text" className="text-xl font-semibold mb-6">
                {currentQuestion.text}
              </h2>
              
              <div className="space-y-3 mb-6">
                {currentQuestion.options.map((option, index) => {
                  const isSelected = selectedAnswer === index;
                  const isCorrect = index === currentQuestion.correctAnswer;
                  const showResult = showFeedback;
                  
                  return (
                    <button
                      key={index}
                      name={`option-${index}`}
                      onClick={() => handleAnswerSelect(index)}
                      disabled={selectedAnswer !== null}
                      className={`w-full text-left p-4 rounded-lg border-2 transition-all ${
                        showResult
                          ? isCorrect
                            ? 'border-green-500 bg-green-50'
                            : isSelected
                            ? 'border-red-500 bg-red-50'
                            : 'border-gray-200'
                          : isSelected
                          ? 'border-blue-500 bg-blue-50'
                          : 'border-gray-200 hover:border-gray-300'
                      }`}
                    >
                      <div className="flex items-center justify-between">
                        <span>{option}</span>
                        {showResult && (
                          <span>
                            {isCorrect ? (
                              <CheckCircle className="w-5 h-5 text-green-600" />
                            ) : isSelected ? (
                              <span className="w-5 h-5 text-red-600">✗</span>
                            ) : null}
                          </span>
                        )}
                      </div>
                    </button>
                  );
                })}
              </div>
              
              {showFeedback && (
                <motion.div
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  data-testid="answer-feedback"
                >
                  <div className={`p-4 rounded-lg mb-6 ${
                    selectedAnswer === currentQuestion.correctAnswer
                      ? 'bg-green-50 border border-green-200'
                      : 'bg-red-50 border border-red-200'
                  }`}>
                  <div className="flex items-start gap-3">
                    {selectedAnswer === currentQuestion.correctAnswer ? (
                      <CheckCircle className="w-5 h-5 text-green-600 mt-0.5" />
                    ) : (
                      <span className="w-5 h-5 text-red-600 mt-0.5">✗</span>
                    )}
                    <div>
                      <p className="font-medium mb-1">
                        {selectedAnswer === currentQuestion.correctAnswer ? 'Correct!' : 'Incorrect'}
                      </p>
                      <p data-testid="answer-explanation" className="text-sm text-gray-700">
                        {currentQuestion.explanation}
                      </p>
                    </div>
                  </div>
                  </div>
                </motion.div>
              )}
              
              {showFeedback && (
                <button
                  onClick={handleNext}
                  className="px-6 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 flex items-center gap-2"
                >
                  {currentQuestionIndex < totalQuestions - 1 ? 'Next Question' : 'View Results'}
                  <ArrowRight size={20} />
                </button>
              )}
              </div>
            </motion.div>
          )}
        </div>
      </div>
    );
  }

  if (showResults && currentCategory) {
    const score = categoryScores[currentCategory.id] || 0;
    const passed = score >= currentCategory.passingScore;
    
    return (
      <div className="min-h-screen bg-gray-50 py-8">
        <div className="max-w-4xl mx-auto px-4">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
          >
            <div className="bg-white rounded-lg shadow-lg p-8 text-center">
            <div className={`w-20 h-20 rounded-full flex items-center justify-center mx-auto mb-6 ${
              passed ? 'bg-green-100' : 'bg-yellow-100'
            }`}>
              {passed ? (
                <Trophy className="w-10 h-10 text-green-600" />
              ) : (
                <Target className="w-10 h-10 text-yellow-600" />
              )}
            </div>
            
            <h1 className="text-3xl font-bold mb-4">
              {currentCategory.title} Assessment Complete!
            </h1>
            
            <div className="text-6xl font-bold mb-4 text-blue-600">
              {score}%
            </div>
            
            <p className="text-xl mb-6">
              {passed ? (
                <span className="text-green-600">🎉 Passed! Great work!</span>
              ) : (
                <span className="text-yellow-600">
                  You need {currentCategory.passingScore}% to pass. Keep practicing!
                </span>
              )}
            </p>
            
            <div className="flex justify-center gap-4">
              <button
                onClick={backToOverview}
                className="px-6 py-3 bg-gray-600 text-white rounded-lg hover:bg-gray-700"
              >
                Back to Overview
              </button>
              
              {!passed && (
                <button
                  onClick={resetCategory}
                  className="px-6 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 flex items-center gap-2"
                >
                  <RefreshCw size={20} />
                  Retake Assessment
                </button>
              )}
            </div>
            </div>
          </motion.div>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-50 py-8">
      <div className="max-w-7xl mx-auto px-4">
        {/* Header */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
        >
          <div className="text-center mb-8">
            <h1 className="text-4xl font-bold mb-4">Course Assessment</h1>
            <p className="text-xl text-gray-600">
              Test your understanding of PPO and reinforcement learning concepts
            </p>
          </div>
        </motion.div>

        {/* Overall Progress */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.1 }}
          data-testid="overall-progress"
        >
          <div className="bg-white rounded-lg shadow-lg p-6 mb-8">
          <h2 className="text-2xl font-bold mb-4">Overall Progress</h2>
          
          <div className="grid md:grid-cols-3 gap-6">
            <div className="text-center">
              <div className="text-3xl font-bold text-blue-600 mb-2">
                {progress.completed}/{progress.total}
              </div>
              <p className="text-gray-600">Assessments Completed</p>
            </div>
            
            <div className="text-center">
              <div className="text-3xl font-bold text-green-600 mb-2">
                {progress.percentage}%
              </div>
              <p className="text-gray-600">Course Completion</p>
            </div>
            
            <div className="text-center">
              <div className="text-3xl font-bold text-purple-600 mb-2">
                {progress.averageScore}%
              </div>
              <p className="text-gray-600">Average Score</p>
            </div>
          </div>
          
          <div className="w-full bg-gray-200 rounded-full h-4 mt-6">
            <div
              className="bg-blue-600 h-4 rounded-full transition-all"
              style={{ width: `${progress.percentage}%` }}
            />
          </div>
          
          {isAllComplete && (
            <motion.div
              initial={{ opacity: 0, scale: 0.9 }}
              animate={{ opacity: 1, scale: 1 }}
              data-testid="certificate-section"
            >
              <div className="mt-6 p-4 bg-gradient-to-r from-yellow-50 to-orange-50 rounded-lg border-2 border-yellow-200">
                <div className="text-center">
                <Trophy className="w-12 h-12 text-yellow-600 mx-auto mb-3" />
                <h3 className="text-xl font-bold text-yellow-800 mb-2">
                  🎉 Congratulations! Course Completed!
                </h3>
                <p className="text-yellow-700 mb-4">
                  You've successfully completed all assessments with a passing grade.
                </p>
                <button className="px-6 py-3 bg-yellow-600 text-white rounded-lg hover:bg-yellow-700 flex items-center gap-2 mx-auto">
                  <Download size={20} />
                  Download Certificate
                </button>
                </div>
              </div>
            </motion.div>
          )}
          </div>
        </motion.div>

        {/* Assessment Categories */}
        <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
          {assessmentCategories.map((category, index) => {
            const Icon = category.icon;
            const score = categoryScores[category.id];
            const hasAttempted = score !== undefined;
            const passed = hasAttempted && score >= category.passingScore;
            
            return (
              <motion.div
                key={category.id}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.2 + index * 0.1 }}
              >
                <button
                  onClick={() => handleCategorySelect(category.id)}
                  className="bg-white rounded-lg shadow-lg overflow-hidden hover:shadow-xl transform hover:-translate-y-1 transition-all text-left w-full"
                >
                <div className={`h-2 ${category.color}`} />
                
                <div className="p-6">
                  <div className="flex items-center justify-between mb-4">
                    <div className={`p-3 rounded-lg ${category.color} bg-opacity-20`}>
                      <Icon className={`w-6 h-6 ${category.color.replace('bg-', 'text-')}`} />
                    </div>
                    
                    {hasAttempted && (
                      <div className="text-right">
                        <div className={`text-2xl font-bold ${
                          passed ? 'text-green-600' : 'text-yellow-600'
                        }`}>
                          {score}%
                        </div>
                        <div className="flex items-center gap-1">
                          {passed ? (
                            <CheckCircle className="w-4 h-4 text-green-600" />
                          ) : (
                            <RefreshCw className="w-4 h-4 text-yellow-600" />
                          )}
                          <span className={`text-xs ${
                            passed ? 'text-green-600' : 'text-yellow-600'
                          }`}>
                            {passed ? 'Passed' : 'Retry'}
                          </span>
                        </div>
                      </div>
                    )}
                  </div>
                  
                  <h3 className="text-xl font-bold mb-2">{category.title}</h3>
                  <p className="text-gray-600 text-sm mb-4">{category.description}</p>
                  
                  <div className="flex items-center justify-between text-sm text-gray-500">
                    <span>{category.questions.length} questions</span>
                    <span>Passing: {category.passingScore}%</span>
                  </div>
                  
                  {hasAttempted && (
                    <div data-testid={`progress-${category.id}`} className="mt-4">
                      <div className="w-full bg-gray-200 rounded-full h-2">
                        <div
                          className={`h-2 rounded-full ${
                            passed ? 'bg-green-500' : 'bg-yellow-500'
                          }`}
                          style={{ width: `${Math.min(score, 100)}%` }}
                        />
                      </div>
                    </div>
                  )}
                </div>
                </button>
              </motion.div>
            );
          })}
        </div>
      </div>
    </div>
  );
}