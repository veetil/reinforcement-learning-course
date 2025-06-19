from fastapi import APIRouter, HTTPException
from typing import List, Dict, Optional
from datetime import datetime
import uuid
from app.schemas.assessment import Quiz, Question, QuizSubmission, QuizResult, Assignment

router = APIRouter()

# In-memory storage for demo
quizzes: Dict[str, Quiz] = {}
assignments: Dict[str, Assignment] = {}
quiz_results: Dict[str, List[QuizResult]] = {}

# Sample quiz data
sample_quiz = Quiz(
    id="quiz-1",
    chapter_id="1",
    title="Introduction to RL Quiz",
    questions=[
        Question(
            id="q1",
            type="multiple-choice",
            content="What is the primary goal of an RL agent?",
            options=[
                {"id": "a", "text": "To minimize computational cost", "correct": False},
                {"id": "b", "text": "To maximize cumulative reward", "correct": True},
                {"id": "c", "text": "To predict the next state", "correct": False},
                {"id": "d", "text": "To model the environment", "correct": False}
            ],
            explanation="RL agents aim to maximize the total reward received over time, not just immediate rewards."
        ),
        Question(
            id="q2",
            type="multiple-choice",
            content="In the RL framework, what does the policy represent?",
            options=[
                {"id": "a", "text": "The reward function", "correct": False},
                {"id": "b", "text": "The environment dynamics", "correct": False},
                {"id": "c", "text": "The agent's strategy for selecting actions", "correct": True},
                {"id": "d", "text": "The value of each state", "correct": False}
            ],
            explanation="A policy π(a|s) defines the probability of taking action a in state s."
        )
    ]
)

quizzes["quiz-1"] = sample_quiz

@router.get("/quiz/{quiz_id}", response_model=Quiz)
async def get_quiz(quiz_id: str):
    """Get a quiz by ID"""
    if quiz_id not in quizzes:
        raise HTTPException(status_code=404, detail="Quiz not found")
    
    return quizzes[quiz_id]

@router.post("/quiz/{quiz_id}/submit", response_model=QuizResult)
async def submit_quiz(quiz_id: str, submission: QuizSubmission):
    """Submit quiz answers and get results"""
    if quiz_id not in quizzes:
        raise HTTPException(status_code=404, detail="Quiz not found")
    
    quiz = quizzes[quiz_id]
    correct_count = 0
    total_questions = len(quiz.questions)
    question_results = []
    
    # Grade each question
    for question in quiz.questions:
        user_answer = submission.answers.get(question.id)
        correct_answer = next(
            (opt["id"] for opt in question.options if opt.get("correct", False)),
            None
        )
        
        is_correct = user_answer == correct_answer
        if is_correct:
            correct_count += 1
            
        question_results.append({
            "question_id": question.id,
            "user_answer": user_answer,
            "correct_answer": correct_answer,
            "is_correct": is_correct,
            "explanation": question.explanation
        })
    
    # Calculate score
    score = (correct_count / total_questions) * 100 if total_questions > 0 else 0
    
    result = QuizResult(
        id=str(uuid.uuid4()),
        quiz_id=quiz_id,
        user_id=submission.user_id,
        score=score,
        passed=score >= 70,  # 70% passing grade
        question_results=question_results,
        submitted_at=datetime.utcnow()
    )
    
    # Store result
    if submission.user_id not in quiz_results:
        quiz_results[submission.user_id] = []
    quiz_results[submission.user_id].append(result)
    
    return result

@router.get("/assignments", response_model=List[Assignment])
async def get_assignments(chapter_id: Optional[str] = None):
    """Get all assignments, optionally filtered by chapter"""
    all_assignments = list(assignments.values())
    
    if chapter_id:
        all_assignments = [a for a in all_assignments if a.chapter_id == chapter_id]
    
    return all_assignments

@router.get("/assignment/{assignment_id}", response_model=Assignment)
async def get_assignment(assignment_id: str):
    """Get an assignment by ID"""
    if assignment_id not in assignments:
        raise HTTPException(status_code=404, detail="Assignment not found")
    
    return assignments[assignment_id]

@router.get("/user/{user_id}/results")
async def get_user_results(user_id: str):
    """Get all quiz results for a user"""
    user_quiz_results = quiz_results.get(user_id, [])
    
    return {
        "user_id": user_id,
        "quiz_results": user_quiz_results,
        "total_quizzes": len(user_quiz_results),
        "average_score": sum(r.score for r in user_quiz_results) / len(user_quiz_results) if user_quiz_results else 0
    }

# Initialize with sample assignment
sample_assignment = Assignment(
    id="assignment-1",
    chapter_id="1",
    title="Build a Simple Grid World Agent",
    description="Implement a basic RL agent that can navigate a grid world",
    objectives=[
        "Create a grid world environment",
        "Implement a random policy",
        "Implement a greedy policy",
        "Compare the performance"
    ],
    starter_code='''class GridWorld:
    def __init__(self, size=5):
        self.size = size
        self.reset()
    
    def reset(self):
        # TODO: Initialize agent position and goal position
        pass
    
    def step(self, action):
        # TODO: Implement environment step
        # Return: next_state, reward, done
        pass

class RandomAgent:
    def __init__(self, action_space):
        self.action_space = action_space
    
    def select_action(self, state):
        # TODO: Implement random action selection
        pass

# Test your implementation
env = GridWorld()
agent = RandomAgent(4)  # 4 actions: up, down, left, right
''',
    test_cases=[
        {
            "name": "Environment Reset",
            "description": "Test that environment resets properly",
            "weight": 20
        },
        {
            "name": "Valid Actions",
            "description": "Test that actions move agent correctly",
            "weight": 30
        },
        {
            "name": "Reward Calculation",
            "description": "Test that rewards are calculated correctly",
            "weight": 30
        },
        {
            "name": "Policy Performance",
            "description": "Test that greedy policy outperforms random",
            "weight": 20
        }
    ]
)

assignments["assignment-1"] = sample_assignment