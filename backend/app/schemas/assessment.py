from pydantic import BaseModel
from typing import List, Dict, Optional, Any
from datetime import datetime

class QuestionOption(BaseModel):
    id: str
    text: str
    correct: bool = False

class Question(BaseModel):
    id: str
    type: str  # multiple-choice, code-completion, drag-drop, numerical
    content: str
    options: Optional[List[Dict[str, Any]]] = None
    correct_answer: Optional[Any] = None
    explanation: str
    points: int = 1
    hints: List[str] = []

class Quiz(BaseModel):
    id: str
    chapter_id: str
    title: str
    description: Optional[str] = None
    questions: List[Question]
    time_limit: Optional[int] = None  # minutes
    passing_score: float = 70.0

class QuizSubmission(BaseModel):
    quiz_id: str
    user_id: str
    answers: Dict[str, Any]  # question_id -> answer
    time_taken: Optional[int] = None  # seconds

class QuizResult(BaseModel):
    id: str
    quiz_id: str
    user_id: str
    score: float
    passed: bool
    question_results: List[Dict[str, Any]]
    submitted_at: datetime
    time_taken: Optional[int] = None

class Assignment(BaseModel):
    id: str
    chapter_id: str
    title: str
    description: str
    objectives: List[str]
    starter_code: str
    test_cases: List[Dict[str, Any]]
    rubric: Optional[Dict[str, float]] = None
    due_date: Optional[datetime] = None
    
class AssignmentSubmission(BaseModel):
    assignment_id: str
    user_id: str
    code: str
    language: str = "python"
    submitted_at: datetime = datetime.utcnow()
    
class AssignmentResult(BaseModel):
    id: str
    assignment_id: str
    user_id: str
    score: float
    passed: bool
    test_results: List[Dict[str, Any]]
    feedback: str
    submitted_at: datetime
    graded_at: datetime