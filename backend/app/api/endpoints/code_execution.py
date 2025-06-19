from fastapi import APIRouter, HTTPException, UploadFile, File
from typing import Dict, Optional
import asyncio
import uuid
from app.schemas.code import CodeSubmission, ExecutionResult, TestCase
from app.services.code_evaluator import CodeEvaluator

router = APIRouter()

# Initialize code evaluator
code_evaluator = CodeEvaluator()

@router.post("/execute", response_model=ExecutionResult)
async def execute_code(submission: CodeSubmission):
    """Execute code in a secure sandbox"""
    try:
        # Create execution ID
        execution_id = str(uuid.uuid4())
        
        # Execute code with timeout
        result = await asyncio.wait_for(
            code_evaluator.execute_code(
                code=submission.code,
                language=submission.language,
                test_cases=submission.test_cases
            ),
            timeout=30.0
        )
        
        return ExecutionResult(
            execution_id=execution_id,
            status="success",
            output=result.output,
            error=result.error,
            test_results=result.test_results,
            execution_time=result.execution_time,
            memory_used=result.memory_used
        )
        
    except asyncio.TimeoutError:
        return ExecutionResult(
            execution_id=execution_id,
            status="timeout",
            output="",
            error="Code execution timed out after 30 seconds",
            test_results=[],
            execution_time=30.0,
            memory_used=0
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/validate")
async def validate_code(submission: CodeSubmission):
    """Validate code syntax without executing"""
    try:
        is_valid, errors = await code_evaluator.validate_syntax(
            code=submission.code,
            language=submission.language
        )
        
        return {
            "valid": is_valid,
            "errors": errors
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/test/{assignment_id}")
async def test_assignment(assignment_id: str, submission: CodeSubmission):
    """Test code against assignment test suite"""
    try:
        # Get test suite for assignment
        test_suite = await code_evaluator.get_assignment_tests(assignment_id)
        
        if not test_suite:
            raise HTTPException(status_code=404, detail="Assignment not found")
        
        # Run tests
        result = await code_evaluator.run_assignment_tests(
            code=submission.code,
            language=submission.language,
            test_suite=test_suite
        )
        
        return {
            "assignment_id": assignment_id,
            "passed": result.all_passed,
            "score": result.score,
            "test_results": result.test_results,
            "feedback": result.feedback
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/playground-examples")
async def get_playground_examples():
    """Get example code snippets for the playground"""
    return {
        "examples": [
            {
                "id": "ppo_basic",
                "title": "Basic PPO Implementation",
                "language": "python",
                "code": '''import numpy as np
import torch
import torch.nn as nn

class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_dim)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return torch.softmax(self.fc3(x), dim=-1)

# Test the network
policy = PolicyNetwork(4, 2)
state = torch.randn(1, 4)
action_probs = policy(state)
print(f"Action probabilities: {action_probs}")'''
            },
            {
                "id": "advantage_calculation",
                "title": "GAE Advantage Calculation",
                "language": "python",
                "code": '''def calculate_gae(rewards, values, next_values, dones, gamma=0.99, lam=0.95):
    """Calculate Generalized Advantage Estimation"""
    advantages = []
    last_advantage = 0
    
    for t in reversed(range(len(rewards))):
        if dones[t]:
            last_advantage = 0
        
        delta = rewards[t] + gamma * next_values[t] * (1 - dones[t]) - values[t]
        last_advantage = delta + gamma * lam * (1 - dones[t]) * last_advantage
        advantages.insert(0, last_advantage)
    
    return advantages

# Example usage
rewards = [1, 2, 3, 4, 5]
values = [0.9, 1.8, 2.7, 3.6, 4.5]
next_values = [1.8, 2.7, 3.6, 4.5, 0]
dones = [False, False, False, False, True]

advantages = calculate_gae(rewards, values, next_values, dones)
print(f"Advantages: {advantages}")'''
            }
        ]
    }