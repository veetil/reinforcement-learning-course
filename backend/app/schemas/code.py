from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from datetime import datetime

class TestCase(BaseModel):
    name: str
    input: Optional[str] = None
    expected_output: Optional[str] = None
    weight: float = 1.0

class CodeSubmission(BaseModel):
    code: str
    language: str = "python"
    test_cases: Optional[List[TestCase]] = None

class TestResult(BaseModel):
    test_name: str
    passed: bool
    actual_output: Optional[str] = None
    expected_output: Optional[str] = None
    error: Optional[str] = None
    execution_time: float

class ExecutionResult(BaseModel):
    execution_id: str
    status: str  # success, error, timeout
    output: str
    error: Optional[str] = None
    test_results: List[TestResult] = []
    execution_time: float
    memory_used: int  # bytes
    
class CodeValidation(BaseModel):
    valid: bool
    errors: List[Dict[str, Any]] = []
    warnings: List[Dict[str, Any]] = []