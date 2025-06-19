import asyncio
import subprocess
import tempfile
import os
import json
from typing import List, Dict, Tuple, Optional, Any
import ast
import docker
from datetime import datetime

class ExecutionResult:
    def __init__(self, output: str = "", error: str = "", test_results: List = None, 
                 execution_time: float = 0.0, memory_used: int = 0):
        self.output = output
        self.error = error
        self.test_results = test_results or []
        self.execution_time = execution_time
        self.memory_used = memory_used

class TestSuiteResult:
    def __init__(self, all_passed: bool, score: float, test_results: List, feedback: str):
        self.all_passed = all_passed
        self.score = score
        self.test_results = test_results
        self.feedback = feedback

class CodeEvaluator:
    def __init__(self):
        # Initialize Docker client for sandboxed execution
        try:
            self.docker_client = docker.from_env()
        except:
            self.docker_client = None
            print("Warning: Docker not available, using subprocess fallback")
            
        # Test suites for assignments
        self.test_suites = self._load_test_suites()
        
    def _load_test_suites(self) -> Dict[str, Any]:
        """Load predefined test suites for assignments"""
        return {
            "assignment-1": {
                "tests": [
                    {
                        "name": "test_grid_world_init",
                        "code": '''
def test_grid_world_init():
    env = GridWorld(size=5)
    assert env.size == 5
    assert hasattr(env, 'reset')
    assert hasattr(env, 'step')
    return True
'''
                    },
                    {
                        "name": "test_reset",
                        "code": '''
def test_reset():
    env = GridWorld(size=5)
    state = env.reset()
    assert state is not None
    return True
'''
                    }
                ]
            }
        }
        
    async def execute_code(
        self,
        code: str,
        language: str = "python",
        test_cases: Optional[List] = None
    ) -> ExecutionResult:
        """Execute code in a sandboxed environment"""
        if self.docker_client and language == "python":
            return await self._execute_docker(code, test_cases)
        else:
            return await self._execute_subprocess(code, language, test_cases)
            
    async def _execute_docker(self, code: str, test_cases: Optional[List] = None) -> ExecutionResult:
        """Execute Python code in Docker container"""
        # Create temporary directory
        with tempfile.TemporaryDirectory() as tmpdir:
            # Write code to file
            code_file = os.path.join(tmpdir, "user_code.py")
            with open(code_file, "w") as f:
                f.write(code)
                
            # If test cases provided, append them
            if test_cases:
                with open(code_file, "a") as f:
                    f.write("\n\n# Test execution\n")
                    for test in test_cases:
                        f.write(f"\nprint('Testing: {test.name}')\n")
                        f.write(f"try:\n")
                        f.write(f"    result = {test.input}\n")
                        f.write(f"    print(f'Result: {{result}}')\n")
                        f.write(f"except Exception as e:\n")
                        f.write(f"    print(f'Error: {{e}}')\n")
                        
            try:
                # Run in Docker container
                container = self.docker_client.containers.run(
                    "python:3.9-slim",
                    f"python /code/user_code.py",
                    volumes={tmpdir: {"bind": "/code", "mode": "ro"}},
                    mem_limit="512m",
                    nano_cpus=500000000,  # 0.5 CPU
                    network_disabled=True,
                    remove=True,
                    stdout=True,
                    stderr=True,
                    timeout=30
                )
                
                return ExecutionResult(
                    output=container.decode("utf-8"),
                    error="",
                    execution_time=0.0  # Docker doesn't provide easy timing
                )
                
            except docker.errors.ContainerError as e:
                return ExecutionResult(
                    output="",
                    error=e.stderr.decode("utf-8") if e.stderr else str(e),
                    execution_time=0.0
                )
            except Exception as e:
                return ExecutionResult(
                    output="",
                    error=str(e),
                    execution_time=0.0
                )
                
    async def _execute_subprocess(
        self,
        code: str,
        language: str,
        test_cases: Optional[List] = None
    ) -> ExecutionResult:
        """Execute code using subprocess (less secure fallback)"""
        if language != "python":
            return ExecutionResult(error=f"Language {language} not supported")
            
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(code)
            if test_cases:
                f.write("\n\n# Test execution\n")
                for test in test_cases:
                    f.write(f"\nprint('Testing: {test.name}')\n")
                    f.write(f"try:\n")
                    f.write(f"    result = {test.input}\n")
                    f.write(f"    print(f'Result: {{result}}')\n")
                    f.write(f"except Exception as e:\n")
                    f.write(f"    print(f'Error: {{e}}')\n")
            f.flush()
            
            try:
                start_time = datetime.now()
                result = subprocess.run(
                    ["python", f.name],
                    capture_output=True,
                    text=True,
                    timeout=30
                )
                execution_time = (datetime.now() - start_time).total_seconds()
                
                return ExecutionResult(
                    output=result.stdout,
                    error=result.stderr,
                    execution_time=execution_time
                )
                
            except subprocess.TimeoutExpired:
                return ExecutionResult(
                    error="Code execution timed out",
                    execution_time=30.0
                )
            except Exception as e:
                return ExecutionResult(error=str(e))
            finally:
                os.unlink(f.name)
                
    async def validate_syntax(self, code: str, language: str) -> Tuple[bool, List[str]]:
        """Validate code syntax without executing"""
        if language != "python":
            return True, []  # No validation for other languages yet
            
        errors = []
        try:
            ast.parse(code)
            return True, []
        except SyntaxError as e:
            errors.append(f"Syntax error at line {e.lineno}: {e.msg}")
            return False, errors
        except Exception as e:
            errors.append(f"Validation error: {str(e)}")
            return False, errors
            
    async def get_assignment_tests(self, assignment_id: str) -> Optional[Dict]:
        """Get test suite for an assignment"""
        return self.test_suites.get(assignment_id)
        
    async def run_assignment_tests(
        self,
        code: str,
        language: str,
        test_suite: Dict
    ) -> TestSuiteResult:
        """Run assignment test suite against submitted code"""
        test_results = []
        total_score = 0
        max_score = len(test_suite["tests"])
        
        for test in test_suite["tests"]:
            # Combine user code with test code
            full_code = code + "\n\n" + test["code"] + f"\n\n# Run test\nresult = {test['name']}()\nprint(f'Test passed: {{result}}')"
            
            # Execute
            result = await self.execute_code(full_code, language)
            
            passed = "Test passed: True" in result.output and not result.error
            test_results.append({
                "name": test["name"],
                "passed": passed,
                "output": result.output,
                "error": result.error
            })
            
            if passed:
                total_score += 1
                
        score_percentage = (total_score / max_score) * 100 if max_score > 0 else 0
        all_passed = total_score == max_score
        
        # Generate feedback
        if all_passed:
            feedback = "Excellent work! All tests passed."
        elif score_percentage >= 70:
            feedback = "Good job! Most tests passed. Review the failed tests to improve."
        else:
            feedback = "Keep working on it. Review the test results and try again."
            
        return TestSuiteResult(
            all_passed=all_passed,
            score=score_percentage,
            test_results=test_results,
            feedback=feedback
        )