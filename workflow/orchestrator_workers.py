from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass
from enum import Enum
import json
from llm_factory.openai_agent import OpenAIAgent
from schemas import TaskType, SubTask, TaskResult

class OrchestratorSystem:
    def __init__(self, agent: OpenAIAgent, max_workers: int = 3):
        self.agent = agent
        self.max_workers = max_workers
        self.results = {}
        
    def _generate_task_planning_prompt(self, task: str) -> str:
        """Generate prompt for the orchestrator to plan subtasks"""
        return f"""As an orchestrator, break down the following task into subtasks.
Respond in JSON format with the following structure:
{{
    "subtasks": [
        {{
            "id": "unique_id",
            "task_type": "code|research|analysis|synthesis",
            "description": "detailed description",
            "context": {{"key": "value"}},
            "dependencies": ["dependency_task_ids"],
            "priority": 0-10
        }}
    ],
    "reasoning": "explanation of the breakdown"
}}

Task: {task}

Consider:
1. Dependencies between subtasks
2. Required context for each subtask
3. Priority of execution
"""

    def _generate_worker_prompt(self, subtask: SubTask) -> str:
        """Generate prompt for worker based on task type and context"""
        base_prompt = f"""Complete the following subtask:
Description: {subtask.description}

Context:
{json.dumps(subtask.context, indent=2)}

Provide your response in JSON format with:
{{
    "result": "your detailed result",
    "confidence": 0.0 to 1.0,
    "metadata": {{"key": "value"}}
}}
"""
        
        if subtask.task_type == TaskType.CODE:
            base_prompt += "\nProvide code changes as git-style patches or complete file contents."
        elif subtask.task_type == TaskType.RESEARCH:
            base_prompt += "\nProvide sources, key findings, and confidence levels."
        elif subtask.task_type == TaskType.ANALYSIS:
            base_prompt += "\nProvide detailed analysis with supporting evidence."
        
        return base_prompt

    def _generate_synthesis_prompt(self, results: Dict[str, TaskResult]) -> str:
        """Generate prompt for synthesizing results"""
        results_dict = {task_id: result.to_dict() for task_id, result in results.items()}
        
        return f"""Synthesize the following subtask results into a coherent final response:

Results:
{json.dumps(results_dict, indent=2)}

Provide a comprehensive response that:
1. Integrates all subtask results
2. Resolves any conflicts
3. Presents a clear final solution
4. Includes relevant details from subtasks
"""

    def _process_subtask(self, subtask: SubTask) -> TaskResult:
        """Process a single subtask using a worker"""
        prompt = self._generate_worker_prompt(subtask)
        response = self.agent.generate_response(prompt)
        
        try:
            result_json = json.loads(response[response.find('{'):response.rfind('}')+1])
            return TaskResult(
                task_id=subtask.id,
                status="completed",
                result=result_json["result"],
                metadata=result_json.get("metadata", {})
            )
        except Exception as e:
            return TaskResult(
                task_id=subtask.id,
                status="failed",
                result=str(e),
                metadata={"error": str(e)}
            )

    def _execute_tasks(self, subtasks: List[SubTask]) -> Dict[str, TaskResult]:
        """Execute all subtasks respecting dependencies"""
        pending_tasks = {task.id: task for task in subtasks}
        completed_tasks = {}
        
        while pending_tasks:
            # Find tasks with satisfied dependencies
            ready_tasks = [
                task for task in pending_tasks.values()
                if not task.dependencies or all(dep in completed_tasks for dep in task.dependencies)
            ]
            
            if not ready_tasks:
                raise ValueError("Circular dependency detected in tasks")
            
            # Execute ready tasks sequentially
            for task in ready_tasks:
                result = self._process_subtask(task)
                completed_tasks[task.id] = result
                del pending_tasks[task.id]
        
        return completed_tasks

    def process_task(self, task: str) -> Dict[str, Any]:
        """Process a complete task using orchestrator-workers pattern"""
        # Plan subtasks
        planning_prompt = self._generate_task_planning_prompt(task)
        planning_response = self.agent.generate_response(planning_prompt)
        print(f"Planning response: {planning_response}")
        
        try:
            plan = json.loads(planning_response[planning_response.find('{'):planning_response.rfind('}')+1])
            subtasks = [
                SubTask(**task_dict)
                for task_dict in plan["subtasks"]
            ]
        except Exception as e:
            raise ValueError(f"Failed to parse task planning response: {e}")
        
        # Execute subtasks
        print(f"Executing subtasks: {subtasks}")
        results = self._execute_tasks(subtasks)
        
        # Synthesize results
        synthesis_prompt = self._generate_synthesis_prompt(results)
        print(f"Executing synthesis response: {synthesis_prompt}")
        final_response = self.agent.generate_response(synthesis_prompt)
        transformed_results = {key: value.to_dict() for key, value in results.items()}
        
        return {
            "final_result": final_response,
            "subtask_results": transformed_results,
            "task_breakdown": plan.get("reasoning"),
        }

# Example usage
if __name__ == "__main__":
    agent = OpenAIAgent()
    orchestrator = OrchestratorSystem(agent)
    
    # Example 1: Code Changes
    code_task = """
    Add input validation to the user registration form in our web app.
    The form should validate:
    - Email format
    - Password strength (min 8 chars, numbers, special chars)
    - Username (alphanumeric, 3-20 chars)
    Update both frontend validation and backend API validation.
    """
    
    print("\nProcessing Code Task:")
    code_result = orchestrator.process_task(code_task)
    print(f"Code Task Result: {code_result}")
    print(json.dumps(code_result, indent=2, ensure_ascii=False))